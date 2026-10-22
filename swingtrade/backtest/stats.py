"""Trade summaries, volatility regimes, drawdown, Monte Carlo and permutation statistics."""

import math
import random
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd


def _annualize_sharpe(resolved: list[dict], sharpe_like: float | None) -> tuple[float | None, float | None]:
    """Shared by summarize_trades()/summarize_trades_weighted() -- sharpe_like
    is a RAW per-trade ratio (mean/std of pnl_pct across trades), never
    annualized, which reads as a tiny number (0.02-0.15 typical here) next
    to the "Sharpe > 1.5" figures usually quoted, since those are almost
    always ANNUALIZED (scaled by sqrt(periods per year)). This applies the
    same standard trade-frequency convention: trades_per_year from the
    pooled trades' own entry_date span, annualized_sharpe_like =
    sharpe_like * sqrt(trades_per_year). Same explicit caveat as
    compute_max_drawdown()'s own docstring: an approximation for reading
    the number on a familiar scale, not a literal dollar-accurate
    portfolio simulation -- additive alongside sharpe_like, not a
    replacement for it.

    Returns (trades_per_year, annualized_sharpe_like), both None if there
    aren't at least 2 distinct entry dates to establish a rate from, if
    sharpe_like itself is None, or if the trades don't carry `entry_date`
    at all -- real callers pool live Trade_Outcomes Mongo documents into a
    minimal {"status", "pnl_pct"} shape (see optimize.py's
    report_live_outcomes_context(), settle_trades.py) with no entry_date
    (those documents use a differently-shaped `signal_date` string
    instead), so this must degrade gracefully rather than assume the field
    exists -- same "missing data doesn't crash the summary" convention as
    everything else in this function."""
    if sharpe_like is None or len(resolved) < 2:
        return None, None
    if any("entry_date" not in t for t in resolved):
        return None, None
    entry_dates = [t["entry_date"] for t in resolved]
    span_days = (max(entry_dates) - min(entry_dates)).days
    if span_days <= 0:
        return None, None
    trades_per_year = len(resolved) / (span_days / 365.25)
    annualized_sharpe_like = sharpe_like * (trades_per_year ** 0.5)
    return round(trades_per_year, 1), round(annualized_sharpe_like, 3)


def summarize_trades(trades: list[dict]) -> dict:
    """Aggregate simulated trades into summary performance metrics. OPEN
    trades (not yet resolved as of the available data) are excluded from
    PnL stats but counted separately."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    open_count = len(trades) - len(resolved)

    if not resolved:
        return {
            "trade_count": 0, "open_count": open_count,
            "win_count": 0, "loss_count": 0, "expired_count": 0,
            "win_rate": None, "avg_pnl_pct": None, "total_pnl_pct": 0.0,
            "pnl_std": None, "sharpe_like": None,
            "trades_per_year": None, "annualized_sharpe_like": None, "k_ratio": None,
        }

    pnls = pd.Series([t["pnl_pct"] for t in resolved])
    win_count = sum(1 for t in resolved if t["status"] == "WIN")
    loss_count = sum(1 for t in resolved if t["status"] == "LOSS")
    expired_count = sum(1 for t in resolved if t["status"] == "EXPIRED")
    pnl_std = float(pnls.std()) if len(pnls) > 1 else 0.0
    sharpe_like = round(float(pnls.mean() / pnl_std), 3) if pnl_std > 0 else None
    trades_per_year, annualized_sharpe_like = _annualize_sharpe(resolved, sharpe_like)

    return {
        "trade_count": len(resolved),
        "open_count": open_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "expired_count": expired_count,
        "win_rate": round(win_count / len(resolved) * 100, 2),
        "avg_pnl_pct": round(float(pnls.mean()), 2),
        "total_pnl_pct": round(float(pnls.sum()), 2),
        "pnl_std": round(pnl_std, 2),
        "sharpe_like": sharpe_like,
        "trades_per_year": trades_per_year,
        "annualized_sharpe_like": annualized_sharpe_like,
        "k_ratio": compute_k_ratio(resolved),
    }


def compute_cluster_weights(trades: list[dict]) -> list[float]:
    """Per-trade weight correcting for same-day, same-sector correlation --
    trades sharing (entry_date, sector) split a combined weight of 1.0
    evenly, so one correlated event (e.g. 30 Technology signals firing the
    same day during one sector move) counts as ONE effective observation
    in aggregate, not 30 independent ones. Unknown-sector trades (no
    sector_lookup entry at signal time) get weight 1.0 each -- there's no
    basis to assume they're correlated with anything.

    This assumes perfect correlation within a cluster and zero correlation
    across clusters -- a conservative approximation, not a measured
    correlation coefficient, but a real correction on the previous
    behavior of treating every pooled trade as fully independent.

    Returns weights in the same order as `trades`. Combine with
    optimize.py's recency fold_weight (multiply the two) for a single
    per-trade weight that accounts for both effects.
    """
    cluster_sizes: dict[tuple, int] = defaultdict(int)
    keys = []
    for t in trades:
        sector = t.get("sector", "Unknown")
        key = None if sector == "Unknown" else (t["entry_date"], sector)
        keys.append(key)
        if key is not None:
            cluster_sizes[key] += 1

    return [1.0 if key is None else 1.0 / cluster_sizes[key] for key in keys]


def summarize_trades_weighted(trades: list[dict], weights: list[float]) -> dict:
    """Like summarize_trades(), but with an arbitrary explicit per-trade
    weight instead of pooling everything uniformly. `weights` must be the
    same length as `trades`, already filtered to exclude OPEN trades.
    compute_cluster_weights() is one source of weights (same-day/same-sector
    correlation); optimize.py's fold_weight is another (fold recency) --
    combine multiple sources by multiplying them elementwise before calling
    this. Uses Kish's effective sample size instead of raw trade_count, so
    heavy weighting can't let a large but low-weight pool fake a
    trustworthy sample size."""
    total_weight = sum(weights) if weights else 0.0
    if not trades or total_weight <= 0:
        return {
            "trade_count": 0, "effective_trade_count": 0.0,
            "win_rate": None, "avg_pnl_pct": None, "pnl_std": None, "sharpe_like": None,
            "trades_per_year": None, "annualized_sharpe_like": None, "k_ratio": None,
        }

    pnl_s = pd.Series([t["pnl_pct"] for t in trades])
    w_s = pd.Series(weights)
    win_weight = sum(w for t, w in zip(trades, weights) if t["status"] == "WIN")
    weighted_mean = float((pnl_s * w_s).sum() / total_weight)
    weighted_var = float((w_s * (pnl_s - weighted_mean) ** 2).sum() / total_weight)
    weighted_std = weighted_var ** 0.5
    effective_n = float(total_weight ** 2 / (w_s ** 2).sum())  # Kish effective sample size
    sharpe_like = round(weighted_mean / weighted_std, 3) if weighted_std > 0 else None
    # Frequency-in-time uses the RAW trade list, not the correlation-weighted
    # effective_n -- how often trades actually occur on the calendar is a
    # separate question from how independent they are.
    trades_per_year, annualized_sharpe_like = _annualize_sharpe(trades, sharpe_like)

    return {
        "trade_count": len(trades),
        "effective_trade_count": round(effective_n, 1),
        "win_rate": round(win_weight / total_weight * 100, 2),
        "avg_pnl_pct": round(weighted_mean, 2),
        "pnl_std": round(weighted_std, 2),
        "sharpe_like": sharpe_like,
        "trades_per_year": trades_per_year,
        "annualized_sharpe_like": annualized_sharpe_like,
        # k_ratio is computed on the RAW trade list (unweighted equity curve),
        # same "frequency-in-time uses raw trades" rationale as trades_per_year
        # above -- the correlation weighting summarize_trades_weighted() applies
        # to sharpe_like/win_rate doesn't have an equivalent for a sequential
        # equity-curve regression (there's no such thing as a "weighted" curve
        # ordering), so this reuses compute_k_ratio() exactly as-is.
        "k_ratio": compute_k_ratio(trades),
    }


def summarize_by_catalyst(trades: list[dict]) -> dict:
    """Split summarize_trades() output by whether each trade carried a
    Catalyst_Warning at entry -- lets you actually see whether performance
    differs around earnings instead of assuming it doesn't. Requires trades
    simulated with earnings_dates passed through (see run_backtest.py
    --with-catalyst); without that, every trade has catalyst_warning=False
    and this just reports everything under the false bucket."""
    with_catalyst = [t for t in trades if t.get("catalyst_warning")]
    without_catalyst = [t for t in trades if not t.get("catalyst_warning")]
    return {
        "catalyst_warning_true": summarize_trades(with_catalyst),
        "catalyst_warning_false": summarize_trades(without_catalyst),
    }


def summarize_by_period(trades: list[dict], summarize_fn=None) -> dict:
    """Split summarize_trades()-style output by calendar YEAR of each
    trade's entry_date -- lets you see whether an edge holds up
    consistently over time or is concentrated in one calendar
    stretch/regime, distinct from the ticker-holdout split (item 69,
    generalization across TICKERS) or k_ratio (item 77, smoothness within
    the one ordering that actually happened). Trades with no entry_date
    (shouldn't happen -- every simulate_*_signals() call sets it) are
    skipped rather than crashing.

    `summarize_fn` lets callers pass their own weighted/cluster-adjusted
    summarizer (e.g. benchmark_random_entry.py's own summarize()) instead
    of the plain summarize_trades() default -- same pattern as
    optimize.average_holdout_summary()'s own summarize_fn param."""
    if summarize_fn is None:
        summarize_fn = summarize_trades
    buckets: dict[int, list[dict]] = {}
    for t in trades:
        entry_date = t.get("entry_date")
        if entry_date is None:
            continue
        buckets.setdefault(entry_date.year, []).append(t)
    return {str(year): summarize_fn(bucket) for year, bucket in sorted(buckets.items())}


VOLATILITY_REGIME_ROLLING_WINDOW_DAYS = 365  # same rolling-not-expanding convention


                                               # levels.SKEW_REGIME_ROLLING_WINDOW_DAYS
                                               # already established (2026-09-03) and the
                                               # same real lesson that motivated it: an
                                               # EXPANDING-since-day-1 baseline stays
                                               # anchored to a stale historical average as
                                               # markets secularly drift, understating how
                                               # "elevated" a recent vol spike really is
                                               # relative to what's actually been typical
                                               # lately. A bounded trailing window instead.
VOLATILITY_REALIZED_WINDOW_DAYS = 20  # trading days of daily returns used for the


                                       # realized-volatility measure itself (roughly
                                       # one calendar month) -- short enough to react to
                                       # a real regime shift, long enough not to be
                                       # dominated by one or two outlier days


def compute_volatility_regime_series(
    market_ohlcv: pd.DataFrame,
    realized_window_days: int = VOLATILITY_REALIZED_WINDOW_DAYS,
    rolling_median_days: int = VOLATILITY_REGIME_ROLLING_WINDOW_DAYS,
) -> pd.Series:
    """Date-indexed Series of "elevated"/"normal" market-volatility-regime
    labels, computed from the market-uptrend proxy's (SPY by convention,
    see MARKET_INDEX_TICKER) OWN realized volatility -- purely a DIAGNOSTIC
    breakdown axis (see summarize_by_volatility_regime() below), not a new
    strategy gate/filter anywhere.

    Motivated by a real, structural fact about this whole codebase: EVERY
    simulate_*_signals() function hard-gates on market_uptrend_from_frame()
    (Close >= SMA_TREND) before ever considering a signal, so a simple
    "bull vs bear" regime breakdown would be moot here -- bear-market days
    are already fully excluded by construction for every strategy. The
    volatility dimension is NOT excluded by that gate (a market can be in a
    confirmed uptrend while realized volatility is elevated or subdued),
    making it the more informative axis for "does this edge hold up in
    different market conditions" within the population every strategy
    actually trades in.

    `realized_window_days` (default 20, ~1 trading month) of trailing daily
    return std defines "how volatile has the market been LATELY"; whether
    that reads as "elevated" or "normal" on any given day is relative to
    its OWN trailing `rolling_median_days` (default 365) median -- a
    BOUNDED rolling window, not an expanding-since-inception one, same
    "avoid a stale, secularly-drifted baseline" reasoning
    SKEW_REGIME_ROLLING_WINDOW_DAYS already established (2026-09-03) for
    the CBOE SKEW regime filter. Days before enough history exists for
    either window read "normal" (conservative default, matches every
    other optional-data-dependent field's "missing data never fabricates
    a signal" convention elsewhere in this codebase)."""
    daily_returns = market_ohlcv["Close"].pct_change()
    realized_vol = daily_returns.rolling(realized_window_days).std()
    rolling_baseline = realized_vol.rolling(f"{rolling_median_days}D").median()
    elevated = realized_vol > rolling_baseline
    return elevated.map({True: "elevated", False: "normal"}).fillna("normal")


def summarize_by_volatility_regime(
    trades: list[dict], regime_series: pd.Series, summarize_fn=None,
) -> dict:
    """Split summarize_trades()-style output by each trade's own entry_date
    volatility regime (see compute_volatility_regime_series()) -- the
    volatility-axis counterpart to summarize_by_period()'s calendar-year
    split, same shape/conventions exactly (including the same
    `summarize_fn` override pattern). A trade whose entry_date isn't found
    in `regime_series` (shouldn't happen given the market data covers the
    full backtest window, but this is a read path, not a gate) is treated
    as "normal" rather than skipped or crashing."""
    if summarize_fn is None:
        summarize_fn = summarize_trades
    regime_by_date = regime_series.to_dict()
    buckets: dict[str, list[dict]] = {"elevated": [], "normal": []}
    for t in trades:
        entry_date = t.get("entry_date")
        if entry_date is None:
            continue
        regime = regime_by_date.get(pd.Timestamp(entry_date), "normal")
        buckets[regime].append(t)
    return {regime: summarize_fn(bucket) for regime, bucket in buckets.items()}


def _concurrency_at_entry(resolved: list[dict]) -> list[int]:
    """For each trade in `resolved` (same order as input), how many trades
    -- including itself -- had an OPEN `[entry_date, exit_date]` interval
    (closed on both ends) at the moment THIS trade itself entered. Pure
    O(n log n) event-delta sweep over dates already present on every
    resolved trade dict (no OHLCV, no new data dependency): +1 on
    `entry_date`, -1 the day AFTER `exit_date` (so a position still counts
    as open ON its own exit_date -- see compute_max_drawdown()'s own
    docstring for why that closed-interval convention was chosen: there's
    no intraday fill-sequencing data across different tickers to know
    whose capital freed up first on a shared boundary day).

    Every trade's own entry event guarantees its own date key exists in
    the running tally, so every returned count is always >= 1 -- callers
    never need a divide-by-zero guard on `1 / count`."""
    delta: dict = defaultdict(int)
    for t in resolved:
        delta[t["entry_date"]] += 1
        delta[t["exit_date"] + timedelta(days=1)] -= 1
    running = 0
    concurrency_as_of: dict = {}
    for d in sorted(delta):
        running += delta[d]
        concurrency_as_of[d] = running
    return [concurrency_as_of[t["entry_date"]] for t in resolved]


def _batched_weighted_returns(resolved: list[dict]) -> list[tuple]:
    """Per-trade weighted returns (`pnl_pct/100` scaled by
    `1/_concurrency_at_entry()`), combined ADDITIVELY within each shared
    `exit_date`, returned as `(exit_date, batch_return)` pairs -- one per
    distinct exit_date, in chronological order -- see
    compute_max_drawdown()'s own docstring for why same-day exits must be
    summed, not compounded against each other one at a time (order-dependent
    phantom-peak bug, verified numerically in
    tests/test_max_drawdown.py::test_same_day_exits_are_order_independent).
    The date half of each pair exists for compute_k_ratio()'s benefit (it
    needs a calendar axis to regress against); compute_max_drawdown() itself
    only needs the return values."""
    concurrency = _concurrency_at_entry(resolved)
    batches: dict = defaultdict(float)
    for t, c in zip(resolved, concurrency):
        batches[t["exit_date"]] += (t["pnl_pct"] / 100) / c
    return [(d, batches[d]) for d in sorted(batches)]


def compute_max_drawdown(trades: list[dict]) -> float | None:
    """Max peak-to-trough decline (%) of a CONCURRENCY-WEIGHTED equity
    curve -- approximates real portfolio risk instead of the old model's
    fiction that every pooled trade (across up to 407 tickers) used 100%
    of capital, one at a time, with zero overlap. optimize.py's default
    single-objective search only ever optimizes sharpe_like (mean/stdev),
    which is blind to tail risk: a config can "win" by juicing consistency
    while quietly tolerating a nastier worst-case drawdown. This gives
    Optuna's optional multi-objective mode (see optimize.py
    --multi-objective) a second axis to actually see that.

    2026-08-31 redesign: found via a real multi-objective search for
    ma_crossover that the OLD sequential-100%-per-trade model produced
    94.8% "drawdown" for an UNTUNED DEFAULT_CONFIG baseline and 98.5% for
    a validated candidate -- numbers too saturated near 100% to
    discriminate a good config from a bad one, because this is a
    swing-trading strategy scanning up to 407 tickers daily with positions
    held up to config.max_holding_days at a time: many trades genuinely
    overlap in real calendar time, and treating each as if it alone
    consumed the whole account manufactures a far worse curve than any
    real (even lightly diversified) account would show.

    Model: `_concurrency_at_entry()` sweep-line-counts how many OTHER
    trades (including itself) had an open [entry_date, exit_date] interval
    (both dates inclusive) at the moment THIS trade itself entered --
    using only entry_date/exit_date, both already present on every
    resolved trade dict (see settlement.py's settle_trade()/
    settle_trade_with_trailing()). Each trade's position_weight = 1 / that
    count -- a fixed snapshot of its estimated share of capital at the
    moment it opened, not a dynamically renormalized daily allocation.
    Trades sharing an exit_date are combined ADDITIVELY (weighted-sum),
    never compounded against each other one at a time -- see
    _batched_weighted_returns()'s own docstring for why. Batches are
    compounded across distinct exit_date values in chronological order:
    P&L realizes (and capital frees up) at EXIT, not entry, so that's when
    each trade's weighted return books into the running equity curve.

    Reduces to the exact OLD sequential-100%-per-trade behavior whenever
    there is no real overlap (every position_weight is 1.0, one trade per
    batch) -- verified in tests/test_max_drawdown.py.

    Still an approximation, not a dollar-accurate portfolio simulation:
      - No hard cap enforces that concurrently-active weights sum to <=1
        -- position_weight is a per-trade snapshot at ITS OWN entry, never
        renormalized against trades that open/close around it, so very
        high-concurrency clusters can be under- or over-weighted relative
        to a real fixed-capital account.
      - No daily mark-to-market: an open position's equity impact is
        invisible until its exit_date batch resolves, so intraday/interday
        unrealized drawdown while a position is still open isn't
        captured -- this function only ever receives lightweight
        trade-summary dicts, never OHLCV, by design.
      - Same-day entry/exit boundary convention: a trade exiting on day D
        and a trade entering on day D are treated as concurrent that day
        (closed-interval convention) -- there's no intraday
        fill-sequencing data across different tickers to know whose
        capital freed up first.
      - Still a proxy for LOSS-CLUSTERING risk under a plausible
        capital-allocation approximation, not a literal "what would my
        account balance have done" claim -- same spirit as the old model,
        just a materially less misleading one.

    Returns None if there are no resolved (non-OPEN) trades, or if any
    resolved trade is missing entry_date/exit_date (same
    degrade-gracefully convention as compute_k_ratio()/annualized-sharpe
    helpers use for live-Mongo-pooled minimal dicts -- no current caller
    triggers this, but it avoids a surprising crash if one ever does)."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if not resolved:
        return None
    if any("entry_date" not in t or "exit_date" not in t for t in resolved):
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for _, r in _batched_weighted_returns(resolved):
        equity *= (1 + r)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return round(max_dd * 100, 2)


def _exit_date_batch_layout(resolved: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Precomputes the (order, boundaries) pair `monte_carlo_drawdown()`
    needs to batch same-exit_date trades additively across every
    simulation without re-deriving the layout each time. `order` is a
    permutation of `resolved`'s indices sorted by exit_date; `boundaries`
    is the start index (within the `order`-permuted array) of each
    distinct exit_date's group, ready to feed straight into
    `np.add.reduceat`."""
    order = np.array(sorted(range(len(resolved)), key=lambda i: resolved[i]["exit_date"]), dtype=int)
    sorted_exit_dates = [resolved[i]["exit_date"] for i in order]
    boundaries = [0]
    for i in range(1, len(sorted_exit_dates)):
        if sorted_exit_dates[i] != sorted_exit_dates[i - 1]:
            boundaries.append(i)
    return order, np.array(boundaries, dtype=int)


def monte_carlo_drawdown(trades: list[dict], n_simulations: int = 1000, seed: int = 42) -> dict | None:
    """How much of compute_max_drawdown()'s single chronological-order
    result is just the LUCK of the particular sequence a fixed set of
    trades happened to occur in? Reshuffles the same resolved trades'
    pnl_pct values into n_simulations random orderings and reports the
    resulting max_drawdown distribution (same concurrency-weighted,
    exit-date-batched equity model as compute_max_drawdown() -- see that
    function's own docstring for the full model/limits) -- a materially
    worse P95/worst case than the actual chronological run means the
    smooth-looking real equity curve was partly lucky ordering, not just
    genuine edge quality.

    2026-08-31: composes with compute_max_drawdown()'s concurrency
    weighting by keeping each trade's OWN entry_date/exit_date -- and
    therefore its own position_weight and its own exit-date batch
    membership -- FIXED across every simulation; only WHICH pnl_pct value
    lands on which (fixed) weighted trade slot is reshuffled. Deliberate:
    concurrency/weighting is a property of when the strategy's signals
    actually fired and how long positions were actually held, which this
    check isn't asking about -- that's a sample-generalization question
    already covered by ticker-holdout (item 69), not an ordering-luck
    question. Weights and exit-date batch layout are computed ONCE via
    _exit_date_batch_layout() and reused across all n_simulations, not
    re-derived per simulation.

    Reshuffles ORDER only, not trade SELECTION -- this answers "how lucky
    was the sequence", not "how lucky was the sample" (that's what
    ticker-holdout already checks, see item 69).

    Returns None with fewer than 3 resolved trades (mirrors
    compute_k_ratio()'s floor -- not enough trades for a distribution to
    mean anything), or if any resolved trade is missing
    entry_date/exit_date (same degrade-gracefully convention
    compute_max_drawdown() now uses)."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if len(resolved) < 3:
        return None
    if any("entry_date" not in t or "exit_date" not in t for t in resolved):
        return None

    weights = np.array([1.0 / c for c in _concurrency_at_entry(resolved)], dtype=float)
    pnl_pct = np.array([t["pnl_pct"] for t in resolved], dtype=float)
    order, boundaries = _exit_date_batch_layout(resolved)

    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(pnl_pct, (n_simulations, 1)), axis=1)
    weighted = (shuffled / 100) * weights
    weighted = weighted[:, order]
    batch_returns = np.add.reduceat(weighted, boundaries, axis=1)

    equity = np.cumprod(1 + batch_returns, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
    max_dd_pct = drawdown.max(axis=1) * 100

    return {
        "n_simulations": n_simulations,
        "n_trades": len(resolved),
        "actual_chronological_max_drawdown": compute_max_drawdown(trades),
        "monte_carlo_mean": round(float(max_dd_pct.mean()), 2),
        "monte_carlo_median": round(float(np.median(max_dd_pct)), 2),
        "monte_carlo_p95": round(float(np.percentile(max_dd_pct, 95)), 2),
        "monte_carlo_worst": round(float(max_dd_pct.max()), 2),
    }


def permutation_test_gap(
    real_trades: list[dict], random_trades: list[dict], summarize_fn=None,
    n_permutations: int = 1000, seed: int = 42,
) -> dict:
    """Quantifies a REAL confidence figure on the real-vs-random GAP itself
    (2026-09-13, improvements.txt item 135) -- a different question from
    every existing check: DSR (optimize.deflated_sharpe_ratio()) corrects
    for selection bias across MANY Optuna trials; this asks, for ONE
    already-fixed real-vs-random comparison, how likely a gap this large
    would be to occur by pure chance if REAL and RANDOM trades were
    actually interchangeable (the null hypothesis: no genuine timing
    skill, real_trades and random_trades differ only by labeling).

    Standard permutation test: pools real_trades + random_trades together,
    then repeatedly reshuffles that pool into two groups of the SAME sizes
    as the real actual real/random split, recomputing the identical gap
    statistic (real sharpe_like minus random sharpe_like, via
    `summarize_fn`) on each shuffle to build a null distribution -- valid
    for a ratio statistic like sharpe_like, not just a plain mean (a
    permutation test makes no distributional assumption about the test
    statistic itself, unlike the closed-form DSR math). `p_value` is the
    fraction of null-distribution gaps that meet or exceed the OBSERVED
    gap (one-sided -- this project only ever cares whether REAL beats
    RANDOM, never the reverse) -- a LOW p_value means the observed gap
    would be unusual/rare if there were truly no timing skill, i.e. real
    evidence of a genuine edge, not just favorable framing of a single
    lucky comparison.

    `summarize_fn` defaults to the plain unweighted summarize_trades() --
    pass a cluster-weighted closure (same pattern summarize_by_period()'s
    own `summarize_fn` override uses) to match whatever weighting the
    caller's own headline gap already used, for a fully apples-to-apples
    p-value.

    Returns {"observed_gap":, "p_value": float in [0,1] or None if either
    summary's own sharpe_like was undefined, "null_mean":, "null_std":,
    "n_permutations": how many shuffles actually produced a defined gap
    (can be fewer than requested if a shuffle's own random split happens
    to produce an all-identical-pnl group with undefined sharpe_like)}."""
    if summarize_fn is None:
        summarize_fn = summarize_trades

    real_summary = summarize_fn(real_trades)
    random_summary = summarize_fn(random_trades)
    real_sharpe = real_summary.get("sharpe_like")
    random_sharpe = random_summary.get("sharpe_like")
    if real_sharpe is None or random_sharpe is None:
        return {"observed_gap": None, "p_value": None, "null_mean": None, "null_std": None, "n_permutations": 0}
    observed_gap = real_sharpe - random_sharpe

    pooled = list(real_trades) + list(random_trades)
    n_real = len(real_trades)
    rng = random.Random(seed)
    null_gaps = []
    for _ in range(n_permutations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        perm_real_summary = summarize_fn(shuffled[:n_real])
        perm_random_summary = summarize_fn(shuffled[n_real:])
        perm_real_sharpe = perm_real_summary.get("sharpe_like")
        perm_random_sharpe = perm_random_summary.get("sharpe_like")
        if perm_real_sharpe is not None and perm_random_sharpe is not None:
            null_gaps.append(perm_real_sharpe - perm_random_sharpe)

    if not null_gaps:
        return {"observed_gap": round(observed_gap, 4), "p_value": None, "null_mean": None, "null_std": None, "n_permutations": 0}

    p_value = sum(1 for g in null_gaps if g >= observed_gap) / len(null_gaps)
    null_series = pd.Series(null_gaps)
    return {
        "observed_gap": round(observed_gap, 4),
        "p_value": round(p_value, 4),
        "null_mean": round(float(null_series.mean()), 4),
        "null_std": round(float(null_series.std()), 4) if len(null_gaps) > 1 else None,
        "n_permutations": len(null_gaps),
    }


def compute_k_ratio(trades: list[dict]) -> float | None:
    """K-ratio: how CONSISTENTLY an equity curve compounds over calendar
    time, distinct from sharpe_like (mean/stdev of per-trade returns) which
    is blind to *ordering* -- two configs with identical win_rate/sharpe_like
    can still differ sharply in whether gains accrue steadily or in one
    lumpy cluster. Fits an OLS regression of log(equity) against elapsed
    calendar days (log-space so compounding growth is linear, the standard
    K-ratio setup). K-ratio is the regression slope's t-statistic (slope
    divided by its own standard error) -- large and positive means equity
    grew steadily and reliably in a straight line; near zero or negative
    means the trend is noisy, flat, or reversing.

    2026-08-31: reworked onto the same CONCURRENCY-WEIGHTED equity curve as
    compute_max_drawdown() (reuses its own `_batched_weighted_returns()`
    helper directly) -- previously compounded every resolved trade
    one-at-a-time in entry_date order as if each alone used 100% of capital,
    the identical "not a real concurrent-position portfolio sim" flaw
    compute_max_drawdown() was redesigned to fix earlier the same day (see
    that function's own docstring for the full model/limits). Each trade's
    position_weight = 1 / how many trades were concurrently open at the
    moment it entered; trades sharing an exit_date combine additively into
    one batch. The regression's calendar axis is now each BATCH's exit_date
    (when its P&L actually realizes and its weighted return books into the
    curve), not each trade's entry_date -- consistent with
    compute_max_drawdown()'s own "P&L realizes at EXIT, not entry" choice.
    Reduces to the exact old per-trade-in-entry-order behavior whenever
    there's no real overlap (every weight is 1.0, one trade per batch,
    exit_date order matches entry_date order for non-overlapping trades).

    Deliberately RAW, not annualized or scaled by sample size -- same
    "comparable only within a similarly-sized/timed run" convention
    sharpe_like already follows (see strategy-validation-pipeline memory
    point 12): only compare k_ratio between runs over the same calendar
    window and trade universe, never across runs of different length.

    Returns None with fewer than 3 resolved trades, when any resolved trade
    is missing entry_date or exit_date (same degrade-gracefully convention
    as compute_max_drawdown()/_annualize_sharpe() -- real callers pool live
    Trade_Outcomes Mongo documents into a minimal {"status", "pnl_pct"}
    shape with neither field, see optimize.py's report_live_outcomes_context(),
    settle_trades.py), when fewer than 3 distinct exit_date batches remain
    after combining same-exit_date trades (no reliable time axis to regress
    against -- can happen even with >=3 resolved trades if several share an
    exit_date), or in the degenerate zero-residual-variance case (avoids a
    divide-by-zero rather than returning a nonsensical infinite value)."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if len(resolved) < 3 or any("entry_date" not in t or "exit_date" not in t for t in resolved):
        return None

    batched = _batched_weighted_returns(resolved)
    if len(batched) < 3:
        return None

    start = batched[0][0]
    equity = 1.0
    xs, ys = [], []
    for exit_date, r in batched:
        equity *= (1 + r)
        if equity <= 0:
            return None
        xs.append(float((exit_date - start).days))
        ys.append(math.log(equity))

    x = pd.Series(xs)
    y = pd.Series(ys)
    n = len(x)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0:
        return None
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - (intercept + slope * x)
    residual_var = float((residuals ** 2).sum()) / (n - 2)
    se_slope = (residual_var / sxx) ** 0.5
    if se_slope == 0:
        return None
    return round(slope / se_slope, 3)
