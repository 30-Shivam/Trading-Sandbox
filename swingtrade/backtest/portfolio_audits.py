"""Portfolio-constrained replay, buy-and-hold baseline and post-hoc audits (dividend drag, overlap, capacity, K-ratio)."""

from collections import defaultdict
import pandas as pd


def simulate_sector_rotation_rebalanced(
    sector_panel: pd.DataFrame,
    lookback_days: int,
    top_n: int,
    rebalance_frequency_days: int,
) -> list[dict]:
    """A STRUCTURALLY DIFFERENT backtest paradigm from every other function
    in this module: an actual REBALANCED PORTFOLIO (hold a sector while it
    keeps qualifying, exit only when it genuinely falls out of the top-N,
    re-evaluated periodically) rather than a discrete daily signal reusing
    a fixed ATR stop/target/max-holding-day exit. This codebase has never
    had a real rebalance-portfolio engine before (see
    simulate_sector_rotation_signals()'s own docstring, which explicitly
    notes the lack) -- this is the first one, built specifically to test
    whether the discrete daily-signal version of sector rotation's own
    negative result (2026-09-16, TUNE/HOLDOUT disagree, NOT promoted) was a
    genuine absence of edge or an artifact of firing on EVERY day a sector
    qualified (98.6% same-ticker overlap, "stays invested" rather than
    "rotates") instead of holding a position across the whole time it
    qualifies.

    Deliberately operates on SECTOR ETFs directly (not individual member
    tickers) -- the classic, real-world-executable form of sector rotation
    (an investor can actually buy/hold/sell XLK/XLF/etc. with real capital,
    no stock-picking-within-a-sector step needed), and it sidesteps the
    prior attempt's own overlap problem entirely (11 positions max, never
    hundreds of overlapping per-ticker entries).

    `sector_panel`: wide DataFrame of Close prices, one column per sector
    NAME (same construction as compute_sector_rotation_rank_frame()'s own
    `sector_panel` argument). `lookback_days`: trailing-return formation
    window (same role as sector_rotation_lookback_days). `top_n`: how many
    of the ~11 sectors to hold at once. `rebalance_frequency_days`: how
    often (in trading days) to re-evaluate which sectors qualify -- between
    rebalance dates, an already-held sector is held UNCHANGED regardless of
    its price path (no ATR stop/target here at all -- this is a genuine
    buy-and-hold-until-rotated-out position, not a swing trade).

    Returns a list of trade dicts SCHEMA-COMPATIBLE with summarize_trades()/
    compute_max_drawdown()/compute_k_ratio() (`ticker` is actually the
    SECTOR name here, `entry_date`/`exit_date`/`pnl_pct`/`status` all
    present) -- one dict per CONTIGUOUS holding span (a sector held across
    several consecutive rebalance dates without ever dropping out of the
    top-N produces ONE trade spanning the whole span, not one per
    rebalance check). A sector still held at the final rebalance date is
    closed out (marked WIN/LOSS, never left OPEN) at that date's own
    price -- deliberately different from every other strategy's "OPEN =
    excluded from PnL stats" convention, since the goal here is a complete,
    fair total-return comparison over the whole tested window, not an
    honest "no outcome yet" for a signal that fired near the data's edge."""
    trailing_return = sector_panel.pct_change(periods=lookback_days)
    rebalance_dates = sector_panel.index[lookback_days::rebalance_frequency_days]

    current_holdings: dict[str, dict] = {}  # sector -> {"entry_date", "entry_price"}
    trades: list[dict] = []

    for rebalance_date in rebalance_dates:
        row = trailing_return.loc[rebalance_date].dropna()
        target = set(row.nlargest(top_n).index) if len(row) > 0 else set()

        for sector in list(current_holdings):
            if sector in target:
                continue
            entry = current_holdings.pop(sector)
            exit_price = float(sector_panel.loc[rebalance_date, sector])
            trades.append(_close_rotation_trade(sector, entry, rebalance_date, exit_price))

        for sector in target:
            if sector in current_holdings:
                continue
            current_holdings[sector] = {
                "entry_date": rebalance_date,
                "entry_price": float(sector_panel.loc[rebalance_date, sector]),
            }

    final_date = sector_panel.index[-1]
    for sector, entry in current_holdings.items():
        exit_price = float(sector_panel.loc[final_date, sector])
        trades.append(_close_rotation_trade(sector, entry, final_date, exit_price))

    return trades


def _close_rotation_trade(sector: str, entry: dict, exit_date, exit_price: float) -> dict:
    """Shared close-out helper for simulate_sector_rotation_rebalanced() --
    both the mid-run "fell out of top-N" exit and the end-of-window
    mark-to-market close-out build the identical trade-dict shape."""
    entry_price = entry["entry_price"]
    pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price else 0.0
    return {
        "ticker": sector,
        "entry_date": entry["entry_date"],
        "exit_date": exit_date,
        "buy_price": entry_price,
        "sell_price": exit_price,
        "pnl_pct": pnl_pct,
        "status": "WIN" if pnl_pct > 0 else ("LOSS" if pnl_pct < 0 else "EXPIRED"),
    }


def compute_equal_weight_buy_and_hold(sector_panel: pd.DataFrame, lookback_days: int) -> list[dict]:
    """The natural passive-benchmark counterpart to
    simulate_sector_rotation_rebalanced() -- buy every sector that has
    valid history ONCE (at the same date the rotation strategy's own first
    rebalance would use, so both start from an identical, fair starting
    line) and hold to the end, unweighted by any rotation skill. Answers
    'does picking winners beat just owning the whole sector universe
    equally', the standard evaluation baseline for a rotation/momentum
    strategy -- a real, executable alternative (an investor could equally
    just buy-and-hold all 11 SPDR sector ETFs), not a synthetic random-entry
    timing baseline like every discrete-signal strategy in this module
    uses (there's no daily entry DECISION here to randomize against).

    Returns the same trade-dict shape as simulate_sector_rotation_rebalanced()
    -- one row per sector, entry at the shared start date, exit at the
    panel's own final date."""
    start_date = sector_panel.index[lookback_days]
    final_date = sector_panel.index[-1]
    trades = []
    for sector in sector_panel.columns:
        entry_price = sector_panel.loc[start_date, sector]
        exit_price = sector_panel.loc[final_date, sector]
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        trades.append(_close_rotation_trade(
            sector, {"entry_date": start_date, "entry_price": float(entry_price)}, final_date, float(exit_price),
        ))
    return trades


def simulate_portfolio_constrained(
    trades: list[dict],
    starting_capital: float,
    position_budget: float,
    max_sector_allocation_pct: float | None = None,
    max_total_deployed_pct: float | None = None,
    sector_lookup: dict[str, str] | None = None,
    group_key: str | None = None,
    max_positions_per_ticker: int | None = None,
) -> dict:
    """Replays already-simulated signals through a REAL, single, finite-
    capital account -- the question nothing before this checked (2026-09-13,
    per allocation.allocate_capital()'s own long-standing docstring caveat:
    "a per-ticker walk-forward backtest doesn't model simultaneous
    cross-ticker exposure either"). Every other metric in this module
    (sharpe_like, k_ratio, compute_max_drawdown's own concurrency-weighted
    equity curve) treats every signal a strategy ever generates as if it
    got taken -- fine for asking "does this signal carry timing
    information," but silent about a different, equally real question: of
    every signal generated, how many could ONE real account, with a fixed
    starting_capital and a flat position_budget per trade (same flat-sizing
    convention ingest.py's own live default uses when no --risk-amount is
    given), have actually AFFORDED to take, and what does that account's
    own dollar equity curve/drawdown/CAGR look like once capital
    constraints are respected rather than pooling every trade's % return as
    if capital were unlimited?

    Reuses allocation.allocate_capital()'s exact three real-world
    constraints (funds/sector-cap/portfolio-cap), applied here as a
    chronological REPLAY across the whole backtest window instead of a
    single day's greedy walk -- trades are processed in entry_date order;
    a trade whose cost can't currently be funded is marked "taken": False
    with a `reason` (mirrors allocate_capital()'s own "Insufficient Funds"/
    "Sector Limit Reached"/"Portfolio Limit Reached" signal relabeling, same
    non-consuming "keep walking, don't stop the whole backtest" semantics)
    rather than crashing or silently still counting it. `max_sector_allocation_pct`/
    `max_total_deployed_pct` are both expressed as a fraction of
    `starting_capital` (NOT a dynamically growing portfolio value) --
    a deliberate simplification, documented below.

    Capital is committed at a trade's entry_date and released back
    (at its REALIZED dollar pnl, not just the cost basis) at its exit_date
    -- same entry/exit-date concurrency convention compute_max_drawdown()
    already established, reused here instead of inventing a second one.
    A same-day exit is assumed to free capital in time to fund a same-day
    new entry (closed-interval release-before-commit ordering) -- the same
    "no intraday fill-sequencing data across tickers" limitation
    compute_max_drawdown() already documents, not a new one.

    Known, deliberate simplifications (mirroring compute_max_drawdown()'s
    own documented gaps, not fixed here either):
      - No daily mark-to-market -- the equity curve only gets a new sample
        at each entry/exit EVENT, never between them, so unrealized
        drawdown of a still-open position is invisible until it resolves.
      - Sector/portfolio caps are a fraction of `starting_capital`, not a
        dynamically growing/shrinking current portfolio value -- caps
        don't loosen as the account compounds gains (a conservative
        simplification, not an attempt at dollar-exact realism).
      - Ties at the same entry_date break by `trade_score` DESCENDING when
        present (mirrors the live dashboard's own Trade_Score-sorted
        `allocate_capital()` greedy walk -- the highest-conviction signal
        gets first claim on the limited capital) -- trades missing a
        `trade_score` (not every strategy's trade dicts carry one, see
        ic_tracking.backtest_ic_check()'s own docstring) fall back to a
        neutral 0 and, among themselves, `trades`' own original order
        (Python's sort is stable). This matters most when POOLING trades
        from multiple strategies into one shared account (2026-09-13,
        improvements.txt item 142) -- a single strategy's own signals
        rarely collide on the exact same entry_date/ticker, but a real
        multi-strategy portfolio does exactly this competition every day.

    Trades missing entry_date/exit_date are silently excluded from the
    replay entirely (same degrade-gracefully convention compute_max_drawdown()
    uses) -- not counted in n_signals, not skippable, not takeable.

    `group_key` (optional, e.g. `"signal"`, present on every trade dict
    every `simulate_*_signals()` function already appends -- 2026-09-13,
    improvements.txt item 142) -- when given, also returns `taken_by_group`
    (a `{group value: count actually taken}` dict) alongside the usual
    aggregate stats. Built for POOLING trades from multiple strategies into
    one shared account (see `benchmark_multi_strategy_portfolio.py`) --
    without this, a pooled multi-strategy run tells you the combined
    capital-constrained outcome but not which strategy's signals actually
    won the limited capital versus got crowded out. None (default)
    preserves the exact original return shape.

    `max_positions_per_ticker` (optional, 2026-09-14, improvements.txt item
    151) -- caps how many OPEN positions the account will hold in the SAME
    ticker at once, mirroring the sector/portfolio caps' own opt-in design.
    None (default) preserves the exact original behavior: unlimited same-
    ticker exposure, since every `simulate_*_signals()` function walks
    eligible days independently with no notion of "already holding this
    ticker" -- see `swingtrade.audit_same_ticker_overlap()`'s own docstring
    for the real, measured gap this exists to let a caller close (a real
    15-ticker/5-year `ma_crossover` sample showed ~3% of trades touching a
    genuine same-ticker overlap). A trade that would breach this cap is
    skipped (`n_skipped_ticker_limit`), same non-consuming "keep walking"
    semantics as the sector/portfolio caps.

    Returns a dict: `starting_capital`, `ending_equity`, `total_return_pct`,
    `cagr_pct` (None if the date span is under a day), `max_drawdown_pct`,
    `n_signals` (resolved trades actually eligible for replay), `n_taken`,
    `n_skipped_insufficient_capital`, `n_skipped_sector_limit`,
    `n_skipped_portfolio_limit`, `n_skipped_ticker_limit`,
    `pct_signals_taken` (n_taken/n_signals as a %, None if n_signals is 0),
    plus `taken_by_group` if `group_key` was given."""
    sector_lookup = sector_lookup or {}
    eligible = [t for t in trades if t.get("status") != "OPEN" and "entry_date" in t and "exit_date" in t]
    n_signals = len(eligible)
    if n_signals == 0:
        result = {
            "starting_capital": starting_capital, "ending_equity": starting_capital,
            "total_return_pct": 0.0, "cagr_pct": None, "max_drawdown_pct": None,
            "n_signals": 0, "n_taken": 0, "n_skipped_insufficient_capital": 0,
            "n_skipped_sector_limit": 0, "n_skipped_portfolio_limit": 0,
            "n_skipped_ticker_limit": 0, "pct_signals_taken": None,
        }
        if group_key:
            result["taken_by_group"] = {}
        return result

    ordered = sorted(eligible, key=lambda t: (t["entry_date"], -t.get("trade_score", 0.0)))
    sector_cap_dollars = (
        max_sector_allocation_pct * starting_capital
        if max_sector_allocation_pct and max_sector_allocation_pct > 0 else None
    )
    total_deployed_cap_dollars = (
        max_total_deployed_pct * starting_capital
        if max_total_deployed_pct and max_total_deployed_pct > 0 else None
    )

    cash = starting_capital
    open_positions: list[dict] = []  # {"exit_date":, "cost":, "sector":, "pnl_pct":, "ticker":}
    deployed_now = 0.0
    sector_spent: dict[str, float] = defaultdict(float)
    open_count_by_ticker: dict = defaultdict(int)
    equity_curve: list[tuple] = [(ordered[0]["entry_date"], starting_capital)]
    n_taken = n_skipped_funds = n_skipped_sector = n_skipped_portfolio = n_skipped_ticker = 0
    taken_by_group: dict = defaultdict(int) if group_key else None

    def _release_through(as_of) -> None:
        nonlocal cash, deployed_now
        still_open = []
        for p in open_positions:
            if p["exit_date"] <= as_of:
                cash += p["cost"] * (1 + p["pnl_pct"] / 100)
                deployed_now -= p["cost"]
                sector_spent[p["sector"]] -= p["cost"]
                open_count_by_ticker[p["ticker"]] -= 1
                equity_curve.append((p["exit_date"], cash + deployed_now))
            else:
                still_open.append(p)
        open_positions[:] = still_open

    for t in ordered:
        _release_through(t["entry_date"])
        sector = sector_lookup.get(t.get("ticker"), t.get("sector", "Unknown"))
        ticker = t.get("ticker")
        cost = position_budget

        if cost > cash:
            n_skipped_funds += 1
            continue
        if sector_cap_dollars is not None and sector_spent[sector] + cost > sector_cap_dollars:
            n_skipped_sector += 1
            continue
        if total_deployed_cap_dollars is not None and deployed_now + cost > total_deployed_cap_dollars:
            n_skipped_portfolio += 1
            continue
        if max_positions_per_ticker is not None and open_count_by_ticker[ticker] >= max_positions_per_ticker:
            n_skipped_ticker += 1
            continue

        cash -= cost
        deployed_now += cost
        sector_spent[sector] += cost
        open_count_by_ticker[ticker] += 1
        open_positions.append({"exit_date": t["exit_date"], "cost": cost, "sector": sector, "pnl_pct": t["pnl_pct"], "ticker": ticker})
        n_taken += 1
        if group_key:
            taken_by_group[t.get(group_key, "Unknown")] += 1
        equity_curve.append((t["entry_date"], cash + deployed_now))

    # Release whatever's still open at the very end so ending_equity is fully realized.
    if open_positions:
        _release_through(max(p["exit_date"] for p in open_positions))

    ending_equity = cash
    equity_curve.sort(key=lambda e: e[0])
    peak = starting_capital
    max_dd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)

    span_days = (ordered[-1]["exit_date"] - ordered[0]["entry_date"]).days if n_taken else 0
    cagr_pct = None
    if span_days > 0 and ending_equity > 0 and starting_capital > 0:
        years = span_days / 365.25
        cagr_pct = round(((ending_equity / starting_capital) ** (1 / years) - 1) * 100, 2)

    result = {
        "starting_capital": starting_capital,
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round((ending_equity / starting_capital - 1) * 100, 2),
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_signals": n_signals,
        "n_taken": n_taken,
        "n_skipped_insufficient_capital": n_skipped_funds,
        "n_skipped_sector_limit": n_skipped_sector,
        "n_skipped_portfolio_limit": n_skipped_portfolio,
        "n_skipped_ticker_limit": n_skipped_ticker,
        "pct_signals_taken": round(n_taken / n_signals * 100, 2),
    }
    if group_key:
        result["taken_by_group"] = dict(taken_by_group)
    return result


def audit_dividend_drag(trades: list[dict], dividend_history: dict) -> dict:
    """Quantifies a real, previously entirely unexamined gap in this
    project's backtest realism (2026-09-13, per the user's "fully
    comprehensive of all variables" directive): every settled trade's
    `pnl_pct` is computed purely from price movement (buy_price vs.
    exit_price, see settlement.py) -- a real position that happened to be
    held THROUGH a stock's ex-dividend date would, in reality, also
    receive that dividend payment, which this backtest currently credits
    nowhere. Slippage/commission already model costs you didn't actually
    avoid; this is the mirror-image gap -- income you actually WOULD have
    received but this backtest never counts, a one-directional bias that
    always UNDERSTATES real returns for dividend-paying tickers, never
    overstates them.

    For growth/tech-heavy universes with low-or-no dividend yields (this
    project's own primary `watchlist.txt`), the effect is expected to be
    negligible. For higher-yield names (utilities/REITs/financials in
    `smallmid_watchlist.txt`, and especially the international ADRs in
    `adr_watchlist.txt` -- several yield 4-8%+), it may not be.

    `trades` should be resolved trades (status != "OPEN") with `ticker`,
    `entry_date`, `exit_date`, `buy_price`, `pnl_pct` -- every field every
    `settle_trade()`/`settle_trade_with_trailing()` trade dict already
    carries. `dividend_history` maps ticker -> a date-indexed pd.Series of
    per-share $ dividend amounts (see `yfinance.Ticker(ticker).dividends`,
    the same real data source `run_backtest.py`'s other fetch_* helpers
    already use elsewhere in this project) -- a ticker missing from this
    dict is treated as having no dividend history (0 missed, same
    "missing optional data never fabricates a worse OR better outcome"
    convention as every other optional field in this codebase). Each
    Series' own index is normalized to a tz-naive calendar DATE internally
    (yfinance's own `.dividends` index is tz-aware with a same-day
    09:30:00 exchange-local timestamp, which would otherwise crash a
    naive comparison against this project's own plain-date entry_date/
    exit_date trade fields) -- callers don't need to pre-clean this.

    For each trade, sums every dividend whose ex-date falls in
    `(entry_date, exit_date]` (the standard convention: you must have
    HELD the position as of the ex-date, i.e. bought before it, to
    receive that payment -- a same-day entry ON the ex-date itself
    already missed it) and expresses it as a % of `buy_price`, directly
    comparable to `pnl_pct`.

    PURE MEASUREMENT ONLY -- does not modify `pnl_pct` or any other trade
    field, and is never called from settle_trade()/simulate_*_signals()
    itself. Changing how real settled pnl_pct is computed is a much more
    invasive, precedent-setting change than an additive audit and is
    deliberately NOT done here without explicit user sign-off, same
    posture the strength-cap-calibration finding (item 140) took toward
    ma_crossover's own live config.

    Returns `{"n_trades": total trades checked, "n_affected": how many
    spanned at least one real ex-dividend date, "pct_affected": that as a
    %, "avg_missed_pct_all_trades": mean missed-dividend-% averaged across
    EVERY trade (0 for unaffected ones -- the honest expected-value drag),
    "avg_missed_pct_affected_only": mean missed-dividend-% averaged only
    across the trades actually affected (the size of the miss WHEN it
    happens), "total_missed_dollars_per_position_budget_dollar"}` --
    the last figure answers "for every $1 nominally invested across these
    trades, how many cents of real dividend income does this backtest
    never count," directly usable as a comparison against slippage/
    commission's own documented cost-sensitivity magnitude (item 130)."""
    dividend_history = dividend_history or {}
    resolved = [t for t in trades if t.get("status") != "OPEN" and "entry_date" in t and "exit_date" in t]
    n_trades = len(resolved)
    if n_trades == 0:
        return {
            "n_trades": 0, "n_affected": 0, "pct_affected": None,
            "avg_missed_pct_all_trades": None, "avg_missed_pct_affected_only": None,
            "total_missed_dollars_per_position_budget_dollar": None,
        }

    normalized_divs_cache: dict = {}

    def _normalized_divs(ticker):
        if ticker not in normalized_divs_cache:
            divs = dividend_history.get(ticker)
            if divs is not None and len(divs) > 0:
                idx = divs.index
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                divs = divs.set_axis(idx.normalize())
            normalized_divs_cache[ticker] = divs
        return normalized_divs_cache[ticker]

    missed_pcts = []
    for t in resolved:
        divs = _normalized_divs(t["ticker"])
        missed_dollars = 0.0
        if divs is not None and len(divs) > 0:
            entry_date = pd.Timestamp(t["entry_date"]).normalize()
            exit_date = pd.Timestamp(t["exit_date"]).normalize()
            in_window = divs[(divs.index > entry_date) & (divs.index <= exit_date)]
            missed_dollars = float(in_window.sum())
        buy_price = t.get("buy_price")
        missed_pcts.append(missed_dollars / buy_price * 100 if buy_price else 0.0)

    n_affected = sum(1 for m in missed_pcts if m > 0)
    return {
        "n_trades": n_trades,
        "n_affected": n_affected,
        "pct_affected": round(n_affected / n_trades * 100, 2),
        "avg_missed_pct_all_trades": round(sum(missed_pcts) / n_trades, 4),
        "avg_missed_pct_affected_only": (
            round(sum(missed_pcts) / n_affected, 4) if n_affected else 0.0
        ),
        "total_missed_dollars_per_position_budget_dollar": round(sum(missed_pcts) / n_trades / 100, 5),
    }


def compute_strategy_correlation(trades_by_strategy: dict) -> dict:
    """Are two strategies' real edges actually DIVERSIFIED, or just the
    same underlying signal wearing different clothes? (2026-09-13, per the
    user's "fully comprehensive of all variables" directive.) Item 142's
    `simulate_portfolio_constrained(group_key=...)` already answers "how
    much capital do these strategies fight over when run together" -- a
    genuinely different question from this one: even with UNLIMITED
    capital (no fighting at all), do these strategies make and lose money
    on the SAME days for the SAME reasons? A real capital-allocation
    decision (how much to size a NEW strategy relative to ones already
    running) depends on this, not on the capital-contention question
    alone -- two strategies that are both real, both individually
    validated, but 90%+ correlated add much less true diversification
    than their separate validation reports would suggest on their own.

    `trades_by_strategy` maps strategy label (e.g. "MA_Crossover", "Pairs"
    -- the same values already in every trade dict's own `signal` field,
    see `benchmark_multi_strategy_portfolio.py`) -> that strategy's own
    list of resolved trade dicts (needs `exit_date`/`pnl_pct` on each).

    Method: for each strategy, builds a DAILY realized-P&L series --
    every trade's `pnl_pct` is booked on its own `exit_date` (summed if
    more than one trade exits that ticker/strategy that day), matching
    this module's own "P&L realizes at EXIT, not entry" convention
    (see `simulate_portfolio_constrained()`'s identical choice). Days
    with no realized trade for a strategy get 0.0, NOT excluded --
    excluding them would only compare cherry-picked overlapping activity
    days, understating how uncorrelated two strategies with different
    trading FREQUENCIES genuinely are. Then computes the standard Pearson
    correlation coefficient between each pair of strategies' aligned
    daily series (aligned over the UNION of every date either strategy
    has any activity on).

    Also reports `ticker_day_overlap_pct` per pair -- what fraction of
    each strategy's own (ticker, entry_date) signal instances are shared
    with the other strategy's own instances -- a simpler, more literal
    "how often do these two actually pick the exact same moment" measure,
    distinct from (and can disagree with) the return-correlation number
    above (e.g. two strategies could rarely trade the identical
    ticker/day, yet still have correlated DAILY P&L because whatever
    moves one moves the whole market and hits every open position that
    day).

    Returns `{"pairs": {"StratA|StratB": {"correlation":, "n_days":,
    "ticker_day_overlap_pct":}}}` -- `correlation`/`n_days` are None for a
    pair where either strategy has fewer than 2 days of realized activity
    (not enough variance to compute a meaningful correlation)."""
    daily_pnl: dict = {}
    ticker_days: dict = {}
    for strategy, trades in trades_by_strategy.items():
        resolved = [t for t in trades if t.get("status") != "OPEN" and "exit_date" in t and "pnl_pct" in t]
        series: dict = defaultdict(float)
        for t in resolved:
            series[pd.Timestamp(t["exit_date"])] += t["pnl_pct"]
        daily_pnl[strategy] = series
        ticker_days[strategy] = {(t.get("ticker"), pd.Timestamp(t["entry_date"])) for t in resolved if "entry_date" in t}

    names = sorted(trades_by_strategy)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            all_dates = sorted(set(daily_pnl[a]) | set(daily_pnl[b]))
            key = f"{a}|{b}"
            if len(all_dates) < 2:
                pairs[key] = {"correlation": None, "n_days": len(all_dates), "ticker_day_overlap_pct": None}
                continue
            series_a = pd.Series([daily_pnl[a].get(d, 0.0) for d in all_dates])
            series_b = pd.Series([daily_pnl[b].get(d, 0.0) for d in all_dates])
            correlation = None
            if series_a.std() > 0 and series_b.std() > 0:
                correlation = round(float(series_a.corr(series_b)), 4)
            shared = ticker_days[a] & ticker_days[b]
            union = ticker_days[a] | ticker_days[b]
            overlap_pct = round(len(shared) / len(union) * 100, 2) if union else None
            pairs[key] = {"correlation": correlation, "n_days": len(all_dates), "ticker_day_overlap_pct": overlap_pct}

    return {"pairs": pairs}


def audit_same_ticker_overlap(trades: list[dict]) -> dict:
    """Real, previously entirely unexamined gap (2026-09-14, per the
    user's "fully comprehensive of all variables" directive, found while
    investigating whether backtest trade counts realistically model
    position-level risk): every `simulate_*_signals()` function in this
    module walks every ELIGIBLE day independently and opens a trade
    whenever its own signal condition fires -- with NO check for whether
    a PREVIOUS trade on the SAME ticker is still open. A real trader
    generally avoids doubling up exposure to the identical underlying
    stock; this backtest can silently count two (or more) overlapping
    positions in the same ticker as if they were two fully independent,
    unrelated bets -- a real risk-concentration dimension distinct from
    `simulate_portfolio_constrained()`'s own sector/portfolio caps, which
    have no notion of "same ticker" at all.

    Confirmed real and non-hypothetical via a direct data check (not just
    theorized): a 15-ticker, 5-year `ma_crossover` sample showed 3 of 94
    real trades (~3%) with a genuine same-ticker overlap -- a new signal
    fired and filled on a ticker before its PREVIOUS trade on that same
    ticker had reached its own `exit_date`.

    `trades` should be resolved trades (status != "OPEN") with `ticker`,
    `entry_date`, `exit_date` -- every field every `settle_trade()`/
    `settle_trade_with_trailing()` trade dict already carries. Groups by
    ticker, sorts by `entry_date`, and counts every CONSECUTIVE pair
    where the later trade's `entry_date` falls at or before the earlier
    trade's own `exit_date` (a closed-interval overlap check, same
    entry/exit-date concurrency convention `compute_max_drawdown()`/
    `simulate_portfolio_constrained()` already use, reused here instead
    of inventing a second one).

    PURE MEASUREMENT ONLY -- does not modify `trades`, does not change
    how any `simulate_*_signals()` function generates trades. See
    `simulate_portfolio_constrained()`'s own new `max_positions_per_ticker`
    parameter (2026-09-14) for the opt-in constraint this measurement
    motivated.

    Returns `{"n_trades":, "n_tickers":, "n_overlapping_pairs":,
    "pct_trades_in_an_overlap": (n trades touching at least one
    overlapping pair, as a % of n_trades), "overlap_examples": (up to 10
    `(ticker, prev_exit_date, next_entry_date)` tuples, for review)}`."""
    resolved = [t for t in trades if t.get("status") != "OPEN" and "entry_date" in t and "exit_date" in t and "ticker" in t]
    n_trades = len(resolved)
    if n_trades == 0:
        return {
            "n_trades": 0, "n_tickers": 0, "n_overlapping_pairs": 0,
            "pct_trades_in_an_overlap": None, "overlap_examples": [],
        }

    by_ticker: dict = defaultdict(list)
    for t in resolved:
        by_ticker[t["ticker"]].append(t)

    n_overlapping_pairs = 0
    trades_in_overlap = set()
    examples = []
    for ticker, ticker_trades in by_ticker.items():
        ordered = sorted(ticker_trades, key=lambda t: t["entry_date"])
        for i in range(1, len(ordered)):
            prev, cur = ordered[i - 1], ordered[i]
            if cur["entry_date"] <= prev["exit_date"]:
                n_overlapping_pairs += 1
                trades_in_overlap.add(id(prev))
                trades_in_overlap.add(id(cur))
                if len(examples) < 10:
                    examples.append((ticker, prev["exit_date"], cur["entry_date"]))

    return {
        "n_trades": n_trades,
        "n_tickers": len(by_ticker),
        "n_overlapping_pairs": n_overlapping_pairs,
        "pct_trades_in_an_overlap": round(len(trades_in_overlap) / n_trades * 100, 2),
        "overlap_examples": examples,
    }


def audit_capital_capacity(
    trades: list[dict],
    position_budget: float,
    capital_levels: list[float],
    max_sector_allocation_pct: float | None = None,
    max_total_deployed_pct: float | None = None,
    sector_lookup: dict[str, str] | None = None,
    plateau_threshold_pct: float = 1.0,
) -> dict:
    """A genuinely new axis (2026-09-14, twentieth iteration of the "fully
    comprehensive of all variables" loop): `simulate_portfolio_constrained()`
    (item 138) already answers "how many of this strategy's real signals
    could ONE real account afford to take" at a single, fixed
    `starting_capital` -- but every real-data run so far has picked that
    one capital figure somewhat arbitrarily (matching whatever the CLI
    default happens to be). This asks the question that actually matters
    for a capital-ALLOCATION decision: as `starting_capital` grows, does
    `pct_signals_taken` keep rising (money is still the binding
    constraint), or does it PLATEAU (the account has hit its structural
    ceiling -- `max_sector_allocation_pct`/`max_total_deployed_pct` are now
    the binding constraint, and no amount of additional capital would let
    this strategy, at this `position_budget`, take meaningfully more of
    its own signals)? Directly relevant to "when we proceed to
    constructing new strategies" (the user's own standing directive): it
    quantifies how much room is actually left in an existing strategy
    before adding more capital stops helping, which is exactly the
    question a real capital-allocation-across-strategies decision needs
    answered.

    Runs `simulate_portfolio_constrained()` once per dollar amount in
    `capital_levels` (e.g. `[3000, 6000, 15000, 30000, 60000]`), same
    `position_budget`/caps/`sector_lookup` held fixed across every level
    (only `starting_capital` varies -- isolates capital scale as the one
    changing variable, the whole point of a sensitivity scan). `trades`
    should already be resolved (status != "OPEN") the same as every other
    `simulate_portfolio_constrained()` caller provides. `capital_levels`
    is expected ascending; not re-sorted here since the plateau detection
    below is inherently order-dependent.

    A level's `dominant_skip_reason` is whichever of
    `n_skipped_insufficient_capital`/`n_skipped_sector_limit`/
    `n_skipped_portfolio_limit`/`n_skipped_ticker_limit` is largest at that
    level (None if nothing was skipped). `capacity_ceiling_capital` is the
    smallest `starting_capital` in the scan where `pct_signals_taken`
    gained less than `plateau_threshold_pct` percentage points over the
    PREVIOUS (smaller) level -- i.e. where money stopped being the thing
    holding the strategy back. None if `pct_signals_taken` was still
    climbing at the largest level scanned (the ceiling, if any, lies
    beyond what was tested here) or if there's only one level to compare.

    Returns `{"levels": [scan results, each a
    simulate_portfolio_constrained() dict plus "dominant_skip_reason"],
    "capacity_ceiling_capital":}`."""
    levels = []
    prev_pct_taken = None
    ceiling = None
    for starting_capital in capital_levels:
        result = simulate_portfolio_constrained(
            trades, starting_capital, position_budget,
            max_sector_allocation_pct, max_total_deployed_pct, sector_lookup,
        )
        skip_counts = {
            "insufficient_capital": result["n_skipped_insufficient_capital"],
            "sector_limit": result["n_skipped_sector_limit"],
            "portfolio_limit": result["n_skipped_portfolio_limit"],
            "ticker_limit": result["n_skipped_ticker_limit"],
        }
        dominant = max(skip_counts, key=skip_counts.get) if any(skip_counts.values()) else None
        result = dict(result, dominant_skip_reason=dominant)
        levels.append(result)

        pct_taken = result["pct_signals_taken"]
        if ceiling is None and prev_pct_taken is not None and pct_taken is not None:
            if pct_taken - prev_pct_taken < plateau_threshold_pct:
                ceiling = starting_capital
        prev_pct_taken = pct_taken

    return {
        "levels": levels,
        "capacity_ceiling_capital": ceiling,
    }


def flatten_out_sample_trades(fold_results: list) -> list[dict]:
    """All resolved (non-OPEN) out-of-sample trades across every fold,
    pooled WITHOUT weighting -- unlike summarize_weighted() (which needs
    each trade's originating fold to apply a recency weight), this is just
    a plain chronological trade list, which is what compute_max_drawdown()
    needs."""
    trades = []
    for fr in fold_results:
        trades.extend(t for t in fr.out_sample_trades if t["status"] != "OPEN")
    return trades
