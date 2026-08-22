"""Walk-Forward backtesting harness (Phase 4). Pure -- takes pre-fetched
historical OHLCV DataFrames in, returns simulated trades and fold-level
metrics out. Reuses compute_levels / add_trade_score / settle_trade /
is_market_uptrend exactly as the live dashboard and nightly settlement job
do, so a backtested TradingConfig is scored by identical rules to a live
signal -- that's the whole point of Phase 1 having pulled this math out of
the Streamlit app in the first place.

Walk-Forward Optimization, not a single static window: a backtest over one
fixed historical range and calling the winner "optimal" is a curve-fitting
trap -- it rewards parameters that memorized that period's noise. Instead,
generate_folds() produces a sequence of rolling in-sample/out-of-sample
windows; Optuna (Phase 5) will score a candidate config on the AGGREGATE
out-of-sample performance across all folds, never a single fold's in-sample
fit. See ARCHITECTURE_PLAN.md amendment #3 for the full rationale.

No look-ahead: on each simulated day `as_of`, only bars up to and including
`as_of` are used to decide whether a signal fires (mirrors the live app,
where the last row of fetched data IS "today"). Settling a trade started on
`as_of` is free to use bars strictly after `as_of` -- grading a past
decision against realized future prices is not look-ahead bias, it's just
how you find out whether the trade won.

Entry timing is realistic, not instantaneous: a signal computed from
`as_of`'s close can't be acted on until the NEXT session at the earliest,
so simulate_signals() no longer assumes entry on the same bar the signal
fired. _find_entry_fill() walks forward looking for the resting limit order
at Buy_Price to actually get touched (gap-aware, same rules as stop/target
fills), within config.max_entry_wait_days -- a signal that's never touched
produces no trade at all, which is the honest outcome for a limit order
nobody would leave open forever.

Catalyst awareness IS simulated, given an optional per-ticker
`earnings_dates` (see run_backtest.fetch_earnings_dates): yfinance's
get_earnings_dates(limit=40) returns roughly a decade of REPORTED earnings
dates, not just upcoming estimates. Only the calendar date is used, never
the EPS/Surprise columns -- once a company schedules a report, the date
itself is a fixed fact that doesn't get revised after the fact the way an
EPS estimate does, so looking up "the next earnings date after `as_of`"
from that static list introduces no look-ahead leakage even for a backtest
day years in the past. Without `earnings_dates` passed in, Catalyst_Warning
is always False (the old behavior) rather than raising -- callers that
don't need catalyst simulation aren't forced to fetch it.

Pooled trades are NOT independent samples, and summarize_trades()'s
mean/std treats them as if they were. A real, observed failure mode: 30+
correlated Technology signals firing the same day during one sector move
aren't 30 independent bets, they're one correlated event counted 30 times,
which understates the true variance of pooled pnl_pct and overstates
sharpe_like's apparent confidence. compute_cluster_weights() (given a
sector_lookup) gives each (entry_date, sector) cluster a combined weight of
1.0 split evenly across its members -- a conservative correction (assumes
perfect intra-cluster correlation, zero inter-cluster), not an exact one,
but a real one. See optimize.py's summarize_weighted() for where this
combines with the existing recency weighting.
"""

import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, TradingConfig
from .levels import (
    adx_trend_entry_levels_from_frame,
    breakout_levels_from_frame,
    breakout_retest_levels_from_frame,
    levels_from_rsi_frame,
    ma_crossover_levels_from_frame,
    market_uptrend_from_frame,
    momentum_burst_levels_from_frame,
    precompute_adx_trend_entry_frame,
    precompute_breakout_frame,
    precompute_breakout_retest_frame,
    precompute_ma_crossover_frame,
    precompute_momentum_burst_frame,
    precompute_pairs_frame,
    precompute_pullback_frame,
    precompute_rsi_frame,
    precompute_squeeze_breakout_frame,
    precompute_week52_frame,
    pairs_levels_from_frame,
    pullback_levels_from_frame,
    squeeze_breakout_levels_from_frame,
    week52_levels_from_frame,
)
from .scoring import add_trade_score
from .settlement import settle_trade, settle_trade_with_trailing

ENTRY_SIGNALS = ("Strong Buy", "Buy")
LOOKBACK_BUFFER_BARS = 60   # extra trailing bars beyond sma_trend_window, for indicator warmup safety


@dataclass(frozen=True)
class Fold:
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_sample_start: pd.Timestamp
    out_sample_end: pd.Timestamp


def generate_folds(
    start,
    end,
    in_sample_days: int = 182,
    out_sample_days: int = 30,
    step_days: int = 30,
) -> list[Fold]:
    """Rolling walk-forward folds over [start, end): each fold trains on
    `in_sample_days` and validates on the immediately following
    `out_sample_days` (no gap, no overlap between the two), then the whole
    window slides forward by `step_days` and repeats. Stops once a fold
    would extend past `end`."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    folds = []
    cursor = start
    while True:
        in_end = cursor + pd.Timedelta(days=in_sample_days)
        out_end = in_end + pd.Timedelta(days=out_sample_days)
        if out_end > end:
            break
        folds.append(Fold(cursor, in_end, in_end, out_end))
        cursor = cursor + pd.Timedelta(days=step_days)
    return folds


def _next_earnings_date(earnings_dates: pd.DatetimeIndex | None, as_of: pd.Timestamp):
    """Mirrors market_data.get_next_earnings_date's live semantics (the
    earliest date >= as_of), just reading from a static historical list
    instead of a fresh API call. `earnings_dates` must be tz-aware UTC (see
    run_backtest.fetch_earnings_dates) -- `as_of` is localized to UTC here
    to match, the same way compute_levels handles its own as_of internally."""
    if earnings_dates is None or len(earnings_dates) == 0:
        return None
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    future = earnings_dates[earnings_dates >= as_of]
    return future.min() if len(future) > 0 else None


def _find_entry_fill(buy_price: float, bars_after_signal: pd.DataFrame, max_wait_days: int):
    """A signal fires using data through `as_of`'s close -- the earliest a
    real limit order at `buy_price` could possibly fill is the NEXT
    session, never the same bar. Walk forward from the first bar strictly
    after the signal, same gap-aware logic already used for stops/targets:
    a gap-down open through the limit fills at that (better, lower) Open; an
    intraday touch (Low <= buy_price) fills at the limit exactly. Stops
    searching after `max_wait_days` -- a resting order nobody would leave
    open forever. Returns (fill_date, fill_price) or None if never filled
    within the window (not a real trade -- correctly excluded, not forced)."""
    for wait_days, (bar_date, bar) in enumerate(bars_after_signal.iterrows(), start=1):
        if wait_days > max_wait_days:
            break
        open_ = float(bar["Open"])
        low = float(bar["Low"])
        if open_ <= buy_price:
            return bar_date, open_
        if low <= buy_price:
            return bar_date, buy_price
    return None


def _find_breakout_fill(trigger_price: float, bars_after_signal: pd.DataFrame, max_wait_days: int):
    """Mirror of _find_entry_fill(), but for a resting STOP-BUY order (fills
    on an UPSIDE cross through `trigger_price`) instead of a resting LIMIT
    order (fills on a downside touch) -- the breakout counterpart. Gap-aware
    the same way stop-losses are in settlement.py: a gap-up open through the
    trigger fills at that (worse, higher) real Open; an intraday touch
    (High >= trigger_price) fills at exactly the trigger. Stops searching
    after `max_wait_days`. Returns (fill_date, fill_price) or None if the
    breakout was never confirmed within the window."""
    for wait_days, (bar_date, bar) in enumerate(bars_after_signal.iterrows(), start=1):
        if wait_days > max_wait_days:
            break
        open_ = float(bar["Open"])
        high = float(bar["High"])
        if open_ >= trigger_price:
            return bar_date, open_
        if high >= trigger_price:
            return bar_date, trigger_price
    return None


def _find_next_open_fill(bars_after_signal: pd.DataFrame):
    """Buy at the very next session's Open, unconditionally -- no waiting
    for a specific price to be touched/crossed (contrast _find_entry_fill()/
    _find_breakout_fill(), both of which wait, possibly days, for a
    specific level). Models a market-buy the morning after a same-day
    confirmation (e.g. momentum_burst) -- the premise there is chasing a
    move already in progress, not waiting for a pullback that may never
    come; a resting limit order at yesterday's Close can systematically
    miss the strongest continuation cases (see swingtrade/config.py's
    momentum_burst_entry_fill). Returns (fill_date, fill_price) or None if
    there's no bar after the signal at all (end of available data)."""
    if bars_after_signal.empty:
        return None
    return bars_after_signal.index[0], float(bars_after_signal.iloc[0]["Open"])


def _settle(
    buy_price: float, stop_loss: float, sell_price: float, atr: float,
    bars_since_entry: pd.DataFrame, config: TradingConfig,
) -> dict:
    """Shared dispatch used by the 3 currently-active strategies' simulate
    functions (breakout, squeeze_breakout, ma_crossover -- both real and
    random-entry counterparts) -- routes to settle_trade_with_trailing()
    when config.trailing_stop_enabled, else the original settle_trade(),
    so each call site doesn't need to repeat this branch inline. The 4
    dormant strategies (pullback/breakout_retest/week52_high/momentum_burst/
    adx_trend_entry) still call settle_trade() directly, unaffected by
    trailing_stop_enabled -- see swingtrade/config.py's own field comment
    for why this is scoped to the active 3 only."""
    if config.trailing_stop_enabled:
        return settle_trade_with_trailing(
            buy_price=buy_price, stop_loss=stop_loss, sell_price=sell_price, atr=atr,
            bars_since_entry=bars_since_entry, config=config,
        )
    return settle_trade(
        buy_price=buy_price, stop_loss=stop_loss, sell_price=sell_price,
        bars_since_entry=bars_since_entry, config=config,
    )


def simulate_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Walk every trading day in [window_start, window_end) for one ticker,
    simulating a trade wherever swingtrade would fire a Strong Buy/Buy
    signal using only data available as of that day. A signal can only be
    known AFTER the close it's computed from, so entry is NOT assumed on
    that same bar -- _find_entry_fill() walks forward from the next session
    looking for the resting limit order at Buy_Price to actually get
    touched (gap-aware, same logic as stop/target fills), up to
    config.max_entry_wait_days out. A signal that's never touched within
    that window produces no trade at all -- correctly excluded, not forced.
    Each trade dict carries both `signal_date` (when the signal fired) and
    `entry_date` (when it actually filled, used for cluster-weighting --
    see compute_cluster_weights); `buy_price` is the real fill price
    (better than the signal's theoretical level on a gap), with the
    original `signal_buy_price` kept for reference. Each filled trade is
    then resolved with settle_trade() against the ticker's actual
    subsequent price history (which may extend past window_end).

    `ohlcv` and `market_ohlcv` must each contain enough leading history
    before window_start to cover config.sma_trend_window -- see
    LOOKBACK_BUFFER_BARS and run_backtest.py's fetch buffering.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False.

    `sector` (optional, see watchlist.read_ticker_sectors) is tagged onto
    every trade so compute_cluster_weights() can identify same-day,
    same-sector clusters -- unknown/untagged tickers default to "Unknown"
    and are never clustered with each other (no basis to assume they're
    correlated).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    # Precomputed ONCE per ticker/fold instead of being recomputed from
    # scratch at every as_of -- see levels.precompute_rsi_frame's docstring
    # for why this is an exact equivalence, not an approximation.
    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_rsi_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue  # insufficient market history yet
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = levels_from_rsi_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue  # insufficient history / macro downtrend / illiquid that day

        scored = add_trade_score(pd.DataFrame([levels]), config).iloc[0]
        if scored["Signal"] not in ENTRY_SIGNALS:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(scored["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue  # resting order never touched within the window -- not a real trade
        entry_date, entry_price = fill

        # Stop/target are risk-management levels relative to the ACTUAL entry
        # price, not the original signal's theoretical Buy_Price -- matters
        # when a gap-down fill gets a better (lower) entry than the signal
        # anticipated. Leaving them anchored to the stale higher Buy_Price
        # would put the stop ABOVE the real entry price on a gap fill,
        # causing a near-instant fake stop-out on what was actually a good
        # (better-than-limit) fill. Re-anchoring to entry_price keeps the
        # stop below and the target above wherever you actually got in --
        # identical to the original scored levels in the non-gap case, since
        # entry_price == signal Buy_Price there.
        atr = float(scored["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": scored["Signal"],
            "trade_score": float(scored["Trade_Score"]),
            "rsi": float(scored["RSI"]),
            "atr": atr,
            "rrr": float(scored["RRR"]),
            "buy_price": entry_price,
            "signal_buy_price": float(scored["Buy_Price"]),
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(scored["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Diagnostic benchmark, NOT a production strategy -- see
    benchmark_random_entry.py. Identical universe gates (macro uptrend,
    liquidity/history via compute_levels) and identical entry-fill/stop/
    target mechanics as simulate_signals(), but the TRIGGER day is chosen
    uniformly at random from every day that passed the gates, instead of
    requiring an RSI-oversold Strong Buy/Buy signal. `n_trades` should be
    the real strategy's signal count for this ticker/window (see
    simulate_signals) so trade volume is matched -- isolates whether RSI
    timing itself adds value over picking days at random with the exact
    same underlying entry/exit structure and macro/liquidity filtering.
    If RSI-timed entries don't meaningfully beat this, the RSI signal
    carries little real predictive information."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_rsi_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the same gates simulate_signals uses
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = levels_from_rsi_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, float(levels["Buy_Price"]), float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_breakout_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
) -> list[dict]:
    """Trend-following counterpart to simulate_signals() -- buys strength (a
    new config.breakout_lookback_days-day closing high in a confirmed
    uptrend) instead of RSI-oversold weakness. Same no-look-ahead discipline
    (only data through `as_of` decides whether a signal fires), same
    entry-timing realism (a signal known at `as_of`'s close can't be acted
    on before the next session -- _find_breakout_fill() walks forward
    looking for the resting stop-buy order to actually get touched, gap-aware,
    within config.max_entry_wait_days), and the exact same ATR-multiple
    stop/target sizing as simulate_signals() -- deliberately kept identical
    so this is a clean test of whether breakout TIMING adds value, without
    also changing the exit model at the same time.

    A breakout already at/above config.breakout_rsi_overbought_threshold is
    skipped -- an over-extended/exhausted move is more likely to fail or
    reverse than one fresh off a quiet consolidation. Default threshold
    (100.0) is a practical no-op, since RSI essentially never reaches it;
    this filter is a documented, unvalidated hypothesis until tested (see
    improvements.txt) -- kept as a plain skip, not baked into the trigger
    definition itself, so it stays a searchable/toggleable parameter.

    A breakout whose Relative_Strength (ticker return minus market return
    over the same breakout_lookback_days window -- see
    compute_relative_strength) is below config.breakout_relative_strength_min
    is also skipped -- a stock breaking out only because the whole market is
    ripping isn't the same as one genuinely beating the market. Default
    (-100.0) is a practical no-op.

    A breakout whose Volume_Ratio (today's Volume over the PRIOR
    volume_lookback_days average -- see compute_breakout_levels) is below
    config.breakout_volume_ratio_min is also skipped -- a genuine breakout
    on high volume vs. a low-volume drift above an old high are different
    events. Default (0.0) is a practical no-op.

    A breakout whose ADX (trend strength, independent of direction -- see
    compute_breakout_levels) is below config.breakout_adx_min is also
    skipped -- a breakout during a weak/choppy trend and one during a
    genuinely strong trend look identical to every other filter here, but
    aren't the same event. Default (0.0) is a practical no-op.

    A breakout whose OBV_Zscore (On-Balance Volume z-scored against its
    own recent baseline -- see compute_breakout_levels) is below
    config.breakout_obv_zscore_min is also skipped -- sustained buying
    pressure building up over weeks is a deeper signal than a single day's
    volume spike. Default (-100.0) is a practical no-op.

    A breakout whose Squeeze_Zscore (prior-day volatility z-scored against
    its own recent baseline -- see compute_breakout_levels) is above
    config.breakout_squeeze_zscore_max is also skipped -- a breakout
    emerging from a period of volatility contraction ("coiled spring") is
    a different, often more reliable event than one that isn't. Default
    (100.0) is a practical no-op.

    `ohlcv`/`market_ohlcv` need the same leading-history buffer as
    simulate_signals() -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_frame(ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Breakout_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.breakout_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.breakout_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.breakout_volume_ratio_min:
            continue
        adx = levels.get("ADX")
        if adx is not None and adx < config.breakout_adx_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.breakout_obv_zscore_min:
            continue
        squeeze_zscore = levels.get("Squeeze_Zscore")
        if squeeze_zscore is not None and squeeze_zscore > config.breakout_squeeze_zscore_max:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.breakout_sector_relative_strength_min:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_breakout_fill(levels["Trigger_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Trigger_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_breakout_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_breakout_signals() -- same idea as
    simulate_random_entries(), but using the breakout strategy's own gates
    (macro uptrend, liquidity via compute_breakout_levels) and stop-buy fill
    mechanics (_find_breakout_fill), so it isolates whether breakout-day
    TIMING adds value over a random day using the identical trigger-price
    formula (that day's own N-day high) and entry/exit structure. `n_trades`
    should be simulate_breakout_signals()'s real signal count for this
    ticker/window, so trade volume is matched."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    # No market_df here -- matches the original's deliberate omission (the
    # random-entry benchmark never applies the relative-strength/volume/
    # overbought filters, only the shared macro-uptrend/liquidity gates).
    frame = precompute_breakout_frame(ohlcv, config)

    candidates = []  # (as_of, trigger_price, atr) for every day that passed the gates, breakout or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Trigger_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, trigger_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_breakout_fill(trigger_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": trigger_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_pullback_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Pullback-in-uptrend counterpart to simulate_signals()/simulate_breakout_signals()
    -- buys a shallow dip toward a rising config.pullback_ma_window-day SMA
    within a confirmed macro uptrend, instead of RSI-oversold weakness
    (simulate_signals) or a fresh N-day closing high (simulate_breakout_signals).
    Same no-look-ahead discipline, same entry-timing realism -- but unlike
    breakout's resting STOP-BUY (_find_breakout_fill), Pullback_Signal's
    Buy_Price is a resting LIMIT order at the pullback MA level, filled via
    the same _find_entry_fill() mechanics as simulate_signals() (a downside
    touch, not an upside cross). Same ATR-multiple stop/target sizing as
    both other strategies, deliberately kept identical so this is a clean
    test of whether pullback TIMING adds value, without also changing the
    exit model.

    `market_ohlcv` is accepted for signature consistency with the other two
    simulate_*() functions (uniform calling convention for run_backtest()),
    but unused here -- v1 of this strategy has no Relative_Strength concept
    (see compute_pullback_levels's docstring on why this is a deliberately
    lean first version).

    `ohlcv` needs the same leading-history buffer as the other two
    strategies -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pullback_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = pullback_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Pullback_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        # Same re-anchor-to-actual-fill-price reasoning as simulate_signals()
        # -- a gap-down fill gets a better (lower) entry than the signal
        # anticipated, so stop/target are relative to entry_price, not the
        # stale signal Buy_Price.
        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Pullback",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_pullback_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_pullback_signals() -- same idea as
    simulate_random_entries()/simulate_random_breakout_entries(), using the
    pullback strategy's own gates (macro uptrend, liquidity, rising MA via
    compute_pullback_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether pullback-day TIMING adds
    value over a random day using the identical Buy_Price formula (that
    day's own pullback MA level) and entry/exit structure. `n_trades`
    should be simulate_pullback_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pullback_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, pullback or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = pullback_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Pullback",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_breakout_retest_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Breakout-retest counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals() -- buys a pullback BACK TO a recent genuine
    breakout's own trigger level, instead of RSI-oversold weakness, a fresh
    N-day high THE SAME DAY, or proximity to a rising MA regardless of any
    prior breakout. Built after both simulate_signals() and
    simulate_pullback_signals() lost to matched-count random-entry timing
    on held-out tickers (see benchmark_random_entry.py), keeping the one
    ingredient that DID show a real edge (simulate_breakout_signals'
    trigger definition) while relaxing its "must fire today" restriction.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at the original breakout's level, filled via
    the same _find_entry_fill() mechanics as simulate_signals()/
    simulate_pullback_signals() (a downside touch), NOT breakout's resting
    STOP-BUY (_find_breakout_fill(), an upside cross). Same ATR-multiple
    stop/target sizing as every other strategy, deliberately kept identical
    so this is a clean test of whether retest TIMING adds value.

    `market_ohlcv` is accepted for signature consistency with the other
    three simulate_*() functions but unused here, same as
    simulate_pullback_signals() -- v1 has no Relative_Strength concept.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_retest_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_retest_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Retest_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Breakout_Retest",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_breakout_retest_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_breakout_retest_signals() -- same
    idea as the other three simulate_random_*_entries() functions, using
    this strategy's own gates (macro uptrend, liquidity, a recent breakout
    to retest via compute_breakout_retest_levels) and limit-order fill
    mechanics (_find_entry_fill), so it isolates whether retest-day TIMING
    adds value over a random day using the identical Buy_Price formula
    (that day's own eligible retest level) and entry/exit structure.
    `n_trades` should be simulate_breakout_retest_signals()'s real signal
    count for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_retest_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, retest-eligible or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_retest_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Breakout_Retest",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_week52_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """52-week-high-momentum counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals() -- buys
    when price is within config.week52_nearness_pct of its own trailing
    config.week52_lookback_days high, in a confirmed macro uptrend. Unlike
    every prior strategy, this is a continuous STATE rather than a
    discrete event, so it can fire on many consecutive days while a stock
    consolidates near its highs.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at essentially today's own Close (already
    near strength, not waiting for a specific dip level), filled via the
    same _find_entry_fill() mechanics as pullback/breakout_retest. Same
    ATR-multiple stop/target sizing as every other strategy.

    `market_ohlcv` is accepted for signature consistency with the other
    four simulate_*() functions but unused here -- no Relative_Strength
    concept in this v1, same as pullback/breakout_retest.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS (note: week52_lookback_days defaults to
    252, noticeably longer than the other strategies' windows, so callers
    fetching history for this strategy need a correspondingly larger
    buffer -- see run_backtest.py's fetch buffering).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_week52_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = week52_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Week52_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Week52_High",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_week52_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_week52_signals() -- same idea as
    the other four simulate_random_*_entries() functions, using this
    strategy's own gates (macro uptrend, liquidity via
    compute_week52_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether being-near-a-52-week-high
    TIMING adds value over a random day using the identical Buy_Price
    formula (that day's own Close) and entry/exit structure. `n_trades`
    should be simulate_week52_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_week52_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, near-high or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = week52_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Week52_High",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_momentum_burst_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Momentum-burst counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()
    -- buys a single day's strong price gain CONFIRMED by unusually high
    volume, in a confirmed macro uptrend. Built to fire MORE OFTEN than any
    prior strategy: unlike breakout (a discrete "new high TODAY" event) or
    breakout_retest/week52_high (both anchored to a specific high level),
    this doesn't require a fresh high at all -- just a forceful, volume-
    backed move, which can happen on many more days than a genuine N-day
    high.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at essentially today's own Close (same
    convention as week52_high: already happening, not waiting for a
    specific dip level), filled via the same _find_entry_fill() mechanics
    as pullback/breakout_retest/week52_high, deliberately NOT a bespoke
    "buy at the next Open" mechanic -- keeping the identical fill mechanic
    across every "Buy_Price = today's Close" strategy keeps this an
    apples-to-apples comparison with week52_high specifically, and matches
    this codebase's one established precedent for this exact situation.
    Same ATR-multiple stop/target sizing as every other strategy.

    `market_ohlcv` is accepted for signature consistency with the other
    five simulate_*() functions but unused here -- no Relative_Strength
    concept in this v1, same as pullback/breakout_retest/week52_high.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_burst_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = momentum_burst_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Momentum_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.momentum_burst_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Momentum_Burst",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_momentum_burst_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_momentum_burst_signals() -- same
    idea as the other five simulate_random_*_entries() functions, using
    this strategy's own gates (macro uptrend, liquidity via
    compute_momentum_burst_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether momentum-burst TIMING adds
    value over a random day using the identical Buy_Price formula (that
    day's own Close) and entry/exit structure. `n_trades` should be
    simulate_momentum_burst_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_burst_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, burst or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = momentum_burst_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.momentum_burst_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Momentum_Burst",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_squeeze_breakout_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
) -> list[dict]:
    """Squeeze-breakout counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()/
    simulate_momentum_burst_signals() -- buys a real directional expansion
    following a recent volatility contraction (squeeze), in a confirmed
    macro uptrend. Deliberately does NOT require a fresh high over any
    window (unlike breakout/breakout_retest/week52_high) and does NOT
    require volume confirmation (unlike momentum_burst) -- a materially
    different, and typically more frequent, trigger.

    Same no-look-ahead discipline, same entry-timing realism.
    config.squeeze_breakout_entry_fill selects the fill model, same
    toggle momentum_burst_entry_fill added: "limit" (default) is a
    resting LIMIT order at today's own Close, filled via
    _find_entry_fill() (same convention week52_high/momentum_burst use);
    "next_open" buys the very next session's Open unconditionally via
    _find_next_open_fill() -- see improvements.txt items 36-37 for why
    this choice can materially move backtest conclusions for a same-day
    "chase an expansion" signal, which is why it's built in here from the
    start rather than retrofitted later. Same ATR-multiple stop/target
    sizing as every other strategy.

    `market_ohlcv` enables the optional Phase 2 Relative_Strength filter
    (see swingtrade/config.py's squeeze_breakout_relative_strength_min) --
    same role as simulate_breakout_signals()'s/simulate_adx_trend_entry_signals()'s
    own market_df threading. Phase 2 also gates on RSI-overbought/
    Volume_Ratio/ADX/OBV_Zscore, all disabled by default (see
    add_squeeze_breakout_trade_score()'s docstring for the full list) --
    kept in sync with that function so live and backtested definitions
    can't silently disagree, same discipline breakout's/adx_trend_entry's
    own filters use.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False --
    same role as simulate_signals()'s own `earnings_dates` param. Previously
    this strategy's backtested Catalyst_Warning was ALWAYS False regardless
    of real earnings proximity (a genuine, previously-undiscovered gap
    between what the live dashboard shows and what was actually backtested
    -- see improvements.txt), since nothing threaded earnings data through
    here before this parameter was added.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_squeeze_breakout_frame(ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = squeeze_breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Squeeze_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.squeeze_breakout_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.squeeze_breakout_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.squeeze_breakout_volume_ratio_min:
            continue
        adx = levels.get("ADX")
        if adx is not None and adx < config.squeeze_breakout_adx_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.squeeze_breakout_obv_zscore_min:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.squeeze_breakout_sector_relative_strength_min:
            continue
        if config.squeeze_breakout_earnings_gate and levels["Catalyst_Warning"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.squeeze_breakout_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Squeeze_Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_squeeze_breakout_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_squeeze_breakout_signals() --
    same idea as the other six simulate_random_*_entries() functions,
    using this strategy's own gates (macro uptrend, liquidity via
    compute_squeeze_breakout_levels) and the SAME
    config.squeeze_breakout_entry_fill-selected fill mechanic, so it
    isolates whether squeeze-breakout TIMING adds value over a random day
    using the identical Buy_Price formula (that day's own Close) and
    entry/exit structure. `n_trades` should be
    simulate_squeeze_breakout_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py.

    `earnings_dates` (optional, see simulate_squeeze_breakout_signals())
    only makes each trade dict's own `catalyst_warning` field honest --
    the candidate pool itself stays UNFILTERED by earnings proximity
    (deliberately, same as every other optional filter's precedent: this
    random baseline answers "does TIMING add value," not "does this
    optional filter help," so it must not apply any of the strategy's own
    optional gates, earnings included)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_squeeze_breakout_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, squeeze or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = squeeze_breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.squeeze_breakout_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Squeeze_Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_pairs_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    peer_prices: pd.DataFrame | None = None,
) -> list[dict]:
    """Mean-reversion PAIRS counterpart to simulate_signals()/simulate_breakout_signals()/
    .../simulate_squeeze_breakout_signals() -- buys a ticker when it has
    diverged unusually far BELOW its most-correlated same-sector peer over
    a recent window (Pair_Spread_Zscore <= config.pairs_zscore_entry_max),
    in a confirmed macro uptrend. See swingtrade/config.py's pairs_* fields
    and swingtrade/levels.precompute_pairs_frame() for the full mechanism.
    LONG-ONLY (no short leg -- this codebase has no short-position support
    anywhere); this is the laggard-convergence side only.

    Same no-look-ahead discipline, same entry-timing realism.
    config.pairs_entry_fill selects the fill model (same "limit" vs.
    "next_open" toggle every other strategy has). Same ATR-multiple
    stop/target sizing.

    `peer_prices` should be a wide DataFrame of Close prices for every
    OTHER ticker in this ticker's own sector (caller's job to pre-filter by
    sector, this ticker's own column excluded) -- see
    precompute_pairs_frame()'s own docstring. Without it, Pair_Signal is
    always False (no partner data, degrades to "never fires," same
    convention as every other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pairs_frame(ohlcv, peer_prices, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Pair_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pairs_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Pairs",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_pairs_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_pairs_signals() -- same idea as
    the other simulate_random_*_entries() functions, using this strategy's
    own gates (macro uptrend, liquidity via pairs_levels_from_frame) and
    the SAME config.pairs_entry_fill-selected fill mechanic, so it isolates
    whether pairs-divergence TIMING adds value over a random day using the
    identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_pairs_signals()'s real signal
    count for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see
    benchmark_random_entry.py.

    Deliberately does NOT take `peer_prices` -- called with peer_prices=None,
    so Pair_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless (same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows -- see
    simulate_random_squeeze_breakout_entries()'s identical reasoning)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pairs_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, pair signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pairs_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Pairs",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_adx_trend_entry_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """ADX-trend-entry counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()/
    simulate_momentum_burst_signals()/simulate_squeeze_breakout_signals() --
    buys while ADX signals a genuinely trending state AND price is above a
    short-term MA for direction, in a confirmed macro uptrend. A
    continuous STATE (like week52_high/squeeze_breakout), so it can fire
    on many consecutive days while a trend persists.

    Same no-look-ahead discipline, same entry-timing realism.
    config.adx_trend_entry_entry_fill selects the fill model, same toggle
    momentum_burst_entry_fill/squeeze_breakout_entry_fill added: "limit"
    (default) is a resting LIMIT order at today's own Close via
    _find_entry_fill(); "next_open" buys the very next session's Open
    unconditionally via _find_next_open_fill() -- built in from the start
    here too, per the lesson from momentum_burst/squeeze_breakout
    (improvements.txt items 36-38). Same ATR-multiple stop/target sizing
    as every other strategy.

    `market_ohlcv` enables the optional Phase 2 Relative_Strength filter
    (see swingtrade/config.py's adx_trend_entry_relative_strength_min) --
    same role as simulate_breakout_signals()'s own market_df threading.
    Phase 2 also gates on RSI-overbought/Volume_Ratio/OBV_Zscore/
    Squeeze_Zscore, all disabled by default (see
    add_adx_trend_entry_trade_score()'s docstring for the full list) --
    kept in sync with that function so live and backtested definitions
    can't silently disagree, same discipline breakout's own six filters use.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_adx_trend_entry_frame(ohlcv, config, market_df=market_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = adx_trend_entry_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["ADX_Trend_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.adx_trend_entry_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.adx_trend_entry_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.adx_trend_entry_volume_ratio_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.adx_trend_entry_obv_zscore_min:
            continue
        squeeze_zscore = levels.get("Squeeze_Zscore")
        if squeeze_zscore is not None and squeeze_zscore > config.adx_trend_entry_squeeze_zscore_max:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.adx_trend_entry_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "ADX_Trend_Entry",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_adx_trend_entry_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_adx_trend_entry_signals() --
    same idea as the other seven simulate_random_*_entries() functions,
    using this strategy's own gates (macro uptrend, liquidity via
    compute_adx_trend_entry_levels) and the SAME
    config.adx_trend_entry_entry_fill-selected fill mechanic, so it
    isolates whether ADX-trend-entry TIMING adds value over a random day
    using the identical Buy_Price formula (that day's own Close) and
    entry/exit structure. `n_trades` should be
    simulate_adx_trend_entry_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_adx_trend_entry_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, trending or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = adx_trend_entry_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.adx_trend_entry_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_ADX_Trend_Entry",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_ma_crossover_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
) -> list[dict]:
    """Moving-average-crossover counterpart to every other simulate_*_signals()
    this session -- buys the day a short-term SMA crosses ABOVE a
    long-term SMA, in a confirmed macro uptrend. A genuinely different
    mechanical trigger from breakout (price-level high), squeeze_breakout
    (volatility-regime shift), or adx_trend_entry (raw trend-strength
    threshold) -- see swingtrade/config.py's own comment on this strategy
    for the full motivation (every momentum/strength-family signal tried
    this session lost to random-entry timing once properly checked).

    Same no-look-ahead discipline, same entry-timing realism.
    config.ma_crossover_entry_fill selects the fill model, same toggle
    every same-day-trigger strategy gets from the start: "limit" (default)
    is a resting LIMIT order at today's own Close via _find_entry_fill();
    "next_open" buys the very next session's Open unconditionally via
    _find_next_open_fill(). Same ATR-multiple stop/target sizing as every
    other strategy.

    Deliberately lean v1, no optional sharpening filters -- this is a NEW
    signal with no prior history to inherit filters from, and this
    project's own established discipline (see improvements.txt) is to
    validate the base trigger via the random-entry benchmark BEFORE
    adding any filters, not after.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False --
    same role as simulate_signals()'s own `earnings_dates` param. Previously
    this strategy's backtested Catalyst_Warning was ALWAYS False regardless
    of real earnings proximity (see improvements.txt for the gap this
    closes, shared with simulate_squeeze_breakout_signals()).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    # market_df/sector_df threaded through here (previously omitted --
    # ma_crossover had no Relative_Strength/sector data at all) purely to
    # enable the new, backtest/Optuna-only Sector_Relative_Strength filter
    # below; nothing else in this strategy reads Relative_Strength, so this
    # is additive and changes no existing gating behavior.
    frame = precompute_ma_crossover_frame(ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = ma_crossover_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["MA_Crossover_Signal"]:
            continue
        if config.ma_crossover_earnings_gate and levels["Catalyst_Warning"]:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.ma_crossover_sector_relative_strength_min:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.ma_crossover_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "MA_Crossover",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_ma_crossover_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_ma_crossover_signals() -- same
    idea as every other simulate_random_*_entries() this session, using
    this strategy's own gates (macro uptrend, liquidity via
    compute_ma_crossover_levels) and the SAME
    config.ma_crossover_entry_fill-selected fill mechanic, so it isolates
    whether MA-crossover TIMING adds value over a random day using the
    identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_ma_crossover_signals()'s real
    signal count for this ticker/window, so trade volume is matched.
    THIS IS THE CRITICAL VALIDATION GATE for this strategy -- see
    benchmark_random_entry.py. Deliberately checked BEFORE any filter or
    tuning work, not after.

    `earnings_dates` (optional, see simulate_squeeze_breakout_signals())
    only makes each trade dict's own `catalyst_warning` field honest --
    the candidate pool itself stays UNFILTERED by earnings proximity,
    same reasoning as simulate_random_squeeze_breakout_entries()."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_ma_crossover_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, crossover or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = ma_crossover_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.ma_crossover_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_MA_Crossover",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def run_backtest(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    sector_lookup: dict[str, str] | None = None,
    strategy: str = "rsi",
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Simulate signals for every ticker in ticker_data over
    [window_start, window_end), settling each against its own subsequent
    history. Returns the combined trade list. `earnings_data` (optional,
    ticker -> tz-aware UTC DatetimeIndex, see run_backtest.fetch_earnings_dates)
    enables honest Catalyst_Warning simulation for "rsi", "squeeze_breakout",
    and "ma_crossover" -- without it, every trade for those three has
    Catalyst_Warning=False. `sector_lookup`
    (optional, ticker -> sector, see watchlist.read_ticker_sectors) tags
    each trade for compute_cluster_weights(). `strategy` selects which
    signal to simulate -- "rsi" (simulate_signals, mean-reversion,
    default), "breakout" (simulate_breakout_signals, trend-following;
    earnings_data is ignored, it has no catalyst concept), "pullback"
    (simulate_pullback_signals, trend-following pullback entry; earnings_data
    also ignored, same reasoning), "breakout_retest"
    (simulate_breakout_retest_signals, pullback to a recent breakout's
    level; earnings_data also ignored, same reasoning), "week52_high"
    (simulate_week52_signals, near a trailing 52-week high; earnings_data
    also ignored, same reasoning), "momentum_burst"
    (simulate_momentum_burst_signals, single-day price+volume burst;
    earnings_data also ignored, same reasoning), "squeeze_breakout"
    (simulate_squeeze_breakout_signals, volatility contraction followed
    by expansion -- earnings_data now honored, see improvements.txt for
    the gap this closes), "adx_trend_entry" (simulate_adx_trend_entry_signals,
    ADX-confirmed trending state; earnings_data still ignored), or
    "ma_crossover" (simulate_ma_crossover_signals, short/long SMA crossover
    -- earnings_data now honored, same as squeeze_breakout).

    `sector_data` (optional, sector NAME -> that sector's own OHLCV, e.g. a
    real SPDR ETF like XLK for Technology -- see watchlist.SECTOR_ETF) backs
    the backtest/Optuna-only Sector_Relative_Strength filter (improvements.txt
    items 68/70) for "breakout"/"squeeze_breakout"/"ma_crossover" only --
    resolved per ticker via `sector_lookup`. `None` (default) means no
    sector data at all -- Sector_Relative_Strength then reads None/NaN and
    every *_sector_relative_strength_min gate never excludes on its own,
    same convention as every other optional filter.

    `pair_price_panels` (optional, sector NAME -> wide Close-price
    DataFrame of every ticker in that sector, same shape/resolution
    convention as `sector_data` above) backs the "pairs" strategy's
    partner-selection/spread-zscore mechanism -- see
    swingtrade/levels.precompute_pairs_frame(). Resolved per ticker into
    `peer_prices` by dropping that ticker's own column from its sector's
    panel. `None` (default) means no peer data at all -- Pair_Signal then
    always reads False, same "missing optional data never fabricates a
    signal" convention as sector_data."""
    earnings_data = earnings_data or {}
    sector_lookup = sector_lookup or {}
    sector_data = sector_data or {}
    pair_price_panels = pair_price_panels or {}
    all_trades = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        sector_ohlcv = sector_data.get(sector)
        pair_panel = pair_price_panels.get(sector)
        peer_prices = (
            pair_panel.drop(columns=[ticker]) if pair_panel is not None and ticker in pair_panel.columns else None
        )
        if strategy == "rsi":
            trades = simulate_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "breakout":
            trades = simulate_breakout_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                sector=sector, sector_ohlcv=sector_ohlcv,
            )
        elif strategy == "pullback":
            trades = simulate_pullback_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "breakout_retest":
            trades = simulate_breakout_retest_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "week52_high":
            trades = simulate_week52_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "momentum_burst":
            trades = simulate_momentum_burst_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "squeeze_breakout":
            trades = simulate_squeeze_breakout_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, sector_ohlcv=sector_ohlcv,
            )
        elif strategy == "adx_trend_entry":
            trades = simulate_adx_trend_entry_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "ma_crossover":
            trades = simulate_ma_crossover_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, sector_ohlcv=sector_ohlcv,
            )
        elif strategy == "pairs":
            trades = simulate_pairs_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, peer_prices=peer_prices,
            )
        else:
            raise ValueError(
                f"unknown strategy: {strategy!r} "
                "(expected 'rsi', 'breakout', 'pullback', 'breakout_retest', 'week52_high', "
                "'momentum_burst', 'squeeze_breakout', 'adx_trend_entry', 'ma_crossover', or 'pairs')"
            )
        all_trades.extend(trades)
    return all_trades


def _annualize_sharpe(resolved: list[dict], sharpe_like: float | None) -> tuple[float | None, float | None]:
    """Shared by summarize_trades()/summarize_trades_weighted() -- sharpe_like
    is a RAW per-trade ratio (mean/std of pnl_pct across trades), never
    annualized, which reads as a tiny number (0.02-0.15 typical here) next
    to the "Sharpe > 1.5" figures usually quoted, since those are almost
    always ANNUALIZED (scaled by sqrt(periods per year)). This applies the
    same standard trade-frequency convention: trades_per_year from the
    pooled trades' own entry_date span, annualized_sharpe_like =
    sharpe_like * sqrt(trades_per_year). Same explicit caveat as
    compute_max_drawdown(): an approximation for reading the number on a
    familiar scale, not a literal compounding portfolio simulation (no
    concurrent-position capital allocation modeled) -- additive alongside
    sharpe_like, not a replacement for it.

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


def compute_max_drawdown(trades: list[dict]) -> float | None:
    """Max peak-to-trough decline (%) of a SEQUENTIAL equity curve built by
    compounding resolved trades in chronological (entry_date) order --
    optimize.py's default single-objective search only ever optimizes
    sharpe_like (mean/stdev), which is blind to tail risk: a config can
    "win" by juicing consistency while quietly tolerating a nastier
    worst-case drawdown. This gives Optuna's optional multi-objective mode
    (see optimize.py --multi-objective) a second axis to actually see that.

    Deliberately a simplification, not a precise portfolio simulation: this
    engine doesn't track concurrent-position capital allocation (multiple
    tickers' trades can genuinely overlap in real calendar time), so this
    treats the pooled trade list as if positions were taken one at a time,
    compounding sequentially. That's a standard, defensible proxy for how
    much LOSS-CLUSTERING a config produces -- not a literal "what would my
    account balance have done" claim. Returns None if there are no
    resolved (non-OPEN) trades."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if not resolved:
        return None
    ordered = sorted(resolved, key=lambda t: t["entry_date"])
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in ordered:
        equity *= (1 + t["pnl_pct"] / 100)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return round(max_dd * 100, 2)


def monte_carlo_drawdown(trades: list[dict], n_simulations: int = 1000, seed: int = 42) -> dict | None:
    """How much of compute_max_drawdown()'s single chronological-order
    result is just the LUCK of the particular sequence a fixed set of
    trades happened to occur in? Reshuffles the same resolved trades'
    pnl_pct values into n_simulations random orderings, sequentially
    compounds each (same "not a real concurrent-position portfolio sim"
    simplification as compute_max_drawdown()), and reports the resulting
    max_drawdown distribution -- a materially worse P95/worst case than
    the actual chronological run means the smooth-looking real equity
    curve was partly lucky ordering, not just genuine edge quality.

    Reshuffles ORDER only, not trade SELECTION -- this answers "how lucky
    was the sequence", not "how lucky was the sample" (that's what
    ticker-holdout already checks, see item 69).

    Returns None with fewer than 3 resolved trades (mirrors
    compute_k_ratio()'s floor -- not enough trades for a distribution to
    mean anything)."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if len(resolved) < 3:
        return None

    pnl_pct = np.array([t["pnl_pct"] for t in resolved], dtype=float)
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(pnl_pct, (n_simulations, 1)), axis=1)

    equity = np.cumprod(1 + shuffled / 100, axis=1)
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


def compute_k_ratio(trades: list[dict]) -> float | None:
    """K-ratio: how CONSISTENTLY a sequential equity curve compounds over
    calendar time, distinct from sharpe_like (mean/stdev of per-trade
    returns) which is blind to *ordering* -- two configs with identical
    win_rate/sharpe_like can still differ sharply in whether gains accrue
    steadily or in one lumpy cluster. Built the same way as
    compute_max_drawdown(): compound resolved trades into a single
    sequential equity curve in chronological (entry_date) order (same
    "not a real concurrent-position portfolio sim" caveat applies), then
    fit an OLS regression of log(equity) against elapsed calendar days
    (log-space so compounding growth is linear, the standard K-ratio
    setup). K-ratio is the regression slope's t-statistic (slope divided
    by its own standard error) -- large and positive means equity grew
    steadily and reliably in a straight line; near zero or negative means
    the trend is noisy, flat, or reversing.

    Deliberately RAW, not annualized or scaled by sample size -- same
    "comparable only within a similarly-sized/timed run" convention
    sharpe_like already follows (see strategy-validation-pipeline memory
    point 12): only compare k_ratio between runs over the same calendar
    window and trade universe, never across runs of different length.

    Returns None with fewer than 3 resolved trades, when trades don't
    carry entry_date at all (same degrade-gracefully convention as
    _annualize_sharpe(), for live-Mongo-pooled trades that lack it), when
    every trade shares one entry_date (no time axis to regress against),
    or in the degenerate zero-residual-variance case (avoids a
    divide-by-zero rather than returning a nonsensical infinite value)."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    if len(resolved) < 3 or any("entry_date" not in t for t in resolved):
        return None

    ordered = sorted(resolved, key=lambda t: t["entry_date"])
    start = ordered[0]["entry_date"]
    equity = 1.0
    xs, ys = [], []
    for t in ordered:
        equity *= (1 + t["pnl_pct"] / 100)
        if equity <= 0:
            return None
        xs.append(float((t["entry_date"] - start).days))
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


@dataclass
class FoldResult:
    fold: Fold
    in_sample_trades: list[dict]
    out_sample_trades: list[dict]
    in_sample_metrics: dict
    out_sample_metrics: dict


def _run_fold_sequential(
    fold: Fold, ticker_data: dict[str, pd.DataFrame], market_data: pd.DataFrame,
    config: TradingConfig, earnings_data: dict, sector_lookup: dict, strategy: str,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
) -> FoldResult:
    """The actual per-fold work, shared by both the sequential and
    parallel paths of run_walk_forward() -- one fold's in-sample and
    out-of-sample backtest, for every ticker. Kept as a plain module-level
    function (not a closure) so it's picklable for ProcessPoolExecutor."""
    in_trades = run_backtest(
        ticker_data, market_data, fold.in_sample_start, fold.in_sample_end, config,
        earnings_data, sector_lookup, strategy, sector_data, pair_price_panels,
    )
    out_trades = run_backtest(
        ticker_data, market_data, fold.out_sample_start, fold.out_sample_end, config,
        earnings_data, sector_lookup, strategy, sector_data, pair_price_panels,
    )
    return FoldResult(
        fold=fold,
        in_sample_trades=in_trades,
        out_sample_trades=out_trades,
        in_sample_metrics=summarize_trades(in_trades),
        out_sample_metrics=summarize_trades(out_trades),
    )


# Populated once per worker process by _init_worker() -- module-level so
# _run_fold_worker() (which only receives small per-task arguments, not the
# full ticker_data/market_data) can reach it without re-sending the whole
# OHLCV universe over IPC for every one of the 56+ folds. Each worker process
# gets its own copy via the ProcessPoolExecutor initializer, set ONCE when
# the pool starts, not once per fold.
_worker_ticker_data: dict[str, pd.DataFrame] | None = None
_worker_market_data: pd.DataFrame | None = None
# Sector-ETF OHLCV (sector NAME -> OHLCV, see watchlist.SECTOR_ETF) -- same
# "loaded once per worker, not once per fold" treatment as ticker_data/
# market_data above, since it's the same kind of shared, sizable data
# (improvements.txt items 68/70). None (default) means no sector data was
# supplied to run_walk_forward() at all -- every worker's Sector_Relative_Strength
# then reads None/NaN, same as before this field existed.
_worker_sector_data: dict[str, pd.DataFrame] | None = None
# Pairs-strategy peer price panels (sector NAME -> wide Close-price
# DataFrame of every ticker in that sector) -- same "loaded once per
# worker, not once per fold" treatment as sector_data above
# (improvements.txt item 83). None means no peer data was supplied at all,
# every worker's Pair_Signal then always reads False.
_worker_pair_price_panels: dict[str, pd.DataFrame] | None = None


def _init_worker(
    ticker_data: dict[str, pd.DataFrame], market_data: pd.DataFrame,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
) -> None:
    global _worker_ticker_data, _worker_market_data, _worker_sector_data, _worker_pair_price_panels
    _worker_ticker_data = ticker_data
    _worker_market_data = market_data
    _worker_sector_data = sector_data
    _worker_pair_price_panels = pair_price_panels


def _run_fold_worker(
    fold: Fold, config: TradingConfig, earnings_data: dict, sector_lookup: dict, strategy: str,
) -> FoldResult:
    """Same work as _run_fold_sequential(), but reads ticker_data/market_data/
    sector_data/pair_price_panels from this worker process's own module-level
    copy (set once by _init_worker()) instead of taking them as arguments --
    only `fold` and `config` (small, cheap to pickle) actually cross the
    process boundary per task."""
    return _run_fold_sequential(
        fold, _worker_ticker_data, _worker_market_data, config, earnings_data, sector_lookup, strategy,
        _worker_sector_data, _worker_pair_price_panels,
    )


def run_walk_forward(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    folds: list[Fold],
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    sector_lookup: dict[str, str] | None = None,
    strategy: str = "rsi",
    parallel: bool = True,
    max_workers: int | None = None,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
) -> list[FoldResult]:
    """Run the backtest across every fold, in-sample and out-of-sample
    separately. Optuna (Phase 5) evaluates a candidate config by scoring it
    across the aggregate of every fold's out_sample_metrics -- never a
    single fold's in-sample fit. `sector_lookup` (optional) tags trades for
    compute_cluster_weights(). `strategy` (see run_backtest) selects "rsi"
    (default) or "breakout".

    Folds are fully independent (no shared state, no data dependency
    between them), so by default (`parallel=True`) this dispatches them
    across a process pool instead of looping sequentially -- this was the
    dominant cost of every Optuna search this session (each trial walks
    every fold, for every ticker, day by day). The pool's initializer loads
    ticker_data/market_data into each worker ONCE when the pool starts, not
    once per fold, so only the lightweight per-fold Fold/config objects
    actually cross the process boundary on each task. Results are returned
    in the same order as `folds` regardless of completion order. Pass
    `parallel=False` to force the old sequential behavior (e.g. for a
    direct correctness comparison, or if a single fold makes pool startup
    overhead not worth it -- also skipped automatically for <=1 fold).

    `max_workers` defaults to cpu_count()-4 (min 1), not the full core
    count -- sustained all-core usage during a long search measurably
    raises CPU temperature, and this leaves headroom for the OS/other apps
    rather than assuming exclusive use of the machine. Pass an explicit
    value (e.g. your full `os.cpu_count()`) for maximum speed if that
    tradeoff is fine for your setup, or 1 for the gentlest possible
    (though `parallel=False` is cheaper than a 1-worker pool for that).

    `sector_data` (optional, sector NAME -> OHLCV, see watchlist.SECTOR_ETF)
    backs the backtest/Optuna-only Sector_Relative_Strength filter
    (improvements.txt items 68/70) -- threaded through to run_backtest() on
    both the sequential and parallel paths (loaded once per worker process
    via the pool initializer on the parallel path, same treatment as
    ticker_data/market_data). None (default) means no sector data at all,
    identical to every caller from before this parameter existed.

    `pair_price_panels` (optional, sector NAME -> wide Close-price
    DataFrame of every ticker in that sector) backs the "pairs" strategy's
    partner-selection/spread-zscore mechanism (improvements.txt item 83) --
    threaded through identically to sector_data above, both paths. None
    (default) means no peer data at all, identical to every caller from
    before this parameter existed."""
    if not parallel or len(folds) <= 1:
        return [
            _run_fold_sequential(
                fold, ticker_data, market_data, config, earnings_data, sector_lookup, strategy, sector_data,
                pair_price_panels,
            )
            for fold in folds
        ]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 4)

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker,
        initargs=(ticker_data, market_data, sector_data, pair_price_panels),
    ) as executor:
        futures = [
            executor.submit(_run_fold_worker, fold, config, earnings_data, sector_lookup, strategy)
            for fold in folds
        ]
        return [f.result() for f in futures]
