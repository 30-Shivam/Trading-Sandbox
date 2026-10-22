"""Shared backtest primitives: Fold, fold generation, entry-fill search and trade settlement."""

from dataclasses import dataclass
import pandas as pd
from ..config import TradingConfig
from ..settlement import settle_trade, settle_trade_with_trailing


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
    within the window (not a real trade -- correctly excluded, not forced).

    Also re-exported publicly as swingtrade.find_entry_fill (2026-08-24) --
    settle_trades.py's live nightly settlement job uses this exact function
    to check whether an unconfirmed signal's resting limit was ever actually
    touched before grading it, instead of (the prior bug) assuming every
    signal filled at its literal buy_price on day one regardless of real
    price action. See settle_trades.py's own settle_one() docstring."""
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
    dollar_volume: float | None = None,
) -> dict:
    """Shared dispatch used by the 3 currently-active strategies' simulate
    functions (breakout, squeeze_breakout, ma_crossover -- both real and
    random-entry counterparts) -- routes to settle_trade_with_trailing()
    when config.trailing_stop_enabled, else the original settle_trade(),
    so each call site doesn't need to repeat this branch inline. The 4
    dormant strategies (pullback/breakout_retest/week52_high/momentum_burst/
    adx_trend_entry) still call settle_trade() directly, unaffected by
    trailing_stop_enabled -- see swingtrade/config.py's own field comment
    for why this is scoped to the active 3 only.

    `dollar_volume` (optional) passes through to whichever settle function
    is chosen -- see settle_trade()'s own docstring for the liquidity-tiered
    slippage rationale."""
    if config.trailing_stop_enabled:
        return settle_trade_with_trailing(
            buy_price=buy_price, stop_loss=stop_loss, sell_price=sell_price, atr=atr,
            bars_since_entry=bars_since_entry, config=config, dollar_volume=dollar_volume,
        )
    return settle_trade(
        buy_price=buy_price, stop_loss=stop_loss, sell_price=sell_price,
        bars_since_entry=bars_since_entry, config=config, dollar_volume=dollar_volume,
    )
