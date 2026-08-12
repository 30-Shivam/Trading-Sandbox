"""Pure, gap-aware trade settlement. No network, no Mongo -- takes the OHLCV
bars since entry in, returns a resolution dict out. Reusable by both the
live nightly settlement job and the Phase 4 walk-forward backtester, so a
trade is graded identically everywhere.

A stop-loss is a *trigger* price, not a guaranteed *fill* price: if a day's
Open has already gapped through the stop, the realistic fill is at that
Open, not at the (better) stop level. A limit sell can't fill worse than its
limit, only better, so a gap up through the target still fills at the
gapped-up Open. That asymmetry is why gap handling only helps the stop side
and only helps (never hurts) the target side. See ARCHITECTURE_PLAN.md
section 3 for the full design rationale.

Execution realism: config.slippage_pct haircuts ONLY stop_hit_intraday fills
-- gap fills already use the real traded Open (already realistically
pessimistic), and target_hit/gap_up_target are limit fills that by
definition can't legitimately execute worse than their limit. Every
resolution additionally pays config.commission_pct_per_trade (a % of trade
value, not a $ amount -- this function has no notion of position size/share
count, so a %-of-value cost is the only form that stays consistent
regardless of how big a position was). Both default to modest-but-nonzero
values so a backtest doesn't implicitly assume frictionless execution.
"""

import pandas as pd

from .config import DEFAULT_CONFIG, TradingConfig


def settle_trade(
    buy_price: float,
    stop_loss: float,
    sell_price: float,
    bars_since_entry: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
) -> dict:
    """`bars_since_entry` must be daily OHLCV rows strictly AFTER the entry
    date, in chronological order (oldest first) -- the entry day itself is
    not part of this walk.

    Returns a dict with `status` in {"WIN", "LOSS", "EXPIRED", "OPEN"}.
    WIN/LOSS/EXPIRED additionally include exit_price, exit_reason,
    holding_days, pnl_pct, exit_date. OPEN carries no other keys -- the
    signal simply hasn't resolved yet and should be re-checked later.
    """
    for holding_days, (bar_date, bar) in enumerate(bars_since_entry.iterrows(), start=1):
        open_ = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        if open_ <= stop_loss:
            return _resolve("LOSS", open_, "gap_down_stop", holding_days, buy_price, bar_date, config)
        if open_ >= sell_price:
            return _resolve("WIN", open_, "gap_up_target", holding_days, buy_price, bar_date, config)
        # Same-day stop-and-target ambiguity resolves conservatively: stop first.
        if low <= stop_loss:
            slipped_stop = stop_loss * (1 - config.slippage_pct)
            return _resolve("LOSS", slipped_stop, "stop_hit_intraday", holding_days, buy_price, bar_date, config)
        if high >= sell_price:
            return _resolve("WIN", sell_price, "target_hit", holding_days, buy_price, bar_date, config)

        if holding_days >= config.max_holding_days:
            return _resolve("EXPIRED", close, "expired", holding_days, buy_price, bar_date, config)

    return {"status": "OPEN"}


def _resolve(
    status: str, exit_price: float, exit_reason: str, holding_days: int, buy_price: float, exit_date,
    config: TradingConfig = DEFAULT_CONFIG,
) -> dict:
    pnl_pct = ((exit_price - buy_price) / buy_price) * 100 - config.commission_pct_per_trade
    return {
        "status": status,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "pnl_pct": round(pnl_pct, 2),
        "exit_date": pd.Timestamp(exit_date).date(),
    }


def settle_trade_with_trailing(
    buy_price: float,
    stop_loss: float,
    sell_price: float,
    atr: float,
    bars_since_entry: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
) -> dict:
    """Hybrid fixed-then-trailing counterpart to settle_trade() -- a trade
    behaves EXACTLY like settle_trade() (same fixed stop_loss/sell_price,
    same gap-aware conservative tie-breaking) until `sell_price` (the
    original target) is first reached. At that point, instead of an
    automatic exit, this switches to a trailing stop
    (config.trailing_stop_atr_multiplier x `atr` below the running highest
    High since target was reached, ratcheting up only, never down) and
    keeps holding -- directly answering "should this trade be held past
    its target for more profit" at the mechanical level, deliberately
    scoped so trades that NEVER reach target are completely unaffected
    (see swingtrade/config.py's trailing_stop_enabled for why this
    minimally-invasive design was chosen over a from-entry chandelier
    trail: it isolates the change's effect cleanly for validation).

    Only called when config.trailing_stop_enabled is True -- callers
    branch between this and settle_trade() themselves (see e.g.
    simulate_breakout_signals()); this function has no fallback path of
    its own, unlike the entry-side optional filters elsewhere in this
    codebase, since exit-model selection is a caller-level decision, not
    a per-row gate.

    Status is determined HONESTLY from the actual exit price vs. buy_price
    (WIN if exit_price > buy_price after commission, else LOSS) rather
    than assumed -- a trailing stop set wider than the initial run-up could
    theoretically still resolve at a loss in a pathological case, and this
    should be reported accurately, not silently mislabeled.

    `atr` is the entry-time ATR already computed by the caller (same value
    used to size the original stop_loss/sell_price) -- passed explicitly
    since settle_trade() itself has no notion of ATR at all."""
    trailing_active = False
    highest_high = None
    trailing_stop = None

    for holding_days, (bar_date, bar) in enumerate(bars_since_entry.iterrows(), start=1):
        open_ = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        if not trailing_active:
            if open_ <= stop_loss:
                return _resolve("LOSS", open_, "gap_down_stop", holding_days, buy_price, bar_date, config)
            if open_ >= sell_price:
                # Gapped up through the original target -- start trailing
                # from this bar's own high instead of exiting immediately.
                trailing_active = True
                highest_high = high
                trailing_stop = highest_high - config.trailing_stop_atr_multiplier * atr
                continue
            # Same-day stop-and-target ambiguity resolves conservatively: stop first.
            if low <= stop_loss:
                slipped_stop = stop_loss * (1 - config.slippage_pct)
                return _resolve("LOSS", slipped_stop, "stop_hit_intraday", holding_days, buy_price, bar_date, config)
            if high >= sell_price:
                # Target reached intraday -- start trailing from this bar's
                # own high, same "continue holding" treatment as the gap-up case.
                trailing_active = True
                highest_high = high
                trailing_stop = highest_high - config.trailing_stop_atr_multiplier * atr
                continue
            if holding_days >= config.max_holding_days:
                return _resolve("EXPIRED", close, "expired", holding_days, buy_price, bar_date, config)
        else:
            # Check today's Open against the trailing stop level as it stood
            # at YESTERDAY's close, BEFORE folding in today's own High --
            # you can't know today's own high before today's session opens,
            # so gapping through a stop that only exists because of today's
            # own subsequent run-up isn't real. Only after this check do we
            # ratchet highest_high/trailing_stop using today's actual
            # intraday action, then check the (possibly now-updated) level
            # against today's Low.
            if open_ <= trailing_stop:
                status = "WIN" if open_ > buy_price else "LOSS"
                return _resolve(status, open_, "gap_down_trailing_stop", holding_days, buy_price, bar_date, config)

            highest_high = max(highest_high, high)
            trailing_stop = max(trailing_stop, highest_high - config.trailing_stop_atr_multiplier * atr)
            if low <= trailing_stop:
                slipped_trailing_stop = trailing_stop * (1 - config.slippage_pct)
                status = "WIN" if slipped_trailing_stop > buy_price else "LOSS"
                return _resolve(
                    status, slipped_trailing_stop, "trailing_stop_hit", holding_days, buy_price, bar_date, config
                )
            if holding_days >= config.max_holding_days:
                status = "WIN" if close > buy_price else "LOSS"
                return _resolve(status, close, "expired_after_target", holding_days, buy_price, bar_date, config)

    return {"status": "OPEN"}
