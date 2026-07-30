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
