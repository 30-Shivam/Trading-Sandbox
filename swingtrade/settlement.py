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
            return _resolve("LOSS", open_, "gap_down_stop", holding_days, buy_price, bar_date)
        if open_ >= sell_price:
            return _resolve("WIN", open_, "gap_up_target", holding_days, buy_price, bar_date)
        # Same-day stop-and-target ambiguity resolves conservatively: stop first.
        if low <= stop_loss:
            return _resolve("LOSS", stop_loss, "stop_hit_intraday", holding_days, buy_price, bar_date)
        if high >= sell_price:
            return _resolve("WIN", sell_price, "target_hit", holding_days, buy_price, bar_date)

        if holding_days >= config.max_holding_days:
            return _resolve("EXPIRED", close, "expired", holding_days, buy_price, bar_date)

    return {"status": "OPEN"}


def _resolve(status: str, exit_price: float, exit_reason: str, holding_days: int, buy_price: float, exit_date) -> dict:
    return {
        "status": status,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "pnl_pct": round(((exit_price - buy_price) / buy_price) * 100, 2),
        "exit_date": pd.Timestamp(exit_date).date(),
    }
