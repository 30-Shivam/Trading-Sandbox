"""Trade_Outcomes schema + indexes.

Populated by the Phase 3 nightly settlement job (not yet implemented --
this module only defines the collection's shape and indexes so far).
Documented document shape, per the gap-aware settlement design in
ARCHITECTURE_PLAN.md:
    {
        "ticker": str,
        "signal_date": str,      # matches the source Trade_Signals document
        "entry_price": float,    # == that signal's buy_price
        "exit_price": float,
        "exit_reason": "target_hit" | "stop_hit_intraday"
                        | "gap_down_stop" | "gap_up_target" | "expired",
        "status": "WIN" | "LOSS" | "EXPIRED",
        "pnl_pct": float,
        "holding_days": int,
        "settled_at": datetime,  # UTC
    }
"""

from .mongo import get_db

COLLECTION_NAME = "Trade_Outcomes"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1), ("signal_date", 1)], unique=True)
