"""Trade_Outcomes persistence.

Populated by the Phase 3 nightly settlement job (settle_trades.py), which
walks each unsettled Trade_Signals document forward through actual price
history using swingtrade.settle_trade(). Document shape:
    {
        "ticker": str,
        "signal_date": str,      # matches the source Trade_Signals document
        "entry_price": float,    # confirmed fill_price if set, else that signal's buy_price
        "exit_price": float,
        "exit_reason": "target_hit" | "stop_hit_intraday"
                        | "gap_down_stop" | "gap_up_target" | "expired",
        "status": "WIN" | "LOSS" | "EXPIRED",
        "pnl_pct": float,
        "holding_days": int,
        "exit_date": str,        # ISO date the trade actually resolved
        "confirmed_filled": bool,# carried over from the source Trade_Signals
                                  # doc -- most outcomes are mechanical
                                  # what-ifs, not real trades; this is how
                                  # you separate the two in reporting
        "settled_at": datetime,  # UTC, when this document was written
    }

Only terminal outcomes (WIN/LOSS/EXPIRED) get a document -- a trade that's
still OPEN simply has no Trade_Outcomes row yet, and its Trade_Signals
document stays settled=False so the next settlement run re-checks it.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Trade_Outcomes"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1), ("signal_date", 1)], unique=True)


def log_trade_outcome(
    ticker: str, signal_date: str, entry_price: float, result: dict, confirmed_filled: bool = False
) -> None:
    """Upsert a Trade_Outcomes document for a resolved trade. `result` is
    the dict returned by swingtrade.settle_trade() for a terminal status
    (WIN/LOSS/EXPIRED) -- never call this with an OPEN result."""
    if result["status"] == "OPEN":
        raise ValueError("cannot log an OPEN result -- the trade hasn't resolved yet")

    db = get_db()
    doc = {
        "ticker": ticker,
        "signal_date": signal_date,
        "entry_price": float(entry_price),
        "exit_price": float(result["exit_price"]),
        "exit_reason": result["exit_reason"],
        "status": result["status"],
        "pnl_pct": float(result["pnl_pct"]),
        "holding_days": int(result["holding_days"]),
        "exit_date": str(result["exit_date"]),
        "confirmed_filled": bool(confirmed_filled),
        "settled_at": datetime.now(timezone.utc),
    }
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date},
        {"$set": doc},
        upsert=True,
    )
