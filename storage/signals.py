"""Trade_Signals persistence.

Document shape (one per ticker per trading day):
    {
        "ticker": str,
        "signal_date": str,          # ISO date, matches compute_levels' As_Of
        "signal": "Strong Buy" | "Buy",
        "trade_score": float,
        "buy_price": float, "sell_price": float, "stop_loss": float,
        "rrr": float, "rsi": float, "atr": float,
        "distance_to_buy_pct": float,
        "shares_to_buy": float, "est_cost": float,
        "next_earnings_date": str | None,
        "catalyst_warning": bool,
        "config_snapshot": dict,     # swingtrade.TradingConfig.to_dict() at trigger time
        "settled": bool,             # flipped by the Phase 3 settlement job
        "created_at": datetime, "updated_at": datetime,
    }

Signals are logged for the *pre-allocation* Signal (Strong Buy / Buy as
scored by swingtrade.add_trade_score), not the capital-allocator's overlay --
"Insufficient Funds" reflects a personal cash constraint on a given day, not
a change in the underlying technical signal, and the learning loop (Phase 5)
needs to judge the signal itself independent of that.
"""

import math
from datetime import datetime, timezone

import pandas as pd

from .mongo import get_db

COLLECTION_NAME = "Trade_Signals"
LOGGABLE_SIGNALS = ("Strong Buy", "Buy")


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1), ("signal_date", 1)], unique=True)


def _native(value):
    """Coerce a pandas/numpy scalar to a plain, BSON-encodable Python value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (float64, bool_, int64, ...)
        return value.item()
    return value


def _build_document(row: dict, config_snapshot: dict, now: datetime) -> dict:
    next_earnings = _native(row.get("Next_Earnings_Date"))
    return {
        "ticker": _native(row["Ticker"]),
        "signal_date": str(row["As_Of"]),
        "signal": _native(row["Signal"]),
        "trade_score": _native(row["Trade_Score"]),
        "buy_price": _native(row["Buy_Price"]),
        "sell_price": _native(row["Sell_Price"]),
        "stop_loss": _native(row["Stop_Loss"]),
        "rrr": _native(row["RRR"]),
        "rsi": _native(row["RSI"]),
        "atr": _native(row["ATR"]),
        "distance_to_buy_pct": _native(row["Distance_to_Buy_Pct"]),
        "shares_to_buy": _native(row["Shares_To_Buy"]),
        "est_cost": _native(row["Est_Cost"]),
        "next_earnings_date": str(next_earnings) if next_earnings is not None else None,
        "catalyst_warning": bool(_native(row["Catalyst_Warning"])),
        "oversold_streak_days": _native(row.get("Oversold_Streak_Days")),
        "extended_decline_warning": bool(_native(row.get("Extended_Decline_Warning", False))),
        "config_snapshot": config_snapshot,
        "settled": False,
        "updated_at": now,
    }


def log_trade_signal(row: dict, config_snapshot: dict) -> None:
    """Upsert one Trade_Signals document, keyed on (ticker, signal_date).
    Re-running the same scan the same day updates the existing document in
    place instead of creating a duplicate."""
    db = get_db()
    now = datetime.now(timezone.utc)
    doc = _build_document(row, config_snapshot, now)

    db[COLLECTION_NAME].update_one(
        {"ticker": doc["ticker"], "signal_date": doc["signal_date"]},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


def log_trade_signals(df: pd.DataFrame, config_snapshot: dict) -> int:
    """Log every Strong Buy / Buy row in df (expects pre-allocation Signal
    values). Returns the number of signals written."""
    eligible = df[df["Signal"].isin(LOGGABLE_SIGNALS)]
    for _, row in eligible.iterrows():
        log_trade_signal(row.to_dict(), config_snapshot)
    return len(eligible)


def get_unsettled_signals() -> list[dict]:
    """Return every Trade_Signals document not yet resolved to a terminal
    outcome. The settlement job re-walks each of these from scratch every
    run -- cheap, and avoids needing to track incremental per-trade state."""
    db = get_db()
    return list(db[COLLECTION_NAME].find({"settled": {"$ne": True}}))


def mark_settled(ticker: str, signal_date: str) -> None:
    db = get_db()
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date},
        {"$set": {"settled": True}},
    )
