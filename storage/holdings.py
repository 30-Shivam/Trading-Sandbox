"""Current_Holdings persistence: a small, manually-maintained record of what
you're actually holding right now -- dollars committed per ticker, and
optionally your average cost per share.

This is deliberately NOT inferred from unsettled Trade_Signals -- a logged
signal doesn't mean you actually got filled (see the fill-tracking gap noted
in TUTORIAL.md), and you may hold things that never came from this system at
all. Instead, you tell it directly (dashboard sidebar). It's used two ways:
  - `amount` feeds swingtrade.allocate_capital's sector-concentration cap,
    measured against your whole portfolio (holdings + today's fresh cash),
    not today's cash pool in isolation.
  - `avg_cost` (optional -- a holding can be recorded with just an amount
    for allocation purposes, no cost basis) feeds swingtrade.review_holding,
    which checks a held position against the active config's ATR-based
    stop/target, anchored to your real entry price.

Document shape: {"ticker": str, "amount": float, "avg_cost": float | None,
"updated_at": datetime}. Whole-collection replace on every save
(set_holdings), not incremental upserts -- mirrors how the user actually
edits it (paste your full current holdings list), not something built up
call by call over time.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Current_Holdings"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1)], unique=True)


def get_holdings() -> dict[str, dict]:
    """Return {ticker: {"amount": dollars committed, "avg_cost": price/share or None}}."""
    db = get_db()
    return {
        doc["ticker"]: {"amount": doc["amount"], "avg_cost": doc.get("avg_cost")}
        for doc in db[COLLECTION_NAME].find({})
    }


def set_holdings(holdings: dict[str, dict]) -> None:
    """Replace the entire holdings record. Each value is
    {"amount": dollars committed, "avg_cost": price/share or None}.
    Tickers with a non-positive/missing amount are dropped rather than
    stored."""
    db = get_db()
    now = datetime.now(timezone.utc)
    docs = []
    for ticker, info in holdings.items():
        amount = float(info.get("amount") or 0)
        if amount <= 0:
            continue
        avg_cost = info.get("avg_cost")
        docs.append({
            "ticker": ticker.strip().upper(),
            "amount": amount,
            "avg_cost": float(avg_cost) if avg_cost else None,
            "updated_at": now,
        })
    db[COLLECTION_NAME].delete_many({})
    if docs:
        db[COLLECTION_NAME].insert_many(docs)
