"""Current_Holdings persistence: a small, manually-maintained record of what
you're actually holding right now, in dollars committed per ticker.

This is deliberately NOT inferred from unsettled Trade_Signals -- a logged
signal doesn't mean you actually got filled (see the fill-tracking gap noted
in TUTORIAL.md), and you may hold things that never came from this system at
all. Instead, you tell it directly (dashboard sidebar), and
swingtrade.allocate_capital uses it to measure sector concentration against
your whole portfolio (holdings + today's fresh cash), not just today's cash
pool in isolation -- a 40% sector cap on a small daily cash pool means
nothing if the sector is already overweight in what you hold.

Document shape: {"ticker": str, "amount": float, "updated_at": datetime}.
Whole-collection replace on every save (set_holdings), not incremental
upserts -- this mirrors how the user actually edits it (paste your full
current holdings list), not something built up call by call over time.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Current_Holdings"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1)], unique=True)


def get_holdings() -> dict[str, float]:
    """Return {ticker: dollar amount currently committed}."""
    db = get_db()
    return {doc["ticker"]: doc["amount"] for doc in db[COLLECTION_NAME].find({})}


def set_holdings(holdings: dict[str, float]) -> None:
    """Replace the entire holdings record. Tickers with a non-positive
    amount are dropped rather than stored."""
    db = get_db()
    now = datetime.now(timezone.utc)
    docs = [
        {"ticker": ticker.strip().upper(), "amount": float(amount), "updated_at": now}
        for ticker, amount in holdings.items()
        if amount and float(amount) > 0
    ]
    db[COLLECTION_NAME].delete_many({})
    if docs:
        db[COLLECTION_NAME].insert_many(docs)
