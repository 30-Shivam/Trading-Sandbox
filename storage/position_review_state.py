"""Position_Review_State persistence: the last Recommendation ("HOLD",
"SELL (stop breached)", "SELL (target hit)") headlessly observed for each
held ticker with a cost basis, via review_positions.py.

Exists purely to detect a FLIP into SELL without re-alerting every single
day a position stays in the same SELL state. Stop_Loss/Sell_Price are
anchored to avg_cost but sized off TODAY's rolling ATR (see
swingtrade.review_holding) -- they move daily, so a genuine SELL -> HOLD
flip-back is possible, not just one-directional, and this state needs to
be compared against on every run rather than assumed monotonic.

Deliberately a separate collection from Current_Holdings (storage/holdings.py)
rather than a field bolted onto it: that collection does a full
delete_many+insert_many on every user edit (set_holdings), which would
silently wipe this state every time someone re-pastes their holdings list.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Position_Review_State"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1)], unique=True)


def get_position_review_state() -> dict[str, str]:
    """Return {ticker: last observed Recommendation string}. A ticker with
    no prior observation (new holding, or never successfully reviewed) is
    simply absent -- callers should treat that as "HOLD" (the default,
    unflagged state) so a newly-added position that's already broken still
    triggers a flip alert rather than being silently skipped."""
    db = get_db()
    return {doc["ticker"]: doc["recommendation"] for doc in db[COLLECTION_NAME].find({})}


def set_position_review_state(recommendations: dict[str, str]) -> None:
    """Upsert this run's observed Recommendation for each ticker given.
    Per-ticker upsert (not a whole-collection replace) so tickers not
    included in this call are left untouched."""
    db = get_db()
    now = datetime.now(timezone.utc)
    for ticker, recommendation in recommendations.items():
        db[COLLECTION_NAME].update_one(
            {"ticker": ticker},
            {"$set": {"recommendation": recommendation, "updated_at": now}},
            upsert=True,
        )


def prune_position_review_state(current_tickers: set[str]) -> None:
    """Delete state for tickers no longer in Current_Holdings -- if removed
    then re-added later, it should be treated as a fresh position (no
    memory of the old flip state), not compared against a stale
    recommendation from an unrelated prior holding period."""
    db = get_db()
    db[COLLECTION_NAME].delete_many({"ticker": {"$nin": sorted(current_tickers)}})
