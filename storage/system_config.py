"""System_Config schema + indexes.

Written by the Phase 5 Optuna learning engine (as "candidate" documents) and
read by Phase 6's Streamlit startup (the "active" document) -- neither is
implemented yet; this module only defines the collection's shape and indexes
so far. Documented document shape:
    {
        "version": int,               # monotonically increasing
        "status": "active" | "candidate",
        "params": dict,                # swingtrade.TradingConfig.to_dict()
        "created_at": datetime,        # UTC
        "promoted_at": datetime | None,
        "notes": str,
    }

Exactly one document should carry status "active" at a time; promotion from
"candidate" to "active" is a deliberate, human-gated step (champion/challenger
pattern), not automatic.
"""

from .mongo import get_db

COLLECTION_NAME = "System_Config"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("status", 1)])
    db[COLLECTION_NAME].create_index([("version", 1)], unique=True)
