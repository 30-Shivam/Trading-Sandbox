"""System_Config schema + indexes + read/write.

Written by the Phase 5 Optuna learning engine (optimize.py) as "candidate"
documents; read by Phase 6's Streamlit startup (the "active" document).
Promotion from candidate to active is a deliberate, human-gated step
(champion/challenger pattern -- see promote_config.py), never automatic.
Document shape:
    {
        "version": int,                # monotonically increasing
        "status": "active" | "candidate" | "retired",
        "params": dict,                # swingtrade.TradingConfig.to_dict()
        "metrics": dict,                # the backtest metrics that earned this candidate its spot
        "created_at": datetime,        # UTC
        "promoted_at": datetime | None,
        "notes": str,
    }

At most one document can have status "active" at a time -- enforced by a
partial unique index, not just application logic, so a concurrent run can't
accidentally create two.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "System_Config"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("version", 1)], unique=True)
    db[COLLECTION_NAME].create_index(
        [("status", 1)],
        unique=True,
        partialFilterExpression={"status": "active"},
        name="status_active_unique",
    )


def get_active_config() -> dict | None:
    """Return the active System_Config document's `params` dict, or None if
    nothing has been promoted yet (callers should fall back to
    swingtrade.DEFAULT_CONFIG)."""
    doc = get_active_config_doc()
    return doc["params"] if doc else None


def get_active_config_doc() -> dict | None:
    """Return the full active System_Config document (version, params,
    metrics, notes, promoted_at, ...), or None if nothing is active. Useful
    when a caller wants to display provenance (e.g. "using v3"), not just
    the bare params dict."""
    db = get_db()
    return db[COLLECTION_NAME].find_one({"status": "active"})


def get_next_version() -> int:
    db = get_db()
    doc = db[COLLECTION_NAME].find_one(sort=[("version", -1)])
    return (doc["version"] + 1) if doc else 1


def write_candidate(params: dict, notes: str = "", metrics: dict | None = None) -> int:
    """Insert a new candidate System_Config document. Never touches
    `active` -- promotion is a separate, deliberate step."""
    db = get_db()
    version = get_next_version()
    doc = {
        "version": version,
        "status": "candidate",
        "params": params,
        "metrics": metrics or {},
        "created_at": datetime.now(timezone.utc),
        "promoted_at": None,
        "notes": notes,
    }
    db[COLLECTION_NAME].insert_one(doc)
    return version


def list_configs() -> list[dict]:
    db = get_db()
    return list(db[COLLECTION_NAME].find({}).sort("version", -1))


def get_config_by_version(version: int) -> dict | None:
    db = get_db()
    return db[COLLECTION_NAME].find_one({"version": version})


def promote_candidate(version: int) -> None:
    """Promote one System_Config document to active, retiring whatever was
    active before. Deliberately manual -- optimize.py never calls this."""
    db = get_db()
    doc = db[COLLECTION_NAME].find_one({"version": version})
    if doc is None:
        raise ValueError(f"no System_Config document with version={version}")
    db[COLLECTION_NAME].update_many({"status": "active"}, {"$set": {"status": "retired"}})
    db[COLLECTION_NAME].update_one(
        {"version": version},
        {"$set": {"status": "active", "promoted_at": datetime.now(timezone.utc)}},
    )
