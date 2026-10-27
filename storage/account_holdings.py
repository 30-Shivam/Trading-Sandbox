"""Account_Holdings persistence: what a REAL brokerage account (e.g. the Scotia iTRADE TFSA) holds -- quantities, native
currency, optional cost basis, and cash balances.

Separate from Current_Holdings (storage/holdings.py), which is a dollars-per-ticker list that review_positions.py reads
for stop/target reviews. Keeping them apart means importing a real account can never overwrite that list or feed it into
those reviews. Whole-ACCOUNT replace on every save (paste/import the full current statement), scoped by `account` so
other accounts are untouched.

Document shape: {"account": str, "symbol": yahoo symbol, "raw_symbol": str, "kind": "security"|"cash", "quantity": float,
"currency": "CAD"|"USD", "avg_cost": float | None, "updated_at": datetime}.
"""

from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Account_Holdings"


def ensure_indexes() -> None:
    get_db()[COLLECTION_NAME].create_index([("account", 1), ("symbol", 1)], unique=True)


def get_account_holdings(account: str) -> list[dict]:
    docs = get_db()[COLLECTION_NAME].find({"account": account}, {"_id": 0, "updated_at": 0, "account": 0})
    return list(docs)


def set_account_holdings(account: str, positions: list[dict]) -> int:
    """Replace ONE account's holdings. Returns the number of documents written. Duplicate symbols are merged
    (quantities summed; cost basis becomes the quantity-weighted average when both rows have one)."""
    merged: dict[str, dict] = {}
    for p in positions:
        cur = merged.get(p["symbol"])
        if cur is None:
            merged[p["symbol"]] = dict(p)
            continue
        q = cur["quantity"] + p["quantity"]
        if cur.get("avg_cost") and p.get("avg_cost") and q:
            cur["avg_cost"] = (cur["avg_cost"] * cur["quantity"] + p["avg_cost"] * p["quantity"]) / q
        else:
            cur["avg_cost"] = cur.get("avg_cost") or p.get("avg_cost")
        cur["quantity"] = q
    now = datetime.now(timezone.utc)
    docs = [{**{k: p.get(k) for k in ("symbol", "raw_symbol", "kind", "quantity", "currency", "avg_cost")},
             "account": account, "updated_at": now} for p in merged.values() if p["quantity"] != 0]
    col = get_db()[COLLECTION_NAME]
    col.delete_many({"account": account})
    if docs:
        col.insert_many(docs)
    return len(docs)
