"""Guards against the exact bug class behind the DuplicateKeyError incident:
Trade_Signals/Trade_Outcomes had BOTH the correct 3-field
(ticker, signal_date, strategy) unique index AND a stale 2-field
(ticker, signal_date) leftover from an earlier migration that was never
dropped -- any two different strategies flagging the same ticker on the
same day collided against the stale index even though the correct one
would have allowed it. See improvements.txt for the incident and fix.

Read-only: only lists existing indexes, never creates/drops/writes
anything. Skips (not fails) if MONGODB_URI isn't available in the
environment -- e.g. a fork's PR without secrets -- same "degrade, don't
break the run" philosophy as the rest of this codebase.
"""
import pytest

import storage

EXPECTED_UNIQUE_KEY = [("ticker", 1), ("signal_date", 1), ("strategy", 1)]
COLLECTIONS_REQUIRING_STRATEGY_KEY = ("Trade_Signals", "Trade_Outcomes")


def _mongo_available() -> bool:
    try:
        storage.get_db()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")


@pytest.mark.parametrize("collection_name", COLLECTIONS_REQUIRING_STRATEGY_KEY)
def test_no_stale_two_field_unique_index(collection_name):
    db = storage.get_db()
    indexes = list(db[collection_name].list_indexes())

    unique_keys = [list(idx["key"].items()) for idx in indexes if idx.get("unique")]

    assert EXPECTED_UNIQUE_KEY in unique_keys, (
        f"{collection_name} is missing the expected 3-field unique index "
        f"{EXPECTED_UNIQUE_KEY} -- got unique indexes: {unique_keys}"
    )

    stale_two_field_key = [("ticker", 1), ("signal_date", 1)]
    assert stale_two_field_key not in unique_keys, (
        f"{collection_name} still has the stale 2-field unique index {stale_two_field_key} "
        "alongside the correct 3-field one -- this is exactly the bug that caused real "
        "DuplicateKeyErrors when two different strategies fired the same ticker/day "
        "(see improvements.txt). Drop it: db[collection].drop_index('ticker_1_signal_date_1')"
    )
