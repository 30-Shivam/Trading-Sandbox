"""User decision/preference tracking (improvements.txt item 92) -- reads
what a user actually DID with logged signals (see
storage.record_user_decision(), storage/signals.py), not whether the
signal itself was good. A genuinely different question from every other
tracking in this project: ic_tracking.py asks "did this methodology's
score rank real outcomes correctly," this asks "does this match what I
actually chose to act on."

Deliberately does NOT attempt automated pattern-mining/clustering of
passed-reason free text (e.g. auto-detecting "you tend to pass on
high-RSI names") -- with realistically sparse early data, that risks
fabricating a false pattern from noise, the exact lesson this project has
re-learned repeatedly this session (ticker-holdout seed sensitivity,
item 69; sector-RS's own small-sample reversal, item 76). This surfaces
the raw digest for a human to notice their own patterns; automated
correlation is explicitly future work, worth building only once real
decisions accumulate.

Same split as ic_tracking.py: the pure aggregation logic
(_summarize_decision_docs) is directly unit-testable; the DB read
(summarize_decisions) is a thin wrapper over it, not separately covered
with a live/mocked database (see tests/test_ic_tracking.py's own
docstring for why that split exists).
"""
from storage.mongo import get_db


def _summarize_decision_docs(docs: list[dict]) -> dict:
    """Pure aggregation over already-fetched Trade_Signals documents (each
    expected to carry `user_decision`/`user_decision_reason`/
    `user_decision_at`, see storage.record_user_decision()). Returns
    {"acted_on_count":, "passed_count":, "passed_with_reason": [...]}
    where passed_with_reason is [{"ticker":, "signal_date":, "reason":,
    "decided_at":}, ...] sorted most-recent-first -- only for passed
    decisions that carry a non-empty reason (an unreasoned pass isn't
    useful to surface in a digest)."""
    acted_on_count = sum(1 for d in docs if d.get("user_decision") == "acted_on")
    passed_count = sum(1 for d in docs if d.get("user_decision") == "passed")

    passed_with_reason = [
        {
            "ticker": d["ticker"], "signal_date": d["signal_date"],
            "reason": d.get("user_decision_reason"), "decided_at": d.get("user_decision_at"),
        }
        for d in docs
        if d.get("user_decision") == "passed" and d.get("user_decision_reason")
    ]
    passed_with_reason.sort(key=lambda d: d["decided_at"] or "", reverse=True)

    return {
        "acted_on_count": acted_on_count,
        "passed_count": passed_count,
        "passed_with_reason": passed_with_reason,
    }


def summarize_decisions(strategy: str | None = None) -> dict:
    """Pools every Trade_Signals document with a recorded user_decision --
    optionally scoped to one `strategy` label, else every strategy pooled
    together. See _summarize_decision_docs() for the returned shape."""
    db = get_db()
    query = {"user_decision": {"$exists": True}}
    if strategy is not None:
        query["strategy"] = strategy
    docs = list(db["Trade_Signals"].find(
        query, {"ticker": 1, "signal_date": 1, "user_decision": 1, "user_decision_reason": 1, "user_decision_at": 1}
    ))
    return _summarize_decision_docs(docs)
