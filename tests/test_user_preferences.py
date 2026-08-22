"""user_preferences.py -- hand-computable unit tests for the pure
aggregation logic (_summarize_decision_docs). summarize_decisions() itself
is a thin Mongo read wrapper and deliberately not covered here with a
live/mocked database -- see tests/test_ic_tracking.py's own docstring for
this codebase's established convention.

Also covers storage.record_user_decision()'s one piece of DB-free logic
(input validation, which raises before ever touching Mongo) -- the actual
field-setting behavior is a thin Mongo write, same "not unit tested"
convention as confirm_fill()/unconfirm_fill() (storage/signals.py), which
have never had dedicated tests either.
"""
import pytest

import storage
from user_preferences import _summarize_decision_docs


def test_record_user_decision_rejects_invalid_decision_value():
    with pytest.raises(ValueError):
        storage.record_user_decision("AAPL", "2026-08-01", "rsi", "maybe")


def _doc(ticker, signal_date, decision, reason=None, decided_at=None):
    d = {"ticker": ticker, "signal_date": signal_date, "user_decision": decision}
    if reason is not None:
        d["user_decision_reason"] = reason
    if decided_at is not None:
        d["user_decision_at"] = decided_at
    return d


def test_counts_acted_on_and_passed_separately():
    docs = [
        _doc("AAPL", "2026-08-01", "acted_on"),
        _doc("MSFT", "2026-08-02", "acted_on"),
        _doc("NVDA", "2026-08-03", "passed", reason="too extended"),
    ]
    result = _summarize_decision_docs(docs)
    assert result["acted_on_count"] == 2
    assert result["passed_count"] == 1


def test_empty_input_returns_zero_counts_and_empty_digest():
    result = _summarize_decision_docs([])
    assert result == {"acted_on_count": 0, "passed_count": 0, "passed_with_reason": []}


def test_passed_without_reason_excluded_from_digest_but_still_counted():
    docs = [_doc("AAPL", "2026-08-01", "passed")]  # no reason
    result = _summarize_decision_docs(docs)
    assert result["passed_count"] == 1
    assert result["passed_with_reason"] == []


def test_acted_on_never_appears_in_passed_digest():
    docs = [_doc("AAPL", "2026-08-01", "acted_on", reason="irrelevant if present")]
    result = _summarize_decision_docs(docs)
    assert result["passed_with_reason"] == []


def test_passed_with_reason_digest_shape():
    docs = [_doc("NVDA", "2026-08-03", "passed", reason="too extended", decided_at="2026-08-03T10:00:00")]
    result = _summarize_decision_docs(docs)
    assert result["passed_with_reason"] == [
        {"ticker": "NVDA", "signal_date": "2026-08-03", "reason": "too extended", "decided_at": "2026-08-03T10:00:00"}
    ]


def test_passed_with_reason_sorted_most_recent_first():
    docs = [
        _doc("A", "2026-08-01", "passed", reason="r1", decided_at="2026-08-01T00:00:00"),
        _doc("B", "2026-08-03", "passed", reason="r3", decided_at="2026-08-03T00:00:00"),
        _doc("C", "2026-08-02", "passed", reason="r2", decided_at="2026-08-02T00:00:00"),
    ]
    result = _summarize_decision_docs(docs)
    tickers_in_order = [d["ticker"] for d in result["passed_with_reason"]]
    assert tickers_in_order == ["B", "C", "A"]
