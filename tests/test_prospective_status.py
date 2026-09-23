"""ic_tracking.prospective_status() -- added 2026-09-23 after the
multiple-testing audit found every strategy ever promoted to real capital
was judged on retrospective tune/holdout evidence alone, with no minimum
elapsed REAL calendar time required before promotion. methodology_report()'s
trust_floor_met already corrects for trade count (tier x cluster weighting),
but a burst of correlated signals on day one could still clear it in days,
not months -- prospective_status() adds a separate, additional calendar-time
gate on top.

Mongo is monkeypatched (a fake db/collection with just the .find_one() shape
this function needs) -- no real network call, same
dependency-injection-free convention as test_settle_one_entry_fill_gate.py.
"""
import pandas as pd

import ic_tracking as ic


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find_one(self, query, sort=None):
        matches = [d for d in self._docs if d.get("strategy") == query.get("strategy")]
        if not matches:
            return None
        if sort:
            key, direction = sort[0]
            matches.sort(key=lambda d: d[key], reverse=(direction == -1))
        return matches[0]


class _FakeDB:
    def __init__(self, docs):
        self._col = _FakeCollection(docs)

    def __getitem__(self, name):
        return self._col


def _patch_db(monkeypatch, docs):
    monkeypatch.setattr(ic, "get_db", lambda: _FakeDB(docs))


def test_no_signals_at_all_returns_none_days_and_every_floor_false(monkeypatch):
    _patch_db(monkeypatch, [])

    result = ic.prospective_status("accruals_rank")

    assert result["days_tracked"] is None
    assert result["days_floor_met"] is False
    assert result["prospective_floor_met"] is False


def test_recent_first_signal_fails_the_days_floor(monkeypatch):
    recent_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    _patch_db(monkeypatch, [{"strategy": "accruals_rank", "signal_date": recent_date}])
    monkeypatch.setattr(ic, "methodology_report",
                         lambda strategy: {"trust_floor_met": True, "effective_n_settled": 999.0, "overall_ic": 0.5})

    result = ic.prospective_status("accruals_rank", min_days=60)

    assert result["days_tracked"] == 5
    assert result["days_floor_met"] is False
    # even with a trust_floor_met=True trade-count report, the calendar gate must still block it
    assert result["prospective_floor_met"] is False


def test_old_first_signal_with_trust_floor_met_passes_both_gates(monkeypatch):
    old_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    _patch_db(monkeypatch, [{"strategy": "accruals_rank", "signal_date": old_date}])
    monkeypatch.setattr(ic, "methodology_report",
                         lambda strategy: {"trust_floor_met": True, "effective_n_settled": 25.0, "overall_ic": 0.3})

    result = ic.prospective_status("accruals_rank", min_days=60)

    assert result["days_tracked"] == 90
    assert result["days_floor_met"] is True
    assert result["prospective_floor_met"] is True


def test_old_first_signal_but_trust_floor_not_met_still_fails(monkeypatch):
    old_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    _patch_db(monkeypatch, [{"strategy": "accruals_rank", "signal_date": old_date}])
    monkeypatch.setattr(ic, "methodology_report",
                         lambda strategy: {"trust_floor_met": False, "effective_n_settled": 3.0, "overall_ic": None})

    result = ic.prospective_status("accruals_rank", min_days=60)

    assert result["days_floor_met"] is True
    assert result["trust_floor_met"] is False
    assert result["prospective_floor_met"] is False


def test_only_matching_strategy_signals_are_considered(monkeypatch):
    old_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    recent_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    _patch_db(monkeypatch, [
        {"strategy": "some_other_strategy", "signal_date": old_date},
        {"strategy": "accruals_rank", "signal_date": recent_date},
    ])
    monkeypatch.setattr(ic, "methodology_report",
                         lambda strategy: {"trust_floor_met": False, "effective_n_settled": 0.0, "overall_ic": None})

    result = ic.prospective_status("accruals_rank", min_days=60)

    assert result["days_tracked"] == 5   # not 200 -- the other strategy's older signal must not leak in
