"""reconcile_settled_outcomes.py -- offline tests for _run()'s verdict
bookkeeping and main()'s dry-run safety. Collections are in-memory fakes and
settle_trades.settle_one is monkeypatched, so no Mongo or yfinance traffic.
"""
import ast
import sys
from types import SimpleNamespace

import pytest

import reconcile_settled_outcomes as rec
import settle_trades
import storage


class FakeCollection:
    def __init__(self, docs=()):
        self.docs = list(docs)
        self.deleted = []

    def find(self, query):
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    def find_one(self, query):
        found = self.find(query)
        return found[0] if found else None

    def delete_one(self, query):
        self.deleted.append(query)


def _outcome(ticker, status="WIN", pnl=5.0, date="2026-08-01"):
    return {"ticker": ticker, "signal_date": date, "strategy": "rsi_mean_reversion", "status": status, "pnl_pct": pnl}


def _signal(ticker, date="2026-08-01"):
    return {"ticker": ticker, "signal_date": date, "strategy": "rsi_mean_reversion"}


def _args(apply=False):
    return SimpleNamespace(strategy="rsi_mean_reversion", apply=apply)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(rec.time, "sleep", lambda s: None)


def _settle_with(monkeypatch, mapping):
    """mapping: ticker -> outcome string, or an Exception to raise."""
    def fake(signal):
        val = mapping[signal["ticker"]]
        if isinstance(val, Exception):
            raise val
        return val, None

    monkeypatch.setattr(settle_trades, "settle_one", fake)


def _summary(out):
    line = [l for l in out.splitlines() if l.startswith("Summary:")][0]
    return ast.literal_eval(line.removeprefix("Summary: "))


def test_unchanged_verdict_within_tolerance(monkeypatch, capsys):
    outcomes = FakeCollection([_outcome("AAA", "WIN", 5.0)])
    signals = FakeCollection([_signal("AAA")])
    _settle_with(monkeypatch, {"AAA": "WIN (target, +5.00%, held 3d, unconfirmed)"})
    rec._run(_args(), outcomes, signals)
    counts = _summary(capsys.readouterr().out)
    assert counts["corrected_same_verdict"] == 1 and counts["corrected_new_verdict"] == 0
    assert outcomes.deleted == []


@pytest.mark.parametrize("new_str", [
    "LOSS (stop, -4.00%, held 2d, unconfirmed)",       # status changed
    "WIN (target, +9.00%, held 3d, CONFIRMED)",         # same status, pnl changed
    "EXPIRED (time, +0.50%, held 20d, unconfirmed)",
])
def test_changed_verdict_is_counted_as_corrected_and_not_deleted(monkeypatch, capsys, new_str):
    outcomes = FakeCollection([_outcome("AAA", "WIN", 5.0)])
    _settle_with(monkeypatch, {"AAA": new_str})
    rec._run(_args(), outcomes, FakeCollection([_signal("AAA")]))
    assert _summary(capsys.readouterr().out)["corrected_new_verdict"] == 1 and outcomes.deleted == []


@pytest.mark.parametrize("new_str", [
    "NEVER_FILLED (resting limit never touched within max_entry_wait_days)",
    "waiting for entry fill",
    "OPEN (still open)",
    "no new bars yet",
])
def test_no_longer_valid_outcome_is_deleted(monkeypatch, capsys, new_str):
    outcomes = FakeCollection([_outcome("AAA", "WIN", 170.98)])
    _settle_with(monkeypatch, {"AAA": new_str})
    rec._run(_args(apply=True), outcomes, FakeCollection([_signal("AAA")]))
    assert outcomes.deleted == [{"ticker": "AAA", "signal_date": "2026-08-01", "strategy": "rsi_mean_reversion"}]
    out = capsys.readouterr().out
    assert "deleting" in out and "WOULD DELETE" not in out


def test_dry_run_wording_says_would_delete_and_footer(monkeypatch, capsys):
    outcomes = FakeCollection([_outcome("AAA")])
    _settle_with(monkeypatch, {"AAA": "OPEN (still open)"})
    rec._run(_args(apply=False), outcomes, FakeCollection([_signal("AAA")]))
    out = capsys.readouterr().out
    assert "WOULD DELETE" in out and "This was a DRY RUN" in out and "DRY RUN (no changes" in out


def test_missing_signal_doc_is_skipped_without_settling(monkeypatch, capsys):
    outcomes = FakeCollection([_outcome("OLD")])
    monkeypatch.setattr(settle_trades, "settle_one", lambda s: pytest.fail("must not settle"))
    rec._run(_args(), outcomes, FakeCollection([]))
    assert _summary(capsys.readouterr().out)["skipped_no_signal_doc"] == 1 and outcomes.deleted == []


def test_settle_error_including_network_timeout_is_counted_and_run_continues(monkeypatch, capsys):
    outcomes = FakeCollection([_outcome("BAD"), _outcome("GOOD")])
    signals = FakeCollection([_signal("BAD"), _signal("GOOD")])
    _settle_with(monkeypatch, {"BAD": TimeoutError("yahoo timeout"),
                               "GOOD": "WIN (target, +5.00%, held 3d, unconfirmed)"})
    rec._run(_args(), outcomes, signals)
    out = capsys.readouterr().out
    counts = _summary(out)
    assert counts["error"] == 1 and counts["corrected_same_verdict"] == 1
    assert "[ERROR] yahoo timeout" in out


def test_empty_strategy_reports_zero_and_makes_no_calls(monkeypatch, capsys):
    monkeypatch.setattr(settle_trades, "settle_one", lambda s: pytest.fail("nothing to settle"))
    rec._run(_args(), FakeCollection([]), FakeCollection([]))
    out = capsys.readouterr().out
    assert "Found 0 existing" in out and "'error': 0" in out


def test_only_outcomes_for_requested_strategy_are_processed(monkeypatch, capsys):
    other = {**_outcome("ZZZ"), "strategy": "breakout"}
    outcomes = FakeCollection([_outcome("AAA"), other])
    _settle_with(monkeypatch, {"AAA": "WIN (target, +5.00%, held 3d, unconfirmed)"})
    rec._run(_args(), outcomes, FakeCollection([_signal("AAA")]))
    assert "Found 1 existing" in capsys.readouterr().out


def test_rate_limit_sleep_between_but_not_before_first(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rec.time, "sleep", sleeps.append)
    names = ["A", "B", "C"]
    outcomes = FakeCollection([_outcome(n) for n in names])
    signals = FakeCollection([_signal(n) for n in names])
    _settle_with(monkeypatch, {n: "WIN (target, +5.00%, held 3d, unconfirmed)" for n in names})
    rec._run(_args(), outcomes, signals)
    assert sleeps == [rec.REQUEST_DELAY_SEC] * 2


def test_unparseable_pnl_in_outcome_string_raises(monkeypatch):
    """Documents the current contract: a WIN/LOSS/EXPIRED string whose pnl field can't be
    parsed is NOT routed to the per-ticker [ERROR] handler (parsing is outside its try)."""
    outcomes = FakeCollection([_outcome("AAA")])
    _settle_with(monkeypatch, {"AAA": "WIN (target, garbage, held 3d)"})
    with pytest.raises(ValueError):
        rec._run(_args(), outcomes, FakeCollection([_signal("AAA")]))


# ---------------------------------------------------------------- main()

class FakeDb:
    def __init__(self, outcomes, signals):
        self.collections = {"Trade_Outcomes": outcomes, "Trade_Signals": signals}

    def __getitem__(self, name):
        return self.collections[name]


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["reconcile_settled_outcomes.py", *argv])
    rec.main()


def test_main_requires_strategy(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch)
    assert exc.value.code == 2


def test_main_mongo_not_configured_exits_1(monkeypatch, capsys):
    def no_mongo():
        raise storage.MongoNotConfigured("MONGODB_URI missing")

    monkeypatch.setattr(storage, "ensure_indexes", no_mongo)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "--strategy", "rsi_mean_reversion")
    assert exc.value.code == 1 and "[ERROR] MONGODB_URI missing" in capsys.readouterr().err


def _wire_main(monkeypatch, outcomes, signals, new_str):
    monkeypatch.setattr(storage, "ensure_indexes", lambda: None)
    monkeypatch.setattr(storage, "get_db", lambda: FakeDb(outcomes, signals))
    writes = []
    monkeypatch.setattr(storage, "log_trade_outcome", lambda *a, **k: writes.append("log_trade_outcome"))
    monkeypatch.setattr(storage, "mark_settled", lambda *a, **k: writes.append("mark_settled"))

    def real_like_settle(signal):
        # Mimics settle_one's write side effects by going through the (possibly patched) storage module.
        storage.log_trade_outcome(signal["ticker"])
        storage.mark_settled(signal["ticker"])
        return new_str, None

    monkeypatch.setattr(settle_trades, "settle_one", real_like_settle)
    return writes


def test_main_dry_run_suppresses_every_write(monkeypatch):
    outcomes = FakeCollection([_outcome("AAA")])
    writes = _wire_main(monkeypatch, outcomes, FakeCollection([_signal("AAA")]), "OPEN (still open)")
    _main(monkeypatch, "--strategy", "rsi_mean_reversion")
    assert writes == [] and outcomes.deleted == []


def test_main_apply_lets_writes_and_delete_through(monkeypatch):
    outcomes = FakeCollection([_outcome("AAA")])
    writes = _wire_main(monkeypatch, outcomes, FakeCollection([_signal("AAA")]), "OPEN (still open)")
    _main(monkeypatch, "--strategy", "rsi_mean_reversion", "--apply")
    assert writes == ["log_trade_outcome", "mark_settled"] and len(outcomes.deleted) == 1


def test_main_dry_run_restores_storage_functions_afterwards(monkeypatch):
    outcomes = FakeCollection([_outcome("AAA")])
    _wire_main(monkeypatch, outcomes, FakeCollection([]), "OPEN (still open)")
    before = (storage.log_trade_outcome, storage.mark_settled)
    _main(monkeypatch, "--strategy", "rsi_mean_reversion")
    assert (storage.log_trade_outcome, storage.mark_settled) == before
