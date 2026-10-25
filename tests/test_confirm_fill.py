"""confirm_fill.py -- CLI dispatch and resettle_if_already_settled(), fully
offline. Every storage.* call and settle_trades.settle_one are monkeypatched
(same convention as test_settle_one_entry_fill_gate.py), so no Mongo or
yfinance traffic can happen.
"""
import sys
from unittest.mock import MagicMock

import pytest

import confirm_fill
import storage


@pytest.fixture
def calls(monkeypatch):
    """Record every storage write the CLI makes; ensure_indexes succeeds."""
    log = []
    monkeypatch.setattr(storage, "ensure_indexes", lambda: log.append(("ensure_indexes",)))
    monkeypatch.setattr(storage, "confirm_fill", lambda *a, **k: log.append(("confirm_fill", a, k)))
    monkeypatch.setattr(storage, "unconfirm_fill", lambda *a, **k: log.append(("unconfirm_fill", a, k)))
    monkeypatch.setattr(storage, "record_user_decision", lambda *a, **k: log.append(("decision", a, k)))
    monkeypatch.setattr(confirm_fill, "resettle_if_already_settled", lambda *a: log.append(("resettle", a)))
    return log


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["confirm_fill.py", *argv])
    confirm_fill.main()


def _names(log):
    return [entry[0] for entry in log]


# ---------------------------------------------------------------- resettle_if_already_settled

def _fake_db(signal):
    collection = MagicMock()
    collection.find_one.return_value = signal
    db = MagicMock()
    db.__getitem__.return_value = collection
    return db, collection


def test_resettle_reruns_settle_one_when_signal_already_settled(monkeypatch, capsys):
    signal = {"ticker": "INTC", "settled": True}
    db, collection = _fake_db(signal)
    monkeypatch.setattr(storage, "get_db", lambda: db)
    settled = []
    monkeypatch.setattr(confirm_fill, "settle_one", lambda s: settled.append(s) or ("WIN (target, +5.00%)", None))
    confirm_fill.resettle_if_already_settled("INTC", "2026-07-24", "rsi")
    db.__getitem__.assert_called_with("Trade_Signals")
    collection.find_one.assert_called_once_with({"ticker": "INTC", "signal_date": "2026-07-24", "strategy": "rsi"})
    assert settled == [signal]
    assert "re-settled: WIN (target, +5.00%)" in capsys.readouterr().out


@pytest.mark.parametrize("signal", [None, {"ticker": "INTC"}, {"ticker": "INTC", "settled": False}])
def test_resettle_noop_for_missing_or_unsettled_signal(monkeypatch, signal):
    db, _ = _fake_db(signal)
    monkeypatch.setattr(storage, "get_db", lambda: db)
    monkeypatch.setattr(confirm_fill, "settle_one", lambda s: pytest.fail("must not re-settle"))
    confirm_fill.resettle_if_already_settled("INTC", "2026-07-24", "rsi")


def test_resettle_propagates_settle_failure(monkeypatch):
    db, _ = _fake_db({"settled": True})
    monkeypatch.setattr(storage, "get_db", lambda: db)

    def boom(s):
        raise TimeoutError("yahoo timeout")

    monkeypatch.setattr(confirm_fill, "settle_one", boom)
    with pytest.raises(TimeoutError):
        confirm_fill.resettle_if_already_settled("INTC", "2026-07-24", "rsi")


def test_resettle_propagates_mongo_failure(monkeypatch):
    def no_mongo():
        raise storage.MongoNotConfigured("MONGODB_URI missing")

    monkeypatch.setattr(storage, "get_db", no_mongo)
    with pytest.raises(storage.MongoNotConfigured):
        confirm_fill.resettle_if_already_settled("INTC", "2026-07-24", "rsi")


# ---------------------------------------------------------------- main(): confirm / undo / pass

def test_confirm_with_price_records_fill_and_decision_then_resettles(monkeypatch, calls, capsys):
    _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24", "--price", "89.75")
    assert _names(calls) == ["ensure_indexes", "confirm_fill", "decision", "resettle"]
    assert calls[1][1] == ("INTC", "2026-07-24", "rsi") and calls[1][2] == {"fill_price": 89.75}
    assert calls[2][1] == ("INTC", "2026-07-24", "rsi", "acted_on")
    assert calls[3][1] == ("INTC", "2026-07-24", "rsi")
    assert "at $89.75" in capsys.readouterr().out


def test_confirm_without_price_uses_logged_buy_price(monkeypatch, calls, capsys):
    _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24", "--strategy", "breakout")
    assert calls[1][1] == ("INTC", "2026-07-24", "breakout") and calls[1][2] == {"fill_price": None}
    assert "using the logged buy_price" in capsys.readouterr().out


def test_undo_unconfirms_and_resettles_without_recording_a_decision(monkeypatch, calls, capsys):
    _run(monkeypatch, "--undo", "--ticker", "INTC", "--date", "2026-07-24")
    assert _names(calls) == ["ensure_indexes", "unconfirm_fill", "resettle"]
    assert "Unconfirmed INTC" in capsys.readouterr().out


def test_pass_records_decision_only_and_never_touches_fill_or_settlement(monkeypatch, calls, capsys):
    _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24", "--pass", "too extended")
    assert _names(calls) == ["ensure_indexes", "decision"]
    assert calls[1][1] == ("INTC", "2026-07-24", "rsi", "passed") and calls[1][2] == {"reason": "too extended"}
    assert "Recorded PASS" in capsys.readouterr().out


def test_undo_takes_precedence_over_pass(monkeypatch, calls):
    _run(monkeypatch, "--undo", "--pass", "x", "--ticker", "INTC", "--date", "2026-07-24")
    assert "unconfirm_fill" in _names(calls) and "decision" not in _names(calls)


def test_ticker_without_date_falls_through_to_listing(monkeypatch, calls, capsys):
    monkeypatch.setattr(storage, "get_signals_pending_confirmation", lambda: [])
    _run(monkeypatch, "--ticker", "INTC")
    assert _names(calls) == ["ensure_indexes"]
    assert "No unconfirmed signals." in capsys.readouterr().out


def test_storage_error_during_confirm_propagates(monkeypatch, calls):
    def boom(*a, **k):
        raise ConnectionError("atlas unreachable")

    monkeypatch.setattr(storage, "confirm_fill", boom)
    with pytest.raises(ConnectionError):
        _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24")
    assert "decision" not in _names(calls)   # no "acted_on" journal entry for a fill that failed


def test_invalid_price_argument_exits_before_any_storage_call(monkeypatch, calls):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24", "--price", "abc")
    assert exc.value.code == 2 and calls == []


def test_mongo_not_configured_exits_1_with_stderr_message(monkeypatch, capsys):
    def no_mongo():
        raise storage.MongoNotConfigured("MONGODB_URI missing")

    monkeypatch.setattr(storage, "ensure_indexes", no_mongo)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--ticker", "INTC", "--date", "2026-07-24")
    assert exc.value.code == 1
    assert "[ERROR] MONGODB_URI missing" in capsys.readouterr().err


# ---------------------------------------------------------------- main(): pending listing

def test_listing_prints_pending_signals_with_strategy_default(monkeypatch, calls, capsys):
    pending = [
        {"ticker": "INTC", "signal_date": "2026-07-24", "strategy": "breakout", "signal": "Strong Buy",
         "buy_price": 89.5, "stop_loss": 80.0, "sell_price": 100.0},
        {"ticker": "AMD", "signal_date": "2026-07-23", "signal": "Buy",   # legacy doc: no strategy key
         "buy_price": 150.0, "stop_loss": 140.0, "sell_price": 170.0},
    ]
    monkeypatch.setattr(storage, "get_signals_pending_confirmation", lambda: pending)
    _run(monkeypatch)
    out = capsys.readouterr().out
    assert "2 unconfirmed signal(s)" in out
    assert "INTC" in out and "breakout" in out and "buy=89.50" in out
    assert "AMD" in out and "rsi" in out
    assert "python confirm_fill.py --ticker" in out


def test_listing_with_malformed_pending_doc_raises_keyerror(monkeypatch, calls):
    """Documents the current contract: a pending doc missing a price field is not
    silently skipped -- the CLI fails loudly."""
    monkeypatch.setattr(storage, "get_signals_pending_confirmation",
                        lambda: [{"ticker": "X", "signal_date": "d", "signal": "Buy"}])
    with pytest.raises(KeyError):
        _run(monkeypatch)
