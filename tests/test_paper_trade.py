from datetime import datetime, timedelta, timezone

import pytest

import broker
import paper_trade as pt

CFG = {"strategies": {"value_rank", "accruals_rank"}, "max_positions": 3, "risk_usd": 100.0,
       "max_notional": 2500.0, "max_exposure_pct": 0.5, "entry_ttl_days": 3, "max_hold_days": 21}


def sig(ticker, strategy="value_rank", buy=50.0, stop=48.0, target=56.0, score=70, **kw):
    return {"ticker": ticker, "strategy": strategy, "signal_date": "2026-10-22", "buy_price": buy,
            "stop_loss": stop, "sell_price": target, "trade_score": score, **kw}


def plan(signals, held=(), open_symbols=(), ids=(), equity=100_000, used=0, n=0, cfg=CFG):
    return pt.plan_orders(signals, set(held), set(open_symbols), set(ids), equity, used, n, cfg)


def test_client_order_id_roundtrip():
    coid = pt.make_client_order_id("value_rank", "AAPL", "2026-10-22")
    assert coid == "ts.value_rank.AAPL.2026-10-22" and len(coid) <= 48
    assert pt.parse_client_order_id(coid) == ("value_rank", "AAPL", "2026-10-22")
    assert pt.parse_client_order_id("manual-order") is None and pt.parse_client_order_id(None) is None


def test_sizing_is_risk_based_and_notional_capped():
    assert pt.size_position(50, 48, 100, 2500) == 50          # $100 risk / $2 per share
    assert pt.size_position(50, 49.9, 100, 2500) == 50        # notional cap: 2500/50
    assert pt.size_position(50, 50, 100, 2500) == 0 and pt.size_position(5000, 4990, 100, 2500) == 0


def test_plan_places_valid_signal_with_rounded_prices():
    orders, skipped = plan([sig("AAA", buy=50.004, stop=48.0, target=56.0)])
    assert orders[0]["qty"] == 49 and orders[0]["limit_price"] == 50.0 and not skipped


@pytest.mark.parametrize("kwargs,reason", [
    ({"signals": [sig("AAA", strategy="ma_crossover")]}, "strategy not enabled"),
    ({"signals": [sig("AAA.TO")]}, "non-USD"),
    ({"signals": [sig("BRK-B")]}, "non-USD"),
    ({"signals": [sig("AAA")], "held": ["AAA"]}, "already open"),
    ({"signals": [sig("AAA")], "ids": ["ts.value_rank.AAA.2026-10-22"]}, "already ordered"),
    ({"signals": [sig("AAA", stop=51)]}, "invalid price"),
    ({"signals": [sig("AAA")], "n": 3}, "max positions"),
    ({"signals": [sig("AAA")], "used": 49_000, "equity": 100_000}, "exposure cap"),
])
def test_plan_skips(kwargs, reason):
    orders, skipped = plan(**kwargs)
    assert not orders and reason in skipped[0][1]


def test_plan_prefers_higher_score_when_slots_limited():
    orders, _ = plan([sig("LOW", score=10), sig("HIGH", score=90)], n=2)
    assert [o["symbol"] for o in orders] == ["HIGH"]


def test_broker_refuses_non_paper_host():
    with pytest.raises(broker.BrokerError):
        broker.AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")
    with pytest.raises(broker.BrokerError):
        broker.AlpacaPaperBroker("k", "s", base_url="https://paper-api.alpaca.markets.evil.com")
    assert broker.AlpacaPaperBroker("k", "s").base_url == broker.PAPER_BASE_URL


def test_broker_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    monkeypatch.setattr(broker, "load_dotenv", lambda: None, raising=False)
    with pytest.raises(broker.BrokerError):
        broker.AlpacaPaperBroker()


class FakeBroker:
    def __init__(self):
        self.cancelled, self.closed = [], []

    def cancel_order(self, oid):
        self.cancelled.append(oid)

    def close_position(self, sym):
        self.closed.append(sym)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_housekeeping_cancels_stale_entries_and_closes_old_positions():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    orders = [
        {"id": "stale", "client_order_id": "ts.value_rank.AAA.2026-10-22", "status": "new",
         "created_at": iso(now - timedelta(days=5)), "symbol": "AAA"},
        {"id": "fresh", "client_order_id": "ts.value_rank.BBB.2026-10-29", "status": "new",
         "created_at": iso(now - timedelta(days=1)), "symbol": "BBB"},
        {"id": "old", "client_order_id": "ts.value_rank.CCC.2026-09-01", "status": "filled", "symbol": "CCC",
         "created_at": iso(now - timedelta(days=50)), "filled_at": iso(now - timedelta(days=40)),
         "legs": [{"id": "leg1", "status": "new"}, {"id": "leg2", "status": "held"}]},
        {"id": "manual", "client_order_id": "mine", "status": "new", "created_at": iso(now - timedelta(days=9)), "symbol": "ZZZ"},
    ]
    fb = FakeBroker()
    cancelled, closed = pt.housekeeping(fb, orders, [{"symbol": "CCC"}], CFG, now)
    assert (cancelled, closed) == (1, 1)
    assert fb.cancelled == ["stale", "leg1", "leg2"] and fb.closed == ["CCC"]


def test_reconcile_confirms_only_unconfirmed_filled_orders(monkeypatch):
    calls = []

    class Col:
        def find_one(self, q):
            return {"confirmed_filled": q["ticker"] == "DONE"}

    class DB:
        def __getitem__(self, name):
            return Col()

    monkeypatch.setattr(pt.storage, "get_db", lambda: DB())
    monkeypatch.setattr(pt.storage, "confirm_fill", lambda *a, **k: calls.append(("confirm", a, k)))
    monkeypatch.setattr(pt.storage, "record_user_decision", lambda *a, **k: calls.append(("decision", a)))
    import confirm_fill
    monkeypatch.setattr(confirm_fill, "resettle_if_already_settled", lambda *a: calls.append(("resettle", a)))
    orders = [
        {"client_order_id": "ts.value_rank.NEW.2026-10-22", "status": "filled", "filled_avg_price": "12.34"},
        {"client_order_id": "ts.value_rank.DONE.2026-10-22", "status": "filled", "filled_avg_price": "9.0"},
        {"client_order_id": "ts.value_rank.OPEN.2026-10-22", "status": "new", "filled_avg_price": None},
    ]
    assert pt.reconcile_fills(orders) == 1
    assert calls[0][0] == "confirm" and calls[0][2] == {"fill_price": 12.34}
