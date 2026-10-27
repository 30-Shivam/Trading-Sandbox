import pytest

import rebalance as rb

T = {"SPY": 0.70, "QQQ": 0.15}


def plan(equity=100_000, cash=100_000, holdings=None, targets=T, reserve=0.15, band=0.03):
    return rb.plan_rebalance(equity, cash, holdings or {}, targets, reserve, band)


def test_initial_build_buys_targets_and_keeps_reserve():
    trades = plan()
    assert {(t["symbol"], t["side"]) for t in trades} == {("SPY", "buy"), ("QQQ", "buy")}
    assert sum(t["notional"] for t in trades) == pytest.approx(85_000)


def test_no_trade_inside_drift_band():
    assert plan(holdings={"SPY": 71_000, "QQQ": 14_000}, cash=15_000) == []


def test_rebalances_drifted_sleeve_sells_first_and_buys_with_proceeds():
    trades = plan(holdings={"SPY": 80_000, "QQQ": 6_000}, cash=14_000)
    assert trades[0]["side"] == "sell" and trades[0]["symbol"] == "SPY"
    assert trades[0]["notional"] == pytest.approx(10_000)
    assert trades[1]["symbol"] == "QQQ" and trades[1]["side"] == "buy"


def test_never_spends_into_satellite_reserve():
    # only 5k cash and a 15k reserve requirement -> nothing may be bought
    assert plan(holdings={"SPY": 70_000, "QQQ": 0}, cash=5_000) == []


def test_rejects_targets_over_100_percent():
    with pytest.raises(ValueError):
        plan(targets={"SPY": 0.9, "QQQ": 0.2})


def test_parse_targets_and_defaults(monkeypatch):
    assert rb.parse_targets("spy:0.6, qqq:0.1") == {"SPY": 0.6, "QQQ": 0.1}
    monkeypatch.delenv("CORE_TARGETS", raising=False)
    assert rb.core_symbols() == {"QQQ", "SPY", "IBIT", "GLD"}


def test_default_policy_sums_to_one_and_builds_from_cash():
    assert sum(rb.DEFAULT_TARGETS.values()) == pytest.approx(1.0)
    trades = plan(targets=rb.DEFAULT_TARGETS, reserve=0.0)
    assert {t["symbol"]: t["notional"] for t in trades} == pytest.approx(
        {"QQQ": 60_000, "SPY": 20_000, "IBIT": 10_000, "GLD": 10_000})
    assert all(t["side"] == "buy" for t in trades)


def test_default_policy_moves_off_old_85_15_by_selling_spy_first():
    trades = plan(targets=rb.DEFAULT_TARGETS, reserve=0.0, holdings={"SPY": 85_000, "QQQ": 15_000}, cash=0)
    assert trades[0] == {"symbol": "SPY", "side": "sell", "notional": 65_000.0, "reason": trades[0]["reason"]}
    assert {t["symbol"]: t["notional"] for t in trades[1:]} == pytest.approx({"QQQ": 45_000, "IBIT": 10_000, "GLD": 10_000})


def test_min_trade_filter():
    assert plan(holdings={"SPY": 69_980, "QQQ": 14_990}, cash=15_030, band=0.0) == []
