from datetime import date

import pandas as pd

import swingtrade


def _index():
    idx = pd.date_range("2026-01-01", periods=10)
    return pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx, dtype=float)


def test_excess_is_trade_minus_index_over_same_window():
    trades = [{"entry_date": date(2026, 1, 1), "exit_date": date(2026, 1, 6), "pnl_pct": 8.0},
              {"entry_date": date(2026, 1, 2), "exit_date": date(2026, 1, 3), "pnl_pct": -1.0}]
    r = swingtrade.audit_vs_index(trades, _index())
    # index: 100->105 = 5%, 101->102 = 0.990%
    assert r["n_trades"] == 2 and r["mean_trade_pnl_pct"] == 3.5
    assert abs(r["mean_index_pnl_pct_same_windows"] - (5.0 + 0.990099) / 2) < 0.01
    assert r["pct_trades_beating_index"] == 50.0


def test_returns_none_without_usable_trades():
    assert swingtrade.audit_vs_index([], _index()) is None
    assert swingtrade.audit_vs_index([{"pnl_pct": 1.0}], _index()) is None
    assert swingtrade.audit_vs_index([{"entry_date": date(2026, 1, 1), "exit_date": date(2026, 1, 2), "pnl_pct": 1}],
                                     pd.Series(dtype=float)) is None


def test_weekend_dates_use_last_available_close():
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])  # Fri, Mon
    close = pd.Series([100.0, 110.0], index=idx)
    t = [{"entry_date": date(2026, 1, 3), "exit_date": date(2026, 1, 5), "pnl_pct": 10.0}]  # Sat -> Mon
    assert swingtrade.audit_vs_index(t, close)["mean_index_pnl_pct_same_windows"] == 10.0


def test_index_gate_requires_positive_excess_and_significance():
    gate = swingtrade.index_gate_verdict
    assert gate({"mean_excess_pct": 0.5, "excess_t_stat_clustered_by_entry_day": 2.0})["passed"]
    assert not gate({"mean_excess_pct": 0.077, "excess_t_stat_clustered_by_entry_day": -0.09})["passed"]
    assert "noise" in gate({"mean_excess_pct": 0.3, "excess_t_stat_clustered_by_entry_day": 1.0})["reason"]
    assert not gate({"mean_excess_pct": -0.1, "excess_t_stat_clustered_by_entry_day": 3.0})["passed"]
    assert not gate(None)["passed"]
