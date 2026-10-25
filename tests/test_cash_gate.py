"""swingtrade.backtest.cash_gate + audit_vs_cash.py CLI -- absolute-return (vs cash) evaluation. Offline, deterministic."""
import math

import numpy as np
import pandas as pd
import pytest

import audit_vs_cash
import swingtrade
from swingtrade.backtest import cash_gate as cg


def _dates(n, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def _series(values, start="2020-01-02"):
    return pd.Series(values, index=_dates(len(values), start), dtype=float)


def _smooth_strategy(n=1008, drift=0.0004, noise=0.0002, seed=3):
    rng = np.random.default_rng(seed)
    return _series(drift + rng.normal(0, noise, n))


# ---------------------------------------------------------------- conversions / alignment

def test_irx_to_daily_returns_converts_percent_yield():
    out = cg.irx_to_daily_returns(_series([5.04, 5.04]))
    assert out.iloc[0] == pytest.approx(0.0504 / 252) and out.iloc[0] == pytest.approx(0.0002)


def test_nav_to_returns():
    r = cg.nav_to_returns(_series([100, 110, 99]))
    assert list(r.round(6)) == [0.1, -0.1]


def test_excess_forward_fills_cash_and_drops_dates_before_first_cash_obs():
    strat = _series([0.01] * 5)
    cash = pd.Series([0.001, 0.002], index=[strat.index[2], strat.index[4]])
    ex = cg.excess_returns(strat, cash)
    assert list(ex.index) == list(strat.index[2:])            # first two dates dropped, not assumed zero-cash
    assert list(ex.round(6)) == [0.009, 0.009, 0.008]          # day 3 ffilled from day 2's 0.001, day 5 uses 0.002


# ---------------------------------------------------------------- Sharpe / SE

def test_annualized_sharpe_known_value_and_degenerate_inputs():
    x = np.array([0.01, -0.01] * 50) + 0.001                   # mean 0.001, sd (ddof=1) known
    expected = x.mean() / x.std(ddof=1) * math.sqrt(252)
    assert cg.annualized_sharpe(x) == pytest.approx(expected)
    assert cg.annualized_sharpe([0.01] * 10) is None            # zero variance
    assert cg.annualized_sharpe([0.01]) is None
    assert cg.annualized_sharpe([np.nan, np.nan]) is None


def test_sharpe_standard_error_and_years_needed_match_hand_calculation():
    assert cg.sharpe_se(1.5, 5.0) == pytest.approx(math.sqrt((1 + 0.5 * 2.25) / 5))      # 0.652
    assert cg.min_years_for_z(1.5, 1.65) == pytest.approx(1.65 ** 2 * 2.125 / 2.25)       # ~2.57
    assert cg.min_years_for_z(1.5, 3.1) == pytest.approx(9.08, abs=0.02)                   # Bonferroni-ish bar


# ---------------------------------------------------------------- drawdown / CAGR / Calmar

def test_max_drawdown_known_curve():
    r = _series([0.10, -0.50, 0.20])                           # 1.10 -> 0.55 (-50%) -> 0.66
    assert cg.max_drawdown_pct(r) == pytest.approx(50.0)


def test_max_drawdown_counts_initial_capital_as_a_peak():
    assert cg.max_drawdown_pct(_series([-0.10, 0.05])) == pytest.approx(10.0)
    assert cg.max_drawdown_pct(_series([0.01, 0.01])) == 0.0
    assert cg.max_drawdown_pct(pd.Series(dtype=float)) is None


def test_cagr_and_calmar():
    r = _series([0.001] * 252)
    assert cg.cagr_pct(r) == pytest.approx((1.001 ** 252 - 1) * 100)
    assert cg.calmar(r) is None                                 # no drawdown -> undefined, not infinity
    r2 = _series([0.10, -0.50, 0.20] + [0.0] * 249)
    assert cg.calmar(r2) == pytest.approx(cg.cagr_pct(r2) / 50.0)


def test_tail_metrics():
    r = _series([0.01] * 95 + [-0.05, -0.04, -0.03, -0.02, -0.01])
    t = cg.tail_metrics(r)
    assert t["worst_day_pct"] == pytest.approx(-5.0)
    assert t["cvar_95_daily_pct"] < -2.0 and t["skew"] < 0
    assert cg.tail_metrics(pd.Series(dtype=float))["worst_day_pct"] is None


# ---------------------------------------------------------------- stationary bootstrap

def test_stationary_bootstrap_indices_shape_range_and_block_structure():
    rng = np.random.default_rng(0)
    idx = cg.stationary_bootstrap_indices(500, 40, mean_block=10, rng=rng)
    assert idx.shape == (40, 500) and idx.min() >= 0 and idx.max() < 500
    continued = ((idx[:, 1:] - idx[:, :-1]) % 500 == 1).mean()    # share of steps that continue a block
    assert 0.85 < continued < 0.95                                # ~ 1 - 1/mean_block = 0.9
    iid = cg.stationary_bootstrap_indices(500, 40, mean_block=1, rng=np.random.default_rng(0))
    assert ((iid[:, 1:] - iid[:, :-1]) % 500 == 1).mean() < 0.05  # mean_block=1 degenerates to iid resampling


def test_bootstrap_sharpe_detects_real_edge_and_is_reproducible():
    ex = _smooth_strategy() - 0.0001
    a = cg.bootstrap_sharpe(ex, n_boot=500, seed=1)
    b = cg.bootstrap_sharpe(ex, n_boot=500, seed=1)
    assert a == b                                               # same seed -> identical
    assert a["ci_low"] > 0 and a["p_le_zero"] == 0.0 and a["ci_low"] < a["sharpe"] < a["ci_high"]


def test_bootstrap_sharpe_zero_edge_ci_straddles_zero():
    rng = np.random.default_rng(11)
    noise = _series(rng.normal(0, 0.01, 1000))
    r = cg.bootstrap_sharpe(noise - noise.mean(), n_boot=500, seed=2)   # mean exactly 0
    assert r["ci_low"] < 0 < r["ci_high"] and 0.2 < r["p_le_zero"] < 0.8


def test_bootstrap_sharpe_needs_enough_data():
    assert cg.bootstrap_sharpe(_series([0.001] * 29)) is None


# ---------------------------------------------------------------- audit + verdict

def _audit(n=1008, **kw):
    cash = pd.Series(0.00015, index=_dates(n))
    return cg.audit_vs_cash(_smooth_strategy(n, **kw), cash, n_boot=500, seed=5)


def test_audit_vs_cash_fields_on_smooth_strategy():
    a = _audit()
    assert a["years"] == pytest.approx(4.0) and a["n_days"] == 1008
    assert a["cagr_pct"] == pytest.approx(((1.0004) ** 252 - 1) * 100, rel=0.05)
    assert a["annualized_excess_pct"] == pytest.approx((0.0004 - 0.00015) * 252 * 100, rel=0.05)
    assert a["sharpe_excess"] > 5 and a["max_drawdown_pct"] < 1
    assert a["bootstrap"]["ci_low"] > 0 and a["tail"]["worst_day_pct"] > -1


def test_audit_returns_none_without_overlap_or_data():
    assert cg.audit_vs_cash(_series([0.001] * 10), pd.Series(0.0001, index=_dates(10))) is None
    far_future_cash = pd.Series(0.0001, index=_dates(100, "2030-01-01"))
    assert cg.audit_vs_cash(_smooth_strategy(100), far_future_cash) is None


def test_verdict_passes_a_smooth_high_sharpe_sleeve():
    v = cg.cash_gate_verdict(_audit(), n_trials=46)
    assert v["passed"] is True, v


@pytest.mark.parametrize("change,fragment", [
    ({"years": 2.0}, "years of data"),
    ({"sharpe_excess": 1.0}, "Sharpe 1.00 < 1.5"),
    ({"cagr_pct": 6.0}, "CAGR 6.00%"),
    ({"max_drawdown_pct": 7.5}, "max drawdown 7.50%"),
])
def test_verdict_names_the_first_failing_condition(change, fragment):
    v = cg.cash_gate_verdict({**_audit(), **change}, n_trials=46)
    assert v["passed"] is False and fragment in v["reason"]


def test_verdict_bonferroni_bar_scales_with_trial_count():
    a = _audit()
    a["bootstrap"] = {**a["bootstrap"], "p_le_zero": 0.002, "n_boot": 5000}
    assert cg.cash_gate_verdict(a, n_trials=1)["passed"] is True          # 0.002 < 0.05
    v = cg.cash_gate_verdict(a, n_trials=46)                               # bar 0.05/46 = 0.00109
    assert v["passed"] is False and "Bonferroni bar" in v["reason"]


def test_verdict_flags_bootstrap_resolution_limit():
    a = _audit()
    a["bootstrap"] = {**a["bootstrap"], "p_le_zero": 0.0, "n_boot": 500}
    a["bootstrap"]["p_le_zero"] = 0.002                                    # one resample in 500 is <= 0
    v = cg.cash_gate_verdict(a, n_trials=1000)                             # bar 0.00005 < 1/500
    assert v["passed"] is False and "resolution" in v["reason"]


def test_verdict_handles_missing_audit():
    assert cg.cash_gate_verdict(None)["passed"] is False


def test_a_fat_tailed_carry_sleeve_fails_on_drawdown_despite_sharpe():
    """Short-vol/carry shape: steady small gains, rare crash. Looks great until the tail hits."""
    r = _smooth_strategy(1008, drift=0.0005, noise=0.0001)
    r.iloc[500] = -0.08
    a = cg.audit_vs_cash(r, pd.Series(0.00015, index=r.index), n_boot=500, seed=5)
    v = cg.cash_gate_verdict(a, n_trials=1)
    assert a["max_drawdown_pct"] > 7 and a["tail"]["skew"] < -5
    assert v["passed"] is False and "drawdown" in v["reason"]


def test_exports_are_reachable_from_package_root():
    assert swingtrade.audit_vs_cash is cg.audit_vs_cash and swingtrade.cash_gate_verdict is cg.cash_gate_verdict


# ---------------------------------------------------------------- CLI

def test_registry_count_dedupes_and_ignores_blank_lines(tmp_path):
    p = tmp_path / "reg.txt"
    p.write_text("a\nb\n\na\n  \nc\n")
    assert audit_vs_cash.registry_count(p) == 3
    assert audit_vs_cash.registry_count(tmp_path / "missing.txt") == 0


def test_cli_end_to_end_with_stubbed_cash_and_no_registry_write(tmp_path, monkeypatch, capsys):
    strat = _smooth_strategy()
    csv = tmp_path / "sleeve.csv"
    strat.rename("return").rename_axis("date").to_csv(csv)
    monkeypatch.setattr(audit_vs_cash, "fetch_cash_returns", lambda kind, s, e: pd.Series(0.00015, index=strat.index))
    before = audit_vs_cash.REGISTRY_PATH.read_text()
    rc = audit_vs_cash.main(["--returns-csv", str(csv), "--strategy", "demo", "--n-boot", "500"])
    out = capsys.readouterr().out
    assert rc == 0 and "CASH GATE: PASS" in out and "Bonferroni bar: alpha 0.05 / 46 trials" in out
    assert audit_vs_cash.REGISTRY_PATH.read_text() == before      # read-only without --register


def test_cli_nav_input_and_failing_gate_returns_1(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(9)
    nav = (1 + _series(rng.normal(0.0, 0.01, 800))).cumprod() * 100
    csv = tmp_path / "nav.csv"
    nav.rename("nav").rename_axis("date").to_csv(csv)
    monkeypatch.setattr(audit_vs_cash, "fetch_cash_returns", lambda kind, s, e: pd.Series(0.0001, index=nav.index))
    rc = audit_vs_cash.main(["--nav-csv", str(csv), "--column", "nav", "--n-boot", "500"])
    assert rc == 1 and "CASH GATE: FAIL" in capsys.readouterr().out


def test_bootstrap_of_a_constant_series_does_not_return_astronomical_sharpes():
    assert cg.bootstrap_sharpe(_series([0.001] * 100), n_boot=200) is None
