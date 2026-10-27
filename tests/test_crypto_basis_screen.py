"""research/crypto_basis/screen.py -- funding yield, fee, underwater and kill-rule arithmetic (docs/CRYPTO_BASIS_SPEC.md). Offline."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "research" / "crypto_basis"))
import screen as S  # noqa: E402


def _funding(rates, start="2022-01-01"):
    idx = pd.date_range(start, periods=len(rates), freq="8h", tz="UTC")
    return pd.Series(rates, index=idx, dtype=float)


def _cash(pct, start="2021-01-01", end="2026-12-31"):
    return pd.Series(pct, index=pd.date_range(start, end, freq="D"), dtype=float)


def test_constant_funding_annualizes_to_rate_x3_x365():
    f = _funding([0.0001] * (3 * 365))                         # exactly calendar 2022
    t = S.yearly_table(f, _cash(4.0), years=[2022])
    assert t.loc[2022, "gross_pct"] == pytest.approx(0.0001 * 3 * 365 * 100)          # 10.95%
    assert t.loc[2022, "days"] == pytest.approx(365.0)
    assert t.loc[2022, "negative_interval_share_pct"] == 0.0


def test_partial_year_is_annualized_by_days_elapsed():
    f = _funding([0.0001] * (3 * 100), start="2026-01-01")      # 100 days of 2026
    t = S.yearly_table(f, _cash(4.0), years=[2026])
    assert t.loc[2026, "days"] == pytest.approx(100.0)
    assert t.loc[2026, "gross_pct"] == pytest.approx(0.0001 * 3 * 365 * 100)           # same annualized rate


def test_fee_models_and_spread_over_cash():
    f = _funding([0.0001] * (3 * 365))
    row = S.yearly_table(f, _cash(4.0), years=[2022]).loc[2022]
    gross = 10.95
    assert row[f"net[{S.PRIMARY}]"] == pytest.approx(gross - 0.20)                       # 4 legs x 0.05%, 1 RT
    assert row["net[maker_0.02%_1RT]"] == pytest.approx(gross - 0.08)
    assert row["net[taker_12RT]"] == pytest.approx(gross - 2.40)
    assert row["net[spot0.10%+perp0.05%_1RT]"] == pytest.approx(gross - 0.30)
    assert row[f"spread[{S.PRIMARY}]"] == pytest.approx(gross - 0.20 - 4.0)
    assert row["spread_on_capital_primary"] == pytest.approx((gross - 0.20 - 4.0) / 1.2)


def test_negative_funding_is_paid_not_ignored():
    f = _funding([0.0001] * (3 * 182) + [-0.0002] * (3 * 183))
    t = S.yearly_table(f, _cash(0.0), years=[2022])
    expected = (0.0001 * 3 * 182 - 0.0002 * 3 * 183) * 100
    assert t.loc[2022, "gross_pct"] == pytest.approx(expected)
    assert t.loc[2022, "gross_pct"] < 0
    assert t.loc[2022, "negative_interval_share_pct"] == pytest.approx(183 / 365 * 100, abs=0.1)


def test_years_without_data_are_skipped():
    f = _funding([0.0001] * 30, start="2023-03-01")
    assert list(S.yearly_table(f, _cash(4.0), years=range(2021, 2026)).index) == [2023]


def test_underwater_never_below_peak():
    u = S.underwater(_funding([0.0001] * 600))
    assert u["max_drawdown_pp"] == 0.0 and u["longest_underwater_days"] == 0.0 and u["underwater_at_end"] is False


def test_underwater_measures_longest_spell_and_recovery():
    # +1 interval gain, then 400 days losing 0.001/interval... then gains recover fully
    up = [0.01]; down = [-0.0001] * (3 * 400); recover = [0.0002] * (3 * 400)
    u = S.underwater(_funding(up + down + recover))
    assert u["max_drawdown_pp"] == pytest.approx(0.0001 * 3 * 400 * 100, rel=1e-6)       # 12 pp
    assert 400 < u["longest_underwater_days"] < 640                                       # down 400d + part of recovery
    assert u["underwater_at_end"] is False


def test_underwater_at_end_is_flagged_and_counted():
    u = S.underwater(_funding([0.01] + [-0.00001] * (3 * 500)))
    assert u["underwater_at_end"] is True and u["longest_underwater_days"] == pytest.approx(500, abs=2)


def test_underwater_ignores_data_before_since():
    f = _funding([-0.01] * (3 * 150) + [0.0001] * (3 * 300), start="2021-06-01")   # losses end Oct 2021; gains run into 2022
    assert S.underwater(f, since="2022-01-01")["max_drawdown_pp"] == 0.0


def _tables(spread_by_asset):
    out = {}
    for asset, spread in spread_by_asset.items():
        gross = spread + 4.0 + 0.20
        out[asset] = S.yearly_table(_funding([gross / 100 / (3 * 365)] * (3 * 365)), _cash(4.0), years=[2022])
    return out


def test_kill_rule_1_threshold_is_four_points_averaged_across_assets():
    ok = S.evaluate_kill_rules(_tables({"BTC": 4.5, "ETH": 4.5}), {"BTC": {"longest_underwater_days": 10}, "ETH": {"longest_underwater_days": 10}})
    assert ok["rule1_fail"] is False and ok["avg_net_spread_pct"] == pytest.approx(4.5)
    mixed = S.evaluate_kill_rules(_tables({"BTC": 6.0, "ETH": 1.0}), {"BTC": {"longest_underwater_days": 0}, "ETH": {"longest_underwater_days": 0}})
    assert mixed["avg_net_spread_pct"] == pytest.approx(3.5) and mixed["rule1_fail"] is True and mixed["dropped"] is True


def test_kill_rule_2_triggers_on_either_asset_over_365_days():
    t = _tables({"BTC": 9.0, "ETH": 9.0})
    assert S.evaluate_kill_rules(t, {"BTC": {"longest_underwater_days": 365}, "ETH": {"longest_underwater_days": 100}})["rule2_fail"] is False
    v = S.evaluate_kill_rules(t, {"BTC": {"longest_underwater_days": 100}, "ETH": {"longest_underwater_days": 366}})
    assert v["rule2_fail"] is True and v["dropped"] is True


def test_kill_rule_1_ignores_years_before_2022():
    t21 = S.yearly_table(_funding([0.0005] * (3 * 365), start="2021-01-01"), _cash(0.05), years=[2021])   # huge 2021 yield
    t22 = S.yearly_table(_funding([0.00001] * (3 * 365), start="2022-01-01"), _cash(4.0), years=[2022])  # tiny 2022 yield
    both = pd.concat([t21, t22])
    v = S.evaluate_kill_rules({"BTC": both, "ETH": both}, {"BTC": {"longest_underwater_days": 0}, "ETH": {"longest_underwater_days": 0}})
    assert v["rule1_fail"] is True
