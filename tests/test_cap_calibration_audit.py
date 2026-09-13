"""audit_cap_calibration() -- data-driven check for a *_strength_cap_pct/
*_strength_cap config field (2026-09-13), closing a gap point 19 of the
strategy-validation pipeline had long flagged: rrr_scoring_ceiling_check()
catches RRR-vs-threshold unreachability, but nothing ever automatically
checked whether a strength cap itself is well-calibrated against real
achievable data -- the exact bug improvements.txt item 97 found (by hand,
once) for ma_crossover's original cap=2.0 vs a real p90 of only 0.52%.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce.
"""
from swingtrade.scoring import audit_cap_calibration


def test_empty_values_returns_none_fields():
    result = audit_cap_calibration([], cap=5.0)
    assert result["n"] == 0
    assert result["median"] is None
    assert result["likely_too_high"] is None
    assert result["likely_too_low"] is None


def test_reproduces_the_real_ma_crossover_v97_incident_shape():
    # Real numbers from the actual incident (improvements.txt item 97):
    # cap=2.0, real p90=0.52 -- ma_crossover's Buy/Strong Buy tier was
    # structurally unreachable because even a p90 "strong" real crossover
    # only used 0.52/2.0 = 26% of this scoring component's points.
    values = [0.14] * 50 + [0.3] * 20 + [0.52] * 10 + [0.8, 1.0, 1.24]
    result = audit_cap_calibration(values, cap=2.0, target_percentile=90.0)
    assert result["p90"] == 0.52
    assert result["pct_of_cap_used_at_p90"] == 26.0
    assert result["likely_too_high"] is True
    assert result["likely_too_low"] is False


def test_reproduces_the_real_ma_crossover_post_fix_calibration():
    # The ACTUAL fix: cap lowered to 0.5 (just under real p90=0.52) --
    # pct_of_cap_used_at_p90 should now be near/at 100%, no longer flagged
    # as too high.
    values = [0.14] * 50 + [0.3] * 20 + [0.52] * 10 + [0.8, 1.0, 1.24]
    result = audit_cap_calibration(values, cap=0.5, target_percentile=90.0)
    assert result["pct_of_cap_used_at_p90"] == 100.0
    assert result["likely_too_high"] is False


def test_cap_set_too_low_flags_saturation():
    # Cap of 1.0 against values mostly at/above 1.0 -- most real
    # observations clip to the SAME max score, losing differentiation.
    values = [1.5, 2.0, 1.2, 3.0, 1.1, 0.9, 1.8]
    result = audit_cap_calibration(values, cap=1.0)
    assert result["pct_values_saturating_cap"] == round(6 / 7 * 100, 2)
    assert result["likely_too_low"] is True


def test_well_calibrated_cap_flags_neither():
    # A cap where a typical strong signal (p90) uses most of the range,
    # and saturation is rare -- the healthy case, matching what item 97's
    # fix was aiming for.
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    result = audit_cap_calibration(values, cap=1.0)
    assert result["likely_too_high"] is False
    assert result["likely_too_low"] is False


def test_median_p99_max_are_real_percentiles_not_placeholders():
    values = list(range(1, 101))  # 1..100
    result = audit_cap_calibration(values, cap=100.0)
    assert result["median"] == 50.5
    assert result["p99"] == 99.01
    assert result["max"] == 100.0
    assert result["n"] == 100
