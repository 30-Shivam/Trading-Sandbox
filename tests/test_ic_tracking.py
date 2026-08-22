"""ic_tracking.py -- hand-computable unit tests for the pure rank-
correlation/IR math (rank_ic, windowed_ic_series, information_ratio,
ensemble_weight). fetch_score_pnl_pairs()/methodology_report() are thin
Mongo read wrappers (same shape as this project's other storage
read-paths, e.g. storage/outcomes.py) and are deliberately not covered
here with a live/mocked database -- see test_mongo_indexes.py for this
codebase's own established "skip if MONGODB_URI unavailable" convention
for anything that needs a real connection; the computational risk lives
entirely in the pure functions below.
"""
import math

import ic_tracking as ic


def test_rank_ic_perfect_positive_correlation():
    assert math.isclose(ic.rank_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)


def test_rank_ic_perfect_negative_correlation():
    assert math.isclose(ic.rank_ic([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)


def test_rank_ic_rank_based_not_value_based():
    # Non-linear but monotonic -- Spearman still reads this as perfect (1.0),
    # unlike a Pearson correlation on the raw values.
    assert math.isclose(ic.rank_ic([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]), 1.0)


def test_rank_ic_none_below_minimum_points():
    assert ic.rank_ic([1, 2], [1, 2]) is None


def test_rank_ic_none_on_constant_scores():
    assert ic.rank_ic([5, 5, 5, 5], [1, 2, 3, 4]) is None


def test_rank_ic_none_on_constant_pnls():
    assert ic.rank_ic([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_rank_ic_none_on_mismatched_lengths():
    assert ic.rank_ic([1, 2, 3], [1, 2]) is None


def test_windowed_ic_series_two_windows_second_too_thin():
    # Window 1 (5 records, >= MIN_TRADES_PER_WINDOW=5): perfect positive IC.
    # Window 2 (2 records): below the 5-record floor, must be skipped.
    records = [
        {"signal_date": "2026-01-01", "score": 1, "pnl_pct": 1},
        {"signal_date": "2026-01-02", "score": 2, "pnl_pct": 2},
        {"signal_date": "2026-01-03", "score": 3, "pnl_pct": 3},
        {"signal_date": "2026-01-04", "score": 4, "pnl_pct": 4},
        {"signal_date": "2026-01-05", "score": 5, "pnl_pct": 5},
        {"signal_date": "2026-02-10", "score": 1, "pnl_pct": -1},
        {"signal_date": "2026-02-11", "score": 2, "pnl_pct": -2},
    ]
    windows = ic.windowed_ic_series(records, window_days=30)
    assert len(windows) == 1
    assert math.isclose(windows[0]["ic"], 1.0)
    assert windows[0]["n"] == 5


def test_windowed_ic_series_empty_input():
    assert ic.windowed_ic_series([]) == []


def test_information_ratio_hand_computed():
    # mean=0.3, sample std (ddof=1) = sqrt(((0.2-0.3)^2+(0.4-0.3)^2+(0.3-0.3)^2)/2)
    #                                = sqrt((0.01+0.01+0)/2) = sqrt(0.01) = 0.1
    # ir = 0.3 / 0.1 = 3.0
    ir = ic.information_ratio([0.2, 0.4, 0.3])
    assert ir is not None
    assert math.isclose(ir, 3.0, rel_tol=1e-9)


def test_information_ratio_none_below_two_windows():
    assert ic.information_ratio([0.5]) is None
    assert ic.information_ratio([]) is None


def test_information_ratio_none_on_zero_std():
    assert ic.information_ratio([0.2, 0.2, 0.2]) is None


def test_ensemble_weight_uses_ir_once_trust_floor_met():
    report = {"trust_floor_met": True, "ir": 0.42}
    assert ic.ensemble_weight(report) == 0.42


def test_ensemble_weight_floors_negative_ir_at_zero():
    report = {"trust_floor_met": True, "ir": -0.9}
    assert ic.ensemble_weight(report) == 0.0


def test_ensemble_weight_neutral_prior_below_trust_floor():
    report = {"trust_floor_met": False, "ir": 0.9}
    assert ic.ensemble_weight(report) == 1.0
    assert ic.ensemble_weight(report, neutral_prior=2.0) == 2.0


def test_ensemble_weight_neutral_prior_when_ir_missing():
    report = {"trust_floor_met": True, "ir": None}
    assert ic.ensemble_weight(report) == 1.0


# --- Tier-weighting (2026-08-21, per explicit user request): actionable
# (real Strong Buy/Buy) signals should count more toward IC/trust-floor
# math than research (Watch-only) signals -- see TIER_WEIGHTS,
# _effective_count(), rank_ic()'s `weights` param.

def test_effective_count_equal_weights_equals_raw_count():
    # Kish ESS collapses to the raw count whenever every weight is equal,
    # regardless of the weight's own value.
    assert math.isclose(ic._effective_count([1.0, 1.0, 1.0, 1.0]), 4.0)
    assert math.isclose(ic._effective_count([0.5, 0.5, 0.5, 0.5]), 4.0)


def test_effective_count_hand_computed_mixed_weights():
    # total=2.0, sum(w^2)=1+0.25+0.25=1.5, effective=4/1.5
    assert math.isclose(ic._effective_count([1.0, 0.5, 0.5]), 4 / 1.5)


def test_effective_count_empty_or_nonpositive():
    assert ic._effective_count([]) == 0.0
    assert ic._effective_count([0.0, 0.0]) == 0.0


def test_rank_ic_weighted_zero_weight_points_are_ignored():
    # Points 4/5 (indices 3/4) carry weight 0 -- the weighted correlation
    # should reduce to exactly the first-3-points' own perfect positive
    # correlation, hand-verified: rank(scores)=[1,2,3,4,5],
    # rank(pnls)=[3,4,5,2,1] (pnls=[10,20,30,5,1]); restricting to the
    # weight>0 subset, [1,2,3] vs [3,4,5] is perfectly monotonic.
    scores = [1, 2, 3, 4, 5]
    pnls = [10, 20, 30, 5, 1]
    weights = [1, 1, 1, 0, 0]
    result = ic.rank_ic(scores, pnls, weights=weights)
    assert result is not None
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_rank_ic_weighted_differs_from_unweighted():
    # Unweighted: hand-computed corr(ranks) = -0.8 (see comment below).
    # Weighted (actionable=2.0 on the first two points, research=0.5 on
    # the last two): the modest-negative actionable pair pulls the result
    # toward a LESS extreme negative correlation than the unweighted pool,
    # which is dominated by the more strongly negative research pair --
    # proves weighting actually changes the answer, not a no-op.
    scores = [1, 2, 3, 4]
    pnls = [10, 20, 5, 1]
    unweighted = ic.rank_ic(scores, pnls)
    assert math.isclose(unweighted, -0.8, abs_tol=1e-9)

    weighted = ic.rank_ic(scores, pnls, weights=[2.0, 2.0, 0.5, 0.5])
    assert math.isclose(weighted, -0.49 / 0.89, rel_tol=1e-9)
    assert weighted > unweighted  # less extreme, exactly as motivated above


def test_rank_ic_weighted_none_on_mismatched_weight_length():
    assert ic.rank_ic([1, 2, 3], [1, 2, 3], weights=[1, 1]) is None


def test_rank_ic_weights_none_reproduces_unweighted_behavior():
    scores, pnls = [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]
    assert ic.rank_ic(scores, pnls) == ic.rank_ic(scores, pnls, weights=None)


def test_windowed_ic_series_low_weight_window_excluded_despite_raw_count():
    # 5 raw records (would have cleared the old unweighted
    # MIN_TRADES_PER_WINDOW=5 floor) but heavily skewed toward low-weight
    # research-tier signals: effective_n = 1.4^2/1.04 ~= 1.88, well below
    # 5 -- the window must now be skipped entirely.
    records = [
        {"signal_date": f"2026-01-0{i}", "score": i, "pnl_pct": i, "weight": w}
        for i, w in zip(range(1, 6), [1.0, 0.1, 0.1, 0.1, 0.1])
    ]
    assert ic.windowed_ic_series(records, window_days=30) == []


def test_windowed_ic_series_weighted_full_weight_matches_unweighted():
    # Same fixture as test_windowed_ic_series_two_windows_second_too_thin,
    # with explicit weight=1.0 on every record -- must reproduce the exact
    # same result as the unweighted case, plus an effective_n field.
    records = [
        {"signal_date": "2026-01-01", "score": 1, "pnl_pct": 1, "weight": 1.0},
        {"signal_date": "2026-01-02", "score": 2, "pnl_pct": 2, "weight": 1.0},
        {"signal_date": "2026-01-03", "score": 3, "pnl_pct": 3, "weight": 1.0},
        {"signal_date": "2026-01-04", "score": 4, "pnl_pct": 4, "weight": 1.0},
        {"signal_date": "2026-01-05", "score": 5, "pnl_pct": 5, "weight": 1.0},
        {"signal_date": "2026-02-10", "score": 1, "pnl_pct": -1, "weight": 1.0},
        {"signal_date": "2026-02-11", "score": 2, "pnl_pct": -2, "weight": 1.0},
    ]
    windows = ic.windowed_ic_series(records, window_days=30)
    assert len(windows) == 1
    assert math.isclose(windows[0]["ic"], 1.0)
    assert windows[0]["n"] == 5
    assert math.isclose(windows[0]["effective_n"], 5.0)
