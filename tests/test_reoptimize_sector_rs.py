"""recommend_lookback() -- the pure decision logic behind
reoptimize_sector_rs.py (improvements.txt item 76): given a per-lookback IC
table, decide whether a different sector_relative_strength_lookback_days
is worth recommending over the current one. Propose-only by design -- this
function never writes anything, just decides what to print."""
import reoptimize_sector_rs as rsr


def test_no_recommendation_when_current_is_already_best():
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.05},
        {"lookback_days": 63, "n": 100, "ic": 0.20},
        {"lookback_days": 126, "n": 100, "ic": 0.10},
    ]
    assert rsr.recommend_lookback(results, current_lookback=63) is None


def test_recommends_when_margin_cleared():
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.20},
        {"lookback_days": 63, "n": 100, "ic": 0.05},
    ]
    rec = rsr.recommend_lookback(results, current_lookback=63, margin=0.02)
    assert rec is not None
    assert rec["lookback_days"] == 21
    assert rec["current_ic"] == 0.05


def test_no_recommendation_when_margin_not_cleared():
    # best is only marginally better than current -- below the required margin
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.06},
        {"lookback_days": 63, "n": 100, "ic": 0.05},
    ]
    assert rsr.recommend_lookback(results, current_lookback=63, margin=0.02) is None


def test_no_recommendation_when_every_ic_undefined():
    results = [
        {"lookback_days": 21, "n": 2, "ic": None},
        {"lookback_days": 63, "n": 2, "ic": None},
    ]
    assert rsr.recommend_lookback(results, current_lookback=63) is None


def test_current_ic_undefined_but_another_candidate_has_one():
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.15},
        {"lookback_days": 63, "n": 2, "ic": None},
    ]
    rec = rsr.recommend_lookback(results, current_lookback=63)
    assert rec is not None
    assert rec["lookback_days"] == 21
    assert rec["current_ic"] is None


def test_no_recommendation_when_current_ic_undefined_and_it_is_also_the_only_candidate():
    results = [{"lookback_days": 63, "n": 2, "ic": None}]
    assert rsr.recommend_lookback(results, current_lookback=63) is None


def test_no_recommendation_when_best_is_negative_even_if_less_bad_than_current():
    # 2026-09-03 fix: previously this recommended 21d purely because -0.01
    # beats -0.20 by more than the margin -- a real incident (item 76) found
    # this exact "least-bad of N weak options" trap fire on evidence too
    # thin to mean anything either way. The winning candidate's own IC must
    # now ALSO clear MIN_ABSOLUTE_IC, not just beat current by the margin.
    results = [
        {"lookback_days": 21, "n": 100, "ic": -0.01},
        {"lookback_days": 63, "n": 100, "ic": -0.20},
    ]
    assert rsr.recommend_lookback(results, current_lookback=63, margin=0.02) is None


def test_negative_ics_still_compared_correctly_once_winner_clears_absolute_floor():
    # Same shape as above, but the winning candidate's own IC now clears
    # the absolute floor -- the relative-margin logic must still work
    # correctly ABOVE that floor, current just also happens to be negative.
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.05},
        {"lookback_days": 63, "n": 100, "ic": -0.20},
    ]
    rec = rsr.recommend_lookback(results, current_lookback=63, margin=0.02)
    assert rec is not None
    assert rec["lookback_days"] == 21


def test_no_recommendation_when_best_ic_below_absolute_floor_even_without_current():
    # Absolute floor applies even when current has no defined IC at all --
    # recommending "switch to this" when the candidate's own IC is itself
    # uninformative is exactly as flawed with or without a current baseline.
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.01},
        {"lookback_days": 63, "n": 2, "ic": None},
    ]
    assert rsr.recommend_lookback(results, current_lookback=63) is None


def test_recommendation_fires_at_exactly_the_absolute_floor():
    results = [
        {"lookback_days": 21, "n": 100, "ic": 0.02},
        {"lookback_days": 63, "n": 100, "ic": -0.20},
    ]
    rec = rsr.recommend_lookback(results, current_lookback=63, margin=0.02, min_absolute_ic=0.02)
    assert rec is not None
    assert rec["lookback_days"] == 21
