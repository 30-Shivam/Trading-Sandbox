"""flag_staleness() -- the pure comparison logic behind Check B of
reoptimize_best_ideas.py (improvements.txt item 76): given a fresh HOLDOUT
summary and a stored baseline, decide whether a live mechanical strategy
(ma_crossover/squeeze_breakout) looks stale enough to flag for a manual
re-look. Propose-only by design -- never writes anything, just decides
what to report."""
import reoptimize_best_ideas as rbi


def test_no_flag_when_no_baseline_exists_yet():
    fresh = {"sharpe_like": 0.01}
    assert rbi.flag_staleness("ma_crossover", fresh, {}) is None


def test_flags_meaningful_sharpe_drop():
    fresh = {"sharpe_like": 0.01}
    baseline = {"holdout_sharpe_like": 0.08}
    flag = rbi.flag_staleness("ma_crossover", fresh, baseline, threshold=0.03)
    assert flag is not None
    assert flag["strategy"] == "ma_crossover"
    assert flag["fresh_sharpe"] == 0.01
    assert flag["baseline_sharpe"] == 0.08


def test_no_flag_when_drop_below_threshold():
    fresh = {"sharpe_like": 0.06}
    baseline = {"holdout_sharpe_like": 0.08}
    assert rbi.flag_staleness("ma_crossover", fresh, baseline, threshold=0.03) is None


def test_no_flag_when_sharpe_improved():
    fresh = {"sharpe_like": 0.12}
    baseline = {"holdout_sharpe_like": 0.08}
    assert rbi.flag_staleness("ma_crossover", fresh, baseline, threshold=0.03) is None


def test_flags_when_fresh_sharpe_undefined():
    fresh = {"sharpe_like": None}
    baseline = {"holdout_sharpe_like": 0.08}
    flag = rbi.flag_staleness("squeeze_breakout", fresh, baseline)
    assert flag is not None
    assert flag["fresh_sharpe"] is None
    assert "undefined" in flag["reason"]


def test_no_flag_when_baseline_sharpe_missing_but_baseline_entry_exists():
    fresh = {"sharpe_like": 0.05}
    baseline = {"config_version": 39}  # entry exists but no holdout_sharpe_like key yet
    assert rbi.flag_staleness("squeeze_breakout", fresh, baseline) is None


def test_exact_threshold_boundary_flags():
    fresh = {"sharpe_like": 0.05}
    baseline = {"holdout_sharpe_like": 0.08}
    # drop is exactly 0.03 -- >= threshold should flag
    flag = rbi.flag_staleness("ma_crossover", fresh, baseline, threshold=0.03)
    assert flag is not None
