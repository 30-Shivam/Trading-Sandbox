"""audit_parameter_stability.py's pure helper functions -- _numeric_param_series()
and _stability(). No MongoDB needed for these (the script's own main()
handles the real MongoDB query separately, not unit-tested here, same
convention every other MongoDB-backed script in this project follows --
only the pure math gets direct test coverage).

Built 2026-09-13 per a real finding: querying this project's own
System_Config history directly showed ma_crossover's own re-optimization
rounds swinging ma_crossover_short_window from 11 to 29 and
atr_take_profit_multiplier from ~1.0 to ~4.95 within a ~3-week span --
a question no prior validation check (DSR, ticker-holdout, real-vs-random)
had ever asked: does Optuna's own "best" answer actually converge across
independent re-optimization rounds, or does it keep swinging?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_parameter_stability import _numeric_param_series, _stability


def _doc(version, **params):
    return {"version": version, "status": "candidate", "params": params}


def test_numeric_param_series_collects_across_docs():
    docs = [
        _doc(1, strategy="ma_crossover", ma_crossover_short_window=10, atr_take_profit_multiplier=2.0),
        _doc(2, strategy="ma_crossover", ma_crossover_short_window=15, atr_take_profit_multiplier=3.0),
    ]
    series = _numeric_param_series(docs)
    assert series["ma_crossover_short_window"] == [10, 15]
    assert series["atr_take_profit_multiplier"] == [2.0, 3.0]


def test_numeric_param_series_excludes_non_numeric_and_bool_fields():
    docs = [_doc(1, strategy="ma_crossover", ma_crossover_entry_fill="limit", earnings_gate=True, window=10)]
    series = _numeric_param_series(docs)
    assert "strategy" not in series          # string
    assert "ma_crossover_entry_fill" not in series  # string
    assert "earnings_gate" not in series      # bool (isinstance(True, int) is True in Python -- must be excluded explicitly)
    assert series["window"] == [10]


def test_numeric_param_series_handles_missing_field_in_some_docs():
    # A field only present in SOME docs (e.g. a param added later) --
    # should collect only the docs that actually have it, not crash or
    # pad with a fabricated default.
    docs = [
        _doc(1, strategy="ma_crossover", a=1.0),
        _doc(2, strategy="ma_crossover", a=2.0, b=5.0),
    ]
    series = _numeric_param_series(docs)
    assert series["a"] == [1.0, 2.0]
    assert series["b"] == [5.0]


def test_stability_zero_variation_gives_zero_range_over_mean():
    result = _stability([2.0, 2.0, 2.0])
    assert result["range_over_mean"] == 0.0
    assert result["mean"] == 2.0
    assert result["min"] == 2.0
    assert result["max"] == 2.0


def test_stability_hand_computed_real_case():
    # Matches the real ma_crossover finding shape: short_window 11..29.
    values = [11, 17, 13, 24, 24, 29, 12, 11, 29, 11, 27, 13, 11]
    result = _stability(values)
    mean = sum(values) / len(values)
    assert result["mean"] == round(mean, 4)
    assert result["min"] == 11
    assert result["max"] == 29
    assert result["range_over_mean"] == round((29 - 11) / mean, 4)


def test_stability_handles_negative_mean():
    # pairs_zscore_entry_max is always negative -- range/|mean| must stay
    # a sensible positive instability measure, not a nonsensical negative one.
    values = [-3.19, -1.86, -2.29, -2.20]
    result = _stability(values)
    assert result["range_over_mean"] > 0
