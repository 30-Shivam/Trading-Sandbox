"""benchmark_random_entry.bonferroni_check() -- the multiple-testing
correction added 2026-09-23 after a real gap was found: DSR
(optimize.deflated_sharpe_ratio()) only corrects for overfitting WITHIN one
strategy's own Optuna search, and permutation_test_gap() only answers
whether ONE already-fixed real-vs-random gap is unlikely by chance -- but
nothing ever corrected for the number of SEPARATE strategy architectures
tried across this project's lifetime (~19 and counting). At an uncorrected
p<0.05 bar tested that many times, the two strategies actually promoted to
real capital-eligible status (ma_crossover p=0.009, rsi_mean_reversion
p=0.020) don't survive a Bonferroni-corrected bar -- and are the ones
confirmed live money-losers.

Uses a real temp file for the registry (not mocked) -- the whole point of
this function is a persistent, cumulative count across real runs, so a
fake in-memory registry wouldn't exercise the part that matters.
"""
import benchmark_random_entry as bre


def test_first_strategy_registers_and_uses_uncorrected_bar(tmp_path):
    registry = tmp_path / "registry.txt"
    result = bre.bonferroni_check("value_rank", 0.001, registry_path=registry)
    assert result["n_trials_ever_tested"] == 1
    assert result["bonferroni_alpha"] == 0.05
    assert result["survives_corrected"] is True
    assert registry.read_text().strip() == "value_rank"


def test_correction_gets_stricter_as_more_strategies_are_registered(tmp_path):
    registry = tmp_path / "registry.txt"
    registry.write_text("\n".join(f"strategy_{i}" for i in range(18)) + "\n")

    # ma_crossover's real recorded p=0.009 -- clears the uncorrected p<0.05
    # bar but must NOT clear the corrected one once it's the 19th trial.
    result = bre.bonferroni_check("ma_crossover", 0.009, registry_path=registry)

    assert result["n_trials_ever_tested"] == 19
    assert abs(result["bonferroni_alpha"] - 0.05 / 19) < 1e-9
    assert result["survives_uncorrected"] is True
    assert result["survives_corrected"] is False


def test_strong_result_survives_even_at_19_trials(tmp_path):
    registry = tmp_path / "registry.txt"
    registry.write_text("\n".join(f"strategy_{i}" for i in range(18)) + "\n")

    # accruals_rank's real recorded p<0.001 -- strong enough to survive
    # even the corrected bar at 19 trials.
    result = bre.bonferroni_check("accruals_rank", 0.0005, registry_path=registry)

    assert result["survives_corrected"] is True


def test_registry_is_deduplicated_and_persists_across_calls(tmp_path):
    registry = tmp_path / "registry.txt"
    bre.bonferroni_check("pairs", 0.5, registry_path=registry)
    bre.bonferroni_check("pairs", 0.5, registry_path=registry)   # same strategy again
    result = bre.bonferroni_check("ma_crossover", 0.5, registry_path=registry)

    assert result["n_trials_ever_tested"] == 2   # not 3 -- "pairs" only counts once


def test_undefined_p_value_gives_undefined_correction(tmp_path):
    registry = tmp_path / "registry.txt"
    result = bre.bonferroni_check("some_strategy", None, registry_path=registry)

    assert result["survives_uncorrected"] is None
    assert result["survives_corrected"] is None
    assert result["n_trials_ever_tested"] == 1   # still registers even with an undefined p_value
