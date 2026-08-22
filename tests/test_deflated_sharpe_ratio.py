"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, improvements.txt item 86)
-- corrects an observed Sharpe ratio for SELECTION BIAS from trying many
candidates, a different question from the ticker-holdout multi-seed
averaging (item 69), which checks generalization across tickers instead.
"""
import math

import optimize


def test_norm_ppf_matches_known_reference_values():
    # Standard normal quantile function -- hand-checkable against any
    # statistics table.
    assert abs(optimize._norm_ppf(0.5) - 0.0) < 1e-9
    assert abs(optimize._norm_ppf(0.975) - 1.959963986) < 1e-6
    assert abs(optimize._norm_ppf(0.95) - 1.644853626) < 1e-6
    assert abs(optimize._norm_ppf(0.025) - (-1.959963986)) < 1e-6


def test_norm_ppf_rejects_invalid_probability():
    import pytest
    with pytest.raises(ValueError):
        optimize._norm_ppf(0.0)
    with pytest.raises(ValueError):
        optimize._norm_ppf(1.0)
    with pytest.raises(ValueError):
        optimize._norm_ppf(1.5)


def test_norm_cdf_matches_known_reference_values():
    assert abs(optimize._norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(optimize._norm_cdf(1.959963986) - 0.975) < 1e-6
    assert abs(optimize._norm_cdf(-1.959963986) - 0.025) < 1e-6


def test_norm_cdf_and_ppf_are_inverses():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        x = optimize._norm_ppf(p)
        assert abs(optimize._norm_cdf(x) - p) < 1e-6


def test_expected_max_sharpe_null_zero_for_single_trial():
    # No selection bias picking a "winner" from exactly one trial.
    assert optimize.expected_max_sharpe_null(sharpe_std=0.5, n_trials=1) == 0.0


def test_expected_max_sharpe_null_none_for_invalid_inputs():
    assert optimize.expected_max_sharpe_null(sharpe_std=0.0, n_trials=50) is None
    assert optimize.expected_max_sharpe_null(sharpe_std=-0.1, n_trials=50) is None
    assert optimize.expected_max_sharpe_null(sharpe_std=0.5, n_trials=0) is None


def test_expected_max_sharpe_null_increases_with_more_trials():
    # More trials searched -> a higher bar you'd expect to clear by chance
    # alone, for the same sharpe_std.
    sr0_10 = optimize.expected_max_sharpe_null(sharpe_std=0.3, n_trials=10)
    sr0_50 = optimize.expected_max_sharpe_null(sharpe_std=0.3, n_trials=50)
    sr0_200 = optimize.expected_max_sharpe_null(sharpe_std=0.3, n_trials=200)
    assert 0 < sr0_10 < sr0_50 < sr0_200


def test_expected_max_sharpe_null_scales_with_sharpe_std():
    sr0_a = optimize.expected_max_sharpe_null(sharpe_std=0.2, n_trials=50)
    sr0_b = optimize.expected_max_sharpe_null(sharpe_std=0.4, n_trials=50)
    assert abs(sr0_b - 2 * sr0_a) < 1e-9


def test_deflated_sharpe_ratio_high_confidence_for_a_clear_winner():
    # Observed Sharpe far above the expected best-of-N null, with a large
    # sample -- should read as high confidence (close to 1.0).
    dsr = optimize.deflated_sharpe_ratio(
        observed_sharpe=2.0, sharpe_std=0.2, n_trials=50, n_observations=500,
    )
    assert dsr is not None
    assert dsr > 0.95


def test_deflated_sharpe_ratio_low_confidence_for_a_marginal_winner():
    # Observed Sharpe barely above (or in line with) what pure chance
    # across many trials would produce -- should read as low/uncertain.
    sr0 = optimize.expected_max_sharpe_null(sharpe_std=0.3, n_trials=100)
    dsr = optimize.deflated_sharpe_ratio(
        observed_sharpe=sr0, sharpe_std=0.3, n_trials=100, n_observations=100,
    )
    assert dsr is not None
    assert abs(dsr - 0.5) < 0.05  # right at the null expectation -> ~50/50


def test_deflated_sharpe_ratio_more_trials_lowers_confidence_for_the_same_result():
    # The exact same observed result should look LESS impressive (lower
    # DSR) the more candidates were searched to find it -- the whole point
    # of a selection-bias correction.
    dsr_few = optimize.deflated_sharpe_ratio(
        observed_sharpe=1.0, sharpe_std=0.3, n_trials=5, n_observations=200,
    )
    dsr_many = optimize.deflated_sharpe_ratio(
        observed_sharpe=1.0, sharpe_std=0.3, n_trials=500, n_observations=200,
    )
    assert dsr_many < dsr_few


def test_deflated_sharpe_ratio_none_for_too_few_observations():
    assert optimize.deflated_sharpe_ratio(
        observed_sharpe=1.0, sharpe_std=0.3, n_trials=50, n_observations=1,
    ) is None
    assert optimize.deflated_sharpe_ratio(
        observed_sharpe=1.0, sharpe_std=0.3, n_trials=50, n_observations=None,
    ) is None


def test_deflated_sharpe_ratio_returns_a_valid_probability():
    for observed in (-1.0, 0.0, 0.05, 0.5, 3.0):
        dsr = optimize.deflated_sharpe_ratio(
            observed_sharpe=observed, sharpe_std=0.25, n_trials=40, n_observations=150,
        )
        assert dsr is not None
        assert 0.0 <= dsr <= 1.0
