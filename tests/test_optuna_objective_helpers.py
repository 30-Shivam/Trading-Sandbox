"""Two independently-testable pieces of optimize.py's objective/candidate-
writing machinery:

taper_drawdown_for_sample_size() -- extends the existing binary
under-sampled gate (MIN_TRADES_FOR_SCORE) to also distrust a trial that
clears the floor but is still thin relative to MIN_TRADES_FOR_TRUSTED_DRAWDOWN.
See improvements.txt item 23's own flagged follow-up: without this, a
--multi-objective search can be fooled into liking a config purely because
it fired too rarely to have accumulated a bad drawdown yet.

build_candidate_config() -- regression coverage for a real bug found while
re-tuning squeeze_breakout with --pin-atr-take-profit-multiplier/
--pin-stop-loss-atr-multiplier for the first time via this CLI path: pinned
tp/sl silently fell back to DEFAULT_CONFIG's values in the written
candidate, since Optuna's own trial.params never contains a pinned field.

is_below_frequency_floor() -- the --min-frequency-fraction gate, added
after a free-filter squeeze_breakout search found a real timing edge that
also collapsed trade frequency ~11x vs. baseline (see improvements.txt) --
nothing in the objective previously penalized a trial for firing
dramatically less often than the baseline it's being compared against.

is_below_win_rate_floor() -- the --min-win-rate gate, added so win_rate
(already reported in every metrics dict) can actually gate selection, not
just be read afterward -- Optuna's own objective only ever optimized
sharpe_like (and optionally max_drawdown), never win_rate directly.
"""
import optimize
import swingtrade


def test_at_floor_fully_penalized():
    got = optimize.taper_drawdown_for_sample_size(40.0, optimize.MIN_TRADES_FOR_SCORE)
    assert got == optimize.UNDER_SAMPLED_DRAWDOWN_PENALTY


def test_at_trusted_threshold_returns_raw():
    got = optimize.taper_drawdown_for_sample_size(40.0, optimize.MIN_TRADES_FOR_TRUSTED_DRAWDOWN)
    assert got == 40.0


def test_above_trusted_threshold_unchanged():
    got = optimize.taper_drawdown_for_sample_size(40.0, optimize.MIN_TRADES_FOR_TRUSTED_DRAWDOWN * 5)
    assert got == 40.0


def test_midpoint_is_linear():
    midpoint = (optimize.MIN_TRADES_FOR_SCORE + optimize.MIN_TRADES_FOR_TRUSTED_DRAWDOWN) / 2
    got = optimize.taper_drawdown_for_sample_size(40.0, midpoint)
    expected = (optimize.UNDER_SAMPLED_DRAWDOWN_PENALTY + 40.0) / 2
    assert abs(got - expected) < 1e-9


def test_a_low_raw_drawdown_still_gets_pulled_toward_penalty_when_thin():
    # A genuinely low (good) raw drawdown at a thin sample should still be
    # pulled MOSTLY toward the penalty -- the whole point is not to trust
    # an attractive-looking number just because the sample is small.
    got = optimize.taper_drawdown_for_sample_size(5.0, optimize.MIN_TRADES_FOR_SCORE + 1)
    assert got > 90.0


def test_build_candidate_config_without_pinning_uses_searched_tp_sl():
    best_params = {
        "squeeze_breakout_zscore_max": -1.5, "atr_take_profit_multiplier": 2.5,
        "stop_loss_atr_multiplier": 1.2,
    }
    config = optimize.build_candidate_config("squeeze_breakout", best_params, None, None)
    assert config.atr_take_profit_multiplier == 2.5
    assert config.stop_loss_atr_multiplier == 1.2


def test_build_candidate_config_with_pinning_preserves_pinned_tp_sl():
    # Regression test for a real bug: when tp/sl is pinned, Optuna's own
    # trial.params never contains those two keys (they're never suggested),
    # so without the explicit override the candidate silently fell back to
    # DEFAULT_CONFIG's tp/sl instead of the pinned values the whole search
    # was actually scored under.
    best_params = {"squeeze_breakout_zscore_max": -1.5}  # no tp/sl keys, matches real Optuna output
    config = optimize.build_candidate_config("squeeze_breakout", best_params, 3.0, 1.0)
    assert config.atr_take_profit_multiplier == 3.0
    assert config.stop_loss_atr_multiplier == 1.0
    assert config.atr_take_profit_multiplier != swingtrade.DEFAULT_CONFIG.atr_take_profit_multiplier


def test_build_candidate_config_strategy_field_set():
    config = optimize.build_candidate_config("ma_crossover", {}, None, None)
    assert config.strategy == "ma_crossover"


def test_frequency_floor_disabled_by_default():
    # None (--min-frequency-fraction omitted or 0) must never penalize,
    # regardless of how low effective_trade_count is -- preserves every
    # existing search's exact behavior unless explicitly opted in.
    assert optimize.is_below_frequency_floor(0.0, None) is False
    assert optimize.is_below_frequency_floor(1000.0, None) is False


def test_frequency_floor_rejects_below_threshold():
    assert optimize.is_below_frequency_floor(50.0, 100.0) is True


def test_frequency_floor_accepts_at_or_above_threshold():
    assert optimize.is_below_frequency_floor(100.0, 100.0) is False
    assert optimize.is_below_frequency_floor(150.0, 100.0) is False


def test_win_rate_floor_disabled_by_default():
    # None (--min-win-rate omitted or 0) must never penalize, regardless of
    # how low win_rate is -- preserves every existing search's exact
    # behavior unless explicitly opted in.
    assert optimize.is_below_win_rate_floor(0.0, None) is False
    assert optimize.is_below_win_rate_floor(None, None) is False


def test_win_rate_floor_rejects_below_threshold():
    assert optimize.is_below_win_rate_floor(30.0, 40.0) is True


def test_win_rate_floor_accepts_at_or_above_threshold():
    assert optimize.is_below_win_rate_floor(40.0, 40.0) is False
    assert optimize.is_below_win_rate_floor(60.0, 40.0) is False


def test_win_rate_floor_rejects_none_win_rate_when_enabled():
    # An under-sampled-to-zero-trades trial has win_rate=None -- can't be
    # trusted to clear a floor it has no measurement for.
    assert optimize.is_below_win_rate_floor(None, 40.0) is True
