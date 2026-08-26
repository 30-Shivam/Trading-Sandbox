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
import pytest

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


# rrr_scoring_ceiling_check() -- pure arithmetic, no Mongo needed. Reproduces
# the real historical numbers from improvements.txt item 42 (the RRR/
# Trade_Score ceiling bug) to prove the formula matches what actually
# happened, not just internal consistency.

def _config_with_rrr(strategy: str, rrr: float) -> swingtrade.TradingConfig:
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(),
        "strategy": strategy,
        "atr_take_profit_multiplier": rrr,
        "stop_loss_atr_multiplier": 1.0,
    })


def test_rrr_ceiling_default_config_clears_buy_threshold():
    result = optimize.rrr_scoring_ceiling_check("breakout", swingtrade.DEFAULT_CONFIG)
    assert result["clears_buy_threshold"] is True
    assert result["rrr"] == 2.0


def test_rrr_ceiling_reproduces_real_v19_breakout_incident():
    # improvements.txt item 42: v19 RRR=0.393, best-case Trade_Score=39.9,
    # never even reaches Watch, below signal_buy_threshold=60.
    result = optimize.rrr_scoring_ceiling_check("breakout", _config_with_rrr("breakout", 0.393))
    assert result["clears_buy_threshold"] is False
    assert abs(result["best_case_score"] - 39.9) < 0.2


def test_rrr_ceiling_reproduces_real_v27_breakout_retest_incident():
    # v27 RRR=0.293, best-case Trade_Score=38.2.
    result = optimize.rrr_scoring_ceiling_check("breakout_retest", _config_with_rrr("breakout_retest", 0.293))
    assert result["clears_buy_threshold"] is False
    assert abs(result["best_case_score"] - 38.2) < 0.2


def test_rrr_ceiling_reproduces_real_v28_week52_high_incident():
    # v28 RRR=0.408, best-case Trade_Score=40.1.
    result = optimize.rrr_scoring_ceiling_check("week52_high", _config_with_rrr("week52_high", 0.408))
    assert result["clears_buy_threshold"] is False
    assert abs(result["best_case_score"] - 40.1) < 0.2


def test_rrr_ceiling_rsi_strategy_uses_non_rescaled_three_term_formula():
    # strategy="rsi" (add_trade_score itself) doesn't rescale to 100 --
    # DEFAULT_CONFIG's rrr_score_weight(40)+rsi_score_weight(40)+
    # distance_score_weight(20) already sum to 100 by convention, so once
    # RRR actually reaches rrr_score_cap the best case is exactly 100,
    # unlike the rescaled family (e.g. breakout, which only has 2 of the 3
    # terms and stays capped at 66.67 even at the same at-cap RRR).
    at_cap_config = _config_with_rrr("rsi", swingtrade.DEFAULT_CONFIG.rrr_score_cap)
    rsi_result = optimize.rrr_scoring_ceiling_check("rsi", at_cap_config)
    assert rsi_result["best_case_score"] == 100.0
    # (every rescaled add_*_trade_score() ALSO reaches exactly 100 once RRR
    # is at its cap, by construction of the 100/total_weight rescale -- the
    # RRR term's cap behavior only differentiates strategies BELOW the cap.)

    # DEFAULT_CONFIG's own RRR=2.0 sits below its rrr_score_cap=4.0, so
    # neither family reaches its absolute best case there -- rsi's 3rd term
    # (rsi_score_weight) still puts it ahead of breakout's 2-term formula.
    rsi_default = optimize.rrr_scoring_ceiling_check("rsi", swingtrade.DEFAULT_CONFIG)
    breakout_default = optimize.rrr_scoring_ceiling_check("breakout", swingtrade.DEFAULT_CONFIG)
    assert rsi_default["best_case_score"] > breakout_default["best_case_score"]


def test_rrr_ceiling_does_not_clear_strong_buy_by_default():
    # DEFAULT_CONFIG's RRR=2.0 needs >2.8 to ever clear signal_strong_buy_threshold
    # (improvements.txt item covering v49) -- documents a real, known, and
    # accepted limitation, not a bug.
    result = optimize.rrr_scoring_ceiling_check("ma_crossover", swingtrade.DEFAULT_CONFIG)
    assert result["clears_buy_threshold"] is True
    assert result["clears_strong_buy_threshold"] is False


# Saturation direction (2026-08-25) -- the mirror-image failure mode the
# check didn't catch before: RRR so far ABOVE rrr_score_cap that it alone
# already clears a tier threshold, regardless of real RSI/Distance data.

def test_rrr_ceiling_reproduces_real_v17_rsi_mean_reversion_incident():
    # RSI Mean-Reversion's live config (v17): RRR=9.75 against
    # rrr_score_cap=4.0. RRR alone contributes exactly rsi_score_weight(40)
    # == signal_watch_threshold(40) under DEFAULT_CONFIG's thresholds --
    # confirmed live 2026-08-25: 295/295 real signals that day had a
    # saturated RRR (9.38-10.18), and 295/407 watchlist tickers (72%)
    # cleared at least Watch regardless of their real RSI (up to 78,
    # badly overbought, for a strategy meant to buy oversold dips).
    result = optimize.rrr_scoring_ceiling_check("rsi", _config_with_rrr("rsi", 9.75))
    assert result["rrr_alone_score"] == 40.0
    assert result["rrr_alone_meets_watch_threshold"] is True
    assert result["rrr_alone_meets_buy_threshold"] is False  # 40 < signal_buy_threshold=60


def test_rrr_ceiling_default_config_is_not_saturated():
    # DEFAULT_CONFIG's RRR=2.0 sits below rrr_score_cap=4.0 -- RRR alone
    # contributes 20 points (half of rrr_score_weight), well under
    # signal_watch_threshold=40, so RSI/Distance still matter.
    result = optimize.rrr_scoring_ceiling_check("rsi", swingtrade.DEFAULT_CONFIG)
    assert result["rrr_alone_meets_watch_threshold"] is False


def test_rrr_ceiling_saturation_detected_in_rescaled_strategy_too():
    # Same failure mode, non-"rsi" (rescaled) formula family -- an extreme
    # RRR should still be caught after the 100/total_weight rescale.
    result = optimize.rrr_scoring_ceiling_check("breakout", _config_with_rrr("breakout", 20.0))
    assert result["rrr_alone_meets_watch_threshold"] is True


def test_rrr_ceiling_severe_saturation_meets_buy_threshold_too():
    # An even more extreme ratio can saturate past signal_buy_threshold
    # itself, not just Watch -- the most severe form of this bug (a
    # ticker could show Strong Buy/Buy from RRR alone with zero real
    # oversold/proximity signal).
    at_cap_config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "rsi",
        "atr_take_profit_multiplier": 30.0, "stop_loss_atr_multiplier": 1.0,
        "rrr_score_cap": 4.0, "rrr_score_weight": 70,
    })
    result = optimize.rrr_scoring_ceiling_check("rsi", at_cap_config)
    assert result["rrr_alone_meets_buy_threshold"] is True


# tp_multiplier_bounds() -- the search-space-level fix (2026-08-25) so
# Optuna can no longer PROPOSE a saturated ratio in the first place (the
# tests above only check the post-hoc validation report). This is the
# exact mechanism behind RSI Mean-Reversion's real v17 incident: ratio 9.75
# against rrr_score_cap=4.0, reachable under the old unbounded upper end of
# ATR_TAKE_PROFIT_RANGE.

def test_tp_bounds_never_exceed_rrr_ceiling():
    import numpy as np
    for stop_loss in np.linspace(*optimize.STOP_LOSS_ATR_RANGE, num=50):
        low, high = optimize.tp_multiplier_bounds(stop_loss)
        assert high / stop_loss <= optimize.RRR_CEILING + 1e-9


def test_tp_bounds_never_invert_across_full_stop_loss_range():
    import numpy as np
    for stop_loss in np.linspace(*optimize.STOP_LOSS_ATR_RANGE, num=200):
        low, high = optimize.tp_multiplier_bounds(stop_loss)
        assert low <= high


def test_tp_bounds_still_respect_rrr_floor():
    low, high = optimize.tp_multiplier_bounds(1.0)
    assert low >= 1.0 * optimize.RRR_FLOOR


def test_tp_bounds_reproduces_real_v17_incident_now_impossible():
    # v17's real values: stop_loss_atr_multiplier=0.250. The old code let
    # atr_take_profit_multiplier range all the way to ATR_TAKE_PROFIT_RANGE[1]
    # (5.0), giving a reachable ratio of 5.0/0.25=20.0 -- the fixed upper
    # bound now caps it at stop_loss * RRR_CEILING = 0.25*4.0 = 1.0.
    low, high = optimize.tp_multiplier_bounds(0.250)
    assert high == 0.250 * optimize.RRR_CEILING
    assert high / 0.250 <= optimize.RRR_CEILING


# find_live_config_for_strategy() -- needs Mongo (reads real System_Config).
# Skips (not fails) if MONGODB_URI isn't available, same convention as
# test_config_candidates_load.py.

def _mongo_available() -> bool:
    try:
        import storage
        storage.get_db()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_find_live_config_returns_none_for_strategy_with_nothing_live():
    # breakout (v19/v43) was never promoted into SECONDARY_STRATEGY_VERSIONS
    # or EXPERIMENTAL_STRATEGY_VERSIONS after the RRR-ceiling investigation
    # (improvements.txt item 42's own "sits unpromoted pending explicit
    # review" bottom line) -- still nothing live for it as of this writing.
    # (momentum_rank was this exact scenario until 2026-08-24, when it was
    # promoted to EXPERIMENTAL_STRATEGY_VERSIONS -- see
    # test_find_live_config_finds_secondary_strategy_by_matching_params_strategy's
    # sibling coverage for the "something IS live" case, which now also
    # covers momentum_rank via EXPERIMENTAL_STRATEGY_VERSIONS.)
    config, label = optimize.find_live_config_for_strategy("breakout")
    assert config is None
    assert label is None


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_find_live_config_finds_secondary_strategy_by_matching_params_strategy():
    import config_loader
    for label, version in config_loader.SECONDARY_STRATEGY_VERSIONS.items():
        config, found_label = optimize.find_live_config_for_strategy(config_loader.load_config_by_version(version)[0].strategy)
        assert config is not None
        assert str(version) in found_label


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_find_live_config_finds_experimental_strategy_too():
    # momentum_rank (v65, promoted 2026-08-24) is EXPERIMENTAL, not
    # SECONDARY -- confirms find_live_config_for_strategy() checks both
    # dicts, not just SECONDARY_STRATEGY_VERSIONS.
    config, label = optimize.find_live_config_for_strategy("momentum_rank")
    assert config is not None
    assert "65" in label
    assert "experimental" in label
