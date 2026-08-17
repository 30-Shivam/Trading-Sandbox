"""best_ideas.py -- hand-computable unit tests for the pure scoring/
blending logic (best_ideas_signal, llm_bullishness_score,
qualitative_composite_score, compute_sector_rs_scores,
gather_candidate_universe, blend_composite). run_best_ideas() itself
orchestrates real network/LLM calls and is exercised via the live smoke
test, not here -- same split this project already uses for
market_data.score_bundle_for_strategy() vs. its own unit-testable pieces.
"""
import math

import pandas as pd

import best_ideas as bi


# ---------------------------------------------------------------- signal thresholds

def test_best_ideas_signal_thresholds():
    assert bi.best_ideas_signal(75.0) == "Strong Buy"
    assert bi.best_ideas_signal(100.0) == "Strong Buy"
    assert bi.best_ideas_signal(74.9) == "Buy"
    assert bi.best_ideas_signal(60.0) == "Buy"
    assert bi.best_ideas_signal(59.9) == "Watch"
    assert bi.best_ideas_signal(45.0) == "Watch"
    assert bi.best_ideas_signal(44.9) == "Ignore"
    assert bi.best_ideas_signal(0.0) == "Ignore"
    assert bi.best_ideas_signal(None) == "Ignore"


# ---------------------------------------------------------------- llm_bullishness_score

def test_llm_bullishness_score_buy_stretches_upward():
    assert bi.llm_bullishness_score("Buy", 80) == 90.0
    assert bi.llm_bullishness_score("Buy", 0) == 50.0
    assert bi.llm_bullishness_score("Buy", 100) == 100.0


def test_llm_bullishness_score_avoid_stretches_downward():
    assert bi.llm_bullishness_score("Avoid", 80) == 10.0
    assert bi.llm_bullishness_score("Avoid", 100) == 0.0


def test_llm_bullishness_score_hold_always_neutral():
    assert bi.llm_bullishness_score("Hold", 0) == 50.0
    assert bi.llm_bullishness_score("Hold", 100) == 50.0


# ---------------------------------------------------------------- qualitative_composite_score

def test_qualitative_composite_score_none_when_nothing_available():
    assert bi.qualitative_composite_score(None) is None
    assert bi.qualitative_composite_score({}) is None
    assert bi.qualitative_composite_score({"analyst": None, "insider": None}) is None


def test_qualitative_composite_score_insider_buying_only():
    result = bi.qualitative_composite_score({"insider": {"net_direction": "Buying"}})
    assert result is not None
    assert result["score"] == 60.0
    assert result["breakdown"] == ["Insider net buying (+10)"]


def test_qualitative_composite_score_combined_bullish_signals():
    qualitative = {
        "insider": {"net_direction": "Buying"},
        "short_interest": {"trend": "Decreasing"},
        "options": {"put_call_volume_ratio": 0.5},
    }
    result = bi.qualitative_composite_score(qualitative)
    assert result is not None
    # 50 + 10 (insider) + 8 (short interest) + 6 (options) = 74
    assert result["score"] == 74.0
    assert len(result["breakdown"]) == 3


def test_qualitative_composite_score_analyst_trend_improving():
    qualitative = {
        "analyst": {
            "trend": [
                "2025-11: strongBuy=1, buy=2, hold=3, sell=1, strongSell=0",  # net = 2*1+2 - (0+1) = 3
                "2026-02: strongBuy=3, buy=2, hold=1, sell=0, strongSell=0",  # net = 2*3+2 - 0 = 8
            ],
        },
    }
    result = bi.qualitative_composite_score(qualitative)
    assert result is not None
    assert result["score"] == 58.0
    assert "Analyst trend improving (+8)" in result["breakdown"]


def test_qualitative_composite_score_analyst_actions_net_zero_still_counts_as_a_signal():
    qualitative = {"analyst": {"recent_actions": ["Firm A: Hold -> Buy (up)", "Firm B: Buy -> Hold (down)"]}}
    result = bi.qualitative_composite_score(qualitative)
    assert result is not None  # a signal fired even though it net to 0
    assert result["score"] == 50.0


def test_qualitative_composite_score_bearish_signals_pull_score_down():
    qualitative = {
        "insider": {"net_direction": "Selling"},
        "short_interest": {"trend": "Increasing"},
        "options": {"put_call_volume_ratio": 1.5},
    }
    result = bi.qualitative_composite_score(qualitative)
    assert result is not None
    # 50 - 10 - 8 - 6 = 26
    assert result["score"] == 26.0


# ---------------------------------------------------------------- compute_sector_rs_scores

def _flat_close_df(closes):
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
        "Volume": [1_000_000] * len(closes),
    })


def test_compute_sector_rs_scores_ranks_correctly():
    lookback_days = 5
    sector_df = _flat_close_df([100, 102, 104, 106, 108, 110])  # +10%
    bundle = {
        "A": {"df": _flat_close_df([50, 55, 60, 65, 70, 80])},   # +60% -> RS = 0.50
        "B": {"df": _flat_close_df([50, 51, 52, 53, 54, 55])},   # +10% -> RS = 0.00
        "C": {"df": _flat_close_df([50, 48, 46, 44, 42, 40])},   # -20% -> RS = -0.30
    }
    sector_lookup = {"A": "Tech", "B": "Tech", "C": "Tech"}
    sector_data = {"Tech": sector_df}

    result = bi.compute_sector_rs_scores(bundle, sector_data, sector_lookup, lookback_days)

    assert math.isclose(result["A"]["relative_strength"], 0.5, rel_tol=1e-6)
    assert math.isclose(result["B"]["relative_strength"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["C"]["relative_strength"], -0.3, rel_tol=1e-6)
    # 3 tickers, ascending rank: C (lowest) -> 1/3, B -> 2/3, A (highest) -> 3/3
    assert math.isclose(result["C"]["percentile"], 33.33, abs_tol=0.01)
    assert math.isclose(result["B"]["percentile"], 66.67, abs_tol=0.01)
    assert result["A"]["percentile"] == 100.0


def test_compute_sector_rs_scores_excludes_ticker_without_sector_data():
    bundle = {"A": {"df": _flat_close_df([50, 55, 60, 65, 70, 80])}}
    result = bi.compute_sector_rs_scores(bundle, {}, {"A": "Tech"}, lookback_days=5)
    assert result == {}


# ---------------------------------------------------------------- gather_candidate_universe

def _mech_df(rows):
    return pd.DataFrame(rows)


def test_gather_candidate_universe_unions_and_ranks_and_caps():
    strategy_frames = {
        "ma_crossover": _mech_df([
            {"Ticker": "X", "Signal": "Buy", "Trade_Score": 80.0},
            {"Ticker": "Y", "Signal": "Buy", "Trade_Score": 60.0},
        ]),
    }
    regime_picks = [{"Ticker": "Z", "Trade_Score": 90.0}]
    sector_rs_scores = {
        "W": {"percentile": 95.0},  # not already present, >= 90 -> eligible
        "V": {"percentile": 50.0},  # below the 90 floor -> excluded
    }
    result = bi.gather_candidate_universe(strategy_frames, regime_picks, sector_rs_scores, max_candidates=4)
    assert result == ["W", "Z", "X", "Y"]


def test_gather_candidate_universe_sector_only_addition_needs_remaining_slots():
    strategy_frames = {
        "ma_crossover": _mech_df([
            {"Ticker": "X", "Signal": "Buy", "Trade_Score": 80.0},
            {"Ticker": "Y", "Signal": "Buy", "Trade_Score": 60.0},
        ]),
    }
    regime_picks = [{"Ticker": "Z", "Trade_Score": 90.0}]
    sector_rs_scores = {"W": {"percentile": 95.0}}
    # max_candidates == len(mechanical+regime) already -- no room left for W.
    result = bi.gather_candidate_universe(strategy_frames, regime_picks, sector_rs_scores, max_candidates=3)
    assert "W" not in result
    assert result == ["Z", "X", "Y"]


def test_gather_candidate_universe_ignore_rows_excluded():
    strategy_frames = {
        "ma_crossover": _mech_df([{"Ticker": "X", "Signal": "Ignore", "Trade_Score": 0.0}]),
    }
    result = bi.gather_candidate_universe(strategy_frames, [], {}, max_candidates=10)
    assert result == []


# ---------------------------------------------------------------- blend_composite

def test_blend_composite_weighted_average_hand_computed():
    scores = {"a": 80.0, "b": 60.0}
    weights = {"a": 1.0, "b": 0.5}
    composite, breakdown = bi.blend_composite(scores, weights)
    # (80*1.0 + 60*0.5) / 1.5 = 110 / 1.5 = 73.333...
    assert math.isclose(composite, 73.33, abs_tol=0.01)
    assert math.isclose(breakdown["a"]["weight"], 1.0 / 1.5, rel_tol=1e-3)
    assert math.isclose(breakdown["b"]["weight"], 0.5 / 1.5, rel_tol=1e-3)


def test_blend_composite_falls_back_to_equal_weight_when_all_weights_zero():
    scores = {"a": 80.0, "b": 60.0}
    weights = {"a": 0.0, "b": 0.0}
    composite, breakdown = bi.blend_composite(scores, weights)
    assert composite == 70.0
    assert breakdown["a"]["weight"] == 0.5
    assert breakdown["b"]["weight"] == 0.5


def test_blend_composite_empty_scores_returns_none():
    composite, breakdown = bi.blend_composite({}, {})
    assert composite is None
    assert breakdown == {}


def test_blend_composite_ignores_none_scores():
    scores = {"a": 80.0, "b": None}
    weights = {"a": 1.0, "b": 1.0}
    composite, breakdown = bi.blend_composite(scores, weights)
    assert composite == 80.0
    assert "b" not in breakdown


def test_blend_composite_missing_weight_defaults_to_one():
    scores = {"a": 80.0, "b": 60.0}
    composite, _ = bi.blend_composite(scores, {})
    assert composite == 70.0
