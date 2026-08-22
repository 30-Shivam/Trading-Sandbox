"""swingtrade's generic LLM-rule interpreter (improvements.txt item 93) --
evaluate_llm_rule_conditions()/rule_exit_to_config()/
precompute_llm_strategy_frame(). This is the safety-critical engine every
LLM-proposed rule actually runs through -- the LLM never generates or
executes code, it only composes a JSON rule this pure, hand-verifiable
machinery interprets.
"""
import numpy as np
import pandas as pd
import pytest

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _frame(rsi_values, adx_values=None):
    n = len(rsi_values)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data = {"RSI": rsi_values}
    if adx_values is not None:
        data["ADX"] = adx_values
    return pd.DataFrame(data, index=idx)


def test_single_condition_less_than():
    frame = _frame([25, 35, 45])
    rule = {"conditions": [{"field": "RSI", "op": "<", "value": 30}], "logic": "AND"}
    result = swingtrade.evaluate_llm_rule_conditions(frame, rule)
    assert list(result) == [True, False, False]


def test_and_logic_requires_all_conditions():
    frame = _frame([25, 25, 45], adx_values=[30, 15, 30])
    rule = {
        "conditions": [{"field": "RSI", "op": "<", "value": 30}, {"field": "ADX", "op": ">", "value": 20}],
        "logic": "AND",
    }
    result = swingtrade.evaluate_llm_rule_conditions(frame, rule)
    assert list(result) == [True, False, False]


def test_or_logic_requires_any_condition():
    frame = _frame([25, 25, 45], adx_values=[30, 15, 30])
    rule = {
        "conditions": [{"field": "RSI", "op": "<", "value": 20}, {"field": "ADX", "op": ">", "value": 20}],
        "logic": "OR",
    }
    result = swingtrade.evaluate_llm_rule_conditions(frame, rule)
    assert list(result) == [True, False, True]


def test_nan_values_evaluate_to_false_not_true():
    frame = _frame([np.nan, 25, np.nan])
    rule = {"conditions": [{"field": "RSI", "op": "<", "value": 30}], "logic": "AND"}
    result = swingtrade.evaluate_llm_rule_conditions(frame, rule)
    assert list(result) == [False, True, False]


def test_raises_on_unknown_field():
    frame = _frame([25, 35])
    rule = {"conditions": [{"field": "NOT_A_REAL_FIELD", "op": "<", "value": 30}], "logic": "AND"}
    with pytest.raises(ValueError, match="unknown field"):
        swingtrade.evaluate_llm_rule_conditions(frame, rule)


def test_raises_on_field_not_present_in_this_frame():
    # ADX is a KNOWN_INDICATOR_FIELD but this particular frame doesn't carry it
    frame = _frame([25, 35])
    rule = {"conditions": [{"field": "ADX", "op": ">", "value": 20}], "logic": "AND"}
    with pytest.raises(ValueError, match="not present"):
        swingtrade.evaluate_llm_rule_conditions(frame, rule)


def test_raises_on_unknown_op():
    frame = _frame([25, 35])
    rule = {"conditions": [{"field": "RSI", "op": "!=", "value": 30}], "logic": "AND"}
    with pytest.raises(ValueError, match="unknown op"):
        swingtrade.evaluate_llm_rule_conditions(frame, rule)


def test_raises_on_empty_conditions():
    frame = _frame([25, 35])
    with pytest.raises(ValueError, match="no conditions"):
        swingtrade.evaluate_llm_rule_conditions(frame, {"conditions": [], "logic": "AND"})


def test_raises_on_non_numeric_value():
    frame = _frame([25, 35])
    rule = {"conditions": [{"field": "RSI", "op": "<", "value": "thirty"}], "logic": "AND"}
    with pytest.raises(ValueError, match="numeric"):
        swingtrade.evaluate_llm_rule_conditions(frame, rule)


# --- rule_exit_to_config(): translates rule["exit"] into a real
# TradingConfig, reusing existing fields -- no new settlement mechanics.

def _base_rule(exit_spec):
    return {"conditions": [{"field": "RSI", "op": "<", "value": 30}], "logic": "AND", "exit": exit_spec}


def test_atr_bracket_exit_sets_multipliers_and_disables_trailing():
    rule = _base_rule({"type": "atr_bracket", "take_profit_atr_multiplier": 3.0, "stop_loss_atr_multiplier": 1.5})
    config = swingtrade.rule_exit_to_config(rule, CONFIG)
    assert config.atr_take_profit_multiplier == 3.0
    assert config.stop_loss_atr_multiplier == 1.5
    assert config.trailing_stop_enabled is False


def test_trailing_stop_exit_enables_trailing_and_sets_trail_multiplier():
    rule = _base_rule({
        "type": "trailing_stop", "take_profit_atr_multiplier": 2.0,
        "stop_loss_atr_multiplier": 1.0, "trailing_stop_atr_multiplier": 1.5,
    })
    config = swingtrade.rule_exit_to_config(rule, CONFIG)
    assert config.trailing_stop_enabled is True
    assert config.trailing_stop_atr_multiplier == 1.5


def test_time_based_exit_overrides_max_holding_days():
    rule = _base_rule({
        "type": "time_based", "take_profit_atr_multiplier": 2.0,
        "stop_loss_atr_multiplier": 1.0, "max_holding_days": 7,
    })
    config = swingtrade.rule_exit_to_config(rule, CONFIG)
    assert config.max_holding_days == 7
    assert config.trailing_stop_enabled is False


def test_rule_exit_to_config_preserves_every_other_base_config_field():
    rule = _base_rule({"type": "atr_bracket", "take_profit_atr_multiplier": 3.0, "stop_loss_atr_multiplier": 1.5})
    config = swingtrade.rule_exit_to_config(rule, CONFIG)
    assert config.sma_trend_window == CONFIG.sma_trend_window
    assert config.min_dollar_volume == CONFIG.min_dollar_volume
