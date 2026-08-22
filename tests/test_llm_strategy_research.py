"""llm_strategy_research.validate_rule() -- the safety-critical gate
between an LLM-proposed rule and a real backtest. Every case here mirrors
a real way a proposal could be malformed or unsafe: unknown field, unknown
op, out-of-bounds numeric param, missing required exit param. See
swingtrade.evaluate_llm_rule_conditions() for the interpreter this rule
DSL feeds (tested separately, tests/test_llm_strategy_engine.py).
"""
import llm_strategy_research as lsr


def _atr_rule(**overrides):
    rule = {
        "conditions": [{"field": "RSI", "op": "<", "value": 30}],
        "logic": "AND",
        "exit": {"type": "atr_bracket", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0},
    }
    rule.update(overrides)
    return rule


def test_valid_atr_bracket_rule_passes():
    ok, reason = lsr.validate_rule(_atr_rule())
    assert ok, reason


def test_valid_trailing_stop_rule_passes():
    rule = _atr_rule(exit={
        "type": "trailing_stop", "take_profit_atr_multiplier": 2.0,
        "stop_loss_atr_multiplier": 1.0, "trailing_stop_atr_multiplier": 1.5,
    })
    ok, reason = lsr.validate_rule(rule)
    assert ok, reason


def test_valid_time_based_rule_passes():
    rule = _atr_rule(exit={
        "type": "time_based", "take_profit_atr_multiplier": 2.0,
        "stop_loss_atr_multiplier": 1.0, "max_holding_days": 10,
    })
    ok, reason = lsr.validate_rule(rule)
    assert ok, reason


def test_multi_condition_or_logic_passes():
    rule = _atr_rule(
        conditions=[{"field": "RSI", "op": "<", "value": 30}, {"field": "ADX", "op": ">", "value": 25}],
        logic="OR",
    )
    ok, reason = lsr.validate_rule(rule)
    assert ok, reason


def test_rejects_non_dict():
    ok, reason = lsr.validate_rule("not a rule")
    assert not ok


def test_rejects_empty_conditions():
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[]))
    assert not ok
    assert "no conditions" in reason


def test_rejects_unknown_field():
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[{"field": "MADE_UP_INDICATOR", "op": "<", "value": 30}]))
    assert not ok
    assert "unknown field" in reason


def test_rejects_unknown_op():
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[{"field": "RSI", "op": "!=", "value": 30}]))
    assert not ok
    assert "unknown op" in reason


def test_rejects_non_numeric_condition_value():
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[{"field": "RSI", "op": "<", "value": "thirty"}]))
    assert not ok
    assert "numeric" in reason


def test_rejects_boolean_condition_value():
    # bool is technically an int subclass in Python -- must be explicitly excluded
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[{"field": "RSI", "op": "<", "value": True}]))
    assert not ok


def test_rejects_out_of_bounds_condition_value():
    ok, reason = lsr.validate_rule(_atr_rule(conditions=[{"field": "RSI", "op": "<", "value": 1e9}]))
    assert not ok
    assert "out of bounds" in reason


def test_rejects_unknown_logic():
    ok, reason = lsr.validate_rule(_atr_rule(logic="XOR"))
    assert not ok
    assert "logic" in reason


def test_rejects_missing_exit():
    rule = _atr_rule()
    del rule["exit"]
    ok, reason = lsr.validate_rule(rule)
    assert not ok
    assert "exit" in reason


def test_rejects_unknown_exit_type():
    ok, reason = lsr.validate_rule(_atr_rule(exit={"type": "moon_landing", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0}))
    assert not ok
    assert "exit type" in reason


def test_rejects_atr_multiplier_out_of_bounds():
    ok, reason = lsr.validate_rule(_atr_rule(exit={
        "type": "atr_bracket", "take_profit_atr_multiplier": 500.0, "stop_loss_atr_multiplier": 1.0,
    }))
    assert not ok
    assert "out of bounds" in reason


def test_rejects_missing_take_profit_multiplier():
    ok, reason = lsr.validate_rule(_atr_rule(exit={"type": "atr_bracket", "stop_loss_atr_multiplier": 1.0}))
    assert not ok


def test_rejects_trailing_stop_missing_trail_multiplier():
    ok, reason = lsr.validate_rule(_atr_rule(exit={
        "type": "trailing_stop", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0,
    }))
    assert not ok
    assert "trailing_stop_atr_multiplier" in reason


def test_rejects_time_based_missing_max_holding_days():
    ok, reason = lsr.validate_rule(_atr_rule(exit={
        "type": "time_based", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0,
    }))
    assert not ok
    assert "max_holding_days" in reason


def test_rejects_time_based_max_holding_days_out_of_bounds():
    ok, reason = lsr.validate_rule(_atr_rule(exit={
        "type": "time_based", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0,
        "max_holding_days": 9000,
    }))
    assert not ok


def test_rejects_time_based_max_holding_days_not_an_int():
    ok, reason = lsr.validate_rule(_atr_rule(exit={
        "type": "time_based", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0,
        "max_holding_days": 10.5,
    }))
    assert not ok


def test_parse_proposal_response_valid():
    text = (
        '{"rule": {"conditions": [{"field": "RSI", "op": "<", "value": 30}], "logic": "AND", '
        '"exit": {"type": "atr_bracket", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0}}, '
        '"parent_cycle_id": null, "rationale": "oversold bounce", "notes_for_next_time": "try tighter RSI next"}'
    )
    result = lsr._parse_proposal_response(text)
    assert result is not None
    assert result["rationale"] == "oversold bounce"
    assert result["parent_cycle_id"] is None
    assert result["notes_for_next_time"] == "try tighter RSI next"


def test_parse_proposal_response_rejects_invalid_rule_even_if_json_valid():
    text = (
        '{"rule": {"conditions": [{"field": "NOT_A_FIELD", "op": "<", "value": 30}], "logic": "AND", '
        '"exit": {"type": "atr_bracket", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0}}, '
        '"parent_cycle_id": null, "rationale": "x", "notes_for_next_time": ""}'
    )
    assert lsr._parse_proposal_response(text) is None


def test_parse_proposal_response_rejects_malformed_json():
    assert lsr._parse_proposal_response("not json") is None


def test_parse_proposal_response_rejects_missing_rationale():
    text = (
        '{"rule": {"conditions": [{"field": "RSI", "op": "<", "value": 30}], "logic": "AND", '
        '"exit": {"type": "atr_bracket", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0}}}'
    )
    assert lsr._parse_proposal_response(text) is None
