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


def test_build_proposal_prompt_nudges_or_logic_after_zero_trade_history():
    """2026-09-01 fix: the first 5 real research cycles all used "AND" with
    3-4 conditions and never once tried "OR" or fewer conditions, despite
    the schema always having supported it -- see
    swingtrade.evaluate_llm_rule_conditions()'s own "AND"/"OR" logic param.
    The prompt now explicitly names this pattern and suggests the
    alternative, rather than silently hoping the LLM discovers it."""
    prompt = lsr._build_proposal_prompt([])
    assert '"logic": "OR"' in prompt
    assert "never OR" in prompt or "OR logic" in prompt.lower() or "logic\": \"OR\"" in prompt


def _fake_proposal(**overrides):
    proposal = {
        "rule": {
            "conditions": [{"field": "RSI", "op": "<", "value": 30}],
            "logic": "AND",
            "exit": {"type": "atr_bracket", "take_profit_atr_multiplier": 2.0, "stop_loss_atr_multiplier": 1.0},
        },
        "rationale": "test",
        "notes_for_next_time": "",
        "parent_cycle_id": None,
    }
    proposal.update(overrides)
    return proposal


def test_run_daily_cycle_sample_is_not_the_fixed_first_n_tickers(monkeypatch):
    """2026-09-01 fix: run_daily_cycle() used to always slice tickers[:sample_size]
    -- the literal first N tickers of watchlist.txt, unrotated across every
    real cycle. Confirmed via real Mongo data that all 5 real cycles so far
    tested the identical, extremely homogeneous first-40 slice (NVDA/AAPL/
    MSFT/AVGO/MU/AMD/...), a highly plausible full explanation for 5
    straight zero-trade cycles independent of the rule itself. Now
    date-seeded random.Random.sample() instead."""
    tickers = [f"T{i:03d}" for i in range(200)]  # ordered so a fixed-slice bug is obvious

    captured = {}

    def fake_fetch_and_backtest(sample_tickers, rule, config, log=print):
        captured["sample"] = sample_tickers
        return {"real": {"sharpe_like": None, "win_rate": None}, "random": {"sharpe_like": None, "win_rate": None}, "n_tickers": len(sample_tickers)}

    monkeypatch.setattr(lsr, "_fetch_and_backtest", fake_fetch_and_backtest)
    monkeypatch.setattr(lsr.storage, "get_recent_cycles", lambda limit=10: [])
    monkeypatch.setattr(lsr.storage, "write_cycle", lambda doc: 1)

    lsr.run_daily_cycle(tickers, sample_size=40, propose_fn=lambda recent: _fake_proposal())

    assert captured["sample"] != tickers[:40], "sample is still the fixed first-N slice, not randomized"
    assert len(captured["sample"]) == 40
    assert len(set(captured["sample"])) == 40  # no duplicates
    assert set(captured["sample"]) <= set(tickers)


def test_run_daily_cycle_sample_is_reproducible_within_the_same_day(monkeypatch):
    """Date-seeded, not call-seeded -- re-running the same day's cycle twice
    (e.g. while debugging a crash) must sample the SAME tickers, not a
    different random draw each time."""
    tickers = [f"T{i:03d}" for i in range(200)]
    samples = []

    def fake_fetch_and_backtest(sample_tickers, rule, config, log=print):
        samples.append(list(sample_tickers))
        return {"real": {"sharpe_like": None, "win_rate": None}, "random": {"sharpe_like": None, "win_rate": None}, "n_tickers": len(sample_tickers)}

    monkeypatch.setattr(lsr, "_fetch_and_backtest", fake_fetch_and_backtest)
    monkeypatch.setattr(lsr.storage, "get_recent_cycles", lambda limit=10: [])
    monkeypatch.setattr(lsr.storage, "write_cycle", lambda doc: 1)

    lsr.run_daily_cycle(tickers, sample_size=40, propose_fn=lambda recent: _fake_proposal())
    lsr.run_daily_cycle(tickers, sample_size=40, propose_fn=lambda recent: _fake_proposal())

    assert samples[0] == samples[1]
