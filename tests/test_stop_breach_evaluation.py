"""llm_agent._parse_stop_breach_response()/_build_stop_breach_prompt() --
pure functions, directly testable; evaluate_stop_breach() itself makes
real API calls (same "not directly tested, only its pure building blocks
are" precedent as call_gemini()/call_groq() -- see
tests/test_llm_audit_verdict.py's own docstring, and
tests/test_llm_resolve_dual.py for _resolve_dual() coverage of
CONSERVATIVE_ORDER_STOP_ACTION).
"""
import llm_agent


def test_parse_stop_breach_response_valid_cut_loss():
    text = '{"action": "Cut Loss", "confidence": 80, "news_sentiment": "Bearish", "rationale": "Real deterioration, no reason to expect recovery."}'
    result = llm_agent._parse_stop_breach_response(text)
    assert result["action"] == "Cut Loss"
    assert result["confidence"] == 80.0
    assert result["news_sentiment"] == "Bearish"


def test_parse_stop_breach_response_valid_hold_through():
    text = '{"action": "Hold Through", "confidence": 55, "news_sentiment": "Neutral", "rationale": "Broad market selloff, not company-specific."}'
    result = llm_agent._parse_stop_breach_response(text)
    assert result["action"] == "Hold Through"


def test_parse_stop_breach_response_invalid_json():
    assert llm_agent._parse_stop_breach_response("not json") is None


def test_parse_stop_breach_response_not_a_dict():
    assert llm_agent._parse_stop_breach_response("[1, 2, 3]") is None


def test_parse_stop_breach_response_bad_action_value():
    """Rejects the profit-side vocabulary too -- "Take Profit" is not a
    valid stop-breach action, confirming the two schemas don't silently
    cross-validate each other."""
    text = '{"action": "Take Profit", "confidence": 80, "news_sentiment": "Bearish", "rationale": "x"}'
    assert llm_agent._parse_stop_breach_response(text) is None


def test_parse_stop_breach_response_confidence_out_of_range():
    text = '{"action": "Cut Loss", "confidence": 150, "news_sentiment": "Bearish", "rationale": "x"}'
    assert llm_agent._parse_stop_breach_response(text) is None


def test_parse_stop_breach_response_missing_rationale():
    text = '{"action": "Cut Loss", "confidence": 80, "news_sentiment": "Bearish"}'
    assert llm_agent._parse_stop_breach_response(text) is None


def test_parse_stop_breach_response_bad_sentiment():
    text = '{"action": "Cut Loss", "confidence": 80, "news_sentiment": "Excited", "rationale": "x"}'
    assert llm_agent._parse_stop_breach_response(text) is None


def _mk_context(**overrides):
    context = {
        "avg_cost": 100.0, "last_close": 88.0, "stop_loss": 90.0,
        "unrealized_pnl_pct": -12.0, "headlines": ["Company X misses earnings"],
    }
    context.update(overrides)
    return context


def test_build_stop_breach_prompt_states_the_breach_and_defaults_to_cut_loss():
    prompt = llm_agent._build_stop_breach_prompt("XYZ", _mk_context())
    assert "BREACHED" in prompt
    assert "stop-loss" in prompt.lower()
    assert '"Cut Loss"' in prompt
    assert "default to" in prompt.lower() and "Cut Loss" in prompt


def test_build_stop_breach_prompt_states_stricter_asymmetry_than_profit_case():
    """The whole point of this being a separate prompt from
    _build_holding_prompt() -- the stricter downside-risk framing must
    actually be present in the text sent to the model, not just in the
    docstring."""
    prompt = llm_agent._build_stop_breach_prompt("XYZ", _mk_context())
    assert "compounding" in prompt.lower() or "opportunity" in prompt.lower()


def test_build_stop_breach_prompt_includes_headlines():
    prompt = llm_agent._build_stop_breach_prompt("XYZ", _mk_context())
    assert "Company X misses earnings" in prompt


def test_build_stop_breach_prompt_handles_no_headlines():
    prompt = llm_agent._build_stop_breach_prompt("XYZ", _mk_context(headlines=[]))
    assert "(none available)" in prompt
