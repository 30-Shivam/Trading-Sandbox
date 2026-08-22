"""llm_agent.audit_verdict() -- adversarial second-pass review of an
already-produced evaluate_ticker() verdict, distinct from _resolve_dual()'s
own PARALLEL dual-provider consensus (see test_llm_resolve_dual.py).
_parse_audit_response()/_build_audit_prompt() are pure functions and
directly testable; audit_verdict() itself makes real API calls (same
"not directly tested, only its pure building blocks are" precedent as
call_gemini()/call_groq() -- see test_llm_resolve_dual.py's own docstring).
"""
import llm_agent


def test_parse_audit_response_valid_pass():
    text = '{"audit_result": "PASS", "audit_notes": "Rationale is fully supported by the given data."}'
    result = llm_agent._parse_audit_response(text)
    assert result == {"audit_result": "PASS", "audit_notes": "Rationale is fully supported by the given data."}


def test_parse_audit_response_valid_fail():
    text = '{"audit_result": "FAIL", "audit_notes": "Rationale cites a 20% earnings beat not present in the data."}'
    result = llm_agent._parse_audit_response(text)
    assert result["audit_result"] == "FAIL"
    assert "earnings beat" in result["audit_notes"]


def test_parse_audit_response_invalid_json():
    assert llm_agent._parse_audit_response("not json") is None


def test_parse_audit_response_not_a_dict():
    assert llm_agent._parse_audit_response("[1, 2, 3]") is None


def test_parse_audit_response_bad_audit_result_value():
    assert llm_agent._parse_audit_response('{"audit_result": "MAYBE", "audit_notes": "x"}') is None


def test_parse_audit_response_missing_audit_result():
    assert llm_agent._parse_audit_response('{"audit_notes": "x"}') is None


def test_parse_audit_response_blank_notes():
    assert llm_agent._parse_audit_response('{"audit_result": "PASS", "audit_notes": ""}') is None
    assert llm_agent._parse_audit_response('{"audit_result": "PASS", "audit_notes": "   "}') is None


def test_parse_audit_response_notes_not_a_string():
    assert llm_agent._parse_audit_response('{"audit_result": "PASS", "audit_notes": 42}') is None


def test_parse_audit_response_strips_whitespace():
    result = llm_agent._parse_audit_response('{"audit_result": "PASS", "audit_notes": "  looks fine  "}')
    assert result["audit_notes"] == "looks fine"


def _sample_context():
    return {
        "last_close": 123.45, "rsi": 42.0, "atr": 3.1,
        "mechanical_scores": {"ma_crossover": 72.5},
        "catalyst_warning": False, "next_earnings_date": None,
        "headlines": ["Company X announces new product line"],
        "fundamentals": {"sector": "Technology"},
    }


def test_build_audit_prompt_embeds_the_verdict_under_review():
    verdict = {"decision": "Buy", "confidence": 85, "rationale": "Strong momentum and a clean earnings beat."}
    prompt = llm_agent._build_audit_prompt("ACME", _sample_context(), verdict)
    assert "ACME" in prompt
    assert "Strong momentum and a clean earnings beat." in prompt
    assert 'decision="Buy"' in prompt
    assert "confidence=85" in prompt


def test_build_audit_prompt_instructs_report_only_never_rewrite():
    verdict = {"decision": "Buy", "confidence": 70, "rationale": "r"}
    prompt = llm_agent._build_audit_prompt("ACME", _sample_context(), verdict)
    assert "may NOT change the decision" in prompt


def test_build_audit_prompt_requests_the_expected_json_shape():
    verdict = {"decision": "Buy", "confidence": 70, "rationale": "r"}
    prompt = llm_agent._build_audit_prompt("ACME", _sample_context(), verdict)
    assert '"audit_result"' in prompt
    assert '"audit_notes"' in prompt
