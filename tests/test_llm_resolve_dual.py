"""Promoted from this session's own ad-hoc verification script into
permanent coverage -- llm_agent._resolve_dual() combines two LLM
providers' results (agree / disagree / only-one-available); a real
disagreement can't be reliably forced from live models on demand, so this
is the only reliable way to exercise every branch.

2026-08-22: _resolve_dual() gained an explicit `secondary_provider_name`
param (previously hardcoded "groq") when the secondary slot became a
Groq-then-OpenRouter fallback pool -- see call_secondary(). Every existing
test below now passes "groq" explicitly for that param, preserving their
original intent (they're all testing Groq-specific scenarios); new tests
cover the "openrouter" label and call_secondary()'s own fallback
sequencing.
"""
import llm_agent

DECISION_KEY, DECISION_ORDER = "decision", llm_agent.CONSERVATIVE_ORDER_DECISION
HOLD_KEY, HOLD_ORDER = "action", llm_agent.CONSERVATIVE_ORDER_HOLD_ACTION


def _mk(decision_key, decision, confidence, rationale="r", sentiment="Neutral"):
    return {decision_key: decision, "confidence": confidence, "rationale": rationale, "news_sentiment": sentiment}


def test_only_gemini_available():
    g = _mk(DECISION_KEY, "Buy", 80.0)
    result = llm_agent._resolve_dual(g, None, None, DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Buy"
    assert result["provider_agreement"] is None
    assert result["secondary_provider"] is None


def test_only_groq_available():
    q = _mk(DECISION_KEY, "Hold", 55.0)
    result = llm_agent._resolve_dual(None, q, "groq", DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Hold"
    assert result["provider_agreement"] is None


def test_neither_available_returns_none():
    assert llm_agent._resolve_dual(None, None, None, DECISION_KEY, DECISION_ORDER) is None


def test_agreement_averages_confidence():
    g = _mk(DECISION_KEY, "Buy", 80.0, rationale="gemini says buy")
    q = _mk(DECISION_KEY, "Buy", 60.0, rationale="groq says buy")
    result = llm_agent._resolve_dual(g, q, "groq", DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Buy"
    assert result["confidence"] == 70.0
    assert result["provider_agreement"] is True
    assert result["secondary_provider"] == "groq"
    assert result["secondary_confidence"] == 60.0


def test_agreement_labels_openrouter_when_openrouter_answered():
    g = _mk(DECISION_KEY, "Buy", 80.0, rationale="gemini says buy")
    o = _mk(DECISION_KEY, "Buy", 60.0, rationale="openrouter says buy")
    result = llm_agent._resolve_dual(g, o, "openrouter", DECISION_KEY, DECISION_ORDER)
    assert result["provider_agreement"] is True
    assert result["secondary_provider"] == "openrouter"


def test_disagreement_picks_conservative_gemini_buy_vs_groq_avoid():
    g = _mk(DECISION_KEY, "Buy", 90.0, rationale="gemini bullish")
    q = _mk(DECISION_KEY, "Avoid", 70.0, rationale="groq bearish")
    result = llm_agent._resolve_dual(g, q, "groq", DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Avoid"
    assert result["confidence"] == 70.0  # the conservative provider's own confidence, NOT averaged
    assert result["provider_agreement"] is False
    assert result["secondary_provider"] == "gemini"
    assert result["secondary_decision"] == "Buy"
    assert "Gemini:" in result["rationale"] and "Groq:" in result["rationale"]


def test_disagreement_picks_conservative_gemini_avoid_vs_groq_buy():
    g = _mk(DECISION_KEY, "Avoid", 65.0, rationale="gemini cautious")
    q = _mk(DECISION_KEY, "Buy", 85.0, rationale="groq bullish")
    result = llm_agent._resolve_dual(g, q, "groq", DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Avoid"
    assert result["confidence"] == 65.0
    assert result["secondary_provider"] == "groq"


def test_disagreement_labels_openrouter_when_openrouter_answered():
    g = _mk(DECISION_KEY, "Avoid", 65.0, rationale="gemini cautious")
    o = _mk(DECISION_KEY, "Buy", 85.0, rationale="openrouter bullish")
    result = llm_agent._resolve_dual(g, o, "openrouter", DECISION_KEY, DECISION_ORDER)
    assert result["decision"] == "Avoid"
    assert result["secondary_provider"] == "openrouter"
    assert "OpenRouter:" in result["rationale"]


def test_holding_disagreement_take_profit_wins():
    g = _mk(HOLD_KEY, "Hold For More", 75.0, rationale="gemini: let it run")
    q = _mk(HOLD_KEY, "Take Profit", 60.0, rationale="groq: lock it in")
    result = llm_agent._resolve_dual(g, q, "groq", HOLD_KEY, HOLD_ORDER)
    assert result["action"] == "Take Profit"
    assert result["confidence"] == 60.0
    assert result["secondary_decision"] == "Hold For More"


def test_holding_agreement_averages_confidence():
    g = _mk(HOLD_KEY, "Take Profit", 70.0)
    q = _mk(HOLD_KEY, "Take Profit", 90.0)
    result = llm_agent._resolve_dual(g, q, "groq", HOLD_KEY, HOLD_ORDER)
    assert result["action"] == "Take Profit"
    assert result["confidence"] == 80.0
    assert result["provider_agreement"] is True


def test_call_secondary_returns_groq_result_when_groq_succeeds(monkeypatch):
    monkeypatch.setattr(llm_agent, "_groq_available", lambda: True)
    monkeypatch.setattr(llm_agent, "_openrouter_available", lambda: True)
    monkeypatch.setattr(llm_agent, "call_groq", lambda prompt, parse_fn=None, max_output_tokens=None: {"decision": "Buy"})
    monkeypatch.setattr(llm_agent, "call_openrouter", lambda prompt, parse_fn=None, max_output_tokens=None: {"decision": "Avoid"})
    result, provider = llm_agent.call_secondary("prompt")
    assert result == {"decision": "Buy"}
    assert provider == "groq"


def test_call_secondary_falls_back_to_openrouter_when_groq_unavailable(monkeypatch):
    monkeypatch.setattr(llm_agent, "_groq_available", lambda: False)
    monkeypatch.setattr(llm_agent, "_openrouter_available", lambda: True)
    monkeypatch.setattr(llm_agent, "call_openrouter", lambda prompt, parse_fn=None, max_output_tokens=None: {"decision": "Hold"})
    result, provider = llm_agent.call_secondary("prompt")
    assert result == {"decision": "Hold"}
    assert provider == "openrouter"


def test_call_secondary_falls_back_to_openrouter_when_groq_call_fails(monkeypatch):
    monkeypatch.setattr(llm_agent, "_groq_available", lambda: True)
    monkeypatch.setattr(llm_agent, "_openrouter_available", lambda: True)
    monkeypatch.setattr(llm_agent, "call_groq", lambda prompt, parse_fn=None, max_output_tokens=None: None)
    monkeypatch.setattr(llm_agent, "call_openrouter", lambda prompt, parse_fn=None, max_output_tokens=None: {"decision": "Hold"})
    result, provider = llm_agent.call_secondary("prompt")
    assert result == {"decision": "Hold"}
    assert provider == "openrouter"


def test_call_secondary_returns_none_none_when_both_unavailable(monkeypatch):
    monkeypatch.setattr(llm_agent, "_groq_available", lambda: False)
    monkeypatch.setattr(llm_agent, "_openrouter_available", lambda: False)
    result, provider = llm_agent.call_secondary("prompt")
    assert result is None
    assert provider is None


def test_call_secondary_returns_none_none_when_both_fail(monkeypatch):
    monkeypatch.setattr(llm_agent, "_groq_available", lambda: True)
    monkeypatch.setattr(llm_agent, "_openrouter_available", lambda: True)
    monkeypatch.setattr(llm_agent, "call_groq", lambda prompt, parse_fn=None, max_output_tokens=None: None)
    monkeypatch.setattr(llm_agent, "call_openrouter", lambda prompt, parse_fn=None, max_output_tokens=None: None)
    result, provider = llm_agent.call_secondary("prompt")
    assert result is None
    assert provider is None
