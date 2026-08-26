"""review_positions._detect_flips() / _build_flip_notification() /
_llm_hold_opinion() -- pure functions and dispatch logic, dict fixtures,
no live Mongo/network/Discord calls, LLM calls monkeypatched. Mirrors
tests/test_settle_trades_notifications.py's own convention.
"""
import review_positions


def _row(ticker, recommendation, avg_cost=100.0, last_close=95.0, pnl_pct=-5.0, sell_price=None, stop_loss=None):
    return {
        "Ticker": ticker, "Recommendation": recommendation,
        "Avg_Cost": avg_cost, "Last_Close": last_close, "Unrealized_PnL_Pct": pnl_pct,
        "Sell_Price": sell_price, "Stop_Loss": stop_loss,
    }


# ---- _detect_flips ----

def test_hold_to_sell_is_a_flip():
    results = [_row("NVDA", "SELL (stop breached)")]
    flips, new_state = review_positions._detect_flips(results, {"NVDA": "HOLD"})
    assert [f["Ticker"] for f in flips] == ["NVDA"]
    assert new_state == {"NVDA": "SELL (stop breached)"}


def test_never_seen_before_defaults_to_hold_and_flips():
    results = [_row("TSLA", "SELL (target hit)")]
    flips, new_state = review_positions._detect_flips(results, {})
    assert [f["Ticker"] for f in flips] == ["TSLA"]


def test_hold_to_hold_is_not_a_flip():
    results = [_row("MSFT", "HOLD")]
    flips, _ = review_positions._detect_flips(results, {"MSFT": "HOLD"})
    assert flips == []


def test_repeated_same_sell_reason_is_not_a_flip():
    results = [_row("AMD", "SELL (stop breached)")]
    flips, _ = review_positions._detect_flips(results, {"AMD": "SELL (stop breached)"})
    assert flips == []


def test_changed_sell_reason_is_still_a_flip():
    results = [_row("AMD", "SELL (stop breached)")]
    flips, _ = review_positions._detect_flips(results, {"AMD": "SELL (target hit)"})
    assert [f["Ticker"] for f in flips] == ["AMD"]


def test_sell_back_to_hold_is_not_flagged_as_a_flip():
    """Only flips INTO SELL are alerted -- a recovery back to HOLD is real
    and worth persisting in new_state, but not itself an alert-worthy
    event per this feature's scope."""
    results = [_row("AMD", "HOLD")]
    flips, new_state = review_positions._detect_flips(results, {"AMD": "SELL (target hit)"})
    assert flips == []
    assert new_state == {"AMD": "HOLD"}


def test_new_state_covers_every_result_regardless_of_flip():
    results = [_row("AAA", "HOLD"), _row("BBB", "SELL (stop breached)")]
    _, new_state = review_positions._detect_flips(results, {})
    assert new_state == {"AAA": "HOLD", "BBB": "SELL (stop breached)"}


# ---- _build_flip_notification ----

def test_empty_flips_returns_none():
    assert review_positions._build_flip_notification([]) is None


def test_single_flip_renders_header_and_line():
    flips = [_row("NVDA", "SELL (stop breached)", avg_cost=120.0, last_close=105.0, pnl_pct=-12.5)]
    message = review_positions._build_flip_notification(flips)
    assert "**Position Review: 1 holding(s) flipped to SELL**" in message
    assert "NVDA: SELL (stop breached) (avg_cost 120.00, last 105.00, -12.50%)" in message


def test_multiple_flips_sorted_worst_pnl_first():
    flips = [
        _row("AAA", "SELL (target hit)", pnl_pct=8.0),
        _row("BBB", "SELL (stop breached)", pnl_pct=-15.0),
        _row("CCC", "SELL (stop breached)", pnl_pct=-3.0),
    ]
    message = review_positions._build_flip_notification(flips)
    lines = message.splitlines()
    assert lines[0] == "**Position Review: 3 holding(s) flipped to SELL**"
    tickers_in_order = [line.split(":")[0] for line in lines[1:]]
    assert tickers_in_order == ["BBB", "CCC", "AAA"]


def test_llm_opinion_line_included_when_present():
    flips = [_row("NVDA", "SELL (stop breached)", pnl_pct=-12.5)]
    opinions = {"NVDA": {"action": "Cut Loss", "confidence": 82.0, "rationale": "Real deterioration, no bounce evidence."}}
    message = review_positions._build_flip_notification(flips, opinions)
    assert "LLM second opinion: **Cut Loss** (confidence 82/100) -- Real deterioration, no bounce evidence." in message


def test_missing_llm_opinion_omits_line_without_error():
    flips = [_row("NVDA", "SELL (stop breached)")]
    message = review_positions._build_flip_notification(flips, {})
    assert "LLM second opinion" not in message


# ---- _llm_hold_opinion ----

def test_llm_hold_opinion_dispatches_evaluate_holding_for_target_hit(monkeypatch):
    captured = {}

    def fake_evaluate_holding(ticker, context):
        captured["ticker"] = ticker
        captured["context"] = context
        return {"action": "Take Profit", "confidence": 70.0, "rationale": "r"}

    monkeypatch.setattr(review_positions.llm_agent, "evaluate_holding", fake_evaluate_holding)
    monkeypatch.setattr(review_positions.market_data, "get_multi_headlines", lambda t: [])
    monkeypatch.setattr(review_positions.market_data, "get_qualitative_snapshot", lambda t: None)

    row = _row("NVDA", "SELL (target hit)", sell_price=150.0)
    result = review_positions._llm_hold_opinion(row, macro_snapshot={})
    assert result["action"] == "Take Profit"
    assert captured["ticker"] == "NVDA"
    assert captured["context"]["sell_price"] == 150.0


def test_llm_hold_opinion_dispatches_evaluate_stop_breach_for_stop_breach(monkeypatch):
    captured = {}

    def fake_evaluate_stop_breach(ticker, context):
        captured["ticker"] = ticker
        captured["context"] = context
        return {"action": "Cut Loss", "confidence": 85.0, "rationale": "r"}

    monkeypatch.setattr(review_positions.llm_agent, "evaluate_stop_breach", fake_evaluate_stop_breach)
    monkeypatch.setattr(review_positions.market_data, "get_multi_headlines", lambda t: [])
    monkeypatch.setattr(review_positions.market_data, "get_qualitative_snapshot", lambda t: None)

    row = _row("DDOG", "SELL (stop breached)", stop_loss=225.0)
    result = review_positions._llm_hold_opinion(row, macro_snapshot={})
    assert result["action"] == "Cut Loss"
    assert captured["ticker"] == "DDOG"
    assert captured["context"]["stop_loss"] == 225.0


def test_llm_hold_opinion_returns_none_for_hold_row():
    row = _row("MSFT", "HOLD")
    assert review_positions._llm_hold_opinion(row, macro_snapshot={}) is None
