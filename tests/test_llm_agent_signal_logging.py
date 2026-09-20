"""ingest.run_llm_agent()'s Trade_Signals logging -- 2026-09-20 real bug
fix: "Avoid" verdicts used to be silently discarded (gate was Buy/Hold
only), so the IC-tracking system could never learn whether llm_agent's
bearish calls were actually right. And Trade_Score used to be the RAW
confidence regardless of decision, so a confident "Hold" looked identical
to a confident "Buy" in the ranking, and a (hypothetically) logged "Avoid"
would have shown a HIGH trade_score for a BEARISH call. Both fixed by
reusing best_ideas.llm_bullishness_score() instead of logging raw
confidence, and widening the logging gate to include "Avoid".

No live network/Mongo -- every external call monkeypatched, same
convention as tests/test_review_positions.py.
"""
import pandas as pd
import pytest

import ingest
import best_ideas


def _mechanical_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "Ticker": "TEST", "Signal": "Buy", "Trade_Score": 70.0,
        "As_Of": pd.Timestamp("2026-09-20").date(), "Last_Close": 100.0,
        "RSI": 45.0, "ATR": 2.0, "Catalyst_Warning": False,
        "Next_Earnings_Date": None, "Currency": "USD",
    }])


def _verdict(decision: str, confidence: float) -> dict:
    return {
        "decision": decision, "confidence": confidence, "rationale": "test",
        "news_sentiment": "neutral", "provider_agreement": True,
        "secondary_provider": None, "secondary_decision": None, "secondary_confidence": None,
    }


@pytest.fixture(autouse=True)
def _stub_common(monkeypatch):
    monkeypatch.setattr(ingest.llm_agent, "is_available", lambda: True)
    monkeypatch.setattr(ingest.market_data, "get_macro_snapshot", lambda: {})
    monkeypatch.setattr(ingest.market_data, "get_multi_headlines", lambda t: [])
    monkeypatch.setattr(ingest.market_data, "get_qualitative_snapshot", lambda t: None)
    monkeypatch.setattr(ingest.llm_agent, "audit_verdict", lambda *a, **k: None)

    class _FakeTicker:
        def __init__(self, ticker):
            self.info = {}
    monkeypatch.setattr(ingest.yf, "Ticker", _FakeTicker)


def test_avoid_verdict_is_now_logged_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(ingest.llm_agent, "evaluate_ticker", lambda t, c, variant="balanced": _verdict("Avoid", 80.0))
    logged_frames = []
    monkeypatch.setattr(ingest.storage, "log_trade_signals", lambda df, cfg: logged_frames.append(df) or {"actionable": 0, "research": len(df)})

    ingest.run_llm_agent({"mechanical": _mechanical_frame()}, ingest.swingtrade.DEFAULT_CONFIG)

    assert logged_frames, "Avoid verdict should have produced a logged row, not been silently dropped"
    row = logged_frames[0].iloc[0]
    assert row["Signal"] == "Watch"  # not in the shared vocabulary, same tier as Hold


def test_avoid_verdict_gets_a_bearish_not_raw_confidence_trade_score(monkeypatch):
    monkeypatch.setattr(ingest.llm_agent, "evaluate_ticker", lambda t, c, variant="balanced": _verdict("Avoid", 80.0))
    logged_frames = []
    monkeypatch.setattr(ingest.storage, "log_trade_signals", lambda df, cfg: logged_frames.append(df) or {"actionable": 0, "research": len(df)})

    ingest.run_llm_agent({"mechanical": _mechanical_frame()}, ingest.swingtrade.DEFAULT_CONFIG)

    trade_score = logged_frames[0].iloc[0]["Trade_Score"]
    expected = best_ideas.llm_bullishness_score("Avoid", 80.0)
    assert trade_score == expected
    assert trade_score < 50.0  # a confident Avoid must read as BEARISH, not a high raw-confidence number
    assert trade_score != 80.0  # the old bug: raw confidence logged verbatim regardless of direction


def test_hold_verdict_scores_flat_50_regardless_of_confidence(monkeypatch):
    monkeypatch.setattr(ingest.llm_agent, "evaluate_ticker", lambda t, c, variant="balanced": _verdict("Hold", 95.0))
    logged_frames = []
    monkeypatch.setattr(ingest.storage, "log_trade_signals", lambda df, cfg: logged_frames.append(df) or {"actionable": 0, "research": len(df)})

    ingest.run_llm_agent({"mechanical": _mechanical_frame()}, ingest.swingtrade.DEFAULT_CONFIG)

    row = logged_frames[0].iloc[0]
    assert row["Signal"] == "Watch"
    assert row["Trade_Score"] == 50.0  # NOT 95.0 (the old bug) -- a confident Hold is still genuinely neutral


def test_buy_verdict_scores_above_50_and_is_logged_as_buy(monkeypatch):
    monkeypatch.setattr(ingest.llm_agent, "evaluate_ticker", lambda t, c, variant="balanced": _verdict("Buy", 80.0))
    logged_frames = []
    monkeypatch.setattr(ingest.storage, "log_trade_signals", lambda df, cfg: logged_frames.append(df) or {"actionable": len(df), "research": 0})

    ingest.run_llm_agent({"mechanical": _mechanical_frame()}, ingest.swingtrade.DEFAULT_CONFIG)

    row = logged_frames[0].iloc[0]
    assert row["Signal"] == "Buy"
    assert row["Trade_Score"] == best_ideas.llm_bullishness_score("Buy", 80.0)
    assert row["Trade_Score"] > 50.0
