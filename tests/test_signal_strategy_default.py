from datetime import datetime, timezone

import storage.signals as signals


def _row():
    return {"Ticker": "AAA", "As_Of": "2026-10-21", "Signal": "Buy", "Trade_Score": 61.0, "Buy_Price": 10.0,
            "Sell_Price": 12.0, "Stop_Loss": 9.0, "RRR": 2.0, "ATR": 0.5, "Last_Close": 10.0,
            "Distance_to_Buy_Pct": 0.0, "Shares_To_Buy": 0.0, "Est_Cost": 0.0,
            "Catalyst_Warning": False}


def test_missing_or_none_strategy_defaults_to_rsi_not_null():
    now = datetime.now(timezone.utc)
    assert signals._build_document(_row(), {}, now)["strategy"] == "rsi"
    assert signals._build_document(_row(), {"strategy": None}, now)["strategy"] == "rsi"
    assert signals._build_document(_row(), {"strategy": "value_rank"}, now)["strategy"] == "value_rank"
