"""dip_buy_analyzer.confirmable_holdings() -- pure function, dict fixtures,
no Streamlit/Mongo involved. Mirrors tests/test_review_positions.py's own
convention for a helper split out of dashboard/script logic for direct
testing.
"""
import dip_buy_analyzer


def _signal(ticker, date, strategy="rsi", buy_price=100.0):
    return {"ticker": ticker, "signal_date": date, "strategy": strategy, "buy_price": buy_price}


def test_holding_with_matching_pending_signal_is_confirmable():
    saved_holdings = {"NVDA": {"amount": 1000.0, "avg_cost": 120.0}}
    pending = [_signal("NVDA", "2026-08-20")]
    result = dip_buy_analyzer.confirmable_holdings(saved_holdings, pending)
    assert list(result.keys()) == ["NVDA"]
    assert result["NVDA"] == [_signal("NVDA", "2026-08-20")]


def test_holding_without_avg_cost_is_never_confirmable():
    saved_holdings = {"NVDA": {"amount": 1000.0, "avg_cost": None}}
    pending = [_signal("NVDA", "2026-08-20")]
    assert dip_buy_analyzer.confirmable_holdings(saved_holdings, pending) == {}


def test_holding_with_no_pending_signal_is_not_confirmable():
    saved_holdings = {"NVDA": {"amount": 1000.0, "avg_cost": 120.0}}
    assert dip_buy_analyzer.confirmable_holdings(saved_holdings, []) == {}


def test_pending_signal_for_unheld_ticker_is_ignored():
    saved_holdings = {"NVDA": {"amount": 1000.0, "avg_cost": 120.0}}
    pending = [_signal("TSLA", "2026-08-20")]
    assert dip_buy_analyzer.confirmable_holdings(saved_holdings, pending) == {}


def test_multiple_pending_signals_for_one_ticker_all_grouped():
    saved_holdings = {"NVDA": {"amount": 1000.0, "avg_cost": 120.0}}
    pending = [_signal("NVDA", "2026-08-20"), _signal("NVDA", "2026-08-15", strategy="breakout")]
    result = dip_buy_analyzer.confirmable_holdings(saved_holdings, pending)
    assert len(result["NVDA"]) == 2


def test_multiple_holdings_only_ones_with_matches_included():
    saved_holdings = {
        "NVDA": {"amount": 1000.0, "avg_cost": 120.0},
        "AAPL": {"amount": 500.0, "avg_cost": 200.0},
        "MSFT": {"amount": 300.0, "avg_cost": None},
    }
    pending = [_signal("NVDA", "2026-08-20")]
    result = dip_buy_analyzer.confirmable_holdings(saved_holdings, pending)
    assert list(result.keys()) == ["NVDA"]
