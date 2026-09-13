"""audit_dividend_drag() -- quantifies a real, previously entirely
unexamined backtest-realism gap (2026-09-13): pnl_pct is computed purely
from price movement, never crediting a real dividend paid while a
position was actually held. Slippage/commission already model costs you
didn't actually avoid; this is the mirror-image, always-understates-real-
returns gap for dividend-paying tickers. PURE MEASUREMENT ONLY -- never
modifies pnl_pct itself.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- same discipline every other audit
this session established.
"""
import datetime

import pandas as pd

from swingtrade.backtest import audit_dividend_drag

D = datetime.date


def _trade(ticker, entry_date, exit_date, buy_price, pnl_pct=1.0, status="WIN"):
    return {
        "ticker": ticker, "entry_date": entry_date, "exit_date": exit_date,
        "buy_price": buy_price, "pnl_pct": pnl_pct, "status": status,
    }


def _divs(pairs, tz=None):
    """pairs: [(date_str, amount), ...]"""
    idx = pd.DatetimeIndex([p[0] for p in pairs])
    if tz:
        idx = idx.tz_localize(tz)
    return pd.Series([p[1] for p in pairs], index=idx)


def test_empty_trades_returns_none_fields():
    result = audit_dividend_drag([], {})
    assert result["n_trades"] == 0
    assert result["pct_affected"] is None


def test_no_dividend_history_means_zero_missed():
    trades = [_trade("AAPL", D(2024, 1, 1), D(2024, 1, 15), 100.0)]
    result = audit_dividend_drag(trades, {})
    assert result["n_trades"] == 1
    assert result["n_affected"] == 0
    assert result["avg_missed_pct_all_trades"] == 0.0


def test_dividend_strictly_after_entry_and_at_or_before_exit_counts():
    # Ex-div date 2024-01-10 falls within (entry=1-1, exit=1-15] -- counted.
    trades = [_trade("BP", D(2024, 1, 1), D(2024, 1, 15), 100.0)]
    dividend_history = {"BP": _divs([("2024-01-10", 2.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_affected"] == 1
    # 2.0 / 100.0 * 100 = 2.0%
    assert result["avg_missed_pct_all_trades"] == 2.0
    assert result["avg_missed_pct_affected_only"] == 2.0


def test_dividend_on_entry_date_itself_does_not_count():
    # Same-day entry ON the ex-date already missed it (you must have held
    # BEFORE the ex-date to receive the payment) -- strictly-after check.
    trades = [_trade("BP", D(2024, 1, 10), D(2024, 1, 20), 100.0)]
    dividend_history = {"BP": _divs([("2024-01-10", 2.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_affected"] == 0


def test_dividend_on_exit_date_itself_counts():
    # Held THROUGH the ex-date (exit is inclusive) -- counts.
    trades = [_trade("BP", D(2024, 1, 1), D(2024, 1, 10), 100.0)]
    dividend_history = {"BP": _divs([("2024-01-10", 2.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_affected"] == 1


def test_dividend_outside_window_does_not_count():
    trades = [_trade("BP", D(2024, 1, 1), D(2024, 1, 5), 100.0)]
    dividend_history = {"BP": _divs([("2024-02-01", 2.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_affected"] == 0
    assert result["avg_missed_pct_all_trades"] == 0.0


def test_multiple_dividends_in_window_are_summed():
    # Two ex-div dates both fall within a long holding window -- summed.
    trades = [_trade("BP", D(2024, 1, 1), D(2024, 6, 1), 100.0)]
    dividend_history = {"BP": _divs([("2024-02-15", 1.5), ("2024-05-15", 1.5)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["avg_missed_pct_all_trades"] == 3.0


def test_tz_aware_dividend_index_handled_without_crashing():
    # yfinance's own .dividends index is tz-aware (America/New_York) --
    # must not crash comparing against this project's plain-date fields.
    trades = [_trade("BP", D(2024, 1, 1), D(2024, 1, 15), 100.0)]
    dividend_history = {"BP": _divs([("2024-01-10", 2.0)], tz="America/New_York")}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_affected"] == 1
    assert result["avg_missed_pct_all_trades"] == 2.0


def test_pct_affected_and_averages_across_mixed_trades():
    # 1 of 2 trades affected: averages must differ between "all trades"
    # (includes the 0% unaffected one) and "affected only".
    trades = [
        _trade("BP", D(2024, 1, 1), D(2024, 1, 15), 100.0),
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 15), 200.0),
    ]
    dividend_history = {"BP": _divs([("2024-01-10", 4.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_trades"] == 2
    assert result["n_affected"] == 1
    assert result["pct_affected"] == 50.0
    # BP: 4/100*100=4%, AAPL: 0% -> avg across all 2 = 2.0%
    assert result["avg_missed_pct_all_trades"] == 2.0
    # avg across just the 1 affected trade = 4.0%
    assert result["avg_missed_pct_affected_only"] == 4.0


def test_ticker_missing_from_dividend_history_treated_as_zero():
    trades = [
        _trade("BP", D(2024, 1, 1), D(2024, 1, 15), 100.0),
        _trade("UNKNOWN_TICKER", D(2024, 1, 1), D(2024, 1, 15), 100.0),
    ]
    dividend_history = {"BP": _divs([("2024-01-10", 2.0)])}
    result = audit_dividend_drag(trades, dividend_history)
    assert result["n_trades"] == 2
    assert result["n_affected"] == 1
