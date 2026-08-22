"""Time-based holdout check (improvements.txt item 87) -- does an edge hold
up consistently across calendar years, or is it concentrated in one
stretch? A different question from the ticker-holdout split (item 69,
generalization across TICKERS) or k_ratio (item 77, smoothness within the
one ordering that actually happened).
"""
import datetime

from swingtrade.backtest import summarize_by_period, summarize_trades


def _trade(entry_date, pnl_pct, status="WIN"):
    return {"entry_date": entry_date, "pnl_pct": pnl_pct, "status": status}


def test_buckets_by_calendar_year():
    trades = [
        _trade(datetime.date(2022, 3, 1), 5.0),
        _trade(datetime.date(2022, 6, 1), -2.0, status="LOSS"),
        _trade(datetime.date(2023, 1, 15), 3.0),
    ]
    result = summarize_by_period(trades)
    assert set(result.keys()) == {"2022", "2023"}
    assert result["2022"]["trade_count"] == 2
    assert result["2023"]["trade_count"] == 1


def test_per_year_summary_matches_manual_split():
    trades_2022 = [_trade(datetime.date(2022, 1, 1), 5.0), _trade(datetime.date(2022, 2, 1), -2.0, status="LOSS")]
    trades_2023 = [_trade(datetime.date(2023, 1, 1), 3.0)]
    result = summarize_by_period(trades_2022 + trades_2023)
    assert result["2022"] == summarize_trades(trades_2022)
    assert result["2023"] == summarize_trades(trades_2023)


def test_trades_missing_entry_date_are_skipped_not_crashed_on():
    trades = [
        {"pnl_pct": 5.0, "status": "WIN"},  # no entry_date at all
        _trade(datetime.date(2022, 1, 1), 3.0),
    ]
    result = summarize_by_period(trades)
    assert set(result.keys()) == {"2022"}
    assert result["2022"]["trade_count"] == 1


def test_single_year_input_returns_single_key_dict():
    trades = [_trade(datetime.date(2024, 1, 1), 5.0), _trade(datetime.date(2024, 12, 1), 1.0)]
    result = summarize_by_period(trades)
    assert list(result.keys()) == ["2024"]


def test_empty_trades_returns_empty_dict():
    assert summarize_by_period([]) == {}


def test_custom_summarize_fn_is_used_instead_of_default():
    calls = []

    def fake_summarize(trades):
        calls.append(len(trades))
        return {"trade_count": len(trades)}

    trades = [_trade(datetime.date(2022, 1, 1), 5.0), _trade(datetime.date(2023, 1, 1), 3.0)]
    result = summarize_by_period(trades, summarize_fn=fake_summarize)
    assert calls == [1, 1]
    assert result == {"2022": {"trade_count": 1}, "2023": {"trade_count": 1}}


def test_years_are_returned_in_chronological_order():
    trades = [
        _trade(datetime.date(2024, 1, 1), 1.0),
        _trade(datetime.date(2021, 1, 1), 1.0),
        _trade(datetime.date(2023, 1, 1), 1.0),
    ]
    result = summarize_by_period(trades)
    assert list(result.keys()) == ["2021", "2023", "2024"]
