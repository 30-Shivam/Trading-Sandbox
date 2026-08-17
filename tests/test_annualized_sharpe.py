"""annualized_sharpe_like -- sharpe_like is a RAW per-trade ratio
(mean/std of pnl_pct across trades), never annualized, which reads as a
tiny number (0.02-0.15 typical in this project) next to the "Sharpe > 1.5"
figures usually quoted elsewhere, since those are almost always ANNUALIZED
(scaled by sqrt(periods per year)). This applies the standard trade-
frequency annualization convention as an additive, clearly-labeled
estimate -- not a replacement for sharpe_like, and not a literal
compounding portfolio simulation (same caveat as compute_max_drawdown()).
"""
import datetime

import swingtrade
from swingtrade.backtest import _annualize_sharpe


def _trade(entry_date, pnl_pct, status="WIN"):
    return {"entry_date": entry_date, "pnl_pct": pnl_pct, "status": status}


def test_annualize_sharpe_known_rate():
    # 10 trades spread exactly across a 365.25-day span -> trades_per_year == 10.
    start = datetime.date(2025, 1, 1)
    step = 365.25 / 9  # 9 gaps between 10 trades spans exactly one year
    trades = [_trade(start + datetime.timedelta(days=round(i * step)), 1.0) for i in range(10)]
    trades_per_year, annualized = _annualize_sharpe(trades, 0.1)
    assert abs(trades_per_year - 10.0) < 0.1
    assert abs(annualized - 0.1 * (trades_per_year ** 0.5)) < 1e-3


def test_annualize_sharpe_none_when_sharpe_none():
    trades = [_trade(datetime.date(2025, 1, 1), 1.0), _trade(datetime.date(2025, 6, 1), -1.0)]
    assert _annualize_sharpe(trades, None) == (None, None)


def test_annualize_sharpe_none_for_single_trade():
    trades = [_trade(datetime.date(2025, 1, 1), 1.0)]
    assert _annualize_sharpe(trades, 0.1) == (None, None)


def test_annualize_sharpe_none_for_same_day_trades():
    trades = [_trade(datetime.date(2025, 1, 1), 1.0), _trade(datetime.date(2025, 1, 1), -1.0)]
    assert _annualize_sharpe(trades, 0.1) == (None, None)


def test_annualize_sharpe_none_when_entry_date_missing():
    # Regression test for a real bug: live Trade_Outcomes documents get
    # pooled into a minimal {"status", "pnl_pct"} shape with no entry_date
    # at all (optimize.py's report_live_outcomes_context(), settle_trades.py)
    # -- crashed with KeyError before this degraded gracefully instead.
    trades = [{"status": "WIN", "pnl_pct": 1.0}, {"status": "LOSS", "pnl_pct": -1.0}]
    assert _annualize_sharpe(trades, 0.1) == (None, None)


def test_summarize_trades_survives_trades_without_entry_date():
    trades = [{"status": "WIN", "pnl_pct": 1.0}, {"status": "LOSS", "pnl_pct": -1.0}]
    metrics = swingtrade.summarize_trades(trades)
    assert metrics["sharpe_like"] is not None
    assert metrics["trades_per_year"] is None
    assert metrics["annualized_sharpe_like"] is None


def test_summarize_trades_includes_annualized_fields():
    start = datetime.date(2025, 1, 1)
    trades = [
        {"entry_date": start + datetime.timedelta(days=i * 20), "pnl_pct": 2.0 if i % 2 == 0 else -1.0,
         "status": "WIN" if i % 2 == 0 else "LOSS"}
        for i in range(10)
    ]
    metrics = swingtrade.summarize_trades(trades)
    assert metrics["sharpe_like"] is not None
    assert metrics["trades_per_year"] is not None
    assert metrics["annualized_sharpe_like"] is not None
    # Rounding order (rounded sharpe_like/trades_per_year recombined here vs.
    # the unrounded intermediates used internally) means this is approximate,
    # not exact -- confirms the relationship holds, not bit-for-bit equality.
    expected = metrics["sharpe_like"] * (metrics["trades_per_year"] ** 0.5)
    assert abs(metrics["annualized_sharpe_like"] - expected) < 0.01


def test_summarize_trades_weighted_includes_annualized_fields():
    start = datetime.date(2025, 1, 1)
    trades = [
        {"entry_date": start + datetime.timedelta(days=i * 20), "pnl_pct": 2.0 if i % 2 == 0 else -1.0,
         "status": "WIN" if i % 2 == 0 else "LOSS"}
        for i in range(10)
    ]
    weights = [1.0] * 10
    metrics = swingtrade.summarize_trades_weighted(trades, weights)
    assert metrics["sharpe_like"] is not None
    assert metrics["trades_per_year"] is not None
    assert metrics["annualized_sharpe_like"] is not None


def test_summarize_trades_empty_returns_none_annualized_fields():
    metrics = swingtrade.summarize_trades([])
    assert metrics["trades_per_year"] is None
    assert metrics["annualized_sharpe_like"] is None


def test_summarize_trades_weighted_empty_returns_none_annualized_fields():
    metrics = swingtrade.summarize_trades_weighted([], [])
    assert metrics["trades_per_year"] is None
    assert metrics["annualized_sharpe_like"] is None
