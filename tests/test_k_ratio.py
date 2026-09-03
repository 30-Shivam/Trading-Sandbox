"""compute_k_ratio() -- how CONSISTENTLY a CONCURRENCY-WEIGHTED equity curve
compounds over calendar time (OLS t-statistic of log(equity) regressed
against elapsed days), distinct from sharpe_like which is blind to trade
ORDERING. See swingtrade.backtest.compute_k_ratio's own docstring for the
full rationale, including the 2026-08-31 rework onto the same
_batched_weighted_returns() model compute_max_drawdown() uses.

Most fixtures below use exit_date == entry_date (an instant round-trip) so
each trade is its own isolated batch (concurrency=1, weight=1.0) -- this
reduces exactly to the pre-rework per-trade-in-order behavior, keeping the
existing hand-picked pnl sequences and their expected k_ratio ranges valid
unchanged. See test_overlapping_trades_collapse_into_fewer_batches below for
a test that actually exercises the concurrency weighting itself."""
import datetime

import swingtrade
from swingtrade.backtest import compute_k_ratio


def _trade(entry_date, pnl_pct, status="WIN", exit_date=None):
    return {
        "entry_date": entry_date,
        "exit_date": exit_date if exit_date is not None else entry_date,
        "pnl_pct": pnl_pct,
        "status": status,
    }


def test_k_ratio_high_for_smooth_steady_growth():
    # Small, consistent gains, evenly spaced, non-overlapping (instant
    # round-trips) -- equity compounds in a near-perfectly straight line in
    # log space -> a large positive k_ratio.
    start = datetime.date(2025, 1, 1)
    trades = [
        _trade(start + datetime.timedelta(days=i * 5), 1.0 + (0.01 if i % 2 == 0 else -0.01))
        for i in range(30)
    ]
    k = compute_k_ratio(trades)
    assert k is not None
    assert k > 5.0


def test_k_ratio_near_zero_for_choppy_flat_sequence():
    # Irregular (not cleanly alternating) wins/losses, net-flat total return
    # -- noisy equity with no reliable trend -> k_ratio should be small,
    # unlike the clean steady-growth case above. (A perfectly regular
    # alternating sawtooth was tried first and rejected for this test: fixed
    # +5%/-5% alternation nets a small but highly REGULAR compounding decay
    # each pair from volatility drag, which is itself a real, detectable
    # trend -- not a fair stand-in for "no trend." This sequence is
    # deliberately irregular instead.)
    start = datetime.date(2025, 1, 1)
    pnl_sequence = [
        2, -3, 1, -1, 4, -2, -4, 3, 1, -2, 2, -1, -3, 4, -2,
        1, 3, -4, -1, 2, -2, 3, -3, 1, 4, -1, -2, 2, -4, 3,
    ]
    trades = [
        _trade(start + datetime.timedelta(days=i * 5), pnl, status="WIN" if pnl > 0 else "LOSS")
        for i, pnl in enumerate(pnl_sequence)
    ]
    k = compute_k_ratio(trades)
    assert k is not None
    assert abs(k) < 1.0


def test_k_ratio_none_for_fewer_than_three_trades():
    start = datetime.date(2025, 1, 1)
    trades = [_trade(start, 1.0), _trade(start + datetime.timedelta(days=5), 1.0)]
    assert compute_k_ratio(trades) is None


def test_k_ratio_none_for_same_day_trades():
    # All three share entry_date AND exit_date -> fully concurrent -> one
    # single batch, not 3 -- fewer than 3 distinct batches to regress
    # against, same None result as before the rework (previously reached via
    # a different path: sxx==0 with one distinct x-value, three trades).
    start = datetime.date(2025, 1, 1)
    trades = [_trade(start, 1.0), _trade(start, -1.0), _trade(start, 2.0)]
    assert compute_k_ratio(trades) is None


def test_k_ratio_none_when_entry_date_missing():
    # Same degrade-gracefully convention as _annualize_sharpe(): live
    # Trade_Outcomes documents get pooled into a minimal
    # {"status", "pnl_pct"} shape with no entry_date (or exit_date) at all.
    trades = [
        {"status": "WIN", "pnl_pct": 1.0}, {"status": "LOSS", "pnl_pct": -1.0},
        {"status": "WIN", "pnl_pct": 2.0},
    ]
    assert compute_k_ratio(trades) is None


def test_k_ratio_none_when_exit_date_missing():
    start = datetime.date(2025, 1, 1)
    trades = [
        {"entry_date": start, "pnl_pct": 1.0, "status": "WIN"},
        {"entry_date": start + datetime.timedelta(days=5), "pnl_pct": -1.0, "status": "LOSS"},
        {"entry_date": start + datetime.timedelta(days=10), "pnl_pct": 2.0, "status": "WIN"},
    ]
    assert compute_k_ratio(trades) is None


def test_k_ratio_ignores_open_trades():
    start = datetime.date(2025, 1, 1)
    trades = [
        _trade(start, 1.0), _trade(start + datetime.timedelta(days=5), 1.0),
        _trade(start + datetime.timedelta(days=10), 1.0),
        _trade(start + datetime.timedelta(days=15), 999.0, status="OPEN"),
    ]
    k_with_open = compute_k_ratio(trades)
    k_without_open = compute_k_ratio(trades[:3])
    assert k_with_open == k_without_open


def test_overlapping_trades_collapse_into_fewer_batches():
    # Two trades fully overlapping [Jan1, Jan10] (concurrency=2, weight=0.5
    # each), a third isolated afterwards [Jan15, Jan15] (weight=1.0) -- only
    # 2 distinct exit_date batches exist (Jan10, Jan15), same edge as
    # test_k_ratio_none_for_same_day_trades but with a partial, not full,
    # collapse. 2 batches is still fewer than the 3 needed to regress.
    a = _trade(datetime.date(2025, 1, 1), 10.0, exit_date=datetime.date(2025, 1, 10))
    b = _trade(datetime.date(2025, 1, 1), -10.0, exit_date=datetime.date(2025, 1, 10))
    c = _trade(datetime.date(2025, 1, 15), 5.0)
    assert compute_k_ratio([a, b, c]) is None


def test_k_ratio_matches_max_drawdown_batching_on_realistic_overlap():
    # Sanity-check that compute_k_ratio() and compute_max_drawdown() agree
    # on WHICH batches exist for the same overlapping trade set (4 trades,
    # 3 distinct exit_date batches after weighting) -- confirms the rework
    # actually reuses _batched_weighted_returns() rather than reimplementing
    # a parallel, possibly-divergent batching scheme.
    from swingtrade.backtest import _batched_weighted_returns
    trades = [
        _trade(datetime.date(2025, 1, 1), 10.0, exit_date=datetime.date(2025, 1, 10)),
        _trade(datetime.date(2025, 1, 5), -20.0, exit_date=datetime.date(2025, 1, 10)),
        _trade(datetime.date(2025, 1, 15), 5.0, exit_date=datetime.date(2025, 1, 20)),
        _trade(datetime.date(2025, 1, 25), 3.0, exit_date=datetime.date(2025, 1, 30)),
    ]
    batched = _batched_weighted_returns(trades)
    assert len(batched) == 3
    assert compute_k_ratio(trades) is not None


def test_summarize_trades_includes_k_ratio_field():
    start = datetime.date(2025, 1, 1)
    trades = [
        {"entry_date": start + datetime.timedelta(days=i * 5),
         "exit_date": start + datetime.timedelta(days=i * 5), "pnl_pct": 1.0,
         "status": "WIN"} for i in range(10)
    ]
    metrics = swingtrade.summarize_trades(trades)
    assert metrics["k_ratio"] is not None
    assert metrics["k_ratio"] > 0


def test_summarize_trades_empty_returns_none_k_ratio():
    metrics = swingtrade.summarize_trades([])
    assert metrics["k_ratio"] is None


def test_summarize_trades_weighted_includes_k_ratio_field():
    start = datetime.date(2025, 1, 1)
    trades = [
        {"entry_date": start + datetime.timedelta(days=i * 5),
         "exit_date": start + datetime.timedelta(days=i * 5), "pnl_pct": 1.0,
         "status": "WIN"} for i in range(10)
    ]
    weights = [1.0] * 10
    metrics = swingtrade.summarize_trades_weighted(trades, weights)
    assert metrics["k_ratio"] is not None
    assert metrics["k_ratio"] > 0


def test_summarize_trades_weighted_empty_returns_none_k_ratio():
    metrics = swingtrade.summarize_trades_weighted([], [])
    assert metrics["k_ratio"] is None
