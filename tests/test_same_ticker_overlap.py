"""audit_same_ticker_overlap() and simulate_portfolio_constrained()'s new
max_positions_per_ticker parameter (2026-09-14) -- a real, previously
entirely unexamined gap: every simulate_*_signals() function walks
eligible days independently and opens a trade whenever its own signal
fires, with no check for whether a PREVIOUS trade on the SAME ticker is
still open. Confirmed real via a direct data check: a 15-ticker, 5-year
ma_crossover sample showed 3 of 94 real trades (~3%) with a genuine
same-ticker overlap.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- same discipline every other audit
this session established.
"""
import datetime

from swingtrade.backtest import audit_same_ticker_overlap, simulate_portfolio_constrained

D = datetime.date


def _trade(ticker, entry_date, exit_date, pnl_pct=0.0, status="WIN"):
    return {"ticker": ticker, "entry_date": entry_date, "exit_date": exit_date, "pnl_pct": pnl_pct, "status": status}


# --- audit_same_ticker_overlap() ---

def test_empty_trades_returns_zero_baseline():
    result = audit_same_ticker_overlap([])
    assert result["n_trades"] == 0
    assert result["n_overlapping_pairs"] == 0
    assert result["pct_trades_in_an_overlap"] is None


def test_no_overlap_when_trades_are_sequential():
    # Trade B enters AFTER trade A exits -- no overlap.
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 10)),
        _trade("AAPL", D(2024, 1, 11), D(2024, 1, 20)),
    ]
    result = audit_same_ticker_overlap(trades)
    assert result["n_overlapping_pairs"] == 0
    assert result["pct_trades_in_an_overlap"] == 0.0


def test_detects_a_real_same_ticker_overlap():
    # Trade B enters BEFORE trade A's own exit_date -- a genuine overlap,
    # matching the real ma_crossover pattern this was built to catch.
    trades = [
        _trade("NVDA", D(2026, 7, 17), D(2026, 8, 7)),
        _trade("NVDA", D(2026, 8, 6), D(2026, 8, 20)),
    ]
    result = audit_same_ticker_overlap(trades)
    assert result["n_overlapping_pairs"] == 1
    assert result["pct_trades_in_an_overlap"] == 100.0
    assert result["overlap_examples"] == [("NVDA", D(2026, 8, 7), D(2026, 8, 6))]


def test_entry_exactly_on_exit_date_counts_as_overlap():
    # Closed-interval convention, matching compute_max_drawdown()'s own.
    trades = [_trade("AAPL", D(2024, 1, 1), D(2024, 1, 10)), _trade("AAPL", D(2024, 1, 10), D(2024, 1, 20))]
    result = audit_same_ticker_overlap(trades)
    assert result["n_overlapping_pairs"] == 1


def test_different_tickers_never_overlap_with_each_other():
    trades = [_trade("AAPL", D(2024, 1, 1), D(2024, 1, 20)), _trade("MSFT", D(2024, 1, 5), D(2024, 1, 15))]
    result = audit_same_ticker_overlap(trades)
    assert result["n_overlapping_pairs"] == 0
    assert result["n_tickers"] == 2


def test_open_trades_are_excluded():
    trades = [_trade("AAPL", D(2024, 1, 1), D(2024, 1, 10), status="OPEN")]
    result = audit_same_ticker_overlap(trades)
    assert result["n_trades"] == 0


def test_pct_trades_in_overlap_only_counts_affected_trades():
    # 3 AAPL trades: first two overlap, third is sequential and clean.
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 10)),
        _trade("AAPL", D(2024, 1, 5), D(2024, 1, 15)),   # overlaps with the first
        _trade("AAPL", D(2024, 1, 16), D(2024, 1, 25)),  # clean, after the second exits
    ]
    result = audit_same_ticker_overlap(trades)
    assert result["n_overlapping_pairs"] == 1
    # 2 of 3 trades touch the one overlapping pair.
    assert result["pct_trades_in_an_overlap"] == round(2 / 3 * 100, 2)


# --- simulate_portfolio_constrained(max_positions_per_ticker=...) ---

def test_max_positions_per_ticker_none_preserves_original_behavior():
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 20), pnl_pct=10.0),
        _trade("AAPL", D(2024, 1, 5), D(2024, 1, 25), pnl_pct=10.0),
    ]
    result = simulate_portfolio_constrained(trades, starting_capital=10_000.0, position_budget=1_000.0)
    assert result["n_taken"] == 2
    assert result["n_skipped_ticker_limit"] == 0


def test_max_positions_per_ticker_one_blocks_a_second_open_position():
    # Both trades in the SAME ticker overlap -- with max_positions_per_ticker=1,
    # only the first can be taken; the second is skipped for ticker limit,
    # NOT insufficient capital (plenty of cash available).
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 20), pnl_pct=10.0),
        _trade("AAPL", D(2024, 1, 5), D(2024, 1, 25), pnl_pct=-5.0),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_positions_per_ticker=1,
    )
    assert result["n_taken"] == 1
    assert result["n_skipped_ticker_limit"] == 1
    assert result["n_skipped_insufficient_capital"] == 0


def test_max_positions_per_ticker_allows_reentry_after_exit():
    # Same ticker, but the second trade enters AFTER the first's exit_date
    # -- max_positions_per_ticker=1 should NOT block this, since the first
    # position is no longer open by then.
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 10), pnl_pct=10.0),
        _trade("AAPL", D(2024, 1, 11), D(2024, 1, 20), pnl_pct=5.0),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_positions_per_ticker=1,
    )
    assert result["n_taken"] == 2
    assert result["n_skipped_ticker_limit"] == 0


def test_max_positions_per_ticker_does_not_affect_different_tickers():
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 1, 20), pnl_pct=10.0),
        _trade("MSFT", D(2024, 1, 1), D(2024, 1, 20), pnl_pct=10.0),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_positions_per_ticker=1,
    )
    assert result["n_taken"] == 2
    assert result["n_skipped_ticker_limit"] == 0


def test_max_positions_per_ticker_two_allows_a_second_but_not_a_third():
    trades = [
        _trade("AAPL", D(2024, 1, 1), D(2024, 2, 1), pnl_pct=1.0),
        _trade("AAPL", D(2024, 1, 2), D(2024, 2, 2), pnl_pct=1.0),
        _trade("AAPL", D(2024, 1, 3), D(2024, 2, 3), pnl_pct=1.0),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_positions_per_ticker=2,
    )
    assert result["n_taken"] == 2
    assert result["n_skipped_ticker_limit"] == 1
