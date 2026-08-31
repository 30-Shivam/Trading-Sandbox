"""compute_max_drawdown() -- concurrency-weighted equity curve (2026-08-31
redesign). See swingtrade/backtest.py's own docstring for the full model:
each trade's position_weight = 1 / (how many trades, including itself, had
an open [entry_date, exit_date] interval at the moment it entered), and
trades sharing an exit_date are combined ADDITIVELY (not compounded one at
a time -- that's order-dependent, see test_same_day_exits_are_order_independent
below for the concrete proof this file's own design was built to catch).

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- see each test's own comment for the
arithmetic.
"""
import datetime

from swingtrade.backtest import _concurrency_at_entry, compute_max_drawdown


def _trade(entry_date, exit_date, pnl_pct, status="WIN"):
    return {"entry_date": entry_date, "exit_date": exit_date, "pnl_pct": pnl_pct, "status": status}


D = datetime.date


def test_returns_none_for_no_resolved_trades():
    assert compute_max_drawdown([]) is None
    assert compute_max_drawdown([_trade(D(2022, 1, 1), D(2022, 1, 5), None, status="OPEN")]) is None


def test_single_trade_matches_old_sequential_behavior():
    # One trade, no possible overlap -- weight is always 1.0, so this must
    # match the pre-redesign formula exactly.
    trades = [_trade(D(2022, 1, 1), D(2022, 1, 10), -30.0)]
    assert compute_max_drawdown(trades) == 30.0


def test_two_fully_overlapping_trades_weighted_by_concurrency():
    # Both open the same window -> concurrency=2 for each -> weight=0.5.
    # Combined additively (same exit_date): 0.5*0 + 0.5*(-0.50) = -0.25.
    # equity 1.0 -> 0.75, drawdown = 25%. The OLD naive sequential
    # algorithm would have given 50% (full-size A no-op, then full-size
    # B's -50% applied to the whole account) -- this is the headline case
    # the redesign exists to fix.
    a = _trade(D(2022, 1, 1), D(2022, 1, 10), 0.0)
    b = _trade(D(2022, 1, 1), D(2022, 1, 10), -50.0)
    assert compute_max_drawdown([a, b]) == 25.0


def test_same_day_exits_are_order_independent():
    # Two fully-overlapping trades, weights 0.5/0.5, pnl +20%/-50%, same
    # exit_date. Hand-verified that naively compounding them ONE AT A TIME
    # (even after weighting) is order-dependent:
    #   A then B: equity 1 -> 1.10 (peak 1.10) -> 0.825, dd = (1.10-0.825)/1.10 = 25.0%
    #   B then A: equity 1 -> 0.75 (peak 1.0)   -> 0.825, dd = (1.0-0.825)/1.0   = 17.5%
    # The correct ADDITIVE combination is order-independent:
    #   0.5*0.20 + 0.5*(-0.50) = -0.15 -> equity 1 -> 0.85, dd = 15.0%
    # Both list orders below must give the same (correct) answer.
    a = _trade(D(2022, 1, 1), D(2022, 1, 10), 20.0)
    b = _trade(D(2022, 1, 1), D(2022, 1, 10), -50.0)
    assert compute_max_drawdown([a, b]) == 15.0
    assert compute_max_drawdown([b, a]) == 15.0


def test_partial_overlap_gives_asymmetric_weights():
    # A: Jan1-10, +10%. B: Jan5-15 (opens while A is still open), -40%.
    # Concurrency at A's own entry (Jan1): just A -> 1 -> weight 1.0.
    # Concurrency at B's own entry (Jan5): A still open + B itself -> 2 -> weight 0.5.
    # Batch 1 (A's exit, Jan10): 1.0*0.10 = 0.10 -> equity 1.10, peak 1.10.
    # Batch 2 (B's exit, Jan15): 0.5*(-0.40) = -0.20 -> equity 0.88.
    # drawdown = (1.10-0.88)/1.10 = 20.0%.
    a = _trade(D(2022, 1, 1), D(2022, 1, 10), 10.0)
    b = _trade(D(2022, 1, 5), D(2022, 1, 15), -40.0)
    assert compute_max_drawdown([a, b]) == 20.0


def test_non_overlapping_trades_matches_old_sequential_behavior():
    # Three back-to-back, non-overlapping trades -- weight=1 for all,
    # reduces exactly to the pre-redesign sequential formula.
    # equity: 1.10 -> 0.88 -> 1.012. Drawdowns: 0%, 20%, 8%. Max = 20%.
    trades = [
        _trade(D(2022, 1, 1), D(2022, 1, 5), 10.0),
        _trade(D(2022, 1, 6), D(2022, 1, 10), -20.0),
        _trade(D(2022, 1, 11), D(2022, 1, 15), 15.0),
    ]
    assert compute_max_drawdown(trades) == 20.0


def test_all_trades_same_day_fully_overlapping():
    # 4 trades, identical [Jan1, Jan10] interval -> weight 0.25 each.
    # Single additive batch: 0.25*(0.10 - 0.10 + 0.20 - 0.40) = -0.05.
    # equity 1.0 -> 0.95, drawdown = 5.0%. (The old naive sequential
    # algorithm, applied in this exact list order, would give 40.0% --
    # order-dependent and far more extreme.)
    trades = [
        _trade(D(2022, 1, 1), D(2022, 1, 10), 10.0),
        _trade(D(2022, 1, 1), D(2022, 1, 10), -10.0),
        _trade(D(2022, 1, 1), D(2022, 1, 10), 20.0),
        _trade(D(2022, 1, 1), D(2022, 1, 10), -40.0),
    ]
    assert compute_max_drawdown(trades) == 5.0


def test_concurrency_at_entry_sweep_line_correctness():
    # T1: Jan1-10. T2: Jan5-15 (opens while T1 open). T3: Jan8-9 (opens
    # while both T1 and T2 open). T4: Jan20-25 (fully isolated, after
    # everything else has closed).
    t1 = _trade(D(2022, 1, 1), D(2022, 1, 10), 1.0)
    t2 = _trade(D(2022, 1, 5), D(2022, 1, 15), 1.0)
    t3 = _trade(D(2022, 1, 8), D(2022, 1, 9), 1.0)
    t4 = _trade(D(2022, 1, 20), D(2022, 1, 25), 1.0)
    assert _concurrency_at_entry([t1, t2, t3, t4]) == [1, 2, 3, 1]


def test_same_day_entry_and_exit_boundary_counts_as_concurrent():
    # U exits Jan10; T enters that same day -- the closed-interval
    # convention treats the shared boundary day as concurrent exposure.
    u = _trade(D(2022, 1, 1), D(2022, 1, 10), 1.0)
    t = _trade(D(2022, 1, 10), D(2022, 1, 20), 1.0)
    assert _concurrency_at_entry([u, t]) == [1, 2]


def test_returns_none_when_exit_date_missing():
    trades = [{"entry_date": D(2022, 1, 1), "pnl_pct": -5.0, "status": "LOSS"}]
    assert compute_max_drawdown(trades) is None


def test_open_trades_excluded():
    trades = [
        _trade(D(2022, 1, 1), D(2022, 1, 5), 10.0),
        _trade(D(2022, 1, 6), D(2022, 1, 10), -20.0),
        {"entry_date": D(2022, 1, 11), "exit_date": None, "pnl_pct": None, "status": "OPEN"},
    ]
    # Same as test_non_overlapping_trades_matches_old_sequential_behavior
    # minus the third (non-overlapping, isolated) leg -- the OPEN trade
    # must have zero effect regardless of its own (garbage) fields.
    assert compute_max_drawdown(trades) == 20.0
