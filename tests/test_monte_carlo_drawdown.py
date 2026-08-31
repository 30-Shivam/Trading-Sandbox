"""Monte Carlo trade-order reshuffling (improvements.txt item 87) -- how
much of compute_max_drawdown()'s single chronological-order result is just
the LUCK of the particular sequence a fixed set of trades happened to occur
in? Reshuffles ORDER only, not trade SELECTION -- a different question from
ticker-holdout (item 69, which checks selection/generalization).
"""
import datetime

from swingtrade.backtest import compute_max_drawdown, monte_carlo_drawdown


# 2026-08-31: compute_max_drawdown()/monte_carlo_drawdown() now need
# exit_date for concurrency weighting -- exit_date=entry_date (a
# zero-day interval) keeps every trade below fully isolated/non-
# overlapping, same as before the redesign, so every hand-verified
# number in this file (about pnl-SEQUENCE/ordering effects, not
# holding-period/concurrency) stays correct unchanged. Real
# concurrency-weighting behavior is covered by tests/test_max_drawdown.py.
def _trade(entry_date, pnl_pct, status="WIN", exit_date=None):
    return {
        "entry_date": entry_date, "exit_date": exit_date or entry_date,
        "pnl_pct": pnl_pct, "status": status,
    }


def test_returns_none_with_fewer_than_three_resolved_trades():
    trades = [_trade(datetime.date(2022, 1, 1), 5.0), _trade(datetime.date(2022, 1, 2), -2.0, status="LOSS")]
    assert monte_carlo_drawdown(trades) is None


def test_open_trades_are_excluded_from_the_resolved_count():
    trades = [
        _trade(datetime.date(2022, 1, 1), 5.0),
        _trade(datetime.date(2022, 1, 2), -2.0, status="LOSS"),
        _trade(datetime.date(2022, 1, 3), None, status="OPEN"),
    ]
    assert monte_carlo_drawdown(trades) is None  # only 2 resolved


def test_deterministic_for_a_fixed_seed():
    trades = [
        _trade(datetime.date(2022, 1, i + 1), pnl)
        for i, pnl in enumerate([-10, -10, 20, 20, 5, -3])
    ]
    first = monte_carlo_drawdown(trades, n_simulations=200, seed=42)
    second = monte_carlo_drawdown(trades, n_simulations=200, seed=42)
    assert first == second


def test_different_seeds_can_give_different_distributions():
    trades = [
        _trade(datetime.date(2022, 1, i + 1), pnl)
        for i, pnl in enumerate([-10, -10, 20, 20, 5, -3])
    ]
    a = monte_carlo_drawdown(trades, n_simulations=200, seed=1)
    b = monte_carlo_drawdown(trades, n_simulations=200, seed=2)
    # Same underlying stats structure, but not required to be bit-identical.
    assert a["n_simulations"] == b["n_simulations"] == 200
    assert a["n_trades"] == b["n_trades"] == 6


def test_reshuffling_reveals_a_worse_ordering_than_a_favorable_actual_order():
    # Two -10% and two +20% trades. Hand-verified: interleaving losses and
    # gains (the actual chronological order below) caps max_drawdown at
    # 10%, but grouping both losses (or both gains) back-to-back produces
    # a worse 19% max_drawdown -- a real ordering effect, not scale
    # invariance (a SINGLE dominant loss would give the same drawdown
    # regardless of placement; it takes >=2 losses interacting with gains
    # to make ordering matter).
    ordered_pnls = [-10, 20, -10, 20]  # alternating -> the favorable actual order
    trades = [_trade(datetime.date(2022, 1, i + 1), pnl) for i, pnl in enumerate(ordered_pnls)]

    actual_dd = compute_max_drawdown(trades)
    assert abs(actual_dd - 10.0) < 1e-9

    result = monte_carlo_drawdown(trades, n_simulations=2000, seed=42)
    assert result["actual_chronological_max_drawdown"] == actual_dd
    # 2000 sims over only 6 distinct multiset orderings will find the 19%
    # worst-case block ordering with overwhelming probability.
    assert abs(result["monte_carlo_worst"] - 19.0) < 0.5
    assert result["monte_carlo_worst"] >= result["actual_chronological_max_drawdown"]
    assert 10.0 <= result["monte_carlo_mean"] <= 19.0
    assert 10.0 <= result["monte_carlo_median"] <= 19.0


def test_result_shape():
    trades = [
        _trade(datetime.date(2022, 1, i + 1), pnl)
        for i, pnl in enumerate([5, -3, 4, -2, 6])
    ]
    result = monte_carlo_drawdown(trades, n_simulations=50, seed=7)
    assert set(result.keys()) == {
        "n_simulations", "n_trades", "actual_chronological_max_drawdown",
        "monte_carlo_mean", "monte_carlo_median", "monte_carlo_p95", "monte_carlo_worst",
    }
    assert result["n_simulations"] == 50
    assert result["n_trades"] == 5
    assert result["monte_carlo_mean"] <= result["monte_carlo_worst"]
    assert result["monte_carlo_median"] <= result["monte_carlo_p95"] <= result["monte_carlo_worst"]
