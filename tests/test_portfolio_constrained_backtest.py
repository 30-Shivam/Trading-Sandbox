"""simulate_portfolio_constrained() -- a REAL, single, finite-capital
account replay of already-simulated signals (2026-09-13), built per the
user's "operate in a continuous loop... i want this backtest to be fully
comprehensive of all variables" directive. Closes a gap
allocation.allocate_capital() itself has documented since before this
session: "a per-ticker walk-forward backtest doesn't model simultaneous
cross-ticker exposure either." Every other metric in backtest.py
(sharpe_like, compute_max_drawdown's own concurrency-weighted curve) treats
every signal as takeable; this asks how many a real, capital-constrained
account could actually have afforded, and what its own dollar equity curve
looks like once funds/sector/portfolio caps are respected.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- same discipline test_max_drawdown.py
established for compute_max_drawdown().
"""
import datetime

from swingtrade.backtest import simulate_portfolio_constrained

D = datetime.date


def _trade(ticker, entry_date, exit_date, pnl_pct, sector="Tech", status="WIN"):
    return {
        "ticker": ticker, "entry_date": entry_date, "exit_date": exit_date,
        "pnl_pct": pnl_pct, "sector": sector, "status": status,
    }


def test_empty_trades_returns_zero_signal_baseline():
    result = simulate_portfolio_constrained([], starting_capital=10_000.0, position_budget=1_000.0)
    assert result["n_signals"] == 0
    assert result["n_taken"] == 0
    assert result["ending_equity"] == 10_000.0
    assert result["total_return_pct"] == 0.0
    assert result["pct_signals_taken"] is None


def test_open_trades_and_missing_dates_are_excluded_from_replay():
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 10.0, status="OPEN"),
        {"ticker": "B", "pnl_pct": 5.0, "status": "WIN"},  # no entry_date/exit_date
    ]
    result = simulate_portfolio_constrained(trades, starting_capital=10_000.0, position_budget=1_000.0)
    assert result["n_signals"] == 0


def test_single_affordable_trade_is_taken_and_realizes_pnl():
    # $10,000 capital, $1,000 position, +20% pnl -> ending equity 10,200.
    trades = [_trade("A", D(2022, 1, 1), D(2022, 1, 10), 20.0)]
    result = simulate_portfolio_constrained(trades, starting_capital=10_000.0, position_budget=1_000.0)
    assert result["n_taken"] == 1
    assert result["n_signals"] == 1
    assert result["ending_equity"] == 10_200.0
    assert result["total_return_pct"] == 2.0
    assert result["pct_signals_taken"] == 100.0


def test_insufficient_capital_skips_excess_trades():
    # $1,000 capital, $1,000 per position -- only ONE position can ever be
    # open at a time. Three trades all open on the same day, so only the
    # first (list order, since ties break by stable sort on entry_date)
    # can be funded; the other two are skipped for insufficient capital.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 10.0),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 10.0),
        _trade("C", D(2022, 1, 1), D(2022, 1, 10), 10.0),
    ]
    result = simulate_portfolio_constrained(trades, starting_capital=1_000.0, position_budget=1_000.0)
    assert result["n_taken"] == 1
    assert result["n_skipped_insufficient_capital"] == 2
    assert result["pct_signals_taken"] == round(1 / 3 * 100, 2)


def test_capital_frees_up_after_exit_for_a_later_entry():
    # $1,000 capital, $1,000 per position. Trade A exits before trade B
    # enters, so B should still be takeable once A's capital (plus its
    # realized gain) is released back. position_budget is FLAT, not
    # compounded with account value -- both trades commit exactly $1,000,
    # not whatever the account grew to.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 5), 10.0),
        _trade("B", D(2022, 1, 6), D(2022, 1, 15), 10.0),
    ]
    result = simulate_portfolio_constrained(trades, starting_capital=1_000.0, position_budget=1_000.0)
    assert result["n_taken"] == 2
    assert result["n_skipped_insufficient_capital"] == 0
    # A: 1000 -> 1100 (all cash, nothing left over). B commits a flat $1000
    # again (not $1100), leaving $100 idle; B resolves +10% -> +$1100.
    # Ending = $100 idle + $1100 realized = $1200.
    assert result["ending_equity"] == 1200.0


def test_sector_cap_skips_a_trade_that_would_breach_it():
    # $10,000 capital, 20% sector cap ($2,000). $1,000 positions -- a
    # second same-sector, same-day trade would push spend to $2,000,
    # exactly at the cap (allowed, <=), a THIRD would breach it (skipped).
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
        _trade("C", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_sector_allocation_pct=0.20,
    )
    assert result["n_taken"] == 2
    assert result["n_skipped_sector_limit"] == 1


def test_portfolio_cap_skips_a_trade_that_would_breach_total_deployed():
    # $10,000 capital, 10% total-deployed cap ($1,000). Only ONE $1,000
    # position can ever be open regardless of sector diversity.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Energy"),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_total_deployed_pct=0.10,
    )
    assert result["n_taken"] == 1
    assert result["n_skipped_portfolio_limit"] == 1


def test_drawdown_reflects_a_real_loss_then_recovery():
    # $10,000 capital, one $1,000 position at a time.
    # Trade A: -50% -> equity 10,000 - 1000 + 500 = 9,500 (peak stays 10,000, dd=5%)
    # Trade B (after A exits): +50% -> equity 9,500 - 1000 + 1500 = 10,000
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 5), -50.0),
        _trade("B", D(2022, 1, 6), D(2022, 1, 10), 50.0),
    ]
    result = simulate_portfolio_constrained(trades, starting_capital=10_000.0, position_budget=1_000.0)
    assert result["ending_equity"] == 10_000.0
    assert result["max_drawdown_pct"] == 5.0


def test_sector_lookup_overrides_trade_own_sector_tag():
    # sector_lookup takes priority over the trade dict's own "sector" field
    # (mirrors allocate_capital()'s own ticker->sector lookup convention).
    # Cap is $1,500/sector: both trades tagged "Tech" would stack to $2,000
    # and breach it, but overridden to two DIFFERENT sectors (Energy,
    # Materials) each only reaches $1,000 -- neither breaches the cap.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 0.0, sector="Tech"),
    ]
    result = simulate_portfolio_constrained(
        trades, starting_capital=10_000.0, position_budget=1_000.0, max_sector_allocation_pct=0.15,
        sector_lookup={"A": "Energy", "B": "Materials"},
    )
    assert result["n_taken"] == 2
    assert result["n_skipped_sector_limit"] == 0
