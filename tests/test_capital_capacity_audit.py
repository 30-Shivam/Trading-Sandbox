"""audit_capital_capacity() (2026-09-14, twentieth iteration of the
"operate in a continuous loop... fully comprehensive of all variables"
directive) -- a genuinely new axis distinct from item 138's own
simulate_portfolio_constrained(): that function answers "how many signals
could ONE fixed-capital account afford" at a single starting_capital
picked somewhat arbitrarily; this scans MULTIPLE capital levels to find
whether pct_signals_taken keeps climbing (money is still the binding
constraint) or plateaus (the sector/portfolio caps become the real
ceiling, and more capital stops helping) -- directly relevant to deciding
how much capital a strategy could actually absorb before adding more is
wasted.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- same discipline every other audit
this session established.
"""
import datetime

from swingtrade.backtest import audit_capital_capacity

D = datetime.date


def _trade(ticker, entry_date, exit_date, pnl_pct=0.0, sector="Tech", status="WIN"):
    return {
        "ticker": ticker, "entry_date": entry_date, "exit_date": exit_date,
        "pnl_pct": pnl_pct, "sector": sector, "status": status,
    }


def test_empty_trades_produces_no_ceiling_and_no_crash():
    result = audit_capital_capacity([], position_budget=1_000.0, capital_levels=[1_000.0, 5_000.0])
    assert len(result["levels"]) == 2
    assert all(lvl["pct_signals_taken"] is None for lvl in result["levels"])
    assert all(lvl["dominant_skip_reason"] is None for lvl in result["levels"])
    assert result["capacity_ceiling_capital"] is None


def test_single_level_never_reports_a_ceiling():
    # No previous level to compare against -- there is nothing to plateau relative to.
    trades = [_trade("A", D(2022, 1, 1), D(2022, 1, 10), 10.0)]
    result = audit_capital_capacity(trades, position_budget=1_000.0, capital_levels=[1_000.0])
    assert len(result["levels"]) == 1
    assert result["capacity_ceiling_capital"] is None


def test_capacity_scan_finds_the_real_ceiling_once_all_signals_are_taken():
    # 3 same-day, same-ticker-budget signals, no sector/portfolio caps.
    # $1,000: only 1 of 3 fundable (33.33%). $2,000: 2 of 3 (66.67%).
    # $3,000: all 3 (100%). $5,000: still all 3 (100%, nothing left to
    # gain) -- this is where the ceiling should be detected.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 5.0),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 5.0),
        _trade("C", D(2022, 1, 1), D(2022, 1, 10), 5.0),
    ]
    result = audit_capital_capacity(
        trades, position_budget=1_000.0,
        capital_levels=[1_000.0, 2_000.0, 3_000.0, 5_000.0],
    )
    pct_taken = [lvl["pct_signals_taken"] for lvl in result["levels"]]
    assert pct_taken == [round(1 / 3 * 100, 2), round(2 / 3 * 100, 2), 100.0, 100.0]
    assert result["capacity_ceiling_capital"] == 5_000.0


def test_dominant_skip_reason_reports_insufficient_capital_when_funds_bind():
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 5.0),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 5.0),
    ]
    result = audit_capital_capacity(trades, position_budget=1_000.0, capital_levels=[1_000.0])
    assert result["levels"][0]["dominant_skip_reason"] == "insufficient_capital"


def test_dominant_skip_reason_reports_sector_limit_when_funds_are_ample():
    # $10,000 cash is ample for both $1,000 trades, but a 5% sector cap
    # ($500) is smaller than a single position -- funds are never the
    # blocker here, the sector cap is.
    trades = [
        _trade("A", D(2022, 1, 1), D(2022, 1, 10), 5.0, sector="Tech"),
        _trade("B", D(2022, 1, 1), D(2022, 1, 10), 5.0, sector="Tech"),
    ]
    result = audit_capital_capacity(
        trades, position_budget=1_000.0, capital_levels=[10_000.0],
        max_sector_allocation_pct=0.05,
    )
    level = result["levels"][0]
    assert level["n_skipped_insufficient_capital"] == 0
    assert level["n_skipped_sector_limit"] == 2
    assert level["dominant_skip_reason"] == "sector_limit"


def test_no_skips_at_a_level_reports_none_dominant_reason():
    trades = [_trade("A", D(2022, 1, 1), D(2022, 1, 10), 5.0)]
    result = audit_capital_capacity(trades, position_budget=1_000.0, capital_levels=[10_000.0])
    assert result["levels"][0]["dominant_skip_reason"] is None
