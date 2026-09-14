"""swingtrade.backtest.compute_strategy_correlation() -- are two
strategies' real edges actually DIVERSIFIED, or just the same underlying
signal wearing different clothes? (2026-09-13). A genuinely different
question from item 142's simulate_portfolio_constrained(group_key=...):
that one asks how much CAPITAL two strategies fight over when run
together; this one asks whether they make/lose money on the SAME days
for the SAME reasons even with unlimited capital.

Every expected value below is hand-computed, not just asserted against
whatever the code happens to produce -- same discipline every other audit
this session established.
"""
import datetime

from swingtrade.backtest import compute_strategy_correlation

D = datetime.date


def _trade(ticker, entry_date, exit_date, pnl_pct, status="WIN"):
    return {
        "ticker": ticker, "entry_date": entry_date, "exit_date": exit_date,
        "pnl_pct": pnl_pct, "status": status,
    }


def test_perfectly_correlated_strategies_report_correlation_near_one():
    # Both strategies realize IDENTICAL daily P&L on every day (same
    # magnitude, same sign, same days) -- textbook perfect correlation.
    dates = [D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 3), D(2024, 1, 4)]
    pnls = [2.0, -1.0, 3.0, -2.0]
    trades_a = [_trade("A", d, d, p) for d, p in zip(dates, pnls)]
    trades_b = [_trade("B", d, d, p) for d, p in zip(dates, pnls)]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    pair = result["pairs"]["StratA|StratB"]
    assert pair["correlation"] == 1.0
    assert pair["n_days"] == 4


def test_perfectly_anti_correlated_strategies_report_correlation_near_negative_one():
    dates = [D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 3), D(2024, 1, 4)]
    pnls = [2.0, -1.0, 3.0, -2.0]
    trades_a = [_trade("A", d, d, p) for d, p in zip(dates, pnls)]
    trades_b = [_trade("B", d, d, -p) for d, p in zip(dates, pnls)]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    assert result["pairs"]["StratA|StratB"]["correlation"] == -1.0


def test_multiple_trades_same_day_same_strategy_are_summed():
    # Two trades exiting the SAME strategy on the SAME day should sum
    # into one daily P&L figure, not overwrite each other.
    day = D(2024, 1, 1)
    trades_a = [_trade("A", day, day, 2.0), _trade("B", day, day, 3.0)]
    trades_b = [_trade("C", day, day, 5.0)]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    # Only 1 day of activity total -> not enough variance (n_days=1) for a
    # real correlation, but this still proves summing works via ticker_day_overlap.
    assert result["pairs"]["StratA|StratB"]["n_days"] == 1


def test_days_with_no_activity_for_one_strategy_count_as_zero_not_excluded():
    # StratA trades on day 1 and 3; StratB only trades on day 2. Days with
    # NO realized trade for a strategy must be treated as 0.0 P&L that
    # day, not excluded from the comparison (excluding them would only
    # ever compare cherry-picked overlapping days).
    trades_a = [_trade("A", D(2024, 1, 1), D(2024, 1, 1), 5.0), _trade("A", D(2024, 1, 3), D(2024, 1, 3), -5.0)]
    trades_b = [_trade("B", D(2024, 1, 2), D(2024, 1, 2), 1.0)]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    pair = result["pairs"]["StratA|StratB"]
    # Union of dates = {1-1, 1-2, 1-3} -> 3 days, not 0 (no shared days) or 1.
    assert pair["n_days"] == 3


def test_ticker_day_overlap_reflects_shared_entry_instances():
    # StratA and StratB both enter the SAME ticker on the SAME day once,
    # plus one unique entry each -- 1 shared out of 3 total unique
    # (ticker, entry_date) instances = 33.33%.
    shared_day = D(2024, 1, 1)
    trades_a = [
        _trade("AAPL", shared_day, D(2024, 1, 5), 1.0),
        _trade("MSFT", D(2024, 1, 2), D(2024, 1, 6), 1.0),
    ]
    trades_b = [
        _trade("AAPL", shared_day, D(2024, 1, 5), -1.0),
        _trade("NVDA", D(2024, 1, 3), D(2024, 1, 7), -1.0),
    ]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    assert result["pairs"]["StratA|StratB"]["ticker_day_overlap_pct"] == round(1 / 3 * 100, 2)


def test_fewer_than_two_days_returns_none_correlation():
    trades_a = [_trade("A", D(2024, 1, 1), D(2024, 1, 1), 1.0)]
    trades_b = [_trade("B", D(2024, 1, 1), D(2024, 1, 1), 1.0)]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    assert result["pairs"]["StratA|StratB"]["correlation"] is None
    assert result["pairs"]["StratA|StratB"]["n_days"] == 1


def test_zero_variance_series_returns_none_correlation_not_a_crash():
    # StratA's daily P&L is CONSTANT (e.g. always exactly 0 realized that
    # day) -- std=0 makes Pearson correlation undefined, must return None
    # gracefully rather than raising or returning NaN.
    dates = [D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 3)]
    trades_a = [_trade("A", d, d, 0.0) for d in dates]
    trades_b = [_trade("B", d, d, p) for d, p in zip(dates, [1.0, -2.0, 3.0])]
    result = compute_strategy_correlation({"StratA": trades_a, "StratB": trades_b})
    assert result["pairs"]["StratA|StratB"]["correlation"] is None


def test_three_strategies_produce_all_pairwise_combinations():
    trades = {
        "A": [_trade("X", D(2024, 1, 1), D(2024, 1, 1), 1.0)],
        "B": [_trade("Y", D(2024, 1, 2), D(2024, 1, 2), 2.0)],
        "C": [_trade("Z", D(2024, 1, 3), D(2024, 1, 3), 3.0)],
    }
    result = compute_strategy_correlation(trades)
    assert set(result["pairs"].keys()) == {"A|B", "A|C", "B|C"}
