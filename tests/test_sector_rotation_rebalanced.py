"""simulate_sector_rotation_rebalanced() -- the first genuine rebalanced-
portfolio backtest engine in this codebase (hold a sector while it keeps
qualifying, exit only when it actually falls out of the top-N), as opposed
to every other strategy's discrete-signal-plus-ATR-stop/target machinery.
Hand-computed small cases so the rebalance/hold/exit timing itself is
verified directly, not just "it runs and produces plausible-looking output."
"""
import numpy as np
import pandas as pd

import swingtrade


def _panel(prices: dict[str, list[float]], start="2020-01-01") -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    return pd.DataFrame(prices, index=pd.date_range(start, periods=n, freq="D"))


def test_rebalanced_holds_a_sector_across_multiple_rebalances_without_fragmenting():
    # 3 sectors, 2 always ahead of the 3rd. lookback=2, rebalance every 2
    # days, top_n=2 -- A and B should stay held continuously (ONE trade
    # each spanning the whole window), C never qualifies at all.
    n = 10
    panel = _panel({
        "A": list(np.linspace(100, 130, n)),
        "B": list(np.linspace(100, 125, n)),
        "C": list(np.linspace(100, 101, n)),  # barely moves -- never top-2
    })
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        panel, lookback_days=2, top_n=2, rebalance_frequency_days=2,
    )
    tickers = {t["ticker"] for t in trades}
    assert tickers == {"A", "B"}
    # ONE trade per sector, not one per rebalance check -- confirms holding
    # spans are merged, not fragmented.
    assert len(trades) == 2
    a_trade = next(t for t in trades if t["ticker"] == "A")
    # Entered at the FIRST rebalance date (index 2, lookback_days=2), held
    # to the FINAL date (index 9) since it never fell out of top-2.
    assert a_trade["entry_date"] == panel.index[2]
    assert a_trade["exit_date"] == panel.index[-1]
    assert a_trade["status"] == "WIN"


def test_rebalanced_exits_when_a_sector_falls_out_of_top_n_and_reenters_later():
    # A leads early, fades, then leads again -- should produce 2 SEPARATE
    # trades for A (exit when it drops out, a fresh entry when it re-qualifies),
    # not one continuous span.
    n = 12
    a_prices = [100, 110, 120, 118, 105, 95, 90, 95, 110, 130, 150, 170]
    b_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    c_prices = [100, 100, 100, 108, 112, 115, 118, 120, 108, 106, 105, 104]
    panel = _panel({"A": a_prices, "B": b_prices, "C": c_prices})
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        panel, lookback_days=2, top_n=1, rebalance_frequency_days=1,
    )
    a_trades = [t for t in trades if t["ticker"] == "A"]
    # A should have re-entered at least once (more than one trade) since it
    # was briefly overtaken by C in the middle of the window.
    assert len(a_trades) >= 2


def test_rebalanced_no_positions_ever_left_open():
    n = 8
    panel = _panel({
        "A": list(np.linspace(100, 120, n)),
        "B": list(np.linspace(100, 90, n)),
    })
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        panel, lookback_days=2, top_n=1, rebalance_frequency_days=2,
    )
    assert trades  # sanity: something actually traded
    assert all(t["status"] != "OPEN" for t in trades)


def test_rebalanced_pnl_pct_matches_hand_computed_value():
    n = 6
    # A: 100 -> 100 -> 110 (flat then a clean 10% move over the held span)
    panel = _panel({
        "A": [100, 100, 100, 105, 110, 110],
        "B": [100, 99, 98, 97, 96, 95],  # always worse -- never qualifies
    })
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        panel, lookback_days=2, top_n=1, rebalance_frequency_days=3,
    )
    a_trades = [t for t in trades if t["ticker"] == "A"]
    assert len(a_trades) == 1
    entry_price = panel.loc[panel.index[2], "A"]  # first rebalance date (index=lookback_days=2)
    exit_price = panel.loc[panel.index[-1], "A"]
    expected_pnl = round((exit_price - entry_price) / entry_price * 100, 4)
    assert a_trades[0]["pnl_pct"] == expected_pnl


def test_rebalanced_schema_compatible_with_summarize_trades():
    n = 10
    panel = _panel({
        "A": list(np.linspace(100, 130, n)),
        "B": list(np.linspace(100, 115, n)),
    })
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        panel, lookback_days=2, top_n=1, rebalance_frequency_days=2,
    )
    summary = swingtrade.summarize_trades(trades)
    assert summary["trade_count"] == len(trades)
    assert summary["open_count"] == 0
    dd = swingtrade.compute_max_drawdown(trades)
    assert dd is None or dd >= 0.0


def test_equal_weight_buy_and_hold_uses_shared_start_and_end_dates():
    n = 6
    panel = _panel({
        "A": [100, 105, 110, 115, 120, 130],  # +30% over the held span
        "B": [100, 90, 80, 70, 60, 50],  # -50% over the held span
    })
    trades = swingtrade.compute_equal_weight_buy_and_hold(panel, lookback_days=2)
    assert {t["ticker"] for t in trades} == {"A", "B"}
    for t in trades:
        assert t["entry_date"] == panel.index[2]
        assert t["exit_date"] == panel.index[-1]
    a_trade = next(t for t in trades if t["ticker"] == "A")
    b_trade = next(t for t in trades if t["ticker"] == "B")
    # A: 110 -> 130 = +18.18%; B: 80 -> 50 = -37.5% (both measured from the
    # SAME shared start date, index 2, not from each ticker's own first row).
    assert a_trade["pnl_pct"] == round((130 - 110) / 110 * 100, 4)
    assert b_trade["pnl_pct"] == round((50 - 80) / 80 * 100, 4)


def test_equal_weight_buy_and_hold_skips_sectors_missing_data_at_shared_dates():
    n = 6
    panel = _panel({
        "A": [100, 105, 110, 115, 120, 130],
        "B": [np.nan, np.nan, np.nan, 70, 60, 50],  # no valid price at the shared start date
    })
    trades = swingtrade.compute_equal_weight_buy_and_hold(panel, lookback_days=2)
    assert {t["ticker"] for t in trades} == {"A"}
