"""Fixes a real methodological gap in benchmark_sector_rotation_rebalanced.py:
summarize_trades()/compute_max_drawdown() were designed for asynchronous
discrete-signal trades, not a synchronized buy-and-hold-everything trade set
(every sector enters/exits on the identical 2 dates) -- that produced a
degenerate "max_drawdown: 0.0" for the buy-and-hold benchmark (an artifact
of only 2 timestamps existing at all, not a real claim that equal-weight
sector exposure never drew down). This builds a REAL periodic NAV curve for
both the rotated strategy and the equal-weight benchmark, using the exact
same rebalance dates and holding logic, so drawdown/total-return are
actually comparable.
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"D:\Trading-Sandbox")
from run_backtest import fetch_history
from watchlist import SECTOR_ETF

start = pd.Timestamp("2019-01-01")
end = pd.Timestamp("2026-09-15")

print("Fetching 11 real sector ETFs + SPY...")
sector_data = {}
for sector, etf in SECTOR_ETF.items():
    df = fetch_history(etf, start, end)
    if not df.empty:
        sector_data[sector] = df
spy = fetch_history("SPY", start, end)
sector_panel = pd.DataFrame({sector: df["Close"] for sector, df in sector_data.items()})


def nav_curve(panel: pd.DataFrame, lookback_days: int, top_n: int | None, rebalance_frequency_days: int) -> pd.Series:
    """top_n=None means 'hold everything, equal-weight, rebalanced periodically'
    (the passive benchmark) -- same rebalance dates/frequency as the rotated
    strategy so both curves are built on an identical timeline."""
    trailing_return = panel.pct_change(periods=lookback_days)
    rebalance_dates = panel.index[lookback_days::rebalance_frequency_days]
    if rebalance_dates[-1] != panel.index[-1]:
        rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([panel.index[-1]]))

    nav = [1.0]
    nav_dates = [rebalance_dates[0]]
    for i in range(len(rebalance_dates) - 1):
        period_start, period_end = rebalance_dates[i], rebalance_dates[i + 1]
        row = trailing_return.loc[period_start].dropna()
        if top_n is None:
            held = list(row.index)  # equal-weight ALL sectors with valid data
        else:
            held = list(row.nlargest(top_n).index)
        if not held:
            period_return = 0.0
        else:
            rets = [
                (panel.loc[period_end, s] / panel.loc[period_start, s]) - 1.0
                for s in held
                if pd.notna(panel.loc[period_start, s]) and pd.notna(panel.loc[period_end, s])
            ]
            period_return = float(np.mean(rets)) if rets else 0.0
        nav.append(nav[-1] * (1.0 + period_return))
        nav_dates.append(period_end)
    return pd.Series(nav, index=pd.DatetimeIndex(nav_dates))


def summarize_nav(nav: pd.Series, label: str) -> None:
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    running_max = nav.cummax()
    drawdown = (nav / running_max - 1) * 100
    max_dd = drawdown.min()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else None
    period_returns = nav.pct_change().dropna()
    sharpe_like = (period_returns.mean() / period_returns.std()) if period_returns.std() > 0 else None
    print(f"  {label}: total_return={total_return:.2f}%  CAGR={cagr:.2f}%  "
          f"max_drawdown={max_dd:.2f}%  sharpe_like(period)={sharpe_like:.3f}  n_periods={len(nav)-1}")


CONFIGS = [
    {"lookback_days": 63, "top_n": 3, "rebalance_frequency_days": 21, "label": "ROTATED 63d/top-3/monthly"},
    {"lookback_days": 126, "top_n": 3, "rebalance_frequency_days": 21, "label": "ROTATED 126d/top-3/monthly"},
    {"lookback_days": 63, "top_n": 4, "rebalance_frequency_days": 21, "label": "ROTATED 63d/top-4/monthly"},
    {"lookback_days": 63, "top_n": 3, "rebalance_frequency_days": 63, "label": "ROTATED 63d/top-3/quarterly"},
]

bh_nav = nav_curve(sector_panel, lookback_days=63, top_n=None, rebalance_frequency_days=21)
summarize_nav(bh_nav, "PASSIVE equal-weight-all-11-sectors (rebalanced monthly)")

spy_start_idx = 63
spy_nav = spy["Close"].iloc[spy_start_idx:] / spy["Close"].iloc[spy_start_idx]
summarize_nav(spy_nav, "PASSIVE SPY buy-and-hold")

for cfg in CONFIGS:
    nav = nav_curve(sector_panel, cfg["lookback_days"], cfg["top_n"], cfg["rebalance_frequency_days"])
    summarize_nav(nav, cfg["label"])
