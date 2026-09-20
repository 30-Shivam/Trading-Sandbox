"""Cheap out-of-sample-style check on the most promising sector-rotation
config (63d lookback / top-3 / quarterly rebalance) from
nav_curve_sector_rotation.py's own real result: does its edge over the
passive equal-weight benchmark hold in BOTH independent halves of the
2019-2026 window, or is it concentrated in (driven by) just one stretch?
Reuses the exact same nav_curve()/summarize_nav() logic, just re-run on
two non-overlapping calendar slices of the SAME already-fetched data.
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"D:\Trading-Sandbox")
from run_backtest import fetch_history
from watchlist import SECTOR_ETF

start = pd.Timestamp("2019-01-01")
end = pd.Timestamp("2026-09-15")

print("Fetching 11 real sector ETFs...")
sector_data = {}
for sector, etf in SECTOR_ETF.items():
    df = fetch_history(etf, start, end)
    if not df.empty:
        sector_data[sector] = df
sector_panel = pd.DataFrame({sector: df["Close"] for sector, df in sector_data.items()})

midpoint = sector_panel.index[len(sector_panel) // 2]
print(f"Full range: {sector_panel.index[0].date()} .. {sector_panel.index[-1].date()}")
print(f"Midpoint (split point): {midpoint.date()}")


def nav_curve(panel: pd.DataFrame, lookback_days: int, top_n: int | None, rebalance_frequency_days: int) -> pd.Series:
    trailing_return = panel.pct_change(periods=lookback_days)
    rebalance_dates = panel.index[lookback_days::rebalance_frequency_days]
    if len(rebalance_dates) == 0 or rebalance_dates[-1] != panel.index[-1]:
        rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([panel.index[-1]]))
    nav, nav_dates = [1.0], [rebalance_dates[0]]
    for i in range(len(rebalance_dates) - 1):
        p0, p1 = rebalance_dates[i], rebalance_dates[i + 1]
        row = trailing_return.loc[p0].dropna()
        held = list(row.index) if top_n is None else list(row.nlargest(top_n).index)
        if held:
            rets = [
                (panel.loc[p1, s] / panel.loc[p0, s]) - 1.0
                for s in held if pd.notna(panel.loc[p0, s]) and pd.notna(panel.loc[p1, s])
            ]
            period_return = float(np.mean(rets)) if rets else 0.0
        else:
            period_return = 0.0
        nav.append(nav[-1] * (1.0 + period_return))
        nav_dates.append(p1)
    return pd.Series(nav, index=pd.DatetimeIndex(nav_dates))


def summarize_nav(nav: pd.Series, label: str) -> float:
    if len(nav) < 2:
        print(f"  {label}: insufficient periods ({len(nav)})")
        return None
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    running_max = nav.cummax()
    max_dd = ((nav / running_max - 1) * 100).min()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else None
    print(f"  {label}: total_return={total_return:.2f}%  CAGR={cagr:.2f}%  max_drawdown={max_dd:.2f}%  n_periods={len(nav)-1}")
    return total_return


for label, sliced in [
    ("FIRST HALF", sector_panel[sector_panel.index <= midpoint]),
    ("SECOND HALF", sector_panel[sector_panel.index >= midpoint]),
]:
    print(f"\n=== {label} ({sliced.index[0].date()} .. {sliced.index[-1].date()}) ===")
    bh_nav = nav_curve(sliced, lookback_days=63, top_n=None, rebalance_frequency_days=63)
    bh_return = summarize_nav(bh_nav, "PASSIVE equal-weight (quarterly-measured)")
    rot_nav = nav_curve(sliced, lookback_days=63, top_n=3, rebalance_frequency_days=63)
    rot_return = summarize_nav(rot_nav, "ROTATED 63d/top-3/quarterly")
    if bh_return is not None and rot_return is not None:
        print(f"  GAP (rotated - passive): {rot_return - bh_return:+.2f}pp")
