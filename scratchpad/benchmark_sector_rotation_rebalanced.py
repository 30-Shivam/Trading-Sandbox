"""Real-data validation for the new simulate_sector_rotation_rebalanced()
engine (2026-09-16) -- fetches all 11 real SPDR sector ETFs + SPY, runs the
rebalanced rotation strategy, and compares against (a) equal-weight
buy-and-hold of the same 11 sectors and (b) plain SPY buy-and-hold. Not a
permanent CLI tool -- a one-off real-data smoke test, mirroring
benchmark_skew_regime.py's own precedent for validating a genuinely new
mechanism before deciding whether it's worth deeper investment.
"""
import sys
import pandas as pd

sys.path.insert(0, r"D:\Trading-Sandbox")
import swingtrade
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
        print(f"  {sector} ({etf}): {len(df)} rows")

spy = fetch_history("SPY", start, end)
print(f"  SPY: {len(spy)} rows")

sector_panel = pd.DataFrame({sector: df["Close"] for sector, df in sector_data.items()})

CONFIGS = [
    {"lookback_days": 63, "top_n": 3, "rebalance_frequency_days": 21, "label": "63d lookback / top-3 / monthly rebalance"},
    {"lookback_days": 126, "top_n": 3, "rebalance_frequency_days": 21, "label": "126d lookback / top-3 / monthly rebalance"},
    {"lookback_days": 63, "top_n": 4, "rebalance_frequency_days": 21, "label": "63d lookback / top-4 / monthly rebalance"},
    {"lookback_days": 63, "top_n": 3, "rebalance_frequency_days": 63, "label": "63d lookback / top-3 / quarterly rebalance"},
]

bh_trades = swingtrade.compute_equal_weight_buy_and_hold(sector_panel, lookback_days=63)
bh_summary = swingtrade.summarize_trades(bh_trades)
bh_dd = swingtrade.compute_max_drawdown(bh_trades)
print(f"\n=== PASSIVE BENCHMARK: equal-weight buy-and-hold, all 11 sectors ===")
print(f"  {bh_summary}")
print(f"  max_drawdown: {bh_dd}")

spy_entry = float(spy["Close"].iloc[63])
spy_exit = float(spy["Close"].iloc[-1])
spy_return = round((spy_exit - spy_entry) / spy_entry * 100, 2)
print(f"\n=== PASSIVE BENCHMARK: SPY buy-and-hold over the same window ===")
print(f"  total_return_pct: {spy_return}")

for cfg in CONFIGS:
    trades = swingtrade.simulate_sector_rotation_rebalanced(
        sector_panel, cfg["lookback_days"], cfg["top_n"], cfg["rebalance_frequency_days"],
    )
    summary = swingtrade.summarize_trades(trades)
    dd = swingtrade.compute_max_drawdown(trades)
    kr = swingtrade.compute_k_ratio(trades)
    total_pnl = summary.get("total_pnl_pct")
    print(f"\n=== ROTATED: {cfg['label']} ({len(trades)} holding-period trades) ===")
    print(f"  {summary}")
    print(f"  max_drawdown: {dd}  k_ratio: {kr}")
