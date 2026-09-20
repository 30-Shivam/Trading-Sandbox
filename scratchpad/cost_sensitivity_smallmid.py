"""One-off stress test, per explicit user request ("how confident are you in
our backtesting... what can we do to make that iron proof") -- re-runs the
SAME real-vs-random check already used to validate ma_crossover/pairs/RSI on
the small/mid-cap universe, but at progressively harsher slippage/commission
assumptions than the current default (slippage_pct=0.001, commission=0.0),
to see whether the GAP (real sharpe_like minus random sharpe_like) -- the
actual evidence any of these strategies has a real edge -- survives more
realistic friction. Reuses optimize.real_vs_random_ratio_check() directly,
zero new production code -- only the config's slippage_pct/commission_pct_per_trade
fields are varied.

config.py's own module docstring already explains why these two fields are
deliberately kept OUT of the Optuna search space ("Letting Optuna tune them
would just teach it to zero out the very realism they exist to add") --
this script does the opposite of that: a human-directed STRESS test at
levels Optuna could never reach on its own, not a search for a favorable
value.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config_loader
import optimize
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

WATCHLIST_FILE = Path(__file__).resolve().parents[1] / "smallmid_watchlist.txt"
end = pd.Timestamp.now().normalize()
start = end - pd.Timedelta(days=365 * 5)

tickers = read_tickers(WATCHLIST_FILE)
sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

print(f"Fetching {len(tickers)} ticker(s) + {MARKET_INDEX_TICKER}, {start.date()}..{end.date()}...")
market_df = fetch_history(MARKET_INDEX_TICKER, start, end)
ticker_data = {}
for t in tickers:
    df = fetch_history(t, start, end)
    if not df.empty:
        ticker_data[t] = df
print(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s).\n")

pair_price_panels = {}
by_sector = {}
for t in ticker_data:
    by_sector.setdefault(sector_lookup.get(t, "Unknown"), []).append(t)
for sector, members in by_sector.items():
    if len(members) >= 2:
        pair_price_panels[sector] = pd.DataFrame({m: ticker_data[m]["Close"] for m in members})

# (slippage_pct, commission_pct_per_trade) -- level 0 is the current
# project-wide default; each subsequent level is deliberately harsher, not
# a realistic-vs-unrealistic binary but a genuine sensitivity SWEEP.
COST_LEVELS = [
    ("current default", 0.001, 0.0),
    ("moderate", 0.003, 0.0005),
    ("harsh", 0.005, 0.001),
    ("very harsh", 0.01, 0.001),
]

CHECKS = [
    ("ma_crossover", config_loader.SMALLMID_MA_CROSSOVER_CONFIG_VERSION, "ma_crossover", None),
    ("pairs", config_loader.SMALLMID_PAIRS_CONFIG_VERSION, "pairs", pair_price_panels),
    ("rsi_mean_reversion", config_loader.SMALLMID_RSI_CONFIG_VERSION, "rsi", None),
]

for label, version, strategy, panels in CHECKS:
    base_config, source = config_loader.load_config_by_version(version)
    if base_config is None:
        print(f"{label}: SKIPPED -- {source}")
        continue
    print(f"\n=== {label} (v{version}) -- cost sensitivity ===")
    for cost_label, slippage, commission in COST_LEVELS:
        config = swingtrade.TradingConfig(**{
            **base_config.to_dict(), "strategy": strategy,
            "slippage_pct": slippage, "commission_pct_per_trade": commission,
        })
        result = optimize.real_vs_random_ratio_check(
            strategy, config, ticker_data, market_df, start, end, sector_lookup,
            pair_price_panels=panels,
        )
        real_sharpe = result["real"].get("sharpe_like")
        random_sharpe = result["random"].get("sharpe_like")
        gap = result["gap"]
        bic = result.get("backtest_ic", {"n": 0, "ic": None})
        print(f"  {cost_label:16s} (slip={slippage:.3f}, comm={commission:.4f}): "
              f"REAL={real_sharpe} RANDOM={random_sharpe} GAP={gap} "
              f"backtest_ic={bic['ic']} (n={bic['n']})")
