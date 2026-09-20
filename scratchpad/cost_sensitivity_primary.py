"""Same stress test as cost_sensitivity_smallmid.py, but for the PRIMARY
watchlist -- specifically ma_crossover v71, the currently ACTIVE,
capital-eligible config, and Mean-Reversion Pairs v58, also capital-eligible.
See that script's own docstring for the full rationale."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config_loader
import optimize
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

WATCHLIST_FILE = Path(__file__).resolve().parents[1] / "watchlist.txt"
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

COST_LEVELS = [
    ("current default", 0.001, 0.0),
    ("moderate", 0.003, 0.0005),
    ("harsh", 0.005, 0.001),
    ("very harsh", 0.01, 0.001),
]

CHECKS = [
    ("ma_crossover (ACTIVE)", 71, "ma_crossover", None),
    ("pairs (Mean-Reversion Pairs, LIVE)", 58, "pairs", pair_price_panels),
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
