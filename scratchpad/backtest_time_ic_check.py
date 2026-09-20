"""One-off diagnostic, per user request to "dig into" why backtest-validated
strategies keep showing flat/negative REAL live IC once deployed (RSI v66,
llm_agent, best_ideas composite, regime_switcher/best_ideas_sector_rs/
squeeze_breakout all showed this pattern this session).

Hypothesis: this project's OFFLINE validation gate (benchmark_random_entry.py/
optimize.py) checks whether a strategy's ENTRY TIMING beats a random-entry
baseline on aggregate sharpe_like/win_rate -- but it has never checked
whether the strategy's own Trade_Score actually RANKS which specific trades
will do better or worse (the same rank-IC question ic_tracking.py uses to
judge LIVE skill). A strategy can have genuinely better-than-random entry
timing while its scoring formula has ~zero real discriminating power --
these are different questions, and only the first has ever gated promotion.

Test: compute BACKTEST-time IC (rank_ic between each strategy's own
Trade_Score and realized pnl_pct, over its own real historical backtest
trades) for every strategy with real live IC already known from this
session's trust-floor sweep, and see if backtest IC predicts live IC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config_loader
import ic_tracking
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

CHECKS = [
    ("ma_crossover", config_loader.SMALLMID_MA_CROSSOVER_CONFIG_VERSION, "ma_crossover"),
    ("rsi_mean_reversion (rsi)", config_loader.SMALLMID_RSI_CONFIG_VERSION, "rsi"),
    ("squeeze_breakout", 53, "squeeze_breakout"),
    ("pairs", config_loader.SMALLMID_PAIRS_CONFIG_VERSION, "pairs"),
    ("momentum_rank", 65, "momentum_rank"),
]

pair_price_panels = {}
by_sector = {}
for t in ticker_data:
    by_sector.setdefault(sector_lookup.get(t, "Unknown"), []).append(t)
for sector, members in by_sector.items():
    if len(members) >= 2:
        pair_price_panels[sector] = pd.DataFrame({m: ticker_data[m]["Close"] for m in members})

momentum_panel = pd.DataFrame({t: ticker_data[t]["Close"] for t in ticker_data})

results = []
for label, version, strategy in CHECKS:
    config, source = config_loader.load_config_by_version(version)
    if config is None:
        print(f"{label}: SKIPPED -- {source}")
        continue
    config = swingtrade.TradingConfig(**{**config.to_dict(), "strategy": strategy})
    momentum_rank_frame = (
        swingtrade.compute_momentum_rank_frame(momentum_panel, config.momentum_lookback_days)
        if strategy == "momentum_rank" else None
    )
    trades = swingtrade.run_backtest(
        ticker_data, market_df, start, end, config, sector_lookup=sector_lookup, strategy=strategy,
        pair_price_panels=pair_price_panels if strategy == "pairs" else None,
        momentum_rank_frame=momentum_rank_frame,
    )
    check = ic_tracking.backtest_ic_check(trades, min_trades=10)
    if check["ic"] is None:
        print(f"{label} (v{version}): only {check['n']} resolved trade(s) w/ trade_score -- too thin, skipped.")
        continue
    print(f"{label} (v{version}): n={check['n']} BACKTEST-TIME IC (Trade_Score vs pnl_pct) = {check['ic']}")
    results.append((label, version, check["n"], check["ic"]))

print("\n=== Summary: backtest-time IC vs known real LIVE IC (from this session's trust-floor sweep) ===")
LIVE_IC = {
    "ma_crossover": 0.794, "rsi_mean_reversion (rsi)": 0.621, "squeeze_breakout": -0.134,
    "pairs": -0.010, "momentum_rank": 0.335,
}
for label, version, n, backtest_ic in results:
    live_ic = LIVE_IC.get(label)
    print(f"  {label}: backtest_ic={backtest_ic:.3f} (n={n})  vs  live_ic={live_ic}")
