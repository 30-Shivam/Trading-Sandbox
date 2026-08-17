"""
Sector relative-strength -- backtest feasibility check.

Question: does a ticker's own SECTOR outperforming or lagging the broader
market (SPY) at signal time predict trade quality, independent of any one
strategy's own edge? Grounded in a real finding from item 65's v54-vs-v51
diagnostic: ma_crossover's edge concentrated in defensive/low-volatility
sectors and broke down in higher-volatility growth/cyclical ones -- sector
context looked like it mattered, just never isolated and tested directly.

Fully backtestable (pure historical OHLCV, no lookahead problem), unlike
regime_switcher.py -- so per strategy_validation_pipeline.txt, this goes
through a real check before any live filter is added.

Reuses swingtrade.compute_relative_strength() unchanged -- it already
computes "series A's return minus series B's return over a window," today
used ticker-vs-SPY (breakout_relative_strength_min /
squeeze_breakout_relative_strength_min). Feeding it a SECTOR ETF in place
of the ticker gives sector relative strength for free, same formula.

Method: run each of the 3 live strategies' own simulate_*_signals() against
their own LIVE config over the full watchlist/5-year window (same trades
benchmark_consensus_signal.py generates). For every trade, compute the
trade's own sector's relative strength (sector ETF return minus SPY return
over --lookback-days, default 63 trading days / ~3 months -- deliberately
longer than any strategy's own ~20-day trigger window, since this tests
broader context, not the same-window ticker-level Relative_Strength that
already exists as an optional filter) as of the trade's signal_date. Bucket
into "tailwind" (sector beating the market) vs. "headwind" (sector lagging
or tied), same ticker-holdout split + cluster-adjusted weighting as every
other benchmark here.

Decision gate: tailwind must beat headwind on TUNE *and* HOLDOUT, not just
ALL, to count as real -- same discipline every other check this session has
used. See improvements.txt for the write-up either way.

Usage:
    python benchmark_sector_relative_strength.py
    python benchmark_sector_relative_strength.py --lookback-days 126
    python benchmark_sector_relative_strength.py --tickers NVDA,AMD,XOM --holdout-frac 0
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import swingtrade
from swingtrade.levels import compute_relative_strength
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import SECTOR_ETF, read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

LIVE_STRATEGY_SIMULATORS = {
    "breakout": swingtrade.simulate_breakout_signals,
    "squeeze_breakout": swingtrade.simulate_squeeze_breakout_signals,
    "ma_crossover": swingtrade.simulate_ma_crossover_signals,
}


def load_live_configs() -> dict[str, tuple[swingtrade.TradingConfig, str]]:
    breakout_config, breakout_label = config_loader.load_active_config()
    squeeze_config, squeeze_reason = config_loader.load_config_by_version(39)
    ma_crossover_config, ma_reason = config_loader.load_config_by_version(51)
    if squeeze_config is None:
        print(f"[ERROR] Could not load squeeze_breakout v39: {squeeze_reason}", file=sys.stderr)
        sys.exit(1)
    if ma_crossover_config is None:
        print(f"[ERROR] Could not load ma_crossover v51: {ma_reason}", file=sys.stderr)
        sys.exit(1)
    return {
        "breakout": (breakout_config, breakout_label),
        "squeeze_breakout": (squeeze_config, "v39 (secondary, live)"),
        "ma_crossover": (ma_crossover_config, "v51 (secondary, live)"),
    }


def summarize(trades: list[dict]) -> dict:
    resolved = [t for t in trades if t["status"] != "OPEN"]
    weights = swingtrade.compute_cluster_weights(resolved)
    return swingtrade.summarize_trades_weighted(resolved, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--holdout-frac", type=float, default=0.25, help="Same ticker-holdout split as optimize.py. 0 disables.")
    parser.add_argument(
        "--holdout-seeds", default=None,
        help="Comma-separated holdout seeds to average TUNE/HOLDOUT over (a single fixed-seed split "
             "carries real sampling noise -- see improvements.txt item 69). "
             f"Default: {','.join(str(s) for s in DEFAULT_HOLDOUT_SEEDS)} (10 seeds).",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=63,
        help="Sector-vs-SPY relative-strength lookback window in trading days. "
             "Default 63 (~3 months, standard sector-momentum convention) -- "
             "deliberately longer than any strategy's own ~20-day trigger window.",
    )
    args = parser.parse_args()

    holdout_seeds = (
        [int(s.strip()) for s in args.holdout_seeds.split(",") if s.strip()]
        if args.holdout_seeds else list(DEFAULT_HOLDOUT_SEEDS)
    )

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 5)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    configs = load_live_configs()
    print("Testing LIVE configs:")
    for strategy, (config, label) in configs.items():
        print(f"  {strategy}: {label}")

    print(f"\nFetching {MARKET_INDEX_TICKER} (market proxy) + {len(SECTOR_ETF)} sector ETFs...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    sector_etf_data = {}
    for i, (sector, etf) in enumerate(SECTOR_ETF.items()):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        df = fetch_history(etf, start, end)
        if df.empty:
            print(f"[ERROR] No data returned for sector ETF {etf} ({sector})", file=sys.stderr)
            sys.exit(1)
        sector_etf_data[sector] = df
    print(f"Fetched {len(sector_etf_data)}/{len(SECTOR_ETF)} sector ETF(s).")

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_history(ticker, start, end)
            if not df.empty:
                ticker_data[ticker] = df
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}", file=sys.stderr)
    print(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        print("[ERROR] No ticker data available.", file=sys.stderr)
        sys.exit(1)

    if args.holdout_frac > 0:
        print(f"Ticker holdout: frac={args.holdout_frac}, averaging TUNE/HOLDOUT over "
              f"{len(holdout_seeds)} seeds ({holdout_seeds}) -- see improvements.txt item 69.")

    print(f"\nSimulating {len(LIVE_STRATEGY_SIMULATORS)} live strategies for "
          f"{len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    all_trades = []
    for strategy, sim_fn in LIVE_STRATEGY_SIMULATORS.items():
        config, _ = configs[strategy]
        strategy_trades = []
        for ticker, ohlcv in ticker_data.items():
            sector = sector_lookup.get(ticker, "Unknown")
            trades = sim_fn(ticker, ohlcv, market_data, start, end, config, sector=sector)
            strategy_trades.extend(trades)
        print(f"  {strategy}: {len(strategy_trades)} signal(s)")
        all_trades.extend(strategy_trades)

    print(f"\nComputing sector relative strength (lookback={args.lookback_days}d) for each trade...")
    unmapped_sectors = set()
    missing_rel_strength = 0
    tailwind_trades = []
    headwind_trades = []
    for t in all_trades:
        sector = t.get("sector", "Unknown")
        etf_df = sector_etf_data.get(sector)
        if etf_df is None:
            unmapped_sectors.add(sector)
            continue
        as_of = pd.Timestamp(t["signal_date"])
        etf_slice = etf_df.loc[:as_of]
        market_slice = market_data.loc[:as_of]
        rel_strength = compute_relative_strength(etf_slice, market_slice, args.lookback_days)
        if rel_strength is None:
            missing_rel_strength += 1
            continue
        t["_sector_relative_strength"] = rel_strength
        (tailwind_trades if rel_strength > 0 else headwind_trades).append(t)

    if unmapped_sectors:
        print(f"  [WARN] {len(unmapped_sectors)} sector(s) with no ETF mapping, skipped: {sorted(unmapped_sectors)}")
    if missing_rel_strength:
        print(f"  [WARN] {missing_rel_strength} trade(s) skipped -- not enough sector/market history yet.")
    print(f"  Tailwind trades (sector beating SPY): {len(tailwind_trades)}")
    print(f"  Headwind trades (sector lagging/tied SPY): {len(headwind_trades)}")

    print("\n=== ALL TICKERS ===")
    print(f"  TAILWIND (sector > SPY): {summarize(tailwind_trades)}")
    print(f"  HEADWIND (sector <= SPY): {summarize(headwind_trades)}")

    if args.holdout_frac > 0:
        tailwind_tune_avg, tailwind_holdout_avg = average_holdout_summary(
            tailwind_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        headwind_tune_avg, headwind_holdout_avg = average_holdout_summary(
            headwind_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )

        print(f"\n=== TUNE (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  TAILWIND: {tailwind_tune_avg}")
        print(f"  HEADWIND: {headwind_tune_avg}")

        print(f"\n=== HOLDOUT (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  TAILWIND: {tailwind_holdout_avg}")
        print(f"  HEADWIND: {headwind_holdout_avg}")

    print()
    print("Decision gate: TAILWIND needs to beat HEADWIND on sharpe_like/win_rate on BOTH")
    print("TUNE and HOLDOUT (averaged across seeds, not a single draw), not just ALL, to")
    print("count as real evidence sector context carries information beyond any one")
    print("strategy's own edge. A HOLDOUT-only failure means stop here -- do not add a live")
    print("sector-relative-strength filter on top of this.")


if __name__ == "__main__":
    main()
