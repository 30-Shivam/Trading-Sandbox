"""
Random-entry benchmark: does RSI-oversold timing carry any real predictive
information, or is this system's backtested edge coming entirely from the
stop/target payoff structure (tight stop, wide target) plus general market
drift?

Three independent rigor upgrades this session (entry-fill timing realism,
tune-vs-ticker-holdout, v3-vs-ticker-holdout -- see improvements.txt) each
collapsed the apparent edge toward zero. Before layering more indicators on
top of the RSI signal, this answers the more basic question directly: for
each ticker, run the REAL RSI-oversold strategy to get its signal count,
then run swingtrade.simulate_random_entries() to fire the SAME NUMBER of
trades on RANDOMLY chosen days instead -- identical universe (macro
uptrend + liquidity gates), identical entry-fill/stop/target mechanics,
only WHICH DAY differs. If RSI-timed entries don't meaningfully beat this
matched-count random baseline, the RSI signal itself carries little real
information.

Applies the same ticker-holdout split as optimize.py (--holdout-frac,
--holdout-seed) so the comparison also holds up (or doesn't) on tickers
never used to pick the config being tested. Uses cluster-adjusted stats
(same-day/same-sector correlation) like run_backtest.py/evaluate_config.py
-- no recency weighting, since this is a single fixed window, not a
walk-forward search (nothing here is being tuned against this data, so the
single-window curve-fitting concern that WFO guards against elsewhere
doesn't apply to a straight before/after comparison of two fixed rules).

Usage:
    python benchmark_random_entry.py
    python benchmark_random_entry.py --start 2021-06-01 --end 2026-07-26 --seed 1
    python benchmark_random_entry.py --tickers NVDA,AMD,INTC --holdout-frac 0
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

import storage
import swingtrade
from optimize import split_tickers_holdout
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5


def load_config_to_test() -> tuple[swingtrade.TradingConfig, str]:
    """The currently-active System_Config -- this benchmark exists to ask
    "does the live signal carry real information," so it should test the
    live params, not a hardcoded snapshot that could silently go stale."""
    try:
        db = storage.get_db()
        doc = db[storage.system_config.COLLECTION_NAME].find_one({"status": "active"})
    except storage.MongoNotConfigured:
        doc = None
    if doc is None:
        return swingtrade.DEFAULT_CONFIG, "DEFAULT_CONFIG (no active config found in Mongo)"
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), **doc["params"]})
    return config, f"v{doc['version']} (active)"


def summarize(trades: list[dict]) -> dict:
    resolved = [t for t in trades if t["status"] != "OPEN"]
    weights = swingtrade.compute_cluster_weights(resolved)
    return swingtrade.summarize_trades_weighted(resolved, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the random-entry day selection.")
    parser.add_argument("--holdout-frac", type=float, default=0.25, help="Same ticker-holdout split as optimize.py. 0 disables.")
    parser.add_argument("--holdout-seed", type=int, default=42, help="Same holdout seed as optimize.py's real runs.")
    args = parser.parse_args()

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

    config, config_label = load_config_to_test()
    print(f"Testing config: {config_label}")
    print(f"  rsi_oversold_threshold={config.rsi_oversold_threshold}, "
          f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
          f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")

    print(f"\nFetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

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

    tune_tickers, holdout_tickers = split_tickers_holdout(
        list(ticker_data.keys()), sector_lookup, args.holdout_frac, args.holdout_seed
    )
    if holdout_tickers:
        print(f"Ticker holdout split (seed={args.holdout_seed}, frac={args.holdout_frac}): "
              f"{len(tune_tickers)} tune / {len(holdout_tickers)} holdout.")
    holdout_set = set(holdout_tickers)

    rng = random.Random(args.seed)

    real_trades = []
    random_trades = []
    real_counts = {}
    print(f"\nSimulating REAL RSI-timed strategy and matched-count RANDOM-entry baseline "
          f"for {len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    for i, (ticker, ohlcv) in enumerate(ticker_data.items()):
        sector = sector_lookup.get(ticker, "Unknown")
        real = swingtrade.simulate_signals(ticker, ohlcv, market_data, start, end, config, sector=sector)
        real_trades.extend(real)
        real_counts[ticker] = len(real)

        rand = swingtrade.simulate_random_entries(
            ticker, ohlcv, market_data, start, end, len(real), rng, config, sector=sector
        )
        random_trades.extend(rand)

    total_real = sum(real_counts.values())
    print(f"Real RSI signals: {total_real} across {len(ticker_data)} ticker(s) "
          f"(entry-fill realized: {len(real_trades)}). "
          f"Random baseline (matched count per ticker): {len(random_trades)} entries filled.")

    def split(trades):
        tune = [t for t in trades if t["ticker"] not in holdout_set]
        holdout = [t for t in trades if t["ticker"] in holdout_set]
        return tune, holdout

    real_tune, real_holdout = split(real_trades)
    random_tune, random_holdout = split(random_trades)

    print("\n=== ALL TICKERS ===")
    print(f"  REAL   (RSI-timed):    {summarize(real_trades)}")
    print(f"  RANDOM (matched count): {summarize(random_trades)}")

    if holdout_tickers:
        print(f"\n=== TUNE tickers ({len(tune_tickers)}) ===")
        print(f"  REAL   (RSI-timed):    {summarize(real_tune)}")
        print(f"  RANDOM (matched count): {summarize(random_tune)}")

        print(f"\n=== HOLDOUT tickers ({len(holdout_tickers)}) ===")
        print(f"  REAL   (RSI-timed):    {summarize(real_holdout)}")
        print(f"  RANDOM (matched count): {summarize(random_holdout)}")

    print()
    print("If REAL's sharpe_like/win_rate isn't meaningfully better than RANDOM's (same trade")
    print("count, same universe, same stop/target/holding-period structure), RSI-oversold timing")
    print("is not adding real predictive information beyond the payoff structure + market drift.")


if __name__ == "__main__":
    main()
