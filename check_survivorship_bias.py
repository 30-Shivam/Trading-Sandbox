"""
Survivorship-bias cross-check (backtest_diagnostic.txt follow-up).

Every backtest in this project runs against watchlist.txt, which was
curated TODAY with the benefit of knowing which tickers are still large,
liquid, and interesting -- projected backward across the whole backtest
window, that's a bias toward names that turned out fine. This script
measures how big that effect actually is, rather than just asserting it
exists: it runs the SAME active config through the SAME walk-forward
harness against two same-sized random ticker samples over the same window --

  1. A random sample from watchlist.txt (today's curated list).
  2. A random sample from the ACTUAL S&P 500 membership as of --start
     (sp500_historical_components.csv, see sp500_membership.py) -- a
     point-in-time universe that wasn't cherry-picked with hindsight.

Also reports how many of the point-in-time sample are even still fetchable
via yfinance today -- tickers that fail are themselves survivorship
evidence (delisted, acquired, renamed since --start).

This does NOT replace the watchlist or fix survivorship bias -- it
quantifies it, as a magnitude check on how optimistic every other backtest
number in this project should be assumed to be.

Usage:
    python check_survivorship_bias.py
    python check_survivorship_bias.py --sample-size 40 --start 2021-06-01
"""

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import sp500_membership
import storage
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5


def fetch_ticker_data(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, label: str) -> dict:
    print(f"Fetching {len(tickers)} {label} ticker(s), {start.date()}..{end.date()}...")
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
    print(f"  -> {len(ticker_data)}/{len(tickers)} fetched successfully.")
    return ticker_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2021-06-01")
    parser.add_argument("--end", default="2025-08-23", help="Default matches the dataset's last snapshot.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    rng = random.Random(args.seed)

    try:
        config, source = config_loader.load_active_config()
    except storage.MongoNotConfigured:
        config, source = swingtrade.DEFAULT_CONFIG, "MongoDB not configured -- using built-in defaults."
    print(f"Config: {source}")

    watchlist_tickers = [t for t in read_tickers(WATCHLIST_FILE) if "." not in t]  # drop CSU.TO etc.
    watchlist_sample = rng.sample(watchlist_tickers, min(args.sample_size, len(watchlist_tickers)))
    print(f"Watchlist sample ({len(watchlist_sample)}): {', '.join(sorted(watchlist_sample))}")

    sp500_tickers = sp500_membership.get_membership_asof(start)
    print(f"\nS&P 500 membership as of {start.date()}: {len(sp500_tickers)} tickers.")
    sp500_sample = rng.sample(sp500_tickers, min(args.sample_size, len(sp500_tickers)))
    print(f"Point-in-time sample ({len(sp500_sample)}): {', '.join(sorted(sp500_sample))}")

    print()
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    watchlist_data = fetch_ticker_data(watchlist_sample, start, end, "watchlist")
    sp500_data = fetch_ticker_data(sp500_sample, start, end, "S&P 500 point-in-time")

    unfetchable = len(sp500_sample) - len(sp500_data)
    print(
        f"\n{unfetchable}/{len(sp500_sample)} point-in-time S&P 500 tickers were NOT fetchable via yfinance "
        f"today (delisted/acquired/renamed since {start.date()}) -- survivorship evidence in its own right."
    )

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"\nGenerated {len(folds)} walk-forward fold(s).")
    if not folds:
        print("[ERROR] Date range too short for even one fold.", file=sys.stderr)
        sys.exit(1)

    print("\nRunning walk-forward on watchlist sample...")
    wl_results = swingtrade.run_walk_forward(watchlist_data, market_data, folds, config)
    wl_metrics = swingtrade.summarize_trades([t for fr in wl_results for t in fr.out_sample_trades])

    print("Running walk-forward on point-in-time S&P 500 sample...")
    sp_results = swingtrade.run_walk_forward(sp500_data, market_data, folds, config)
    sp_metrics = swingtrade.summarize_trades([t for fr in sp_results for t in fr.out_sample_trades])

    print()
    print(f"=== Today's watchlist.txt sample ({len(watchlist_data)} tickers) ===")
    print(wl_metrics)
    print()
    print(f"=== Point-in-time S&P 500 sample ({len(sp500_data)} tickers, as of {start.date()}) ===")
    print(sp_metrics)

    if wl_metrics["win_rate"] is not None and sp_metrics["win_rate"] is not None:
        win_delta = wl_metrics["win_rate"] - sp_metrics["win_rate"]
        sharpe_delta = (wl_metrics["sharpe_like"] or 0) - (sp_metrics["sharpe_like"] or 0)
        print()
        print(f"Win-rate delta (watchlist - point-in-time S&P 500): {win_delta:+.2f}pp")
        print(f"Sharpe-like delta (watchlist - point-in-time S&P 500): {sharpe_delta:+.3f}")
        print("A meaningfully positive delta means the current watchlist's curation is itself")
        print("inflating backtested performance beyond what a genuinely unbiased universe shows.")


if __name__ == "__main__":
    main()
