"""
Manual config evaluation CLI (Phase 5 companion to optimize.py).

Runs one specific, manually-chosen TradingConfig through the same Phase 4
walk-forward harness optimize.py's Optuna trials use, and reports its
pooled out-of-sample metrics. For validating a manual override -- e.g.
capping a parameter Optuna pinned at a search-space boundary with a more
conservative hand-picked value -- BEFORE writing it as a System_Config
candidate. Never promotes anything itself; use promote_config.py for that.

Usage:
    python evaluate_config.py --tickers NVDA,AMD,INTC --start 2021-06-01 --end 2026-07-26 \
        --rsi-oversold-threshold 52.19 --atr-take-profit-multiplier 1.86 \
        --stop-loss-atr-multiplier 0.5 \
        --write-candidate --notes "trial 38 RSI/ATR-TP, manually capped stop-loss"
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import storage
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--rsi-oversold-threshold", type=float, default=None)
    parser.add_argument("--atr-take-profit-multiplier", type=float, default=None)
    parser.add_argument("--stop-loss-atr-multiplier", type=float, default=None)
    parser.add_argument("--extended-decline-penalty-per-day", type=float, default=None)
    parser.add_argument("--extended-decline-penalty-cap", type=float, default=None)
    parser.add_argument("--write-candidate", action="store_true", help="Write this config to System_Config as a candidate.")
    parser.add_argument("--notes", default="", help="Notes to store with the candidate (with --write-candidate).")
    args = parser.parse_args()

    overrides = {}
    if args.rsi_oversold_threshold is not None:
        overrides["rsi_oversold_threshold"] = args.rsi_oversold_threshold
    if args.atr_take_profit_multiplier is not None:
        overrides["atr_take_profit_multiplier"] = args.atr_take_profit_multiplier
    if args.stop_loss_atr_multiplier is not None:
        overrides["stop_loss_atr_multiplier"] = args.stop_loss_atr_multiplier
    if args.extended_decline_penalty_per_day is not None:
        overrides["extended_decline_penalty_per_day"] = args.extended_decline_penalty_per_day
    if args.extended_decline_penalty_cap is not None:
        overrides["extended_decline_penalty_cap"] = args.extended_decline_penalty_cap
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), **overrides})

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Evaluating config: {overrides}")
    print(f"Fetching history for {len(tickers)} ticker(s) + {MARKET_INDEX_TICKER}, {start.date()}..{end.date()}...")
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

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"Generated {len(folds)} walk-forward fold(s).")
    if not folds:
        print("[ERROR] Date range too short for even one fold.", file=sys.stderr)
        sys.exit(1)

    fold_results = swingtrade.run_walk_forward(ticker_data, market_data, folds, config, sector_lookup=sector_lookup)
    pooled_oos_trades = [t for fr in fold_results for t in fr.out_sample_trades]
    resolved_oos_trades = [t for t in pooled_oos_trades if t["status"] != "OPEN"]
    metrics = swingtrade.summarize_trades(pooled_oos_trades)
    cluster_metrics = swingtrade.summarize_trades_weighted(
        resolved_oos_trades, swingtrade.compute_cluster_weights(resolved_oos_trades)
    )

    baseline_results = swingtrade.run_walk_forward(
        ticker_data, market_data, folds, swingtrade.DEFAULT_CONFIG, sector_lookup=sector_lookup
    )
    baseline_metrics = swingtrade.summarize_trades([t for fr in baseline_results for t in fr.out_sample_trades])

    print()
    print(f"Pooled out-of-sample metrics for this config: {metrics}")
    print(f"Correlation-adjusted (same-day/same-sector counted once): {cluster_metrics}")
    print(f"Baseline (DEFAULT_CONFIG) for comparison:      {baseline_metrics}")

    if args.write_candidate:
        version = storage.write_candidate(config.to_dict(), notes=args.notes, metrics=metrics)
        print()
        print(f"Wrote candidate System_Config version={version} (status=candidate, NOT active).")
        print(f"Review it, then run: python promote_config.py --promote {version}")


if __name__ == "__main__":
    main()
