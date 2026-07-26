"""
Walk-Forward backtest CLI (Phase 4).

Fetches historical OHLCV for the watchlist (+ SPY as the market-uptrend
proxy) via yfinance, runs the swingtrade walk-forward harness across a
sequence of rolling in-sample/out-of-sample folds, and prints per-fold
in-sample vs. out-of-sample performance for the given TradingConfig.

This is what will let Optuna (Phase 5) score a candidate RSI/ATR parameter
set against years of history instantly, instead of waiting months for live
Trade_Outcomes to accumulate -- see swingtrade/backtest.py's module
docstring for the full walk-forward rationale and its one real limitation
(catalyst/earnings awareness is NOT simulated here; yfinance has no
point-in-time historical earnings calendar).

Usage:
    python run_backtest.py
    python run_backtest.py --start 2023-01-01 --end 2026-07-01
    python run_backtest.py --in-sample-days 120 --out-sample-days 20 --step-days 20
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import swingtrade
from watchlist import read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
MARKET_INDEX_TICKER = "SPY"
LOOKBACK_BUFFER_DAYS = 320  # calendar-day buffer before window start, for SMA200 warmup
REQUEST_DELAY_SEC = 0.5


def fetch_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    buffered_start = start - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
    df = yf.download(ticker, start=buffered_start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def print_fold_table(fold_results: list) -> None:
    headers = ["Fold", "In-Sample", "Out-of-Sample", "IS #", "IS Win%", "OOS #", "OOS Win%", "OOS AvgPnL%", "OOS Sharpe"]
    widths = [6, 24, 24, 6, 9, 6, 9, 13, 11]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for i, fr in enumerate(fold_results, start=1):
        is_range = f"{fr.fold.in_sample_start.date()}..{fr.fold.in_sample_end.date()}"
        oos_range = f"{fr.fold.out_sample_start.date()}..{fr.fold.out_sample_end.date()}"
        ism, oom = fr.in_sample_metrics, fr.out_sample_metrics
        row = [
            str(i), is_range, oos_range,
            str(ism["trade_count"]), _fmt(ism["win_rate"]),
            str(oom["trade_count"]), _fmt(oom["win_rate"]),
            _fmt(oom["avg_pnl_pct"]), _fmt(oom["sharpe_like"]),
        ]
        print("".join(v.ljust(w) for v, w in zip(row, widths)))


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 1y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)

    print(f"Backtesting {len(tickers)} ticker(s) over {start.date()} to {end.date()}")

    print(f"Fetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        print(f"[{i + 1}/{len(tickers)}] {ticker} ...", end=" ")
        try:
            df = fetch_history(ticker, start, end)
            if df.empty:
                print("no data")
                continue
            ticker_data[ticker] = df
            print("ok")
        except Exception as exc:
            print("skipped")
            print(f"  [WARN] {ticker}: {exc}", file=sys.stderr)

    print(f"\nFetched history for {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        print("[ERROR] No ticker data available to backtest.", file=sys.stderr)
        sys.exit(1)

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"Generated {len(folds)} walk-forward fold(s) "
          f"(in-sample={args.in_sample_days}d, out-of-sample={args.out_sample_days}d, step={args.step_days}d).")
    if not folds:
        print("[ERROR] Date range too short to generate even one fold with these window sizes.", file=sys.stderr)
        sys.exit(1)

    fold_results = swingtrade.run_walk_forward(ticker_data, market_data, folds, swingtrade.DEFAULT_CONFIG)

    print()
    print_fold_table(fold_results)

    all_oos_trades = [t for fr in fold_results for t in fr.out_sample_trades]
    overall = swingtrade.summarize_trades(all_oos_trades)
    print()
    print(f"Aggregate out-of-sample performance across all {len(fold_results)} fold(s): {overall}")
    print()
    print("NOTE: catalyst/earnings awareness is not simulated -- see swingtrade/backtest.py.")
    print("This is a mechanical replay of historical price data, not a forecast.")


if __name__ == "__main__":
    main()
