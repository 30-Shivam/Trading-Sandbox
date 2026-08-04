"""
Standalone scheduled scan + signal-logging service (Phase 7).

Runs the exact same fetch -> compute -> score pipeline as the Streamlit
dashboard (dip_buy_analyzer.py), via the shared `market_data`/`swingtrade`/
`config_loader` modules, but as a one-shot headless process instead of
something that only runs when a human happens to have the dashboard open.
This is the concrete gap it closes: previously, Trade_Signals only
accumulated on days someone opened the dashboard and it happened to compute
a Buy/Strong Buy -- starving the settlement job (Phase 3) and the learning
loop (Phase 5) of data whenever nobody looked. Running this on a schedule
(cron locally, a Kubernetes CronJob once Phase 8 containerizes it) makes
signal generation independent of the UI.

Not rewritten in Go (amendment #1) -- see market_data.py's docstring.

Position sizing here uses --position-budget (default matches the dashboard's
default) purely to populate the shares_to_buy/est_cost fields Trade_Signals
expects. This script does no capital allocation across signals -- greedily
spending a cash pool across today's top signals is a personal, interactive
decision that belongs in the dashboard, not something to bake into a
scheduled job or log as if it were part of the technical signal.

Usage:
    python ingest.py
    python ingest.py --watchlist my_list.txt --position-budget 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import ai_context
import config_loader
import market_data
import storage
import swingtrade
from watchlist import read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
DEFAULT_POSITION_BUDGET = 250.0


def run(watchlist_path: Path, position_budget: float, with_ai_context: bool = False) -> int:
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    config, config_source = config_loader.load_active_config()
    print(f"{config_source} (strategy={config.strategy})")

    tickers = tuple(read_tickers(watchlist_path))
    if not tickers:
        print(f"[ERROR] no tickers found in {watchlist_path}", file=sys.stderr)
        return 1
    print(f"Scanning {len(tickers)} ticker(s)...")

    try:
        market_uptrend, market_close, market_sma200 = market_data.check_market_uptrend(config)
    except Exception as exc:
        print(f"[ERROR] could not evaluate {market_data.MARKET_INDEX_TICKER} macro trend: {exc}", file=sys.stderr)
        return 1

    if not market_uptrend:
        print(
            f"{market_data.MARKET_INDEX_TICKER} is in a macro downtrend "
            f"(Last_Close {market_close:.2f} < SMA200 {market_sma200:.2f}) -- "
            "skipping scan, no signals logged."
        )
        return 0

    results, skipped = market_data.scan_tickers(tickers, config)
    if not results:
        print("[ERROR] no tickers were successfully analyzed.", file=sys.stderr)
        for ticker, reason in skipped:
            print(f"  skipped {ticker}: {reason}", file=sys.stderr)
        return 1

    results_df = pd.DataFrame(results)
    results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(config.fractional_share_decimals)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
    if config.strategy == "breakout":
        results_df = swingtrade.add_breakout_trade_score(results_df, config)
    else:
        results_df = swingtrade.add_trade_score(results_df, config)

    logged_count = storage.log_trade_signals(results_df, config.to_dict())
    strong_buys = int((results_df["Signal"] == "Strong Buy").sum())
    buys = int((results_df["Signal"] == "Buy").sum())
    print(f"Analyzed {len(results_df)}/{len(tickers)} ticker(s): {strong_buys} Strong Buy, {buys} Buy.")
    print(f"Logged {logged_count} signal(s) to MongoDB.")

    if with_ai_context:
        if not ai_context.is_available():
            print("[WARN] --with-ai-context requested but unavailable (set GEMINI_API_KEY).", file=sys.stderr)
        else:
            signal_rows = results_df[results_df["Signal"].isin(["Strong Buy", "Buy"])]
            for _, row in signal_rows.iterrows():
                headlines = market_data.get_multi_headlines(row["Ticker"])
                summary = ai_context.summarize_ticker_context(row["Ticker"], row["Signal"], headlines)
                if summary:
                    print(f"\n[{row['Ticker']} ({row['Signal']})] {summary}")

    if skipped:
        print(f"Skipped {len(skipped)} ticker(s):")
        for ticker, reason in skipped:
            print(f"  {ticker}: {reason}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Scan the watchlist and log Strong Buy/Buy signals to MongoDB.")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST_FILE)
    parser.add_argument("--position-budget", type=float, default=DEFAULT_POSITION_BUDGET)
    parser.add_argument(
        "--with-ai-context", action="store_true",
        help="Print an AI-generated news summary (informational only, not a rating) for each "
             "Strong Buy/Buy signal found. Requires GEMINI_API_KEY (free tier); degrades to a "
             "warning if unavailable rather than failing the scan.",
    )
    args = parser.parse_args()
    sys.exit(run(args.watchlist, args.position_budget, args.with_ai_context))


if __name__ == "__main__":
    main()
