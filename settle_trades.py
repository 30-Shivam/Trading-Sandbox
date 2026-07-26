"""
Nightly settlement job (Phase 3).

Walks every unsettled Trade_Signals document forward through actual price
history and resolves it to WIN / LOSS / EXPIRED via swingtrade.settle_trade()
(gap-aware: a stop-loss is a trigger price, not a guaranteed fill -- see
ARCHITECTURE_PLAN.md section 3). Terminal outcomes are written to
Trade_Outcomes and the source Trade_Signals document is marked settled.
Trades that haven't resolved yet are left untouched and simply get re-walked
from scratch on the next run -- there's no incremental state to corrupt, so
this is safe to run as often as you like (once a day, via cron, is enough
since it's all daily-bar data).

Usage:
    python settle_trades.py
"""

import sys
import time

import pandas as pd
import yfinance as yf

import storage
import swingtrade

REQUEST_DELAY_SEC = 0.5   # pause between yfinance calls to avoid rate-limiting
CONFIG = swingtrade.DEFAULT_CONFIG


def fetch_bars_since(ticker: str, signal_date: str) -> pd.DataFrame:
    """Daily OHLCV strictly AFTER signal_date, chronological order."""
    df = yf.download(ticker, start=signal_date, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[df.index > pd.Timestamp(signal_date)]


def settle_one(signal: dict) -> str:
    ticker = signal["ticker"]
    signal_date = signal["signal_date"]
    bars = fetch_bars_since(ticker, signal_date)
    if bars.empty:
        return "no new bars yet"

    result = swingtrade.settle_trade(
        buy_price=signal["buy_price"],
        stop_loss=signal["stop_loss"],
        sell_price=signal["sell_price"],
        bars_since_entry=bars,
        config=CONFIG,
    )

    if result["status"] == "OPEN":
        return "OPEN (still open)"

    storage.log_trade_outcome(ticker, signal_date, signal["buy_price"], result)
    storage.mark_settled(ticker, signal_date)
    return (
        f"{result['status']} ({result['exit_reason']}, "
        f"{result['pnl_pct']:+.2f}%, held {result['holding_days']}d)"
    )


def main():
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    unsettled = storage.get_unsettled_signals()
    print(f"Found {len(unsettled)} unsettled signal(s).")

    counts: dict[str, int] = {}
    for i, signal in enumerate(unsettled):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        ticker = signal["ticker"]
        try:
            outcome = settle_one(signal)
        except Exception as exc:
            outcome = f"ERROR: {exc}"
        print(f"  {ticker} ({signal['signal_date']}): {outcome}")
        if outcome.startswith("ERROR"):
            key = "ERROR"
        elif outcome == "no new bars yet":
            key = "PENDING_NO_DATA"
        else:
            key = outcome.split()[0]  # WIN / LOSS / EXPIRED / OPEN
        counts[key] = counts.get(key, 0) + 1

    print()
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
