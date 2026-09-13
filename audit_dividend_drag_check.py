"""
Dividend-drag audit -- real-data check for swingtrade.audit_dividend_drag().

Every settled trade's pnl_pct is computed purely from price movement
(buy_price vs exit_price) -- a real position held THROUGH a stock's
ex-dividend date would, in reality, also receive that dividend payment,
which this backtest currently credits nowhere. Slippage/commission already
model costs you didn't actually avoid; this is the mirror-image gap --
real income never counted, a one-directional bias that always UNDERSTATES
real returns for dividend-paying tickers, never overstates them.

Compares two real universes already built this session: this project's
own low/no-dividend, tech-heavy primary watchlist.txt (expected: small
effect) against the international ADR universe (adr_watchlist.txt, item
137) -- several of which (BP, BTI, HSBC, TotalEnergies, National Grid, ...)
yield 4-8%+, a genuinely different regime.

PURE MEASUREMENT -- does not modify any config or trade data. See
swingtrade.audit_dividend_drag()'s own docstring for the full method.

Usage:
    python audit_dividend_drag_check.py
    python audit_dividend_drag_check.py --tickers BP,BTI,HSBC --strategy ma_crossover
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import config_loader
import swingtrade
from run_backtest import fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
REQUEST_DELAY_SEC = 0.5
MARKET_INDEX_TICKER = "SPY"


def _fetch_dividends(tickers: list[str]) -> dict:
    dividend_history = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            divs = yf.Ticker(ticker).dividends
            if len(divs) > 0:
                dividend_history[ticker] = divs
        except Exception as exc:
            print(f"  [WARN] {ticker} dividends: {exc}", file=sys.stderr)
    return dividend_history


def _run_universe(label: str, watchlist_file: Path, tickers_override, config, market_ohlcv, start, end):
    if tickers_override:
        tickers = tickers_override
    else:
        tickers = read_tickers(watchlist_file)
    sector_lookup = read_ticker_sectors(watchlist_file)

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
    print(f"[{label}] Fetched {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        return None, None

    trades = []
    for ticker, ohlcv in ticker_data.items():
        trades.extend(swingtrade.simulate_ma_crossover_signals(
            ticker, ohlcv, market_ohlcv, start, end, config, sector=sector_lookup.get(ticker, "Unknown"),
        ))
    print(f"[{label}] {len(trades)} real ma_crossover signal(s).")

    print(f"[{label}] Fetching real dividend history for {len(ticker_data)} ticker(s)...")
    dividend_history = _fetch_dividends(list(ticker_data.keys()))
    print(f"[{label}] {len(dividend_history)}/{len(ticker_data)} ticker(s) have real dividend history.")
    return trades, dividend_history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 3y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to check ONE universe instead of both.")
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 3)

    print(f"Fetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
    market_ohlcv = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_ohlcv.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    config, label = config_loader.load_active_config()
    print(f"Testing config: {label} (strategy=ma_crossover)\n")

    tickers_override = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    universes = (
        [("CUSTOM", SCRIPT_DIR / "watchlist.txt", tickers_override)]
        if tickers_override else
        [
            ("PRIMARY (low/no-dividend, tech-heavy)", SCRIPT_DIR / "watchlist.txt", None),
            ("ADR (international, several high-yield)", SCRIPT_DIR / "adr_watchlist.txt", None),
        ]
    )

    for label_name, watchlist_file, override in universes:
        print(f"\n=== {label_name} ===")
        trades, dividend_history = _run_universe(label_name, watchlist_file, override, config, market_ohlcv, start, end)
        if not trades:
            print("  No real signals -- skipping.")
            continue
        result = swingtrade.audit_dividend_drag(trades, dividend_history)
        print(f"  {result}")
        if result["avg_missed_pct_all_trades"]:
            print(f"  On average, this backtest understates real total return by "
                  f"{result['avg_missed_pct_all_trades']}pp per trade across this universe "
                  f"(compare against known cost-sensitivity magnitudes, improvements.txt item 130).")


if __name__ == "__main__":
    main()
