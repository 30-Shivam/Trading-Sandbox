"""One-off diagnostic (not committed): investigates the RSI v66 random-
baseline fill-count anomaly flagged 2026-08-31 -- benchmark_random_entry.py
requested a random-day sample matched to REAL's own trade count, but only a
small fraction of those random attempts actually filled (real's own report:
~9320 real filled trades vs only ~1786 of the matched-count random draw
actually filling).

Mechanism (confirmed by reading the code before running anything): RSI's
real buy_signal requires `last_close <= buy_price` (buy_price = trailing
support_lookback_days low) as a PRECONDITION to fire at all -- so on a real
signal day, price is already at/below the resting limit order, and
_find_entry_fill() almost always fills near-immediately. simulate_random_
entries() picks ANY macro-uptrend/liquid day as a candidate, with no such
precondition -- most random days start well above the trailing low, so
requiring price to fall to it within max_entry_wait_days is a much rarer
event. This script does NOT change that mechanism (that's expected, honest
behavior of the check) -- it tests the actual open question: does the small
REALIZED random sample (rather than the full population of days that COULD
fill if attempted) give a biased or just a noisier answer than an
exhaustive draw would?

Method: for each ticker, run the REAL strategy, the CURRENT random draw
(n_trades = len(real), the same convention benchmark_random_entry.py uses),
and an EXHAUSTIVE random draw (n_trades set far above the total eligible-day
count, so effectively every eligible day gets a fill attempt -- the maximum
achievable random-fill sample). Compares aggregate win_rate/sharpe_like
between the small and exhaustive random samples -- if they agree, the small
sample was just noisier, not biased, and the "RSI loses to random" finding
stands unweakened. If they disagree meaningfully, that's a real finding.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import pandas as pd

import storage
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

REQUEST_DELAY_SEC = 0.3
WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "watchlist.txt"


def summarize(trades):
    return swingtrade.summarize_trades(trades)


def main():
    storage.mongo.get_db()
    doc66 = storage.system_config.get_config_by_version(66)
    config = swingtrade.TradingConfig.from_dict(doc66["params"])
    print(f"RSI v66: max_entry_wait_days={config.max_entry_wait_days}, "
          f"support_lookback_days={config.support_lookback_days}, "
          f"rsi_oversold_threshold={config.rsi_oversold_threshold}")

    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=365 * 5)

    tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)
    print(f"Fetching history for {MARKET_INDEX_TICKER} + {len(tickers)} ticker(s), {start.date()}..{end.date()}...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)

    real_trades, random_small_trades, random_exhaustive_trades = [], [], []
    rng_small = random.Random(1)
    rng_exhaustive = random.Random(1)
    n_fetched = 0
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            ohlcv = fetch_history(ticker, start, end)
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}")
            continue
        if ohlcv.empty:
            continue
        n_fetched += 1
        sector = sector_lookup.get(ticker, "Unknown")

        real = swingtrade.simulate_signals(ticker, ohlcv, market_data, start, end, config, sector=sector)
        real_trades.extend(real)

        small = swingtrade.simulate_random_entries(
            ticker, ohlcv, market_data, start, end, len(real), rng_small, config, sector=sector
        )
        random_small_trades.extend(small)

        # Effectively unlimited request -- simulate_random_entries() internally
        # caps at min(n_trades, len(candidates)), so this realizes every
        # eligible candidate day's fill attempt, not just a matched-count subset.
        exhaustive = swingtrade.simulate_random_entries(
            ticker, ohlcv, market_data, start, end, 10**9, rng_exhaustive, config, sector=sector
        )
        random_exhaustive_trades.extend(exhaustive)

    print(f"\nFetched {n_fetched}/{len(tickers)} ticker(s).")
    print(f"\nREAL filled trades:               {len(real_trades)}")
    print(f"RANDOM (matched-count, small):     {len(random_small_trades)}  "
          f"({len(random_small_trades) / max(len(real_trades), 1):.1%} of real's count)")
    print(f"RANDOM (exhaustive, every candidate attempted): {len(random_exhaustive_trades)}  "
          f"({len(random_exhaustive_trades) / max(len(real_trades), 1):.1%} of real's count)")

    print()
    print(f"REAL:                {summarize(real_trades)}")
    print(f"RANDOM (small):      {summarize(random_small_trades)}")
    print(f"RANDOM (exhaustive): {summarize(random_exhaustive_trades)}")


if __name__ == "__main__":
    main()
