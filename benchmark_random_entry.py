"""
Random-entry benchmark: does a signal's entry TIMING carry any real
predictive information, or is this system's backtested edge coming
entirely from the stop/target payoff structure (tight stop, wide target)
plus general market drift?

Three independent rigor upgrades this session (entry-fill timing realism,
tune-vs-ticker-holdout, v3-vs-ticker-holdout -- see improvements.txt) each
collapsed RSI-oversold's apparent edge toward zero. This answers the more
basic question directly: for each ticker, run the REAL strategy to get its
signal count, then fire the SAME NUMBER of trades on RANDOMLY chosen days
instead -- identical universe (macro uptrend + liquidity gates), identical
entry-fill/stop/target mechanics, only WHICH DAY differs. If REAL entries
don't meaningfully beat this matched-count random baseline, the signal's
TIMING carries little real information (the RSI result: it doesn't --
REAL lost to RANDOM on every ticker-holdout cut).

--strategy rsi (default) tests the original RSI-oversold mean-reversion
signal (swingtrade.simulate_signals / simulate_random_entries).
--strategy breakout tests the newer trend-following signal (buy a new
config.breakout_lookback_days-day closing high in a confirmed uptrend --
swingtrade.simulate_breakout_signals / simulate_random_breakout_entries),
built specifically because the RSI result showed pure mean-reversion timing
adds no value -- see improvements.txt's STRATEGIC PIVOT section.
--strategy pullback tests the pullback-in-uptrend signal (buy a shallow
dip toward a rising config.pullback_ma_window-day SMA in a confirmed
uptrend -- swingtrade.simulate_pullback_signals / simulate_random_pullback_entries),
built to fire more often than breakout's fresh-high requirement -- this IS
the critical validation gate for that strategy before it's trusted at all.

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
    python benchmark_random_entry.py --strategy breakout --breakout-lookback-days 55
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
    parser.add_argument("--strategy", choices=["rsi", "breakout", "pullback"], default="rsi",
                         help="Which signal to benchmark against random entries. Default: rsi.")
    parser.add_argument("--breakout-lookback-days", type=int, default=None,
                         help="Override config.breakout_lookback_days (--strategy breakout only). "
                              "Default: whatever the tested config already has (20 by default).")
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
    if args.breakout_lookback_days is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "breakout_lookback_days": args.breakout_lookback_days})
        config_label += f" (breakout_lookback_days overridden to {args.breakout_lookback_days})"
    print(f"Testing config: {config_label} -- strategy={args.strategy}")
    if args.strategy == "rsi":
        print(f"  rsi_oversold_threshold={config.rsi_oversold_threshold}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "breakout":
        print(f"  breakout_lookback_days={config.breakout_lookback_days}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    else:
        print(f"  pullback_ma_window={config.pullback_ma_window}, "
              f"pullback_ma_slope_window={config.pullback_ma_slope_window}, "
              f"pullback_band_pct={config.pullback_band_pct}, "
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

    if args.strategy == "rsi":
        real_fn, random_fn = swingtrade.simulate_signals, swingtrade.simulate_random_entries
        real_label = "RSI-timed"
    elif args.strategy == "breakout":
        real_fn, random_fn = swingtrade.simulate_breakout_signals, swingtrade.simulate_random_breakout_entries
        real_label = "Breakout-timed"
    else:
        real_fn, random_fn = swingtrade.simulate_pullback_signals, swingtrade.simulate_random_pullback_entries
        real_label = "Pullback-timed"

    real_trades = []
    random_trades = []
    real_counts = {}
    print(f"\nSimulating REAL {real_label} strategy and matched-count RANDOM-entry baseline "
          f"for {len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    for i, (ticker, ohlcv) in enumerate(ticker_data.items()):
        sector = sector_lookup.get(ticker, "Unknown")
        real = real_fn(ticker, ohlcv, market_data, start, end, config, sector=sector)
        real_trades.extend(real)
        real_counts[ticker] = len(real)

        rand = random_fn(
            ticker, ohlcv, market_data, start, end, len(real), rng, config, sector=sector
        )
        random_trades.extend(rand)

    total_real = sum(real_counts.values())
    print(f"Real {real_label} signals: {total_real} across {len(ticker_data)} ticker(s) "
          f"(entry-fill realized: {len(real_trades)}). "
          f"Random baseline (matched count per ticker): {len(random_trades)} entries filled.")

    def split(trades):
        tune = [t for t in trades if t["ticker"] not in holdout_set]
        holdout = [t for t in trades if t["ticker"] in holdout_set]
        return tune, holdout

    real_tune, real_holdout = split(real_trades)
    random_tune, random_holdout = split(random_trades)

    print("\n=== ALL TICKERS ===")
    print(f"  REAL   ({real_label}): {summarize(real_trades)}")
    print(f"  RANDOM (matched count): {summarize(random_trades)}")

    if holdout_tickers:
        print(f"\n=== TUNE tickers ({len(tune_tickers)}) ===")
        print(f"  REAL   ({real_label}): {summarize(real_tune)}")
        print(f"  RANDOM (matched count): {summarize(random_tune)}")

        print(f"\n=== HOLDOUT tickers ({len(holdout_tickers)}) ===")
        print(f"  REAL   ({real_label}): {summarize(real_holdout)}")
        print(f"  RANDOM (matched count): {summarize(random_holdout)}")

    print()
    print(f"If REAL's sharpe_like/win_rate isn't meaningfully better than RANDOM's (same trade")
    print(f"count, same universe, same stop/target/holding-period structure), {args.strategy} timing")
    print("is not adding real predictive information beyond the payoff structure + market drift.")


if __name__ == "__main__":
    main()
