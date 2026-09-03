"""
Macro SKEW-index regime -- backtest feasibility check.

Question: does the broad options-market tail-risk regime (CBOE SKEW Index
elevated vs normal) predict ma_crossover's trade quality, independent of
its own edge? Grounded in the same reasoning as benchmark_macro_regime.py
(yield curve) and benchmark_sector_relative_strength.py -- a factor that
looks plausible in theory needs a real check before any live filter gets
built on top of it, per strategy_validation_pipeline.txt.

Originated from a user-supplied PDF ("Build Your Own Skew Map") describing
a PER-STOCK 25-delta put/call IV skew map -- confirmed infeasible to
backtest here (yfinance only exposes LIVE option chains, no historical
per-contract IV archive; a real paid feed, ~$30/mo, would be needed). The
CBOE SKEW Index (ticker ^SKEW) is a related but coarser, INDEX-LEVEL
tail-risk measure -- not the same formula (CBOE's own OTM-SPX-based
skewness calculation, not put IV minus call IV over ATM IV), but free via
yfinance with real daily history back to 1990, so THIS is what's actually
checkable.

Uses ^SKEW itself, no API key needed (unlike the FRED-based yield-curve
check) -- fetched via the same yfinance path every other ticker in this
project already uses.

No natural fixed threshold exists for SKEW the way T10Y2Y has a
theory-driven zero (CBOE's own "no tail risk" reference point, 100, is
almost never actually observed -- min in the real 1990-2026 series is
101.2, mean 123). So the regime split uses a ROLLING median over the
trailing SKEW_ROLLING_WINDOW_DAYS calendar days STRICTLY BEFORE each
trade's own signal_date (never including same-day or future values) --
"elevated" vs "normal" relative to SKEW's own RECENT history, not a fixed
constant. Point-in-time safe: both the threshold itself and the day being
classified are computed from data available before the trade, same
no-look-ahead discipline every other indicator in this codebase already
follows.

Deliberately a BOUNDED rolling window, not an expanding-since-1990 one --
a real bug caught in a smoke test before the full run: an expanding median
anchored all the way back to 1990 stays dragged down by SKEW's own decades
of secular drift (SKEW has trended meaningfully higher over time), so
almost every recent observation reads "elevated" relative to a 36-year-old
baseline that's no longer representative -- a 5-ticker smoke test split
38 elevated vs 3 normal, nowhere near a usable regime split. A rolling
window stays adaptive to wherever SKEW's own level has actually settled
recently. Fetches an extra SKEW_WARMUP_DAYS of ^SKEW history before
`--start` purely to warm up the rolling window before the real test window
begins (never contributes trades of its own).

Method: run ma_crossover's own simulate_ma_crossover_signals() against its
LIVE config over the full watchlist/5-year window, same as
benchmark_macro_regime.py. Compare via the same ticker-holdout +
cluster-adjusted, multi-seed-averaged methodology every other benchmark
script here uses.

Decision gate: ELEVATED and NORMAL regimes must show a REAL, consistent
difference on TUNE *and* HOLDOUT, not just ALL, before this is worth
building into a live filter -- same discipline every other factor check
this project has used.

Usage:
    python benchmark_skew_regime.py
    python benchmark_skew_regime.py --tickers NVDA,AMD,XOM --holdout-frac 0
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import swingtrade
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
SKEW_TICKER = "^SKEW"  # CBOE SKEW Index -- see this module's own docstring
SKEW_ROLLING_WINDOW_DAYS = 365  # ~1 trading year -- bounded, adaptive to secular
                                  # drift, see docstring for the real bug this fixes
SKEW_WARMUP_DAYS = 365 * 2  # extra history fetched before --start, comfortably
                             # covers SKEW_ROLLING_WINDOW_DAYS before the real
                             # test window begins, see docstring
MIN_HISTORY_FOR_MEDIAN = 126  # ~6 trading months -- below this, don't trust the
                               # rolling median yet (matters only if the warmup
                               # fetch itself came back thin, e.g. a data gap)


def summarize(trades: list[dict]) -> dict:
    resolved = [t for t in trades if t["status"] != "OPEN"]
    weights = swingtrade.compute_cluster_weights(resolved)
    return swingtrade.summarize_trades_weighted(resolved, weights)


def _regime_as_of(skew_series: pd.Series, signal_date) -> str | None:
    """"elevated" (today's SKEW above the trailing SKEW_ROLLING_WINDOW_DAYS'
    own median) or "normal" (at or below), using ONLY ^SKEW observations
    STRICTLY BEFORE signal_date, within that trailing window, for both the
    threshold and the value being classified -- never the signal date
    itself or later, and never anything older than the rolling window (see
    this module's own docstring for the real secular-drift bug a naive
    expanding-since-1990 median hit before this). Returns None if fewer
    than MIN_HISTORY_FOR_MEDIAN observations exist within the window yet."""
    as_of = pd.Timestamp(signal_date)
    window_start = as_of - pd.Timedelta(days=SKEW_ROLLING_WINDOW_DAYS)
    window = skew_series[(skew_series.index >= window_start) & (skew_series.index < as_of)]
    if len(window) < MIN_HISTORY_FOR_MEDIAN:
        return None
    threshold = window.median()
    current = window.iloc[-1]
    return "elevated" if current > threshold else "normal"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--holdout-frac", type=float, default=0.25, help="Same ticker-holdout split as optimize.py. 0 disables.")
    parser.add_argument(
        "--holdout-seeds", default=None,
        help="Comma-separated holdout seeds to average TUNE/HOLDOUT over -- a single fixed-seed "
             "split carries real sampling noise (see improvements.txt item 69). "
             f"Default: {','.join(str(s) for s in DEFAULT_HOLDOUT_SEEDS)} (10 seeds).",
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

    config, label = config_loader.load_active_config()
    if config.strategy != "ma_crossover":
        print(
            f"[ERROR] Active config's strategy is {config.strategy!r}, not ma_crossover -- "
            "this script is scoped to ma_crossover specifically (the one live strategy with a "
            "real, standing beat-random validation right now). Re-scope if the active slot has changed.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Testing LIVE config: {label} (strategy={config.strategy})")

    skew_fetch_start = start - pd.Timedelta(days=SKEW_WARMUP_DAYS)
    print(f"\nFetching {SKEW_TICKER} (CBOE SKEW Index) from {skew_fetch_start.date()}..{end.date()} "
          f"({SKEW_WARMUP_DAYS} days of extra warmup before {start.date()})...")
    skew_df = fetch_history(SKEW_TICKER, skew_fetch_start, end)
    if skew_df.empty:
        print(f"[ERROR] Could not fetch {SKEW_TICKER}.", file=sys.stderr)
        sys.exit(1)
    skew_series = skew_df["Close"]
    print(f"Fetched {len(skew_series)} observation(s), median={skew_series.median():.2f}, "
          f"range [{skew_series.min():.2f}, {skew_series.max():.2f}].")

    print(f"\nFetching {MARKET_INDEX_TICKER} (market proxy)...")
    market_ohlcv = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_ohlcv.empty:
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

    if args.holdout_frac > 0:
        print(f"Ticker holdout: frac={args.holdout_frac}, averaging TUNE/HOLDOUT over "
              f"{len(holdout_seeds)} seeds ({holdout_seeds}) -- see improvements.txt item 69.")

    print(f"\nSimulating ma_crossover for {len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    all_trades = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        trades = swingtrade.simulate_ma_crossover_signals(
            ticker, ohlcv, market_ohlcv, start, end, config, sector=sector,
        )
        all_trades.extend(trades)
    print(f"  {len(all_trades)} signal(s)")

    print(f"\nBucketing trades by {SKEW_TICKER} regime as of each trade's own signal_date "
          "(rolling median, point-in-time safe)...")
    missing_regime = 0
    elevated_trades = []
    normal_trades = []
    for t in all_trades:
        regime = _regime_as_of(skew_series, t["signal_date"])
        if regime is None:
            missing_regime += 1
            continue
        t["_skew_regime"] = regime
        (elevated_trades if regime == "elevated" else normal_trades).append(t)

    if missing_regime:
        print(f"  [WARN] {missing_regime} trade(s) skipped -- insufficient prior {SKEW_TICKER} history.")
    print(f"  Elevated-SKEW trades: {len(elevated_trades)}")
    print(f"  Normal-SKEW trades:   {len(normal_trades)}")

    print("\n=== ALL TICKERS ===")
    print(f"  ELEVATED (SKEW > its own trailing 1yr median): {summarize(elevated_trades)}")
    print(f"  NORMAL   (SKEW <= its own trailing 1yr median): {summarize(normal_trades)}")

    if args.holdout_frac > 0:
        elevated_tune_avg, elevated_holdout_avg = average_holdout_summary(
            elevated_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        normal_tune_avg, normal_holdout_avg = average_holdout_summary(
            normal_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )

        print(f"\n=== TUNE (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  ELEVATED: {elevated_tune_avg}")
        print(f"  NORMAL:   {normal_tune_avg}")

        print(f"\n=== HOLDOUT (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  ELEVATED: {elevated_holdout_avg}")
        print(f"  NORMAL:   {normal_holdout_avg}")

    print()
    print("Decision gate: ELEVATED and NORMAL need to show a REAL, consistent difference in")
    print("sharpe_like/win_rate on BOTH TUNE and HOLDOUT (averaged across seeds, not a single")
    print("draw), not just ALL, to count as real evidence SKEW regime carries information beyond")
    print("ma_crossover's own edge. A HOLDOUT-only or inconsistent result means stop here -- do")
    print("not build a live SKEW-regime filter on top of this.")


if __name__ == "__main__":
    main()
