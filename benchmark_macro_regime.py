"""
Macro yield-curve regime -- backtest feasibility check.

Question: does the broad macro regime (yield curve inverted vs. normal)
predict ma_crossover's trade quality, independent of its own edge? Grounded
in the same reasoning as benchmark_sector_relative_strength.py -- a factor
that looks plausible in theory needs a real check before any live filter
gets built on top of it, per strategy_validation_pipeline.txt.

Uses the 10-year/2-year Treasury yield spread (FRED series T10Y2Y) as the
regime signal -- market-observed, published same-day, NEVER revised after
the fact. Deliberately NOT a survey/estimate-based series like ISM PMI or
GDP: those get revised after initial release, which would silently graft a
new look-ahead-bias problem onto a codebase that's gone to real lengths
elsewhere (entry-fill timing, .shift(1) indicators, ticker-holdout) to
avoid exactly this class of bug. See market_data.fetch_fred_series()'s own
docstring for the same reasoning.

Requires a free FRED_API_KEY (fred.stlouisfed.org/docs/api/api_key.html,
no cost) -- degrades to a clear error, not a crash, if unset.

Method: run ma_crossover's own simulate_ma_crossover_signals() against its
LIVE config (config_loader.load_active_config() -- ma_crossover is
currently primary) over the full watchlist/5-year window. For every
resolved trade, look up the most recent T10Y2Y value STRICTLY BEFORE its
signal_date (a 1-day-minimum lag, same "don't assume same-day availability"
caution every other indicator in this codebase already takes) and bucket
into "inverted" (spread < 0) vs. "normal" (spread >= 0). Compare via the
same ticker-holdout + cluster-adjusted, multi-seed-averaged methodology
every other benchmark script here uses.

Decision gate: inverted and normal regimes must show a REAL, consistent
difference on TUNE *and* HOLDOUT, not just ALL, before this is worth
building into a live filter -- same discipline every other factor check
this project has used.

Usage:
    python benchmark_macro_regime.py
    python benchmark_macro_regime.py --tickers NVDA,AMD,XOM --holdout-frac 0
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import market_data
import swingtrade
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
YIELD_CURVE_SERIES_ID = "T10Y2Y"  # 10-year minus 2-year Treasury constant maturity
                                   # rate spread -- market-observed, daily, never
                                   # revised. See this module's own docstring.


def summarize(trades: list[dict]) -> dict:
    resolved = [t for t in trades if t["status"] != "OPEN"]
    weights = swingtrade.compute_cluster_weights(resolved)
    return swingtrade.summarize_trades_weighted(resolved, weights)


def _regime_as_of(yield_curve: pd.Series, signal_date) -> str | None:
    """"inverted" (spread < 0) or "normal" (spread >= 0) as of the most
    recent T10Y2Y observation STRICTLY BEFORE signal_date -- never the
    signal date itself or later, same no-same-day-availability caution
    every other point-in-time lookup in this codebase already takes.
    Returns None if no observation exists before signal_date yet (too
    early in the series)."""
    as_of = pd.Timestamp(signal_date)
    prior = yield_curve[yield_curve.index < as_of]
    if prior.empty:
        return None
    return "inverted" if prior.iloc[-1] < 0 else "normal"


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

    if not market_data.fred_available():
        print(
            "[ERROR] FRED_API_KEY not set -- get a free key at "
            "fred.stlouisfed.org/docs/api/api_key.html and set it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

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

    print(f"\nFetching {YIELD_CURVE_SERIES_ID} (10Y-2Y Treasury spread) from FRED, {start.date()}..{end.date()}...")
    yield_curve = market_data.fetch_fred_series(YIELD_CURVE_SERIES_ID, start, end)
    if yield_curve is None or yield_curve.empty:
        print(f"[ERROR] Could not fetch {YIELD_CURVE_SERIES_ID} from FRED.", file=sys.stderr)
        sys.exit(1)
    n_inverted_days = int((yield_curve < 0).sum())
    print(
        f"Fetched {len(yield_curve)} observation(s) -- {n_inverted_days} inverted-curve day(s) "
        f"({n_inverted_days / len(yield_curve):.1%} of the series)."
    )

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

    print(f"\nBucketing trades by {YIELD_CURVE_SERIES_ID} regime as of each trade's own signal_date...")
    missing_regime = 0
    inverted_trades = []
    normal_trades = []
    for t in all_trades:
        regime = _regime_as_of(yield_curve, t["signal_date"])
        if regime is None:
            missing_regime += 1
            continue
        t["_macro_regime"] = regime
        (inverted_trades if regime == "inverted" else normal_trades).append(t)

    if missing_regime:
        print(f"  [WARN] {missing_regime} trade(s) skipped -- no FRED observation yet before signal_date.")
    print(f"  Inverted-curve trades: {len(inverted_trades)}")
    print(f"  Normal-curve trades:   {len(normal_trades)}")

    print("\n=== ALL TICKERS ===")
    print(f"  INVERTED (T10Y2Y < 0): {summarize(inverted_trades)}")
    print(f"  NORMAL   (T10Y2Y >= 0): {summarize(normal_trades)}")

    if args.holdout_frac > 0:
        inverted_tune_avg, inverted_holdout_avg = average_holdout_summary(
            inverted_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        normal_tune_avg, normal_holdout_avg = average_holdout_summary(
            normal_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )

        print(f"\n=== TUNE (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  INVERTED: {inverted_tune_avg}")
        print(f"  NORMAL:   {normal_tune_avg}")

        print(f"\n=== HOLDOUT (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  INVERTED: {inverted_holdout_avg}")
        print(f"  NORMAL:   {normal_holdout_avg}")

    print()
    print("Decision gate: INVERTED and NORMAL need to show a REAL, consistent difference in")
    print("sharpe_like/win_rate on BOTH TUNE and HOLDOUT (averaged across seeds, not a single")
    print("draw), not just ALL, to count as real evidence macro regime carries information")
    print("beyond ma_crossover's own edge. A HOLDOUT-only or inconsistent result means stop")
    print("here -- do not build a live macro-regime filter on top of this.")


if __name__ == "__main__":
    main()
