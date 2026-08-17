"""
Multi-strategy consensus signal -- backtest feasibility check.

Question: when 2+ of the 3 currently-LIVE mechanical strategies (breakout
v43, squeeze_breakout v39, ma_crossover v51) fire on the SAME ticker/day,
do those "consensus" trades carry real predictive value beyond what any one
strategy shows alone -- the same "agreement = trustworthy" logic already
validated for the dual-provider LLM cross-check (llm_agent.py)?

Unlike regime_switcher.py (LLM/news-driven, structurally can't be
backtested), this is pure mechanical signal overlap -- no lookahead
problem -- so it goes through this project's real validation gate before
any live build, per strategy_validation_pipeline.txt's own instruction to
default back to the full pipeline unless explicitly told to skip it again.

Method: run each of the 3 strategies' own simulate_*_signals() independently
against their own LIVE config (not DEFAULT_CONFIG) over the full watchlist,
5-year window. Group every resulting trade by its (ticker, signal_date) key
-- the pre-entry-fill trigger date, before fill-timing noise. Any key where
2+ distinct strategies fired is "consensus" (pool ALL agreeing legs, not
just a cherry-picked "winner" leg, to test the hypothesis honestly); every
other key is "solo". Same ticker-holdout split + cluster-adjusted weighting
as benchmark_random_entry.py, so the comparison holds up (or doesn't) out
of sample too.

Decision gate: consensus must beat solo on TUNE *and* HOLDOUT, not just
ALL, to count as real -- same discipline every other strategy here has been
held to. See improvements.txt for the write-up either way.

Usage:
    python benchmark_consensus_signal.py
    python benchmark_consensus_signal.py --tickers NVDA,AMD,INTC --holdout-frac 0
    python benchmark_consensus_signal.py --start 2021-06-01 --end 2026-08-15
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import swingtrade
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_earnings_dates, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

# The 3 currently-live, already-promoted mechanical strategies -- deliberately
# NOT undecided candidates (v53/v54 etc), same reasoning regime_switcher.py
# used: testing the versions actually trusted enough to be capital-eligible
# today, not something still under consideration itself.
LIVE_STRATEGY_SIMULATORS = {
    "breakout": swingtrade.simulate_breakout_signals,
    "squeeze_breakout": swingtrade.simulate_squeeze_breakout_signals,
    "ma_crossover": swingtrade.simulate_ma_crossover_signals,
}
EARNINGS_AWARE_STRATEGIES = ("squeeze_breakout", "ma_crossover")


def load_live_configs() -> dict[str, tuple[swingtrade.TradingConfig, str]]:
    """The 3 real, currently-live configs -- config_loader.py is the single
    source of truth ingest.py/dip_buy_analyzer.py already use, so this
    benchmark can't silently drift from what's actually running."""
    breakout_config, breakout_label = config_loader.load_active_config()
    squeeze_config, squeeze_reason = config_loader.load_config_by_version(39)
    ma_crossover_config, ma_reason = config_loader.load_config_by_version(51)
    if squeeze_config is None:
        print(f"[ERROR] Could not load squeeze_breakout v39: {squeeze_reason}", file=sys.stderr)
        sys.exit(1)
    if ma_crossover_config is None:
        print(f"[ERROR] Could not load ma_crossover v51: {ma_reason}", file=sys.stderr)
        sys.exit(1)
    return {
        "breakout": (breakout_config, breakout_label),
        "squeeze_breakout": (squeeze_config, "v39 (secondary, live)"),
        "ma_crossover": (ma_crossover_config, "v51 (secondary, live)"),
    }


def summarize(trades: list[dict]) -> dict:
    resolved = [t for t in trades if t["status"] != "OPEN"]
    weights = swingtrade.compute_cluster_weights(resolved)
    return swingtrade.summarize_trades_weighted(resolved, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--holdout-frac", type=float, default=0.25, help="Same ticker-holdout split as optimize.py. 0 disables.")
    parser.add_argument(
        "--holdout-seeds", default=None,
        help="Comma-separated holdout seeds to average TUNE/HOLDOUT over (a single fixed-seed split "
             "carries real sampling noise -- see improvements.txt item 69). "
             f"Default: {','.join(str(s) for s in DEFAULT_HOLDOUT_SEEDS)} (10 seeds).",
    )
    parser.add_argument(
        "--with-catalyst", action="store_true",
        help="Fetch historical earnings dates (one extra yfinance call per ticker) so "
             "Catalyst_Warning is computed honestly for squeeze_breakout/ma_crossover "
             "instead of always False. Off by default, same convention as run_backtest.py.",
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

    configs = load_live_configs()
    print("Testing LIVE configs:")
    for strategy, (config, label) in configs.items():
        print(f"  {strategy}: {label}")

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

    earnings_data = {}
    if args.with_catalyst:
        print(f"\nFetching historical earnings dates for {len(ticker_data)} ticker(s)...")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            earnings_data[ticker] = fetch_earnings_dates(ticker)
        found = sum(1 for d in earnings_data.values() if len(d) > 0)
        print(f"Got earnings history for {found}/{len(ticker_data)} ticker(s).")

    if args.holdout_frac > 0:
        print(f"Ticker holdout: frac={args.holdout_frac}, averaging TUNE/HOLDOUT over "
              f"{len(holdout_seeds)} seeds ({holdout_seeds}) -- see improvements.txt item 69.")

    earnings_kwargs = lambda strategy, ticker: (  # noqa: E731
        {"earnings_dates": earnings_data.get(ticker)}
        if strategy in EARNINGS_AWARE_STRATEGIES and args.with_catalyst else {}
    )

    print(f"\nSimulating {len(LIVE_STRATEGY_SIMULATORS)} live strategies for "
          f"{len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    all_trades: dict[str, list[dict]] = {s: [] for s in LIVE_STRATEGY_SIMULATORS}
    for strategy, sim_fn in LIVE_STRATEGY_SIMULATORS.items():
        config, _ = configs[strategy]
        for ticker, ohlcv in ticker_data.items():
            sector = sector_lookup.get(ticker, "Unknown")
            trades = sim_fn(
                ticker, ohlcv, market_data, start, end, config,
                **earnings_kwargs(strategy, ticker), sector=sector,
            )
            for t in trades:
                t["_strategy"] = strategy
            all_trades[strategy].extend(trades)
        print(f"  {strategy}: {len(all_trades[strategy])} signal(s)")

    # (ticker, signal_date) -> set of strategies that fired that key
    fire_map: dict[tuple, set] = {}
    for strategy, trades in all_trades.items():
        for t in trades:
            key = (t["ticker"], t["signal_date"])
            fire_map.setdefault(key, set()).add(strategy)

    consensus_keys = {k for k, v in fire_map.items() if len(v) >= 2}
    combo_counts: dict[tuple, int] = {}
    for k in consensus_keys:
        combo = tuple(sorted(fire_map[k]))
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    print(f"\nDistinct (ticker, signal_date) keys: {len(fire_map)}")
    print(f"  -- 2+ strategies agreed: {len(consensus_keys)}")
    print("  Agreement combos:")
    for combo, n in sorted(combo_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {combo}: {n}")

    consensus_trades = []
    solo_trades = []
    for strategy, trades in all_trades.items():
        for t in trades:
            key = (t["ticker"], t["signal_date"])
            (consensus_trades if key in consensus_keys else solo_trades).append(t)

    print("\n=== ALL TICKERS ===")
    print(f"  CONSENSUS (2+ strategies agree): {summarize(consensus_trades)}")
    print(f"  SOLO      (1 strategy fires)   : {summarize(solo_trades)}")

    if args.holdout_frac > 0:
        consensus_tune_avg, consensus_holdout_avg = average_holdout_summary(
            consensus_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        solo_tune_avg, solo_holdout_avg = average_holdout_summary(
            solo_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )

        print(f"\n=== TUNE (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  CONSENSUS: {consensus_tune_avg}")
        print(f"  SOLO     : {solo_tune_avg}")

        print(f"\n=== HOLDOUT (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  CONSENSUS: {consensus_holdout_avg}")
        print(f"  SOLO     : {solo_holdout_avg}")

    print()
    print("Decision gate: CONSENSUS needs to beat SOLO on sharpe_like/win_rate on BOTH")
    print("TUNE and HOLDOUT (averaged across seeds, not a single draw), not just ALL, to")
    print("count as real evidence agreement carries information beyond any one strategy's")
    print("own edge. A HOLDOUT-only failure means stop here -- do not build a live")
    print("consensus_signal feature on top of this.")


if __name__ == "__main__":
    main()
