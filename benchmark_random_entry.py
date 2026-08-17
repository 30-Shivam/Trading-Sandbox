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
(Result: pullback lost too, both on untuned defaults and on its own
Optuna-tuned candidate -- see improvements.txt item 26.)
--strategy breakout_retest tests the breakout-retest signal (buy a
pullback BACK TO a recent genuine breakout's own trigger level, within
config.retest_window_days -- swingtrade.simulate_breakout_retest_signals /
simulate_random_breakout_retest_entries), built to keep the one ingredient
that's actually shown a real edge (breakout's trigger) while relaxing its
same-day-only restriction -- this IS the critical validation gate for that
strategy too. (Result: PASSED, both on untuned defaults and on its own
Optuna-tuned candidate -- see improvements.txt item 27.)
--strategy week52_high tests the 52-week-high-momentum signal (buy when
price is within config.week52_nearness_pct of its own trailing
week52_lookback_days high -- swingtrade.simulate_week52_signals /
simulate_random_week52_entries), a well-documented academic factor and a
continuous STATE rather than a discrete event, unlike every prior
strategy -- this IS the critical validation gate for that strategy too.
--strategy momentum_burst tests the momentum-burst signal (buy a single
day's Close-vs-prior-Close gain of at least config.momentum_burst_gain_pct_min,
CONFIRMED by Volume at least config.momentum_burst_volume_ratio_min times
its prior average -- swingtrade.simulate_momentum_burst_signals /
simulate_random_momentum_burst_entries), built to fire more often than any
prior strategy (no fresh-high requirement at all) -- this IS the critical
validation gate for that strategy too. (Result: mixed at untuned defaults
-- beats RANDOM on holdout/aggregate, loses on tune; Optuna-tuning made it
WORSE, not better -- holdout sharpe went net negative; an alternate
entry-fill model didn't resolve it either -- see improvements.txt items
35-37. Deprioritized in favor of a different signal formulation.)
--strategy squeeze_breakout tests the squeeze-breakout signal (buy a real
directional expansion -- config.squeeze_breakout_gain_pct_min -- following
a recent volatility contraction -- Recent_Min_Squeeze_Zscore at/below
config.squeeze_breakout_zscore_max within the trailing
config.squeeze_breakout_lookback_days -- swingtrade.simulate_squeeze_breakout_signals
/ simulate_random_squeeze_breakout_entries), built after momentum_burst
proved thin/fragile: deliberately does NOT require a fresh high over any
window (unlike breakout/breakout_retest/week52_high -- an earlier design
draft did, rejected because requiring both a squeeze AND a fresh high is
the intersection of two conditions, necessarily rarer than either alone)
and does NOT require volume confirmation (unlike momentum_burst) -- this
IS the critical validation gate for that strategy too, checked under BOTH
entry-fill models from the start (see --squeeze-breakout-entry-fill and
improvements.txt's validation-pipeline step 7).
--strategy adx_trend_entry tests the ADX-trend-entry signal (buy while
ADX -- config.adx_window -- is at/above config.adx_trend_entry_threshold
AND price is above a short-term MA -- config.adx_trend_entry_ma_window --
for direction, in a confirmed macro uptrend -- swingtrade.simulate_adx_trend_entry_signals
/ simulate_random_adx_trend_entry_entries), a continuous STATE like
week52_high/squeeze_breakout rather than a discrete event, so it can fire
on many consecutive days a trend persists. Deliberately lean v1, mirroring
breakout's OWN real history as the template -- v19 didn't launch with its
six optional "sharpening" filters either, they were added incrementally
after it was already trusted; the same treatment is a planned follow-up
for this strategy if this lean version clears the same bar. Checked under
BOTH entry-fill models from the start too (see --adx-trend-entry-entry-fill).

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
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_earnings_dates, fetch_history
from watchlist import SECTOR_ETF, read_ticker_sectors, read_tickers

EARNINGS_AWARE_STRATEGIES = ("squeeze_breakout", "ma_crossover")  # the only simulate_*_signals()/
                                                                   # simulate_random_*_entries() that
                                                                   # accept earnings_dates -- see
                                                                   # swingtrade/backtest.py
SECTOR_AWARE_STRATEGIES = ("breakout", "squeeze_breakout", "ma_crossover")  # the only REAL
                                                                   # simulate_*_signals() (not the
                                                                   # random baselines -- see
                                                                   # improvements.txt items 68/70/71)
                                                                   # that accept sector_ohlcv

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5


def load_config_to_test(version: int | None = None) -> tuple[swingtrade.TradingConfig, str]:
    """The currently-active System_Config by default -- this benchmark
    exists to ask "does the live signal carry real information," so it
    should test the live params, not a hardcoded snapshot that could
    silently go stale. Pass `version` to instead test one specific
    candidate (status=candidate, not yet promoted) -- e.g. a fresh Optuna
    result for a SECONDARY strategy (squeeze_breakout/ma_crossover), which
    never has status=active at all (only the single PRIMARY breakout slot
    does -- see improvements.txt item 60)."""
    if version is not None:
        doc = storage.get_config_by_version(version)
        if doc is None:
            print(f"[ERROR] No System_Config document with version={version}.", file=sys.stderr)
            sys.exit(1)
        config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), **doc["params"]})
        return config, f"v{version} ({doc.get('status', 'unknown status')})"
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
    parser.add_argument(
        "--strategy",
        choices=[
            "rsi", "breakout", "pullback", "breakout_retest", "week52_high",
            "momentum_burst", "squeeze_breakout", "adx_trend_entry", "ma_crossover",
        ],
        default="rsi",
        help="Which signal to benchmark against random entries. Default: rsi.",
    )
    parser.add_argument("--breakout-lookback-days", type=int, default=None,
                         help="Override config.breakout_lookback_days (--strategy breakout only). "
                              "Default: whatever the tested config already has (20 by default).")
    parser.add_argument(
        "--momentum-burst-entry-fill", choices=["limit", "next_open"], default=None,
        help="Override config.momentum_burst_entry_fill (--strategy momentum_burst only) -- "
             "'limit' waits for a downside touch back to the signal price (today's default, "
             "same convention week52_high uses); 'next_open' buys the very next session's Open "
             "unconditionally, no waiting -- a real test of whether the limit-fill model is "
             "systematically excluding genuine momentum continuations. Default: whatever the "
             "tested config already has ('limit').",
    )
    parser.add_argument(
        "--squeeze-breakout-entry-fill", choices=["limit", "next_open"], default=None,
        help="Override config.squeeze_breakout_entry_fill (--strategy squeeze_breakout only) -- "
             "same 'limit' vs. 'next_open' choice as --momentum-burst-entry-fill, see that flag's "
             "help. Default: whatever the tested config already has ('limit').",
    )
    parser.add_argument(
        "--adx-trend-entry-entry-fill", choices=["limit", "next_open"], default=None,
        help="Override config.adx_trend_entry_entry_fill (--strategy adx_trend_entry only) -- "
             "same 'limit' vs. 'next_open' choice as --momentum-burst-entry-fill, see that flag's "
             "help. Default: whatever the tested config already has ('limit').",
    )
    parser.add_argument(
        "--ma-crossover-entry-fill", choices=["limit", "next_open"], default=None,
        help="Override config.ma_crossover_entry_fill (--strategy ma_crossover only) -- "
             "same 'limit' vs. 'next_open' choice as --momentum-burst-entry-fill, see that flag's "
             "help. Default: whatever the tested config already has ('limit').",
    )
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the random-entry day selection.")
    parser.add_argument("--holdout-frac", type=float, default=0.25, help="Same ticker-holdout split as optimize.py. 0 disables.")
    parser.add_argument(
        "--holdout-seeds", default=None,
        help="Comma-separated holdout seeds to average TUNE/HOLDOUT over (a single fixed-seed split "
             "carries real sampling noise -- see improvements.txt item 69). "
             f"Default: {','.join(str(s) for s in DEFAULT_HOLDOUT_SEEDS)} (10 seeds).",
    )
    parser.add_argument(
        "--config-version", type=int, default=None,
        help="Test one specific System_Config version instead of the active one -- e.g. a fresh "
             "Optuna candidate for a secondary strategy (squeeze_breakout/ma_crossover), which "
             "never has status=active at all. Default: the active config (today's default behavior).",
    )
    parser.add_argument(
        "--with-catalyst", action="store_true",
        help="Fetch historical earnings dates (one extra yfinance call per ticker, see "
             "run_backtest.fetch_earnings_dates) so Catalyst_Warning is computed honestly "
             "instead of always False. Only affects --strategy squeeze_breakout/ma_crossover "
             "(the only two whose simulate_*_signals()/simulate_random_*_entries() accept "
             "earnings_dates) -- a no-op flag for every other --strategy value. Off by default, "
             "same convention as run_backtest.py's own --with-catalyst.",
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

    config, config_label = load_config_to_test(args.config_version)
    if args.breakout_lookback_days is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "breakout_lookback_days": args.breakout_lookback_days})
        config_label += f" (breakout_lookback_days overridden to {args.breakout_lookback_days})"
    if args.momentum_burst_entry_fill is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "momentum_burst_entry_fill": args.momentum_burst_entry_fill})
        config_label += f" (momentum_burst_entry_fill overridden to {args.momentum_burst_entry_fill})"
    if args.squeeze_breakout_entry_fill is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "squeeze_breakout_entry_fill": args.squeeze_breakout_entry_fill})
        config_label += f" (squeeze_breakout_entry_fill overridden to {args.squeeze_breakout_entry_fill})"
    if args.adx_trend_entry_entry_fill is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "adx_trend_entry_entry_fill": args.adx_trend_entry_entry_fill})
        config_label += f" (adx_trend_entry_entry_fill overridden to {args.adx_trend_entry_entry_fill})"
    if args.ma_crossover_entry_fill is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "ma_crossover_entry_fill": args.ma_crossover_entry_fill})
        config_label += f" (ma_crossover_entry_fill overridden to {args.ma_crossover_entry_fill})"
    print(f"Testing config: {config_label} -- strategy={args.strategy}")
    if args.strategy == "rsi":
        print(f"  rsi_oversold_threshold={config.rsi_oversold_threshold}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "breakout":
        print(f"  breakout_lookback_days={config.breakout_lookback_days}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "pullback":
        print(f"  pullback_ma_window={config.pullback_ma_window}, "
              f"pullback_ma_slope_window={config.pullback_ma_slope_window}, "
              f"pullback_band_pct={config.pullback_band_pct}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "breakout_retest":
        print(f"  breakout_lookback_days={config.breakout_lookback_days}, "
              f"retest_window_days={config.retest_window_days}, "
              f"retest_band_pct={config.retest_band_pct}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "week52_high":
        print(f"  week52_lookback_days={config.week52_lookback_days}, "
              f"week52_nearness_pct={config.week52_nearness_pct}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "momentum_burst":
        print(f"  momentum_burst_gain_pct_min={config.momentum_burst_gain_pct_min}, "
              f"momentum_burst_volume_ratio_min={config.momentum_burst_volume_ratio_min}, "
              f"momentum_burst_entry_fill={config.momentum_burst_entry_fill}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "squeeze_breakout":
        print(f"  squeeze_breakout_zscore_max={config.squeeze_breakout_zscore_max}, "
              f"squeeze_breakout_lookback_days={config.squeeze_breakout_lookback_days}, "
              f"squeeze_breakout_gain_pct_min={config.squeeze_breakout_gain_pct_min}, "
              f"squeeze_breakout_entry_fill={config.squeeze_breakout_entry_fill}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    elif args.strategy == "adx_trend_entry":
        print(f"  adx_trend_entry_threshold={config.adx_trend_entry_threshold}, "
              f"adx_trend_entry_ma_window={config.adx_trend_entry_ma_window}, "
              f"adx_trend_entry_entry_fill={config.adx_trend_entry_entry_fill}, "
              f"atr_take_profit_multiplier={config.atr_take_profit_multiplier}, "
              f"stop_loss_atr_multiplier={config.stop_loss_atr_multiplier}")
    else:
        print(f"  ma_crossover_short_window={config.ma_crossover_short_window}, "
              f"ma_crossover_long_window={config.ma_crossover_long_window}, "
              f"ma_crossover_entry_fill={config.ma_crossover_entry_fill}, "
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

    earnings_data = {}
    if args.with_catalyst and args.strategy in EARNINGS_AWARE_STRATEGIES:
        print(f"\nFetching historical earnings dates for {len(ticker_data)} ticker(s)...")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            earnings_data[ticker] = fetch_earnings_dates(ticker)
        found = sum(1 for d in earnings_data.values() if len(d) > 0)
        print(f"Got earnings history for {found}/{len(ticker_data)} ticker(s).")

    sector_data: dict[str, pd.DataFrame] = {}
    if args.strategy in SECTOR_AWARE_STRATEGIES:
        present_sectors = sorted({sector_lookup[t] for t in ticker_data if t in sector_lookup} & set(SECTOR_ETF))
        if present_sectors:
            print(f"\nFetching {len(present_sectors)} sector ETF(s) for Sector_Relative_Strength...")
            for i, sector in enumerate(present_sectors):
                if i > 0:
                    time.sleep(REQUEST_DELAY_SEC)
                etf_df = fetch_history(SECTOR_ETF[sector], start, end)
                if not etf_df.empty:
                    sector_data[sector] = etf_df

    if args.holdout_frac > 0:
        print(f"Ticker holdout: frac={args.holdout_frac}, averaging TUNE/HOLDOUT over "
              f"{len(holdout_seeds)} seeds ({holdout_seeds}) -- see improvements.txt item 69.")

    rng = random.Random(args.seed)

    if args.strategy == "rsi":
        real_fn, random_fn = swingtrade.simulate_signals, swingtrade.simulate_random_entries
        real_label = "RSI-timed"
    elif args.strategy == "breakout":
        real_fn, random_fn = swingtrade.simulate_breakout_signals, swingtrade.simulate_random_breakout_entries
        real_label = "Breakout-timed"
    elif args.strategy == "pullback":
        real_fn, random_fn = swingtrade.simulate_pullback_signals, swingtrade.simulate_random_pullback_entries
        real_label = "Pullback-timed"
    elif args.strategy == "breakout_retest":
        real_fn, random_fn = swingtrade.simulate_breakout_retest_signals, swingtrade.simulate_random_breakout_retest_entries
        real_label = "Breakout_Retest-timed"
    elif args.strategy == "week52_high":
        real_fn, random_fn = swingtrade.simulate_week52_signals, swingtrade.simulate_random_week52_entries
        real_label = "Week52_High-timed"
    elif args.strategy == "momentum_burst":
        real_fn, random_fn = swingtrade.simulate_momentum_burst_signals, swingtrade.simulate_random_momentum_burst_entries
        real_label = "Momentum_Burst-timed"
    elif args.strategy == "squeeze_breakout":
        real_fn, random_fn = swingtrade.simulate_squeeze_breakout_signals, swingtrade.simulate_random_squeeze_breakout_entries
        real_label = "Squeeze_Breakout-timed"
    elif args.strategy == "adx_trend_entry":
        real_fn, random_fn = swingtrade.simulate_adx_trend_entry_signals, swingtrade.simulate_random_adx_trend_entry_entries
        real_label = "ADX_Trend_Entry-timed"
    else:
        real_fn, random_fn = swingtrade.simulate_ma_crossover_signals, swingtrade.simulate_random_ma_crossover_entries
        real_label = "MA_Crossover-timed"

    real_trades = []
    random_trades = []
    real_counts = {}
    print(f"\nSimulating REAL {real_label} strategy and matched-count RANDOM-entry baseline "
          f"for {len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    earnings_kwargs = lambda ticker: (  # noqa: E731 -- only squeeze_breakout/ma_crossover accept earnings_dates
        {"earnings_dates": earnings_data.get(ticker)} if args.strategy in EARNINGS_AWARE_STRATEGIES else {}
    )
    sector_kwargs = lambda ticker: (  # noqa: E731
        {"sector_ohlcv": sector_data.get(sector_lookup.get(ticker, "Unknown"))}
        if args.strategy in SECTOR_AWARE_STRATEGIES else {}
    )
    for i, (ticker, ohlcv) in enumerate(ticker_data.items()):
        sector = sector_lookup.get(ticker, "Unknown")
        real = real_fn(
            ticker, ohlcv, market_data, start, end, config,
            **earnings_kwargs(ticker), sector=sector, **sector_kwargs(ticker),
        )
        real_trades.extend(real)
        real_counts[ticker] = len(real)

        rand = random_fn(
            ticker, ohlcv, market_data, start, end, len(real), rng, config,
            **earnings_kwargs(ticker), sector=sector,
        )
        random_trades.extend(rand)

    total_real = sum(real_counts.values())
    print(f"Real {real_label} signals: {total_real} across {len(ticker_data)} ticker(s) "
          f"(entry-fill realized: {len(real_trades)}). "
          f"Random baseline (matched count per ticker): {len(random_trades)} entries filled.")

    print("\n=== ALL TICKERS ===")
    print(f"  REAL   ({real_label}): {summarize(real_trades)}")
    print(f"  RANDOM (matched count): {summarize(random_trades)}")

    if args.holdout_frac > 0:
        real_tune_avg, real_holdout_avg = average_holdout_summary(
            real_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        random_tune_avg, random_holdout_avg = average_holdout_summary(
            random_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )

        print(f"\n=== TUNE (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  REAL   ({real_label}): {real_tune_avg}")
        print(f"  RANDOM (matched count): {random_tune_avg}")

        print(f"\n=== HOLDOUT (avg of {len(holdout_seeds)} seeds) ===")
        print(f"  REAL   ({real_label}): {real_holdout_avg}")
        print(f"  RANDOM (matched count): {random_holdout_avg}")

    print()
    print(f"If REAL's sharpe_like/win_rate isn't meaningfully better than RANDOM's (same trade")
    print(f"count, same universe, same stop/target/holding-period structure), {args.strategy} timing")
    print("is not adding real predictive information beyond the payoff structure + market drift.")


if __name__ == "__main__":
    main()
