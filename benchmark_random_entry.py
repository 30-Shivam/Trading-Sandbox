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
--strategy pairs tests the mean-reversion PAIRS signal (buy a ticker when
it has diverged unusually far BELOW its most-correlated same-sector peer
over a recent window -- Pair_Spread_Zscore at/below config.pairs_zscore_entry_max
-- swingtrade.simulate_pairs_signals / simulate_random_pairs_entries),
a genuinely different signal family from every trend/volatility-following
strategy above (ticker-vs-peer divergence, not ticker/sector-vs-market
momentum). LONG-ONLY laggard-convergence (no short leg -- this codebase
has no short-position support anywhere). Deliberately lean v1, same
"no optional filters yet, add them only if this clears the bar" treatment
adx_trend_entry got -- this IS the critical validation gate for this
strategy too.

--strategy insider_buying tests the INSIDER-BUYING signal (buy a ticker
when real, open-market Form-4 insider purchases have clustered within a
recent window -- config.insider_lookback_days, dollar/distinct-buyer
gated -- swingtrade.simulate_insider_buying_signals / simulate_random_insider_buying_entries),
a fundamentally different data source from every other strategy (fundamentals/
ownership activity, not price/volume). Real, hard constraint worth knowing
before trusting any result here: run_backtest.fetch_insider_purchases()'s
own yfinance source only carries ~12-22 months of real history (NOT this
project's usual 5-year window), so any validation on this strategy is
necessarily weaker evidence than every other strategy's own, by
construction -- this IS still the critical validation gate for it, just
over a shorter real window than usual.

--strategy momentum_rank tests cross-sectional MOMENTUM RANK (buy a ticker
whose trailing-return percentile rank -- config.momentum_lookback_days,
against the WHOLE watchlist, not just its own sector -- clears
config.momentum_top_percentile_min, in a confirmed macro uptrend --
swingtrade.simulate_momentum_signals / simulate_random_momentum_entries),
the first strategy in this project whose signal for one ticker depends on
every OTHER ticker's own return the same day (Jegadeesh & Titman
cross-sectional momentum) -- every prior strategy scores a ticker from its
own price history alone. Continuous STATE like week52_high/squeeze_breakout/
adx_trend_entry, not a discrete event. Deliberately lean v1, same "no
optional filters yet" treatment every other strategy's first pass got --
this IS the critical validation gate for it.

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
import yfinance as yf

import ic_tracking
import storage
import swingtrade
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary, average_summaries
from swingtrade.levels import (
    precompute_breakout_frame,
    precompute_ma_crossover_frame,
    precompute_momentum_frame,
    precompute_pairs_frame,
    precompute_rsi_frame,
    precompute_squeeze_breakout_frame,
)
from run_backtest import (
    LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, RESEARCH_HOLDOUT_CUTOFF, fetch_earnings_dates,
    fetch_earnings_surprises, fetch_history, fetch_insider_purchases,
)
from watchlist import SECTOR_ETF, read_ticker_sectors, read_tickers

EARNINGS_AWARE_STRATEGIES = (
    "squeeze_breakout", "ma_crossover", "pairs", "insider_buying", "momentum_rank", "pead",
)  # the only
                                                                   # simulate_*_signals()/
                                                                   # simulate_random_*_entries() that
                                                                   # accept earnings_dates -- see
                                                                   # swingtrade/backtest.py
SECTOR_AWARE_STRATEGIES = ("breakout", "squeeze_breakout", "ma_crossover")  # the only REAL
                                                                   # simulate_*_signals() (not the
                                                                   # random baselines -- see
                                                                   # improvements.txt items 68/70/71)
                                                                   # that accept sector_ohlcv
PAIR_AWARE_STRATEGIES = ("pairs",)  # the only REAL simulate_*_signals() (not the random
                                                                   # baseline -- same "answers whether
                                                                   # TIMING adds value" precedent as
                                                                   # SECTOR_AWARE_STRATEGIES above) that
                                                                   # accepts peer_prices -- see
                                                                   # improvements.txt item 82
INSIDER_AWARE_STRATEGIES = ("insider_buying",)  # the only REAL simulate_*_signals() (not the
                                                                   # random baseline -- same reasoning as
                                                                   # PAIR_AWARE_STRATEGIES above) that
                                                                   # accepts insider_purchases
MOMENTUM_AWARE_STRATEGIES = ("momentum_rank",)  # the only REAL simulate_*_signals() (not the
                                                                   # random baseline -- same "answers
                                                                   # whether TIMING adds value"
                                                                   # precedent as PAIR_AWARE_STRATEGIES
                                                                   # above) that accepts rank_column
PEAD_AWARE_STRATEGIES = ("pead",)  # the only REAL simulate_*_signals() (not the random
                                                                   # baseline -- same reasoning as
                                                                   # INSIDER_AWARE_STRATEGIES above) that
                                                                   # accepts earnings_surprises

# Strategy -> its own *_strength_cap_pct/*_strength_cap config field name
# (2026-09-13, improvements.txt item 145) -- every strategy here now also
# carries a real `signal_strength_pct` on its own real trade dicts (see
# swingtrade/backtest.py's own simulate_*_signals() trades.append() calls),
# so swingtrade.audit_cap_calibration() can run directly against real_trades
# without a separate collector script. Strategies without an entry here
# either don't use this scoring pattern (rsi/breakout/etc.) or are
# dormant/retired (momentum_burst/squeeze_breakout's own live status --
# see config_loader.py -- adx_trend_entry).
STRENGTH_CAP_FIELD = {
    "ma_crossover": "ma_crossover_strength_cap_pct",
    "pairs": "pairs_zscore_strength_cap",
    "momentum_rank": "momentum_strength_cap_pct",
    "pead": "pead_strength_cap_pct",
}

# Strategy -> its own precompute_*_frame() function (2026-09-13,
# improvements.txt item 146) -- backs the opt-in --with-lookahead-audit
# flag. Only the 6 real, actively-used strategies swingtrade.audit_no_lookahead()
# was already real-data-confirmed clean against (improvements.txt item
# 139 + its same-day extension) -- dormant/retired strategies
# (momentum_burst/adx_trend_entry/etc.) are deliberately excluded, same
# scoping STRENGTH_CAP_FIELD above uses.
LOOKAHEAD_PRECOMPUTE_FN = {
    "rsi": precompute_rsi_frame,
    "breakout": precompute_breakout_frame,
    "squeeze_breakout": precompute_squeeze_breakout_frame,
    "ma_crossover": precompute_ma_crossover_frame,
    "pairs": precompute_pairs_frame,
    "momentum_rank": precompute_momentum_frame,
}
LOOKAHEAD_AUDIT_MAX_TICKERS = 5   # bounded regardless of universe size -- this
LOOKAHEAD_AUDIT_MAX_DATES = 5     # doubles precompute cost per sample, so it's
                                   # deliberately opt-in and small by default

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
RANDOM_BASELINE_SEEDS = [1, 42, 7, 99, 2024]  # deliberately fewer than DEFAULT_HOLDOUT_SEEDS'
                                               # 10 -- each extra seed here re-runs a full
                                               # random-entry simulation pass across every
                                               # ticker (real compute cost), unlike ticker-
                                               # holdout re-partitioning (free, just relabels
                                               # already-computed trades) -- 5 seeds already
                                               # captures most of the variance-reduction
                                               # benefit for the ALL-TICKERS headline number
                                               # this backs (see --random-baseline-seeds)


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


def _average_by_key(per_seed_dicts: list[dict]) -> dict:
    """Averages a list of {key: summary} dicts (one per random-baseline
    seed, e.g. summarize_by_period()'s own year->summary shape) into one
    {key: averaged_summary} -- the BY YEAR/BY VOLATILITY REGIME counterpart
    to optimize.average_summaries() (2026-09-13, improvements.txt item 141),
    which only averages a single flat summary, not a dict of them keyed by
    year/regime. A key missing from some seeds (e.g. a year with zero
    random trades under an unlucky seed) is averaged over however many
    seeds actually produced it, not treated as 0 or excluded entirely."""
    all_keys = set()
    for d in per_seed_dicts:
        all_keys.update(d.keys())
    return {
        key: average_summaries([d[key] for d in per_seed_dicts if key in d])
        for key in all_keys
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strategy",
        choices=[
            "rsi", "breakout", "pullback", "breakout_retest", "week52_high",
            "momentum_burst", "squeeze_breakout", "adx_trend_entry", "ma_crossover", "pairs",
            "insider_buying", "momentum_rank", "pead",
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
        "--momentum-entry-fill", choices=["limit", "next_open"], default=None,
        help="Override config.momentum_entry_fill (--strategy momentum_rank only) -- same "
             "'limit' vs. 'next_open' choice as --momentum-burst-entry-fill, checked from day one "
             "per the item-37 lesson rather than added after promotion. Default: whatever the "
             "tested config already has ('limit').",
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
    parser.add_argument(
        "--use-research-cutoff", action="store_true",
        help="Use run_backtest.RESEARCH_HOLDOUT_CUTOFF instead of today as --end's default "
             "(ignored if --end is also given explicitly) -- see that constant's own docstring: "
             "reserves every day after the cutoff as genuine, never-yet-tuned-against future "
             "data, meant to be checked ONCE, right before a final live-promotion decision. Use "
             "this for any NEW strategy validation run -- the plain today-default remains fine "
             "for live-status checks, which this flag isn't for.",
    )
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument(
        "--watchlist-file", type=Path, default=None,
        help="Use a different watchlist JSON file instead of watchlist.txt (e.g. smallmid_watchlist.txt, "
             "adr_watchlist.txt) -- same schema, read via watchlist.read_tickers()/read_ticker_sectors(). "
             "Ignored if --tickers is also given. Lets this script validate a strategy's edge against a "
             "genuinely different universe without a one-off script per universe (2026-09-13, item 137 -- "
             "the third-universe generalization test).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for the random-entry day selection.")
    parser.add_argument(
        "--random-baseline-seeds", default=None,
        help="Comma-separated seeds to average the RANDOM baseline's OWN ALL-TICKERS summary over, "
             "instead of trusting a single draw (2026-09-13, improvements.txt item 132) -- the same "
             "sampling-noise problem --holdout-seeds already fixed for ticker-holdout splits, never "
             "applied to the random-entry draw itself until now. Real trades are unaffected (they're "
             "deterministic). Default: 5 seeds. Pass a single seed (e.g. '1') to reproduce the old "
             "one-draw ALL-TICKERS behavior exactly, e.g. for a fast smoke test. NOTE: only the "
             "ALL-TICKERS section below is multi-seed-averaged this way -- BY YEAR/TUNE/HOLDOUT/"
             "MONTE CARLO still use the single --seed draw (a known, not-yet-extended gap, see "
             "[[feedback_strategy_validation_pipeline]] point 21b).",
    )
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
        "--portfolio-starting-capital", type=float, default=10_000.0,
        help="Starting capital for the new PORTFOLIO-CONSTRAINED REPLAY section (2026-09-13, "
             "swingtrade.simulate_portfolio_constrained()) -- replays REAL trades through a single, "
             "finite-capital account (flat --portfolio-position-budget per trade, "
             "config.max_sector_allocation_pct/max_total_deployed_pct caps) instead of treating every "
             "signal as independently takeable. Default: $10,000 (an illustrative figure, not derived "
             "from any real account).",
    )
    parser.add_argument(
        "--portfolio-position-budget", type=float, default=250.0,
        help="Flat $ per position for the portfolio-constrained replay -- matches ingest.py's own "
             "DEFAULT_POSITION_BUDGET (the live flat-sizing default when no --risk-amount is given).",
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
    parser.add_argument(
        "--with-dividend-drag", action="store_true",
        help="Fetch real dividend history (one extra yfinance call per ticker, see "
             "yfinance.Ticker(ticker).dividends) and run swingtrade.audit_dividend_drag() against "
             "the REAL trades (2026-09-13, improvements.txt item 143) -- quantifies how much this "
             "backtest's pnl_pct understates real total return by never crediting dividends paid "
             "during a holding period. Pure measurement, adds a report section only -- never "
             "changes pnl_pct itself. Off by default (extra network cost); most informative for "
             "higher-dividend-yield universes (adr_watchlist.txt, smallmid financials/REITs/utilities) "
             "-- negligible for this project's own low-dividend primary watchlist.",
    )
    parser.add_argument(
        "--with-lookahead-audit", action="store_true",
        help="Run swingtrade.audit_no_lookahead() against a small, bounded sample "
             f"({LOOKAHEAD_AUDIT_MAX_TICKERS} tickers x {LOOKAHEAD_AUDIT_MAX_DATES} dates, regardless "
             "of universe size) of this strategy's own precompute_*_frame() function (2026-09-13, "
             "improvements.txt item 146) -- confirms no future code change accidentally introduced a "
             "look-ahead leak. No extra network fetch (reuses data already fetched for the real "
             "backtest itself), but doubles precompute cost for the sampled tickers/dates, so opt-in. "
             "A no-op for strategies without an entry in LOOKAHEAD_PRECOMPUTE_FN (dormant/retired "
             "strategies, or ones never real-data-audited this way).",
    )
    args = parser.parse_args()

    holdout_seeds = (
        [int(s.strip()) for s in args.holdout_seeds.split(",") if s.strip()]
        if args.holdout_seeds else list(DEFAULT_HOLDOUT_SEEDS)
    )
    random_baseline_seeds = (
        [int(s.strip()) for s in args.random_baseline_seeds.split(",") if s.strip()]
        if args.random_baseline_seeds else list(RANDOM_BASELINE_SEEDS)
    )

    if args.end:
        end = pd.Timestamp(args.end)
    elif args.use_research_cutoff:
        end = RESEARCH_HOLDOUT_CUTOFF
        print(f"Using research holdout cutoff as --end: {end.date()} "
              "(see run_backtest.RESEARCH_HOLDOUT_CUTOFF's own docstring).")
    else:
        end = pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 5)

    watchlist_file = args.watchlist_file if args.watchlist_file else WATCHLIST_FILE
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not watchlist_file.exists():
            print(f"[ERROR] Watchlist file not found: {watchlist_file}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(watchlist_file)
    sector_lookup = read_ticker_sectors(watchlist_file) if watchlist_file.exists() else {}

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
    if args.momentum_entry_fill is not None:
        config = swingtrade.TradingConfig(**{**config.to_dict(), "momentum_entry_fill": args.momentum_entry_fill})
        config_label += f" (momentum_entry_fill overridden to {args.momentum_entry_fill})"
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
    elif args.strategy == "momentum_rank":
        print(f"  momentum_lookback_days={config.momentum_lookback_days}, "
              f"momentum_top_percentile_min={config.momentum_top_percentile_min}, "
              f"momentum_entry_fill={config.momentum_entry_fill}, "
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

    # Data-quality audit (2026-09-13, improvements.txt item 144) -- free
    # (no extra fetch, checks data already in memory), so unconditional
    # rather than opt-in: a vendor-side bad tick (Low > High, a zero/
    # negative price) could otherwise silently corrupt a signal or
    # settlement decision with nothing to catch it. Large single-day moves
    # (>50%) are flagged too but never treated as an error -- a real
    # crash/spike is a legitimate event, not a bug, and this codebase's
    # own no-look-ahead discipline must still respect it either way.
    structural_issues = 0
    for ticker, df in ticker_data.items():
        dq = swingtrade.audit_data_quality(df)
        if not dq["is_clean"]:
            structural_issues += dq["n_structural_violations"]
            print(f"  [DATA QUALITY] {ticker}: {dq['n_structural_violations']} structural violation(s) "
                  f"at {dq['structural_violation_dates']} -- see swingtrade.audit_data_quality()'s own "
                  "docstring for what this checks.", file=sys.stderr)
    if structural_issues == 0:
        print(f"Data quality: 0 structural violations across {len(ticker_data)} ticker(s) (clean).")
    else:
        print(f"[WARN] Data quality: {structural_issues} TOTAL structural violation(s) found -- "
              "see above for which ticker(s)/date(s).", file=sys.stderr)

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

    dividend_history: dict = {}
    if args.with_dividend_drag:
        print(f"\nFetching real dividend history for {len(ticker_data)} ticker(s)...")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            try:
                divs = yf.Ticker(ticker).dividends
                if len(divs) > 0:
                    dividend_history[ticker] = divs
            except Exception as exc:
                print(f"  [WARN] {ticker} dividends: {exc}", file=sys.stderr)
        print(f"Got dividend history for {len(dividend_history)}/{len(ticker_data)} ticker(s).")

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

    insider_data: dict[str, pd.DataFrame] = {}
    if args.strategy in INSIDER_AWARE_STRATEGIES:
        print(f"\nFetching insider purchase history for {len(ticker_data)} ticker(s)...")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            insider_data[ticker] = fetch_insider_purchases(ticker, config)
        found = sum(1 for d in insider_data.values() if len(d) > 0)
        total_events = sum(len(d) for d in insider_data.values())
        print(f"Got {total_events} real purchase event(s) across {found}/{len(ticker_data)} ticker(s).")

    earnings_surprise_data: dict[str, pd.DataFrame] = {}
    if args.strategy in PEAD_AWARE_STRATEGIES:
        print(f"\nFetching earnings-surprise history for {len(ticker_data)} ticker(s)... "
              "(see run_backtest.fetch_earnings_surprises()'s own EXPLICIT point-in-time-integrity "
              "caveat -- read it before trusting any result below)")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            earnings_surprise_data[ticker] = fetch_earnings_surprises(ticker)
        found = sum(1 for d in earnings_surprise_data.values() if len(d) > 0)
        total_events = sum(len(d) for d in earnings_surprise_data.values())
        print(f"Got {total_events} real reported quarter(s) across {found}/{len(ticker_data)} ticker(s).")

    sector_price_panels: dict[str, pd.DataFrame] = {}
    if args.strategy in PAIR_AWARE_STRATEGIES:
        # No new network fetch -- every ticker's OHLCV is already in
        # ticker_data. One wide Close-price panel per sector (>= 2 members),
        # built once and sliced per ticker below (see peer_kwargs()).
        by_sector: dict[str, list[str]] = {}
        for t in ticker_data:
            by_sector.setdefault(sector_lookup.get(t, "Unknown"), []).append(t)
        for sector, members in by_sector.items():
            if len(members) < 2:
                continue
            sector_price_panels[sector] = pd.DataFrame({m: ticker_data[m]["Close"] for m in members})

    momentum_rank_frame: pd.DataFrame | None = None
    if args.strategy in MOMENTUM_AWARE_STRATEGIES:
        # No new network fetch -- every ticker's OHLCV is already in
        # ticker_data. ONE wide Close-price panel across the WHOLE universe
        # (no sector partitioning, unlike pairs' sector_price_panels above --
        # ranking is universe-wide by design), rank-computed ONCE via
        # swingtrade.compute_momentum_rank_frame() and sliced per ticker
        # below (see momentum_kwargs()). Built from the FULL ticker set,
        # not just tune-side tickers -- this deliberately matches pairs'
        # own precedent (its sector panels are never split by tune/holdout
        # either) and the design decision confirmed for this strategy: no
        # forward-looking leak results, since ranking only ever uses past
        # prices, and it matches what a live scan would realistically use.
        momentum_panel = pd.DataFrame({t: ticker_data[t]["Close"] for t in ticker_data})
        momentum_rank_frame = swingtrade.compute_momentum_rank_frame(momentum_panel, config.momentum_lookback_days)

    if args.with_lookahead_audit:
        precompute_fn = LOOKAHEAD_PRECOMPUTE_FN.get(args.strategy)
        if precompute_fn is None:
            print(f"\n[LOOKAHEAD AUDIT] --strategy {args.strategy} has no entry in "
                  "LOOKAHEAD_PRECOMPUTE_FN -- skipped.")
        else:
            sample_tickers = list(ticker_data.keys())[:LOOKAHEAD_AUDIT_MAX_TICKERS]
            print(f"\n=== LOOK-AHEAD AUDIT ({len(sample_tickers)} ticker(s) x up to "
                  f"{LOOKAHEAD_AUDIT_MAX_DATES} date(s) -- see improvements.txt item 146) ===")
            total_checked = total_mismatches = 0
            for ticker in sample_tickers:
                df = ticker_data[ticker]
                if len(df) < 260:
                    continue
                sample_dates = list(df.index[250::max(1, (len(df) - 250) // LOOKAHEAD_AUDIT_MAX_DATES)][:LOOKAHEAD_AUDIT_MAX_DATES])
                extra_series = {}
                if args.strategy in ("ma_crossover", "squeeze_breakout", "breakout"):
                    extra_series = {
                        "market_df": market_data,
                        "sector_df": sector_data.get(sector_lookup.get(ticker, "Unknown")),
                    }
                elif args.strategy == "pairs":
                    panel = sector_price_panels.get(sector_lookup.get(ticker, "Unknown"))
                    extra_series = {
                        "peer_prices": panel.drop(columns=[ticker]) if panel is not None and ticker in panel.columns else None,
                    }
                elif args.strategy == "momentum_rank":
                    extra_series = {
                        "rank_column": momentum_rank_frame[ticker] if momentum_rank_frame is not None and ticker in momentum_rank_frame.columns else None,
                    }
                result = swingtrade.audit_no_lookahead(precompute_fn, df, sample_dates, config, extra_series=extra_series)
                total_checked += result["dates_checked"]
                total_mismatches += len(result["mismatches"])
                if result["mismatches"]:
                    print(f"  [LOOKAHEAD AUDIT] {ticker}: {len(result['mismatches'])} mismatch(es) -- "
                          f"{result['mismatches'][:3]}")
            if total_mismatches == 0:
                print(f"  Clean: 0 mismatches across {total_checked} sampled date-check(s).")
            else:
                print(f"  [WARN] {total_mismatches} TOTAL mismatch(es) found across {total_checked} "
                      "sampled date-check(s) -- see above for which ticker(s)/date(s)/column(s).")

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
    elif args.strategy == "pairs":
        real_fn, random_fn = swingtrade.simulate_pairs_signals, swingtrade.simulate_random_pairs_entries
        real_label = "Pairs-timed"
    elif args.strategy == "insider_buying":
        real_fn, random_fn = swingtrade.simulate_insider_buying_signals, swingtrade.simulate_random_insider_buying_entries
        real_label = "Insider_Buying-timed"
    elif args.strategy == "momentum_rank":
        real_fn, random_fn = swingtrade.simulate_momentum_signals, swingtrade.simulate_random_momentum_entries
        real_label = "Momentum_Rank-timed"
    elif args.strategy == "pead":
        real_fn, random_fn = swingtrade.simulate_pead_signals, swingtrade.simulate_random_pead_entries
        real_label = "PEAD-timed"
    else:
        real_fn, random_fn = swingtrade.simulate_ma_crossover_signals, swingtrade.simulate_random_ma_crossover_entries
        real_label = "MA_Crossover-timed"

    real_trades = []
    random_trades = []
    real_counts = {}
    print(f"\nSimulating REAL {real_label} strategy and matched-count RANDOM-entry baseline "
          f"for {len(ticker_data)} ticker(s), {start.date()}..{end.date()}...")
    earnings_kwargs = lambda ticker: (  # noqa: E731 -- see EARNINGS_AWARE_STRATEGIES
        {"earnings_dates": earnings_data.get(ticker)} if args.strategy in EARNINGS_AWARE_STRATEGIES else {}
    )
    sector_kwargs = lambda ticker: (  # noqa: E731
        {"sector_ohlcv": sector_data.get(sector_lookup.get(ticker, "Unknown"))}
        if args.strategy in SECTOR_AWARE_STRATEGIES else {}
    )
    def peer_kwargs(ticker):
        if args.strategy not in PAIR_AWARE_STRATEGIES:
            return {}
        panel = sector_price_panels.get(sector_lookup.get(ticker, "Unknown"))
        if panel is None or ticker not in panel.columns:
            return {"peer_prices": None}
        return {"peer_prices": panel.drop(columns=[ticker])}
    insider_kwargs = lambda ticker: (  # noqa: E731 -- see INSIDER_AWARE_STRATEGIES
        {"insider_purchases": insider_data.get(ticker)} if args.strategy in INSIDER_AWARE_STRATEGIES else {}
    )
    pead_kwargs = lambda ticker: (  # noqa: E731 -- see PEAD_AWARE_STRATEGIES
        {"earnings_surprises": earnings_surprise_data.get(ticker)} if args.strategy in PEAD_AWARE_STRATEGIES else {}
    )
    def momentum_kwargs(ticker):
        if args.strategy not in MOMENTUM_AWARE_STRATEGIES:
            return {}
        if momentum_rank_frame is None or ticker not in momentum_rank_frame.columns:
            return {"rank_column": None}
        return {"rank_column": momentum_rank_frame[ticker]}
    for i, (ticker, ohlcv) in enumerate(ticker_data.items()):
        sector = sector_lookup.get(ticker, "Unknown")
        real = real_fn(
            ticker, ohlcv, market_data, start, end, config,
            **earnings_kwargs(ticker), sector=sector, **sector_kwargs(ticker), **peer_kwargs(ticker),
            **insider_kwargs(ticker), **momentum_kwargs(ticker), **pead_kwargs(ticker),
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

    # Multi-seed-averaged RANDOM baseline (2026-09-13, improvements.txt items
    # 132/141) -- a single random draw (the `random_trades` above, from
    # --seed alone) carries the same sampling-noise risk --holdout-seeds
    # already fixed for ticker-holdout splits, just never applied to the
    # random-ENTRY draw itself. Reuses `random_trades`'s own seed (args.seed)
    # as one of the seeds averaged over, if it's already in
    # random_baseline_seeds, rather than wastefully re-simulating it.
    # Originally (item 132) this only covered the ALL-TICKERS headline;
    # item 141 extended it to BY YEAR/TUNE/HOLDOUT/MONTE CARLO below too, by
    # keeping each seed's own raw trades (not just its summary) so every
    # section can reuse the identical per-seed draws rather than each
    # picking its own single seed independently.
    per_seed_random_trades = []
    for s in random_baseline_seeds:
        if s == args.seed:
            per_seed_random_trades.append(random_trades)
            continue
        seed_rng = random.Random(s)
        seed_random_trades = []
        for ticker, ohlcv in ticker_data.items():
            sector = sector_lookup.get(ticker, "Unknown")
            rand = random_fn(
                ticker, ohlcv, market_data, start, end, real_counts.get(ticker, 0), seed_rng, config,
                **earnings_kwargs(ticker), sector=sector,
            )
            seed_random_trades.extend(rand)
        per_seed_random_trades.append(seed_random_trades)
    random_all_avg = average_summaries([summarize(t) for t in per_seed_random_trades])

    print("\n=== ALL TICKERS ===")
    print(f"  REAL   ({real_label}): {summarize(real_trades)}")
    print(f"  RANDOM (matched count, avg of {len(random_baseline_seeds)} seeds {random_baseline_seeds}): "
          f"{random_all_avg}")

    backtest_ic = ic_tracking.backtest_ic_check(real_trades)
    print(f"\n=== BACKTEST-TIME IC (does Trade_Score itself rank REAL trades' outcomes, "
          f"not just beat random on aggregate returns?) ===")
    if backtest_ic["ic"] is None:
        print(f"  n={backtest_ic['n']} -- too thin (< {ic_tracking.MIN_TRADES_FOR_BACKTEST_IC}) to trust, "
              "or every trade_score's strategy doesn't record one yet (see swingtrade/backtest.py).")
    else:
        print(f"  n={backtest_ic['n']}  backtest_ic={backtest_ic['ic']:.3f}")
        print("  This is a DIFFERENT question from the ALL/TUNE/HOLDOUT sharpe_like/win_rate checks "
              "above -- those ask 'does this beat random on aggregate', this asks 'does the score "
              "actually rank which specific trades do better or worse'. A strategy can pass the "
              "former with ~zero real answer to the latter (see improvements.txt for the real "
              "finding that motivated this check).")

    # 2026-09-13 (improvements.txt item 145) -- is the *_strength_cap_pct
    # field itself well-calibrated against what these REAL trades actually
    # achieved? Free now that real_trades carry their own signal_strength_pct
    # directly (previously needed a separate collector script per strategy,
    # see audit_strength_cap_calibration.py). Unconditional, like the data-
    # quality check -- a no-op print for strategies without an entry in
    # STRENGTH_CAP_FIELD (rsi/breakout/etc. don't use this scoring pattern).
    cap_field = STRENGTH_CAP_FIELD.get(args.strategy)
    if cap_field:
        strength_values = [t["signal_strength_pct"] for t in real_trades if "signal_strength_pct" in t]
        cap_value = getattr(config, cap_field)
        cap_result = swingtrade.audit_cap_calibration(strength_values, cap_value)
        print(f"\n=== STRENGTH-CAP CALIBRATION (is config.{cap_field}={cap_value} well-calibrated "
              "against what these REAL trades actually achieved? see improvements.txt item 140) ===")
        print(f"  {cap_result}")
        if cap_result["likely_too_high"]:
            print("  [FLAG] cap looks miscalibrated TOO HIGH -- real strong signals barely use this component's points.")
        if cap_result["likely_too_low"]:
            print("  [FLAG] cap looks miscalibrated TOO LOW -- many real signals saturate to the same max score.")

    # 2026-09-13 (improvements.txt item 135) -- a REAL confidence figure on
    # the gap itself, not just eyeballing whether REAL's sharpe_like number
    # looks bigger than RANDOM's. See swingtrade.permutation_test_gap()'s
    # own docstring for the full method (a standard permutation/shuffle
    # test) and how this differs from DSR (selection bias across MANY
    # Optuna trials, not sampling uncertainty within this ONE comparison).
    perm_test = swingtrade.permutation_test_gap(real_trades, random_trades, summarize)
    print("\n=== PERMUTATION TEST (how likely is this GAP by pure chance, if REAL/RANDOM were truly interchangeable?) ===")
    if perm_test["p_value"] is None:
        print(f"  observed_gap={perm_test['observed_gap']} -- p_value undefined (too few resolved trades "
              "for a meaningful shuffle, or every shuffle produced an undefined sharpe_like).")
    else:
        print(f"  observed_gap={perm_test['observed_gap']}  p_value={perm_test['p_value']:.4f}  "
              f"(null distribution: mean={perm_test['null_mean']}, std={perm_test['null_std']}, "
              f"{perm_test['n_permutations']} shuffles)")
        print("  LOW p_value = the observed gap would be unusual/rare under 'no real timing skill' -- "
              "real evidence of a genuine edge, not just a single favorably-framed comparison.")

    print(f"\n=== BY YEAR (does the edge hold up over time, or is it concentrated in one stretch? "
          f"RANDOM avg of {len(random_baseline_seeds)} seeds) ===")
    real_by_year = swingtrade.summarize_by_period(real_trades, summarize)
    per_seed_by_year = [swingtrade.summarize_by_period(t, summarize) for t in per_seed_random_trades]
    random_by_year_avg = _average_by_key(per_seed_by_year)
    for year in sorted(set(real_by_year) | set(random_by_year_avg)):
        print(f"  {year}  REAL   ({real_label}): {real_by_year.get(year)}")
        print(f"  {year}  RANDOM (matched count): {random_by_year_avg.get(year)}")

    # 2026-09-13 (improvements.txt item 134) -- a DIFFERENT regime axis from
    # BY YEAR above: EVERY simulate_*_signals() function already hard-gates
    # on market_uptrend_from_frame() (SPY Close >= its own SMA_TREND), so a
    # bull-vs-bear breakdown would be moot here (bear days are excluded by
    # construction for every strategy). Volatility regime is NOT excluded
    # by that gate -- a confirmed uptrend can still be realized-volatility
    # elevated or subdued -- making it the informative axis within the
    # population every strategy actually trades in. See
    # swingtrade.compute_volatility_regime_series()'s own docstring.
    volatility_regime = swingtrade.compute_volatility_regime_series(market_data)
    print(f"\n=== BY VOLATILITY REGIME (does the edge hold in both calm and turbulent "
          f"markets, or is it concentrated in one? RANDOM avg of {len(random_baseline_seeds)} seeds) ===")
    real_by_regime = swingtrade.summarize_by_volatility_regime(real_trades, volatility_regime, summarize)
    per_seed_by_regime = [
        swingtrade.summarize_by_volatility_regime(t, volatility_regime, summarize) for t in per_seed_random_trades
    ]
    random_by_regime_avg = _average_by_key(per_seed_by_regime)
    for regime in ("elevated", "normal"):
        print(f"  {regime:9s}  REAL   ({real_label}): {real_by_regime.get(regime)}")
        print(f"  {regime:9s}  RANDOM (matched count): {random_by_regime_avg.get(regime)}")

    if args.holdout_frac > 0:
        real_tune_avg, real_holdout_avg = average_holdout_summary(
            real_trades, sector_lookup, args.holdout_frac, holdout_seeds, summarize
        )
        # RANDOM's TUNE/HOLDOUT is now averaged over BOTH axes (2026-09-13,
        # item 141): for EACH random-baseline seed's own trades, re-split by
        # ticker-holdout across `holdout_seeds` (cheap -- average_holdout_summary()
        # only re-partitions already-simulated trades, no re-simulation), then
        # average those per-seed TUNE/HOLDOUT results across the random-baseline
        # seeds too -- closing the "known, not-yet-extended gap" item 132's own
        # help text flagged (BY YEAR/TUNE/HOLDOUT/MONTE CARLO previously stuck
        # on the single --seed draw even after ALL-TICKERS got multi-seed
        # averaging).
        per_seed_tune, per_seed_holdout = [], []
        for t in per_seed_random_trades:
            tune, holdout = average_holdout_summary(t, sector_lookup, args.holdout_frac, holdout_seeds, summarize)
            per_seed_tune.append(tune)
            per_seed_holdout.append(holdout)
        random_tune_avg = average_summaries(per_seed_tune)
        random_holdout_avg = average_summaries(per_seed_holdout)

        print(f"\n=== TUNE (REAL avg of {len(holdout_seeds)} holdout seeds; "
              f"RANDOM avg of {len(holdout_seeds)} holdout seeds x {len(random_baseline_seeds)} baseline seeds) ===")
        print(f"  REAL   ({real_label}): {real_tune_avg}")
        print(f"  RANDOM (matched count): {random_tune_avg}")

        print(f"\n=== HOLDOUT (REAL avg of {len(holdout_seeds)} holdout seeds; "
              f"RANDOM avg of {len(holdout_seeds)} holdout seeds x {len(random_baseline_seeds)} baseline seeds) ===")
        print(f"  REAL   ({real_label}): {real_holdout_avg}")
        print(f"  RANDOM (matched count): {random_holdout_avg}")

    print(f"\n=== MONTE CARLO DRAWDOWN (1000 reshuffles of each's own trade order; "
          f"RANDOM avg of {len(random_baseline_seeds)} seeds) ===")
    per_seed_mc = [swingtrade.monte_carlo_drawdown(t) for t in per_seed_random_trades]
    random_mc_avg = average_summaries([d for d in per_seed_mc if d is not None])
    print(f"  REAL   ({real_label}): {swingtrade.monte_carlo_drawdown(real_trades)}")
    print(f"  RANDOM (matched count): {random_mc_avg}")

    print(f"\n=== PORTFOLIO-CONSTRAINED REPLAY (${args.portfolio_starting_capital:,.0f} capital, "
          f"${args.portfolio_position_budget:,.0f}/position, sector cap {config.max_sector_allocation_pct*100:.0f}%"
          f"{f', portfolio cap {config.max_total_deployed_pct*100:.0f}%' if config.max_total_deployed_pct else ''}"
          f", RANDOM avg of {len(random_baseline_seeds)} seeds) ===")
    print("  A different question from every check above: of every signal generated, how many could a REAL,")
    print("  single, finite-capital account actually have AFFORDED to take, and what does that account's own")
    print("  dollar equity curve/drawdown/CAGR look like -- vs. every prior metric pooling every signal as if")
    print("  capital were unlimited. See swingtrade.simulate_portfolio_constrained()'s own docstring.")
    per_seed_portfolio = [
        swingtrade.simulate_portfolio_constrained(
            t, args.portfolio_starting_capital, args.portfolio_position_budget,
            config.max_sector_allocation_pct, config.max_total_deployed_pct, sector_lookup,
        )
        for t in per_seed_random_trades
    ]
    random_portfolio_avg = average_summaries(per_seed_portfolio)
    print(f"  REAL   ({real_label}): {swingtrade.simulate_portfolio_constrained(real_trades, args.portfolio_starting_capital, args.portfolio_position_budget, config.max_sector_allocation_pct, config.max_total_deployed_pct, sector_lookup)}")
    print(f"  RANDOM (matched count): {random_portfolio_avg}")

    # 2026-09-14 (improvements.txt item 151) -- free (no extra fetch, just
    # analyzes real_trades already generated), so unconditional like the
    # data-quality/cap-calibration checks. Every simulate_*_signals()
    # function walks eligible days independently with no notion of
    # "already holding this ticker" -- this measures how often that
    # actually produces a same-ticker double-exposure event nothing else
    # catches (distinct from the sector/portfolio caps above, which have
    # no notion of "same ticker" either).
    overlap_result = swingtrade.audit_same_ticker_overlap(real_trades)
    print(f"\n=== SAME-TICKER OVERLAP (does this backtest ever double up exposure to one ticker? "
          f"see improvements.txt item 151) ===")
    print(f"  REAL ({real_label}): {overlap_result}")
    if overlap_result["n_overlapping_pairs"]:
        print(f"  {overlap_result['pct_trades_in_an_overlap']}% of real trades touch at least one same-ticker "
              "overlap -- consider swingtrade.simulate_portfolio_constrained()'s own max_positions_per_ticker "
              "param if this matters for a live capital decision.")

    if args.with_dividend_drag:
        print("\n=== DIVIDEND DRAG (does pnl_pct understate real total return by never crediting "
              "dividends paid during a holding period? see improvements.txt item 143) ===")
        dividend_result = swingtrade.audit_dividend_drag(real_trades, dividend_history)
        print(f"  REAL ({real_label}): {dividend_result}")
        if dividend_result["avg_missed_pct_all_trades"]:
            print(f"  On average, this backtest understates real total return by "
                  f"{dividend_result['avg_missed_pct_all_trades']}pp per trade on this universe "
                  "(pure measurement -- pnl_pct itself is unchanged).")

    print()
    print(f"If REAL's sharpe_like/win_rate isn't meaningfully better than RANDOM's (same trade")
    print(f"count, same universe, same stop/target/holding-period structure), {args.strategy} timing")
    print("is not adding real predictive information beyond the payoff structure + market drift.")


if __name__ == "__main__":
    main()
