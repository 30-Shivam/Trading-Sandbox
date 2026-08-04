"""
Optuna learning engine (Phase 5).

Searches the RSI/ATR parameters named in the original architecture goal --
rsi_oversold_threshold, atr_take_profit_multiplier, stop_loss_atr_multiplier
-- via Bayesian optimization (Optuna's TPE sampler), scoring each candidate
on its AGGREGATE out-of-sample performance pooled across every Phase 4
walk-forward fold. Never a single fold's in-sample fit -- see
swingtrade/backtest.py for why that would be curve-fitting.

Historical OHLCV is fetched once up front and reused across every trial;
each trial then only re-runs the in-memory backtest (fast, pure
computation), which is what makes a 50+ trial search practical.

Objective: sharpe_like (mean / stdev of pooled out-of-sample pnl_pct) --
rewards consistency, not just raw returns, matching the risk-management
focus already baked into the rest of this system (Stop_Loss, RRR, capital
allocation). Candidates that produce too few pooled trades to trust
(< MIN_TRADES_FOR_SCORE) are penalized rather than scored on a lucky small
sample -- the same anti-curve-fitting spirit as walk-forward itself, just
applied within a single trial too.

The winning trial is written to MongoDB's System_Config collection as a new
`candidate` document -- it is NEVER auto-promoted to `active`. A human
reviews it and runs promote_config.py to actually put it live
(champion/challenger pattern): a noisy optimization run should not be able
to silently degrade live trading.

Recency-weighted scoring: out-of-sample folds closer to --end (i.e. closer
to today, the period your actual live trades fall in) count exponentially
more toward the pooled score than older folds, via --recency-half-life-days
(default 180). This is deliberately NOT the same thing as mixing live
Trade_Outcomes documents directly into the score -- those reflect one
specific historical config, not each trial's varying candidate, so grafting
them in would conflate two different questions (see summarize_weighted()).
Instead, every trial gets re-simulated fairly over the SAME recent calendar
window under ITS OWN candidate config, and that recent window is simply
weighted more heavily -- so what's been working lately (the regime your
live trades are actually part of) has outsized influence on the winner,
without corrupting the cross-candidate comparison. Pass --recency-half-life-days
0 to fall back to old-style uniform pooling.

Correlation-adjusted scoring: pooled trades also aren't independent of each
other -- a real, observed failure mode is dozens of correlated Technology
signals firing the same day during one sector move, which a naive pooled
mean/std would count as dozens of independent data points instead of one
correlated event. summarize_weighted() combines the recency weight above
with swingtrade.compute_cluster_weights() (same-day/same-sector clusters
split a combined weight of 1.0), via watchlist.read_ticker_sectors. Both
weighting effects flow into the same Kish effective-sample-size gate for
MIN_TRADES_FOR_SCORE, so a candidate can't look well-sampled just because it
produced a large trade_count that's secretly a handful of correlated events
repeated many times.

Live Trade_Outcomes are still reported separately for context (how many
exist, their aggregate performance) -- useful as a sanity check against what
the backtest predicts, but not an input to the score.

Ticker-universe holdout validation: walk-forward already holds out TIME
rigorously (in-sample/out-of-sample folds), but until now every search also
tuned AND "validated" on the exact same ticker universe -- a config could be
quietly overfit to the specific names in the watchlist without anyone
catching it (the walk-forward folds only ever ask "does this generalize to
a later date for these same tickers," never "does this generalize to a
ticker the search never saw at all"). split_tickers_holdout() partitions the
fetched tickers into a TUNE set (all trials -- objective, baseline, Optuna
search -- only ever see this) and a HOLDOUT set that Optuna never touches
during the search, stratified by sector so neither set accidentally
concentrates in one sector the way the original live-vs-backtest gap did.
After the search picks a winner, that winning config (and DEFAULT_CONFIG,
for reference) get re-run ONE extra time against the holdout tickers over
the same folds -- reported alongside the tune-set numbers, and stored on
the candidate doc as `holdout_metrics`, so whoever reviews the candidate
before promoting can see whether the edge generalizes to unseen names or
was quietly specific to the tuning watchlist. Pass --holdout-frac 0 to
disable (old behavior: tune and "validate" on every ticker).

Strategy-agnostic since the breakout/trend-following signal was added
(improvements.txt's STRATEGIC PIVOT section, built after benchmark_random_entry.py
showed RSI-oversold timing carries no real predictive value over random entry
days). --strategy rsi (default) searches the original RSI/ATR/stop-loss/
streak-penalty space; --strategy breakout searches breakout_lookback_days/
atr_take_profit_multiplier/stop_loss_atr_multiplier/
breakout_rsi_overbought_threshold/breakout_relative_strength_min/
breakout_volume_ratio_min instead -- everything above (WFO, recency
weighting, correlation-adjustment, ticker-holdout, champion/challenger)
applies identically to either.

Usage:
    python optimize.py --trials 50 --start 2023-01-01 --end 2026-07-01
    python optimize.py --trials 50 --recency-half-life-days 90   # weight recent regime more heavily
    python optimize.py --trials 50 --holdout-frac 0.3            # hold out 30% of tickers by sector
    python optimize.py --trials 30 --strategy breakout           # search the breakout signal instead
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import optuna
import pandas as pd

import storage
import swingtrade
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

MIN_TRADES_FOR_SCORE = 15    # penalize candidates with too few pooled trades to trust
UNDER_SAMPLED_PENALTY = -10.0  # well below any realistic sharpe_like, but finite (no inf/NaN into Optuna)
UNDER_SAMPLED_DRAWDOWN_PENALTY = 100.0  # --multi-objective only: worst-possible (100%) drawdown for an
                                         # under-sampled trial, so it's dominated on both axes rather than
                                         # accidentally looking "safe" by having no trades to draw down from

# Search space bounds for the tunables this study is allowed to move.
# A prior run pinned rsi_oversold_threshold at 54.9/55 and stop_loss_atr_multiplier
# at 0.53/0.5 -- both right against their old boundaries, which usually means the
# bound was cutting off the true optimum rather than the search finding an interior
# sweet spot. Widened accordingly; atr_take_profit_multiplier landed interior
# (2.13 of 1.0-3.0) so it's untouched.
RSI_OVERSOLD_RANGE = (15.0, 70.0)
ATR_TAKE_PROFIT_RANGE = (1.0, 3.0)
# Upper bound widened 3.0 -> 5.0: the breakout search has now pinned
# stop_loss_atr_multiplier at exactly 3.0 in TWO separate searches
# (candidates v14 and v16) -- two data points is a stronger signal than
# one that the old ceiling was cutting off the true optimum, the same
# boundary-pinning pattern that showed up earlier this session for RSI.
# Shared between both strategies' search spaces; harmless for RSI (no
# evidence it wants a wider stop than its already-interior ~2.18 optimum).
STOP_LOSS_ATR_RANGE = (0.25, 5.0)
EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE = (0.0, 4.0)
EXTENDED_DECLINE_PENALTY_CAP_RANGE = (0.0, 50.0)

# Breakout strategy's own search space. A single-window benchmark_random_entry.py
# comparison already showed 20 has no value and 55 shows a real (if modest) edge
# on held-out tickers (see improvements.txt) -- this range lets a real WFO search
# find the actual optimum instead of guessing between two hand-picked points.
BREAKOUT_LOOKBACK_RANGE = (10, 80)
# 100 = practical "disabled" (RSI essentially never reaches it); 50 is a fairly
# restrictive cutoff. Lets Optuna find out whether skipping over-extended
# breakouts (see simulate_breakout_signals) helps at all, and by how much,
# rather than assuming a hand-picked threshold -- this is itself an unvalidated
# hypothesis (improvements.txt item 7a) until a real search says otherwise.
BREAKOUT_RSI_OVERBOUGHT_RANGE = (50.0, 100.0)
# -100 = practical "disabled"; 0 requires the ticker to at least match the
# market over the breakout window, positive values require genuine
# outperformance. Lower bound (-50) is intentionally permissive so the
# search can effectively find "no meaningful filtering" if that's actually
# best, rather than being forced to always filter -- same "let a real
# search decide" reasoning as the overbought threshold above
# (improvements.txt item 5).
BREAKOUT_RELATIVE_STRENGTH_RANGE = (-50.0, 15.0)
# 0.0 = practical "disabled" (a real ratio is always >= 0); upper bound (3.0)
# requires today's volume to be 3x the prior average -- a genuinely
# demanding confirmation threshold. Same "let a real search decide"
# reasoning as the other two breakout filters (improvements.txt item 6).
BREAKOUT_VOLUME_RATIO_RANGE = (0.0, 3.0)
# 0.0 = practical "disabled" (ADX is always >= 0); upper bound (40.0) is
# already a demanding bar in classical TA terms (ADX > 25 is commonly read
# as "trending", > 40 "strong trend"). ADX measures trend STRENGTH,
# independent of direction -- a different dimension than RSI/Relative_Strength/
# Volume_Ratio. Same "let a real search decide" reasoning as the other
# breakout filters (improvements.txt item 17).
BREAKOUT_ADX_MIN_RANGE = (0.0, 40.0)
# Both OBV_Zscore and Squeeze_Zscore (see levels.precompute_breakout_frame)
# are genuine z-scores -- realistically span roughly -3 to +3, so that range
# covers everywhere from "essentially always passes" to "essentially always
# blocks" at either end, same "let a real search decide" reasoning as every
# other breakout filter (improvements.txt item 18).
BREAKOUT_OBV_ZSCORE_MIN_RANGE = (-3.0, 3.0)
BREAKOUT_SQUEEZE_ZSCORE_MAX_RANGE = (-3.0, 3.0)

# slippage_pct / commission_pct_per_trade are deliberately NEVER in this search
# space: they model execution friction, not strategy behavior. Letting Optuna
# tune them would just teach it to zero out the very realism they exist to add.


def split_tickers_holdout(
    tickers: list[str], sector_lookup: dict[str, str], holdout_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Partition `tickers` into (tune, holdout) sets, stratified by sector so
    neither set accidentally over-concentrates in one sector (the exact
    mechanism behind the original live-vs-backtest gap -- see
    backtest_diagnostic.txt). Deterministic given the same seed. A sector
    with only 1-2 tickers may contribute 0 to holdout (round() can floor to
    0) -- expected, not a bug, with small watchlists or small sectors."""
    if holdout_frac <= 0:
        return list(tickers), []

    rng = random.Random(seed)
    by_sector: dict[str, list[str]] = defaultdict(list)
    for t in tickers:
        by_sector[sector_lookup.get(t, "Unknown")].append(t)

    tune, holdout = [], []
    for sector in sorted(by_sector):
        group = sorted(by_sector[sector])
        rng.shuffle(group)
        n_holdout = round(len(group) * holdout_frac)
        holdout.extend(group[:n_holdout])
        tune.extend(group[n_holdout:])
    return tune, holdout


def fold_weight(fold: swingtrade.Fold, end: pd.Timestamp, half_life_days: float) -> float:
    """Exponential recency weight for one fold's out-of-sample window: 1.0
    at `end`, halving every `half_life_days`. half_life_days <= 0 disables
    weighting (every fold counts equally, matching the old behavior)."""
    if half_life_days <= 0:
        return 1.0
    age_days = max((end - fold.out_sample_end).days, 0)
    return 0.5 ** (age_days / half_life_days)


def summarize_weighted(fold_results: list, end: pd.Timestamp, half_life_days: float) -> dict:
    """Like swingtrade.summarize_trades, but pools out-of-sample trades
    weighted by BOTH fold recency (the current market regime counts for
    more than one from a year ago) AND same-day/same-sector correlation
    (see swingtrade.compute_cluster_weights -- 30 correlated Technology
    signals firing one day are one effective observation, not 30
    independent ones). Folds don't overlap in calendar time, so clustering
    across the whole pooled set is equivalent to clustering per-fold and
    concatenating. Uses Kish's effective sample size (not raw trade_count)
    for MIN_TRADES_FOR_SCORE, so heavy weighting -- from either source --
    can't let a big pile of low-weight trades masquerade as a trustworthy
    sample."""
    flat_trades, recency_weights = [], []
    for fr in fold_results:
        w = fold_weight(fr.fold, end, half_life_days)
        for t in fr.out_sample_trades:
            if t["status"] == "OPEN":
                continue
            flat_trades.append(t)
            recency_weights.append(w)

    if not flat_trades:
        return swingtrade.summarize_trades_weighted([], [])

    cluster_weights = swingtrade.compute_cluster_weights(flat_trades)
    combined_weights = [r * c for r, c in zip(recency_weights, cluster_weights)]
    return swingtrade.summarize_trades_weighted(flat_trades, combined_weights)


def build_objective(
    ticker_data: dict, market_data: pd.DataFrame, folds: list, end: pd.Timestamp, half_life_days: float,
    sector_lookup: dict[str, str], strategy: str = "rsi", max_workers: int | None = None,
    multi_objective: bool = False,
):
    """`multi_objective=False` (default) is byte-for-byte the original
    single-scalar-objective behavior -- unchanged, so every existing script/
    workflow that assumes a single-objective study (study.best_trial, the
    candidate-writing flow in main()) keeps working exactly as before.

    `multi_objective=True` makes the objective return a 2-tuple instead:
    (sharpe_like, max_drawdown) -- see swingtrade.compute_max_drawdown()'s
    docstring for why sharpe_like alone is blind to tail risk. Optuna then
    searches for the Pareto front (configs where no other trial is BOTH a
    higher sharpe_like AND a lower drawdown) instead of a single winner --
    see main()'s post-search handling for how one candidate still gets
    selected from that front to actually write to System_Config."""
    def objective(trial: optuna.Trial):
        if strategy == "rsi":
            params = {
                "rsi_oversold_threshold": trial.suggest_float("rsi_oversold_threshold", *RSI_OVERSOLD_RANGE),
                "atr_take_profit_multiplier": trial.suggest_float("atr_take_profit_multiplier", *ATR_TAKE_PROFIT_RANGE),
                "stop_loss_atr_multiplier": trial.suggest_float("stop_loss_atr_multiplier", *STOP_LOSS_ATR_RANGE),
                "extended_decline_penalty_per_day": trial.suggest_float(
                    "extended_decline_penalty_per_day", *EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE
                ),
                "extended_decline_penalty_cap": trial.suggest_float(
                    "extended_decline_penalty_cap", *EXTENDED_DECLINE_PENALTY_CAP_RANGE
                ),
            }
        else:
            params = {
                "breakout_lookback_days": trial.suggest_int("breakout_lookback_days", *BREAKOUT_LOOKBACK_RANGE),
                "atr_take_profit_multiplier": trial.suggest_float("atr_take_profit_multiplier", *ATR_TAKE_PROFIT_RANGE),
                "stop_loss_atr_multiplier": trial.suggest_float("stop_loss_atr_multiplier", *STOP_LOSS_ATR_RANGE),
                "breakout_rsi_overbought_threshold": trial.suggest_float(
                    "breakout_rsi_overbought_threshold", *BREAKOUT_RSI_OVERBOUGHT_RANGE
                ),
                "breakout_relative_strength_min": trial.suggest_float(
                    "breakout_relative_strength_min", *BREAKOUT_RELATIVE_STRENGTH_RANGE
                ),
                "breakout_volume_ratio_min": trial.suggest_float(
                    "breakout_volume_ratio_min", *BREAKOUT_VOLUME_RATIO_RANGE
                ),
                "breakout_adx_min": trial.suggest_float("breakout_adx_min", *BREAKOUT_ADX_MIN_RANGE),
                "breakout_obv_zscore_min": trial.suggest_float(
                    "breakout_obv_zscore_min", *BREAKOUT_OBV_ZSCORE_MIN_RANGE
                ),
                "breakout_squeeze_zscore_max": trial.suggest_float(
                    "breakout_squeeze_zscore_max", *BREAKOUT_SQUEEZE_ZSCORE_MAX_RANGE
                ),
            }
        candidate = swingtrade.TradingConfig(**{
            **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy, **params,
        })

        fold_results = swingtrade.run_walk_forward(
            ticker_data, market_data, folds, candidate, sector_lookup=sector_lookup, strategy=strategy,
            max_workers=max_workers,
        )
        metrics = summarize_weighted(fold_results, end, half_life_days)
        trial.set_user_attr("metrics", metrics)

        under_sampled = metrics["effective_trade_count"] < MIN_TRADES_FOR_SCORE or metrics["sharpe_like"] is None

        if not multi_objective:
            return UNDER_SAMPLED_PENALTY if under_sampled else metrics["sharpe_like"]

        max_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(fold_results))
        trial.set_user_attr("max_drawdown", max_dd)
        if under_sampled:
            return UNDER_SAMPLED_PENALTY, UNDER_SAMPLED_DRAWDOWN_PENALTY
        return metrics["sharpe_like"], (max_dd if max_dd is not None else UNDER_SAMPLED_DRAWDOWN_PENALTY)

    return objective


def report_live_outcomes_context() -> None:
    """Informational only -- see module docstring for why live outcomes are
    kept out of the per-trial score itself (recency-weighting is the
    mechanism that lets the score respond to what's working lately). Splits
    out confirmed fills (see confirm_fill.py) from every mechanical signal's
    hypothetical outcome, AND splits actionable (Strong Buy/Buy) from
    research (Watch) tier (see storage/signals.py) -- most logged signals
    were never actually traded, and the research tier specifically was
    never meant to be, so the pooled-everything number overstates both the
    sample of real trades and the sample of signals you'd actually act on."""
    try:
        db = storage.get_db()
    except storage.MongoNotConfigured:
        return
    docs = list(db[storage.outcomes.COLLECTION_NAME].find({}))
    if not docs:
        print("Live Trade_Outcomes so far: 0 (too early to factor into scoring).")
        return
    as_trades = lambda ds: [{"status": d["status"], "pnl_pct": d["pnl_pct"]} for d in ds]  # noqa: E731
    print(f"Live Trade_Outcomes so far (every signal, all tiers): {len(docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(docs))}")

    actionable_docs = [d for d in docs if d.get("tier", "actionable") == "actionable"]
    research_docs = [d for d in docs if d.get("tier") == "research"]
    print(f"  Actionable tier (Strong Buy/Buy) only: {len(actionable_docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(actionable_docs))}")

    confirmed_docs = [d for d in actionable_docs if d.get("confirmed_filled")]
    if confirmed_docs:
        print(f"    ...of which CONFIRMED real fills: {len(confirmed_docs)} -- "
              f"pooled metrics: {swingtrade.summarize_trades(as_trades(confirmed_docs))}")
    else:
        print("    ...of which CONFIRMED real fills: 0 -- see confirm_fill.py.")

    print(f"  Research tier (Watch, never traded/tradeable) only: {len(research_docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(research_docs))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["rsi", "breakout"], default="rsi",
                         help="Which signal to search parameters for. Default: rsi.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 1y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed, for reproducible searches.")
    parser.add_argument(
        "--recency-half-life-days", type=float, default=180.0,
        help="Out-of-sample folds this many days before --end count half as much as the most "
             "recent fold (exponential decay). 0 disables weighting (uniform pooling).",
    )
    parser.add_argument(
        "--holdout-frac", type=float, default=0.25,
        help="Fraction of tickers (stratified by sector) held out from tuning entirely, "
             "then used once at the end to validate the winning config on names Optuna "
             "never saw. 0 disables (tune and validate on every ticker, old behavior).",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=42,
        help="Seed for the ticker tune/holdout split, so the same split is reproducible run to run.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
        help="CPU cores used to parallelize walk-forward folds (see swingtrade.run_walk_forward). "
             "Defaults to cpu_count()-4 (leaves headroom for the OS/other apps and keeps sustained "
             "thermal load down) rather than using every core -- pass a higher number (or your full "
             "core count) for maximum speed, or 1 to fall back to the old sequential behavior.",
    )
    parser.add_argument(
        "--multi-objective", action="store_true",
        help="Also optimize for max_drawdown (see swingtrade.compute_max_drawdown), not just "
             "sharpe_like -- finds the Pareto front (configs where no trial is both a higher "
             "sharpe_like AND a lower drawdown) instead of a single scalar winner, then picks "
             "the highest-sharpe_like trial on that front (still clearing MIN_TRADES_FOR_SCORE) "
             "to actually write as a candidate. Off by default -- the original single-objective "
             "search is unchanged unless you pass this.",
    )
    args = parser.parse_args()

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Fetching history for {len(tickers)} ticker(s) + {MARKET_INDEX_TICKER}, "
          f"{start.date()}..{end.date()} (once, reused across all {args.trials} trials)...")
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

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"Generated {len(folds)} walk-forward fold(s).")
    if not folds:
        print("[ERROR] Date range too short for even one fold.", file=sys.stderr)
        sys.exit(1)

    tune_tickers, holdout_tickers = split_tickers_holdout(
        list(ticker_data.keys()), sector_lookup, args.holdout_frac, args.holdout_seed
    )
    tune_ticker_data = {t: ticker_data[t] for t in tune_tickers}
    holdout_ticker_data = {t: ticker_data[t] for t in holdout_tickers}
    if holdout_tickers:
        print(f"Ticker holdout split (seed={args.holdout_seed}, frac={args.holdout_frac}): "
              f"{len(tune_tickers)} tune / {len(holdout_tickers)} holdout. "
              f"Holdout tickers never seen during tuning: {sorted(holdout_tickers)}")
    else:
        print("Ticker holdout validation disabled (--holdout-frac 0 or too few tickers) -- "
              "tuning and reporting on every ticker, old behavior.")

    print(f"Using up to {args.max_workers} CPU core(s) for walk-forward parallelization "
          f"(--max-workers to change; {os.cpu_count()} available).")

    baseline_config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": args.strategy})
    baseline_results = swingtrade.run_walk_forward(
        tune_ticker_data, market_data, folds, baseline_config, sector_lookup=sector_lookup, strategy=args.strategy,
        max_workers=args.max_workers,
    )
    baseline_metrics = summarize_weighted(baseline_results, end, args.recency_half_life_days)
    weight_note = (
        f"(recency-weighted, half-life={args.recency_half_life_days:.0f}d)"
        if args.recency_half_life_days > 0 else "(uniform pooling)"
    )
    print(f"\nBaseline (DEFAULT_CONFIG, strategy={args.strategy}) pooled out-of-sample metrics on TUNE tickers {weight_note}: {baseline_metrics}")
    if args.multi_objective:
        baseline_drawdown = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(baseline_results))
        print(f"  Baseline max_drawdown: {baseline_drawdown}%")
    report_live_outcomes_context()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.multi_objective:
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        build_objective(
            tune_ticker_data, market_data, folds, end, args.recency_half_life_days, sector_lookup, args.strategy,
            max_workers=args.max_workers, multi_objective=args.multi_objective,
        ),
        n_trials=args.trials, show_progress_bar=False,
    )

    if args.multi_objective:
        pareto = study.best_trials
        print()
        print(f"Pareto front: {len(pareto)} non-dominated trial(s) (sharpe_like, max_drawdown%)")
        for t in sorted(pareto, key=lambda t: -t.values[0]):
            print(f"  Trial #{t.number}: sharpe_like={t.values[0]:.4f}, max_drawdown={t.values[1]:.2f}%  params={t.params}")

        eligible = [t for t in pareto if t.values[0] > UNDER_SAMPLED_PENALTY]
        if not eligible:
            print()
            print("[WARN] No Pareto-optimal trial cleared the under-sampled floor -- not writing "
                  "a candidate. Try a wider date range, more tickers, or a longer in-sample window.")
            return
        # Selection heuristic, stated plainly: highest sharpe_like among the
        # Pareto-optimal trials (i.e. among configs where no OTHER trial beat
        # them on both axes simultaneously) -- keeps the existing "one
        # candidate gets written" champion/challenger flow intact rather than
        # asking a human to pick blindly, while the full front (printed
        # above) still shows what was traded off to get there.
        best = max(eligible, key=lambda t: t.values[0])
        best_metrics = best.user_attrs.get("metrics", {})
        best_drawdown = best.user_attrs.get("max_drawdown")
        print()
        print(f"Selected from Pareto front, trial #{best.number}: sharpe_like={best.values[0]:.4f}, max_drawdown={best.values[1]:.2f}%")
        print(f"  params: {best.params}")
        print(f"  metrics (TUNE tickers): {best_metrics}")
    else:
        best = study.best_trial
        best_metrics = best.user_attrs.get("metrics", {})
        best_drawdown = None
        print()
        print(f"Best trial #{best.number}: score(sharpe_like or penalty)={best.value:.4f}")
        print(f"  params: {best.params}")
        print(f"  metrics (TUNE tickers): {best_metrics}")

        if best.value <= UNDER_SAMPLED_PENALTY:
            print()
            print("[WARN] Even the best trial was under-sampled or had no valid sharpe_like -- "
                  "not writing a candidate. Try a wider date range, more tickers, or a longer "
                  "in-sample window.")
            return

    candidate_config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": args.strategy, **best.params,
    })

    holdout_metrics = {}
    if holdout_ticker_data:
        holdout_baseline_results = swingtrade.run_walk_forward(
            holdout_ticker_data, market_data, folds, baseline_config, sector_lookup=sector_lookup,
            strategy=args.strategy, max_workers=args.max_workers,
        )
        holdout_baseline_metrics = summarize_weighted(holdout_baseline_results, end, args.recency_half_life_days)

        holdout_candidate_results = swingtrade.run_walk_forward(
            holdout_ticker_data, market_data, folds, candidate_config, sector_lookup=sector_lookup,
            strategy=args.strategy, max_workers=args.max_workers,
        )
        holdout_candidate_metrics = summarize_weighted(holdout_candidate_results, end, args.recency_half_life_days)

        print()
        print(f"=== Ticker-universe holdout validation ({len(holdout_tickers)} tickers Optuna never saw) ===")
        print(f"  baseline (DEFAULT_CONFIG, strategy={args.strategy}) on holdout: {holdout_baseline_metrics}")
        print(f"  candidate (winning trial) on holdout: {holdout_candidate_metrics}")
        if args.multi_objective:
            holdout_baseline_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(holdout_baseline_results))
            holdout_candidate_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(holdout_candidate_results))
            print(f"  baseline max_drawdown on holdout: {holdout_baseline_dd}%")
            print(f"  candidate max_drawdown on holdout: {holdout_candidate_dd}%")
        if (holdout_candidate_metrics.get("effective_trade_count") or 0) < MIN_TRADES_FOR_SCORE:
            print("  [WARN] Holdout effective_trade_count is thin -- treat this validation as a weak "
                  "signal, not proof either way. Consider a lower --holdout-frac or more tickers.")

        holdout_metrics = {
            "holdout_tickers": sorted(holdout_tickers),
            "tune_tickers": sorted(tune_tickers),
            "holdout_frac": args.holdout_frac,
            "holdout_seed": args.holdout_seed,
            "baseline": holdout_baseline_metrics,
            "candidate": holdout_candidate_metrics,
        }
        if args.multi_objective:
            holdout_metrics["baseline_max_drawdown"] = holdout_baseline_dd
            holdout_metrics["candidate_max_drawdown"] = holdout_candidate_dd

    best_sharpe = best.values[0] if args.multi_objective else best.value
    notes = (
        f"Optuna search (strategy={args.strategy}{'  multi-objective' if args.multi_objective else ''}): "
        f"{args.trials} trials, {start.date()}..{end.date()}, "
        f"{len(tune_ticker_data)} tune / {len(holdout_ticker_data)} holdout ticker(s), {len(folds)} fold(s), "
        f"recency_half_life_days={args.recency_half_life_days:.0f}. "
        f"Baseline sharpe_like={baseline_metrics.get('sharpe_like')}, best sharpe_like={best_sharpe:.4f}."
    )
    if args.multi_objective:
        notes += f" Selected trial max_drawdown={best_drawdown}% (Pareto front had {len(pareto)} trial(s))."
    if holdout_metrics:
        notes += (
            f" Holdout ({len(holdout_tickers)} tickers): baseline sharpe_like="
            f"{holdout_metrics['baseline'].get('sharpe_like')}, candidate sharpe_like="
            f"{holdout_metrics['candidate'].get('sharpe_like')}."
        )
    version = storage.write_candidate(
        candidate_config.to_dict(), notes=notes, metrics=best_metrics, holdout_metrics=holdout_metrics
    )

    print()
    print(f"Wrote candidate System_Config version={version} (status=candidate, NOT active).")
    print(f"Review it, then run: python promote_config.py --promote {version}")


if __name__ == "__main__":
    main()
