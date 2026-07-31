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
without corrupting the cross-candidate comparison. Weighting uses Kish's
effective sample size (not raw trade_count) for the MIN_TRADES_FOR_SCORE
check, so a large pool of old, near-zero-weight trades can't fake
statistical significance. Pass --recency-half-life-days 0 to fall back to
old-style uniform pooling.

Live Trade_Outcomes are still reported separately for context (how many
exist, their aggregate performance) -- useful as a sanity check against what
the backtest predicts, but not an input to the score.

Usage:
    python optimize.py --trials 50 --start 2023-01-01 --end 2026-07-01
    python optimize.py --trials 50 --recency-half-life-days 90   # weight recent regime more heavily
"""

import argparse
import sys
import time
from pathlib import Path

import optuna
import pandas as pd

import storage
import swingtrade
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_history
from watchlist import read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

MIN_TRADES_FOR_SCORE = 15    # penalize candidates with too few pooled trades to trust
UNDER_SAMPLED_PENALTY = -10.0  # well below any realistic sharpe_like, but finite (no inf/NaN into Optuna)

# Search space bounds for the tunables this study is allowed to move.
# A prior run pinned rsi_oversold_threshold at 54.9/55 and stop_loss_atr_multiplier
# at 0.53/0.5 -- both right against their old boundaries, which usually means the
# bound was cutting off the true optimum rather than the search finding an interior
# sweet spot. Widened accordingly; atr_take_profit_multiplier landed interior
# (2.13 of 1.0-3.0) so it's untouched.
RSI_OVERSOLD_RANGE = (15.0, 70.0)
ATR_TAKE_PROFIT_RANGE = (1.0, 3.0)
STOP_LOSS_ATR_RANGE = (0.25, 3.0)
EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE = (0.0, 4.0)
EXTENDED_DECLINE_PENALTY_CAP_RANGE = (0.0, 50.0)

# slippage_pct / commission_pct_per_trade are deliberately NEVER in this search
# space: they model execution friction, not strategy behavior. Letting Optuna
# tune them would just teach it to zero out the very realism they exist to add.


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
    weighted by fold recency instead of uniformly -- the current market
    regime counts for more than one from a year ago. Uses Kish's effective
    sample size (not raw trade_count) so heavy weighting can't let a big
    pile of old, near-zero-weight trades masquerade as a trustworthy
    sample."""
    pnls, weights, win_weight, total_weight = [], [], 0.0, 0.0
    raw_count = 0
    for fr in fold_results:
        w = fold_weight(fr.fold, end, half_life_days)
        for t in fr.out_sample_trades:
            if t["status"] == "OPEN":
                continue
            raw_count += 1
            pnls.append(t["pnl_pct"])
            weights.append(w)
            total_weight += w
            if t["status"] == "WIN":
                win_weight += w

    if not pnls or total_weight <= 0:
        return {
            "trade_count": 0, "effective_trade_count": 0.0,
            "win_rate": None, "avg_pnl_pct": None, "pnl_std": None, "sharpe_like": None,
        }

    pnl_s = pd.Series(pnls)
    w_s = pd.Series(weights)
    weighted_mean = float((pnl_s * w_s).sum() / total_weight)
    weighted_var = float((w_s * (pnl_s - weighted_mean) ** 2).sum() / total_weight)
    weighted_std = weighted_var ** 0.5
    effective_n = float(total_weight ** 2 / (w_s ** 2).sum())  # Kish effective sample size

    return {
        "trade_count": raw_count,
        "effective_trade_count": round(effective_n, 1),
        "win_rate": round(win_weight / total_weight * 100, 2),
        "avg_pnl_pct": round(weighted_mean, 2),
        "pnl_std": round(weighted_std, 2),
        "sharpe_like": round(weighted_mean / weighted_std, 3) if weighted_std > 0 else None,
    }


def build_objective(ticker_data: dict, market_data: pd.DataFrame, folds: list, end: pd.Timestamp, half_life_days: float):
    def objective(trial: optuna.Trial) -> float:
        candidate = swingtrade.TradingConfig(**{
            **swingtrade.DEFAULT_CONFIG.to_dict(),
            "rsi_oversold_threshold": trial.suggest_float("rsi_oversold_threshold", *RSI_OVERSOLD_RANGE),
            "atr_take_profit_multiplier": trial.suggest_float("atr_take_profit_multiplier", *ATR_TAKE_PROFIT_RANGE),
            "stop_loss_atr_multiplier": trial.suggest_float("stop_loss_atr_multiplier", *STOP_LOSS_ATR_RANGE),
            "extended_decline_penalty_per_day": trial.suggest_float(
                "extended_decline_penalty_per_day", *EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE
            ),
            "extended_decline_penalty_cap": trial.suggest_float(
                "extended_decline_penalty_cap", *EXTENDED_DECLINE_PENALTY_CAP_RANGE
            ),
        })

        fold_results = swingtrade.run_walk_forward(ticker_data, market_data, folds, candidate)
        metrics = summarize_weighted(fold_results, end, half_life_days)
        trial.set_user_attr("metrics", metrics)

        if metrics["effective_trade_count"] < MIN_TRADES_FOR_SCORE or metrics["sharpe_like"] is None:
            return UNDER_SAMPLED_PENALTY
        return metrics["sharpe_like"]

    return objective


def report_live_outcomes_context() -> None:
    """Informational only -- see module docstring for why live outcomes are
    kept out of the per-trial score itself (recency-weighting is the
    mechanism that lets the score respond to what's working lately). Splits
    out confirmed fills (see confirm_fill.py) from every mechanical signal's
    hypothetical outcome -- most logged signals were never actually traded,
    so the pooled-everything number overstates the sample of real trades."""
    try:
        db = storage.get_db()
    except storage.MongoNotConfigured:
        return
    docs = list(db[storage.outcomes.COLLECTION_NAME].find({}))
    if not docs:
        print("Live Trade_Outcomes so far: 0 (too early to factor into scoring).")
        return
    trades = [{"status": d["status"], "pnl_pct": d["pnl_pct"]} for d in docs]
    metrics = swingtrade.summarize_trades(trades)
    print(f"Live Trade_Outcomes so far (every signal): {len(docs)} -- pooled metrics: {metrics}")

    confirmed_docs = [d for d in docs if d.get("confirmed_filled")]
    if confirmed_docs:
        confirmed_trades = [{"status": d["status"], "pnl_pct": d["pnl_pct"]} for d in confirmed_docs]
        confirmed_metrics = swingtrade.summarize_trades(confirmed_trades)
        print(f"  ...of which CONFIRMED real fills: {len(confirmed_docs)} -- pooled metrics: {confirmed_metrics}")
    else:
        print("  ...of which CONFIRMED real fills: 0 -- see confirm_fill.py.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    baseline_results = swingtrade.run_walk_forward(ticker_data, market_data, folds, swingtrade.DEFAULT_CONFIG)
    baseline_metrics = summarize_weighted(baseline_results, end, args.recency_half_life_days)
    weight_note = (
        f"(recency-weighted, half-life={args.recency_half_life_days:.0f}d)"
        if args.recency_half_life_days > 0 else "(uniform pooling)"
    )
    print(f"\nBaseline (current DEFAULT_CONFIG) pooled out-of-sample metrics {weight_note}: {baseline_metrics}")
    report_live_outcomes_context()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        build_objective(ticker_data, market_data, folds, end, args.recency_half_life_days),
        n_trials=args.trials, show_progress_bar=False,
    )

    best = study.best_trial
    best_metrics = best.user_attrs.get("metrics", {})
    print()
    print(f"Best trial #{best.number}: score(sharpe_like or penalty)={best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  metrics: {best_metrics}")

    if best.value <= UNDER_SAMPLED_PENALTY:
        print()
        print("[WARN] Even the best trial was under-sampled or had no valid sharpe_like -- "
              "not writing a candidate. Try a wider date range, more tickers, or a longer "
              "in-sample window.")
        return

    candidate_config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), **best.params})
    notes = (
        f"Optuna search: {args.trials} trials, {start.date()}..{end.date()}, "
        f"{len(ticker_data)} ticker(s), {len(folds)} fold(s), "
        f"recency_half_life_days={args.recency_half_life_days:.0f}. "
        f"Baseline sharpe_like={baseline_metrics.get('sharpe_like')}, best sharpe_like={best.value:.4f}."
    )
    version = storage.write_candidate(candidate_config.to_dict(), notes=notes, metrics=best_metrics)

    print()
    print(f"Wrote candidate System_Config version={version} (status=candidate, NOT active).")
    print(f"Review it, then run: python promote_config.py --promote {version}")


if __name__ == "__main__":
    main()
