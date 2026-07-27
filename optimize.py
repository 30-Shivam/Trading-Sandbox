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

Live Trade_Outcomes are reported for context (how many exist, their
aggregate performance) but are NOT mixed into the per-trial score: live
outcomes were generated under one specific historical config, not the
trial's varying candidate config, so blending them into a comparison across
candidates would conflate two different questions. As Trade_Outcomes
accumulates, a future revision can incorporate it more directly.

Usage:
    python optimize.py --trials 50 --start 2023-01-01 --end 2026-07-01
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


def build_objective(ticker_data: dict, market_data: pd.DataFrame, folds: list):
    def objective(trial: optuna.Trial) -> float:
        candidate = swingtrade.TradingConfig(**{
            **swingtrade.DEFAULT_CONFIG.to_dict(),
            "rsi_oversold_threshold": trial.suggest_float("rsi_oversold_threshold", *RSI_OVERSOLD_RANGE),
            "atr_take_profit_multiplier": trial.suggest_float("atr_take_profit_multiplier", *ATR_TAKE_PROFIT_RANGE),
            "stop_loss_atr_multiplier": trial.suggest_float("stop_loss_atr_multiplier", *STOP_LOSS_ATR_RANGE),
        })

        fold_results = swingtrade.run_walk_forward(ticker_data, market_data, folds, candidate)
        pooled_oos_trades = [t for fr in fold_results for t in fr.out_sample_trades]
        metrics = swingtrade.summarize_trades(pooled_oos_trades)
        trial.set_user_attr("metrics", metrics)

        if metrics["trade_count"] < MIN_TRADES_FOR_SCORE or metrics["sharpe_like"] is None:
            return UNDER_SAMPLED_PENALTY
        return metrics["sharpe_like"]

    return objective


def report_live_outcomes_context() -> None:
    """Informational only -- see module docstring for why live outcomes
    aren't mixed into the per-trial score yet."""
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
    print(f"Live Trade_Outcomes so far: {len(docs)} -- pooled metrics: {metrics}")


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
    baseline_metrics = swingtrade.summarize_trades([t for fr in baseline_results for t in fr.out_sample_trades])
    print(f"\nBaseline (current DEFAULT_CONFIG) pooled out-of-sample metrics: {baseline_metrics}")
    report_live_outcomes_context()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(build_objective(ticker_data, market_data, folds), n_trials=args.trials, show_progress_bar=False)

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
        f"{len(ticker_data)} ticker(s), {len(folds)} fold(s). "
        f"Baseline sharpe_like={baseline_metrics.get('sharpe_like')}, best sharpe_like={best.value:.4f}."
    )
    version = storage.write_candidate(candidate_config.to_dict(), notes=notes, metrics=best_metrics)

    print()
    print(f"Wrote candidate System_Config version={version} (status=candidate, NOT active).")
    print(f"Review it, then run: python promote_config.py --promote {version}")


if __name__ == "__main__":
    main()
