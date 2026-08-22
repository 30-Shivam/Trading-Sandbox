"""
Periodic re-validation driver for the "Best Ideas" tab (improvements.txt
items 74-76) -- runs both checks and posts one combined report via Discord.
Meant to run on a MONTHLY schedule (see .github/workflows/monthly_reoptimize.yml),
never daily -- this is a much heavier job than ingest.py's own daily scan
(full 5-year re-simulation across 2 strategies x 5 sector-RS lookback
candidates, plus a real-vs-random HOLDOUT re-check for each strategy).

PROPOSE ONLY, same hard rule as every prior build touching Best Ideas or
System_Config this session: this script NEVER writes to MongoDB's
System_Config collection, NEVER calls promote_config.py, NEVER edits
best_ideas.py's own constants. It produces a report; a human decides what
to do with it -- exactly the same champion/challenger discipline
promote_config.py's own docstring describes ("a noisy optimization run
should not be able to silently degrade real trading"), just applied to an
unattended monthly job instead of an interactively-run one.

Check A (see reoptimize_sector_rs.py): does a different
sector_relative_strength_lookback_days show a meaningfully higher
Information Coefficient against real historical trades than the current
63-day default? The only Best Ideas methodology that's genuinely
backtestable (best_ideas_qualitative/best_ideas_meta structurally aren't --
no point-in-time archived data, same limit as llm_agent.py).

Check B (this file): have ma_crossover (v55, primary), squeeze_breakout
(v39), RSI mean-reversion (v17), or mean-reversion pairs (v58) -- the live
mechanical strategies Best Ideas actually draws from (see
LIVE_STRATEGIES_FOR_STALENESS_CHECK) -- decayed since their own last real
validation? Reuses optimize.average_holdout_summary() (item 69's
multi-seed methodology) for a real-vs-random HOLDOUT comparison, compared
against a stored baseline (best_ideas_baseline_metrics.json) -- the exact
kind of drift that was found and acted on twice in one day already this
session (squeeze_breakout, improvements.txt item 62; breakout's full
retirement, item 73).

Usage:
    python reoptimize_best_ideas.py
    python reoptimize_best_ideas.py --tickers NVDA,AMD,XOM --lookback-candidates 21,63,126
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import notifications
import reoptimize_sector_rs
import storage
import swingtrade
from optimize import DEFAULT_HOLDOUT_SEEDS, average_holdout_summary
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
BASELINE_FILE = SCRIPT_DIR / "best_ideas_baseline_metrics.json"
REQUEST_DELAY_SEC = 0.5

# A HOLDOUT sharpe_like drop of at least this much (absolute, current minus
# baseline) is flagged -- same order of magnitude as the real decays this
# project has already found and acted on (v39: item 62; breakout: item 73),
# well above the seed-to-seed noise floor item 69's own investigation
# measured for a single strategy's HOLDOUT sharpe_like.
STALENESS_SHARPE_DROP_THRESHOLD = 0.03

# (strategy, config_version, real_fn_name, random_fn_name) -- every live
# mechanical strategy best_ideas.METHODOLOGIES actually draws from
# (breakout/v43 was retired, item 73, and was never in that list anyway).
# Pinned config_version numbers must stay in sync with
# config_loader.SECONDARY_STRATEGY_VERSIONS (the live-dashboard source of
# truth for which candidate version each secondary strategy runs) --
# ma_crossover alone uses None/active since it's the PRIMARY slot, not a
# pinned secondary.
LIVE_STRATEGIES_FOR_STALENESS_CHECK = [
    ("ma_crossover", None, "simulate_ma_crossover_signals", "simulate_random_ma_crossover_entries"),
    ("squeeze_breakout", 39, "simulate_squeeze_breakout_signals", "simulate_random_squeeze_breakout_entries"),
    ("rsi_mean_reversion", 17, "simulate_signals", "simulate_random_entries"),
    ("pairs", 58, "simulate_pairs_signals", "simulate_random_pairs_entries"),
]


def load_baseline() -> dict:
    """{} if the baseline file doesn't exist yet (first run) -- Check B
    degrades to "no baseline to compare against, nothing to flag" rather
    than erroring, same "missing data doesn't crash" convention this
    codebase uses everywhere."""
    if not BASELINE_FILE.exists():
        return {}
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def flag_staleness(
    strategy: str, fresh_holdout: dict, baseline: dict, threshold: float = STALENESS_SHARPE_DROP_THRESHOLD,
) -> dict | None:
    """Pure comparison logic, directly unit-testable: `fresh_holdout` is one
    average_holdout_summary() bucket dict (has "sharpe_like"), `baseline` is
    that strategy's own entry from best_ideas_baseline_metrics.json (or {}
    if there is none yet). Flags staleness when the fresh HOLDOUT sharpe_like
    has dropped by >= `threshold` versus the stored baseline, OR when the
    fresh sharpe_like is not None but the strategy's own baseline value is
    missing/None (can't compare, but a defined-vs-undefined flip is itself
    worth a human look). Returns None when there's nothing to compare
    against (no baseline recorded yet) or no meaningful drop."""
    fresh_sharpe = fresh_holdout.get("sharpe_like")
    baseline_sharpe = baseline.get("holdout_sharpe_like") if baseline else None
    if not baseline:
        return None
    if fresh_sharpe is None:
        return {"strategy": strategy, "reason": "fresh HOLDOUT sharpe_like undefined (too thin a sample)",
                "fresh_sharpe": None, "baseline_sharpe": baseline_sharpe}
    if baseline_sharpe is None:
        return None
    drop = baseline_sharpe - fresh_sharpe
    if drop >= threshold:
        return {"strategy": strategy, "reason": f"HOLDOUT sharpe_like dropped {drop:.4f} vs baseline",
                "fresh_sharpe": fresh_sharpe, "baseline_sharpe": baseline_sharpe}
    return None


def run_staleness_check(
    strategy: str, config_version: int | None, real_fn_name: str, random_fn_name: str,
    ticker_data: dict, market_data: pd.DataFrame, sector_lookup: dict, start, end,
    pair_price_panels: dict[str, pd.DataFrame] | None = None, log=print,
) -> dict:
    """Real-vs-random HOLDOUT re-check for one live mechanical strategy,
    reusing the exact optimize.average_holdout_summary() multi-seed
    methodology (item 69) benchmark_random_entry.py itself uses -- so this
    stays honestly comparable to every other HOLDOUT number this project
    has reported this session.

    `pair_price_panels` (one wide Close-price panel per sector, see
    main()'s own construction) is only consulted for strategy="pairs" --
    without it, simulate_pairs_signals() silently degrades to "Pair_Signal
    always False" (see its own docstring), which would make this check
    report an undefined HOLDOUT sharpe_like and falsely flag pairs as
    stale every single run."""
    if config_version is not None:
        config, label = config_loader.load_config_by_version(config_version)
        if config is None:
            raise RuntimeError(f"could not load {strategy} v{config_version}: {label}")
    else:
        config, label = config_loader.load_active_config()
        if config.strategy != strategy:
            raise RuntimeError(f"active primary config's strategy is {config.strategy!r}, expected {strategy!r}")
        # load_active_config() doesn't return the version number itself --
        # resolve it separately so the baseline file records a real version,
        # not `null`, for whichever strategy currently holds the primary slot.
        active_doc = storage.get_active_config_doc()
        config_version = active_doc["version"] if active_doc else None
    log(f"  {strategy}: {label}")

    real_fn = getattr(swingtrade, real_fn_name)
    random_fn = getattr(swingtrade, random_fn_name)
    rng = random.Random(1)
    real_trades, random_trades = [], []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        peer_kwargs = {}
        if strategy == "pairs" and pair_price_panels is not None:
            panel = pair_price_panels.get(sector)
            if panel is not None and ticker in panel.columns:
                peer_kwargs = {"peer_prices": panel.drop(columns=[ticker])}
        real = real_fn(ticker, ohlcv, market_data, start, end, config, sector=sector, **peer_kwargs)
        real_trades.extend(real)
        random_trades.extend(
            random_fn(ticker, ohlcv, market_data, start, end, len(real), rng, config, sector=sector)
        )

    def summarize(trades):
        resolved = [t for t in trades if t["status"] != "OPEN"]
        weights = swingtrade.compute_cluster_weights(resolved)
        return swingtrade.summarize_trades_weighted(resolved, weights)

    _, real_holdout = average_holdout_summary(real_trades, sector_lookup, 0.25, list(DEFAULT_HOLDOUT_SEEDS), summarize)
    _, random_holdout = average_holdout_summary(random_trades, sector_lookup, 0.25, list(DEFAULT_HOLDOUT_SEEDS), summarize)
    return {
        "strategy": strategy, "config_version": config_version, "label": label,
        "real_holdout": real_holdout, "random_holdout": random_holdout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument(
        "--lookback-candidates", default=",".join(str(x) for x in reoptimize_sector_rs.DEFAULT_LOOKBACK_CANDIDATES),
        help="Comma-separated sector-RS lookback windows to compare (Check A).",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="After reporting, overwrite best_ideas_baseline_metrics.json with today's fresh HOLDOUT numbers -- "
             "use once you've reviewed a report and want future runs to compare against it. Off by default so a "
             "routine run never silently moves the goalposts.",
    )
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 5)
    lookback_candidates = [int(x.strip()) for x in args.lookback_candidates.split(",") if x.strip()]

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    report_lines: list[str] = []

    def log(msg):
        print(msg)
        report_lines.append(str(msg))

    log(f"=== Best Ideas periodic re-validation -- {end.date()} ===")
    log(f"Universe: {len(tickers)} ticker(s), window {start.date()}..{end.date()}\n")

    log("--- Check A: sector_rs lookback re-tune ---")
    try:
        sector_rs_results = reoptimize_sector_rs.compute_sector_rs_ic_table(tickers, lookback_candidates, start, end, log=log)
    except RuntimeError as exc:
        log(f"[ERROR] Check A failed: {exc}")
        sector_rs_results = []
    if sector_rs_results:
        log(f"\n{'lookback_days':>14} | {'n':>6} | {'ic':>8}")
        for r in sector_rs_results:
            marker = "  <-- CURRENT" if r["lookback_days"] == reoptimize_sector_rs.CURRENT_LOOKBACK else ""
            ic_str = f"{r['ic']:.4f}" if r["ic"] is not None else "None"
            log(f"{r['lookback_days']:>14} | {r['n']:>6} | {ic_str:>8}{marker}")
        rec = reoptimize_sector_rs.recommend_lookback(sector_rs_results)
        if rec is None:
            log(f"No sector-RS lookback change recommended (margin required: "
                f"{reoptimize_sector_rs.MEANINGFUL_IC_MARGIN} IC).")
        else:
            log(f"RECOMMENDATION: sector_relative_strength_lookback_days -> {rec['lookback_days']} "
                f"(IC {rec['ic']:.4f} vs current {rec['current_ic']}). NOT applied automatically.")

    log("\n--- Check B: mechanical-strategy staleness re-check ---")
    print(f"\nFetching {MARKET_INDEX_TICKER}...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
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
    log(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s) for staleness check.")

    # One wide Close-price panel per sector (>= 2 members), for the "pairs"
    # strategy's partner-selection mechanism -- no new fetch, built from
    # ticker_data already in hand. Same construction benchmark_random_entry.py
    # uses (items 82/85); unused by every other strategy in the loop below.
    pair_price_panels: dict[str, pd.DataFrame] = {}
    by_sector: dict[str, list[str]] = {}
    for ticker in ticker_data:
        by_sector.setdefault(sector_lookup.get(ticker, "Unknown"), []).append(ticker)
    for sector, members in by_sector.items():
        if len(members) < 2:
            continue
        pair_price_panels[sector] = pd.DataFrame({m: ticker_data[m]["Close"] for m in members})

    baseline = load_baseline()
    new_baseline = dict(baseline)
    staleness_flags = []
    for strategy, config_version, real_fn_name, random_fn_name in LIVE_STRATEGIES_FOR_STALENESS_CHECK:
        try:
            check = run_staleness_check(
                strategy, config_version, real_fn_name, random_fn_name,
                ticker_data, market_data, sector_lookup, start, end,
                pair_price_panels=pair_price_panels, log=log,
            )
        except RuntimeError as exc:
            log(f"  [WARN] {strategy}: staleness check skipped -- {exc}")
            continue
        real_holdout = check["real_holdout"]
        log(f"    HOLDOUT (avg of {len(DEFAULT_HOLDOUT_SEEDS)} seeds): "
            f"REAL sharpe_like={real_holdout.get('sharpe_like')}, "
            f"win_rate={real_holdout.get('win_rate')}, "
            f"RANDOM sharpe_like={check['random_holdout'].get('sharpe_like')}")
        flag = flag_staleness(strategy, real_holdout, baseline.get(strategy, {}))
        if flag is not None:
            staleness_flags.append(flag)
            log(f"    FLAGGED: {flag['reason']}")
        new_baseline[strategy] = {
            "config_version": check["config_version"], "date": str(end.date()),
            "holdout_sharpe_like": real_holdout.get("sharpe_like"),
            "holdout_win_rate": real_holdout.get("win_rate"),
            "holdout_annualized_sharpe_like": real_holdout.get("annualized_sharpe_like"),
        }

    if staleness_flags:
        log(f"\n{len(staleness_flags)} strategy(ies) flagged for possible drift -- worth a manual "
            "re-investigation (same process as improvements.txt items 62/73), NOT auto-acted on.")
    else:
        log("\nNo mechanical strategy flagged for staleness this run.")

    if args.update_baseline:
        BASELINE_FILE.write_text(json.dumps(new_baseline, indent=2), encoding="utf-8")
        log(f"\nBaseline updated: {BASELINE_FILE}")
    else:
        log(f"\n(--update-baseline not passed -- {BASELINE_FILE.name} left unchanged.)")

    report = "\n".join(report_lines)
    summary = f"Best Ideas re-validation ({end.date()}): "
    summary += f"{len(staleness_flags)} staleness flag(s), "
    rec = reoptimize_sector_rs.recommend_lookback(sector_rs_results) if sector_rs_results else None
    summary += "1 sector-RS lookback recommendation" if rec else "no sector-RS change recommended"
    webhook_url = notifications.get_strategy_webhook_url("best_ideas_reoptimize")
    notifications.notify_with_file(summary, "best_ideas_revalidation_report.txt", report, webhook_url=webhook_url)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
