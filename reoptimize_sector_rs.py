"""
Periodic re-check (Check A of the Best Ideas re-validation job, see
reoptimize_best_ideas.py) of best_ideas_sector_rs's lookback window --
`config.sector_relative_strength_lookback_days`, currently 63 trading days,
shared by both the backtest/Optuna-only *_sector_relative_strength_min
filters (items 68-71) and best_ideas.compute_sector_rs_scores()'s own live
momentum ranking (item 74).

Unlike best_ideas_qualitative/best_ideas_meta (structurally unbacktestable
-- no point-in-time archived data), sector_rs is pure historical price
action, so it's the one Best Ideas methodology that CAN be periodically
re-optimized against real history rather than relying solely on prospective
IC/IR.

Method: run the 2 live mechanical strategies that actually feed the Best
Ideas ensemble (ma_crossover, squeeze_breakout -- see best_ideas.METHODOLOGIES)
over the full 5-year watchlist window under their real live configs, same as
benchmark_sector_relative_strength.py. For each candidate lookback, compute
each trade's own sector's relative-strength score AS OF its signal_date
(swingtrade.levels.compute_relative_strength(), the exact primitive
best_ideas.compute_sector_rs_scores() calls live) and pool every
(score, realized pnl_pct) pair into ic_tracking.rank_ic() -- the SAME IC
math the live dashboard already uses to judge every other methodology, so
this re-tune stays honestly comparable to what's already being measured
prospectively.

PROPOSE ONLY: prints a comparison table and, if a candidate lookback beats
the current one by a real margin, a recommendation -- never writes to
System_Config, never edits swingtrade/config.py's own default. Changing
sector_relative_strength_lookback_days is a deliberate, separate, human
decision, same as every config change this project has ever made.

Usage:
    python reoptimize_sector_rs.py
    python reoptimize_sector_rs.py --tickers NVDA,AMD,XOM --lookback-candidates 21,63,126
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import ic_tracking
import swingtrade
from swingtrade.levels import compute_relative_strength
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import SECTOR_ETF, read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

# The 2 live mechanical strategies best_ideas.METHODOLOGIES actually draws
# from today (breakout/v43 was retired, see improvements.txt item 73 -- not
# included here since best_ideas.py itself never includes it either).
LIVE_STRATEGY_SIMULATORS = {
    "ma_crossover": swingtrade.simulate_ma_crossover_signals,
    "squeeze_breakout": swingtrade.simulate_squeeze_breakout_signals,
}

DEFAULT_LOOKBACK_CANDIDATES = [21, 42, 63, 84, 126]
CURRENT_LOOKBACK = swingtrade.DEFAULT_CONFIG.sector_relative_strength_lookback_days  # 63
MEANINGFUL_IC_MARGIN = 0.02  # a candidate must beat the current lookback's IC by at
                              # least this much to be worth flagging -- IC differences
                              # smaller than this are within the kind of noise this
                              # project has repeatedly found in similarly-sized samples
                              # this session (see improvements.txt item 69's own
                              # seed-sensitivity investigation)

MIN_ABSOLUTE_IC = 0.02  # 2026-09-03 fix (improvements.txt): recommend_lookback() used to
                         # compare candidates ONLY relative to each other -- no floor on
                         # whether the WINNING one was itself actually informative. Real
                         # incident this closes: a full-scale re-run (item 76) found 21d
                         # IC=-0.0028 (essentially zero, not a real edge) still "won" by
                         # beating 63d's more-negative IC by more than MEANINGFUL_IC_MARGIN
                         # -- the mechanical recommendation fired on evidence too thin to
                         # mean anything either way, a "least-bad of N weak options" trap.
                         # Reuses MEANINGFUL_IC_MARGIN's own value deliberately -- the same
                         # magnitude this project already uses as "a real difference" is
                         # equally reasonable as the floor for "actually informative at all".


def recommend_lookback(
    results: list[dict], current_lookback: int = CURRENT_LOOKBACK, margin: float = MEANINGFUL_IC_MARGIN,
    min_absolute_ic: float = MIN_ABSOLUTE_IC,
) -> dict | None:
    """Pure decision logic, factored out for direct unit testing:
    `results` is [{"lookback_days": int, "n": int, "ic": float|None}, ...]
    (one entry per candidate lookback, `ic` from ic_tracking.rank_ic() over
    that lookback's pooled (score, pnl_pct) pairs). Returns a recommendation
    dict {"lookback_days":, "ic":, "current_ic":, "margin":} only when a
    DIFFERENT lookback's IC beats the current one by at least `margin` --
    None otherwise (including when every candidate's IC is undefined, or
    the current lookback already IS the best one). Never recommends
    "switching" to the same lookback that's already active.

    Also requires the WINNING candidate's own IC to clear `min_absolute_ic`
    (see MIN_ABSOLUTE_IC's own docstring for the real incident this fixes)
    -- a candidate only being LESS BAD than an even weaker current lookback
    is not, on its own, a real basis to recommend anything. This is an
    absolute floor on top of (not a replacement for) the relative `margin`
    check above -- both must pass."""
    current = next((r for r in results if r["lookback_days"] == current_lookback), None)
    current_ic = current["ic"] if current and current["ic"] is not None else None
    best = max((r for r in results if r["ic"] is not None), key=lambda r: r["ic"], default=None)
    if best is None:
        return None
    if best["ic"] < min_absolute_ic:
        return None
    if current_ic is None:
        if best["lookback_days"] == current_lookback:
            return None
        return {"lookback_days": best["lookback_days"], "ic": best["ic"], "current_ic": None, "margin": margin}
    if best["lookback_days"] != current_lookback and (best["ic"] - current_ic) >= margin:
        return {"lookback_days": best["lookback_days"], "ic": best["ic"], "current_ic": current_ic, "margin": margin}
    return None


def load_live_configs() -> dict[str, tuple[swingtrade.TradingConfig, str]]:
    """The 2 real, currently-live mechanical configs Best Ideas draws from
    -- config_loader.py is the single source of truth ingest.py/best_ideas.py
    already use, so this can't silently drift from what's actually running.
    ma_crossover is PRIMARY now (item 73), not a secondary -- load via
    load_active_config(), not a hardcoded version."""
    ma_crossover_config, ma_label = config_loader.load_active_config()
    if ma_crossover_config.strategy != "ma_crossover":
        print(
            f"[WARN] Active primary config's strategy is {ma_crossover_config.strategy!r}, "
            "not 'ma_crossover' -- best_ideas.py's own METHODOLOGIES list assumes ma_crossover "
            "is live; this re-check's ma_crossover leg will be skipped.",
            file=sys.stderr,
        )
        ma_crossover_config = None
    squeeze_config, squeeze_reason = config_loader.load_config_by_version(
        config_loader.SECONDARY_STRATEGY_VERSIONS.get("Squeeze Breakout", 39)
    )
    if squeeze_config is None:
        print(f"[ERROR] Could not load squeeze_breakout: {squeeze_reason}", file=sys.stderr)
        sys.exit(1)
    configs = {"squeeze_breakout": (squeeze_config, "live secondary")}
    if ma_crossover_config is not None:
        configs["ma_crossover"] = (ma_crossover_config, ma_label)
    return configs


def compute_sector_rs_ic_table(
    tickers: list[str], lookback_candidates: list[int], start: pd.Timestamp, end: pd.Timestamp,
    log=print,
) -> list[dict]:
    """The full fetch -> simulate -> pooled-IC-per-lookback pipeline, factored
    out of main() so reoptimize_best_ideas.py can call it directly (no
    subprocess/output-parsing) instead of just re-running the CLI. `log`
    defaults to print but can be swapped for a list-collecting callable by
    a caller building a combined report string. Returns the same
    `results` list `recommend_lookback()` expects."""
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)
    configs = load_live_configs()
    log("Testing LIVE configs (the same 2 strategies best_ideas.METHODOLOGIES draws from):")
    for strategy, (config, label) in configs.items():
        log(f"  {strategy}: {label}")

    log(f"\nFetching {MARKET_INDEX_TICKER} + {len(SECTOR_ETF)} sector ETFs...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        raise RuntimeError(f"No data returned for {MARKET_INDEX_TICKER}")
    sector_etf_data = {}
    for i, (sector, etf) in enumerate(SECTOR_ETF.items()):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        df = fetch_history(etf, start, end)
        if not df.empty:
            sector_etf_data[sector] = df
    log(f"Fetched {len(sector_etf_data)}/{len(SECTOR_ETF)} sector ETF(s).")

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_history(ticker, start, end)
            if not df.empty:
                ticker_data[ticker] = df
        except Exception as exc:
            log(f"  [WARN] {ticker}: {exc}")
    log(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        raise RuntimeError("No ticker data available.")

    log(f"\nSimulating {len(configs)} live strategies for {len(ticker_data)} ticker(s), "
        f"{start.date()}..{end.date()}...")
    all_trades = []
    for strategy, (config, _) in configs.items():
        sim_fn = LIVE_STRATEGY_SIMULATORS[strategy]
        strategy_trades = []
        for ticker, ohlcv in ticker_data.items():
            sector = sector_lookup.get(ticker, "Unknown")
            trades = sim_fn(ticker, ohlcv, market_data, start, end, config, sector=sector)
            strategy_trades.extend(t for t in trades if t["status"] != "OPEN")
            all_trades.extend(t for t in trades if t["status"] != "OPEN")
        log(f"  {strategy}: {len(strategy_trades)} settled trade(s)")

    log(f"\nComputing sector relative-strength score at each trade's own signal_date, "
        f"for {len(lookback_candidates)} candidate lookback(s): {lookback_candidates}")
    results = []
    for lookback in lookback_candidates:
        scores, pnls = [], []
        for t in all_trades:
            sector = t.get("sector", "Unknown")
            etf_df = sector_etf_data.get(sector)
            if etf_df is None:
                continue
            as_of = pd.Timestamp(t["signal_date"])
            etf_slice = etf_df.loc[:as_of]
            market_slice = market_data.loc[:as_of]
            rs = compute_relative_strength(etf_slice, market_slice, lookback)
            if rs is None:
                continue
            scores.append(rs)
            pnls.append(t["pnl_pct"])
        ic = ic_tracking.rank_ic(scores, pnls)
        results.append({"lookback_days": lookback, "n": len(scores), "ic": ic})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 5y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument(
        "--lookback-candidates", default=",".join(str(x) for x in DEFAULT_LOOKBACK_CANDIDATES),
        help=f"Comma-separated lookback windows (trading days) to compare. Default: {DEFAULT_LOOKBACK_CANDIDATES}.",
    )
    args = parser.parse_args()

    lookback_candidates = [int(x.strip()) for x in args.lookback_candidates.split(",") if x.strip()]

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 5)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)

    try:
        results = compute_sector_rs_ic_table(tickers, lookback_candidates, start, end)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Sector-RS IC by lookback window (pooled across all settled trades) ===")
    print(f"{'lookback_days':>14} | {'n':>6} | {'ic':>8}")
    for r in results:
        marker = "  <-- CURRENT" if r["lookback_days"] == CURRENT_LOOKBACK else ""
        ic_str = f"{r['ic']:.4f}" if r["ic"] is not None else "None"
        print(f"{r['lookback_days']:>14} | {r['n']:>6} | {ic_str:>8}{marker}")

    recommendation = recommend_lookback(results)

    print()
    if recommendation is None:
        print(f"No candidate meaningfully beats the current {CURRENT_LOOKBACK}d lookback "
              f"(margin required: {MEANINGFUL_IC_MARGIN} IC) AND clears the {MIN_ABSOLUTE_IC} "
              "absolute-informativeness floor (a candidate only being LESS BAD than a weak "
              "current lookback is not a real basis to recommend anything on its own) -- "
              "no change recommended.")
    elif recommendation["current_ic"] is None:
        print(f"Current lookback ({CURRENT_LOOKBACK}d) has no defined IC -- "
              f"candidate {recommendation['lookback_days']}d (IC={recommendation['ic']:.4f}) is worth a look. "
              "NOT applied automatically.")
    else:
        print(f"RECOMMENDATION: {recommendation['lookback_days']}d beats the current {CURRENT_LOOKBACK}d by "
              f"{recommendation['ic'] - recommendation['current_ic']:.4f} IC "
              f"(>= {MEANINGFUL_IC_MARGIN} margin) -- worth considering for "
              "config.sector_relative_strength_lookback_days. NOT applied automatically.")


if __name__ == "__main__":
    main()
