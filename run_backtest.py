"""
Walk-Forward backtest CLI (Phase 4).

Fetches historical OHLCV for the watchlist (+ SPY as the market-uptrend
proxy) via yfinance, runs the swingtrade walk-forward harness across a
sequence of rolling in-sample/out-of-sample folds, and prints per-fold
in-sample vs. out-of-sample performance for the given TradingConfig.

This is what will let Optuna (Phase 5) score a candidate RSI/ATR parameter
set against years of history instantly, instead of waiting months for live
Trade_Outcomes to accumulate -- see swingtrade/backtest.py's module
docstring for the full walk-forward rationale.

Also prints a correlation-adjusted summary alongside the raw aggregate:
pooled trades aren't independent when many fire the same day in the same
sector (a real, observed failure mode), so the raw trade_count/sharpe_like
above can overstate confidence -- see swingtrade.compute_cluster_weights.

Pass --with-catalyst to also fetch each ticker's historical earnings dates
(yfinance's get_earnings_dates(limit=40), only the calendar date, never the
EPS/Surprise columns -- see fetch_earnings_dates and swingtrade/backtest.py
for why this introduces no look-ahead leakage) and print a catalyst-vs-non
performance breakdown. Off by default since it doesn't change which trades
get simulated (Catalyst_Warning isn't consumed by Trade_Score/Signal
anywhere) -- it costs one extra yfinance call per ticker purely for
reporting/analysis.

Usage:
    python run_backtest.py
    python run_backtest.py --start 2023-01-01 --end 2026-07-01
    python run_backtest.py --in-sample-days 120 --out-sample-days 20 --step-days 20
    python run_backtest.py --with-catalyst
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import swingtrade
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
MARKET_INDEX_TICKER = "SPY"
LOOKBACK_BUFFER_DAYS = 420  # calendar-day buffer before window start -- sized for the
                             # LARGEST indicator window across all strategies:
                             # week52_high's week52_lookback_days (252 trading days,
                             # default), not just SMA200's 200. 252 trading days is
                             # roughly 365 calendar days; this leaves a real safety
                             # margin on top for holidays/warmup, same margin the old
                             # 320-day constant left for SMA200's shorter window.
REQUEST_DELAY_SEC = 0.5
EARNINGS_HISTORY_LIMIT = 40  # ~10 years of quarterly reports

# Research holdout cutoff (2026-09-13, improvements.txt item 132) -- per
# explicit user request to make this project's backtesting "iron proof."
# Every strategy tuning/selection/comparison run this project has EVER done
# (including everything this whole session) has used `end=today` by
# default -- meaning there is no longer any stretch of calendar time that
# hasn't already been looked at, directly or indirectly, while deciding
# which strategies to trust. This constant draws a line, starting NOW: any
# NEW strategy-selection work (a fresh Optuna search, a real-vs-random
# validation run, a "does this candidate beat the live config" comparison)
# should pass `--use-research-cutoff` (optimize.py/benchmark_random_entry.py)
# so it only ever sees data up to this date -- reserving everything AFTER
# it as genuine, never-yet-tuned-against future data.
#
# The whole point only holds if this discipline is actually followed:
# checking data after this cutoff should happen EXACTLY ONCE, immediately
# before a final live-promotion decision for whatever candidate survives
# every other step -- not repeatedly "peeked at" as new real dates accrue,
# which would silently recreate the exact whole-program data-snooping
# problem this constant exists to prevent (see
# [[feedback_strategy_validation_pipeline]] point 21's own docstring for
# the fuller argument: point 17's Deflated Sharpe Ratio corrects for many
# trials WITHIN one search; nothing before this corrected for many
# DIFFERENT strategy families all being iterated against the same,
# ever-more-thoroughly-examined historical window across an entire
# research program).
#
# Deliberately NOT the default for `--end` everywhere -- live-status
# monitoring, trust-floor sweeps, and smoke tests all have a legitimate
# reason to use today's real date, and forcing this cutoff on those would
# just be confusing, not safer. This is opt-in, for tuning/selection work
# specifically, via `--use-research-cutoff`.
#
# Update this value only when a genuine final promotion check against
# everything after the current cutoff has actually been run -- moving it
# forward "just because time passed" without that check defeats the
# purpose.
RESEARCH_HOLDOUT_CUTOFF = pd.Timestamp("2026-09-13")


def fetch_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    buffered_start = start - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
    df = yf.download(ticker, start=buffered_start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 2026-09-17: repair the rare-but-real structural OHLC violations
    # audit_data_quality() catches (see swingtrade.sanitize_ohlcv()'s own
    # docstring -- the same 2021-05-05 vendor anomaly recurred across 4
    # unrelated tickers in both this project's universes) BEFORE any
    # strategy's ATR/stop-loss/entry-fill math ever sees the row.
    return swingtrade.sanitize_ohlcv(df)


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """Historical + a few upcoming earnings report dates, tz-aware UTC
    (matching market_data.get_next_earnings_date's live convention) so they
    compare directly against backtest `as_of` timestamps. Only the .index
    (the calendar date) is used -- never the EPS Estimate/Reported
    EPS/Surprise columns, which ARE genuinely forward-looking and would be
    real leakage. The date itself, once a company schedules a report, is a
    fixed fact that doesn't get revised the way an EPS estimate does, so
    using it for a backtest day years in the past introduces no look-ahead
    bias -- see swingtrade/backtest.py's module docstring."""
    try:
        dates = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_HISTORY_LIMIT).index
    except Exception:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates).tz_convert("UTC")


def fetch_insider_purchases(ticker: str, config: swingtrade.TradingConfig = swingtrade.DEFAULT_CONFIG) -> pd.DataFrame:
    """Real, open-market insider Form-4 purchases -- see
    swingtrade/config.py's insider_* fields and
    swingtrade/levels.precompute_insider_buying_frame() for how this feeds
    the "insider_buying" strategy. yfinance's own insider_transactions
    caps at ~150 rows/ticker (roughly 12-22 months of real history, NOT
    this project's usual 5-year backtest window -- a real, structural data
    limitation, not something fixable here) and its "Transaction" column
    returns empty in the currently installed version (1.5.1), so
    classification comes from parsing the free-text "Text" field via
    swingtrade.classify_insider_transaction().

    IMPORTANT no-look-ahead handling: yfinance's only date field ("Start
    Date") is ambiguous between the actual transaction date and the SEC
    filing date, and Form 4 filings can legitimately lag the trade by a
    few business days -- unlike fetch_earnings_dates()'s date (a fixed,
    already-public fact once scheduled), this one needs a conservative,
    EXPLICIT assumption rather than a silent one. `effective_date` = the
    raw "Start Date" plus `config.insider_reporting_lag_days` calendar
    days -- the date this purchase is treated as PUBLICLY known, not when
    it actually happened. Rows missing a real $ Value (mostly option
    exercises, not confirmed open-market buys) are dropped.

    Degrades to an empty DataFrame (same shape, zero rows) rather than
    crashing when a ticker has no insider data at all -- same convention
    fetch_earnings_dates() already follows."""
    columns = ["effective_date", "value", "insider"]
    try:
        raw = yf.Ticker(ticker).insider_transactions
    except Exception:
        raw = None
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)

    tier = raw["Text"].apply(swingtrade.classify_insider_transaction)
    purchases = raw[(tier == "purchase") & raw["Value"].notna()].copy()
    if purchases.empty:
        return pd.DataFrame(columns=columns)

    lag = pd.Timedelta(days=config.insider_reporting_lag_days)
    effective_date = pd.to_datetime(purchases["Start Date"]) + lag
    effective_date = effective_date.dt.tz_localize("UTC") if effective_date.dt.tz is None else effective_date.dt.tz_convert("UTC")

    return pd.DataFrame({
        "effective_date": effective_date.values,
        "value": purchases["Value"].astype(float).values,
        "insider": purchases["Insider"].values,
    })


def fetch_earnings_surprises(ticker: str) -> pd.DataFrame:
    """Real historical earnings-surprise events for the "pead" (post-
    earnings-announcement drift) strategy -- see swingtrade/config.py's
    pead_* fields and swingtrade/levels.precompute_pead_frame() for how
    this feeds it. Reuses the SAME yfinance endpoint fetch_earnings_dates()
    already calls (get_earnings_dates(limit=EARNINGS_HISTORY_LIMIT)), but
    -- unlike that function, which deliberately discards everything except
    the calendar date -- this one keeps the "EPS Estimate"/"Reported EPS"/
    "Surprise(%)" columns too, since PEAD's entire signal IS the surprise
    magnitude.

    CRITICAL, EXPLICIT DATA-INTEGRITY CAVEAT, not a silent assumption:
    fetch_earnings_dates()'s own docstring notes the calendar date is a
    fixed fact (once scheduled, it doesn't get revised) but explicitly
    calls the EPS/Surprise columns "genuinely forward-looking" for a
    reason -- this project has NOT independently verified that yfinance's
    historical "EPS Estimate" figure for an OLD quarter reflects what the
    consensus estimate actually was right before that report (point-in-
    time), versus some vendor-side backfill/revision after the fact (a
    known general risk with free financial data, unlike raw OHLCV, which
    never gets revised). If this data is NOT point-in-time, backtesting
    against it would introduce real look-ahead leakage this project's own
    no-look-ahead discipline otherwise guards against everywhere else.
    Shipped anyway, same "best available free data, explicit assumption,
    not a silent one" posture run_backtest.fetch_insider_purchases() itself
    already takes for its own effective_date ambiguity -- but flag this
    caveat plainly whenever reporting a backtested PEAD result, and treat
    an implausibly strong result (bigger than the modest effect size the
    academic PEAD literature documents) as a reason to suspect this leak
    specifically, not as confirmation of a great strategy.

    `effective_date` is the raw earnings-report timestamp itself (tz-aware
    UTC) -- deliberately NO added lag (unlike fetch_insider_purchases()'s
    Start-Date ambiguity): a report timestamped after that day's market
    close naturally excludes that same calendar day from
    precompute_pead_frame()'s own day-index comparison (midnight < that
    day's close-time timestamp), so the signal only starts counting from
    the NEXT trading day onward -- correctly modeling "you can't react
    until the market can," with no separate before/after-market lag
    parameter needed the way insider Form-4 filings need one.

    Degrades to an empty DataFrame (same shape, zero rows) rather than
    crashing when a ticker has no earnings-surprise data at all -- same
    convention fetch_earnings_dates()/fetch_insider_purchases() already
    follow. Rows with a NaN Surprise(%) (unreported estimate, or a report
    still pending) are dropped."""
    columns = ["effective_date", "surprise_pct"]
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_HISTORY_LIMIT)
    except Exception:
        raw = None
    if raw is None or raw.empty or "Surprise(%)" not in raw.columns:
        return pd.DataFrame(columns=columns)

    reported = raw[raw["Surprise(%)"].notna()].copy()
    if reported.empty:
        return pd.DataFrame(columns=columns)

    effective_date = pd.DatetimeIndex(reported.index)
    effective_date = effective_date.tz_localize("UTC") if effective_date.tz is None else effective_date.tz_convert("UTC")

    return pd.DataFrame({
        "effective_date": effective_date,
        "surprise_pct": reported["Surprise(%)"].astype(float).values,
    })


def print_fold_table(fold_results: list) -> None:
    headers = ["Fold", "In-Sample", "Out-of-Sample", "IS #", "IS Win%", "OOS #", "OOS Win%", "OOS AvgPnL%", "OOS Sharpe"]
    widths = [6, 24, 24, 6, 9, 6, 9, 13, 11]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for i, fr in enumerate(fold_results, start=1):
        is_range = f"{fr.fold.in_sample_start.date()}..{fr.fold.in_sample_end.date()}"
        oos_range = f"{fr.fold.out_sample_start.date()}..{fr.fold.out_sample_end.date()}"
        ism, oom = fr.in_sample_metrics, fr.out_sample_metrics
        row = [
            str(i), is_range, oos_range,
            str(ism["trade_count"]), _fmt(ism["win_rate"]),
            str(oom["trade_count"]), _fmt(oom["win_rate"]),
            _fmt(oom["avg_pnl_pct"]), _fmt(oom["sharpe_like"]),
        ]
        print("".join(v.ljust(w) for v, w in zip(row, widths)))


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 1y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument(
        "--with-catalyst", action="store_true",
        help="Fetch historical earnings dates and simulate Catalyst_Warning honestly, printing a "
             "catalyst-vs-non breakdown at the end. One extra yfinance call per ticker.",
    )
    args = parser.parse_args()

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

    print(f"Backtesting {len(tickers)} ticker(s) over {start.date()} to {end.date()}")

    print(f"Fetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        print(f"[{i + 1}/{len(tickers)}] {ticker} ...", end=" ")
        try:
            df = fetch_history(ticker, start, end)
            if df.empty:
                print("no data")
                continue
            ticker_data[ticker] = df
            print("ok")
        except Exception as exc:
            print("skipped")
            print(f"  [WARN] {ticker}: {exc}", file=sys.stderr)

    print(f"\nFetched history for {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        print("[ERROR] No ticker data available to backtest.", file=sys.stderr)
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

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"Generated {len(folds)} walk-forward fold(s) "
          f"(in-sample={args.in_sample_days}d, out-of-sample={args.out_sample_days}d, step={args.step_days}d).")
    if not folds:
        print("[ERROR] Date range too short to generate even one fold with these window sizes.", file=sys.stderr)
        sys.exit(1)

    fold_results = swingtrade.run_walk_forward(
        ticker_data, market_data, folds, swingtrade.DEFAULT_CONFIG, earnings_data, sector_lookup
    )

    print()
    print_fold_table(fold_results)

    all_oos_trades = [t for fr in fold_results for t in fr.out_sample_trades]
    resolved_oos_trades = [t for t in all_oos_trades if t["status"] != "OPEN"]
    overall = swingtrade.summarize_trades(all_oos_trades)
    print()
    print(f"Aggregate out-of-sample performance across all {len(fold_results)} fold(s): {overall}")

    cluster_weights = swingtrade.compute_cluster_weights(resolved_oos_trades)
    cluster_adjusted = swingtrade.summarize_trades_weighted(resolved_oos_trades, cluster_weights)
    print(f"Correlation-adjusted (same-day/same-sector trades counted once, not N times): {cluster_adjusted}")
    if cluster_adjusted["effective_trade_count"] and overall["trade_count"]:
        shrink_pct = (1 - cluster_adjusted["effective_trade_count"] / overall["trade_count"]) * 100
        print(f"  -> raw trade_count is {shrink_pct:.0f}% inflated by same-day/same-sector correlation.")

    if args.with_catalyst:
        breakdown = swingtrade.summarize_by_catalyst(all_oos_trades)
        print()
        print(f"  Catalyst_Warning=True : {breakdown['catalyst_warning_true']}")
        print(f"  Catalyst_Warning=False: {breakdown['catalyst_warning_false']}")

    print()
    print("This is a mechanical replay of historical price data, not a forecast. Execution "
          "assumptions (slippage_pct, commission_pct_per_trade) are baked into pnl_pct above.")


if __name__ == "__main__":
    main()
