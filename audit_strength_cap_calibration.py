"""
Strength-cap calibration audit -- real-data check for every
`*_strength_cap_pct`/`*_strength_cap` config field.

Every `add_*_trade_score()` function (swingtrade/scoring.py) clips its own
`Signal_Strength_Pct` at one of these caps and normalizes it as
`clipped / cap` toward config.distance_score_weight points. A badly
calibrated cap silently breaks scoring in one of two directions without
ever raising an error:

  - Set too HIGH relative to what real signals actually achieve: almost
    every real observation earns only a tiny fraction of this component's
    points. The CONFIRMED real incident (improvements.txt item 97):
    ma_crossover's original ma_crossover_strength_cap_pct=2.0 against a
    real p90 of only 0.52% -- on its own enough to make Buy/Strong Buy
    STRUCTURALLY UNREACHABLE. Found by hand, once, via an ad-hoc script;
    fixed by lowering the cap to 0.5.
  - Set too LOW: most/many real signals already clip to the SAME max
    score, losing differentiation among genuinely different-strength
    signals at the top end.

This script is the REUSABLE version of that one-off investigation --
swingtrade.audit_cap_calibration() does the actual statistics (see its
own docstring); this script's job is just gathering REAL
Signal_Strength_Pct observations (every day a strategy's own boolean
signal condition was True, across real watchlist data) to feed it.

Scope this round: ma_crossover (re-verifying the item 97 fix still holds
under today's live config/data) and pairs (never previously audited this
way). momentum_rank/pead need cross-sectional rank/earnings-surprise data
respectively -- deliberately NOT built this round, an explicit, scoped
gap for a future iteration, not an oversight.

Usage:
    python audit_strength_cap_calibration.py
    python audit_strength_cap_calibration.py --tickers AAPL,MSFT,NVDA --strategy ma_crossover
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import swingtrade
from run_backtest import fetch_history
from swingtrade.levels import (
    ma_crossover_levels_from_frame,
    market_uptrend_from_frame,
    pairs_levels_from_frame,
    precompute_ma_crossover_frame,
    precompute_pairs_frame,
)
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
MARKET_INDEX_TICKER = "SPY"


def _collect_ma_crossover_strengths(ticker_data: dict, market_ohlcv: pd.DataFrame, config) -> list[float]:
    """Every REAL Signal_Strength_Pct observed on a day MA_Crossover_Signal
    was True, across every ticker -- the population a real Trade_Score
    would actually need to differentiate among."""
    market_frame = precompute_ma_crossover_frame(market_ohlcv, config)
    values = []
    for ticker, ohlcv in ticker_data.items():
        frame = precompute_ma_crossover_frame(ohlcv, config)
        for as_of in ohlcv.index:
            try:
                market_uptrend_from_frame(market_frame, as_of, config)
            except RuntimeError:
                continue
            try:
                levels = ma_crossover_levels_from_frame(ticker, frame, as_of, config)
            except RuntimeError:
                continue
            if levels["MA_Crossover_Signal"]:
                values.append(levels["Signal_Strength_Pct"])
    return values


def _collect_pairs_strengths(ticker_data: dict, market_ohlcv: pd.DataFrame, sector_lookup: dict, config) -> list[float]:
    """Every REAL Signal_Strength_Pct observed on a day Pair_Signal was
    True, built from a real same-sector peer panel per improvements.txt
    item 83's own convention (see optimize.py's pair_price_panels)."""
    market_frame = precompute_ma_crossover_frame(market_ohlcv, config)  # only used for market_uptrend_from_frame's own frame shape
    by_sector: dict[str, list[str]] = {}
    for t in ticker_data:
        by_sector.setdefault(sector_lookup.get(t, "Unknown"), []).append(t)
    pair_price_panels = {
        sector: pd.DataFrame({m: ticker_data[m]["Close"] for m in members})
        for sector, members in by_sector.items() if len(members) >= 2
    }

    values = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        panel = pair_price_panels.get(sector)
        peer_prices = panel.drop(columns=[ticker]) if panel is not None and ticker in panel.columns else None
        frame = precompute_pairs_frame(ohlcv, peer_prices, config)
        for as_of in ohlcv.index:
            try:
                market_uptrend_from_frame(market_frame, as_of, config)
            except RuntimeError:
                continue
            try:
                levels = pairs_levels_from_frame(ticker, frame, as_of, config)
            except RuntimeError:
                continue
            if levels["Pair_Signal"]:
                values.append(levels["Signal_Strength_Pct"])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 3y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument(
        "--strategy", choices=["ma_crossover", "pairs", "both"], default="both",
        help="Which strategy's cap to audit. Default: both.",
    )
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 3)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Fetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
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
    print(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s), {start.date()}..{end.date()}.\n")
    if not ticker_data:
        print("[ERROR] No ticker data available.", file=sys.stderr)
        sys.exit(1)

    if args.strategy in ("ma_crossover", "both"):
        config, label = config_loader.load_active_config()
        print(f"=== MA_CROSSOVER ({label}, cap={config.ma_crossover_strength_cap_pct}) ===")
        values = _collect_ma_crossover_strengths(ticker_data, market_ohlcv, config)
        result = swingtrade.audit_cap_calibration(values, config.ma_crossover_strength_cap_pct)
        print(f"  {result}")
        if result["likely_too_high"]:
            print("  [FLAG] cap looks miscalibrated TOO HIGH -- real strong signals barely use this component's points.")
        if result["likely_too_low"]:
            print("  [FLAG] cap looks miscalibrated TOO LOW -- many real signals saturate to the same max score.")
        print()

    if args.strategy in ("pairs", "both"):
        config, label = config_loader.load_config_by_version(config_loader.SECONDARY_STRATEGY_VERSIONS["Mean-Reversion Pairs"])
        if config is None:
            print(f"[ERROR] Could not load Mean-Reversion Pairs config: {label}", file=sys.stderr)
        else:
            print(f"=== PAIRS ({label}, cap={config.pairs_zscore_strength_cap}) ===")
            values = _collect_pairs_strengths(ticker_data, market_ohlcv, sector_lookup, config)
            result = swingtrade.audit_cap_calibration(values, config.pairs_zscore_strength_cap)
            print(f"  {result}")
            if result["likely_too_high"]:
                print("  [FLAG] cap looks miscalibrated TOO HIGH -- real strong signals barely use this component's points.")
            if result["likely_too_low"]:
                print("  [FLAG] cap looks miscalibrated TOO LOW -- many real signals saturate to the same max score.")
            print()

    print("momentum_rank/pead not audited this round (need cross-sectional rank / earnings-surprise data) -- scoped gap, not an oversight.")


if __name__ == "__main__":
    main()
