"""One-off verification (not committed): re-run v60 (live) and v71 (candidate)
ma_crossover configs against the EXACT SAME 102-ticker holdout set stored on
v71's own System_Config document, using the current (2026-08-31 concurrency-
weighted) compute_k_ratio(). Compares the fresh result against the OLD
(pre-fix, naive entry-date-ordered) k_ratio values already on record:
  - v71 candidate, HOLDOUT: OLD 11.446 (stored in Mongo, holdout_metrics.candidate.k_ratio)
  - v71 candidate, TUNE:    OLD 39.034 (stored in Mongo, metrics.k_ratio)
  - v60 live, HOLDOUT:      OLD 24.57  (from the 2026-08-31 head-to-head one-off
                             script's own result, recorded in session notes/memory)
Mirrors optimize.py's own holdout evaluation path exactly (same
run_walk_forward + summarize_weighted call shape) so the recomputed numbers
are apples-to-apples with what's already stored.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config_loader
import market_data as market_data_module
import storage
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import SECTOR_ETF, read_ticker_sectors

REQUEST_DELAY_SEC = 0.3
WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "watchlist.txt"


def main():
    storage.mongo.get_db()

    doc60 = storage.system_config.get_config_by_version(60)
    doc71 = storage.system_config.get_config_by_version(71)
    config60 = swingtrade.TradingConfig.from_dict(doc60["params"])
    config71 = swingtrade.TradingConfig.from_dict(doc71["params"])

    hm = doc71["holdout_metrics"]
    holdout_tickers = hm["holdout_tickers"]
    old_v71_holdout_k = hm["candidate"]["k_ratio"]
    old_v71_tune_k = doc71["metrics"]["k_ratio"]
    print(f"Reusing v71's own stored 102-ticker holdout set (seed={hm['holdout_seed']}, frac={hm['holdout_frac']}).")
    print(f"OLD (pre-fix) v71 candidate k_ratio: TUNE={old_v71_tune_k}, HOLDOUT={old_v71_holdout_k}")
    print(f"OLD (pre-fix) v60 head-to-head holdout k_ratio (from session notes): 24.57")
    print()

    # Notes on v71 say: "20 trials, 2021-08-30..2026-08-30, ... 54 fold(s),
    # recency_half_life_days=180" -- reconstruct the identical window/fold
    # shape (optimize.py's own --in-sample-days/--out-sample-days/--step-days
    # defaults: 182/30/30).
    start = pd.Timestamp("2021-08-30")
    end = pd.Timestamp("2026-08-30")
    recency_half_life_days = 180.0

    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Fetching history for {MARKET_INDEX_TICKER} + {len(holdout_tickers)} holdout ticker(s), "
          f"{start.date()}..{end.date()}...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)

    present_sectors = sorted({sector_lookup[t] for t in holdout_tickers if t in sector_lookup} & set(SECTOR_ETF))
    sector_data = {}
    for i, sector in enumerate(present_sectors):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        etf_df = fetch_history(SECTOR_ETF[sector], start, end)
        if not etf_df.empty:
            sector_data[sector] = etf_df
    print(f"Fetched {len(sector_data)} sector ETF(s).")

    yield_curve = None
    if market_data_module.fred_available():
        yield_curve = market_data_module.fetch_fred_series("T10Y2Y", start, end)
        print("Fetched FRED T10Y2Y yield-curve spread.")
    else:
        print("FRED_API_KEY not set -- yield_curve=None (matches how v60 has always run; v71's own "
              "yield-curve filter reads Yield_Curve_Spread as missing, i.e. a no-op, same as live production today).")

    ticker_data = {}
    for i, ticker in enumerate(holdout_tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_history(ticker, start, end)
            if not df.empty:
                ticker_data[ticker] = df
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}")
    print(f"Fetched {len(ticker_data)}/{len(holdout_tickers)} holdout ticker(s).")

    folds = swingtrade.generate_folds(start, end, 182, 30, 30)
    print(f"Generated {len(folds)} fold(s) (stored notes say 54).")

    for label, config in (("v60 (LIVE)", config60), ("v71 (CANDIDATE)", config71)):
        results = swingtrade.run_walk_forward(
            ticker_data, market_data, folds, config, strategy="ma_crossover",
            sector_lookup=sector_lookup, sector_data=sector_data, yield_curve=yield_curve,
        )
        import optimize
        metrics = optimize.summarize_weighted(results, end, recency_half_life_days)
        dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(results))
        print()
        print(f"=== {label} on holdout (NEW concurrency-weighted k_ratio) ===")
        print(f"  k_ratio={metrics['k_ratio']}, max_drawdown={dd}%, sharpe_like={metrics['sharpe_like']}, "
              f"win_rate={metrics['win_rate']}, trade_count={metrics['trade_count']}, "
              f"effective_trade_count={metrics['effective_trade_count']}")


if __name__ == "__main__":
    main()
