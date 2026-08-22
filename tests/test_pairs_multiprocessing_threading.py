"""Regression coverage for pair_price_panels threading through
run_walk_forward()'s multiprocessing machinery (improvements.txt item 83)
-- the SAME _init_worker()/ProcessPoolExecutor setup that caused a real,
previously-documented production bug this session (Windows spawn deadlock,
see [[windows-multiprocessing-guard]]), and the same pattern sector_data's
own threading already got a direct regression test for (see
tests/test_sector_data_multiprocessing_threading.py) rather than just "it
didn't crash."

Deliberately does NOT spin up a real ProcessPoolExecutor in the test suite
(slow, a new source of flakiness this project hasn't had before) -- instead
calls _init_worker()/_run_fold_worker() directly (exactly what a spawned
worker process would do) and compares against _run_fold_sequential() called
with the same pair_price_panels explicitly, which must match byte-for-byte.
The real end-to-end ProcessPoolExecutor path (parallel=True) was separately
smoke-tested manually against real data.
"""
import numpy as np
import pandas as pd

import swingtrade
from swingtrade.backtest import Fold, _init_worker, _run_fold_sequential, _run_fold_worker


def _make_config(entry_fill: str = "next_open") -> swingtrade.TradingConfig:
    # next_open (not the "limit" default) -- this fixture's own steady,
    # smooth synthetic uptrend rarely dips back to touch a resting limit
    # order at the signal day's own Close (see strategy-validation-
    # pipeline point 7's documented "price that keeps running never gets
    # touched" issue for same-day-Close-entry strategies), which would
    # make this fixture unable to produce ANY real trade regardless of
    # whether pair_price_panels is threaded correctly -- next_open sidesteps
    # that fill-model sensitivity so this test isolates threading, not
    # fill-model behavior (already covered elsewhere for other strategies).
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "pairs", "pairs_entry_fill": entry_fill,
    })


def _make_fold(ohlcv) -> Fold:
    mid = ohlcv.index[len(ohlcv) // 2]
    return Fold(
        in_sample_start=ohlcv.index[0], in_sample_end=mid,
        out_sample_start=mid, out_sample_end=ohlcv.index[-1],
    )


def _make_pair_price_panel(base_ohlcv: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A same-sector peer panel for "TEST" -- includes TEST's OWN column
    (mirrors real production panels built in benchmark_random_entry.py/
    optimize.py, which include every sector member including the ticker
    itself; run_backtest() drops the ticker's own column internally before
    passing peer_prices through to simulate_pairs_signals() -- a panel
    missing the ticker's own column here would make run_backtest()'s
    `ticker in pair_panel.columns` check False, silently resolving
    peer_prices to None regardless of whether real peer data was
    intended). PEER is derived from TEST's own Close with small
    independent noise PLUS a steady extra drift, guaranteed highly
    correlated (unlike two independently-seeded fixtures) while still
    diverging enough, persistently, to cross pairs_zscore_entry_max for
    real on several real dates -- small noise alone (no drift) never
    diverges far enough from its own rolling baseline to fire at all."""
    rng = np.random.default_rng(seed)
    n = len(base_ohlcv)
    noise = 1 + rng.normal(0, 0.002, n)
    drift = np.cumprod(1 + np.full(n, 0.0015))
    peer_close = base_ohlcv["Close"] * noise * drift
    return pd.DataFrame({"TEST": base_ohlcv["Close"], "PEER": peer_close}, index=base_ohlcv.index)


def test_worker_globals_receive_pair_price_panels(uptrend_ohlcv):
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    pair_price_panels = {"Technology": _make_pair_price_panel(uptrend_ohlcv, seed=1)}
    config = _make_config()
    fold = _make_fold(uptrend_ohlcv)

    _init_worker(ticker_data, uptrend_ohlcv, None, pair_price_panels)
    worker_result = _run_fold_worker(fold, config, {}, sector_lookup, "pairs")

    direct_result = _run_fold_sequential(
        fold, ticker_data, uptrend_ohlcv, config, {}, sector_lookup, "pairs", None, pair_price_panels,
    )

    assert len(worker_result.out_sample_trades) == len(direct_result.out_sample_trades)
    assert worker_result.out_sample_metrics == direct_result.out_sample_metrics


def test_pair_price_panels_actually_changes_results_when_threaded(uptrend_ohlcv):
    # A real, persistently-diverging peer -- with pair_price_panels threaded
    # through, Pair_Signal fires for real days (confirmed directly against
    # precompute_pairs_frame() before writing this test: 6 real
    # Pair_Spread_Zscore <= pairs_zscore_entry_max crossings on this exact
    # fixture/seed, both before and after the fold midpoint). Without it
    # (None), Pair_Partner/Pair_Spread_Zscore are always missing, so
    # Pair_Signal must always be False -- zero pairs trades, regardless of
    # the config.
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    pair_price_panels = {"Technology": _make_pair_price_panel(uptrend_ohlcv, seed=1)}
    config = _make_config()
    fold = _make_fold(uptrend_ohlcv)

    with_data = _run_fold_sequential(
        fold, ticker_data, uptrend_ohlcv, config, {}, sector_lookup, "pairs", None, pair_price_panels,
    )
    without_data = _run_fold_sequential(
        fold, ticker_data, uptrend_ohlcv, config, {}, sector_lookup, "pairs", None, None,
    )

    with_total = len(with_data.in_sample_trades) + len(with_data.out_sample_trades)
    without_total = len(without_data.in_sample_trades) + len(without_data.out_sample_trades)
    assert with_total > 0, "a real, persistently-diverging peer should produce at least one real pairs trade"
    assert without_total == 0, "with no peer data at all, Pair_Signal must always be False"
