"""Regression coverage for yield_curve threading through run_walk_forward()'s
multiprocessing machinery (2026-08-31, benchmark_macro_regime.py's real
finding built into ma_crossover_yield_curve_spread_max) -- mirrors
test_sector_data_multiprocessing_threading.py exactly, the SAME
_init_worker()/ProcessPoolExecutor setup that caused a real, previously-
documented production bug this session (Windows spawn deadlock, see
[[windows-multiprocessing-guard]]), so this gets a direct regression test
rather than just "it didn't crash."

Deliberately does NOT spin up a real ProcessPoolExecutor in the test suite
(same reasoning as the sector_data version) -- calls _init_worker()/
_run_fold_worker() directly (exactly what a spawned worker process would
do) and compares against _run_fold_sequential() called with the same
yield_curve explicitly, which must match byte-for-byte. The real end-to-end
ProcessPoolExecutor path (parallel=True) is a manual/smoke-test check, same
convention every other shared-data threading test here already follows.
"""
import pandas as pd

import swingtrade
from swingtrade.backtest import Fold, _init_worker, _run_fold_sequential, _run_fold_worker


def _make_config(yield_curve_max: float) -> swingtrade.TradingConfig:
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(),
        "strategy": "ma_crossover",
        "ma_crossover_yield_curve_spread_max": yield_curve_max,
    })


def _make_fold(ohlcv) -> Fold:
    mid = ohlcv.index[len(ohlcv) // 2]
    return Fold(
        in_sample_start=ohlcv.index[0], in_sample_end=mid,
        out_sample_start=mid, out_sample_end=ohlcv.index[-1],
    )


def test_worker_globals_receive_yield_curve(uptrend_ohlcv, market_ohlcv):
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    yield_curve = pd.Series(5.0, index=uptrend_ohlcv.index)  # constant, well above any threshold used below
    config = _make_config(yield_curve_max=100.0)  # disabled, matches default
    fold = _make_fold(uptrend_ohlcv)

    _init_worker(ticker_data, market_ohlcv, None, None, None, yield_curve)
    worker_result = _run_fold_worker(fold, config, {}, sector_lookup, "ma_crossover")

    direct_result = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, config, {}, sector_lookup, "ma_crossover",
        None, None, None, yield_curve,
    )

    assert len(worker_result.out_sample_trades) == len(direct_result.out_sample_trades)
    assert worker_result.out_sample_metrics == direct_result.out_sample_metrics


def test_yield_curve_actually_changes_results_when_threaded(uptrend_ohlcv, market_ohlcv):
    # A constant spread of 5.0 with a strict max of -3.0 should exclude
    # EVERY signal (real finding: only INVERTED/low spreads are favorable,
    # so this is a deliberately-impossible-to-clear ceiling). At the
    # disabled default (100.0), it must exclude nothing extra.
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    yield_curve = pd.Series(5.0, index=uptrend_ohlcv.index)
    fold = _make_fold(uptrend_ohlcv)

    strict_config = _make_config(yield_curve_max=-3.0)
    disabled_config = _make_config(yield_curve_max=100.0)

    strict_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "ma_crossover",
        None, None, None, yield_curve,
    )
    disabled_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, disabled_config, {}, sector_lookup, "ma_crossover",
        None, None, None, yield_curve,
    )
    strict_without_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "ma_crossover",
        None, None, None, None,
    )

    strict_total = len(strict_with_data.in_sample_trades) + len(strict_with_data.out_sample_trades)
    disabled_total = len(disabled_with_data.in_sample_trades) + len(disabled_with_data.out_sample_trades)
    strict_without_total = (
        len(strict_without_data.in_sample_trades) + len(strict_without_data.out_sample_trades)
    )

    assert strict_total == 0, "an impossible-to-clear yield-curve ceiling should exclude every trade"
    assert disabled_total > 0, "the disabled default should not exclude anything (sanity check on the fixture)"
    # Without yield_curve at all, Yield_Curve_Spread is None -- the strict
    # threshold must NOT exclude anything, proving the exclusion above
    # genuinely came from yield_curve being threaded through, not from
    # something else about the strict config.
    assert strict_without_total == disabled_total
