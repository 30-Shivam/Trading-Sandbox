"""Regression coverage for skew_regime threading through run_walk_forward()'s
multiprocessing machinery (2026-09-03, benchmark_skew_regime.py's real
finding built into ma_crossover_skew_regime_min) -- mirrors
test_yield_curve_multiprocessing_threading.py exactly, the SAME
_init_worker()/ProcessPoolExecutor setup that caused a real, previously-
documented production bug this session (Windows spawn deadlock, see
[[windows-multiprocessing-guard]]), so this gets a direct regression test
rather than just "it didn't crash."

Deliberately does NOT spin up a real ProcessPoolExecutor in the test suite
(same reasoning as the sector_data/yield_curve versions) -- calls
_init_worker()/_run_fold_worker() directly (exactly what a spawned worker
process would do) and compares against _run_fold_sequential() called with
the same skew_regime explicitly, which must match byte-for-byte. The real
end-to-end ProcessPoolExecutor path (parallel=True) is a manual/smoke-test
check, same convention every other shared-data threading test here already
follows.
"""
import pandas as pd

import swingtrade
from swingtrade.backtest import Fold, _init_worker, _run_fold_sequential, _run_fold_worker


def _make_config(skew_regime_min: float) -> swingtrade.TradingConfig:
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(),
        "strategy": "ma_crossover",
        "ma_crossover_skew_regime_min": skew_regime_min,
    })


def _make_fold(ohlcv) -> Fold:
    mid = ohlcv.index[len(ohlcv) // 2]
    return Fold(
        in_sample_start=ohlcv.index[0], in_sample_end=mid,
        out_sample_start=mid, out_sample_end=ohlcv.index[-1],
    )


def test_worker_globals_receive_skew_regime(uptrend_ohlcv, market_ohlcv):
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    # Constant series -- its own rolling median equals itself, so
    # Skew_Regime_Diff reads ~0.0 everywhere, well above any disabled
    # (-1000.0) threshold used below.
    skew_regime = pd.Series(130.0, index=uptrend_ohlcv.index)
    config = _make_config(skew_regime_min=-1000.0)  # disabled, matches default
    fold = _make_fold(uptrend_ohlcv)

    _init_worker(ticker_data, market_ohlcv, None, None, None, None, skew_regime)
    worker_result = _run_fold_worker(fold, config, {}, sector_lookup, "ma_crossover")

    direct_result = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, config, {}, sector_lookup, "ma_crossover",
        None, None, None, None, skew_regime,
    )

    assert len(worker_result.out_sample_trades) == len(direct_result.out_sample_trades)
    assert worker_result.out_sample_metrics == direct_result.out_sample_metrics


def test_skew_regime_actually_changes_results_when_threaded(uptrend_ohlcv, market_ohlcv):
    # A constant ^SKEW series has Skew_Regime_Diff ~= 0.0 everywhere (its
    # own rolling median equals itself) -- a strict min of 10.0 should
    # exclude EVERY signal (real finding: only ELEVATED-relative-to-recent-
    # history diffs are favorable, so this is a deliberately-impossible-to-
    # clear floor). At the disabled default (-1000.0), it must exclude
    # nothing extra.
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    skew_regime = pd.Series(130.0, index=uptrend_ohlcv.index)
    fold = _make_fold(uptrend_ohlcv)

    strict_config = _make_config(skew_regime_min=10.0)
    disabled_config = _make_config(skew_regime_min=-1000.0)

    strict_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "ma_crossover",
        None, None, None, None, skew_regime,
    )
    disabled_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, disabled_config, {}, sector_lookup, "ma_crossover",
        None, None, None, None, skew_regime,
    )
    strict_without_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "ma_crossover",
        None, None, None, None, None,
    )

    strict_total = len(strict_with_data.in_sample_trades) + len(strict_with_data.out_sample_trades)
    disabled_total = len(disabled_with_data.in_sample_trades) + len(disabled_with_data.out_sample_trades)
    strict_without_total = (
        len(strict_without_data.in_sample_trades) + len(strict_without_data.out_sample_trades)
    )

    assert strict_total == 0, "an impossible-to-clear skew-regime floor should exclude every trade"
    assert disabled_total > 0, "the disabled default should not exclude anything (sanity check on the fixture)"
    # Without skew_regime at all, Skew_Regime_Diff is None -- the strict
    # threshold must NOT exclude anything, proving the exclusion above
    # genuinely came from skew_regime being threaded through, not from
    # something else about the strict config.
    assert strict_without_total == disabled_total
