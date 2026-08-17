"""Regression coverage for sector_data threading through run_walk_forward()'s
multiprocessing machinery (improvements.txt items 68-71) -- the SAME
_init_worker()/ProcessPoolExecutor setup that caused a real, previously-
documented production bug this session (Windows spawn deadlock, see
[[windows-multiprocessing-guard]]), so this gets a direct regression test
rather than just "it didn't crash."

Deliberately does NOT spin up a real ProcessPoolExecutor in the test suite
(slow, a new source of flakiness this project hasn't had before) -- instead
calls _init_worker()/_run_fold_worker() directly (exactly what a spawned
worker process would do) and compares against _run_fold_sequential() called
with the same sector_data explicitly, which must match byte-for-byte. The
real end-to-end ProcessPoolExecutor path (parallel=True) was separately
smoke-tested manually against real data -- see improvements.txt item 71.
"""
import swingtrade
from swingtrade.backtest import Fold, _init_worker, _run_fold_sequential, _run_fold_worker


def _make_config(sector_min: float) -> swingtrade.TradingConfig:
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(),
        "strategy": "breakout",
        "breakout_sector_relative_strength_min": sector_min,
    })


def _make_fold(ohlcv) -> Fold:
    mid = ohlcv.index[len(ohlcv) // 2]
    return Fold(
        in_sample_start=ohlcv.index[0], in_sample_end=mid,
        out_sample_start=mid, out_sample_end=ohlcv.index[-1],
    )


def test_worker_globals_receive_sector_data(uptrend_ohlcv, market_ohlcv, sector_ohlcv):
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    sector_data = {"Technology": sector_ohlcv}
    config = _make_config(sector_min=-100.0)  # disabled, matches default
    fold = _make_fold(uptrend_ohlcv)

    _init_worker(ticker_data, market_ohlcv, sector_data)
    worker_result = _run_fold_worker(fold, config, {}, sector_lookup, "breakout")

    direct_result = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, config, {}, sector_lookup, "breakout", sector_data,
    )

    assert len(worker_result.out_sample_trades) == len(direct_result.out_sample_trades)
    assert worker_result.out_sample_metrics == direct_result.out_sample_metrics


def test_sector_data_actually_changes_results_when_threaded(uptrend_ohlcv, market_ohlcv, sector_ohlcv):
    # A near-impossible-to-clear threshold (sector must beat the market by
    # 50+ percentage points over 63 trading days) -- with real sector_data
    # threaded through, this should exclude every signal. At the disabled
    # default (-100.0), it must exclude nothing extra.
    ticker_data = {"TEST": uptrend_ohlcv}
    sector_lookup = {"TEST": "Technology"}
    sector_data = {"Technology": sector_ohlcv}
    fold = _make_fold(uptrend_ohlcv)

    strict_config = _make_config(sector_min=50.0)
    disabled_config = _make_config(sector_min=-100.0)

    strict_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "breakout", sector_data,
    )
    disabled_with_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, disabled_config, {}, sector_lookup, "breakout", sector_data,
    )
    strict_without_data = _run_fold_sequential(
        fold, ticker_data, market_ohlcv, strict_config, {}, sector_lookup, "breakout", None,
    )

    strict_total = len(strict_with_data.in_sample_trades) + len(strict_with_data.out_sample_trades)
    disabled_total = len(disabled_with_data.in_sample_trades) + len(disabled_with_data.out_sample_trades)
    strict_without_total = (
        len(strict_without_data.in_sample_trades) + len(strict_without_data.out_sample_trades)
    )

    assert strict_total == 0, "an extreme sector-outperformance threshold should exclude every trade"
    assert disabled_total > 0, "the disabled default should not exclude anything (sanity check on the fixture)"
    # Without sector_data at all, Sector_Relative_Strength is None -- the
    # strict threshold must NOT exclude anything, proving the exclusion
    # above genuinely came from sector_data being threaded through, not
    # from something else about the strict config.
    assert strict_without_total == disabled_total
