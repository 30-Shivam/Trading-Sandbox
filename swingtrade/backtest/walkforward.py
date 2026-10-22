"""Walk-forward driver: per-fold execution (sequential and multiprocessing) and run_walk_forward()."""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .core import Fold
from .engine import run_backtest
from .stats import summarize_trades


@dataclass
class FoldResult:
    fold: Fold
    in_sample_trades: list[dict]
    out_sample_trades: list[dict]
    in_sample_metrics: dict
    out_sample_metrics: dict


def _run_fold_sequential(
    fold: Fold, ticker_data: dict[str, pd.DataFrame], market_data: pd.DataFrame,
    config: TradingConfig, earnings_data: dict, sector_lookup: dict, strategy: str,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
    momentum_rank_frame: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
) -> FoldResult:
    """The actual per-fold work, shared by both the sequential and
    parallel paths of run_walk_forward() -- one fold's in-sample and
    out-of-sample backtest, for every ticker. Kept as a plain module-level
    function (not a closure) so it's picklable for ProcessPoolExecutor."""
    in_trades = run_backtest(
        ticker_data, market_data, fold.in_sample_start, fold.in_sample_end, config,
        earnings_data, sector_lookup, strategy, sector_data, pair_price_panels,
        momentum_rank_frame=momentum_rank_frame, yield_curve=yield_curve, skew_regime=skew_regime,
    )
    out_trades = run_backtest(
        ticker_data, market_data, fold.out_sample_start, fold.out_sample_end, config,
        earnings_data, sector_lookup, strategy, sector_data, pair_price_panels,
        momentum_rank_frame=momentum_rank_frame, yield_curve=yield_curve, skew_regime=skew_regime,
    )
    return FoldResult(
        fold=fold,
        in_sample_trades=in_trades,
        out_sample_trades=out_trades,
        in_sample_metrics=summarize_trades(in_trades),
        out_sample_metrics=summarize_trades(out_trades),
    )


# Populated once per worker process by _init_worker() -- module-level so
# _run_fold_worker() (which only receives small per-task arguments, not the
# full ticker_data/market_data) can reach it without re-sending the whole
# OHLCV universe over IPC for every one of the 56+ folds. Each worker process
# gets its own copy via the ProcessPoolExecutor initializer, set ONCE when
# the pool starts, not once per fold.
_worker_ticker_data: dict[str, pd.DataFrame] | None = None


_worker_market_data: pd.DataFrame | None = None


# Sector-ETF OHLCV (sector NAME -> OHLCV, see watchlist.SECTOR_ETF) -- same
# "loaded once per worker, not once per fold" treatment as ticker_data/
# market_data above, since it's the same kind of shared, sizable data
# (improvements.txt items 68/70). None (default) means no sector data was
# supplied to run_walk_forward() at all -- every worker's Sector_Relative_Strength
# then reads None/NaN, same as before this field existed.
_worker_sector_data: dict[str, pd.DataFrame] | None = None


# Pairs-strategy peer price panels (sector NAME -> wide Close-price
# DataFrame of every ticker in that sector) -- same "loaded once per
# worker, not once per fold" treatment as sector_data above
# (improvements.txt item 83). None means no peer data was supplied at all,
# every worker's Pair_Signal then always reads False.
_worker_pair_price_panels: dict[str, pd.DataFrame] | None = None


# Cross-sectional momentum-rank strategy's universe-wide percentile frame
# (dates x tickers, see swingtrade.levels.compute_momentum_rank_frame()) --
# same "loaded once per worker, not once per fold" treatment as
# pair_price_panels above. THIS ONE IS NOT OPTIONAL TO THREAD THROUGH THE
# PARALLEL PATH THE WAY insider_data currently is: unlike an add-on filter,
# a ticker's ENTIRE momentum_rank signal depends on this frame, so a worker
# that never received it would silently produce zero signals in every
# trial (the exact failure mode flagged in optimize.py's own "insider_buying
# deliberately not included yet" comment) -- None means no percentile data
# was supplied at all, every worker's Momentum_Signal then always reads
# False.
_worker_momentum_rank_frame: pd.DataFrame | None = None


# Shared yield-curve Series (see run_backtest()'s own docstring) -- same
# "loaded once per worker, not once per fold" treatment as the above.
# Unlike sector_data/pair_price_panels, this is NOT keyed by anything --
# the exact same Series object is handed unchanged to every ticker.
_worker_yield_curve: pd.Series | None = None


# Shared CBOE SKEW Index Series (raw closes, see run_backtest()'s own
# docstring) -- same "loaded once per worker, not once per fold" treatment,
# same "not keyed by anything, handed unchanged to every ticker" shape as
# _worker_yield_curve above.
_worker_skew_regime: pd.Series | None = None


def _init_worker(
    ticker_data: dict[str, pd.DataFrame], market_data: pd.DataFrame,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
    momentum_rank_frame: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
) -> None:
    global _worker_ticker_data, _worker_market_data, _worker_sector_data, _worker_pair_price_panels
    global _worker_momentum_rank_frame, _worker_yield_curve, _worker_skew_regime
    _worker_ticker_data = ticker_data
    _worker_market_data = market_data
    _worker_sector_data = sector_data
    _worker_pair_price_panels = pair_price_panels
    _worker_momentum_rank_frame = momentum_rank_frame
    _worker_yield_curve = yield_curve
    _worker_skew_regime = skew_regime


def _run_fold_worker(
    fold: Fold, config: TradingConfig, earnings_data: dict, sector_lookup: dict, strategy: str,
) -> FoldResult:
    """Same work as _run_fold_sequential(), but reads ticker_data/market_data/
    sector_data/pair_price_panels/momentum_rank_frame from this worker
    process's own module-level copy (set once by _init_worker()) instead of
    taking them as arguments -- only `fold` and `config` (small, cheap to
    pickle) actually cross the process boundary per task."""
    return _run_fold_sequential(
        fold, _worker_ticker_data, _worker_market_data, config, earnings_data, sector_lookup, strategy,
        _worker_sector_data, _worker_pair_price_panels, _worker_momentum_rank_frame, _worker_yield_curve,
        _worker_skew_regime,
    )


def run_walk_forward(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    folds: list[Fold],
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    sector_lookup: dict[str, str] | None = None,
    strategy: str = "rsi",
    parallel: bool = True,
    max_workers: int | None = None,
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
    momentum_rank_frame: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
) -> list[FoldResult]:
    """Run the backtest across every fold, in-sample and out-of-sample
    separately. Optuna (Phase 5) evaluates a candidate config by scoring it
    across the aggregate of every fold's out_sample_metrics -- never a
    single fold's in-sample fit. `sector_lookup` (optional) tags trades for
    compute_cluster_weights(). `strategy` (see run_backtest) selects "rsi"
    (default) or "breakout".

    Folds are fully independent (no shared state, no data dependency
    between them), so by default (`parallel=True`) this dispatches them
    across a process pool instead of looping sequentially -- this was the
    dominant cost of every Optuna search this session (each trial walks
    every fold, for every ticker, day by day). The pool's initializer loads
    ticker_data/market_data into each worker ONCE when the pool starts, not
    once per fold, so only the lightweight per-fold Fold/config objects
    actually cross the process boundary on each task. Results are returned
    in the same order as `folds` regardless of completion order. Pass
    `parallel=False` to force the old sequential behavior (e.g. for a
    direct correctness comparison, or if a single fold makes pool startup
    overhead not worth it -- also skipped automatically for <=1 fold).

    `max_workers` defaults to cpu_count()-4 (min 1), not the full core
    count -- sustained all-core usage during a long search measurably
    raises CPU temperature, and this leaves headroom for the OS/other apps
    rather than assuming exclusive use of the machine. Pass an explicit
    value (e.g. your full `os.cpu_count()`) for maximum speed if that
    tradeoff is fine for your setup, or 1 for the gentlest possible
    (though `parallel=False` is cheaper than a 1-worker pool for that).

    `sector_data` (optional, sector NAME -> OHLCV, see watchlist.SECTOR_ETF)
    backs the backtest/Optuna-only Sector_Relative_Strength filter
    (improvements.txt items 68/70) -- threaded through to run_backtest() on
    both the sequential and parallel paths (loaded once per worker process
    via the pool initializer on the parallel path, same treatment as
    ticker_data/market_data). None (default) means no sector data at all,
    identical to every caller from before this parameter existed.

    `pair_price_panels` (optional, sector NAME -> wide Close-price
    DataFrame of every ticker in that sector) backs the "pairs" strategy's
    partner-selection/spread-zscore mechanism (improvements.txt item 83) --
    threaded through identically to sector_data above, both paths. None
    (default) means no peer data at all, identical to every caller from
    before this parameter existed.

    `momentum_rank_frame` (optional, a wide DataFrame -- dates x tickers --
    of every ticker's own trailing-return percentile, see
    swingtrade.levels.compute_momentum_rank_frame()) backs the
    "momentum_rank" strategy -- threaded through BOTH paths (sequential
    directly, parallel via the pool initializer's `initargs`, same
    treatment as pair_price_panels) since, unlike insider_data, this
    strategy's entire signal depends on it -- forgetting the parallel
    thread would mean every Optuna trial run with `parallel=True` silently
    sees no percentile data and produces zero signals. None (default) means
    no percentile data at all, identical to every caller from before this
    parameter existed.

    `yield_curve` (optional, a single date-indexed Series shared across
    every ticker unchanged, see market_data.fetch_fred_series()) backs
    "ma_crossover"'s BACKTEST/OPTUNA-ONLY ma_crossover_yield_curve_spread_max
    filter -- threaded through BOTH paths identically to sector_data/
    pair_price_panels (loaded once per worker via the pool initializer on
    the parallel path). None (default) means no yield-curve data at all,
    identical to every caller from before this parameter existed.

    `skew_regime` (optional, a single date-indexed Series of RAW CBOE SKEW
    Index closes shared across every ticker unchanged, see
    run_backtest.fetch_history() called on ticker "^SKEW") backs
    "ma_crossover"'s BACKTEST/OPTUNA-ONLY ma_crossover_skew_regime_min
    filter -- threaded through BOTH paths identically to yield_curve above.
    None (default) means no SKEW data at all, identical to every caller
    from before this parameter existed."""
    if not parallel or len(folds) <= 1:
        return [
            _run_fold_sequential(
                fold, ticker_data, market_data, config, earnings_data, sector_lookup, strategy, sector_data,
                pair_price_panels, momentum_rank_frame, yield_curve, skew_regime,
            )
            for fold in folds
        ]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 4)

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker,
        initargs=(
            ticker_data, market_data, sector_data, pair_price_panels, momentum_rank_frame, yield_curve,
            skew_regime,
        ),
    ) as executor:
        futures = [
            executor.submit(_run_fold_worker, fold, config, earnings_data, sector_lookup, strategy)
            for fold in folds
        ]
        return [f.result() for f in futures]
