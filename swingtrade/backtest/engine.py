"""Single-window backtest engine: run_backtest() and run_random_backtest()."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .signals_technical import (
    simulate_breakout_retest_signals,
    simulate_breakout_signals,
    simulate_momentum_burst_signals,
    simulate_pullback_signals,
    simulate_random_breakout_entries,
    simulate_random_breakout_retest_entries,
    simulate_random_entries,
    simulate_random_momentum_burst_entries,
    simulate_random_pullback_entries,
    simulate_random_squeeze_breakout_entries,
    simulate_random_week52_entries,
    simulate_signals,
    simulate_squeeze_breakout_signals,
    simulate_week52_signals,
)
from .signals_cross_sectional import (
    simulate_lowvol_signals,
    simulate_momentum_signals,
    simulate_pairs_signals,
    simulate_random_lowvol_entries,
    simulate_random_momentum_entries,
    simulate_random_pairs_entries,
    simulate_random_sector_rotation_entries,
    simulate_sector_rotation_signals,
)
from .signals_fundamental import (
    simulate_accruals_signals,
    simulate_analyst_revision_signals,
    simulate_cash_profitability_signals,
    simulate_insider_buying_signals,
    simulate_pead_signals,
    simulate_quality_signals,
    simulate_random_accruals_entries,
    simulate_random_analyst_revision_entries,
    simulate_random_cash_profitability_entries,
    simulate_random_insider_buying_entries,
    simulate_random_pead_entries,
    simulate_random_quality_entries,
    simulate_random_value_entries,
    simulate_value_signals,
)
from .signals_other import (
    simulate_adx_trend_entry_signals,
    simulate_ma_crossover_signals,
    simulate_random_adx_trend_entry_entries,
    simulate_random_ma_crossover_entries,
)


def run_backtest(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    sector_lookup: dict[str, str] | None = None,
    strategy: str = "rsi",
    sector_data: dict[str, pd.DataFrame] | None = None,
    pair_price_panels: dict[str, pd.DataFrame] | None = None,
    insider_data: dict[str, pd.DataFrame] | None = None,
    momentum_rank_frame: pd.DataFrame | None = None,
    lowvol_rank_frame: pd.DataFrame | None = None,
    sector_rotation_rank_frame: pd.DataFrame | None = None,
    quality_rank_frame: pd.DataFrame | None = None,
    value_rank_frame: pd.DataFrame | None = None,
    accruals_rank_frame: pd.DataFrame | None = None,
    cash_profitability_rank_frame: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
    earnings_surprise_data: dict[str, pd.DataFrame] | None = None,
    analyst_revision_data: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Simulate signals for every ticker in ticker_data over
    [window_start, window_end), settling each against its own subsequent
    history. Returns the combined trade list. `earnings_data` (optional,
    ticker -> tz-aware UTC DatetimeIndex, see run_backtest.fetch_earnings_dates)
    enables honest Catalyst_Warning simulation for "rsi", "squeeze_breakout",
    and "ma_crossover" -- without it, every trade for those three has
    Catalyst_Warning=False. `sector_lookup`
    (optional, ticker -> sector, see watchlist.read_ticker_sectors) tags
    each trade for compute_cluster_weights(). `strategy` selects which
    signal to simulate -- "rsi" (simulate_signals, mean-reversion,
    default), "breakout" (simulate_breakout_signals, trend-following;
    earnings_data is ignored, it has no catalyst concept), "pullback"
    (simulate_pullback_signals, trend-following pullback entry; earnings_data
    also ignored, same reasoning), "breakout_retest"
    (simulate_breakout_retest_signals, pullback to a recent breakout's
    level; earnings_data also ignored, same reasoning), "week52_high"
    (simulate_week52_signals, near a trailing 52-week high; earnings_data
    also ignored, same reasoning), "momentum_burst"
    (simulate_momentum_burst_signals, single-day price+volume burst;
    earnings_data also ignored, same reasoning), "squeeze_breakout"
    (simulate_squeeze_breakout_signals, volatility contraction followed
    by expansion -- earnings_data now honored, see improvements.txt for
    the gap this closes), "adx_trend_entry" (simulate_adx_trend_entry_signals,
    ADX-confirmed trending state; earnings_data still ignored), or
    "ma_crossover" (simulate_ma_crossover_signals, short/long SMA crossover
    -- earnings_data now honored, same as squeeze_breakout).

    `sector_data` (optional, sector NAME -> that sector's own OHLCV, e.g. a
    real SPDR ETF like XLK for Technology -- see watchlist.SECTOR_ETF) backs
    the backtest/Optuna-only Sector_Relative_Strength filter (improvements.txt
    items 68/70) for "breakout"/"squeeze_breakout"/"ma_crossover" only --
    resolved per ticker via `sector_lookup`. `None` (default) means no
    sector data at all -- Sector_Relative_Strength then reads None/NaN and
    every *_sector_relative_strength_min gate never excludes on its own,
    same convention as every other optional filter.

    `pair_price_panels` (optional, sector NAME -> wide Close-price
    DataFrame of every ticker in that sector, same shape/resolution
    convention as `sector_data` above) backs the "pairs" strategy's
    partner-selection/spread-zscore mechanism -- see
    swingtrade/levels.precompute_pairs_frame(). Resolved per ticker into
    `peer_prices` by dropping that ticker's own column from its sector's
    panel. `None` (default) means no peer data at all -- Pair_Signal then
    always reads False, same "missing optional data never fabricates a
    signal" convention as sector_data.

    `insider_data` (optional, ticker -> run_backtest.fetch_insider_purchases()'s
    own per-ticker output) backs the "insider_buying" strategy -- resolved
    per ticker directly (no cross-ticker panel needed, unlike pairs). `None`
    (default) means no purchase data at all -- Insider_Buy_Signal then
    always reads False, same convention as pair_price_panels/sector_data.
    NOTE: not yet threaded through run_walk_forward()'s multiprocessing
    path (ProcessPoolExecutor initargs) -- only this single-call sequential
    path -- since no Optuna tuning pass for this strategy exists yet; add
    that threading if/when one does.

    `momentum_rank_frame` (optional, a wide DataFrame -- dates x tickers --
    of every ticker's own trailing-return percentile, see
    swingtrade.levels.compute_momentum_rank_frame()) backs the
    "momentum_rank" strategy. Unlike `pair_price_panels` (sector-
    partitioned, resolved per ticker by dropping that ticker's own column),
    this is ONE universe-wide frame -- resolved per ticker by slicing out
    that ticker's own column (a plain Series), since the cross-ticker
    ranking work is already done by the caller. `None` (default) means no
    percentile data at all -- Momentum_Signal then always reads False, same
    "missing optional data never fabricates a signal" convention as
    sector_data/pair_price_panels. THREADED THROUGH run_walk_forward()'s
    multiprocessing path (ProcessPoolExecutor initargs) from day one -- see
    the insider_data note immediately above for why forgetting this for a
    strategy whose ENTIRE signal depends on the cross-ticker data would be
    a silent, not-a-crash failure in every parallel Optuna trial.

    `yield_curve` (optional, a single date-indexed Series shared across
    EVERY ticker unchanged -- see market_data.fetch_fred_series(),
    currently only ever T10Y2Y) backs "ma_crossover"'s BACKTEST/OPTUNA-ONLY
    ma_crossover_yield_curve_spread_max filter (see config.py's own
    comment on that field). Unlike sector_data/pair_price_panels/
    momentum_rank_frame, this is NOT resolved per-ticker at all -- there's
    no per-sector or per-ticker lookup, the same Series is passed straight
    through to every ticker's own simulate_ma_crossover_signals() call.
    `None` (default) means no yield-curve data at all -- Yield_Curve_Spread
    then reads None/NaN and the gate never excludes on its own, same
    convention as every other optional filter here.

    `skew_regime` (optional, a single date-indexed Series of RAW CBOE SKEW
    Index closes shared across EVERY ticker unchanged -- see
    run_backtest.fetch_history() called on ticker "^SKEW") backs
    "ma_crossover"'s BACKTEST/OPTUNA-ONLY ma_crossover_skew_regime_min
    filter -- same "not resolved per-ticker, passed straight through"
    treatment as `yield_curve` above. `None` (default) means no SKEW data
    at all -- Skew_Regime_Diff then reads None/NaN and the gate never
    excludes on its own, same convention as every other optional filter
    here.

    `earnings_surprise_data` (optional, ticker ->
    run_backtest.fetch_earnings_surprises()'s own per-ticker output) backs
    the "pead" strategy -- resolved per ticker directly, same shape/
    resolution convention as `insider_data`. `None` (default) means no
    surprise data at all -- PEAD_Signal then always reads False, same
    convention as insider_data/pair_price_panels/sector_data. NOTE: same
    limitation as insider_data -- not yet threaded through
    run_walk_forward()'s multiprocessing path, only this single-call
    sequential path, since no Optuna tuning pass for this strategy exists
    yet either.

    `lowvol_rank_frame` (optional, a wide DataFrame -- dates x tickers --
    of every ticker's own trailing-volatility percentile, INVERTED so LOW
    volatility reads as a HIGH percentile, see
    swingtrade.levels.compute_lowvol_rank_frame()) backs the "lowvol_rank"
    strategy -- same resolution convention as `momentum_rank_frame`
    (universe-wide, sliced per ticker). `None` (default) means no
    percentile data at all -- LowVol_Signal then always reads False, same
    convention as momentum_rank_frame. NOTE: same limitation as
    insider_data/earnings_surprise_data -- not yet threaded through
    run_walk_forward()'s multiprocessing path, only this single-call
    sequential path, since no Optuna tuning pass for this strategy exists
    yet either.

    `sector_rotation_rank_frame` (optional, a wide DataFrame -- dates x
    SECTOR NAMES, not tickers -- of every GICS sector ETF's own trailing-
    return percentile among the OTHER sectors, see
    swingtrade.levels.compute_sector_rotation_rank_frame()) backs the
    "sector_rotation" strategy. UNLIKE `momentum_rank_frame`/
    `lowvol_rank_frame` (resolved by TICKER), this is resolved by the
    ticker's own SECTOR via `sector_lookup` -- every ticker in the same
    sector shares the identical column. `None` (default) means no
    percentile data at all -- SectorRotation_Signal then always reads
    False, same convention as momentum_rank_frame. NOTE: same limitation as
    momentum_rank_frame's own insider_data/earnings_surprise_data note --
    not yet threaded through run_walk_forward()'s multiprocessing path,
    only this single-call sequential path.

    `quality_rank_frame` (optional, a wide DataFrame -- dates x tickers --
    of every ticker's own point-in-time ROE percentile, see
    swingtrade.levels.compute_quality_rank_frame() and
    sec_fundamentals.build_roe_panel() for the real data source -- SEC
    EDGAR's free XBRL API, not yfinance) backs the "quality_rank" strategy
    -- same resolution convention as `momentum_rank_frame` (universe-wide,
    sliced per ticker). `None` (default) means no ROE data at all --
    Quality_Signal then always reads False, same convention as
    momentum_rank_frame. NOTE: same limitation as insider_data/
    earnings_surprise_data -- not yet threaded through run_walk_forward()'s
    multiprocessing path, only this single-call sequential path.

    `value_rank_frame` (optional, a wide DataFrame -- dates x tickers -- of
    every ticker's own Price-to-Book percentile, INVERTED so the CHEAPEST
    ratio reads as the HIGHEST percentile, see
    swingtrade.levels.compute_value_rank_frame() and
    sec_fundamentals.build_book_value_per_share_panel() for the real data
    source) backs the "value_rank" strategy -- same resolution convention
    as `quality_rank_frame` (universe-wide, sliced per ticker). `None`
    (default) means no data at all -- Value_Signal then always reads
    False, same convention as quality_rank_frame. NOTE: same limitation as
    quality_rank_frame -- not yet threaded through run_walk_forward()'s
    multiprocessing path, only this single-call sequential path.

    `analyst_revision_data` (optional, ticker ->
    run_backtest.fetch_analyst_revisions()'s own per-ticker output) backs
    the "analyst_revision" strategy -- resolved per ticker directly (no
    cross-ticker panel needed), same shape/resolution convention as
    `insider_data`. `None` (default) means no revision data at all --
    Analyst_Revision_Signal then always reads False, same convention as
    insider_data/earnings_surprise_data. NOTE: same limitation as
    insider_data -- not yet threaded through run_walk_forward()'s
    multiprocessing path, only this single-call sequential path.

    `accruals_rank_frame` (optional, a wide DataFrame -- dates x tickers --
    of every ticker's own accruals percentile, INVERTED so the LOWEST
    accruals reads as the HIGHEST percentile, see
    swingtrade.levels.compute_accruals_rank_frame() and
    sec_fundamentals.build_accruals_panel() for the real data source) backs
    the "accruals_rank" strategy -- same resolution convention as
    `quality_rank_frame`/`value_rank_frame` (universe-wide, sliced per
    ticker). `None` (default) means no data at all -- Accruals_Signal then
    always reads False. NOTE: same limitation as quality_rank_frame/
    value_rank_frame -- not yet threaded through run_walk_forward()'s
    multiprocessing path, only this single-call sequential path."""
    earnings_data = earnings_data or {}
    sector_lookup = sector_lookup or {}
    sector_data = sector_data or {}
    pair_price_panels = pair_price_panels or {}
    insider_data = insider_data or {}
    earnings_surprise_data = earnings_surprise_data or {}
    analyst_revision_data = analyst_revision_data or {}
    all_trades = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        sector_ohlcv = sector_data.get(sector)
        pair_panel = pair_price_panels.get(sector)
        peer_prices = (
            pair_panel.drop(columns=[ticker]) if pair_panel is not None and ticker in pair_panel.columns else None
        )
        insider_purchases = insider_data.get(ticker)
        earnings_surprises = earnings_surprise_data.get(ticker)
        analyst_revisions = analyst_revision_data.get(ticker)
        rank_column = (
            momentum_rank_frame[ticker]
            if momentum_rank_frame is not None and ticker in momentum_rank_frame.columns
            else None
        )
        lowvol_rank_column = (
            lowvol_rank_frame[ticker]
            if lowvol_rank_frame is not None and ticker in lowvol_rank_frame.columns
            else None
        )
        sector_rotation_rank_column = (
            sector_rotation_rank_frame[sector]
            if sector_rotation_rank_frame is not None and sector in sector_rotation_rank_frame.columns
            else None
        )
        quality_rank_column = (
            quality_rank_frame[ticker]
            if quality_rank_frame is not None and ticker in quality_rank_frame.columns
            else None
        )
        value_rank_column = (
            value_rank_frame[ticker]
            if value_rank_frame is not None and ticker in value_rank_frame.columns
            else None
        )
        accruals_rank_column = (
            accruals_rank_frame[ticker]
            if accruals_rank_frame is not None and ticker in accruals_rank_frame.columns
            else None
        )
        cash_profitability_rank_column = (
            cash_profitability_rank_frame[ticker]
            if cash_profitability_rank_frame is not None and ticker in cash_profitability_rank_frame.columns
            else None
        )
        if strategy == "rsi":
            trades = simulate_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "breakout":
            trades = simulate_breakout_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                sector=sector, sector_ohlcv=sector_ohlcv,
            )
        elif strategy == "pullback":
            trades = simulate_pullback_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "breakout_retest":
            trades = simulate_breakout_retest_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "week52_high":
            trades = simulate_week52_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "momentum_burst":
            trades = simulate_momentum_burst_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "squeeze_breakout":
            trades = simulate_squeeze_breakout_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, sector_ohlcv=sector_ohlcv,
            )
        elif strategy == "adx_trend_entry":
            trades = simulate_adx_trend_entry_signals(
                ticker, ohlcv, market_data, window_start, window_end, config, sector=sector,
            )
        elif strategy == "ma_crossover":
            trades = simulate_ma_crossover_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, sector_ohlcv=sector_ohlcv,
                yield_curve=yield_curve, skew_regime=skew_regime,
            )
        elif strategy == "pairs":
            trades = simulate_pairs_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, peer_prices=peer_prices,
            )
        elif strategy == "insider_buying":
            trades = simulate_insider_buying_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, insider_purchases=insider_purchases,
            )
        elif strategy == "momentum_rank":
            trades = simulate_momentum_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=rank_column,
            )
        elif strategy == "pead":
            trades = simulate_pead_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, earnings_surprises=earnings_surprises,
            )
        elif strategy == "lowvol_rank":
            trades = simulate_lowvol_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=lowvol_rank_column,
            )
        elif strategy == "sector_rotation":
            trades = simulate_sector_rotation_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=sector_rotation_rank_column,
            )
        elif strategy == "quality_rank":
            trades = simulate_quality_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=quality_rank_column,
            )
        elif strategy == "value_rank":
            trades = simulate_value_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=value_rank_column,
            )
        elif strategy == "analyst_revision":
            trades = simulate_analyst_revision_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, analyst_revisions=analyst_revisions,
            )
        elif strategy == "accruals_rank":
            trades = simulate_accruals_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=accruals_rank_column,
            )
        elif strategy == "cash_profitability_rank":
            trades = simulate_cash_profitability_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker), sector=sector, rank_column=cash_profitability_rank_column,
            )
        else:
            raise ValueError(
                f"unknown strategy: {strategy!r} "
                "(expected 'rsi', 'breakout', 'pullback', 'breakout_retest', 'week52_high', "
                "'momentum_burst', 'squeeze_breakout', 'adx_trend_entry', 'ma_crossover', 'pairs', "
                "'insider_buying', 'momentum_rank', 'pead', 'lowvol_rank', 'sector_rotation', "
                "'quality_rank', 'value_rank', 'analyst_revision', 'accruals_rank', or 'cash_profitability_rank')"
            )
        all_trades.extend(trades)
    return all_trades


def run_random_backtest(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    window_start,
    window_end,
    real_trade_counts: dict[str, int],
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    sector_lookup: dict[str, str] | None = None,
    strategy: str = "rsi",
) -> list[dict]:
    """Matched-count RANDOM-entry counterpart to run_backtest() -- the ONE
    shared dispatch table for every simulate_random_*_entries() function,
    reused by both benchmark_random_entry.py's own real-vs-random
    comparison and optimize.py's automatic real-vs-random ratio reporting
    (2026-08-24), so this mapping is never duplicated a third time. This
    codebase's own history has already been bitten twice by exactly this
    class of bug (ingest.py's and dip_buy_analyzer.py's independently
    hand-maintained _score_for_strategy() copies drifting out of sync when
    a new strategy was added) -- a shared, single source of truth here is
    a deliberate fix, not an accident of convenience.

    For each ticker, fires exactly `real_trade_counts[ticker]` randomly-
    chosen entries (matching that ticker's own real signal count under the
    SAME config, so trade volume is never a confound) through the same
    macro-uptrend/liquidity gates and entry-fill/stop-target mechanics as
    the real strategy -- only WHICH DAY differs. `real_trade_counts` should
    come from a prior run_backtest() call's own trade list (grouped by
    ticker), same "count matched, single fixed seed reused per caller"
    convention benchmark_random_entry.py's own loop already establishes.

    Deliberately does NOT accept `sector_data`/`pair_price_panels`/
    `momentum_rank_frame`/`insider_data` -- every simulate_random_*_entries()
    function is deliberately called with peer_prices=None/rank_column=None/
    insider_purchases=None/sector_ohlcv=None itself (confirmed directly
    against benchmark_random_entry.py's own dispatch loop, which never
    passes these to a random_fn call either): the random baseline answers
    whether ENTRY TIMING adds value over the same underlying gates, not
    whether a cross-ticker/sector/fundamental filter helps -- see e.g.
    simulate_random_pairs_entries()'s own docstring for this exact
    reasoning. `earnings_data` IS still honored, for the same 5 strategies
    run_backtest() honors it for (squeeze_breakout/ma_crossover/pairs/
    insider_buying/momentum_rank) -- Catalyst_Warning is a property of the
    calendar date a trade lands on, not of which signal chose that date,
    so it applies identically to a random entry."""
    earnings_data = earnings_data or {}
    sector_lookup = sector_lookup or {}
    all_trades = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        n_trades = real_trade_counts.get(ticker, 0)
        if strategy == "rsi":
            trades = simulate_random_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "breakout":
            trades = simulate_random_breakout_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "pullback":
            trades = simulate_random_pullback_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "breakout_retest":
            trades = simulate_random_breakout_retest_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "week52_high":
            trades = simulate_random_week52_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "momentum_burst":
            trades = simulate_random_momentum_burst_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "squeeze_breakout":
            trades = simulate_random_squeeze_breakout_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "adx_trend_entry":
            trades = simulate_random_adx_trend_entry_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config, sector=sector,
            )
        elif strategy == "ma_crossover":
            trades = simulate_random_ma_crossover_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "pairs":
            trades = simulate_random_pairs_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "insider_buying":
            trades = simulate_random_insider_buying_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "momentum_rank":
            trades = simulate_random_momentum_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "pead":
            trades = simulate_random_pead_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "lowvol_rank":
            trades = simulate_random_lowvol_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "sector_rotation":
            trades = simulate_random_sector_rotation_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "quality_rank":
            trades = simulate_random_quality_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "value_rank":
            trades = simulate_random_value_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "analyst_revision":
            trades = simulate_random_analyst_revision_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "accruals_rank":
            trades = simulate_random_accruals_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        elif strategy == "cash_profitability_rank":
            trades = simulate_random_cash_profitability_entries(
                ticker, ohlcv, market_data, window_start, window_end, n_trades, rng, config,
                earnings_dates=earnings_data.get(ticker), sector=sector,
            )
        else:
            raise ValueError(
                f"unknown strategy: {strategy!r} "
                "(expected 'rsi', 'breakout', 'pullback', 'breakout_retest', 'week52_high', "
                "'momentum_burst', 'squeeze_breakout', 'adx_trend_entry', 'ma_crossover', 'pairs', "
                "'insider_buying', 'momentum_rank', 'pead', 'lowvol_rank', 'sector_rotation', "
                "'quality_rank', 'value_rank', 'analyst_revision', 'accruals_rank', or 'cash_profitability_rank')"
            )
        all_trades.extend(trades)
    return all_trades
