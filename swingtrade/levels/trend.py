"""ADX-trend and MA-crossover level calculators."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .technical import precompute_breakout_frame


def precompute_adx_trend_entry_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every column adx_trend_entry_levels_from_frame()
    needs -- built ON TOP of precompute_breakout_frame() (reused wholesale:
    ADX/SMA_TREND/RSI/ATR/AvgVolume/etc. are identical to what breakout
    already computes -- ADX in particular is the SAME column breakout's
    own optional breakout_adx_min filter already uses, not recomputed
    here).

    Adds one new column: Short_MA = a short-term rolling mean of Close
    (config.adx_trend_entry_ma_window) -- used only for directional
    confirmation (ADX itself measures trend STRENGTH, not direction)."""
    df = precompute_breakout_frame(df, config, market_df=market_df)
    df["Short_MA"] = df["Close"].rolling(window=config.adx_trend_entry_ma_window).mean()
    return df


def adx_trend_entry_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_adx_trend_entry_levels()'s dict for one row of a
    frame already built by precompute_adx_trend_entry_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Business logic is verbatim compute_adx_trend_entry_levels(), just
    reading from a precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi, adx, short_ma, last_volume, avg_volume_prior = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"],
        last_row["RSI"], last_row["ADX"], last_row["Short_MA"],
        last_row["Volume"], last_row["AvgVolume_Prior"],
    )
    obv_zscore, squeeze_zscore = last_row["OBV_Zscore"], last_row["Squeeze_Zscore"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(adx):
        raise RuntimeError(f"insufficient history to compute {config.adx_window}-day ADX")
    if pd.isna(short_ma):
        raise RuntimeError(f"insufficient history to compute {config.adx_trend_entry_ma_window}-day short MA")
    # RSI is informational only for adx_trend_entry's CORE signal (not used
    # for gating there) -- same treatment as every other trend-following
    # strategy. It IS used by the optional Phase 2 filters below (applied
    # downstream in add_adx_trend_entry_trade_score()/
    # simulate_adx_trend_entry_signals(), same "informational unless the
    # filter is explicitly enabled" pattern breakout's own six filters use).
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    # Same informational treatment for the volume ratio (Phase 2 filter --
    # see breakout_levels_from_frame's identical inline computation).
    if pd.isna(avg_volume_prior) or float(avg_volume_prior) == 0:
        volume_ratio = None
    else:
        volume_ratio = round(float(last_volume) / float(avg_volume_prior), 3)
    # Same informational treatment for OBV/squeeze z-scores (Phase 2 filters).
    obv_zscore = None if pd.isna(obv_zscore) else round(float(obv_zscore), 3)
    squeeze_zscore = None if pd.isna(squeeze_zscore) else round(float(squeeze_zscore), 3)

    last_close, sma_trend, atr, avg_volume, adx, short_ma = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume), float(adx), float(short_ma),
    )

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    adx_trend_signal = bool(adx >= config.adx_trend_entry_threshold and last_close > short_ma)

    # Buy_Price = today's own Close -- this is a continuous-STATE
    # confirmation (same convention as week52_high/momentum_burst/
    # squeeze_breakout), not a specific price level to wait for.
    # Distance_to_Buy_Pct is therefore always 0 by construction, same
    # uninformative-but-schema-compatible treatment those strategies
    # already established.
    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    # Relative_Strength (Phase 2 filter) needs market_df -- only present
    # in the frame if precompute_adx_trend_entry_frame() was called with
    # it (see breakout_levels_from_frame's identical pattern).
    relative_strength = None
    if "Relative_Strength" in frame.columns:
        rs_val = last_row["Relative_Strength"]
        if pd.notna(rs_val):
            relative_strength = round(float(rs_val), 4)

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "ADX": round(adx, 2),
        "Short_MA": round(short_ma, 2),
        "ADX_Trend_Signal": adx_trend_signal,
        "Relative_Strength": relative_strength,
        "Volume_Ratio": volume_ratio,
        "OBV_Zscore": obv_zscore,
        "Squeeze_Zscore": squeeze_zscore,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        # How far ADX clears its own minimum "trending" bar -- see
        # add_adx_trend_entry_trade_score, replaces Distance_to_Buy_Pct
        # (always 0 here) as the score's ticker-differentiating term.
        "Signal_Strength_Pct": round(adx - config.adx_trend_entry_threshold, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_adx_trend_entry_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
) -> dict:
    """Compute ADX-trend-entry levels for one ticker's OHLCV history -- a
    ninth strategy, a continuous STATE (like week52_high/squeeze_breakout,
    not a discrete event): fires whenever ADX is at/above
    adx_trend_entry_threshold (genuinely trending, independent of
    direction) AND price is above a short-term MA (direction
    confirmation). Core trigger stayed lean through Phase 1 validation
    (see improvements.txt item 40 -- real, if modest, edge); Phase 2 (same
    item) added the SAME family of optional "sharpening" filters breakout
    (v19) itself accumulated over its own real history -- RSI-overbought,
    Relative_Strength, Volume_Ratio, OBV_Zscore, Squeeze_Zscore -- all
    disabled by default (0/off = today's exact Phase-1 behavior), applied
    as ADDITIONAL gates downstream in add_adx_trend_entry_trade_score()/
    simulate_adx_trend_entry_signals(), NOT baked into ADX_Trend_Signal
    itself, mirroring breakout's own architecture exactly (see
    breakout_levels_from_frame()/add_breakout_trade_score()).

    Buy_Price is today's own Close (see adx_trend_entry_levels_from_frame's
    inline comment for why -- same convention as week52_high/momentum_burst/
    squeeze_breakout, NOT a resting level to wait for). `market_df`
    (optional), if given, enables the Relative_Strength filter -- same
    role as compute_breakout_levels()'s own `market_df` parameter.

    The returned dict is schema-compatible with every other strategy's
    (Buy_Price, Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct,
    Catalyst_Warning, etc. all present) -- only add_adx_trend_entry_trade_score
    (swingtrade/scoring.py) differs downstream. RSI/Relative_Strength/
    Volume_Ratio/OBV_Zscore/Squeeze_Zscore are informational unless their
    matching filter field is changed from its disabled default.

    Thin wrapper over precompute_adx_trend_entry_frame()/adx_trend_entry_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard, matching
    every other strategy's same rationale. NOT called from ingest.py in
    this v1 -- this strategy is dashboard-only/experimental until it
    passes the same random-entry-timing validation every other strategy
    here was held to (see benchmark_random_entry.py).
    """
    frame = precompute_adx_trend_entry_frame(df, config, market_df=market_df)
    as_of = frame.index[-1]
    return adx_trend_entry_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


SKEW_REGIME_ROLLING_WINDOW_DAYS = 365  # ~1 trading year -- see precompute_ma_crossover_frame()'s


                                        # own docstring for why this must be a BOUNDED window,
                                        # not an expanding-since-1990 one (real secular-drift bug
                                        # caught in benchmark_skew_regime.py's own smoke test).


def precompute_ma_crossover_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every column ma_crossover_levels_from_frame()
    needs -- built ON TOP of precompute_breakout_frame() wholesale, like
    every other strategy this session (SMA_TREND/RSI/ATR/AvgVolume/ADX/
    OBV_Zscore/Squeeze_Zscore all reused for free).

    Adds two new columns (short/long SMA) plus their own .shift(1)
    counterparts -- MA_Short_Prev/MA_Long_Prev are needed to detect the
    ACTUAL crossover day (short was <= long yesterday, short > long today),
    not just "short is currently above long," which would fire on every
    day the relationship holds rather than only the day it changes.

    `yield_curve` (optional, a single date-indexed Series shared across
    every ticker -- see market_data.fetch_fred_series(), currently only
    ever T10Y2Y) backs the BACKTEST/OPTUNA-ONLY
    ma_crossover_yield_curve_spread_max filter (see that field's own
    comment in config.py for the real finding motivating it). Forward-
    filled onto `df`'s own trading-day index first (FRED's own calendar
    doesn't align to trading days -- weekends/holidays would otherwise
    read NaN), then `.shift(1)`'d -- the same no-look-ahead discipline
    every other indicator here uses, and the same "strictly BEFORE
    signal_date" convention benchmark_macro_regime.py's own
    _regime_as_of() already established, so a live signal can never read
    a same-day-or-later FRED observation. `Yield_Curve_Spread` is NaN
    wherever no yield_curve observation exists yet, same missing-data
    convention Sector_Relative_Strength already follows -- never excludes
    a signal on its own (see the scoring/backtest gates, which both treat
    NaN as "unavailable, don't exclude").

    `skew_regime` (optional, a single date-indexed Series of RAW CBOE SKEW
    Index closes shared across every ticker, see run_backtest.fetch_history()
    called on ticker "^SKEW") backs the BACKTEST/OPTUNA-ONLY
    ma_crossover_skew_regime_min filter (see that field's own comment in
    config.py for the real finding motivating it, benchmark_skew_regime.py).
    Unlike Yield_Curve_Spread (a raw level passthrough), this computes a
    ROLLING-RELATIVE value: `Skew_Regime_Diff = today's ^SKEW minus its own
    trailing SKEW_REGIME_ROLLING_WINDOW_DAYS median` -- ^SKEW has no stable
    fixed threshold the way T10Y2Y has a theory-driven zero (real secular
    drift over its own 36-year history means a fixed/expanding threshold
    would silently misclassify most of a live run, exactly the bug
    benchmark_skew_regime.py's own smoke test caught before its real
    validation run). The rolling median itself is computed on `skew_regime`'s
    OWN native daily index BEFORE reindexing onto `df` (so a ticker's
    occasional missing trading day never shrinks the window), then the
    resulting diff series is forward-filled onto `df`'s own index and
    `.shift(1)`'d -- identical no-look-ahead treatment to Yield_Curve_Spread
    above, both the window's own median and the value being classified are
    always computed from data strictly before the day being scored."""
    df = precompute_breakout_frame(df, config, market_df=market_df, sector_df=sector_df)
    df["MA_Short"] = df["Close"].rolling(window=config.ma_crossover_short_window).mean()
    df["MA_Long"] = df["Close"].rolling(window=config.ma_crossover_long_window).mean()
    df["MA_Short_Prev"] = df["MA_Short"].shift(1)
    df["MA_Long_Prev"] = df["MA_Long"].shift(1)
    if yield_curve is not None:
        df["Yield_Curve_Spread"] = yield_curve.reindex(df.index, method="ffill").shift(1)
    if skew_regime is not None:
        skew_diff = skew_regime - skew_regime.rolling(f"{SKEW_REGIME_ROLLING_WINDOW_DAYS}D").median()
        df["Skew_Regime_Diff"] = skew_diff.reindex(df.index, method="ffill").shift(1)
    return df


def ma_crossover_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_ma_crossover_levels()'s dict for one row of a frame
    already built by precompute_ma_crossover_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Business logic
    is verbatim compute_ma_crossover_levels(), just reading from a
    precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi, adx = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
        last_row["ADX"],
    )
    ma_short, ma_long, ma_short_prev, ma_long_prev = (
        last_row["MA_Short"], last_row["MA_Long"], last_row["MA_Short_Prev"], last_row["MA_Long_Prev"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(ma_short):
        raise RuntimeError(f"insufficient history to compute {config.ma_crossover_short_window}-day short MA")
    if pd.isna(ma_long):
        raise RuntimeError(f"insufficient history to compute {config.ma_crossover_long_window}-day long MA")
    if pd.isna(ma_short_prev) or pd.isna(ma_long_prev):
        raise RuntimeError("insufficient history: no prior day's MA values to detect a crossover")
    # RSI/ADX are informational only, same treatment as every other
    # trend-following strategy this session.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    adx = None if pd.isna(adx) else round(float(adx), 2)

    last_close, sma_trend, atr, avg_volume, ma_short, ma_long, ma_short_prev, ma_long_prev = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume),
        float(ma_short), float(ma_long), float(ma_short_prev), float(ma_long_prev),
    )

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    # The actual crossover event: short was AT OR BELOW long yesterday,
    # short is ABOVE long today -- fires exactly once per cross, not on
    # every subsequent day the short MA merely stays above the long one.
    crossover_signal = bool(ma_short_prev <= ma_long_prev and ma_short > ma_long)

    # Buy_Price = today's own Close -- same "already happening, buy near
    # the current price" convention week52_high/momentum_burst/
    # squeeze_breakout/adx_trend_entry all use for a same-day trigger, not
    # a resting level to wait for. Distance_to_Buy_Pct is therefore always
    # 0 by construction, same schema-compatible-but-uninformative
    # treatment those strategies already established.
    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    # How far the crossover's own gap (short MA minus long MA, as a % of
    # price) clears zero -- replaces Distance_to_Buy_Pct (always 0 here)
    # as the score's ticker-differentiating term, same pattern
    # momentum_burst/squeeze_breakout/adx_trend_entry already established.
    signal_strength_pct = round((ma_short - ma_long) / last_close * 100, 3)

    # Sector_Relative_Strength (backtest/Optuna-only) -- ma_crossover's
    # first optional filter field of any kind; same graceful missing-
    # column/NaN treatment breakout/squeeze_breakout's own use.
    sector_relative_strength = None
    if "Sector_Relative_Strength" in frame.columns:
        srs_val = last_row["Sector_Relative_Strength"]
        if pd.notna(srs_val):
            sector_relative_strength = round(float(srs_val), 4)

    # Yield_Curve_Spread (backtest/Optuna-only, see config.py's
    # ma_crossover_yield_curve_spread_max) -- same graceful missing-
    # column/NaN treatment as Sector_Relative_Strength above.
    yield_curve_spread = None
    if "Yield_Curve_Spread" in frame.columns:
        ycs_val = last_row["Yield_Curve_Spread"]
        if pd.notna(ycs_val):
            yield_curve_spread = round(float(ycs_val), 4)

    # Skew_Regime_Diff (backtest/Optuna-only, see config.py's
    # ma_crossover_skew_regime_min) -- same graceful missing-column/NaN
    # treatment as Sector_Relative_Strength/Yield_Curve_Spread above.
    skew_regime_diff = None
    if "Skew_Regime_Diff" in frame.columns:
        srd_val = last_row["Skew_Regime_Diff"]
        if pd.notna(srd_val):
            skew_regime_diff = round(float(srd_val), 4)

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "ADX": adx,
        "Sector_Relative_Strength": sector_relative_strength,
        "Yield_Curve_Spread": yield_curve_spread,
        "Skew_Regime_Diff": skew_regime_diff,
        "MA_Short": round(ma_short, 2),
        "MA_Long": round(ma_long, 2),
        "MA_Crossover_Signal": crossover_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Dollar_Volume": round(dollar_volume, 2),
    }


def compute_ma_crossover_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
) -> dict:
    """Compute moving-average-crossover levels for one ticker's OHLCV
    history -- a genuinely different mechanical trigger from every
    strategy tried in this project: fires the day a short-term SMA
    crosses ABOVE a long-term SMA (trend CONFIRMATION via relative
    moving-average positioning), not a price-level breakout, not
    RSI-based, not a volatility-regime shift, not a raw trend-strength
    threshold. See swingtrade/config.py's own comment on this strategy
    for the full motivation.

    MA_Crossover_Signal requires the short SMA to have been AT OR BELOW
    the long SMA yesterday and ABOVE it today -- a discrete event (like
    breakout), fires once per cross, not every day the relationship
    merely holds.

    Buy_Price is today's own Close (see ma_crossover_levels_from_frame's
    inline comment for why -- same convention as week52_high/
    momentum_burst/squeeze_breakout/adx_trend_entry).

    The returned dict is schema-compatible with every other strategy's
    (Buy_Price, Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct,
    Catalyst_Warning, etc. all present) -- only add_ma_crossover_trade_score
    (swingtrade/scoring.py) differs downstream. RSI is informational only,
    same treatment as the others.

    Thin wrapper over precompute_ma_crossover_frame()/
    ma_crossover_levels_from_frame() -- kept as a single-call convenience
    for the live dashboard, matching every other strategy's same
    rationale. ma_crossover cleared its own random-entry-timing validation
    (benchmark_random_entry.py) and was promoted to the PRIMARY, real-capital
    strategy on 2026-08-16 (improvements.txt item 73) -- this function is
    what market_data.score_bundle_for_strategy() actually calls for every
    live "ma_crossover" scan (both the dashboard and ingest.py).

    `yield_curve` (optional, a single FRED T10Y2Y Series -- see
    market_data.fetch_ticker_bundle()) backs Yield_Curve_Spread /
    ma_crossover_yield_curve_spread_max (improvements.txt items 109/110),
    live as well as backtest/Optuna now that it's threaded here. Omitted
    (default None) means Yield_Curve_Spread reads None and the filter never
    excludes anything, same graceful-degradation convention as sector_df.
    """
    frame = precompute_ma_crossover_frame(
        df, config, market_df=market_df, sector_df=sector_df, yield_curve=yield_curve
    )
    as_of = frame.index[-1]
    return ma_crossover_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)
