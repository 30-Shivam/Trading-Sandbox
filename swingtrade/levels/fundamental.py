"""Fundamentals/event-driven level calculators: quality, value, accruals, cash profitability, insider, analyst revisions, PEAD."""

import warnings
import numpy as np
import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .technical import precompute_breakout_frame


def compute_quality_rank_frame(roe_panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own point-in-time
    ROE (Return on Equity) -- the fundamentals-data counterpart to
    compute_momentum_rank_frame()/compute_lowvol_rank_frame(), same
    "whole-universe operation done once" rationale. Structurally SIMPLER
    than either of those: ROE is already a LEVEL (this year's profitability
    reading), not a trailing return needing a pct_change(periods=lookback)
    step -- a straight cross-sectional rank of whatever `roe_panel` already
    contains on each day.

    `roe_panel` should be a wide DataFrame of point-in-time ROE, one column
    per ticker (see sec_fundamentals.build_roe_panel() for the real
    caller-side construction -- each cell already forward-filled from that
    ticker's own most recently FILED 10-K as of that day, never a future
    value). Returns a DataFrame the same shape, each cell = that ticker's
    ROE percentile among every OTHER ticker with a real value that same
    day (100 = highest ROE in the universe). A ticker with no SEC data at
    all (Canadian cross-listed names, fetch failures) reads NaN every day,
    same "missing data doesn't fabricate a signal" convention as
    compute_momentum_rank_frame()."""
    return roe_panel.rank(axis=1, pct=True) * 100


def precompute_quality_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional QUALITY/PROFITABILITY
    RANK strategy's columns -- built on top of precompute_breakout_frame()
    (reused wholesale for SMA_TREND/ATR/AvgVolume/etc., the same
    macro-uptrend/liquidity gates every strategy shares), exact structural
    mirror of precompute_momentum_frame()/precompute_lowvol_frame().

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_quality_rank_frame() over every ticker's point-in-time ROE.
    Missing (None) degrades to "never fires" rather than excluding the
    ticker outright, same convention as precompute_momentum_frame()."""
    df = precompute_breakout_frame(df, config)
    df["Quality_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def quality_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional QUALITY/PROFITABILITY RANK strategy's
    dict for one row of a frame already built by precompute_quality_frame()
    -- exact structural mirror of momentum_levels_from_frame()/
    lowvol_levels_from_frame(), differing only in which precomputed
    percentile column gates the signal. Same macro-uptrend/liquidity gates
    every strategy shares (via precompute_breakout_frame(), reused
    wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    quality_percentile = last_row["Quality_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    if pd.isna(quality_percentile):
        quality_signal = False
        quality_percentile_out = None
        signal_strength_pct = 0.0
    else:
        quality_percentile = float(quality_percentile)
        quality_signal = bool(quality_percentile >= config.quality_top_percentile_min)
        quality_percentile_out = round(quality_percentile, 2)
        signal_strength_pct = (
            round(quality_percentile - config.quality_top_percentile_min, 2) if quality_signal else 0.0
        )

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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Quality_Percentile": quality_percentile_out,
        "Quality_Signal": quality_signal,
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


def compute_quality_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional QUALITY/PROFITABILITY RANK levels for one
    ticker's OHLCV history -- see precompute_quality_frame()'s own
    docstring for the full mechanism. Thin wrapper, exact mirror of
    compute_momentum_levels()/compute_lowvol_levels()."""
    frame = precompute_quality_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return quality_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_value_rank_frame(price_panel: pd.DataFrame, bvps_panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own daily
    Price-to-Book ratio, INVERTED so the CHEAPEST ticker (lowest P/B) gets
    the HIGHEST percentile (100) -- the VALUE-factor counterpart to
    compute_quality_rank_frame()'s ROE-based ranking. Deliberately the
    opposite ranking direction from every other rank frame in this
    codebase: momentum/lowvol/quality all rank "biggest raw value = best,"
    value ranks "smallest raw ratio = best."

    `price_panel` (wide Close-price panel, e.g. market_data.build_momentum_panel())
    and `bvps_panel` (wide, already-forward-filled point-in-time Book Value
    Per Share, from sec_fundamentals.build_book_value_per_share_panel())
    must share the same tickers/columns -- aligned via pandas' own
    elementwise division on their shared index/columns, any ticker/date
    combination present in one but not the other reads NaN, same
    "missing data doesn't fabricate a signal" convention as every other
    optional per-ticker input. A ticker with no BVPS data at all (Canadian
    cross-listed names, fetch failures) reads NaN every day and is
    correctly excluded from the day's percentile computation entirely
    (pandas .rank() already skips NaN by construction)."""
    price_to_book = price_panel / bvps_panel
    # ascending=False means the LARGEST raw ratio (most expensive) gets
    # rank=1 (worst), and the SMALLEST raw ratio (cheapest) gets rank=N
    # (best) -- pct=True then turns that into cheapest=100%, most
    # expensive=~1/N, the same "best case reads near 100" convention
    # compute_quality_rank_frame() uses for its own (non-inverted) ROE rank.
    return price_to_book.rank(axis=1, pct=True, ascending=False) * 100


def precompute_value_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional VALUE RANK strategy's
    columns -- exact structural mirror of precompute_quality_frame(),
    differing only in which precomputed percentile column/label is used
    (`Value_Percentile` here, already inverted so higher=cheaper=better by
    the time it reaches this function -- see compute_value_rank_frame()).

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_value_rank_frame(). Missing (None) degrades to "never fires,"
    same convention as every sibling cross-sectional strategy."""
    df = precompute_breakout_frame(df, config)
    df["Value_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def value_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional VALUE RANK strategy's dict for one row
    of a frame already built by precompute_value_frame() -- exact
    structural mirror of quality_levels_from_frame(), differing only in
    which percentile column gates the signal."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    value_percentile = last_row["Value_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    if pd.isna(value_percentile):
        value_signal = False
        value_percentile_out = None
        signal_strength_pct = 0.0
    else:
        value_percentile = float(value_percentile)
        value_signal = bool(value_percentile >= config.value_top_percentile_min)
        value_percentile_out = round(value_percentile, 2)
        signal_strength_pct = (
            round(value_percentile - config.value_top_percentile_min, 2) if value_signal else 0.0
        )

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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Value_Percentile": value_percentile_out,
        "Value_Signal": value_signal,
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


def compute_value_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional VALUE RANK levels for one ticker's OHLCV
    history -- see precompute_value_frame()'s own docstring for the full
    mechanism. Thin wrapper, exact mirror of compute_quality_levels()."""
    frame = precompute_value_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return value_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_accruals_rank_frame(accruals_panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own point-in-time
    ACCRUALS ratio, INVERTED so the LOWEST accruals (earnings most backed
    by real cash flow, per Sloan 1996's classic anomaly) gets the HIGHEST
    percentile -- the accruals-factor counterpart to
    compute_value_rank_frame()'s own inverted Price-to-Book ranking, same
    "smallest raw value = best" direction, same rank(ascending=False)
    mechanism. See sec_fundamentals.fetch_point_in_time_accruals()'s own
    docstring for the full economic reasoning.

    `accruals_panel` should be a wide DataFrame of point-in-time accruals,
    one column per ticker (see sec_fundamentals.build_accruals_panel()).
    Returns a DataFrame the same shape, each cell = that ticker's accruals
    percentile among every OTHER ticker with a real value that same day
    (100 = lowest/best accruals in the universe). A ticker with no SEC
    data at all reads NaN every day, same "missing data doesn't fabricate
    a signal" convention as every sibling rank frame."""
    return accruals_panel.rank(axis=1, pct=True, ascending=False) * 100


def precompute_accruals_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional ACCRUALS RANK
    strategy's columns -- built on top of precompute_breakout_frame()
    (reused wholesale for SMA_TREND/ATR/AvgVolume/etc.), exact structural
    mirror of precompute_quality_frame() -- accruals, like ROE, is already
    a LEVEL (this year's ratio), not a trailing return, so this is a
    plain rank-column reindex, no lookback/pct_change math.

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_accruals_rank_frame(). Missing (None) degrades to "never
    fires," same convention as every sibling cross-sectional strategy."""
    df = precompute_breakout_frame(df, config)
    df["Accruals_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def accruals_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional ACCRUALS RANK strategy's dict for one
    row of a frame already built by precompute_accruals_frame() -- exact
    structural mirror of quality_levels_from_frame()."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    accruals_percentile = last_row["Accruals_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    if pd.isna(accruals_percentile):
        accruals_signal = False
        accruals_percentile_out = None
        signal_strength_pct = 0.0
    else:
        accruals_percentile = float(accruals_percentile)
        accruals_signal = bool(accruals_percentile >= config.accruals_top_percentile_min)
        accruals_percentile_out = round(accruals_percentile, 2)
        signal_strength_pct = (
            round(accruals_percentile - config.accruals_top_percentile_min, 2) if accruals_signal else 0.0
        )

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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Accruals_Percentile": accruals_percentile_out,
        "Accruals_Signal": accruals_signal,
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


def compute_accruals_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional ACCRUALS RANK levels for one ticker's OHLCV
    history -- see precompute_accruals_frame()'s own docstring for the
    full mechanism. Thin wrapper, exact mirror of compute_quality_levels()."""
    frame = precompute_accruals_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return accruals_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_cash_profitability_rank_frame(cash_profitability_panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own point-in-time
    CASH-BASED OPERATING PROFITABILITY (CFO/Assets) -- NOT inverted, same
    "highest raw value = best" direction as compute_quality_rank_frame()'s
    ROE ranking (unlike compute_accruals_rank_frame()'s/compute_value_rank_frame()'s
    inverted "lowest is best" convention). See
    sec_fundamentals.fetch_point_in_time_cash_profitability()'s own
    docstring for the full economic reasoning.

    `cash_profitability_panel` should be a wide DataFrame of point-in-time
    cash profitability, one column per ticker (see
    sec_fundamentals.build_cash_profitability_panel()). Returns a
    DataFrame the same shape, each cell = that ticker's cash-profitability
    percentile among every OTHER ticker with a real value that same day
    (100 = highest cash profitability in the universe). A ticker with no
    SEC data at all reads NaN every day, same "missing data doesn't
    fabricate a signal" convention as every sibling rank frame."""
    return cash_profitability_panel.rank(axis=1, pct=True) * 100


def precompute_cash_profitability_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional CASH PROFITABILITY
    RANK strategy's columns -- exact structural mirror of
    precompute_quality_frame() (same non-inverted ranking direction)."""
    df = precompute_breakout_frame(df, config)
    df["Cash_Profitability_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def cash_profitability_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional CASH PROFITABILITY RANK strategy's dict
    for one row of a frame already built by precompute_cash_profitability_frame()
    -- exact structural mirror of quality_levels_from_frame()."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    cash_profitability_percentile = last_row["Cash_Profitability_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    if pd.isna(cash_profitability_percentile):
        cash_profitability_signal = False
        cash_profitability_percentile_out = None
        signal_strength_pct = 0.0
    else:
        cash_profitability_percentile = float(cash_profitability_percentile)
        cash_profitability_signal = bool(cash_profitability_percentile >= config.cash_profitability_top_percentile_min)
        cash_profitability_percentile_out = round(cash_profitability_percentile, 2)
        signal_strength_pct = (
            round(cash_profitability_percentile - config.cash_profitability_top_percentile_min, 2)
            if cash_profitability_signal else 0.0
        )

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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Cash_Profitability_Percentile": cash_profitability_percentile_out,
        "Cash_Profitability_Signal": cash_profitability_signal,
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


def compute_cash_profitability_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional CASH PROFITABILITY RANK levels for one
    ticker's OHLCV history -- see precompute_cash_profitability_frame()'s
    own docstring for the full mechanism. Thin wrapper, exact mirror of
    compute_quality_levels()."""
    frame = precompute_cash_profitability_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return cash_profitability_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def classify_insider_transaction(text) -> str:
    """Classifies one yfinance insider_transactions row's free-text "Text"
    field -- the "Transaction" column itself returns empty in the
    currently installed yfinance version (1.5.1), so this is the only
    available classifier. Returns "purchase" only for a genuine open-market
    buy ("Purchase at price X per share."); everything else (sales, stock
    gifts, option exercises with a blank/NaN Text field, or any other
    free-text shape) classifies as "other" -- deliberately conservative,
    since a false "purchase" would fabricate a signal from a non-buy
    event. Public (not module-private) since run_backtest.fetch_insider_purchases()
    needs it via swingtrade's own public API, and it's independently
    unit-testable."""
    if not isinstance(text, str):
        return "other"
    return "purchase" if text.strip().lower().startswith("purchase at price") else "other"


def precompute_insider_buying_frame(
    df: pd.DataFrame,
    insider_purchases: pd.DataFrame | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the INSIDER-BUYING strategy's columns --
    built on top of precompute_breakout_frame() (reused wholesale for
    SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/liquidity gates
    every strategy shares).

    `insider_purchases`, if given, should be
    run_backtest.fetch_insider_purchases()'s own output: a DataFrame with
    columns ["effective_date", "value", "insider"], one row per genuine
    open-market insider Purchase, already reporting-lag-adjusted (see that
    function's own no-look-ahead reasoning -- "effective_date" is when the
    purchase is treated as PUBLICLY known, not the raw transaction date).

    For each trading day, sums the `value` of every purchase whose
    effective_date falls within the trailing config.insider_lookback_days
    CALENDAR days (inclusive of that day), and counts the number of
    DISTINCT insiders contributing to that sum -- a single insider
    repeatedly buying small amounts doesn't count as broad conviction the
    way several different insiders buying does.

    Fully vectorized via a (days x events) boolean window matrix, looping
    only over DISTINCT insiders (not days or raw events) for the
    distinct-buyer count -- cheap given insider_purchases is always a
    small event count per ticker (tens, not thousands, per the real
    watchlist probe behind this strategy's own scoping)."""
    df = precompute_breakout_frame(df, config)
    df["Insider_Purchase_Value"] = 0.0
    df["Insider_Distinct_Buyers"] = 0

    if insider_purchases is None or insider_purchases.empty:
        return df

    # insider_purchases["effective_date"] is tz-aware UTC (see
    # fetch_insider_purchases()); df.index is tz-naive (every OHLCV frame
    # in this codebase is) -- normalize to naive-UTC to match, rather than
    # touching df.index itself.
    event_dates_idx = pd.DatetimeIndex(insider_purchases["effective_date"])
    if event_dates_idx.tz is not None:
        event_dates_idx = event_dates_idx.tz_convert("UTC").tz_localize(None)

    day_index = df.index.values
    event_dates = event_dates_idx.values
    event_values = insider_purchases["value"].to_numpy(dtype=float)
    event_insiders = insider_purchases["insider"].to_numpy()

    lookback = np.timedelta64(config.insider_lookback_days, "D")
    # window: this day is within insider_lookback_days AFTER the event --
    # i.e. the event is still "recent" as of this day.
    in_window = (
        (day_index[:, None] >= event_dates[None, :])
        & (day_index[:, None] <= (event_dates[None, :] + lookback))
    )

    df["Insider_Purchase_Value"] = (in_window * event_values[None, :]).sum(axis=1)

    distinct_by_day = np.zeros(len(day_index), dtype=int)
    for insider_name in pd.unique(event_insiders):
        mask = event_insiders == insider_name
        distinct_by_day += in_window[:, mask].any(axis=1)
    df["Insider_Distinct_Buyers"] = distinct_by_day

    return df


def insider_buying_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the INSIDER-BUYING strategy's dict for one row of a frame
    already built by precompute_insider_buying_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    insider_value, insider_distinct_buyers = (
        last_row["Insider_Purchase_Value"], last_row["Insider_Distinct_Buyers"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating (this strategy's trigger is
    # insider purchase activity, not RSI) -- same treatment squeeze_breakout/
    # breakout/ma_crossover/pairs already give it.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    insider_value = float(insider_value) if not pd.isna(insider_value) else 0.0
    insider_distinct_buyers = int(insider_distinct_buyers) if not pd.isna(insider_distinct_buyers) else 0
    insider_signal = bool(
        insider_distinct_buyers >= config.insider_min_distinct_buyers
        and insider_value >= config.insider_min_purchase_value
    )
    # How many EXTRA distinct insiders bought, beyond the minimum required
    # -- same "distance past the trigger" differentiating-term role
    # Signal_Strength_Pct plays for squeeze_breakout/pairs, in distinct-buyer
    # units here, NOT a % (see pairs_zscore_strength_cap's identical "reused
    # field name, different units" precedent). Raw dollar value isn't used
    # for this -- real purchase sizes ($1M-$10M+) would blow past any sane
    # cap almost immediately and stop differentiating anything; several
    # different insiders buying independently is also the better-regarded
    # signal in the insider-trading literature over one large purchase.
    signal_strength_pct = (
        round(float(insider_distinct_buyers - config.insider_min_distinct_buyers), 2)
        if insider_signal else 0.0
    )

    # Buy_Price = today's own Close -- same "already happening, buy near
    # market" convention squeeze_breakout/pairs use, not a discount-limit
    # like RSI's structural-support wait. Distance_to_Buy_Pct is therefore
    # always 0 by construction, same schema-compatible treatment those
    # strategies already established.
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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Insider_Purchase_Value": round(insider_value, 2),
        "Insider_Distinct_Buyers": insider_distinct_buyers,
        "Insider_Buy_Signal": insider_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_insider_buying_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    insider_purchases: pd.DataFrame | None = None,
) -> dict:
    """Compute INSIDER-BUYING levels for one ticker's OHLCV history -- see
    precompute_insider_buying_frame()'s own docstring for the full
    mechanism. Thin wrapper over precompute_insider_buying_frame()/
    insider_buying_levels_from_frame() -- kept as a single-call convenience
    for the live dashboard, matching every other strategy's same rationale
    (see compute_squeeze_breakout_levels()).

    `insider_purchases` should be run_backtest.fetch_insider_purchases()'s
    own output for this ticker. Without it, Insider_Buy_Signal always
    reads False (no purchase data)."""
    frame = precompute_insider_buying_frame(df, insider_purchases, config)
    as_of = frame.index[-1]
    return insider_buying_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_analyst_revision_frame(
    df: pd.DataFrame,
    analyst_revisions: pd.DataFrame | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the ANALYST REVISION MOMENTUM strategy's
    columns -- built on top of precompute_breakout_frame() (reused
    wholesale for SMA_TREND/ATR/AvgVolume/etc.), exact structural mirror
    of precompute_insider_buying_frame() -- a small real per-ticker event
    count, not a cross-sectional universe-wide rank.

    `analyst_revisions`, if given, should be
    run_backtest.fetch_analyst_revisions()'s own output: a DataFrame with
    columns ["effective_date", "direction"] (direction +1.0=upgrade,
    -1.0=downgrade; reiterations/initiations already excluded), already
    reporting-lag-adjusted.

    For each trading day, sums `direction` over every event whose
    effective_date falls within the trailing config.analyst_revision_lookback_days
    CALENDAR days (inclusive) -- a net-upgrade count that can go negative
    (net downgrades). Same (days x events) boolean window-matrix
    vectorization as precompute_insider_buying_frame(), cheap given a real
    ticker's revision history is tens to low hundreds of events, not
    thousands."""
    df = precompute_breakout_frame(df, config)
    df["Net_Upgrades"] = 0.0

    if analyst_revisions is None or analyst_revisions.empty:
        return df

    event_dates_idx = pd.DatetimeIndex(analyst_revisions["effective_date"])
    if event_dates_idx.tz is not None:
        event_dates_idx = event_dates_idx.tz_convert("UTC").tz_localize(None)

    day_index = df.index.values
    event_dates = event_dates_idx.values
    event_directions = analyst_revisions["direction"].to_numpy(dtype=float)

    lookback = np.timedelta64(config.analyst_revision_lookback_days, "D")
    in_window = (
        (day_index[:, None] >= event_dates[None, :])
        & (day_index[:, None] <= (event_dates[None, :] + lookback))
    )
    df["Net_Upgrades"] = (in_window * event_directions[None, :]).sum(axis=1)

    return df


def analyst_revision_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the ANALYST REVISION MOMENTUM strategy's dict for one row of
    a frame already built by precompute_analyst_revision_frame() -- exact
    structural mirror of insider_buying_levels_from_frame()."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    net_upgrades = last_row["Net_Upgrades"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating -- same treatment every other
    # non-RSI-triggered strategy gives it.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    net_upgrades = float(net_upgrades) if not pd.isna(net_upgrades) else 0.0
    analyst_revision_signal = bool(net_upgrades >= config.analyst_revision_min_net_upgrades)
    # How far past the minimum net-upgrade count -- same "distance past the
    # trigger" differentiating-term role Signal_Strength_Pct plays for
    # insider_buying/pairs/squeeze_breakout.
    signal_strength_pct = (
        round(net_upgrades - config.analyst_revision_min_net_upgrades, 2) if analyst_revision_signal else 0.0
    )

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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Net_Upgrades": round(net_upgrades, 2),
        "Analyst_Revision_Signal": analyst_revision_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_analyst_revision_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    analyst_revisions: pd.DataFrame | None = None,
) -> dict:
    """Compute ANALYST REVISION MOMENTUM levels for one ticker's OHLCV
    history -- see precompute_analyst_revision_frame()'s own docstring for
    the full mechanism. Thin wrapper, exact mirror of
    compute_insider_buying_levels()."""
    frame = precompute_analyst_revision_frame(df, analyst_revisions, config)
    as_of = frame.index[-1]
    return analyst_revision_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_pead_frame(
    df: pd.DataFrame,
    earnings_surprises: pd.DataFrame | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the PEAD (post-earnings-announcement drift)
    strategy's columns -- built on top of precompute_breakout_frame()
    (reused wholesale for SMA_TREND/ATR/AvgVolume/etc., the same
    macro-uptrend/liquidity gates every strategy shares). Mirrors
    precompute_insider_buying_frame()'s own vectorized (days x events)
    window-matrix shape exactly, adapted to PEAD's own event schema.

    `earnings_surprises`, if given, should be
    run_backtest.fetch_earnings_surprises()'s own output: a DataFrame with
    columns ["effective_date", "surprise_pct"], one row per real reported
    quarter -- see that function's own EXPLICIT point-in-time-integrity
    caveat before trusting this.

    For each trading day, takes the MAX surprise_pct among events whose
    effective_date falls within the trailing config.pead_signal_window_days
    CALENDAR days (inclusive) -- normally at most one earnings report can
    plausibly fall in a multi-day window (quarterly cadence), so "max" is
    simple and correct rather than an arbitrary aggregation choice; NaN
    (no qualifying event in window) if none."""
    df = precompute_breakout_frame(df, config)
    df["PEAD_Surprise_Pct"] = np.nan

    if earnings_surprises is None or earnings_surprises.empty:
        return df

    # earnings_surprises["effective_date"] is tz-aware UTC (see
    # fetch_earnings_surprises()); df.index is tz-naive (every OHLCV frame
    # in this codebase is) -- normalize to naive-UTC to match, rather than
    # touching df.index itself. Same handling precompute_insider_buying_frame()
    # already established.
    event_dates_idx = pd.DatetimeIndex(earnings_surprises["effective_date"])
    if event_dates_idx.tz is not None:
        event_dates_idx = event_dates_idx.tz_convert("UTC").tz_localize(None)

    day_index = df.index.values
    event_dates = event_dates_idx.values
    event_surprises = earnings_surprises["surprise_pct"].to_numpy(dtype=float)

    window = np.timedelta64(config.pead_signal_window_days, "D")
    # window: this day is within pead_signal_window_days AFTER the report
    # -- i.e. the report is still "fresh" as of this day. A report
    # timestamped after that same day's close naturally excludes that same
    # calendar day (midnight < close-time timestamp), so the signal only
    # starts counting from the NEXT trading day onward -- see
    # fetch_earnings_surprises()'s own docstring for why this needs no
    # separate lag parameter the way insider Form-4 filings do.
    in_window = (
        (day_index[:, None] >= event_dates[None, :])
        & (day_index[:, None] <= (event_dates[None, :] + window))
    )

    masked = np.where(in_window, event_surprises[None, :], np.nan)
    with warnings.catch_warnings():
        # nanmax's own RuntimeWarning for an all-NaN row (no qualifying
        # event that day -- the overwhelmingly common case day-to-day,
        # given quarterly reporting cadence) is expected here, not a real
        # numerical error -- np.errstate() does NOT suppress this (it only
        # covers floating-point exceptions, not Python's warnings module).
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        df["PEAD_Surprise_Pct"] = np.nanmax(masked, axis=1)

    return df


def pead_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the PEAD strategy's dict for one row of a frame already
    built by precompute_pead_frame() -- the O(1)-per-row counterpart a
    walk-forward loop calls once per `as_of`. Same macro-uptrend/liquidity
    gates every strategy shares (via precompute_breakout_frame(), reused
    wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    surprise_pct = last_row["PEAD_Surprise_Pct"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating (this strategy's trigger is
    # an earnings surprise, not RSI) -- same treatment squeeze_breakout/
    # breakout/ma_crossover/pairs/insider_buying already give it.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

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

    surprise_pct = float(surprise_pct) if not pd.isna(surprise_pct) else None
    pead_signal = bool(surprise_pct is not None and surprise_pct >= config.pead_surprise_pct_min)
    # How far past the minimum the surprise was, in PERCENTAGE-POINT units
    # (NOT a % of the minimum) -- same "distance past the trigger"
    # differentiating-term role Signal_Strength_Pct plays for every other
    # strategy, see add_pead_trade_score().
    signal_strength_pct = (
        round(surprise_pct - config.pead_surprise_pct_min, 2) if pead_signal else 0.0
    )

    # Buy_Price = today's own Close -- same "already happening, buy near
    # market" convention squeeze_breakout/pairs/insider_buying use, not a
    # discount-limit like RSI's structural-support wait. Distance_to_Buy_Pct
    # is therefore always 0 by construction, same schema-compatible
    # treatment those strategies already established. config.pead_entry_fill
    # (default "next_open", NOT "limit" -- see its own config.py comment)
    # is what actually determines the real fill price/day downstream.
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

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "PEAD_Surprise_Pct": round(surprise_pct, 2) if surprise_pct is not None else None,
        "PEAD_Signal": pead_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_pead_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    earnings_surprises: pd.DataFrame | None = None,
) -> dict:
    """Compute PEAD levels for one ticker's OHLCV history -- see
    precompute_pead_frame()'s own docstring for the full mechanism (AND its
    own linked EXPLICIT data-integrity caveat). Thin wrapper over
    precompute_pead_frame()/pead_levels_from_frame() -- kept as a
    single-call convenience for the live dashboard, matching every other
    strategy's same rationale (see compute_squeeze_breakout_levels()).

    `earnings_surprises` should be run_backtest.fetch_earnings_surprises()'s
    own output for this ticker. Without it, PEAD_Signal always reads False
    (no surprise data)."""
    frame = precompute_pead_frame(df, earnings_surprises, config)
    as_of = frame.index[-1]
    return pead_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)
