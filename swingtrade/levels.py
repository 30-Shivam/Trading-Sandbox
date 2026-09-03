"""Pure technical-level calculations. No network calls, no UI -- takes
already-fetched OHLCV data (and already-resolved catalyst info) in, returns
plain dicts/tuples out. Safe to call from Streamlit, the settlement job, or a
backtest loop replaying years of historical bars.
"""

import operator

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import DEFAULT_CONFIG, TradingConfig


def is_market_uptrend(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> tuple[bool, float, float]:
    """Return (is_uptrend, last_close, sma) for a broad-market index's OHLCV data."""
    sma = df["Close"].rolling(window=config.sma_trend_window).mean().iloc[-1]
    last_close = float(df["Close"].iloc[-1])
    sma = float(sma)
    if pd.isna(sma):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    return last_close >= sma, last_close, sma


def market_uptrend_from_frame(frame: pd.DataFrame, as_of, config: TradingConfig = DEFAULT_CONFIG) -> tuple[bool, float, float]:
    """Vectorized counterpart to is_market_uptrend() -- reads SMA_TREND from
    a frame already built by precompute_rsi_frame() instead of recomputing
    the rolling SMA on a freshly truncated window. SMA_TREND is a plain
    fixed-window rolling mean, so this is an exact equivalence (see
    precompute_rsi_frame's docstring), not an approximation.

    Uses `.loc[:as_of].iloc[-1]` ("as-of" lookup) rather than an exact
    `.loc[as_of]` match -- preserves the original is_market_uptrend() call
    sites' behavior of tolerating a market calendar that doesn't share every
    exact trading date with the ticker being evaluated (e.g. a data-fetch
    edge case), falling back to the most recent prior market bar instead of
    raising a KeyError on a date the market frame doesn't have.
    """
    last_row = frame.loc[:as_of].iloc[-1]
    last_close = float(last_row["Close"])
    sma = last_row["SMA_TREND"]
    if pd.isna(sma):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    sma = float(sma)
    return last_close >= sma, last_close, sma


def precompute_rsi_frame(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Vectorized precompute of every rolling/EMA column compute_levels()
    needs, run ONCE over the full df instead of being recomputed from
    scratch at every `as_of` inside a walk-forward backtest loop -- that
    per-day full re-derivation (a ~260-day rolling recompute to learn about
    1 new row, repeated for every day of every fold) was the dominant cost
    of a full-scale search. See improvements.txt's vectorization item.

    SMA/SMA_TREND/AvgVolume are plain fixed-window rolling functions --
    identical whether computed once over a long df and sliced, or
    recomputed on a freshly truncated trailing window ending at the same
    row, by definition (a rolling window only ever looks at its own trailing
    N rows). RSI/ATR (Wilder-smoothed, pandas_ta) technically retain
    infinitesimal memory of data before any window start, but with the
    ~260-day trailing buffer this codebase always keeps (LOOKBACK_BUFFER_BARS
    in swingtrade/backtest.py), that residual is many orders of magnitude
    below the 2-decimal rounding applied everywhere downstream -- verified
    empirically against the old truncate-and-recompute path across real
    watchlist data before this was trusted (see improvements.txt).

    Oversold_Streak_Days is vectorized via a groupby-cumcount consecutive-run
    trick: `below` marks NaN or RSI>=threshold as a streak-breaker (matching
    the original backward-walking loop exactly). `run_id` increments AT each
    streak-breaking row, so every run's group is [breaking row, then the
    True rows that follow] -- cumcount() within that group already starts
    at 0 on the breaking row itself, so the first True row is cumcount=1,
    the second is 2, etc.: cumcount() alone (no +1) is the streak length.
    Verified against a brute-force backward walk over the identical RSI
    series (986 rows, 0 mismatches) before trusting this.
    """
    df = df.copy()
    df["SMA"] = df["Close"].rolling(window=config.ma_window).mean()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()

    below = df["RSI"].notna() & (df["RSI"] < config.rsi_oversold_threshold)
    run_id = (~below).cumsum()
    streak = below.groupby(run_id).cumcount()
    df["Oversold_Streak_Days"] = streak.where(below, 0).astype(int)
    df["Extended_Decline_Warning"] = df["Oversold_Streak_Days"] >= config.extended_decline_warning_days
    return df


def levels_from_rsi_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_levels()'s dict for one row of a frame already built
    by precompute_rsi_frame() -- the O(1)-per-row counterpart to compute_levels
    that a walk-forward loop calls once per `as_of` instead of paying the full
    indicator-recompute cost each time. Business logic (gates, buy signal,
    support/stop/target math, catalyst window) is verbatim compute_levels(),
    just reading from a precomputed row instead of computing it inline.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    # NaN/None-check every raw value BEFORE casting to float -- ta.rsi/ta.atr
    # can leave a genuinely bad row (e.g. a ticker symbol reused by a thin,
    # sparsely-traded stock after the original company renamed/delisted) as
    # a plain None rather than a float NaN, and float(None) raises a raw
    # TypeError instead of the intended graceful RuntimeError below.
    last_close, sma, sma_trend, rsi, atr, avg_volume = (
        last_row["Close"], last_row["SMA"], last_row["SMA_TREND"],
        last_row["RSI"], last_row["ATR"], last_row["AvgVolume"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma):
        raise RuntimeError(f"insufficient history to compute {config.ma_window}-day SMA")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(rsi):
        raise RuntimeError(f"insufficient history to compute {config.rsi_window}-day RSI")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma, sma_trend, rsi, atr, avg_volume = (
        float(last_close), float(sma), float(sma_trend), float(rsi), float(atr), float(avg_volume),
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

    oversold_streak_days = int(last_row["Oversold_Streak_Days"])
    extended_decline_warning = bool(last_row["Extended_Decline_Warning"])

    recent_window = frame.loc[:as_of].tail(config.support_lookback_days)
    support_level = float(recent_window["Low"].min())
    support_date = recent_window["Low"].idxmin()
    ma_discount_price = sma * (1 - config.ma_discount_pct)

    buy_price = round(support_level, 2)
    buy_basis = f"structural swing low (last {config.support_lookback_days}d, {support_date.date()})"

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    # risk can be <= 0 only if ATR rounds to 0 (flat price action); fall back
    # to 0.0 instead of None so the column stays numeric in the DataFrame/CSV.
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    distance_to_buy_pct = ((last_close - buy_price) / buy_price) * 100
    buy_signal = (last_close <= buy_price) and (rsi < config.rsi_oversold_threshold)

    # Reference point for the catalyst window is the bar's own date, not
    # wall-clock now() -- keeps this function backtest-safe.
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
        "SMA20": round(sma, 2),
        "MA_Discount_Price": round(ma_discount_price, 2),
        "Support_Level": round(support_level, 2),
        "Support_Date": support_date.date(),
        "RSI": round(rsi, 2),
        "ATR": round(atr, 2),
        "Buy_Price": buy_price,
        "Buy_Basis": buy_basis,
        "Buy_Signal": buy_signal,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": round(distance_to_buy_pct, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Oversold_Streak_Days": oversold_streak_days,
        "Extended_Decline_Warning": extended_decline_warning,
    }


def compute_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Compute buy/sell/stop levels, RSI/ATR, RRR, and catalyst flags for one
    ticker's OHLCV history.

    `next_earnings_date` (tz-aware Timestamp or None) and `top_headline` are
    passed in already-resolved rather than fetched here, so this function has
    no dependency on yfinance or wall-clock time -- the catalyst window is
    measured against the data's own last bar date, not `now()`, which is what
    makes it safe to reuse verbatim inside a historical backtest.

    Thin wrapper over precompute_rsi_frame()/levels_from_rsi_frame() -- kept
    as a single-call convenience for the live dashboard/ingest.py (which only
    ever need "today"'s row) so live and backtested signal generation always
    run through the exact same code path, with zero risk of the two drifting
    apart over time.
    """
    frame = precompute_rsi_frame(df, config)
    as_of = frame.index[-1]
    return levels_from_rsi_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_relative_strength(
    ticker_df: pd.DataFrame, market_df: pd.DataFrame, lookback_days: int
) -> float | None:
    """Ticker's total return over the trailing `lookback_days`, minus the
    market's (e.g. SPY) return over the identical window -- a simple
    additive relative-strength measure. Positive means the ticker beat the
    market over that window, negative means it lagged. Both `ticker_df` and
    `market_df` should already be truncated at `as_of` by the caller (no
    look-ahead) and are read via `.iloc[-(lookback_days + 1)]` /
    `.iloc[-1]`, so this is safe to call anywhere compute_breakout_levels()
    is. Returns None if either series doesn't have enough history yet --
    informational, not a reason to exclude a ticker on its own."""
    if len(ticker_df) <= lookback_days or len(market_df) <= lookback_days:
        return None
    ticker_start = float(ticker_df["Close"].iloc[-(lookback_days + 1)])
    ticker_end = float(ticker_df["Close"].iloc[-1])
    market_start = float(market_df["Close"].iloc[-(lookback_days + 1)])
    market_end = float(market_df["Close"].iloc[-1])
    if ticker_start == 0 or market_start == 0:
        return None
    ticker_return = (ticker_end - ticker_start) / ticker_start
    market_return = (market_end - market_start) / market_start
    return ticker_return - market_return


def precompute_breakout_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every rolling/EMA column compute_breakout_levels()
    needs, run ONCE over the full df -- the breakout counterpart to
    precompute_rsi_frame() (see its docstring for why this is an exact
    equivalence to the old per-`as_of` truncate-and-recompute approach, not
    an approximation, for every column here: SMA_TREND/AvgVolume/
    AvgVolume_Prior/Highest_High are all plain fixed-window rolling
    functions, and Relative_Strength is a fixed-lag pct_change, none of
    which carry RSI/ATR's Wilder-smoothing warmup subtlety). ADX is also
    Wilder-smoothed (like RSI/ATR) -- same reasoning applies: the ~260-day
    trailing buffer this codebase always keeps is far more than enough for
    its exponential decay to converge to the same value a fresh truncated
    recompute would have produced. OBV_Zscore and Squeeze_Zscore are both
    genuinely exact (not just converged) regardless of leading history --
    see each one's inline comment below for the specific invariance
    argument.

    `market_df`, if given, should be the market index's OHLCV over the same
    (or a superset) date range -- Relative_Strength is computed via
    `pct_change(periods=breakout_lookback_days)` on both series, aligned by
    date via reindex, matching compute_relative_strength()'s semantics
    (ticker return minus market return over the identical window).

    `sector_df`, if given ALONGSIDE `market_df` (both required -- this is a
    sector-vs-market comparison, not a ticker-vs-market one), should be the
    ticker's own sector's OHLCV (e.g. a real SPDR sector ETF like XLK for
    Technology) over the same date range -- Sector_Relative_Strength is
    computed the identical way as Relative_Strength (pct_change over
    `sector_relative_strength_lookback_days`, both series reindexed to
    `df`'s own index), just comparing the SECTOR's return to the market's
    instead of the ticker's own. BACKTEST/OPTUNA-ONLY as of its
    introduction -- market_data.py's live scan path never supplies
    `sector_df`, so this column simply doesn't exist there, and every
    caller downstream (breakout_levels_from_frame() etc.) already treats a
    missing/NaN Sector_Relative_Strength as "don't exclude," same
    convention as every other optional filter here.
    """
    df = df.copy()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()
    df["AvgVolume_Prior"] = df["AvgVolume"].shift(1)
    df["Highest_High"] = df["High"].rolling(window=config.breakout_lookback_days).max().shift(1)
    # ta.adx() returns None (not a NaN-filled DataFrame) when there's too
    # little history to compute even one window -- a real ticker can hit
    # this with a genuinely tiny row count (e.g. a recent delisting/take-
    # private leaving only a handful of trailing bars under the old
    # symbol, confirmed for EA in improvements.txt). Missing ADX already
    # doesn't exclude a signal on its own (same convention as OBV/squeeze
    # z-scores below), so degrade to NaN here rather than crashing --
    # matters for optimize.py/benchmark_random_entry.py, which (unlike
    # market_data.score_bundle_for_strategy's per-ticker try/except) had
    # no protection against this.
    adx_result = ta.adx(df["High"], df["Low"], df["Close"], length=config.adx_window)
    df["ADX"] = adx_result[f"ADX_{config.adx_window}"] if adx_result is not None else np.nan

    # On-Balance Volume, z-scored against its own trailing obv_window
    # mean/stdev -- OBV's raw cumulative magnitude is arbitrary (depends on
    # how much leading history precedes the window), but this z-score is
    # NOT: OBV and its rolling mean shift by the same constant offset for
    # any amount of extra leading history, so their difference (and this
    # z-score) is invariant to it.
    signed_volume = np.sign(df["Close"].diff()).fillna(0) * df["Volume"]
    obv = signed_volume.cumsum()
    obv_mean = obv.rolling(window=config.obv_window).mean()
    obv_std = obv.rolling(window=config.obv_window).std()
    df["OBV_Zscore"] = (obv - obv_mean) / obv_std

    # Bollinger-style volatility squeeze: raw normalized volatility
    # (stdev/mean of Close -- deliberately NOT multiplied by a band-width
    # constant like the classic "2 std" convention, since any constant
    # multiplier cancels out of the z-score below anyway), z-scored
    # against its own trailing bb_squeeze_window history, then shift(1) so
    # TODAY's own breakout move (which would itself add volatility) can't
    # contaminate the "was this a quiet coil beforehand" reading -- the
    # same no-look-ahead-into-today convention Highest_High uses.
    bb_vol = df["Close"].rolling(window=config.bb_window).std() / df["Close"].rolling(window=config.bb_window).mean()
    bb_vol_mean = bb_vol.rolling(window=config.bb_squeeze_window).mean()
    bb_vol_std = bb_vol.rolling(window=config.bb_squeeze_window).std()
    df["Squeeze_Zscore"] = ((bb_vol - bb_vol_mean) / bb_vol_std).shift(1)

    if market_df is not None:
        ticker_return = df["Close"].pct_change(periods=config.breakout_lookback_days)
        market_return = market_df["Close"].pct_change(periods=config.breakout_lookback_days).reindex(df.index)
        df["Relative_Strength"] = (ticker_return - market_return).replace([float("inf"), float("-inf")], pd.NA)

        if sector_df is not None:
            sector_return = sector_df["Close"].pct_change(
                periods=config.sector_relative_strength_lookback_days
            ).reindex(df.index)
            market_return_sector_window = market_df["Close"].pct_change(
                periods=config.sector_relative_strength_lookback_days
            ).reindex(df.index)
            df["Sector_Relative_Strength"] = (
                (sector_return - market_return_sector_window).replace([float("inf"), float("-inf")], pd.NA)
            )
    return df


def breakout_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_breakout_levels()'s dict for one row of a frame
    already built by precompute_breakout_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of` instead of
    paying the full indicator-recompute cost each time. Business logic is
    verbatim compute_breakout_levels(), just reading from a precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, highest_high, rsi, last_volume, avg_volume_prior, adx = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"],
        last_row["AvgVolume"], last_row["Highest_High"], last_row["RSI"],
        last_row["Volume"], last_row["AvgVolume_Prior"], last_row["ADX"],
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
    if pd.isna(highest_high):
        raise RuntimeError(f"insufficient history to compute {config.breakout_lookback_days}-day high")
    # RSI is informational only for breakout (not used for gating/signal
    # decisions) -- don't exclude a ticker just because its RSI warmup
    # hasn't filled yet; leave it None rather than raising.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    # Same informational treatment for the volume ratio.
    if pd.isna(avg_volume_prior) or float(avg_volume_prior) == 0:
        volume_ratio = None
    else:
        volume_ratio = round(float(last_volume) / float(avg_volume_prior), 3)
    # Same informational treatment for ADX -- missing/warming-up shouldn't
    # exclude a ticker on its own, only the breakout_adx_min gate (applied
    # downstream in simulate_breakout_signals/add_breakout_trade_score) does.
    adx = None if pd.isna(adx) else round(float(adx), 2)
    # Same informational treatment for OBV/squeeze z-scores.
    obv_zscore = None if pd.isna(obv_zscore) else round(float(obv_zscore), 3)
    squeeze_zscore = None if pd.isna(squeeze_zscore) else round(float(squeeze_zscore), 3)

    last_close, sma_trend, atr, avg_volume, highest_high = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume), float(highest_high),
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

    trigger_price = round(highest_high, 2)
    breakout_signal = last_close > highest_high

    buy_price = trigger_price
    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0
    distance_to_buy_pct = ((last_close - buy_price) / buy_price) * 100

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    relative_strength = None
    if "Relative_Strength" in frame.columns:
        rs_val = last_row["Relative_Strength"]
        if pd.notna(rs_val):
            relative_strength = round(float(rs_val), 4)

    # Sector_Relative_Strength (backtest/Optuna-only, see precompute_breakout_frame's
    # own docstring) -- only present in the frame if sector_df was ALSO
    # supplied alongside market_df; missing/NaN never excludes a ticker on
    # its own, same convention as Relative_Strength above.
    sector_relative_strength = None
    if "Sector_Relative_Strength" in frame.columns:
        srs_val = last_row["Sector_Relative_Strength"]
        if pd.notna(srs_val):
            sector_relative_strength = round(float(srs_val), 4)

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Trigger_Price": trigger_price,
        "Trigger_Basis": f"{config.breakout_lookback_days}-day high (prior days only)",
        "Breakout_Signal": breakout_signal,
        "Relative_Strength": relative_strength,
        "Sector_Relative_Strength": sector_relative_strength,
        "Volume_Ratio": volume_ratio,
        "ADX": adx,
        "OBV_Zscore": obv_zscore,
        "Squeeze_Zscore": squeeze_zscore,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": round(distance_to_buy_pct, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_breakout_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> dict:
    """Compute breakout trigger/stop/target levels for one ticker's OHLCV
    history -- the trend-following counterpart to compute_levels()'s
    RSI-oversold mean-reversion logic. Same macro-uptrend and liquidity
    gates (a breakout in a stock nobody can actually trade, or one that's
    still in a macro downtrend, isn't a real opportunity either), but the
    trigger is a new N-day CLOSING high instead of an RSI reading, and
    Trigger_Price is a level price gets bought ABOVE, not below.

    `Highest_High` uses `.shift(1)` -- the highest High of the PRIOR
    `config.breakout_lookback_days` days, excluding today's own bar. Today's
    Close is then compared against that prior-days-only level, so this stays
    safe to call with `df` truncated at any `as_of` inside a backtest loop,
    exactly like compute_levels().

    The returned dict is schema-compatible with compute_levels() (Buy_Price,
    Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct, Catalyst_Warning, etc.
    all present) so live callers (market_data.scan_tickers, the dashboard,
    storage) don't need to special-case which strategy produced a row --
    only add_trade_score vs add_breakout_trade_score (swingtrade/scoring.py)
    differ. RSI is computed here too even though breakout doesn't use it for
    its own gating -- purely informational today (storage requires the
    field, and it's a natural input for a future "skip overbought breakouts"
    filter, see improvements.txt). `next_earnings_date`/`top_headline` are
    optional and behave exactly as in compute_levels(); backtest callers
    omit them (Catalyst_Warning stays False, matching simulate_signals'
    same behavior when earnings_dates isn't supplied).

    `market_df` (optional) enables Relative_Strength (see
    compute_relative_strength()) -- the ticker's return over
    config.breakout_lookback_days minus the market's return over the same
    window, informational unless config.breakout_relative_strength_min is
    changed from its disabled default. None if market_df isn't supplied.

    Volume_Ratio is today's Volume divided by the PRIOR (`.shift(1)`, same
    no-look-ahead convention as Highest_High) volume_lookback_days average
    -- a genuine breakout on high volume vs. a low-volume drift above an
    old high are different events (see improvements.txt item 6). None if
    there isn't enough history for the prior-average yet.

    ADX (config.adx_window, default 14) measures the STRENGTH of the
    current trend, independent of direction -- a different dimension than
    RSI (momentum level) or Relative_Strength (direction vs. market). A
    breakout during a weak/choppy trend and one during a genuinely strong
    trend look identical to every other field here, but aren't the same
    event; informational unless config.breakout_adx_min is changed from
    its disabled default (see improvements.txt).

    OBV_Zscore is On-Balance Volume (cumulative signed volume) z-scored
    against its own trailing obv_window mean/stdev -- rising relative to
    its own recent baseline reflects sustained buying pressure building up
    over WEEKS, a deeper signal than Volume_Ratio's single-day spike;
    informational unless config.breakout_obv_zscore_min is changed from
    its disabled default.

    Squeeze_Zscore is a volatility (stdev/mean of Close) z-score against
    its own trailing bb_squeeze_window history, read as of the PRIOR day
    (`.shift(1)`, so today's own breakout move can't contaminate it) --
    classic "coiled spring" pattern, a breakout emerging from unusually
    contracted volatility is a different event than one that isn't;
    informational unless config.breakout_squeeze_zscore_max is changed
    from its disabled default.

    Thin wrapper over precompute_breakout_frame()/breakout_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard/ingest.py so
    live and backtested signal generation always run through the exact same
    code path (see compute_levels()'s docstring for the same rationale).
    """
    frame = precompute_breakout_frame(df, config, market_df=market_df, sector_df=sector_df)
    as_of = frame.index[-1]
    return breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_pullback_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of every rolling column pullback_levels_from_frame()
    needs, run ONCE over the full df -- the pullback counterpart to
    precompute_rsi_frame()/precompute_breakout_frame() (see precompute_rsi_frame's
    docstring for why this is an exact equivalence to a fresh truncated
    recompute, not an approximation: SMA_TREND/AvgVolume/MA_Pullback are all
    plain fixed-window rolling functions; RSI/ATR/ADX are Wilder-smoothed
    but converge well within the ~260-day trailing buffer this codebase
    always keeps -- see LOOKBACK_BUFFER_BARS in swingtrade/backtest.py).
    """
    df = df.copy()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"], length=config.adx_window)[f"ADX_{config.adx_window}"]
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()
    df["MA_Pullback"] = df["Close"].rolling(window=config.pullback_ma_window).mean()
    df["MA_Pullback_Prior"] = df["MA_Pullback"].shift(config.pullback_ma_slope_window)
    return df


def pullback_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_pullback_levels()'s dict for one row of a frame
    already built by precompute_pullback_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Business logic
    is verbatim compute_pullback_levels(), just reading from a precomputed
    row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, rsi, atr, avg_volume, adx, ma_pullback, ma_pullback_prior = (
        last_row["Close"], last_row["SMA_TREND"], last_row["RSI"], last_row["ATR"],
        last_row["AvgVolume"], last_row["ADX"], last_row["MA_Pullback"], last_row["MA_Pullback_Prior"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(ma_pullback):
        raise RuntimeError(f"insufficient history to compute {config.pullback_ma_window}-day pullback MA")
    if pd.isna(ma_pullback_prior):
        raise RuntimeError(
            f"insufficient history to confirm {config.pullback_ma_window}-day MA slope "
            f"({config.pullback_ma_slope_window}d lookback)"
        )
    # RSI/ADX are informational only for pullback (not used for gating) --
    # same treatment as breakout's RSI/ADX (see breakout_levels_from_frame).
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    adx = None if pd.isna(adx) else round(float(adx), 2)

    last_close, sma_trend, atr, avg_volume, ma_pullback, ma_pullback_prior = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume),
        float(ma_pullback), float(ma_pullback_prior),
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

    buy_price = round(ma_pullback, 2)
    distance_to_buy_pct = ((last_close - buy_price) / buy_price) * 100
    ma_rising = ma_pullback > ma_pullback_prior
    pullback_signal = bool(ma_rising and (abs(distance_to_buy_pct) <= config.pullback_band_pct))

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
        "ADX": adx,
        "MA_Pullback": buy_price,
        "MA_Basis": f"{config.pullback_ma_window}-day SMA (rising over prior {config.pullback_ma_slope_window}d)",
        "Pullback_Signal": pullback_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": round(distance_to_buy_pct, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_pullback_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Compute pullback-in-uptrend levels for one ticker's OHLCV history --
    a third strategy, distinct from both compute_levels() (RSI-oversold
    mean-reversion -- shown by benchmark_random_entry.py to carry no real
    timing edge) and compute_breakout_levels() (requires a fresh N-day
    closing high the SAME day, which is rare). Buys a shallow dip toward a
    RISING pullback_ma_window-day SMA within a confirmed macro uptrend
    (Last_Close > SMA_TREND) -- trend-following like breakout, but fires
    far more often since it doesn't require price to be at a fresh extreme.

    Pullback_Signal requires BOTH: the pullback MA itself rising over the
    trailing pullback_ma_slope_window days (filters out a topping/rolling-
    over MA, not a genuine uptrend pullback) AND Distance_to_Buy_Pct within
    +/- pullback_band_pct of the MA (close enough to call it a genuine
    support test, not still extended above or already broken below).
    Buy_Price is the MA level itself -- a resting LIMIT order, filled via
    the same _find_entry_fill() mechanics as the RSI strategy (see
    simulate_pullback_signals), not a stop-buy like breakout's.

    The returned dict is schema-compatible with compute_levels()/
    compute_breakout_levels() (Buy_Price, Sell_Price, Stop_Loss, RRR,
    Distance_to_Buy_Pct, Catalyst_Warning, etc. all present) -- only
    add_pullback_trade_score (swingtrade/scoring.py) differs downstream.
    RSI/ADX are informational only here, same treatment as breakout's.

    Thin wrapper over precompute_pullback_frame()/pullback_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard/ingest.py,
    matching compute_levels()/compute_breakout_levels()'s same rationale.
    """
    frame = precompute_pullback_frame(df, config)
    as_of = frame.index[-1]
    return pullback_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_breakout_retest_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every column breakout_retest_levels_from_frame()
    needs -- built ON TOP of precompute_breakout_frame() (reused wholesale,
    not reimplemented: SMA_TREND/RSI/ATR/AvgVolume/Highest_High/ADX/etc. are
    identical to what a genuine breakout needs, since a retest is
    fundamentally "did a genuine breakout happen recently, and has price
    pulled back to it").

    Adds three retest-specific columns:
    - Breakout_Flag: True on any day Close > Highest_High (today's own
      breakout) -- exactly compute_breakout_levels' Breakout_Signal
      definition, just computed for every row instead of one `as_of`.
    - Last_Breakout_Level: the Highest_High value from the MOST RECENT
      Breakout_Flag day, forward-filled -- NaN before the first breakout
      ever occurs. ffill() only ever propagates an EARLIER row's value
      forward, so this carries no look-ahead: on any given day it reflects
      only breakouts that have already happened by that day's close,
      exactly like every other rolling column in this codebase.
    - Days_Since_Breakout: row-position distance back to that same most
      recent breakout day (0 on the breakout day itself, NaN before any
      breakout has occurred) -- same forward-fill-of-position technique,
      no look-ahead for the identical reason.
    """
    df = precompute_breakout_frame(df, config, market_df=market_df)
    breakout_flag = df["Close"] > df["Highest_High"]
    df["Last_Breakout_Level"] = df["Highest_High"].where(breakout_flag).ffill()
    row_position = pd.Series(range(len(df)), index=df.index)
    breakout_day_position = row_position.where(breakout_flag).ffill()
    df["Days_Since_Breakout"] = row_position - breakout_day_position
    return df


def breakout_retest_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_breakout_retest_levels()'s dict for one row of a
    frame already built by precompute_breakout_retest_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Business logic is verbatim compute_breakout_retest_levels(), just
    reading from a precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi, adx, last_breakout_level, days_since_breakout = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"],
        last_row["RSI"], last_row["ADX"], last_row["Last_Breakout_Level"], last_row["Days_Since_Breakout"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    # RSI/ADX are informational only for breakout_retest (not used for
    # gating) -- same treatment as breakout's/pullback's.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    adx = None if pd.isna(adx) else round(float(adx), 2)

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)

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

    # No breakout has ever occurred yet -- Last_Breakout_Level/Days_Since_Breakout
    # are both NaN. Not an error (a totally normal, common state for a
    # ticker with no recent breakout) -- just means Retest_Signal is False.
    has_breakout = pd.notna(last_breakout_level) and pd.notna(days_since_breakout)
    if has_breakout:
        buy_price = round(float(last_breakout_level), 2)
        days_since = int(days_since_breakout)
        distance_to_buy_pct = ((last_close - buy_price) / buy_price) * 100
        retest_signal = bool(
            0 < days_since <= config.retest_window_days and abs(distance_to_buy_pct) <= config.retest_band_pct
        )
    else:
        # No prior breakout to retest -- Buy_Price/Distance_to_Buy_Pct still
        # need SOME numeric value for schema-compatibility (storage/display
        # code reads these fields unconditionally), so fall back to
        # Last_Close itself (Distance_to_Buy_Pct becomes 0 by construction)
        # -- harmless, since Retest_Signal is always False here and the
        # hard gate in add_breakout_retest_trade_score zeroes the score
        # regardless of what these fallback values are.
        buy_price = round(last_close, 2)
        days_since = None
        distance_to_buy_pct = 0.0
        retest_signal = False

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
        "ADX": adx,
        "Days_Since_Breakout": days_since,
        "Retest_Signal": retest_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": round(distance_to_buy_pct, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_breakout_retest_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
) -> dict:
    """Compute breakout-retest levels for one ticker's OHLCV history -- a
    fourth strategy, built after BOTH RSI-oversold (compute_levels) and
    pullback-in-uptrend (compute_pullback_levels) lost to matched-count
    random-entry timing on held-out tickers (see benchmark_random_entry.py),
    while breakout (compute_breakout_levels) was the one signal that beat
    it. Keeps that validated ingredient -- Retest_Signal requires a genuine
    breakout_lookback_days-day closing high to have occurred within the
    last retest_window_days days -- but relaxes breakout's single most
    restrictive property (must fire THE SAME DAY) by allowing entry when
    price pulls BACK to that breakout's own trigger level within the
    window, instead of only on the breakout day itself.

    Buy_Price is the ORIGINAL breakout's trigger level (Highest_High on the
    day it fired, forward-filled) -- a resting LIMIT order, filled via the
    same _find_entry_fill() mechanics as RSI/pullback (buying a dip back
    down to a known level), NOT breakout's own stop-buy _find_breakout_fill().

    The returned dict is schema-compatible with the other three strategies'
    (Buy_Price, Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct,
    Catalyst_Warning, etc. all present) -- only add_breakout_retest_trade_score
    (swingtrade/scoring.py) differs downstream. RSI/ADX are informational
    only, same treatment as breakout's/pullback's.

    No extra filters (RSI-overbought/relative-strength/volume/ADX/OBV/squeeze)
    in this v1 -- same lean-first-version reasoning as compute_pullback_levels.

    Thin wrapper over precompute_breakout_retest_frame()/breakout_retest_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard/ingest.py,
    matching every other strategy's same rationale.
    """
    frame = precompute_breakout_retest_frame(df, config, market_df=market_df)
    as_of = frame.index[-1]
    return breakout_retest_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_week52_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of every rolling column week52_levels_from_frame()
    needs -- the 52-week-high counterpart to precompute_rsi_frame()/
    precompute_pullback_frame(). SMA_TREND/RSI/ATR/AvgVolume are the same
    shared macro/liquidity/informational columns every strategy computes.

    Week52_High = High.rolling(week52_lookback_days).max() -- deliberately
    NOT shifted (unlike breakout's Highest_High), because this is a
    continuous STATE (how close is today's own price to its own trailing
    high, inclusive of today), not a discrete EVENT (did today cross a
    level established before today) -- see compute_week52_levels()'s
    docstring for the full reasoning. Still fully backtest-safe: only ever
    reads data through and including the current row, same as SMA_TREND/
    RSI/ATR everywhere else in this codebase.
    """
    df = df.copy()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()
    df["Week52_High"] = df["High"].rolling(window=config.week52_lookback_days).max()
    return df


def week52_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_week52_levels()'s dict for one row of a frame
    already built by precompute_week52_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Business logic
    is verbatim compute_week52_levels(), just reading from a precomputed
    row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, rsi, atr, avg_volume, week52_high = (
        last_row["Close"], last_row["SMA_TREND"], last_row["RSI"], last_row["ATR"],
        last_row["AvgVolume"], last_row["Week52_High"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(week52_high):
        raise RuntimeError(f"insufficient history to compute {config.week52_lookback_days}-day high")
    # RSI is informational only for week52_high (not used for gating) --
    # same treatment as the other trend-following strategies.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

    last_close, sma_trend, atr, avg_volume, week52_high = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume), float(week52_high),
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

    buy_price = round(last_close, 2)
    # Distance BELOW the 52-week high -- >=0 normally; can go slightly
    # negative if today's own Close is itself a fresh week52_lookback_days
    # high (Week52_High includes today, see precompute_week52_frame).
    distance_to_buy_pct = ((week52_high - last_close) / week52_high) * 100
    week52_signal = bool(distance_to_buy_pct <= config.week52_nearness_pct)

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
        "Week52_High": round(week52_high, 2),
        "Week52_Signal": week52_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": round(distance_to_buy_pct, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_week52_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Compute 52-week-high-momentum levels for one ticker's OHLCV history
    -- a fifth strategy, a well-documented academic factor (George & Hwang
    2004, "The 52-Week High and Momentum Investing") distinct from every
    prior attempt here: unlike breakout (a discrete "new high TODAY" event,
    compute_breakout_levels) or breakout_retest (a bounded window after one
    specific event, compute_breakout_retest_levels), this is a continuous
    STATE -- how close is price, right now, to its own trailing
    week52_lookback_days high -- so it can stay true for many consecutive
    days while a stock consolidates near its highs.

    Week52_High deliberately does NOT use .shift(1) the way breakout's
    Highest_High does: breakout's shift exists because it's detecting a
    discrete EVENT (did today cross a level established BEFORE today);
    this is a continuous STATE description (how close is today's own
    price to its own trailing high, inclusive of today) -- the standard
    academic definition, and still fully backtest-safe, since both
    formulations only ever use data through `as_of`, no future leakage
    either way.

    Buy_Price is today's own Close -- a resting LIMIT order at essentially
    the current price (this strategy means "already near strength," not
    "wait for a specific dip level"), filled via the same
    _find_entry_fill() mechanics as pullback/breakout_retest, not
    breakout's stop-buy. Distance_to_Buy_Pct is repurposed here to mean
    "distance BELOW the 52-week high" (>=0 normally, can go slightly
    negative if today's Close itself is a fresh 252-day high) -- consistent
    with how every prior strategy gives this field its own strategy-
    specific meaning while feeding the identical scoring formula.

    The returned dict is schema-compatible with the other four strategies'
    -- only add_week52_trade_score (swingtrade/scoring.py) differs
    downstream. RSI is informational only, same treatment as the others.

    Thin wrapper over precompute_week52_frame()/week52_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard/ingest.py,
    matching every other strategy's same rationale.
    """
    frame = precompute_week52_frame(df, config)
    as_of = frame.index[-1]
    return week52_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_momentum_burst_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every column momentum_burst_levels_from_frame()
    needs -- built ON TOP of precompute_breakout_frame() (reused wholesale,
    not reimplemented: SMA_TREND/RSI/ATR/AvgVolume/AvgVolume_Prior/ADX/etc.
    are identical to what breakout already computes, and this strategy's
    volume-confirmation leg is the SAME Volume_Ratio concept breakout's own
    optional breakout_volume_ratio_min filter already uses -- see
    breakout_levels_from_frame()'s inline Volume_Ratio computation, mirrored
    the same way here rather than added as its own frame column).

    Adds one new column: Day_Gain_Pct = today's Close vs. PRIOR Close %
    gain (pct_change() -- no shift needed, this reads only today's and
    yesterday's already-closed bars, no look-ahead)."""
    df = precompute_breakout_frame(df, config, market_df=market_df)
    df["Day_Gain_Pct"] = df["Close"].pct_change() * 100
    return df


def momentum_burst_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_momentum_burst_levels()'s dict for one row of a
    frame already built by precompute_momentum_burst_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Business logic is verbatim compute_momentum_burst_levels(), just
    reading from a precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, avg_volume_prior, last_volume, rsi, day_gain_pct = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"],
        last_row["AvgVolume_Prior"], last_row["Volume"], last_row["RSI"], last_row["Day_Gain_Pct"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(day_gain_pct):
        raise RuntimeError("insufficient history: no prior Close to compute today's % gain")
    # RSI is informational only for momentum_burst (not used for gating) --
    # same treatment as every other trend-following strategy.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

    last_close, sma_trend, atr, avg_volume, day_gain_pct = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume), float(day_gain_pct),
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

    # Same informational-unless-thresholded treatment as breakout's own
    # Volume_Ratio (see breakout_levels_from_frame) -- here it's the
    # trigger's own gating leg, not optional, so a missing/zero prior
    # average correctly means "can't confirm volume," not "skip the check."
    if pd.isna(avg_volume_prior) or float(avg_volume_prior) == 0:
        volume_ratio = None
    else:
        volume_ratio = round(float(last_volume) / float(avg_volume_prior), 3)

    momentum_signal = bool(
        day_gain_pct >= config.momentum_burst_gain_pct_min
        and volume_ratio is not None
        and volume_ratio >= config.momentum_burst_volume_ratio_min
    )

    # Buy_Price = today's own Close -- same "already happening, buy near
    # the current price" convention week52_high uses, not a resting level
    # to wait for (this is a same-day confirmation of a move already in
    # progress, not a discrete price level). Distance_to_Buy_Pct is
    # therefore always 0 by construction (Buy_Price IS Last_Close) --
    # uninformative for this strategy the same way breakout_retest's own
    # "no prior breakout yet" fallback case is, kept only for
    # schema-compatibility with the shared scoring formula.
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
        "Day_Gain_Pct": round(day_gain_pct, 2),
        "Volume_Ratio": volume_ratio,
        "Momentum_Signal": momentum_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        # How far today's gain clears its own minimum bar -- see
        # add_momentum_burst_trade_score, replaces Distance_to_Buy_Pct
        # (always 0 here) as the score's ticker-differentiating term.
        "Signal_Strength_Pct": round(day_gain_pct - config.momentum_burst_gain_pct_min, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_momentum_burst_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Compute momentum-burst levels for one ticker's OHLCV history -- a
    sixth strategy, built specifically to fire MORE OFTEN than any prior
    one: unlike breakout (a discrete "fresh N-day high TODAY" event) or
    breakout_retest/week52_high (both anchored to a specific high level),
    this fires on a single day's strong price gain CONFIRMED by unusually
    high volume -- a genuinely different phenomenon (sudden, volume-backed
    conviction) rather than a more-sensitive version of an existing
    trigger, so it can fire on days none of the other five strategies do.

    Momentum_Signal requires BOTH Day_Gain_Pct >= momentum_burst_gain_pct_min
    AND Volume_Ratio >= momentum_burst_volume_ratio_min -- price movement
    alone (a volatile, low-conviction day) and volume alone (elevated
    trading with no real price movement) are each, individually, weak
    evidence; both together is the actual "conviction" signal.

    Buy_Price is today's own Close (see momentum_burst_levels_from_frame's
    inline comment for why -- same convention as week52_high, NOT a
    resting level to wait for).

    The returned dict is schema-compatible with every other strategy's
    (Buy_Price, Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct,
    Catalyst_Warning, etc. all present) -- only add_momentum_burst_trade_score
    (swingtrade/scoring.py) differs downstream. RSI is informational only,
    same treatment as the others.

    Thin wrapper over precompute_momentum_burst_frame()/momentum_burst_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard, matching
    every other strategy's same rationale. NOT called from ingest.py in
    this v1 -- this strategy is dashboard-only/experimental until it
    passes the same random-entry-timing validation every other strategy
    here was held to (see benchmark_random_entry.py).
    """
    frame = precompute_momentum_burst_frame(df, config)
    as_of = frame.index[-1]
    return momentum_burst_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def precompute_squeeze_breakout_frame(
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute of every column squeeze_breakout_levels_from_frame()
    needs -- built ON TOP of precompute_breakout_frame() (reused wholesale:
    SMA_TREND/RSI/ATR/AvgVolume/Squeeze_Zscore/etc. are identical to what
    breakout already computes -- Squeeze_Zscore in particular is the SAME
    column breakout's own optional breakout_squeeze_zscore_max filter
    already uses, not recomputed here).

    Adds two new columns:
    - Recent_Min_Squeeze_Zscore: rolling MIN of Squeeze_Zscore over the
      trailing squeeze_breakout_lookback_days -- "was there a genuine
      squeeze at some point in the last few days," not just yesterday
      specifically (squeezes often persist several days before
      releasing). Squeeze_Zscore itself is already .shift(1)'d (see
      precompute_breakout_frame), so a rolling min over already-lagged
      values stays fully no-look-ahead.
    - Day_Gain_Pct: today's Close vs. PRIOR Close % gain (pct_change() --
      no shift needed, reads only today's and yesterday's already-closed
      bars) -- same concept precompute_momentum_burst_frame() introduced,
      computed independently here since these are separate frame-builder
      functions."""
    df = precompute_breakout_frame(df, config, market_df=market_df, sector_df=sector_df)
    df["Recent_Min_Squeeze_Zscore"] = df["Squeeze_Zscore"].rolling(window=config.squeeze_breakout_lookback_days).min()
    df["Day_Gain_Pct"] = df["Close"].pct_change() * 100
    return df


def squeeze_breakout_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract compute_squeeze_breakout_levels()'s dict for one row of a
    frame already built by precompute_squeeze_breakout_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Business logic is verbatim compute_squeeze_breakout_levels(), just
    reading from a precomputed row.
    """
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi, recent_min_squeeze, day_gain_pct = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"],
        last_row["RSI"], last_row["Recent_Min_Squeeze_Zscore"], last_row["Day_Gain_Pct"],
    )
    # Phase 2 sharpening filters (improvements.txt item 42/43) -- ADX/
    # Volume_Ratio/OBV_Zscore informational, same treatment breakout's own
    # optional filters use (missing/NaN never excludes on its own).
    adx, last_volume, avg_volume_prior, obv_zscore = (
        last_row["ADX"], last_row["Volume"], last_row["AvgVolume_Prior"], last_row["OBV_Zscore"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")
    if pd.isna(day_gain_pct):
        raise RuntimeError("insufficient history: no prior Close to compute today's % gain")
    if pd.isna(recent_min_squeeze):
        raise RuntimeError(
            f"insufficient history to compute {config.squeeze_breakout_lookback_days}-day trailing Squeeze_Zscore"
        )
    # RSI is informational only for squeeze_breakout (not used for
    # gating) -- same treatment as every other trend-following strategy.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    adx = None if pd.isna(adx) else round(float(adx), 2)
    obv_zscore = None if pd.isna(obv_zscore) else round(float(obv_zscore), 3)
    if pd.isna(avg_volume_prior) or float(avg_volume_prior) == 0:
        volume_ratio = None
    else:
        volume_ratio = round(float(last_volume) / float(avg_volume_prior), 3)

    last_close, sma_trend, atr, avg_volume, day_gain_pct, recent_min_squeeze = (
        float(last_close), float(sma_trend), float(atr), float(avg_volume),
        float(day_gain_pct), float(recent_min_squeeze),
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

    squeeze_signal = bool(
        recent_min_squeeze <= config.squeeze_breakout_zscore_max
        and day_gain_pct >= config.squeeze_breakout_gain_pct_min
    )

    # Buy_Price = today's own Close -- this is a same-day confirmation of
    # an expansion already happening (same convention momentum_burst/
    # week52_high use), not a specific price level to wait for.
    # Distance_to_Buy_Pct is therefore always 0 by construction, same
    # uninformative-but-schema-compatible treatment momentum_burst
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

    # Relative_Strength (Phase 2 filter) needs market_df -- only present in
    # the frame if precompute_squeeze_breakout_frame() was called with it
    # (see breakout_levels_from_frame's/adx_trend_entry_levels_from_frame's
    # identical pattern).
    relative_strength = None
    if "Relative_Strength" in frame.columns:
        rs_val = last_row["Relative_Strength"]
        if pd.notna(rs_val):
            relative_strength = round(float(rs_val), 4)

    # Sector_Relative_Strength (backtest/Optuna-only) -- same graceful
    # missing-column/NaN treatment as breakout_levels_from_frame's own.
    sector_relative_strength = None
    if "Sector_Relative_Strength" in frame.columns:
        srs_val = last_row["Sector_Relative_Strength"]
        if pd.notna(srs_val):
            sector_relative_strength = round(float(srs_val), 4)

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "ADX": adx,
        "Relative_Strength": relative_strength,
        "Sector_Relative_Strength": sector_relative_strength,
        "Volume_Ratio": volume_ratio,
        "OBV_Zscore": obv_zscore,
        "Day_Gain_Pct": round(day_gain_pct, 2),
        "Recent_Min_Squeeze_Zscore": round(recent_min_squeeze, 3),
        "Squeeze_Signal": squeeze_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        # How far today's gain clears its own minimum bar -- see
        # add_squeeze_breakout_trade_score, replaces Distance_to_Buy_Pct
        # (always 0 here) as the score's ticker-differentiating term.
        "Signal_Strength_Pct": round(day_gain_pct - config.squeeze_breakout_gain_pct_min, 2),
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def precompute_pairs_frame(
    df: pd.DataFrame,
    peer_prices: pd.DataFrame | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the mean-reversion PAIRS strategy's columns
    -- built on top of precompute_breakout_frame() (reused wholesale for
    SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/liquidity gates
    every strategy shares).

    `peer_prices`, if given, should be a wide DataFrame of Close prices for
    every OTHER ticker in this ticker's own sector (this ticker's own
    column must NOT be included -- the caller's job to pre-filter by
    sector_lookup, keeping this function a pure DataFrame-in computation
    with no fetching/lookup of its own), reindexed to (at least cover)
    `df`'s own date range.

    For each trading day, picks whichever peer has the highest rolling
    correlation (pairs_lookback_days window) to this ticker's own daily
    returns, PROVIDED it clears pairs_min_correlation -- Pair_Partner (the
    winning peer's ticker) and Pair_Correlation are both fully vectorized
    via pandas' own rolling().corr(), genuinely causal (each day's value
    uses only the trailing window ending that day, never a future value) --
    not a per-day Python loop.

    Pair_Spread_Zscore measures how unusual TODAY's divergence from the
    winning partner is, relative to what's been NORMAL for this specific
    pair recently: spread = this ticker's cumulative return over
    pairs_spread_window_days minus the partner's own cumulative return over
    the identical window; the z-score baseline (rolling mean/stdev of that
    spread over pairs_zscore_window_days) is itself .shift(1)'d before use
    -- same "don't let today's own extreme value inflate the very baseline
    it's being compared against" discipline Squeeze_Zscore already
    established (see precompute_breakout_frame's own comment) -- otherwise
    a big divergence would partly absorb into its own rolling mean/stdev
    and dampen the very signal it's supposed to flag.

    Every same-sector peer's spread/z-score is computed (not just the
    eventual daily winner's), since that's cheap/vectorized regardless;
    only the winning partner's own value is kept per day via a row-wise
    select against that day's own Pair_Partner column."""
    df = precompute_breakout_frame(df, config)
    df["Pair_Partner"] = None
    df["Pair_Correlation"] = np.nan
    df["Pair_Spread_Zscore"] = np.nan

    if peer_prices is None or peer_prices.empty:
        return df

    peer_prices = peer_prices.reindex(df.index)
    ticker_returns = df["Close"].pct_change()
    ticker_cum_spread = df["Close"].pct_change(periods=config.pairs_spread_window_days)

    corr_by_peer: dict[str, pd.Series] = {}
    zscore_by_peer: dict[str, pd.Series] = {}
    for peer in peer_prices.columns:
        peer_close = peer_prices[peer]
        if peer_close.isna().all():
            continue
        peer_returns = peer_close.pct_change()
        corr_by_peer[peer] = ticker_returns.rolling(window=config.pairs_lookback_days).corr(peer_returns)

        peer_cum_spread = peer_close.pct_change(periods=config.pairs_spread_window_days)
        spread = (ticker_cum_spread - peer_cum_spread).replace([float("inf"), float("-inf")], pd.NA)

        spread_mean = spread.shift(1).rolling(window=config.pairs_zscore_window_days).mean()
        spread_std = spread.shift(1).rolling(window=config.pairs_zscore_window_days).std()
        zscore_by_peer[peer] = (spread - spread_mean) / spread_std

    if not corr_by_peer:
        return df

    corr_df = pd.DataFrame(corr_by_peer)
    zscore_df = pd.DataFrame(zscore_by_peer)

    # DataFrame.idxmax(axis=1) raises ValueError on a row that's entirely
    # NaN (every day before the pairs_lookback_days rolling window has
    # filled is exactly this, across every peer column simultaneously) --
    # compute idxmax/max only over rows with at least one real value, and
    # leave the rest as NaN rather than crashing.
    best_corr = pd.Series(np.nan, index=corr_df.index)
    best_partner = pd.Series(np.nan, index=corr_df.index, dtype=object)
    has_any_value = corr_df.notna().any(axis=1)
    if has_any_value.any():
        valid_subset = corr_df.loc[has_any_value]
        best_corr.loc[has_any_value] = valid_subset.max(axis=1)
        best_partner.loc[has_any_value] = valid_subset.idxmax(axis=1)
    # Explicitly null out days where the winning correlation is missing or
    # below the minimum threshold, rather than fabricating a partner.
    valid = best_corr.notna() & (best_corr >= config.pairs_min_correlation)
    best_partner = best_partner.where(valid)
    best_corr = best_corr.where(valid)

    best_zscore = pd.Series(np.nan, index=df.index)
    for peer in zscore_df.columns:
        is_winner = best_partner == peer
        best_zscore = best_zscore.where(~is_winner, zscore_df[peer])

    df["Pair_Partner"] = best_partner
    df["Pair_Correlation"] = best_corr.round(4)
    df["Pair_Spread_Zscore"] = best_zscore
    return df


def pairs_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the mean-reversion PAIRS strategy's dict for one row of a
    frame already built by precompute_pairs_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    pair_partner, pair_correlation, pair_spread_zscore = (
        last_row["Pair_Partner"], last_row["Pair_Correlation"], last_row["Pair_Spread_Zscore"],
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
    # the peer-spread z-score, not RSI) -- same treatment every other
    # non-RSI strategy (squeeze_breakout, breakout, ma_crossover) gives it.
    # Including it (unlike the first draft, which omitted it entirely) --
    # storage/signals.py's _build_document() unconditionally read row["RSI"]
    # for every strategy ever built before this one, a real crash caught
    # only once a real full-watchlist signal actually reached logging (see
    # improvements.txt item 82's live-wiring follow-up); fixing that one
    # call site AND restoring RSI here closes the whole gap, not just the
    # one instance found.
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

    # Missing partner/z-score (no peer_prices supplied, or no same-sector
    # peer cleared pairs_min_correlation) degrades to "never fires" rather
    # than excluding the ticker outright -- same "optional data missing
    # doesn't crash, just no signal" convention every other strategy uses.
    if pd.isna(pair_partner) or pd.isna(pair_spread_zscore):
        pair_signal = False
        pair_spread_zscore_out = None
        signal_strength_pct = 0.0
    else:
        pair_spread_zscore = float(pair_spread_zscore)
        pair_signal = bool(pair_spread_zscore <= config.pairs_zscore_entry_max)
        pair_spread_zscore_out = round(pair_spread_zscore, 3)
        # How far below the trigger the z-score fell -- same "distance past
        # the trigger" differentiating-term role Signal_Strength_Pct plays
        # for squeeze_breakout/momentum_burst, just in z-score units here.
        signal_strength_pct = (
            round(config.pairs_zscore_entry_max - pair_spread_zscore, 3) if pair_signal else 0.0
        )

    # Buy_Price = today's own Close -- same "confirm and enter near market"
    # convention squeeze_breakout/ma_crossover use, not a discount-limit
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
        "Pair_Partner": None if pd.isna(pair_partner) else str(pair_partner),
        "Pair_Correlation": None if pd.isna(pair_correlation) else round(float(pair_correlation), 4),
        "Pair_Spread_Zscore": pair_spread_zscore_out,
        "Pair_Signal": pair_signal,
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


def compute_pairs_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    peer_prices: pd.DataFrame | None = None,
) -> dict:
    """Compute mean-reversion PAIRS levels for one ticker's OHLCV history --
    see precompute_pairs_frame()'s own docstring for the full mechanism
    (partner selection via rolling correlation, spread z-score). Thin
    wrapper over precompute_pairs_frame()/pairs_levels_from_frame() -- kept
    as a single-call convenience for the live dashboard, matching every
    other strategy's same rationale (see compute_squeeze_breakout_levels()).

    `peer_prices` should be a wide DataFrame of Close prices for every
    OTHER ticker in this ticker's own sector (this ticker's own column
    excluded) -- caller's job to build from the already-fetched watchlist
    bundle, same as market_data.py's own sector_data resolution. Without
    it, Pair_Signal always reads False (no partner data)."""
    frame = precompute_pairs_frame(df, peer_prices, config)
    as_of = frame.index[-1]
    return pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_momentum_rank_frame(panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own trailing return,
    computed ONCE for the whole universe -- the key structural difference
    from pairs' precompute_pairs_frame(), which recomputes its cross-ticker
    correlation math separately inside every single ticker's own call.
    Percentile rank is inherently a whole-universe operation (a ticker's
    rank today depends on every OTHER ticker's return that same day), so
    doing it once here and having every ticker's own precompute_momentum_frame()
    just look up its own already-computed column is both cheaper and more
    correct than recomputing it per ticker.

    `panel` should be a wide DataFrame of Close prices, one column per
    ticker in the ranking universe (see market_data.build_momentum_panel()
    for the live-path caller, or the equivalent inline construction in
    optimize.py/benchmark_random_entry.py for backtest callers) -- no
    sector partitioning, unlike pairs' own per-sector panels, since ranking
    is universe-wide by design.

    Returns a DataFrame the same shape as `panel` (dates x tickers), each
    cell = that ticker's trailing-return percentile (0-100 scale, matching
    best_ideas.compute_sector_rs_scores()'s identical rank(pct=True)*100
    convention) on that day. A ticker with fewer than `lookback_days` of
    prior history reads NaN until its trailing return itself becomes
    computable, same "missing data doesn't fabricate a signal" convention
    as every other optional field in this codebase."""
    trailing_return = panel.pct_change(periods=lookback_days)
    return trailing_return.rank(axis=1, pct=True) * 100


def precompute_momentum_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional MOMENTUM RANK strategy's
    columns -- built on top of precompute_breakout_frame() (reused
    wholesale for SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/
    liquidity gates every strategy shares).

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_momentum_rank_frame() over every ticker's Close prices --
    unlike pairs' `peer_prices` (a DataFrame of every OTHER ticker, with
    correlation computed per-ticker inside precompute_pairs_frame()), the
    rank computation itself is inherently a whole-universe operation done
    ONCE by the caller; this function only reads this ticker's own
    already-computed percentile, reindexed to df's own date range. Missing
    (None) degrades to "never fires" rather than excluding the ticker
    outright, same convention as pairs' missing peer_prices."""
    df = precompute_breakout_frame(df, config)
    df["Momentum_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def momentum_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional MOMENTUM RANK strategy's dict for one row
    of a frame already built by precompute_momentum_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Same macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    momentum_percentile = last_row["Momentum_Percentile"]
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
    # the universe-wide percentile rank, not RSI) -- same treatment every
    # other non-RSI strategy (squeeze_breakout, breakout, ma_crossover,
    # pairs) gives it, per the real storage/signals.py bug pairs' own
    # rollout caught (row["RSI"] bracket access assumed every strategy
    # carries this key -- see pairs_levels_from_frame()'s identical comment).
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

    # Missing percentile (no rank_column supplied, or too little history
    # for this ticker's own trailing return yet) degrades to "never fires"
    # rather than excluding the ticker outright -- same "optional data
    # missing doesn't crash, just no signal" convention every other
    # strategy uses.
    if pd.isna(momentum_percentile):
        momentum_signal = False
        momentum_percentile_out = None
        signal_strength_pct = 0.0
    else:
        momentum_percentile = float(momentum_percentile)
        momentum_signal = bool(momentum_percentile >= config.momentum_top_percentile_min)
        momentum_percentile_out = round(momentum_percentile, 2)
        # How far past the trigger the percentile cleared -- same
        # "distance past the trigger" differentiating-term role
        # Signal_Strength_Pct plays for squeeze_breakout/pairs, just in
        # percentile points here.
        signal_strength_pct = (
            round(momentum_percentile - config.momentum_top_percentile_min, 2) if momentum_signal else 0.0
        )

    # Buy_Price = today's own Close -- same "confirm and enter near market"
    # convention squeeze_breakout/ma_crossover/pairs use, not a discount-
    # limit like RSI's structural-support wait. Distance_to_Buy_Pct is
    # therefore always 0 by construction, same schema-compatible treatment
    # those strategies already established.
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
        "Momentum_Percentile": momentum_percentile_out,
        "Momentum_Signal": momentum_signal,
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


def compute_momentum_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional MOMENTUM RANK levels for one ticker's OHLCV
    history -- see precompute_momentum_frame()'s own docstring for the full
    mechanism. Thin wrapper over precompute_momentum_frame()/
    momentum_levels_from_frame() -- kept as a single-call convenience for
    the live dashboard, matching every other strategy's same rationale (see
    compute_pairs_levels()).

    `rank_column` should be this ticker's own column already sliced from a
    shared universe-wide rank frame (see compute_momentum_rank_frame()) --
    caller's job to build from the already-fetched watchlist bundle, same
    as market_data.py's own pair_price_panels resolution. Without it,
    Momentum_Signal always reads False (no percentile data)."""
    frame = precompute_momentum_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return momentum_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


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


# LLM-invented strategy research (2026-08-22, improvements.txt item 93) --
# the LLM composes a structured JSON rule (which already-computed
# indicators to condition on, which already-built exit mechanic to use),
# never generates or executes code. See llm_strategy_research.py for the
# proposal/validation/orchestration layer; everything here is the generic,
# human-written interpreter every rule runs through.

KNOWN_INDICATOR_FIELDS = (
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "ATR", "ADX", "SMA_TREND", "AvgVolume", "AvgVolume_Prior",
    "Highest_High", "OBV_Zscore", "Squeeze_Zscore", "Relative_Strength",
)  # every column precompute_breakout_frame() actually guarantees (raw OHLCV
   # always; Relative_Strength only when market_df is supplied, same as
   # every other caller of that function) -- deliberately NOT the full set
   # every OTHER strategy's own extension adds (e.g. Recent_Min_Squeeze_Zscore
   # is squeeze_breakout-specific, Insider_Purchase_Value needs external
   # fetched data) -- this generic engine only calls the shared base, so
   # only the shared base's own columns are honest to offer.

_ALLOWED_CONDITION_OPS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
}


def evaluate_llm_rule_conditions(frame: pd.DataFrame, rule: dict) -> pd.Series:
    """Pure, vectorized evaluation of one LLM-proposed rule's `conditions`
    against an already-precomputed frame (any frame carrying
    KNOWN_INDICATOR_FIELDS' columns -- precompute_llm_strategy_frame()'s
    own output). Each condition is `frame[field] {op} value`; combined via
    `rule["logic"]` ("AND" or "OR") across every condition.

    Raises ValueError (never silently ignores or coerces) on any
    unrecognized field/op/logic, a field not actually present in this
    specific frame (e.g. Relative_Strength requested but market_df wasn't
    supplied upstream), or a non-numeric condition value -- same
    "malformed input is never more trustworthy than no input" discipline
    llm_agent._parse_response() already applies to raw LLM JSON elsewhere
    in this codebase. Callers (llm_strategy_research.validate_rule()) are
    expected to catch and reject a malformed rule BEFORE it ever reaches
    a real backtest.

    NaN comparisons (insufficient warmup history for a given indicator)
    evaluate to False, never True -- a condition can only fire once its
    own indicator has real data, same as every other strategy's own
    NaN-during-warmup handling."""
    conditions = rule.get("conditions") or []
    if not conditions:
        raise ValueError("rule has no conditions")
    logic = rule.get("logic", "AND")
    if logic not in ("AND", "OR"):
        raise ValueError(f"unknown logic: {logic!r} (expected 'AND' or 'OR')")

    masks = []
    for cond in conditions:
        field = cond.get("field")
        op = cond.get("op")
        value = cond.get("value")
        if field not in KNOWN_INDICATOR_FIELDS:
            raise ValueError(f"unknown field: {field!r}")
        if field not in frame.columns:
            raise ValueError(f"field {field!r} not present in this frame (was market_df supplied?)")
        if op not in _ALLOWED_CONDITION_OPS:
            raise ValueError(f"unknown op: {op!r} (expected one of {sorted(_ALLOWED_CONDITION_OPS)})")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"condition value must be numeric, got {value!r}")
        masks.append(_ALLOWED_CONDITION_OPS[op](frame[field], value))

    combined = masks[0]
    for mask in masks[1:]:
        combined = (combined & mask) if logic == "AND" else (combined | mask)
    return combined.fillna(False)


def rule_exit_to_config(rule: dict, config: TradingConfig = DEFAULT_CONFIG) -> TradingConfig:
    """Builds a per-rule effective TradingConfig from `rule["exit"]`'s own
    chosen exit mechanic/params, reusing TradingConfig's EXISTING
    atr_take_profit_multiplier/stop_loss_atr_multiplier/
    trailing_stop_atr_multiplier/trailing_stop_enabled/max_holding_days
    fields -- the LLM only SELECTS and PARAMETERIZES among this project's
    already-built, already-tested exit mechanics
    (swingtrade.settle_trade()/settle_trade_with_trailing()), never
    invents new settlement code. Every numeric param is expected to
    already be validated/clamped by llm_strategy_research.validate_rule()
    before this is called -- this function trusts its input, it doesn't
    re-validate it."""
    exit_spec = rule["exit"]
    overrides = {
        "atr_take_profit_multiplier": exit_spec["take_profit_atr_multiplier"],
        "stop_loss_atr_multiplier": exit_spec["stop_loss_atr_multiplier"],
        "trailing_stop_enabled": exit_spec["type"] == "trailing_stop",
    }
    if exit_spec["type"] == "trailing_stop":
        overrides["trailing_stop_atr_multiplier"] = exit_spec["trailing_stop_atr_multiplier"]
    if exit_spec["type"] == "time_based":
        overrides["max_holding_days"] = exit_spec["max_holding_days"]
    return TradingConfig(**{**config.to_dict(), **overrides})


def precompute_llm_strategy_frame(
    df: pd.DataFrame,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute for one LLM-proposed rule -- built on top of
    precompute_breakout_frame() (reused wholesale, exactly like every
    other strategy, for SMA_TREND/ATR/RSI/etc. and the shared
    macro-uptrend/liquidity gate every strategy respects unconditionally).
    `LLM_Strategy_Signal` is the rule's own conditions evaluated via
    evaluate_llm_rule_conditions() -- the ONLY thing this rule actually
    controls; the macro-uptrend/liquidity checks stay enforced identically
    for every rule, same as every other strategy family, in
    llm_strategy_levels_from_frame() below."""
    df = precompute_breakout_frame(df, config, market_df=market_df)
    df["LLM_Strategy_Signal"] = evaluate_llm_rule_conditions(df, rule)
    return df


def llm_strategy_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract one LLM-proposed rule's dict for one row of a frame already
    built by precompute_llm_strategy_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale) -- NOT something any
    rule can configure around.

    `config` is expected to already be the RULE's own effective config
    (see rule_exit_to_config()) -- Sell_Price/Stop_Loss below read
    config.atr_take_profit_multiplier/stop_loss_atr_multiplier directly,
    same as every other strategy's own levels_from_frame()."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    llm_signal = bool(last_row["LLM_Strategy_Signal"])
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
    # the LLM's own rule, not RSI specifically) -- same treatment every
    # other non-RSI strategy gives it.
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

    # Buy_Price = today's own Close -- same "already happening, buy near
    # market" convention squeeze_breakout/pairs/insider_buying use.
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
        "LLM_Strategy_Signal": llm_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_llm_strategy_levels(
    ticker: str,
    df: pd.DataFrame,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
) -> dict:
    """Compute one LLM-proposed rule's levels for one ticker's OHLCV
    history -- see precompute_llm_strategy_frame()'s own docstring for the
    full mechanism. Thin wrapper, same pattern as every other
    compute_X_levels() (see compute_squeeze_breakout_levels())."""
    effective_config = rule_exit_to_config(rule, config)
    frame = precompute_llm_strategy_frame(df, rule, effective_config, market_df=market_df)
    as_of = frame.index[-1]
    return llm_strategy_levels_from_frame(ticker, frame, as_of, effective_config, next_earnings_date, top_headline)


def compute_squeeze_breakout_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> dict:
    """Compute squeeze-breakout levels for one ticker's OHLCV history -- a
    seventh strategy, built to fire MORE OFTEN than momentum_burst (which
    proved thin/fragile under tuning and entry-fill sensitivity checks --
    see improvements.txt items 35-37) via a materially different trigger:
    fires when volatility was recently CONTRACTED (a squeeze -- reusing
    Squeeze_Zscore, already computed for breakout's own optional filter)
    and today shows a real directional EXPANSION (a meaningful same-day
    gain). Deliberately does NOT also require a fresh high over any
    window (an earlier design draft did -- rejected: requiring both a
    squeeze AND a fresh high is the intersection of two conditions,
    necessarily rarer than either alone, defeating the point of a
    faster-firing signal) and does NOT require volume confirmation
    (unlike momentum_burst) -- kept deliberately distinct from the
    existing fast-firing candidate rather than a near-duplicate.

    Squeeze_Signal requires BOTH Recent_Min_Squeeze_Zscore <=
    squeeze_breakout_zscore_max (a genuine contraction within the last
    squeeze_breakout_lookback_days) AND Day_Gain_Pct >=
    squeeze_breakout_gain_pct_min (a real expansion today).

    Buy_Price is today's own Close (see squeeze_breakout_levels_from_frame's
    inline comment for why -- same convention as week52_high/momentum_burst,
    NOT a resting level to wait for).

    The returned dict is schema-compatible with every other strategy's
    (Buy_Price, Sell_Price, Stop_Loss, RRR, Distance_to_Buy_Pct,
    Catalyst_Warning, etc. all present) -- only add_squeeze_breakout_trade_score
    (swingtrade/scoring.py) differs downstream. RSI is informational only,
    same treatment as the others.

    Thin wrapper over precompute_squeeze_breakout_frame()/squeeze_breakout_levels_from_frame()
    -- kept as a single-call convenience for the live dashboard, matching
    every other strategy's same rationale. NOT called from ingest.py in
    this v1 -- this strategy is dashboard-only/experimental until it
    passes the same random-entry-timing validation every other strategy
    here was held to (see benchmark_random_entry.py).

    `market_df` (added alongside the optional sharpening filters --
    improvements.txt item 42/43) is needed for Relative_Strength; None
    when not supplied, same backward-compatible treatment
    compute_breakout_levels()/compute_adx_trend_entry_levels() use.
    """
    frame = precompute_squeeze_breakout_frame(df, config, market_df=market_df, sector_df=sector_df)
    as_of = frame.index[-1]
    return squeeze_breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


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


def review_holding(
    ticker: str,
    df: pd.DataFrame,
    avg_cost: float,
    config: TradingConfig = DEFAULT_CONFIG,
) -> dict:
    """Evaluate an already-held position against the system's own mechanical
    exit rules, anchored to your actual avg_cost rather than a freshly
    computed structural support level. Same ATR-based stop/target formulas
    compute_levels uses for a new candidate -- Stop_Loss = avg_cost -
    stop_loss_atr_multiplier * ATR, Sell_Price (target) = avg_cost +
    atr_take_profit_multiplier * ATR -- just applied to a position you
    already own instead of one you're screening. Recommends a SELL if the
    current price has breached either level, HOLD otherwise. Informational,
    not a guarantee -- carries the exact same reliability caveats as every
    other signal this system produces, since it uses the same config.
    """
    df = df.copy()
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)

    last_row = df.iloc[-1]
    last_close = float(last_row["Close"])
    last_date = df.index[-1]
    atr = float(last_row["ATR"])
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")

    stop_loss = round(avg_cost - (config.stop_loss_atr_multiplier * atr), 2)
    sell_price = round(avg_cost + (config.atr_take_profit_multiplier * atr), 2)
    unrealized_pnl_pct = round((last_close - avg_cost) / avg_cost * 100, 2)

    if last_close <= stop_loss:
        recommendation = "SELL (stop breached)"
    elif last_close >= sell_price:
        recommendation = "SELL (target hit)"
    else:
        recommendation = "HOLD"

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Avg_Cost": round(avg_cost, 2),
        "Last_Close": round(last_close, 2),
        "ATR": round(atr, 2),
        "Stop_Loss": stop_loss,
        "Sell_Price": sell_price,
        "Unrealized_PnL_Pct": unrealized_pnl_pct,
        "Recommendation": recommendation,
    }
