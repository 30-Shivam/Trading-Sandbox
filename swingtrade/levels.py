"""Pure technical-level calculations. No network calls, no UI -- takes
already-fetched OHLCV data (and already-resolved catalyst info) in, returns
plain dicts/tuples out. Safe to call from Streamlit, the settlement job, or a
backtest loop replaying years of historical bars.
"""

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
    """
    df = df.copy()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()
    df["AvgVolume_Prior"] = df["AvgVolume"].shift(1)
    df["Highest_High"] = df["High"].rolling(window=config.breakout_lookback_days).max().shift(1)
    df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"], length=config.adx_window)[f"ADX_{config.adx_window}"]

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
    frame = precompute_breakout_frame(df, config, market_df=market_df)
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
