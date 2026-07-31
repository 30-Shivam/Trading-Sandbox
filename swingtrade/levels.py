"""Pure technical-level calculations. No network calls, no UI -- takes
already-fetched OHLCV data (and already-resolved catalyst info) in, returns
plain dicts/tuples out. Safe to call from Streamlit, the settlement job, or a
backtest loop replaying years of historical bars.
"""

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
    """
    df = df.copy()
    df["SMA"] = df["Close"].rolling(window=config.ma_window).mean()
    df["SMA_TREND"] = df["Close"].rolling(window=config.sma_trend_window).mean()
    df["RSI"] = ta.rsi(df["Close"], length=config.rsi_window)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
    df["AvgVolume"] = df["Volume"].rolling(window=config.volume_lookback_days).mean()

    last_row = df.iloc[-1]
    last_date = df.index[-1]
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

    oversold_streak_days = 0
    for rsi_val in reversed(df["RSI"].tolist()):
        if pd.isna(rsi_val) or rsi_val >= config.rsi_oversold_threshold:
            break
        oversold_streak_days += 1
    extended_decline_warning = oversold_streak_days >= config.extended_decline_warning_days

    recent_window = df.tail(config.support_lookback_days)
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
    as_of = pd.Timestamp(last_date)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of).total_seconds() / 86400
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
