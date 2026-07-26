"""
Interactive Streamlit dashboard for a mechanical swing-trading watchlist scan.

This app does NOT predict price direction and is NOT investment advice.
It applies a fixed, deterministic rule set to historical price data to compute,
for every scanned ticker:

  1. A limit BUY price, defined as the structural support level: the lowest
     daily Low over the last SUPPORT_LOOKBACK_DAYS trading days (recent swing
     low). The 20-day SMA discount price is still reported for context but no
     longer used to set the buy price.

     Buy_Signal is True only when both are satisfied:
       a) the last close is at/below the structural support buy price, and
       b) the 14-day RSI is below RSI_OVERSOLD_THRESHOLD (mathematically
          oversold), computed via pandas_ta.

  2. A limit SELL (take-profit) price, defined as
       buy_price + (ATR_TAKE_PROFIT_MULTIPLIER * ATR14)
     where ATR14 is the 14-day Average True Range (via pandas_ta), so the
     take-profit target scales with the stock's actual current volatility
     instead of a static percentage.

  3. A Stop_Loss level, defined as buy_price - (1.0 * ATR14), and a
     Risk-to-Reward Ratio (RRR) = (sell_price - buy_price) / (buy_price - stop_loss).

  4. Distance_to_Buy_Pct: how far the last close is from the buy trigger,
       ((last_close - buy_price) / buy_price) * 100
     Positive = still above the buy trigger (needs to fall further).
     Negative/zero = last close is already at or below the buy trigger.

  5. Catalyst awareness: the next upcoming earnings date and the most recent
     news headline (yfinance). Catalyst_Warning is True when the next
     earnings date falls within EARNINGS_WARNING_DAYS days, flagging a
     volatile binary event -- shown in the table (highlighted red) rather
     than removed, so it can inform rather than hide the decision.

  6. Macro trend and liquidity gates: tickers are excluded from the results
     when the last close is below the 200-day SMA (macro downtrend) or when
     20-day average dollar volume is below MIN_DOLLAR_VOLUME (too illiquid to
     swing-trade safely). A broad-market gate (MARKET_INDEX_TICKER vs. its
     own 200-day SMA) halts the whole scan when the index itself is in a
     macro downtrend.

  7. Shares_To_Buy = position_budget / buy_price, rounded to
     FRACTIONAL_SHARE_DECIMALS places -- a fixed-dollar-budget position size,
     recalculated live from the sidebar "Position Budget" input.

  8. Trade_Score (0-100) blends Risk-to-Reward Ratio, RSI, and how close the
     last close is to the buy trigger into a single priority score, mapped to
     a Signal of Strong Buy / Buy / Watch / Ignore.

Every number is derived mechanically from the fetched data using the constants
below.
"""

import re
import time
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf

# ----------------------------- Configuration -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"

LOOKBACK_PERIOD = "1y"           # data window to fetch (needs 200d+ for SMA200)
MA_WINDOW = 20                   # moving-average window (trading days), context only
SUPPORT_LOOKBACK_DAYS = 20       # window to scan for the structural swing low
MA_DISCOUNT_PCT = 0.05           # 5% below the 20-day MA, context only

RSI_WINDOW = 14                  # RSI lookback (trading days)
ATR_WINDOW = 14                  # ATR lookback (trading days)
RSI_OVERSOLD_THRESHOLD = 45      # buy signal requires RSI below this
ATR_TAKE_PROFIT_MULTIPLIER = 1.5 # sell_price = buy_price + multiplier * ATR14
STOP_LOSS_ATR_MULTIPLIER = 1.0   # stop_loss = buy_price - multiplier * ATR14

SMA_TREND_WINDOW = 200           # macro trend filter window (trading days)
VOLUME_LOOKBACK_DAYS = 20        # window for average volume / liquidity check
MIN_DOLLAR_VOLUME = 5_000_000    # exclude tickers below this 20d $ volume
DEFAULT_POSITION_BUDGET = 250    # default $ sidebar value for position sizing
DEFAULT_TOTAL_CASH = 5_000       # default $ sidebar value for total available cash
FRACTIONAL_SHARE_DECIMALS = 4    # precision for fractional-share sizing

MARKET_INDEX_TICKER = "SPY"      # broad-market proxy for the macro gate

NEWS_HEADLINE_COUNT = 3          # recent news articles to fetch per ticker
EARNINGS_WARNING_DAYS = 14       # flag Catalyst_Warning if earnings within N days

REQUEST_DELAY_SEC = 0.5          # pause between API calls to avoid rate-limiting
SCAN_CACHE_TTL_SEC = 900         # how long a scan result stays cached (15 min)

# Trade_Score weights (must sum to 100)
RRR_SCORE_WEIGHT = 40            # points for Risk-to-Reward Ratio
RRR_SCORE_CAP = 4.0              # RRR at/above this earns full RRR points
RSI_SCORE_WEIGHT = 40            # points for RSI (oversold-ness)
RSI_SCORE_FLOOR = 30             # RSI at/below this earns full RSI points
RSI_SCORE_CEILING = 60           # RSI at/above this earns zero RSI points
DISTANCE_SCORE_WEIGHT = 20       # points for proximity to the buy trigger
DISTANCE_SCORE_CAP_PCT = 20      # distance at/above this earns zero points

SIGNAL_COLORS = {
    "Strong Buy": "background-color: #1b7a3d; color: #ffffff;",
    "Buy": "background-color: #8bc34a; color: #1a1a1a;",
    "Watch": "background-color: #f6c945; color: #1a1a1a;",
    "Ignore": "background-color: #e57373; color: #1a1a1a;",
    "Insufficient Funds": "background-color: #78909c; color: #ffffff;",
}
CATALYST_WARNING_STYLE = "background-color: #c62828; color: #ffffff; font-weight: 600;"

DISPLAY_COLUMNS = [
    "Ticker", "Signal", "Trade_Score", "Last_Close", "Buy_Price", "Stop_Loss",
    "Sell_Price", "RRR", "RSI", "ATR", "Distance_to_Buy_Pct", "Shares_To_Buy",
    "Est_Cost", "Next_Earnings_Date", "Catalyst_Warning", "Top_Headline", "As_Of",
]

# ---------------------------------------------------------------------------


def read_tickers(path: Path) -> list[str]:
    """Read tickers from watchlist.txt. Supports either:
      - a JSON object/array with {"ticker": "..."} entries (or a "watchlist" key), or
      - a plain text file with one ticker per line.
    Whitespace is stripped and empty lines/entries are ignored.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()

    if stripped.startswith("{") or stripped.startswith("["):
        import json
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            entries = data.get("watchlist", []) if isinstance(data, dict) else data
            tickers = []
            for entry in entries:
                if isinstance(entry, dict) and entry.get("ticker"):
                    tickers.append(str(entry["ticker"]).strip().upper())
                elif isinstance(entry, str) and entry.strip():
                    tickers.append(entry.strip().upper())
            if tickers:
                return tickers

    # Fallback: plain text, one ticker per line
    return [line.strip().upper() for line in raw.splitlines() if line.strip()]


def parse_ticker_text(raw: str) -> list[str]:
    """Parse a free-form, user-pasted ticker list (newline and/or comma separated)."""
    if not raw or not raw.strip():
        return []
    seen = set()
    tickers = []
    for token in re.split(r"[\s,]+", raw.strip()):
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=LOOKBACK_PERIOD, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no data returned for ticker '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_next_earnings_date(ticker_obj: yf.Ticker, now_utc: pd.Timestamp):
    """Return the next upcoming earnings date (tz-aware, UTC) or None if
    unavailable / no future date is listed."""
    try:
        earnings = ticker_obj.get_earnings_dates(limit=12)
    except Exception:
        return None
    if earnings is None or earnings.empty:
        return None
    dates_utc = earnings.index.tz_convert("UTC")
    future_dates = dates_utc[dates_utc >= now_utc]
    if future_dates.empty:
        return None
    return future_dates.min()


def get_recent_headlines(ticker_obj: yf.Ticker, count: int = NEWS_HEADLINE_COUNT) -> list[str]:
    """Return up to `count` most recent news headline titles."""
    try:
        news_items = ticker_obj.news or []
    except Exception:
        return []

    titles = []
    for item in news_items[:count]:
        title = None
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, dict):
                title = content.get("title")
            if not title:
                title = item.get("title")
        if title:
            titles.append(str(title))
    return titles


def check_market_uptrend(index_ticker: str = MARKET_INDEX_TICKER) -> tuple[bool, float, float]:
    """Return (is_uptrend, last_close, sma200) for the broad-market index."""
    df = fetch_data(index_ticker)
    sma200 = df["Close"].rolling(window=SMA_TREND_WINDOW).mean().iloc[-1]
    last_close = float(df["Close"].iloc[-1])
    sma200 = float(sma200)
    if pd.isna(sma200):
        raise RuntimeError(f"insufficient history to compute {SMA_TREND_WINDOW}-day SMA for {index_ticker}")
    return last_close >= sma200, last_close, sma200


def compute_levels(ticker: str, df: pd.DataFrame, ticker_obj: yf.Ticker) -> dict:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(window=MA_WINDOW).mean()
    df["SMA200"] = df["Close"].rolling(window=SMA_TREND_WINDOW).mean()
    df["RSI"] = ta.rsi(df["Close"], length=RSI_WINDOW)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_WINDOW)
    df["AvgVolume"] = df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS).mean()

    last_row = df.iloc[-1]
    last_close = float(last_row["Close"])
    last_date = df.index[-1]
    sma20 = float(last_row["SMA20"])
    sma200 = float(last_row["SMA200"])
    rsi = float(last_row["RSI"])
    atr = float(last_row["ATR"])
    avg_volume = float(last_row["AvgVolume"])
    if pd.isna(sma20):
        raise RuntimeError(f"insufficient history to compute {MA_WINDOW}-day SMA")
    if pd.isna(sma200):
        raise RuntimeError(f"insufficient history to compute {SMA_TREND_WINDOW}-day SMA")
    if pd.isna(rsi):
        raise RuntimeError(f"insufficient history to compute {RSI_WINDOW}-day RSI")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {ATR_WINDOW}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {VOLUME_LOOKBACK_DAYS}-day average volume")

    if last_close < sma200:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA200 {sma200:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < MIN_DOLLAR_VOLUME:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${MIN_DOLLAR_VOLUME:,.0f})"
        )

    recent_window = df.tail(SUPPORT_LOOKBACK_DAYS)
    support_level = float(recent_window["Low"].min())
    support_date = recent_window["Low"].idxmin()
    ma_discount_price = sma20 * (1 - MA_DISCOUNT_PCT)

    buy_price = round(support_level, 2)
    buy_basis = f"structural swing low (last {SUPPORT_LOOKBACK_DAYS}d, {support_date.date()})"

    sell_price = round(buy_price + (ATR_TAKE_PROFIT_MULTIPLIER * atr), 2)
    stop_loss = round(buy_price - (STOP_LOSS_ATR_MULTIPLIER * atr), 2)
    risk = buy_price - stop_loss
    # risk can be <= 0 only if ATR rounds to 0 (flat price action); fall back
    # to 0.0 instead of None so the column stays numeric in the DataFrame/CSV.
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    distance_to_buy_pct = ((last_close - buy_price) / buy_price) * 100
    buy_signal = (last_close <= buy_price) and (rsi < RSI_OVERSOLD_THRESHOLD)

    now_utc = pd.Timestamp.now(tz="UTC")
    next_earnings = get_next_earnings_date(ticker_obj, now_utc)
    if next_earnings is not None:
        days_to_earnings = (next_earnings - now_utc).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= EARNINGS_WARNING_DAYS
        next_earnings_date = next_earnings.date()
    else:
        catalyst_warning = False
        next_earnings_date = None

    headlines = get_recent_headlines(ticker_obj)
    top_headline = headlines[0] if headlines else ""

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "SMA20": round(sma20, 2),
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
        "Next_Earnings_Date": next_earnings_date,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def signal_for_score(score: float) -> str:
    if score > 80:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 40:
        return "Watch"
    return "Ignore"


def add_trade_score(df: pd.DataFrame) -> pd.DataFrame:
    """Blend RRR, RSI, and Distance_to_Buy_Pct into a 0-100 Trade_Score and
    map it to a Strong Buy / Buy / Watch / Ignore Signal."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=RRR_SCORE_CAP) / RRR_SCORE_CAP) * RRR_SCORE_WEIGHT

    rsi_clipped = df["RSI"].clip(lower=RSI_SCORE_FLOOR, upper=RSI_SCORE_CEILING)
    rsi_score = (
        (RSI_SCORE_CEILING - rsi_clipped) / (RSI_SCORE_CEILING - RSI_SCORE_FLOOR)
    ) * RSI_SCORE_WEIGHT

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=DISTANCE_SCORE_CAP_PCT)
    distance_score = (1 - distance_clipped / DISTANCE_SCORE_CAP_PCT) * DISTANCE_SCORE_WEIGHT

    df["Trade_Score"] = (rrr_score + rsi_score + distance_score).round(1)
    df["Signal"] = df["Trade_Score"].apply(signal_for_score)
    return df


def allocate_capital(df: pd.DataFrame, total_cash: float) -> tuple[pd.DataFrame, float]:
    """Greedily allocate total_cash down the (already Trade_Score-sorted) list
    of Strong Buy / Buy signals, in order. A trade whose Est_Cost exceeds the
    remaining cash is relabeled Insufficient Funds -- without consuming any
    cash or stopping the walk, so a cheaper trade further down the list can
    still be funded. Returns the updated DataFrame and total capital spent."""
    df = df.copy()
    remaining_cash = total_cash
    signals = df["Signal"].tolist()
    costs = df["Est_Cost"].tolist()
    for i in range(len(df)):
        if signals[i] not in ("Strong Buy", "Buy"):
            continue
        if costs[i] <= remaining_cash:
            remaining_cash -= costs[i]
        else:
            signals[i] = "Insufficient Funds"
    df["Signal"] = signals
    capital_allocated = round(total_cash - remaining_cash, 2)
    return df, capital_allocated


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Checking broad-market macro trend...")
def cached_market_uptrend() -> tuple[bool, float, float]:
    return check_market_uptrend()


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Scanning watchlist...")
def scan_watchlist(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Fetch data and compute levels for every ticker. This is the only
    network-heavy step and is cached so sidebar tweaks (budget, sorting)
    don't re-trigger a full re-scan."""
    results = []
    skipped = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_data(ticker)
            ticker_obj = yf.Ticker(ticker)
            results.append(compute_levels(ticker, df, ticker_obj))
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    results_df = pd.DataFrame(results)
    return results_df, skipped


def style_results(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    formats = {
        "Trade_Score": "{:.1f}",
        "Last_Close": "{:.2f}",
        "Buy_Price": "{:.2f}",
        "Stop_Loss": "{:.2f}",
        "Sell_Price": "{:.2f}",
        "RRR": "{:.2f}",
        "RSI": "{:.1f}",
        "ATR": "{:.2f}",
        "Distance_to_Buy_Pct": "{:.2f}%",
        "Shares_To_Buy": "{:.4f}",
        "Est_Cost": "{:.2f}",
    }
    return (
        df.style
        .format(formats, na_rep="-")
        .map(lambda v: SIGNAL_COLORS.get(v, ""), subset=["Signal"])
        .map(lambda v: CATALYST_WARNING_STYLE if v else "", subset=["Catalyst_Warning"])
    )


def main():
    st.set_page_config(page_title="Swing-Trading Dashboard", layout="wide")
    st.title("Swing-Trading Dashboard")
    st.caption(
        "Mechanical, rule-based output from historical data -- not a forecast or "
        "recommendation. Verify live price/liquidity before placing orders."
    )

    with st.sidebar:
        st.header("Configuration")
        position_budget = st.number_input(
            "Position Budget ($)",
            min_value=1.0,
            value=float(DEFAULT_POSITION_BUDGET),
            step=10.0,
            help="Max $ allocated per trade; drives the fractional Shares_To_Buy column below.",
        )
        total_cash = st.number_input(
            "Total Available Cash ($)",
            min_value=0.0,
            value=float(DEFAULT_TOTAL_CASH),
            step=100.0,
            help="Capital pool spent greedily down the Trade_Score-ranked Buy/Strong Buy list; "
                 "trades that no longer fit are marked Insufficient Funds.",
        )
        default_ticker_text = "\n".join(read_tickers(WATCHLIST_FILE)) if WATCHLIST_FILE.exists() else ""
        ticker_text = st.text_area(
            "Watchlist (one ticker per line, or comma-separated)",
            value=default_ticker_text,
            height=280,
        )
        tickers = tuple(parse_ticker_text(ticker_text))
        st.caption(f"{len(tickers)} ticker(s) loaded.")

    if not tickers:
        st.warning("No tickers to scan. Paste some tickers in the sidebar watchlist box.")
        st.stop()

    try:
        market_uptrend, market_close, market_sma200 = cached_market_uptrend()
    except Exception as exc:
        st.error(f"Could not evaluate {MARKET_INDEX_TICKER} macro trend: {exc}")
        st.stop()

    if not market_uptrend:
        st.error(
            f"**{MARKET_INDEX_TICKER} is in a macro downtrend** "
            f"(Last_Close {market_close:.2f} < SMA200 {market_sma200:.2f}). "
            "Individual structural-support levels are unreliable when the broad market "
            "itself is breaking down -- watchlist analysis has been skipped."
        )
        st.stop()
    st.success(f"{MARKET_INDEX_TICKER} is above its 200-day SMA ({market_close:.2f} >= {market_sma200:.2f}).")

    results_df, skipped = scan_watchlist(tickers)

    if results_df.empty:
        st.error("No tickers were successfully analyzed. See skipped tickers below.")
        if skipped:
            with st.expander(f"Skipped tickers ({len(skipped)})"):
                st.dataframe(pd.DataFrame(skipped, columns=["Ticker", "Reason"]), hide_index=True)
        st.stop()

    results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(FRACTIONAL_SHARE_DECIMALS)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
    results_df = add_trade_score(results_df)
    results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)
    results_df, capital_allocated = allocate_capital(results_df, total_cash)
    remaining_idle_cash = round(total_cash - capital_allocated, 2)

    strong_buys = int((results_df["Signal"] == "Strong Buy").sum())
    catalyst_warnings = int(results_df["Catalyst_Warning"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickers Scanned", len(tickers))
    col2.metric("Successfully Analyzed", len(results_df))
    col3.metric("Active Strong Buys", strong_buys)
    col4.metric("Catalyst Warnings", catalyst_warnings)

    col5, col6, col7 = st.columns(3)
    col5.metric("Starting Cash", f"${total_cash:,.2f}")
    col6.metric("Capital Allocated to Orders", f"${capital_allocated:,.2f}")
    col7.metric("Remaining Idle Cash", f"${remaining_idle_cash:,.2f}")

    st.subheader("Scan Results")
    st.dataframe(style_results(results_df[DISPLAY_COLUMNS]), width="stretch", hide_index=True)

    st.download_button(
        "Download full results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="swing_orders.csv",
        mime="text/csv",
    )

    if skipped:
        with st.expander(f"Skipped tickers ({len(skipped)})"):
            st.dataframe(pd.DataFrame(skipped, columns=["Ticker", "Reason"]), hide_index=True)


if __name__ == "__main__":
    main()
