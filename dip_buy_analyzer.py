"""
Interactive Streamlit dashboard for a mechanical swing-trading watchlist scan.

This app does NOT predict price direction and is NOT investment advice.
It applies a fixed, deterministic rule set to historical price data to compute,
for every scanned ticker:

  1. A limit BUY price, defined as the structural support level: the lowest
     daily Low over the last support_lookback_days trading days (recent swing
     low). The SMA discount price is still reported for context but no
     longer used to set the buy price.

     Buy_Signal is True only when both are satisfied:
       a) the last close is at/below the structural support buy price, and
       b) the 14-day RSI is below rsi_oversold_threshold (mathematically
          oversold), computed via pandas_ta.

  2. A limit SELL (take-profit) price, defined as
       buy_price + (atr_take_profit_multiplier * ATR14)
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
     earnings date falls within earnings_warning_days days, flagging a
     volatile binary event -- shown in the table (highlighted red) rather
     than removed, so it can inform rather than hide the decision.

  6. Macro trend and liquidity gates: tickers are excluded from the results
     when the last close is below the 200-day SMA (macro downtrend) or when
     20-day average dollar volume is below min_dollar_volume (too illiquid to
     swing-trade safely). A broad-market gate (MARKET_INDEX_TICKER vs. its
     own 200-day SMA) halts the whole scan when the index itself is in a
     macro downtrend.

  7. Shares_To_Buy = position_budget / buy_price, rounded to
     fractional_share_decimals places -- a fixed-dollar-budget position size,
     recalculated live from the sidebar "Position Budget" input.

  8. Trade_Score (0-100) blends Risk-to-Reward Ratio, RSI, and how close the
     last close is to the buy trigger into a single priority score, mapped to
     a Signal of Strong Buy / Buy / Watch / Ignore.

  9. Every "Strong Buy"/"Buy" signal is logged to MongoDB's Trade_Signals
     collection (idempotent per ticker/day), using the *pre-allocation*
     Signal -- a personal cash shortfall ("Insufficient Funds") shouldn't be
     recorded as if the underlying technical signal changed. If MONGODB_URI
     isn't configured, the dashboard still works; logging is just skipped
     with a sidebar note.

The actual level/score/allocation math lives in the `swingtrade` package
(no yfinance/streamlit dependency there); persistence lives in the
`storage` package (no yfinance/streamlit dependency there either). This
file only handles data fetching, Streamlit UI, and wiring it all together.
"""

import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

import storage
import swingtrade

# ----------------------------- Configuration -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"

CONFIG = swingtrade.DEFAULT_CONFIG   # Phase 6 will load this from MongoDB System_Config

LOOKBACK_PERIOD = "1y"           # data window to fetch (needs 200d+ for SMA200)
MARKET_INDEX_TICKER = "SPY"      # broad-market proxy for the macro gate

NEWS_HEADLINE_COUNT = 3          # recent news articles to fetch per ticker

DEFAULT_POSITION_BUDGET = 250    # default $ sidebar value for position sizing
DEFAULT_TOTAL_CASH = 5_000       # default $ sidebar value for total available cash

REQUEST_DELAY_SEC = 0.5          # pause between API calls to avoid rate-limiting
SCAN_CACHE_TTL_SEC = 900         # how long a scan result stays cached (15 min)

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
    """Fetch the broad-market index and evaluate it via swingtrade.is_market_uptrend."""
    df = fetch_data(index_ticker)
    return swingtrade.is_market_uptrend(df, CONFIG)


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
            now_utc = pd.Timestamp.now(tz="UTC")
            next_earnings = get_next_earnings_date(ticker_obj, now_utc)
            headlines = get_recent_headlines(ticker_obj)
            top_headline = headlines[0] if headlines else ""
            levels = swingtrade.compute_levels(
                ticker, df, CONFIG, next_earnings_date=next_earnings, top_headline=top_headline
            )
            results.append(levels)
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


@st.cache_resource(show_spinner=False)
def init_storage() -> tuple[bool, str]:
    """One-time-per-process MongoDB connectivity check + index setup.
    Returns (ok, message) rather than raising, so a missing/unreachable
    database degrades the dashboard gracefully instead of crashing it."""
    try:
        storage.ensure_indexes()
        return True, ""
    except storage.MongoNotConfigured as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Could not connect to MongoDB: {exc}"


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

    results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(CONFIG.fractional_share_decimals)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
    results_df = swingtrade.add_trade_score(results_df, CONFIG)
    results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)

    # Log signals BEFORE the capital-allocation overlay: Trade_Signals should
    # reflect the underlying technical signal, not whether cash happened to
    # be available today.
    storage_ok, storage_message = init_storage()
    if storage_ok:
        try:
            logged_count = storage.log_trade_signals(results_df, CONFIG.to_dict())
            st.sidebar.caption(f"Logged {logged_count} signal(s) to MongoDB.")
        except Exception as exc:
            st.sidebar.warning(f"Signal logging failed: {exc}")
    else:
        st.sidebar.caption(f"MongoDB not connected ({storage_message}) -- signals aren't being logged.")

    results_df, capital_allocated = swingtrade.allocate_capital(results_df, total_cash)
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
