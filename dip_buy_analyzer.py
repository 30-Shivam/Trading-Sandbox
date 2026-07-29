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
     recalculated live from the sidebar "Position Budget" input. Capital is
     then allocated greedily down the Trade_Score-ranked list against "Total
     Available Cash": a trade that no longer fits the remaining cash is
     marked Insufficient Funds, and (if watchlist.txt's JSON form supplies a
     "sector" per ticker) a trade that would push its sector's cumulative
     spend past max_sector_allocation_pct of your whole portfolio value is
     marked Sector Limit Reached instead -- five Strong Buys in one sector on
     the same day are one concentrated bet wearing five tickers, not five
     independent ones, and nothing else in this pipeline catches that.
     "Whole portfolio value" includes the sidebar's "Current Holdings" box
     (persisted to MongoDB's Current_Holdings collection via a Save button,
     manually maintained rather than inferred from unsettled signals, since a
     logged signal doesn't guarantee you actually got filled) -- a sector
     you're already overweight in from prior holdings gets little or no new
     room today, even though holdings never reduce Total Available Cash
     itself.

  8. Trade_Score (0-100) blends Risk-to-Reward Ratio, RSI, and how close the
     last close is to the buy trigger into a single priority score, mapped to
     a Signal of Strong Buy / Buy / Watch / Ignore.

  9. Every "Strong Buy"/"Buy" signal is logged to MongoDB's Trade_Signals
     collection (idempotent per ticker/day), using the *pre-allocation*
     Signal -- a personal cash shortfall ("Insufficient Funds") shouldn't be
     recorded as if the underlying technical signal changed. If MONGODB_URI
     isn't configured, the dashboard still works; logging is just skipped
     with a sidebar note. The same scan-and-log pipeline also runs headless,
     independent of anyone having this dashboard open, via `ingest.py`
     (see ARCHITECTURE_PLAN.md Phase 7) -- both share the fetch logic in
     `market_data.py` and the config-loading logic in `config_loader.py` so
     they can never silently drift apart.

  10. All of the above run on whichever TradingConfig is currently "active"
      in MongoDB's System_Config (the output of optimize.py + a deliberate
      promote_config.py decision -- see ARCHITECTURE_PLAN.md Phase 5),
      re-checked periodically at the same cadence as the rest of the scan
      cache. If nothing is active yet, or Mongo is unreachable, this falls
      back to swingtrade.DEFAULT_CONFIG rather than crashing -- the sidebar
      always shows which one is actually in effect.

The actual level/score/allocation math lives in the `swingtrade` package
(no yfinance/streamlit dependency there); persistence lives in the
`storage` package (no yfinance/streamlit dependency there either). This
file only handles data fetching, Streamlit UI, and wiring it all together.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import config_loader
import market_data
import storage
import swingtrade
from watchlist import parse_ticker_text, read_ticker_sectors, read_tickers

# ----------------------------- Configuration -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"

DEFAULT_POSITION_BUDGET = 250    # default $ sidebar value for position sizing
DEFAULT_TOTAL_CASH = 5_000       # default $ sidebar value for total available cash

SCAN_CACHE_TTL_SEC = 900         # how long a scan result stays cached (15 min)

SIGNAL_COLORS = {
    "Strong Buy": "background-color: #1b7a3d; color: #ffffff;",
    "Buy": "background-color: #8bc34a; color: #1a1a1a;",
    "Watch": "background-color: #f6c945; color: #1a1a1a;",
    "Ignore": "background-color: #e57373; color: #1a1a1a;",
    "Insufficient Funds": "background-color: #78909c; color: #ffffff;",
    "Sector Limit Reached": "background-color: #5e35b1; color: #ffffff;",
}
CATALYST_WARNING_STYLE = "background-color: #c62828; color: #ffffff; font-weight: 600;"

DISPLAY_COLUMNS = [
    "Ticker", "Signal", "Trade_Score", "Last_Close", "Buy_Price", "Stop_Loss",
    "Sell_Price", "RRR", "RSI", "ATR", "Distance_to_Buy_Pct", "Shares_To_Buy",
    "Est_Cost", "Next_Earnings_Date", "Catalyst_Warning", "Top_Headline", "As_Of",
]

# ---------------------------------------------------------------------------


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Checking broad-market macro trend...")
def cached_market_uptrend(config: swingtrade.TradingConfig) -> tuple[bool, float, float]:
    return market_data.check_market_uptrend(config)


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Scanning watchlist...")
def scan_watchlist(
    tickers: tuple[str, ...], config: swingtrade.TradingConfig
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Fetch data and compute levels for every ticker, via the shared
    market_data module (also used standalone by ingest.py -- see
    ARCHITECTURE_PLAN.md Phase 7). Cached so sidebar tweaks (budget, sorting)
    don't re-trigger a full re-scan; `config` is part of the cache key, so a
    newly-promoted System_Config correctly triggers a fresh scan instead of
    serving results computed under the old parameters."""
    results, skipped = market_data.scan_tickers(tickers, config)
    return pd.DataFrame(results), skipped


def parse_holdings_text(raw: str) -> dict[str, float]:
    """Parse 'TICKER,AMOUNT' (or 'TICKER AMOUNT') lines into {ticker: dollars
    currently committed}. Blank/malformed lines and non-positive amounts are
    skipped rather than erroring -- this is a manually-typed box, not a
    validated form."""
    holdings = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) != 2:
            continue
        ticker, amount_str = parts
        try:
            amount = float(amount_str)
        except ValueError:
            continue
        if amount > 0:
            holdings[ticker.upper()] = amount
    return holdings


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


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def load_active_config() -> tuple[swingtrade.TradingConfig, str]:
    """Load the active TradingConfig from MongoDB's System_Config (re-checked
    on the same TTL cadence as the rest of the scan cache, so a newly
    promoted config takes effect without restarting the app). Logic lives in
    config_loader (shared with ingest.py, unwrapped there) so the dashboard
    and the standalone scan can never disagree on what "active" means."""
    return config_loader.load_active_config()


def main():
    st.set_page_config(page_title="Swing-Trading Dashboard", layout="wide")
    st.title("Swing-Trading Dashboard")
    st.caption(
        "Mechanical, rule-based output from historical data -- not a forecast or "
        "recommendation. Verify live price/liquidity before placing orders."
    )

    config, config_source = load_active_config()

    with st.sidebar:
        st.header("Configuration")
        st.caption(config_source)
        with st.expander("Active trading parameters"):
            st.json({
                "rsi_oversold_threshold": config.rsi_oversold_threshold,
                "atr_take_profit_multiplier": config.atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": config.stop_loss_atr_multiplier,
                "max_holding_days": config.max_holding_days,
                "max_sector_allocation_pct": config.max_sector_allocation_pct,
            })
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
                 "trades that no longer fit are marked Insufficient Funds, and trades that would "
                 "over-concentrate one sector are marked Sector Limit Reached (see "
                 "max_sector_allocation_pct in the active config above).",
        )

        st.subheader("Current Holdings")
        try:
            saved_holdings = storage.get_holdings()
        except Exception:
            saved_holdings = {}
        default_holdings_text = "\n".join(f"{t},{a:g}" for t, a in saved_holdings.items())
        holdings_text = st.text_area(
            "What you're actually holding right now (TICKER,AMOUNT $ per line)",
            value=default_holdings_text,
            height=100,
            help="Counted toward the sector cap below as already-deployed capital (your whole "
                 "portfolio, not just today's cash pool) -- never subtracted from Total "
                 "Available Cash itself. Used live from this box every run; click Save to "
                 "persist it as the default for next time.",
        )
        existing_holdings = parse_holdings_text(holdings_text)
        if st.button("Save holdings"):
            try:
                storage.set_holdings(existing_holdings)
                st.success(f"Saved {len(existing_holdings)} holding(s).")
            except Exception as exc:
                st.warning(f"Could not save holdings: {exc}")

        default_ticker_text = "\n".join(read_tickers(WATCHLIST_FILE)) if WATCHLIST_FILE.exists() else ""
        ticker_text = st.text_area(
            "Watchlist (one ticker per line, or comma-separated)",
            value=default_ticker_text,
            height=280,
        )
        tickers = tuple(parse_ticker_text(ticker_text))
        st.caption(f"{len(tickers)} ticker(s) loaded.")

    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    if not tickers:
        st.warning("No tickers to scan. Paste some tickers in the sidebar watchlist box.")
        st.stop()

    try:
        market_uptrend, market_close, market_sma200 = cached_market_uptrend(config)
    except Exception as exc:
        st.error(f"Could not evaluate {market_data.MARKET_INDEX_TICKER} macro trend: {exc}")
        st.stop()

    if not market_uptrend:
        st.error(
            f"**{market_data.MARKET_INDEX_TICKER} is in a macro downtrend** "
            f"(Last_Close {market_close:.2f} < SMA200 {market_sma200:.2f}). "
            "Individual structural-support levels are unreliable when the broad market "
            "itself is breaking down -- watchlist analysis has been skipped."
        )
        st.stop()
    st.success(f"{market_data.MARKET_INDEX_TICKER} is above its 200-day SMA ({market_close:.2f} >= {market_sma200:.2f}).")

    results_df, skipped = scan_watchlist(tickers, config)

    if results_df.empty:
        st.error("No tickers were successfully analyzed. See skipped tickers below.")
        if skipped:
            with st.expander(f"Skipped tickers ({len(skipped)})"):
                st.dataframe(pd.DataFrame(skipped, columns=["Ticker", "Reason"]), hide_index=True)
        st.stop()

    results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(config.fractional_share_decimals)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
    results_df = swingtrade.add_trade_score(results_df, config)
    results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)

    # Log signals BEFORE the capital-allocation overlay: Trade_Signals should
    # reflect the underlying technical signal, not whether cash happened to
    # be available today.
    storage_ok, storage_message = init_storage()
    if storage_ok:
        try:
            logged_count = storage.log_trade_signals(results_df, config.to_dict())
            st.sidebar.caption(f"Logged {logged_count} signal(s) to MongoDB.")
        except Exception as exc:
            st.sidebar.warning(f"Signal logging failed: {exc}")
    else:
        st.sidebar.caption(f"MongoDB not connected ({storage_message}) -- signals aren't being logged.")

    results_df, capital_allocated = swingtrade.allocate_capital(
        results_df, total_cash,
        sector_lookup=sector_lookup, max_sector_allocation_pct=config.max_sector_allocation_pct,
        existing_holdings=existing_holdings,
    )
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

    if sector_lookup:
        funded = results_df[results_df["Signal"].isin(["Strong Buy", "Buy"])].copy()
        funded["Sector"] = funded["Ticker"].map(sector_lookup).fillna("Unknown")
        new_by_sector = funded.groupby("Sector")["Est_Cost"].sum()

        holdings_by_sector: dict[str, float] = {}
        for ticker, amount in existing_holdings.items():
            sector = sector_lookup.get(ticker, "Unknown")
            holdings_by_sector[sector] = holdings_by_sector.get(sector, 0.0) + amount
        holdings_by_sector = pd.Series(holdings_by_sector, dtype=float)

        breakdown = pd.DataFrame({
            "Existing Holdings ($)": holdings_by_sector,
            "New Today ($)": new_by_sector,
        }).fillna(0.0)
        breakdown["Total ($)"] = breakdown["Existing Holdings ($)"] + breakdown["New Today ($)"]
        breakdown = breakdown.sort_values("Total ($)", ascending=False)

        portfolio_value = total_cash + sum(existing_holdings.values())
        sector_cap_dollars = (
            config.max_sector_allocation_pct * portfolio_value if config.max_sector_allocation_pct > 0 else None
        )
        with st.expander("Capital allocated by sector"):
            if sector_cap_dollars is not None:
                st.caption(
                    f"Cap: {config.max_sector_allocation_pct * 100:.0f}% of total portfolio value "
                    f"(${portfolio_value:,.2f} = today's cash + existing holdings) per sector "
                    f"= ${sector_cap_dollars:,.2f}."
                )
            st.dataframe(breakdown.reset_index(names="Sector"), width="stretch", hide_index=True)

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
