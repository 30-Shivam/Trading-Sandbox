"""Market data fetching (yfinance): OHLCV, earnings dates, news headlines,
and the broad-market macro-trend gate. Zero `streamlit` dependency so it can
be shared identically by the interactive dashboard (dip_buy_analyzer.py) and
the standalone scheduled scan (ingest.py) -- see ARCHITECTURE_PLAN.md Phase 7.
Reimplementing this in Go was explicitly rejected (amendment #1): yfinance's
reliability depends on replicating Yahoo's cookie/crumb auth handshake, which
changes without notice, and that's ongoing maintenance debt best absorbed by
an actively-maintained Python library.
"""

import time

import pandas as pd
import yfinance as yf

import swingtrade

LOOKBACK_PERIOD = "1y"           # data window to fetch (needs 200d+ for SMA200)
MARKET_INDEX_TICKER = "SPY"      # broad-market proxy for the macro gate
NEWS_HEADLINE_COUNT = 3          # recent news articles to fetch per ticker
REQUEST_DELAY_SEC = 0.5          # pause between API calls to avoid rate-limiting


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


def check_market_uptrend(
    config: swingtrade.TradingConfig, index_ticker: str = MARKET_INDEX_TICKER
) -> tuple[bool, float, float]:
    """Fetch the broad-market index and evaluate it via swingtrade.is_market_uptrend."""
    df = fetch_data(index_ticker)
    return swingtrade.is_market_uptrend(df, config)


def scan_tickers(
    tickers: tuple[str, ...], config: swingtrade.TradingConfig
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch data and compute levels for every ticker. The only network-heavy
    step; callers decide whether/how to cache it (Streamlit wraps this in
    st.cache_data, ingest.py calls it once per process). Dispatches on
    `config.strategy` -- "rsi" (default, compute_levels) or "breakout"
    (compute_breakout_levels) -- so callers don't need to know which
    strategy produced the active config; the returned dicts are
    schema-compatible either way (see compute_breakout_levels' docstring).

    For "breakout", also fetches the market index ONCE (not per ticker) to
    enable Relative_Strength -- see compute_relative_strength(). Failure to
    fetch it degrades gracefully (Relative_Strength stays None, informational
    only unless config.breakout_relative_strength_min is enabled) rather than
    failing the whole scan over a single extra data point."""
    market_df = None
    if config.strategy == "breakout":
        try:
            market_df = fetch_data(MARKET_INDEX_TICKER)
        except Exception:
            market_df = None

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
            if config.strategy == "breakout":
                levels = swingtrade.compute_breakout_levels(
                    ticker, df, config, next_earnings_date=next_earnings,
                    top_headline=top_headline, market_df=market_df,
                )
            else:
                levels = swingtrade.compute_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            results.append(levels)
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    return results, skipped


def review_holdings(
    holdings: dict[str, float], config: swingtrade.TradingConfig
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch current data and evaluate each held ticker (ticker -> avg_cost)
    against the active config's stop/target rules, via
    swingtrade.review_holding. Same rate-limited fetch pattern as
    scan_tickers."""
    results = []
    skipped = []
    for i, (ticker, avg_cost) in enumerate(holdings.items()):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_data(ticker)
            results.append(swingtrade.review_holding(ticker, df, avg_cost, config))
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    return results, skipped
