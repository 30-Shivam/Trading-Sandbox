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
MACRO_HEADLINE_TICKER = "^GSPC"  # broad-index proxy for macro/market-wide headlines (LLM Agent tab)
MACRO_HEADLINE_COUNT = 5         # macro headlines to fetch per get_macro_snapshot() call
VIX_TICKER = "^VIX"              # CBOE volatility index -- free numeric market-fear proxy


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=LOOKBACK_PERIOD, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no data returned for ticker '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # yfinance/Yahoo occasionally publishes a trailing row for the most
    # recent session with a real Volume but a NaN Close (the day's data
    # isn't fully finalized yet at fetch time) -- a NaN anywhere in the
    # window NaNs out every rolling indicator computed through that row
    # (SMA/RSI/ATR), which otherwise surfaces as a confusing "insufficient
    # history" error despite having a full year of real data behind it.
    # Drop any such trailing rows so the newest bar used downstream is
    # always a genuinely complete one, not a same-day placeholder.
    while len(df) and pd.isna(df["Close"].iloc[-1]):
        df = df.iloc[:-1]
    if df.empty:
        raise RuntimeError(f"no usable (non-NaN Close) data returned for ticker '{ticker}'")
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


def get_multi_headlines(ticker: str, count: int = NEWS_HEADLINE_COUNT) -> list[str]:
    """Fetch up to `count` recent headlines for `ticker` on demand -- a thin
    wrapper around get_recent_headlines() for callers (e.g. ai_context.py
    via the dashboard) that want more than the single Top_Headline already
    carried on a scan_tickers() result, without re-fetching OHLCV or paying
    the cost of storing every ticker's full headline list during the main
    scan. Returns [] on any fetch failure -- informational only, never
    worth failing a scan over."""
    try:
        return get_recent_headlines(yf.Ticker(ticker), count=count)
    except Exception:
        return []


def get_macro_headlines(count: int = MACRO_HEADLINE_COUNT) -> list[str]:
    """Fetch up to `count` recent headlines for a broad market index
    (MACRO_HEADLINE_TICKER) rather than any single company -- market-wide
    news (Fed decisions, major economic/political events) tends to surface
    on a broad index's own news feed. Same mechanism as get_multi_headlines()
    (yfinance's Ticker.news), just pointed at an index instead of a ticker.
    Returns [] on any fetch failure -- informational only, never worth
    failing a scan over."""
    try:
        return get_recent_headlines(yf.Ticker(MACRO_HEADLINE_TICKER), count=count)
    except Exception:
        return []


def get_macro_snapshot() -> dict:
    """One-per-run market-wide backdrop for the LLM Agent tab (see
    llm_agent.py) -- VIX level/change plus broad-market headlines, fetched
    ONCE per dashboard page load and shared across every candidate ticker
    that run (not one fetch per ticker). Each field degrades independently
    to None/[] on its own fetch failure rather than the whole snapshot
    failing -- same resilience philosophy as every other fetcher here."""
    vix = None
    vix_change_pct = None
    try:
        vix_df = fetch_data(VIX_TICKER)
        vix = round(float(vix_df["Close"].iloc[-1]), 2)
        if len(vix_df) >= 2:
            prev_close = float(vix_df["Close"].iloc[-2])
            if prev_close:
                vix_change_pct = round((vix - prev_close) / prev_close * 100, 2)
    except Exception:
        pass

    return {"vix": vix, "vix_change_pct": vix_change_pct, "headlines": get_macro_headlines()}


def _get_analyst_data(ticker_obj: yf.Ticker) -> dict | None:
    """Analyst price targets + recommendation trend (last 4 months, oldest
    to newest -- shows DIRECTION, not just a snapshot) + the most recent 3
    upgrades/downgrades. Returns None (not raise) on total failure; a
    partial result (e.g. targets but no trend) is still returned rather
    than discarded, since each sub-field is independently useful."""
    try:
        targets = dict(ticker_obj.analyst_price_targets or {})
    except Exception:
        targets = {}

    trend = []
    try:
        recs = ticker_obj.recommendations
        if recs is not None and not recs.empty:
            for _, row in recs.iloc[::-1].iterrows():  # oldest -> newest
                trend.append(
                    f"{row.get('period')}: strongBuy={row.get('strongBuy')}, buy={row.get('buy')}, "
                    f"hold={row.get('hold')}, sell={row.get('sell')}, strongSell={row.get('strongSell')}"
                )
    except Exception:
        pass

    recent_actions = []
    try:
        ud = ticker_obj.upgrades_downgrades
        if ud is not None and not ud.empty:
            for _, row in ud.head(3).iterrows():
                recent_actions.append(
                    f"{row.get('Firm')}: {row.get('FromGrade')} -> {row.get('ToGrade')} ({row.get('Action')})"
                )
    except Exception:
        pass

    if not targets and not trend and not recent_actions:
        return None
    return {"targets": targets, "trend": trend, "recent_actions": recent_actions}


def _get_insider_data(ticker_obj: yf.Ticker) -> dict | None:
    """Net buy/sell $ direction from the 5 most recent insider transactions
    (classified by keyword in the Text field -- yfinance has no separate
    clean buy/sell flag), plus static institutional/insider ownership
    levels from Ticker.info. Non-market transactions (gifts, awards) are
    excluded from the net-direction calculation but don't block ownership
    levels from being returned."""
    net_direction = None
    try:
        txns = ticker_obj.insider_transactions
        if txns is not None and not txns.empty:
            buy_value, sell_value = 0.0, 0.0
            for _, row in txns.head(5).iterrows():
                text = str(row.get("Text") or "").lower()
                value = row.get("Value")
                if value is None or pd.isna(value):
                    continue
                if "sale" in text:
                    sell_value += float(value)
                elif "purchase" in text or "buy" in text:
                    buy_value += float(value)
            if buy_value or sell_value:
                if buy_value > sell_value * 1.5:
                    net_direction = "Buying"
                elif sell_value > buy_value * 1.5:
                    net_direction = "Selling"
                else:
                    net_direction = "Mixed"
    except Exception:
        pass

    pct_institutions, pct_insiders = None, None
    try:
        info = ticker_obj.info or {}
        pct_institutions = info.get("heldPercentInstitutions")
        pct_insiders = info.get("heldPercentInsiders")
    except Exception:
        pass

    if net_direction is None and pct_institutions is None and pct_insiders is None:
        return None
    return {"net_direction": net_direction, "pct_institutions": pct_institutions, "pct_insiders": pct_insiders}


def _get_short_interest(ticker_obj: yf.Ticker) -> dict | None:
    """Short interest as a % of float, plus its direction vs. the prior
    month -- a positioning signal distinct in character from news/analyst
    text. >5% relative change is treated as a real move; smaller than that
    is "Flat" rather than noise being reported as a trend."""
    try:
        info = ticker_obj.info or {}
        pct_of_float = info.get("shortPercentOfFloat")
        shares_short = info.get("sharesShort")
        shares_short_prior = info.get("sharesShortPriorMonth")
    except Exception:
        return None

    trend = None
    if shares_short is not None and shares_short_prior:
        change_pct = (shares_short - shares_short_prior) / shares_short_prior
        if change_pct > 0.05:
            trend = "Increasing"
        elif change_pct < -0.05:
            trend = "Decreasing"
        else:
            trend = "Flat"

    if pct_of_float is None and trend is None:
        return None
    return {"pct_of_float": round(pct_of_float * 100, 2) if pct_of_float is not None else None, "trend": trend}


def _get_recent_filings(ticker_obj: yf.Ticker, count: int = 3) -> list[dict]:
    """Most recent `count` SEC filings (type/date/title) -- a primary-source
    signal distinct from aggregated news headlines. Returns [] on any
    failure or if no filings are available."""
    try:
        filings = ticker_obj.sec_filings or []
    except Exception:
        return []
    out = []
    for f in filings[:count]:
        if not isinstance(f, dict):
            continue
        out.append({"type": f.get("type"), "date": str(f.get("date")), "title": f.get("title")})
    return out


def _get_options_sentiment(ticker_obj: yf.Ticker) -> dict | None:
    """Nearest-expiry put/call VOLUME ratio -- a "smart money" positioning
    proxy unrelated to news/analyst/insider signal. Returns None if no
    options chain is available (illiquid/no listed options) or on any
    fetch failure -- not every ticker has one, and that's not an error."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None
        chain = ticker_obj.option_chain(expirations[0])
        call_volume = float(chain.calls["volume"].fillna(0).sum())
        put_volume = float(chain.puts["volume"].fillna(0).sum())
    except Exception:
        return None
    if call_volume <= 0:
        return None
    return {"put_call_volume_ratio": round(put_volume / call_volume, 3)}


def get_qualitative_snapshot(ticker: str) -> dict:
    """Richer, qualitative-beyond-numbers context for the LLM Agent tab and
    Position Review overlay (see llm_agent.py) -- analyst consensus/trend,
    insider activity, short interest, recent SEC filings, and options
    positioning. Each of the five sub-sections degrades INDEPENDENTLY to
    None/[] on its own fetch failure (never raises, and one sub-fetcher
    failing never blocks the other four) -- same resilience philosophy as
    get_macro_snapshot(). One `yf.Ticker` object is built once and reused
    across all five sub-fetchers, same convention get_next_earnings_date()
    already established (ticker_obj passed in, not a raw ticker string).

    Returns {"analyst": dict|None, "insider": dict|None,
    "short_interest": dict|None, "filings": list[dict], "options": dict|None}
    -- see llm_agent._build_qualitative_block() for how each key renders
    into the LLM prompt."""
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        return {"analyst": None, "insider": None, "short_interest": None, "filings": [], "options": None}

    return {
        "analyst": _get_analyst_data(ticker_obj),
        "insider": _get_insider_data(ticker_obj),
        "short_interest": _get_short_interest(ticker_obj),
        "filings": _get_recent_filings(ticker_obj),
        "options": _get_options_sentiment(ticker_obj),
    }


def check_market_uptrend(
    config: swingtrade.TradingConfig, index_ticker: str = MARKET_INDEX_TICKER
) -> tuple[bool, float, float]:
    """Fetch the broad-market index and evaluate it via swingtrade.is_market_uptrend."""
    df = fetch_data(index_ticker)
    return swingtrade.is_market_uptrend(df, config)


CANADIAN_EXCHANGE_SUFFIXES = (".TO", ".V")  # Toronto Stock Exchange / TSX Venture -- yfinance's
                                             # own suffix convention for Canadian listings


def get_ticker_currency(ticker: str) -> str:
    """The currency a ticker trades in -- "CAD" for a Toronto Stock
    Exchange/TSX Venture listing (see CANADIAN_EXCHANGE_SUFFIXES), "USD"
    otherwise. Deliberately a pure, zero-cost suffix check rather than a
    yfinance Ticker.info call: this project's watchlist only ever mixes
    these two currencies (via the .TO suffix convention it already uses
    for CSU.TO and friends), so the ticker symbol itself already carries
    this information -- no need to pay a network round-trip per ticker,
    266 times a day, for something the symbol already tells you.

    This system does NOT do FX conversion -- Currency is purely an
    informational/display field (see the 'Currency' column added to
    every scan result, and the 'currency' field persisted alongside
    every logged Trade_Signals document) so a CAD-denominated position's
    dollar figures are never silently mistaken for USD. A flat-dollar
    position budget still spends that many units of whatever currency
    the ticker trades in."""
    return "CAD" if ticker.endswith(CANADIAN_EXCHANGE_SUFFIXES) else "USD"


def fetch_ticker_bundle(
    tickers: tuple[str, ...],
) -> tuple[dict[str, dict], pd.DataFrame | None, list[tuple[str, str]]]:
    """Fetch OHLCV + earnings + headlines for every ticker ONCE, independent
    of any strategy config -- the pure-fetch half of what scan_tickers()
    used to do in one fetch-then-compute-then-discard loop. Also fetches the
    market index (SPY) exactly once here, unconditionally -- previously
    only fetched for "breakout" configs (and, for breakout specifically, a
    SECOND time inside scan_tickers' old per-ticker loop, an existing minor
    inefficiency this also fixes). Fetching it unconditionally now is a
    small, one-time cost that's basically free relative to the per-ticker
    loop, and lets every strategy share the same bundle without needing to
    know ahead of time which of them will actually use it.

    Returns `(bundle, market_df, skipped)`: `bundle` maps ticker ->
    `{"df": OHLCV DataFrame, "next_earnings": Timestamp | None,
    "top_headline": str, "currency": "USD" | "CAD"}` (see
    get_ticker_currency() -- a pure suffix check, not a fetch, so it
    never contributes to `skipped`); `market_df` is SPY's OHLCV (`None` if it failed
    to fetch -- degrades gracefully, same as before: Relative_Strength
    stays `None` rather than failing the whole scan over one extra data
    point); `skipped` is `(ticker, reason)` pairs for tickers whose OWN
    data failed to fetch.

    Callers running more than one strategy against the same tickers (see
    dip_buy_analyzer.py's multi-strategy dashboard sections) should call
    this ONCE and then `score_bundle_for_strategy()` once per strategy
    against the same bundle, instead of paying the fetch cost per strategy."""
    try:
        market_df = fetch_data(MARKET_INDEX_TICKER)
    except Exception:
        market_df = None

    bundle: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    now_utc = pd.Timestamp.now(tz="UTC")
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_data(ticker)
            ticker_obj = yf.Ticker(ticker)
            next_earnings = get_next_earnings_date(ticker_obj, now_utc)
            headlines = get_recent_headlines(ticker_obj)
            top_headline = headlines[0] if headlines else ""
            bundle[ticker] = {
                "df": df, "next_earnings": next_earnings, "top_headline": top_headline,
                "currency": get_ticker_currency(ticker),
            }
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    return bundle, market_df, skipped


def score_bundle_for_strategy(
    bundle: dict[str, dict], market_df: pd.DataFrame | None, config: swingtrade.TradingConfig
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Compute levels for every ticker in an already-fetched bundle (see
    fetch_ticker_bundle()), dispatching on `config.strategy`: "rsi"
    (default, compute_levels), "breakout" (compute_breakout_levels, uses
    `market_df` for Relative_Strength), "pullback" (compute_pullback_levels),
    "breakout_retest" (compute_breakout_retest_levels), "week52_high"
    (compute_week52_levels), "momentum_burst" (compute_momentum_burst_levels),
    "squeeze_breakout" (compute_squeeze_breakout_levels), or
    "adx_trend_entry" (compute_adx_trend_entry_levels) -- the returned
    dicts are schema-compatible across all eight (see
    compute_breakout_levels' docstring), each with an added "Currency"
    key ("USD"/"CAD", from fetch_ticker_bundle()'s own get_ticker_currency()
    tag -- informational only, this system does no FX conversion). Pure
    computation, no network calls -- safe and cheap to call once per
    strategy against the SAME bundle."""
    results = []
    skipped = []
    for ticker, entry in bundle.items():
        df, next_earnings, top_headline = entry["df"], entry["next_earnings"], entry["top_headline"]
        try:
            if config.strategy == "breakout":
                levels = swingtrade.compute_breakout_levels(
                    ticker, df, config, next_earnings_date=next_earnings,
                    top_headline=top_headline, market_df=market_df,
                )
            elif config.strategy == "pullback":
                levels = swingtrade.compute_pullback_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            elif config.strategy == "breakout_retest":
                levels = swingtrade.compute_breakout_retest_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            elif config.strategy == "week52_high":
                levels = swingtrade.compute_week52_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            elif config.strategy == "momentum_burst":
                levels = swingtrade.compute_momentum_burst_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            elif config.strategy == "squeeze_breakout":
                levels = swingtrade.compute_squeeze_breakout_levels(
                    ticker, df, config, next_earnings_date=next_earnings,
                    top_headline=top_headline, market_df=market_df,
                )
            elif config.strategy == "adx_trend_entry":
                levels = swingtrade.compute_adx_trend_entry_levels(
                    ticker, df, config, next_earnings_date=next_earnings,
                    top_headline=top_headline, market_df=market_df,
                )
            elif config.strategy == "ma_crossover":
                levels = swingtrade.compute_ma_crossover_levels(
                    ticker, df, config, next_earnings_date=next_earnings,
                    top_headline=top_headline, market_df=market_df,
                )
            else:
                levels = swingtrade.compute_levels(
                    ticker, df, config, next_earnings_date=next_earnings, top_headline=top_headline
                )
            # Purely informational -- see get_ticker_currency()'s own docstring for why this
            # system deliberately does NOT do FX conversion, just honest labeling.
            levels["Currency"] = entry.get("currency", "USD")
            results.append(levels)
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    return results, skipped


def scan_tickers(
    tickers: tuple[str, ...], config: swingtrade.TradingConfig
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch data and compute levels for every ticker. The only network-heavy
    step; callers decide whether/how to cache it (Streamlit wraps this in
    st.cache_data, ingest.py calls it once per process). Signature and
    behavior are unchanged from before the fetch/compute split below --
    existing single-strategy callers (ingest.py, the dashboard's primary
    scan) need no changes. Thin wrapper over fetch_ticker_bundle() +
    score_bundle_for_strategy() -- callers running MULTIPLE strategies
    against the same tickers should call those two directly instead (see
    fetch_ticker_bundle()'s docstring), to fetch once and score many times
    rather than paying the fetch cost per strategy."""
    bundle, market_df, fetch_skipped = fetch_ticker_bundle(tickers)
    results, score_skipped = score_bundle_for_strategy(bundle, market_df, config)
    return results, fetch_skipped + score_skipped


def review_holdings(
    holdings: dict[str, float], config: swingtrade.TradingConfig
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch current data and evaluate each held ticker (ticker -> avg_cost)
    against the active config's stop/target rules, via
    swingtrade.review_holding. Same rate-limited fetch pattern as
    scan_tickers. Each result dict gains a "Currency" key (see
    get_ticker_currency()) -- Avg_Cost/Last_Close/Stop_Loss/Sell_Price are
    always in THIS currency; this system does no FX conversion, so a
    manually-entered Avg_Cost for a CAD ticker needs to have been entered
    in CAD to compare correctly against the ticker's own CAD price data."""
    results = []
    skipped = []
    for i, (ticker, avg_cost) in enumerate(holdings.items()):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_data(ticker)
            review = swingtrade.review_holding(ticker, df, avg_cost, config)
            review["Currency"] = get_ticker_currency(ticker)
            results.append(review)
        except Exception as exc:
            skipped.append((ticker, str(exc)))
    return results, skipped
