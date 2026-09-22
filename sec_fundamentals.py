"""Free, official, point-in-time fundamentals via the SEC's own XBRL
company-facts API -- built 2026-09-16 as the fix for a real, confirmed gap:
yfinance's quarterly_financials/financials only return 5-7 quarters / 4-5
annual periods per ticker (verified directly against AAPL/MSFT/JPM/XOM),
far too thin for this project's own multi-year validation standard (see
improvements.txt / the algorithm-derivation-loop's fundamentals-data-blocker
finding). SEC EDGAR instead gives real, filed-date-tagged history spanning
well over a decade for most tickers (confirmed: AAPL back to 2009, HUBB -- a
mid-cap -- back to 2008), each fact carrying its own real `filed` date
separate from the `end` (period-covered) date, which is exactly the
point-in-time correctness a backtest needs and yfinance's live-only
snapshot fields cannot provide.

REAL LIMITATION, not a silent gap: only covers SEC-registered (effectively
US-listed) filers -- Canadian cross-listed tickers in this project's own
watchlist.txt (ATD.TO, TD.TO, CNQ.TO, BCE.TO, SU.TO, CNR.TO, NTR.TO, ABX.TO)
are NOT in SEC's ticker-to-CIK map and will silently return no data (same
"missing optional data never fabricates a signal" convention every other
strategy in this codebase uses, not a crash).

No API key required -- SEC's own fair-use policy just asks for a
descriptive User-Agent identifying the requester and reasonable request
pacing (a courtesy, not an enforced rate limit the way most vendor APIs
are), hence REQUEST_DELAY_SEC below.

REAL BUG FOUND+FIXED 2026-09-21, while validating the "value_rank" strategy
built on top of fetch_point_in_time_book_value_per_share() below: a stock
split creates a systematic Price-to-Book mismatch if left uncorrected.
yfinance's historical Close (see run_backtest.fetch_history(), even with
auto_adjust=False) is always expressed on TODAY's post-split share-count
basis -- confirmed directly (NVDA's real 2023-01-03 close was ~$143;
yfinance reports ~$14.3, already divided by its June-2024 10:1 split
ratio). SEC's own filed `dei:EntityCommonStockSharesOutstanding`, by
contrast, is whatever the company's cover page said AT THAT TIME -- the
PRE-split count for any filing before a later split. Dividing a split-
ADJUSTED price by a book-value-per-share built from PRE-split shares
deflates the ratio by roughly the split factor for every date before the
split, making a real, expensive growth stock (NVDA's real historical P/B
was ~16x) look falsely CHEAP (computed as ~1.6x) throughout its entire
pre-split history -- exactly backwards. Fixed via `_cumulative_split_factor()`
below, which multiplies each filing's raw shares-outstanding by the
product of every real stock split (from yfinance's own `Ticker.splits`)
that happened AFTER that filing, so the denominator lands on the SAME
modern-share-count basis the price series already uses.
"""
import time

import pandas as pd
import requests
import yfinance as yf

USER_AGENT = "Trading-Sandbox research contact: ks3032004@gmail.com"
REQUEST_DELAY_SEC = 0.15  # SEC's own fair-use guidance is ~10 req/sec max; well under that
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_cik_map_cache: dict[str, str] | None = None


def fetch_cik_map() -> dict[str, str]:
    """Ticker -> 10-digit zero-padded CIK, fetched once and cached for the
    life of the process (this file is small -- ~10K tickers -- and doesn't
    change intra-session). Returns {} on any fetch failure rather than
    raising, same "missing optional data degrades gracefully" convention
    every other fetch function in this codebase follows."""
    global _cik_map_cache
    if _cik_map_cache is not None:
        return _cik_map_cache
    try:
        resp = requests.get(TICKER_MAP_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        _cik_map_cache = {}
        return _cik_map_cache
    _cik_map_cache = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}
    return _cik_map_cache


def fetch_point_in_time_roe(ticker: str) -> pd.DataFrame:
    """Real, point-in-time-correct annual Return-on-Equity history for one
    ticker, built from SEC EDGAR's own filed 10-Ks -- the fundamentals
    counterpart to run_backtest.fetch_earnings_surprises()/
    fetch_insider_purchases().

    ROE = NetIncomeLoss / StockholdersEquity, matched by FISCAL YEAR (same
    `end` date on both concepts) within the SAME 10-K filing (same `accn`
    accession number, so a restated/amended figure never gets silently
    paired with the wrong period's originally-filed counterpart). Uses only
    `form == "10-K"` entries -- deliberately ANNUAL, not a trailing-twelve-
    month quarterly reconstruction, for v1: simpler, less bug-prone, and
    academically standard for a quality/profitability factor (unlike
    momentum, low-turnover fundamentals factors are conventionally rebuilt
    once a year, not continuously).

    Returns a DataFrame with columns ["filed_date", "roe"], one row per
    fiscal year, sorted by filed_date -- `filed_date` is deliberately the
    REAL date the public first saw this figure (when the 10-K was actually
    filed), NOT the fiscal year-end date it covers (which can lag by
    60-90+ days) -- using `end` instead of `filed` anywhere downstream
    would silently introduce real look-ahead bias, exactly the class of
    bug this project's own audit_no_lookahead() discipline exists to catch.
    Degrades to an empty DataFrame (ticker not in SEC's map, no CIK match,
    fetch failure, or no matched NetIncomeLoss/StockholdersEquity pairs)
    rather than crashing, same convention as every other optional-data
    fetch function here."""
    columns = ["filed_date", "roe"]
    cik_map = fetch_cik_map()
    cik = cik_map.get(ticker)
    if cik is None:
        return pd.DataFrame(columns=columns)

    try:
        resp = requests.get(
            COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        if resp.status_code != 200:
            return pd.DataFrame(columns=columns)
        facts = resp.json().get("facts", {}).get("us-gaap", {})
    except Exception:
        return pd.DataFrame(columns=columns)

    net_income = facts.get("NetIncomeLoss", {}).get("units", {}).get("USD", [])
    equity = facts.get("StockholdersEquity", {}).get("units", {}).get("USD", [])
    if not net_income or not equity:
        return pd.DataFrame(columns=columns)

    ni_10k = {u["accn"]: u for u in net_income if u.get("form") == "10-K" and u.get("end")}
    eq_10k = {u["accn"]: u for u in equity if u.get("form") == "10-K" and u.get("end")}

    rows = []
    for accn, ni in ni_10k.items():
        eq = eq_10k.get(accn)
        if eq is None or eq.get("end") != ni.get("end"):
            continue
        equity_val = eq.get("val")
        if not equity_val or equity_val < 0:  # zero or negative equity -- ROE undefined/meaningless, skip rather than divide
                                            # (negative equity with negative net income would otherwise
                                            # divide out to a POSITIVE ratio, hiding financial distress
                                            # instead of flagging it -- a well-known ROE pitfall, caught by
                                            # this module's own test before it ever reached real data)
            continue
        rows.append({"filed_date": ni["filed"], "roe": ni["val"] / equity_val, "fiscal_year_end": ni["end"]})

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows).sort_values("filed_date")
    # A ticker can have more than one 10-K/A (amendment) for the same fiscal
    # year -- keep only the LAST (most recent) filed value per fiscal year
    # end, so a later restatement correctly supersedes the original rather
    # than both appearing as separate point-in-time observations.
    out = out.drop_duplicates(subset="fiscal_year_end", keep="last")
    out["filed_date"] = pd.to_datetime(out["filed_date"])
    return out[columns].reset_index(drop=True)


def _cumulative_split_factor(splits: pd.Series, filed_date) -> float:
    """Product of every real stock-split ratio in `splits` (yfinance's own
    Ticker.splits, a date-indexed Series of ratios like 10.0 for a 10:1
    split) that occurred AFTER `filed_date` -- see this module's own
    top-of-file bug writeup (2026-09-21) for why this matters. Returns 1.0
    (no adjustment) when there are no later splits or `splits` is empty,
    same "missing optional data never fabricates a signal" convention as
    every other optional lookup here."""
    if splits is None or splits.empty:
        return 1.0
    filed_ts = pd.Timestamp(filed_date)
    if filed_ts.tzinfo is None and splits.index.tz is not None:
        filed_ts = filed_ts.tz_localize(splits.index.tz)
    elif filed_ts.tzinfo is not None and splits.index.tz is None:
        filed_ts = filed_ts.tz_localize(None)
    later_splits = splits[splits.index > filed_ts]
    if later_splits.empty:
        return 1.0
    return float(later_splits.prod())


def fetch_point_in_time_book_value_per_share(ticker: str) -> pd.DataFrame:
    """Real, point-in-time-correct annual Book Value Per Share history for
    one ticker -- the VALUE-factor counterpart to fetch_point_in_time_roe()
    (a QUALITY factor). BVPS = StockholdersEquity / shares outstanding,
    matched by FISCAL YEAR (same `end` date) within the SAME 10-K filing
    (same `accn` accession number), exact same discipline as
    fetch_point_in_time_roe() -- see that function's docstring for the
    filed-vs-end date reasoning, which applies identically here.

    Shares outstanding comes from `dei:EntityCommonStockSharesOutstanding`
    (a cover-page fact, not `us-gaap`) -- a required disclosure on every
    10-K's cover page for essentially all SEC filers, unlike some us-gaap
    concepts that vary by company's own chart-of-accounts choices, making
    it a reliably populated fact to build a universe-wide panel from.

    Returns a DataFrame with columns ["filed_date", "book_value_per_share"],
    one row per fiscal year, sorted by filed_date. Degrades to an empty
    DataFrame (ticker not in SEC's map, no CIK match, fetch failure, or no
    matched StockholdersEquity/shares-outstanding pairs) rather than
    crashing, same convention as fetch_point_in_time_roe().

    Each row's raw (as-filed) share count is scaled by
    _cumulative_split_factor() to land on the SAME modern post-split share
    basis run_backtest.fetch_history()'s Close prices already use -- without
    this, any ticker that split during (or after) the backtest window would
    show a falsely deflated Price-to-Book for its entire pre-split history
    (real bug found+fixed 2026-09-21, see this module's own top-of-file
    writeup)."""
    columns = ["filed_date", "book_value_per_share"]
    cik_map = fetch_cik_map()
    cik = cik_map.get(ticker)
    if cik is None:
        return pd.DataFrame(columns=columns)

    try:
        resp = requests.get(
            COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        if resp.status_code != 200:
            return pd.DataFrame(columns=columns)
        payload = resp.json().get("facts", {})
    except Exception:
        return pd.DataFrame(columns=columns)

    equity = payload.get("us-gaap", {}).get("StockholdersEquity", {}).get("units", {}).get("USD", [])
    shares = (
        payload.get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    )
    if not equity or not shares:
        return pd.DataFrame(columns=columns)

    eq_10k = {u["accn"]: u for u in equity if u.get("form") == "10-K" and u.get("end")}
    # dei:EntityCommonStockSharesOutstanding is a COVER-PAGE fact -- its own
    # "end" date is the cover page's as-of date (close to, but not always
    # identical to, the fiscal year end), so match by accession number only
    # (same filing), not by exact end-date equality like ROE's NetIncomeLoss
    # match does -- accession number alone is sufficient since both facts
    # come from the SAME 10-K.
    sh_10k = {u["accn"]: u for u in shares if u.get("form") == "10-K"}

    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        splits = pd.Series(dtype=float)

    rows = []
    for accn, eq in eq_10k.items():
        sh = sh_10k.get(accn)
        if sh is None:
            continue
        shares_val = sh.get("val")
        equity_val = eq.get("val")
        if not shares_val or shares_val <= 0 or equity_val is None:
            continue
        raw_bvps = equity_val / shares_val
        adjusted_bvps = raw_bvps / _cumulative_split_factor(splits, eq["filed"])
        rows.append({"filed_date": eq["filed"], "book_value_per_share": adjusted_bvps, "fiscal_year_end": eq.get("end")})

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows).sort_values("filed_date")
    out = out.drop_duplicates(subset="fiscal_year_end", keep="last")
    out["filed_date"] = pd.to_datetime(out["filed_date"])
    return out[columns].reset_index(drop=True)


def build_book_value_per_share_panel(tickers: list[str], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the wide (dates x tickers) point-in-time Book Value Per Share
    panel the "value_rank" strategy needs -- exact structural mirror of
    build_roe_panel(), same forward-fill-never-backfill discipline."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        bvps_history = fetch_point_in_time_book_value_per_share(ticker)
        if bvps_history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            bvps_history["book_value_per_share"].values, index=pd.DatetimeIndex(bvps_history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


BVPS_CACHE_COLLECTION = "Fundamentals_Cache_BVPS"
BVPS_CACHE_MAX_AGE_DAYS = 7  # book value per share only changes once per
                                           # fiscal year per ticker (a new
                                           # 10-K) -- re-fetching SEC's full
                                           # company-facts payload (2
                                           # network calls per ticker as of
                                           # the 2026-09-21 split-adjustment
                                           # fix) on every single daily
                                           # automation run is unnecessary
                                           # network cost and unnecessary
                                           # load on SEC's own servers. A
                                           # week is a deliberately
                                           # conservative refresh cadence,
                                           # not tuned to filing frequency.


def fetch_point_in_time_book_value_per_share_cached(
    ticker: str, max_age_days: int = BVPS_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached wrapper around fetch_point_in_time_book_value_per_share() --
    see BVPS_CACHE_MAX_AGE_DAYS above for why caching matters here. Checks
    MongoDB's `Fundamentals_Cache_BVPS` collection first (one document per
    ticker, `{"_id": ticker, "rows": [...], "fetched_at": <UTC datetime>}`);
    only re-fetches from SEC EDGAR/yfinance if the cached entry is missing
    or older than `max_age_days`. Deliberately a SEPARATE function from
    fetch_point_in_time_book_value_per_share() rather than a modification
    of it -- the backtest path (benchmark_random_entry.py) keeps calling
    the uncached original directly, unaffected by this at all, matching
    this module's own "small independent functions" convention. Safe to
    call from a GH Actions cron (a fresh container every run, no local
    disk persistence) since the cache lives in the already-persistent
    Mongo cluster this project already depends on, not local disk.
    Degrades to an uncached live fetch (never crashes) if Mongo is
    unavailable -- same "missing optional infrastructure never blocks a
    real signal" convention as every other storage-touching function in
    this codebase."""
    import storage  # local import: keeps this module's own test suite
                                           # (network-mocked, Mongo-free)
                                           # working unchanged for every
                                           # function that doesn't need
                                           # this cache, same "storage is
                                           # optional infrastructure"
                                           # posture as the rest of this
                                           # codebase
    from datetime import datetime, timezone

    columns = ["filed_date", "book_value_per_share"]
    db = None
    try:
        db = storage.get_db()
        cached = db[BVPS_CACHE_COLLECTION].find_one({"_id": ticker})
    except Exception:
        cached = None

    if cached is not None:
        fetched_at = cached.get("fetched_at")
        if fetched_at is not None:
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
            if age_days <= max_age_days:
                rows = cached.get("rows", [])
                if not rows:
                    return pd.DataFrame(columns=columns)
                out = pd.DataFrame(rows)
                out["filed_date"] = pd.to_datetime(out["filed_date"])
                return out[columns]

    fresh = fetch_point_in_time_book_value_per_share(ticker)
    if db is not None:
        try:
            db[BVPS_CACHE_COLLECTION].replace_one(
                {"_id": ticker},
                {
                    "_id": ticker,
                    "rows": fresh.assign(filed_date=fresh["filed_date"].astype(str)).to_dict("records"),
                    "fetched_at": datetime.now(timezone.utc),
                },
                upsert=True,
            )
        except Exception:
            pass  # cache write failure never blocks returning the real, freshly-fetched data
    return fresh


def build_book_value_per_share_panel_cached(
    tickers: list[str], date_index: pd.DatetimeIndex, max_age_days: int = BVPS_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached counterpart to build_book_value_per_share_panel() -- exact
    same forward-fill-never-backfill construction, but sourced from
    fetch_point_in_time_book_value_per_share_cached() instead of the
    uncached fetch. Intended for the LIVE daily automation path
    (ingest.py), not the backtest path -- benchmark_random_entry.py keeps
    calling build_book_value_per_share_panel() directly, unaffected."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        bvps_history = fetch_point_in_time_book_value_per_share_cached(ticker, max_age_days)
        if bvps_history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            bvps_history["book_value_per_share"].values, index=pd.DatetimeIndex(bvps_history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


def fetch_point_in_time_accruals(ticker: str) -> pd.DataFrame:
    """Real, point-in-time-correct annual ACCRUALS history for one ticker
    -- the "accruals_rank" strategy's data source, implementing Sloan
    (1996)'s classic accruals anomaly (one of the most robust, longest-
    replicated findings in the academic asset-pricing literature): firms
    whose earnings are mostly real cash flow (LOW accruals) outperform
    firms whose earnings are mostly accounting adjustments (HIGH
    accruals), because investors systematically overweight the less-
    persistent accrual component.

    Accruals = (NetIncomeLoss - NetCashProvidedByUsedInOperatingActivities)
    / Assets -- scaled by TOTAL ASSETS, deliberately NOT by market cap or
    shares outstanding, which structurally avoids the real stock-split
    mismatch class of bug fetch_point_in_time_book_value_per_share() above
    had to fix (see this module's own top-of-file writeup) -- this ratio
    never touches price or share count at all.

    NetIncomeLoss and NetCashProvidedByUsedInOperatingActivities are both
    FLOW concepts (matched by accession number AND `end` date, same
    discipline as fetch_point_in_time_roe()'s NetIncomeLoss/StockholdersEquity
    pairing). Assets is a POINT-IN-TIME concept (matched by accession
    number alone, same "cover-page fact from the same filing" convention
    fetch_point_in_time_book_value_per_share() uses for shares outstanding).
    Uses only `form == "10-K"` entries -- deliberately ANNUAL, same
    "low-turnover fundamentals factor" reasoning as fetch_point_in_time_roe().

    Returns a DataFrame with columns ["filed_date", "accruals"], one row
    per fiscal year, sorted by filed_date -- `filed_date` is the real date
    the public first saw this figure (the 10-K filing date), never the
    fiscal year-end date, same no-look-ahead discipline as every other
    function here. Degrades to an empty DataFrame rather than crashing,
    same convention as fetch_point_in_time_roe()."""
    columns = ["filed_date", "accruals"]
    cik_map = fetch_cik_map()
    cik = cik_map.get(ticker)
    if cik is None:
        return pd.DataFrame(columns=columns)

    try:
        resp = requests.get(
            COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        if resp.status_code != 200:
            return pd.DataFrame(columns=columns)
        facts = resp.json().get("facts", {}).get("us-gaap", {})
    except Exception:
        return pd.DataFrame(columns=columns)

    net_income = facts.get("NetIncomeLoss", {}).get("units", {}).get("USD", [])
    cfo = facts.get("NetCashProvidedByUsedInOperatingActivities", {}).get("units", {}).get("USD", [])
    assets = facts.get("Assets", {}).get("units", {}).get("USD", [])
    if not net_income or not cfo or not assets:
        return pd.DataFrame(columns=columns)

    ni_10k = {u["accn"]: u for u in net_income if u.get("form") == "10-K" and u.get("end")}
    cfo_10k = {u["accn"]: u for u in cfo if u.get("form") == "10-K" and u.get("end")}
    assets_10k = {u["accn"]: u for u in assets if u.get("form") == "10-K" and u.get("end")}

    rows = []
    for accn, ni in ni_10k.items():
        cf = cfo_10k.get(accn)
        if cf is None or cf.get("end") != ni.get("end"):
            continue
        at = assets_10k.get(accn)
        if at is None:
            continue
        assets_val = at.get("val")
        ni_val = ni.get("val")
        cfo_val = cf.get("val")
        if not assets_val or assets_val <= 0 or ni_val is None or cfo_val is None:
            continue
        accruals = (ni_val - cfo_val) / assets_val
        rows.append({"filed_date": ni["filed"], "accruals": accruals, "fiscal_year_end": ni.get("end")})

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows).sort_values("filed_date")
    out = out.drop_duplicates(subset="fiscal_year_end", keep="last")
    out["filed_date"] = pd.to_datetime(out["filed_date"])
    return out[columns].reset_index(drop=True)


def build_accruals_panel(tickers: list[str], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the wide (dates x tickers) point-in-time ACCRUALS panel the
    "accruals_rank" strategy needs -- exact structural mirror of
    build_roe_panel(), same forward-fill-never-backfill discipline."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        accruals_history = fetch_point_in_time_accruals(ticker)
        if accruals_history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            accruals_history["accruals"].values, index=pd.DatetimeIndex(accruals_history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


ACCRUALS_CACHE_COLLECTION = "Fundamentals_Cache_Accruals"
ACCRUALS_CACHE_MAX_AGE_DAYS = 7  # same reasoning as BVPS_CACHE_MAX_AGE_DAYS
                                           # above -- accruals only changes
                                           # once per fiscal year per ticker


def fetch_point_in_time_accruals_cached(
    ticker: str, max_age_days: int = ACCRUALS_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached wrapper around fetch_point_in_time_accruals() -- exact
    structural mirror of fetch_point_in_time_book_value_per_share_cached().
    Checks MongoDB's `Fundamentals_Cache_Accruals` collection first; only
    re-fetches from SEC EDGAR if the cached entry is missing or older than
    `max_age_days`. Deliberately a SEPARATE function from
    fetch_point_in_time_accruals() -- the backtest path
    (benchmark_random_entry.py) keeps calling the uncached original
    directly, unaffected. Degrades to an uncached live fetch (never
    crashes) if Mongo is unavailable, same convention as every other
    storage-touching function here."""
    import storage  # local import: see fetch_point_in_time_book_value_per_share_cached()'s
                                           # own identical comment
    from datetime import datetime, timezone

    columns = ["filed_date", "accruals"]
    db = None
    try:
        db = storage.get_db()
        cached = db[ACCRUALS_CACHE_COLLECTION].find_one({"_id": ticker})
    except Exception:
        cached = None

    if cached is not None:
        fetched_at = cached.get("fetched_at")
        if fetched_at is not None:
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
            if age_days <= max_age_days:
                rows = cached.get("rows", [])
                if not rows:
                    return pd.DataFrame(columns=columns)
                out = pd.DataFrame(rows)
                out["filed_date"] = pd.to_datetime(out["filed_date"])
                return out[columns]

    fresh = fetch_point_in_time_accruals(ticker)
    if db is not None:
        try:
            db[ACCRUALS_CACHE_COLLECTION].replace_one(
                {"_id": ticker},
                {
                    "_id": ticker,
                    "rows": fresh.assign(filed_date=fresh["filed_date"].astype(str)).to_dict("records"),
                    "fetched_at": datetime.now(timezone.utc),
                },
                upsert=True,
            )
        except Exception:
            pass
    return fresh


def build_accruals_panel_cached(
    tickers: list[str], date_index: pd.DatetimeIndex, max_age_days: int = ACCRUALS_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached counterpart to build_accruals_panel() -- exact same forward-
    fill-never-backfill construction, but sourced from
    fetch_point_in_time_accruals_cached() instead of the uncached fetch.
    Intended for the LIVE daily automation path (ingest.py), not the
    backtest path."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        accruals_history = fetch_point_in_time_accruals_cached(ticker, max_age_days)
        if accruals_history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            accruals_history["accruals"].values, index=pd.DatetimeIndex(accruals_history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


def fetch_point_in_time_cash_profitability(ticker: str) -> pd.DataFrame:
    """Real, point-in-time-correct annual CASH-BASED OPERATING PROFITABILITY
    history for one ticker -- the "cash_profitability_rank" strategy's data
    source. A close cousin of fetch_point_in_time_accruals() above (same 3
    SEC concepts, same fetch), but answers a DIFFERENT question: accruals
    measures earnings QUALITY (how much of reported earnings is real cash
    vs. accounting adjustment); this measures raw cash PROFITABILITY LEVEL
    (how much operating cash flow a business actually generates relative
    to its asset base) -- the literature (Ball, Gerakos, Linnainmaa,
    Nikolaev 2016) finds cash-based operating profitability measures
    outperform profitability measures that still include the accrual
    component, i.e. this is a genuinely different signal from accruals
    even though it's built from the same three underlying numbers.

    Cash Profitability = NetCashProvidedByUsedInOperatingActivities /
    Assets -- HIGHER is better here (buy the top decile, same direction as
    quality_rank's ROE, NOT accruals_rank's inverted "lowest is best"
    convention), since more cash generated per dollar of assets is
    straightforwardly good, unlike accruals where the sign matters for a
    different reason (earnings composition, not magnitude). Scaled by
    total assets, never price or shares outstanding -- same structural
    immunity to the stock-split confound as fetch_point_in_time_accruals().

    Same FLOW-vs-POINT-IN-TIME concept matching discipline as
    fetch_point_in_time_accruals() (CFO matched to Assets by accession
    number, using CFO's own `end` date as the fiscal-year key). Returns a
    DataFrame with columns ["filed_date", "cash_profitability"], one row
    per fiscal year. Degrades to an empty DataFrame rather than crashing,
    same convention as every other function here."""
    columns = ["filed_date", "cash_profitability"]
    cik_map = fetch_cik_map()
    cik = cik_map.get(ticker)
    if cik is None:
        return pd.DataFrame(columns=columns)

    try:
        resp = requests.get(
            COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        if resp.status_code != 200:
            return pd.DataFrame(columns=columns)
        facts = resp.json().get("facts", {}).get("us-gaap", {})
    except Exception:
        return pd.DataFrame(columns=columns)

    cfo = facts.get("NetCashProvidedByUsedInOperatingActivities", {}).get("units", {}).get("USD", [])
    assets = facts.get("Assets", {}).get("units", {}).get("USD", [])
    if not cfo or not assets:
        return pd.DataFrame(columns=columns)

    cfo_10k = {u["accn"]: u for u in cfo if u.get("form") == "10-K" and u.get("end")}
    assets_10k = {u["accn"]: u for u in assets if u.get("form") == "10-K" and u.get("end")}

    rows = []
    for accn, cf in cfo_10k.items():
        at = assets_10k.get(accn)
        if at is None:
            continue
        assets_val = at.get("val")
        cfo_val = cf.get("val")
        if not assets_val or assets_val <= 0 or cfo_val is None:
            continue
        cash_profitability = cfo_val / assets_val
        rows.append({"filed_date": cf["filed"], "cash_profitability": cash_profitability, "fiscal_year_end": cf.get("end")})

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows).sort_values("filed_date")
    out = out.drop_duplicates(subset="fiscal_year_end", keep="last")
    out["filed_date"] = pd.to_datetime(out["filed_date"])
    return out[columns].reset_index(drop=True)


def build_cash_profitability_panel(tickers: list[str], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the wide (dates x tickers) point-in-time CASH PROFITABILITY
    panel the "cash_profitability_rank" strategy needs -- exact structural
    mirror of build_accruals_panel()/build_roe_panel()."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        history = fetch_point_in_time_cash_profitability(ticker)
        if history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            history["cash_profitability"].values, index=pd.DatetimeIndex(history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


CASH_PROFITABILITY_CACHE_COLLECTION = "Fundamentals_Cache_CashProfitability"
CASH_PROFITABILITY_CACHE_MAX_AGE_DAYS = 7  # same reasoning as ACCRUALS_CACHE_MAX_AGE_DAYS above


def fetch_point_in_time_cash_profitability_cached(
    ticker: str, max_age_days: int = CASH_PROFITABILITY_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached wrapper around fetch_point_in_time_cash_profitability() --
    exact structural mirror of fetch_point_in_time_accruals_cached()."""
    import storage  # local import: see fetch_point_in_time_book_value_per_share_cached()'s own identical comment
    from datetime import datetime, timezone

    columns = ["filed_date", "cash_profitability"]
    db = None
    try:
        db = storage.get_db()
        cached = db[CASH_PROFITABILITY_CACHE_COLLECTION].find_one({"_id": ticker})
    except Exception:
        cached = None

    if cached is not None:
        fetched_at = cached.get("fetched_at")
        if fetched_at is not None:
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
            if age_days <= max_age_days:
                rows = cached.get("rows", [])
                if not rows:
                    return pd.DataFrame(columns=columns)
                out = pd.DataFrame(rows)
                out["filed_date"] = pd.to_datetime(out["filed_date"])
                return out[columns]

    fresh = fetch_point_in_time_cash_profitability(ticker)
    if db is not None:
        try:
            db[CASH_PROFITABILITY_CACHE_COLLECTION].replace_one(
                {"_id": ticker},
                {
                    "_id": ticker,
                    "rows": fresh.assign(filed_date=fresh["filed_date"].astype(str)).to_dict("records"),
                    "fetched_at": datetime.now(timezone.utc),
                },
                upsert=True,
            )
        except Exception:
            pass
    return fresh


def build_cash_profitability_panel_cached(
    tickers: list[str], date_index: pd.DatetimeIndex, max_age_days: int = CASH_PROFITABILITY_CACHE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Cached counterpart to build_cash_profitability_panel() -- intended
    for the LIVE daily automation path, not the backtest path."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        history = fetch_point_in_time_cash_profitability_cached(ticker, max_age_days)
        if history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(
            history["cash_profitability"].values, index=pd.DatetimeIndex(history["filed_date"]),
        )
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)


def build_roe_panel(tickers: list[str], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the wide (dates x tickers) point-in-time ROE panel every
    cross-sectional rank strategy in this codebase needs (same shape/role
    as market_data.build_momentum_panel()) -- fetches each ticker's own
    fetch_point_in_time_roe() history, then forward-fills each ticker's
    real filed-date observations across `date_index` so every trading day
    reads "the most recently FILED ROE as of that day," never a future
    value. A ticker with no matched SEC data at all (Canadian cross-listed
    names, fetch failures) contributes an all-NaN column -- degrades to
    "never fires" for that ticker, same convention as every other optional
    per-ticker input in this codebase.

    Network-bound (one companyfacts fetch per ticker) -- paces itself
    internally (REQUEST_DELAY_SEC between requests) across a real
    multi-ticker universe, same courtesy pattern as every other bulk fetch
    loop in this codebase (see run_backtest.py's own fetch loops)."""
    columns = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        roe_history = fetch_point_in_time_roe(ticker)
        if roe_history.empty:
            columns[ticker] = pd.Series(index=date_index, dtype=float)
            continue
        series = pd.Series(roe_history["roe"].values, index=pd.DatetimeIndex(roe_history["filed_date"]))
        # A ticker's SEC history may start well after date_index's own
        # start (early years) or a filing may land between trading days --
        # reindex onto the full date_index, forward-filling from each real
        # filed_date, deliberately NEVER back-filling (a day before the
        # first-ever filed value must stay NaN, not silently borrow a
        # FUTURE figure).
        columns[ticker] = series.reindex(series.index.union(date_index)).ffill().reindex(date_index)
    return pd.DataFrame(columns)
