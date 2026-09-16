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
"""
import time

import pandas as pd
import requests

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
