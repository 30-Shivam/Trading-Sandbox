"""market_data.py -- offline unit tests (no yfinance/FRED/Mongo traffic).

test_batch_fetch.py already covers fetch_ohlcv_batch's happy/chunk-failure
paths and _with_retry's recovery; this file covers the rest: OHLCV cleanup
edge cases, network-timeout degradation, FRED parsing, the qualitative
sub-fetchers, bundle assembly and holdings review. Every network entry point
(yf.download, yf.Ticker, requests.get, time.sleep) is monkeypatched.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

import market_data
import swingtrade


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff / request delays must never actually wait in tests."""
    monkeypatch.setattr(market_data.time, "sleep", lambda s: None)


def _ohlcv(n=30, close=10.0):
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": 1000.0}, index=idx)


# ---------------------------------------------------------------- _finalize_ohlcv / fetch_data

def test_finalize_empty_frame_raises():
    with pytest.raises(RuntimeError, match="no data returned for ticker 'AAA'"):
        market_data._finalize_ohlcv(pd.DataFrame(), "AAA")


def test_finalize_drops_trailing_nan_close_rows():
    df = _ohlcv()
    df.iloc[-2:, df.columns.get_loc("Close")] = np.nan
    out = market_data._finalize_ohlcv(df, "AAA")
    assert len(out) == 28 and not pd.isna(out["Close"].iloc[-1])


def test_finalize_nan_close_only_in_the_middle_is_kept():
    df = _ohlcv()
    df.iloc[5, df.columns.get_loc("Close")] = np.nan
    assert len(market_data._finalize_ohlcv(df, "AAA")) == 30


def test_finalize_all_nan_close_raises_no_usable_data():
    df = _ohlcv()
    df["Close"] = np.nan
    with pytest.raises(RuntimeError, match="no usable"):
        market_data._finalize_ohlcv(df, "AAA")


def test_finalize_flattens_multiindex_columns():
    df = _ohlcv()
    df.columns = pd.MultiIndex.from_product([df.columns, ["AAA"]])
    out = market_data._finalize_ohlcv(df, "AAA")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_finalize_raises_when_sanitizer_leaves_nothing(monkeypatch):
    monkeypatch.setattr(market_data.swingtrade, "sanitize_ohlcv", lambda df: df.iloc[0:0])
    with pytest.raises(RuntimeError, match="after sanitizing"):
        market_data._finalize_ohlcv(_ohlcv(), "AAA")


def test_fetch_data_passes_timeout_and_returns_clean_frame(monkeypatch):
    seen = {}

    def fake_download(ticker, **kw):
        seen.update(kw, ticker=ticker)
        return _ohlcv()

    monkeypatch.setattr(market_data.yf, "download", fake_download)
    out = market_data.fetch_data("AAA")
    assert seen["timeout"] == market_data.YF_REQUEST_TIMEOUT_SEC and seen["ticker"] == "AAA"
    assert len(out) == 30


def test_fetch_data_timeout_propagates(monkeypatch):
    def hang(*a, **k):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(market_data.yf, "download", hang)
    with pytest.raises(requests.exceptions.Timeout):
        market_data.fetch_data("AAA")


def test_fetch_data_empty_download_raises(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data returned"):
        market_data.fetch_data("ZZZ")


# ---------------------------------------------------------------- _with_retry / batch timeouts

def test_with_retry_backoff_is_linear_and_exhausts_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(market_data.time, "sleep", sleeps.append)
    calls = []

    def dead():
        calls.append(1)
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        market_data._with_retry(dead)
    assert len(calls) == market_data.RETRY_ATTEMPTS
    assert sleeps == [5.0, 10.0]  # no sleep after the final failed attempt


def test_with_retry_first_success_never_sleeps(monkeypatch):
    sleeps = []
    monkeypatch.setattr(market_data.time, "sleep", sleeps.append)
    assert market_data._with_retry(lambda a, b=0: a + b, 1, b=2) == 3 and sleeps == []


def test_batch_download_timeout_is_retried_then_skips_chunk(monkeypatch):
    calls = []

    def timeout(*a, **k):
        calls.append(1)
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(market_data.yf, "download", timeout)
    frames, skipped = market_data.fetch_ohlcv_batch(("AAA", "BBB"))
    assert frames == {} and len(calls) == market_data.RETRY_ATTEMPTS
    assert [s[0] for s in skipped] == ["AAA", "BBB"] and "batch download failed" in skipped[0][1]


def test_batch_empty_tickers_makes_no_download(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: pytest.fail("should not download"))
    assert market_data.fetch_ohlcv_batch(()) == ({}, [])


def test_batch_ticker_missing_from_multiindex_is_skipped_not_fatal(monkeypatch):
    data = pd.concat({"AAA": _ohlcv()}, axis=1)  # BBB absent -> KeyError per ticker
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: data)
    frames, skipped = market_data.fetch_ohlcv_batch(("AAA", "BBB"))
    assert list(frames) == ["AAA"] and [s[0] for s in skipped] == ["BBB"]


# ---------------------------------------------------------------- earnings / headlines

class _Ticker:
    """Minimal stand-in for yf.Ticker; any attribute may be a value or an exception to raise."""

    def __init__(self, **attrs):
        self._attrs = attrs

    def __getattr__(self, name):
        val = self.__dict__["_attrs"].get(name)
        if isinstance(val, Exception):
            raise val
        return val

    def get_earnings_dates(self, limit):
        val = self._attrs.get("earnings")
        if isinstance(val, Exception):
            raise val
        return val


NOW = pd.Timestamp("2026-06-01", tz="UTC")


def _earnings(*dates):
    return pd.DataFrame({"EPS": [1.0] * len(dates)}, index=pd.DatetimeIndex(dates, tz="America/New_York"))


def test_next_earnings_picks_earliest_future_date():
    t = _Ticker(earnings=_earnings("2026-04-20", "2026-08-05", "2026-07-20"))
    assert market_data.get_next_earnings_date(t, NOW) == pd.Timestamp("2026-07-20", tz="America/New_York").tz_convert("UTC")


@pytest.mark.parametrize("earnings", [None, pd.DataFrame(), RuntimeError("yahoo 429"), _earnings("2026-01-15")])
def test_next_earnings_unavailable_or_all_past_is_none(earnings):
    assert market_data.get_next_earnings_date(_Ticker(earnings=earnings), NOW) is None


def test_headlines_handles_nested_flat_missing_and_garbage_items():
    news = [
        {"content": {"title": "nested"}},
        {"title": "flat"},
        {"content": {"title": None}, "title": "fallback"},
        {"content": {}},  # no title anywhere -> skipped
        "not-a-dict",
        {"title": 123},
    ]
    assert market_data.get_recent_headlines(_Ticker(news=news), count=10) == ["nested", "flat", "fallback", "123"]


def test_headlines_respects_count_and_failure_modes():
    news = [{"title": f"h{i}"} for i in range(5)]
    assert market_data.get_recent_headlines(_Ticker(news=news), count=2) == ["h0", "h1"]
    assert market_data.get_recent_headlines(_Ticker(news=None)) == []
    assert market_data.get_recent_headlines(_Ticker(news=RuntimeError("boom"))) == []


def test_multi_and_macro_headlines_degrade_to_empty_on_constructor_failure(monkeypatch):
    def boom(symbol):
        raise requests.exceptions.ConnectionError("dns")

    monkeypatch.setattr(market_data.yf, "Ticker", boom)
    assert market_data.get_multi_headlines("AAA") == []
    assert market_data.get_macro_headlines() == []


def test_multi_headlines_happy_path(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: _Ticker(news=[{"title": f"{s} up"}]))
    assert market_data.get_multi_headlines("AAA") == ["AAA up"]


# ---------------------------------------------------------------- macro snapshot

def test_macro_snapshot_computes_vix_level_and_change(monkeypatch):
    vix = _ohlcv(n=3)
    vix["Close"] = [20.0, 20.0, 22.0]
    monkeypatch.setattr(market_data, "fetch_data", lambda t: vix)
    monkeypatch.setattr(market_data, "get_macro_headlines", lambda: ["fed"])
    assert market_data.get_macro_snapshot() == {"vix": 22.0, "vix_change_pct": 10.0, "headlines": ["fed"]}


def test_macro_snapshot_single_bar_has_level_but_no_change(monkeypatch):
    monkeypatch.setattr(market_data, "fetch_data", lambda t: _ohlcv(n=1, close=18.0))
    monkeypatch.setattr(market_data, "get_macro_headlines", lambda: [])
    snap = market_data.get_macro_snapshot()
    assert snap["vix"] == 18.0 and snap["vix_change_pct"] is None


def test_macro_snapshot_zero_prev_close_avoids_division_by_zero(monkeypatch):
    vix = _ohlcv(n=2)
    vix["Close"] = [0.0, 5.0]
    monkeypatch.setattr(market_data, "fetch_data", lambda t: vix)
    monkeypatch.setattr(market_data, "get_macro_headlines", lambda: [])
    assert market_data.get_macro_snapshot()["vix_change_pct"] is None


def test_macro_snapshot_fetch_timeout_degrades_each_field(monkeypatch):
    def timeout(t):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(market_data, "fetch_data", timeout)
    monkeypatch.setattr(market_data, "get_macro_headlines", lambda: ["still here"])
    assert market_data.get_macro_snapshot() == {"vix": None, "vix_change_pct": None, "headlines": ["still here"]}


# ---------------------------------------------------------------- FRED

def _fred_response(payload=None, status_error=None, json_error=None):
    def raise_for_status():
        if status_error:
            raise status_error

    def json_():
        if json_error:
            raise json_error
        return payload

    return SimpleNamespace(raise_for_status=raise_for_status, json=json_)


def test_fred_available_follows_env(monkeypatch):
    monkeypatch.delenv(market_data.FRED_API_KEY_ENV, raising=False)
    assert market_data.fred_available() is False
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "k")
    assert market_data.fred_available() is True
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "")
    assert market_data.fred_available() is False


def test_fred_without_key_makes_no_request(monkeypatch):
    monkeypatch.delenv(market_data.FRED_API_KEY_ENV, raising=False)
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: pytest.fail("no request expected"))
    assert market_data.fetch_fred_series("T10Y2Y", "2026-01-01", "2026-02-01") is None


def test_fred_parses_sorts_and_drops_placeholders(monkeypatch):
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "secret")
    captured = {}
    payload = {"observations": [
        {"date": "2026-01-05", "value": "0.30"},
        {"date": "2026-01-02", "value": "0.25"},
        {"date": "2026-01-03", "value": "."},        # FRED's missing-value placeholder
        {"date": "2026-01-04", "value": ""},
        {"date": "2026-01-06", "value": None},
        {"date": "2026-01-07", "value": "abc"},      # unparseable
        {"value": "1.0"},                            # missing date
        {"date": "not-a-date", "value": "1.0"},
    ]}

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return _fred_response(payload)

    monkeypatch.setattr(market_data.requests, "get", fake_get)
    out = market_data.fetch_fred_series("T10Y2Y", "2026-01-01", pd.Timestamp("2026-02-01"))
    assert list(out.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert list(out.values) == [0.25, 0.30]
    assert captured["url"] == market_data.FRED_BASE_URL and captured["timeout"] == 15
    assert captured["params"]["series_id"] == "T10Y2Y"
    assert captured["params"]["observation_start"] == "2026-01-01"
    assert captured["params"]["observation_end"] == "2026-02-01"


@pytest.mark.parametrize("response", [
    _fred_response(status_error=requests.exceptions.HTTPError("500")),
    _fred_response(json_error=ValueError("not json")),
    _fred_response({"observations": []}),
    _fred_response({}),
    _fred_response({"observations": [{"date": "2026-01-02", "value": "."}]}),
])
def test_fred_bad_responses_return_none(monkeypatch, response):
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "secret")
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: response)
    assert market_data.fetch_fred_series("T10Y2Y", "2026-01-01", "2026-02-01") is None


@pytest.mark.parametrize("exc", [requests.exceptions.Timeout("t/o"), requests.exceptions.ConnectionError("down")])
def test_fred_network_failure_returns_none(monkeypatch, exc):
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "secret")

    def fail(*a, **k):
        raise exc

    monkeypatch.setattr(market_data.requests, "get", fail)
    assert market_data.fetch_fred_series("T10Y2Y", "2026-01-01", "2026-02-01") is None


# ---------------------------------------------------------------- qualitative sub-fetchers

def test_short_interest_trend_thresholds():
    def si(now, prior, pct=0.1):
        t = _Ticker(info={"shortPercentOfFloat": pct, "sharesShort": now, "sharesShortPriorMonth": prior})
        return market_data._get_short_interest(t)

    assert si(106, 100) == {"pct_of_float": 10.0, "trend": "Increasing"}
    assert si(94, 100)["trend"] == "Decreasing"
    assert si(103, 100)["trend"] == "Flat"
    assert si(100, 0)["trend"] is None          # zero prior month -> no division
    assert si(None, 100)["trend"] is None


def test_short_interest_none_when_nothing_available_or_info_fails():
    assert market_data._get_short_interest(_Ticker(info={})) is None
    assert market_data._get_short_interest(_Ticker(info=RuntimeError("429"))) is None


def test_insider_direction_classification():
    def txns(rows):
        return _Ticker(insider_transactions=pd.DataFrame(rows), info={})

    buy = [{"Text": "Purchase at price", "Value": 1000.0}, {"Text": "Sale", "Value": 100.0}]
    sell = [{"Text": "Sale at price", "Value": 1000.0}, {"Text": "Purchase", "Value": 100.0}]
    mixed = [{"Text": "Sale", "Value": 100.0}, {"Text": "Purchase", "Value": 100.0}]
    assert market_data._get_insider_data(txns(buy))["net_direction"] == "Buying"
    assert market_data._get_insider_data(txns(sell))["net_direction"] == "Selling"
    assert market_data._get_insider_data(txns(mixed))["net_direction"] == "Mixed"


def test_insider_ignores_gifts_and_nan_values_and_reads_ownership():
    rows = [{"Text": "Gift to charity", "Value": 9e9}, {"Text": "Sale", "Value": np.nan}]
    t = _Ticker(insider_transactions=pd.DataFrame(rows),
                info={"heldPercentInstitutions": 0.7, "heldPercentInsiders": 0.02})
    assert market_data._get_insider_data(t) == {"net_direction": None, "pct_institutions": 0.7, "pct_insiders": 0.02}


def test_insider_none_when_everything_fails():
    t = _Ticker(insider_transactions=RuntimeError("x"), info=RuntimeError("y"))
    assert market_data._get_insider_data(t) is None


def test_analyst_data_partial_and_total_failure():
    recs = pd.DataFrame([{"period": "0m", "strongBuy": 1, "buy": 2, "hold": 3, "sell": 0, "strongSell": 0},
                         {"period": "-1m", "strongBuy": 0, "buy": 2, "hold": 4, "sell": 0, "strongSell": 0}])
    t = _Ticker(analyst_price_targets=RuntimeError("x"), recommendations=recs,
                upgrades_downgrades=RuntimeError("y"))
    out = market_data._get_analyst_data(t)
    assert out["targets"] == {} and out["recent_actions"] == []
    assert out["trend"][0].startswith("-1m:") and out["trend"][1].startswith("0m:")  # oldest -> newest
    broken = _Ticker(analyst_price_targets=RuntimeError("a"), recommendations=RuntimeError("b"),
                     upgrades_downgrades=RuntimeError("c"))
    assert market_data._get_analyst_data(broken) is None


def test_recent_filings_limits_and_skips_non_dicts():
    filings = [{"type": "10-K", "date": "2026-02-01", "title": "Annual"}, "junk",
               {"type": "8-K", "date": "2026-03-01", "title": "Event"}, {"type": "4"}]
    out = market_data._get_recent_filings(_Ticker(sec_filings=filings), count=3)
    assert out == [{"type": "10-K", "date": "2026-02-01", "title": "Annual"},
                   {"type": "8-K", "date": "2026-03-01", "title": "Event"}]
    assert market_data._get_recent_filings(_Ticker(sec_filings=RuntimeError("x"))) == []
    assert market_data._get_recent_filings(_Ticker(sec_filings=None)) == []


def test_options_sentiment_ratio_and_degradation():
    chain = SimpleNamespace(calls=pd.DataFrame({"volume": [100.0, np.nan]}), puts=pd.DataFrame({"volume": [50.0, 25.0]}))

    class Opts(_Ticker):
        def option_chain(self, exp):
            return chain

    assert market_data._get_options_sentiment(Opts(options=("2026-07-17",))) == {"put_call_volume_ratio": 0.75}
    assert market_data._get_options_sentiment(Opts(options=())) is None          # no listed options
    zero_calls = SimpleNamespace(calls=pd.DataFrame({"volume": [0.0]}), puts=pd.DataFrame({"volume": [5.0]}))

    class ZeroOpts(_Ticker):
        def option_chain(self, exp):
            return zero_calls

    assert market_data._get_options_sentiment(ZeroOpts(options=("x",))) is None

    class Broken(_Ticker):
        def option_chain(self, exp):
            raise requests.exceptions.Timeout()

    assert market_data._get_options_sentiment(Broken(options=("x",))) is None


def test_qualitative_snapshot_constructor_failure_returns_empty_shape(monkeypatch):
    def boom(s):
        raise RuntimeError("no yahoo")

    monkeypatch.setattr(market_data.yf, "Ticker", boom)
    assert market_data.get_qualitative_snapshot("AAA") == {
        "analyst": None, "insider": None, "short_interest": None, "filings": [], "options": None}


def test_qualitative_snapshot_one_failing_section_does_not_block_others(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: _Ticker(
        sec_filings=[{"type": "10-K", "date": "d", "title": "t"}], info=RuntimeError("x"),
        analyst_price_targets=RuntimeError("x"), recommendations=None, upgrades_downgrades=None,
        insider_transactions=None, options=None))
    snap = market_data.get_qualitative_snapshot("AAA")
    assert snap["filings"] == [{"type": "10-K", "date": "d", "title": "t"}]
    assert snap["analyst"] is None and snap["short_interest"] is None and snap["options"] is None


# ---------------------------------------------------------------- currency / market gate

@pytest.mark.parametrize("ticker,expected", [("CSU.TO", "CAD"), ("ABC.V", "CAD"), ("AAPL", "USD"), ("BRK-B", "USD"),
                                             ("TOWER", "USD"), ("", "USD")])
def test_ticker_currency_suffix(ticker, expected):
    assert market_data.get_ticker_currency(ticker) == expected


def test_check_market_uptrend_delegates_to_swingtrade(monkeypatch):
    seen = {}
    monkeypatch.setattr(market_data, "fetch_data", lambda t: seen.setdefault("t", t) and _ohlcv())
    monkeypatch.setattr(market_data.swingtrade, "is_market_uptrend", lambda df, cfg: (True, 1.0, 2.0))
    assert market_data.check_market_uptrend(swingtrade.DEFAULT_CONFIG) == (True, 1.0, 2.0)
    assert seen["t"] == "SPY"


def test_check_market_uptrend_fetch_failure_propagates(monkeypatch):
    def boom(t):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(market_data, "fetch_data", boom)
    with pytest.raises(requests.exceptions.Timeout):
        market_data.check_market_uptrend(swingtrade.DEFAULT_CONFIG)


# ---------------------------------------------------------------- fetch_ticker_bundle

def _patch_bundle_deps(monkeypatch, frames, skipped=(), spy=True, extras=(None, "headline")):
    def fake_fetch_data(ticker):
        if ticker == "SPY" and not spy:
            raise requests.exceptions.Timeout("spy timeout")
        return _ohlcv()

    monkeypatch.setattr(market_data, "fetch_data", fake_fetch_data)
    monkeypatch.setattr(market_data, "fetch_ohlcv_batch", lambda tickers: (frames, list(skipped)))
    monkeypatch.setattr(market_data, "_fetch_extras", lambda t, now: extras)
    monkeypatch.delenv(market_data.FRED_API_KEY_ENV, raising=False)


def test_bundle_happy_path_shape(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"AAA": _ohlcv(), "CSU.TO": _ohlcv()})
    bundle, market_df, skipped, sector_data, yield_curve = market_data.fetch_ticker_bundle(("AAA", "CSU.TO"))
    assert set(bundle) == {"AAA", "CSU.TO"} and market_df is not None
    assert skipped == [] and sector_data == {} and yield_curve is None
    assert bundle["AAA"]["top_headline"] == "headline" and bundle["AAA"]["next_earnings"] is None
    assert bundle["CSU.TO"]["currency"] == "CAD" and bundle["AAA"]["currency"] == "USD"


def test_bundle_spy_timeout_gives_none_market_df_but_keeps_tickers(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"AAA": _ohlcv()}, spy=False)
    bundle, market_df, *_ = market_data.fetch_ticker_bundle(("AAA",))
    assert market_df is None and "AAA" in bundle


def test_bundle_surfaces_fetch_skips(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"AAA": _ohlcv()}, skipped=[("BBB", "no data")])
    _, _, skipped, _, _ = market_data.fetch_ticker_bundle(("AAA", "BBB"))
    assert skipped == [("BBB", "no data")]


def test_bundle_extras_failure_does_not_drop_ticker(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"AAA": _ohlcv()})

    def boom(t, now):
        raise RuntimeError("429")

    monkeypatch.setattr(market_data, "_fetch_extras", boom)
    bundle, *_ = market_data.fetch_ticker_bundle(("AAA",))
    assert bundle["AAA"]["next_earnings"] is None and bundle["AAA"]["top_headline"] == ""


def test_bundle_fetches_each_distinct_sector_etf_once_and_tolerates_failure(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"A": _ohlcv(), "B": _ohlcv(), "C": _ohlcv()})
    fetched = []
    etf = market_data.SECTOR_ETF
    sectors = sorted(etf)[:2]

    def fake_fetch_data(ticker):
        fetched.append(ticker)
        if ticker == etf[sectors[1]]:
            raise requests.exceptions.Timeout("etf timeout")
        return _ohlcv()

    monkeypatch.setattr(market_data, "fetch_data", fake_fetch_data)
    lookup = {"A": sectors[0], "B": sectors[0], "C": sectors[1], "D": "Not A Sector"}
    _, _, _, sector_data, _ = market_data.fetch_ticker_bundle(("A", "B", "C", "D"), sector_lookup=lookup)
    assert fetched.count(etf[sectors[0]]) == 1
    assert list(sector_data) == [sectors[0]]   # failed sector ETF dropped, unknown sector ignored


def test_bundle_fetches_yield_curve_only_with_key_and_survives_failure(monkeypatch):
    _patch_bundle_deps(monkeypatch, {"AAA": _ohlcv()})
    series = pd.Series([0.1], index=pd.DatetimeIndex(["2026-01-02"]))
    monkeypatch.setenv(market_data.FRED_API_KEY_ENV, "k")
    monkeypatch.setattr(market_data, "fetch_fred_series", lambda *a, **k: series)
    assert market_data.fetch_ticker_bundle(("AAA",))[4] is series

    def boom(*a, **k):
        raise RuntimeError("fred down")

    monkeypatch.setattr(market_data, "fetch_fred_series", boom)
    assert market_data.fetch_ticker_bundle(("AAA",))[4] is None


# ---------------------------------------------------------------- panels / scoring / scan / review

def test_build_pair_price_panels_single_member_sectors_excluded():
    bundle = {t: {"df": _ohlcv()} for t in "ABC"}
    panels = market_data.build_pair_price_panels(bundle, {"A": "Tech", "B": "Tech", "C": "Energy"})
    assert list(panels) == ["Tech"] and list(panels["Tech"].columns) == ["A", "B"]
    assert market_data.build_pair_price_panels(bundle, None) != {}   # all three share "Unknown"


def test_build_momentum_panel_is_wide_close_frame():
    bundle = {"A": {"df": _ohlcv(close=1.0)}, "B": {"df": _ohlcv(close=2.0)}}
    panel = market_data.build_momentum_panel(bundle)
    assert list(panel.columns) == ["A", "B"] and panel["B"].iloc[0] == 2.0


def test_score_bundle_skips_failing_ticker_and_tags_currency(uptrend_ohlcv):
    bundle = {
        "GOOD": {"df": uptrend_ohlcv, "next_earnings": None, "top_headline": "", "currency": "CAD"},
        "SHORT": {"df": uptrend_ohlcv.iloc[:10], "next_earnings": None, "top_headline": "", "currency": "USD"},
    }
    results, skipped = market_data.score_bundle_for_strategy(bundle, None, swingtrade.DEFAULT_CONFIG)
    assert [r["Ticker"] for r in results] == ["GOOD"] and results[0]["Currency"] == "CAD"
    assert [s[0] for s in skipped] == ["SHORT"] and skipped[0][1]


def test_score_bundle_empty_bundle():
    assert market_data.score_bundle_for_strategy({}, None, swingtrade.DEFAULT_CONFIG) == ([], [])


def test_scan_tickers_combines_fetch_and_score_skips(monkeypatch):
    monkeypatch.setattr(market_data, "fetch_ticker_bundle",
                        lambda tickers, sector_lookup=None: ({}, None, [("F", "fetch")], {}, None))
    monkeypatch.setattr(market_data, "score_bundle_for_strategy",
                        lambda *a, **k: ([{"Ticker": "OK"}], [("S", "score")]))
    results, skipped = market_data.scan_tickers(("F", "S", "OK"), swingtrade.DEFAULT_CONFIG)
    assert results == [{"Ticker": "OK"}] and skipped == [("F", "fetch"), ("S", "score")]


def test_review_holdings_collects_results_and_skips_failures(monkeypatch):
    def fake_fetch(ticker):
        if ticker == "BAD":
            raise requests.exceptions.Timeout("slow")
        return _ohlcv()

    monkeypatch.setattr(market_data, "fetch_data", fake_fetch)
    monkeypatch.setattr(market_data.swingtrade, "review_holding",
                        lambda ticker, df, avg_cost, cfg: {"Ticker": ticker, "Avg_Cost": avg_cost})
    results, skipped = market_data.review_holdings({"AAA": 10.0, "BAD": 5.0, "CSU.TO": 100.0}, swingtrade.DEFAULT_CONFIG)
    assert [(r["Ticker"], r["Currency"]) for r in results] == [("AAA", "USD"), ("CSU.TO", "CAD")]
    assert skipped == [("BAD", "slow")]


def test_review_holdings_empty_and_throttles_between_calls(monkeypatch):
    assert market_data.review_holdings({}, swingtrade.DEFAULT_CONFIG) == ([], [])
    sleeps = []
    monkeypatch.setattr(market_data.time, "sleep", sleeps.append)
    monkeypatch.setattr(market_data, "fetch_data", lambda t: _ohlcv())
    monkeypatch.setattr(market_data.swingtrade, "review_holding", lambda *a: {})
    market_data.review_holdings({"A": 1.0, "B": 1.0, "C": 1.0}, swingtrade.DEFAULT_CONFIG)
    assert sleeps == [market_data.REQUEST_DELAY_SEC] * 2   # not before the first call
