"""sec_fundamentals.py -- real, point-in-time fundamentals via SEC EDGAR's
free XBRL company-facts API (2026-09-16, the fix for yfinance's confirmed
5-7-quarter fundamentals depth limit). All network calls mocked (no live
SEC requests in the test suite, same `monkeypatch` convention every other
network-touching test in this codebase uses) -- these tests exist
specifically to verify the ONE property that matters most here: a fact
filed AFTER `as_of` must NEVER be visible to a lookup at `as_of` (real
look-ahead risk, not a hypothetical one -- SEC filings routinely lag their
own period-end date by 45-90+ days).
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import sec_fundamentals
import storage


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _reset_cik_cache():
    sec_fundamentals._cik_map_cache = None
    yield
    sec_fundamentals._cik_map_cache = None


def test_fetch_cik_map_parses_and_caches(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        })

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    m1 = sec_fundamentals.fetch_cik_map()
    m2 = sec_fundamentals.fetch_cik_map()  # second call should hit the cache, not refetch
    assert m1 == {"AAPL": "0000320193", "MSFT": "0000789019"}
    assert m2 is m1
    assert len(calls) == 1


def test_fetch_cik_map_degrades_to_empty_dict_on_failure(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise ConnectionError("network down")

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    assert sec_fundamentals.fetch_cik_map() == {}


def _fake_companyfacts(net_income_rows, equity_rows):
    return _FakeResponse({
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": net_income_rows}},
                "StockholdersEquity": {"units": {"USD": equity_rows}},
            }
        }
    })


def test_fetch_point_in_time_roe_computes_from_matched_10k_pair(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts(
            net_income_rows=[
                {"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"},
            ],
            equity_rows=[
                {"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"},
            ],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    roe = sec_fundamentals.fetch_point_in_time_roe("TEST")
    assert len(roe) == 1
    assert roe.iloc[0]["roe"] == pytest.approx(0.1)
    assert roe.iloc[0]["filed_date"] == pd.Timestamp("2021-02-15")


def test_fetch_point_in_time_roe_ignores_non_10k_forms(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts(
            net_income_rows=[
                {"accn": "Q1", "end": "2020-09-30", "val": 250_000, "form": "10-Q", "filed": "2020-11-01"},
            ],
            equity_rows=[
                {"accn": "Q1", "end": "2020-09-30", "val": 5_000_000, "form": "10-Q", "filed": "2020-11-01"},
            ],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    roe = sec_fundamentals.fetch_point_in_time_roe("TEST")
    assert roe.empty


def test_fetch_point_in_time_roe_wont_pair_mismatched_accession_or_period(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts(
            net_income_rows=[
                {"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"},
            ],
            equity_rows=[
                # Different accession AND different period end -- must NOT pair with A1's net income.
                {"accn": "A2", "end": "2019-12-31", "val": 9_000_000, "form": "10-K", "filed": "2020-02-10"},
            ],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    roe = sec_fundamentals.fetch_point_in_time_roe("TEST")
    assert roe.empty


def test_fetch_point_in_time_roe_keeps_latest_filed_amendment_per_fiscal_year(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts(
            net_income_rows=[
                {"accn": "ORIG", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"},
                {"accn": "AMEND", "end": "2020-12-31", "val": 900_000, "form": "10-K", "filed": "2021-06-01"},
            ],
            equity_rows=[
                {"accn": "ORIG", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"},
                {"accn": "AMEND", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-06-01"},
            ],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    roe = sec_fundamentals.fetch_point_in_time_roe("TEST")
    assert len(roe) == 1
    assert roe.iloc[0]["roe"] == pytest.approx(0.09)  # 900,000 / 10,000,000 -- the LATER-filed figure
    assert roe.iloc[0]["filed_date"] == pd.Timestamp("2021-06-01")


def test_fetch_point_in_time_roe_skips_zero_or_negative_equity(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts(
            net_income_rows=[
                {"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"},
            ],
            equity_rows=[
                {"accn": "A1", "end": "2020-12-31", "val": -500_000, "form": "10-K", "filed": "2021-02-15"},
            ],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    roe = sec_fundamentals.fetch_point_in_time_roe("TEST")
    assert roe.empty


def test_fetch_point_in_time_roe_unmapped_ticker_returns_empty_without_network_call(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {})
    calls = []
    monkeypatch.setattr(sec_fundamentals.requests, "get", lambda *a, **k: calls.append(1))
    roe = sec_fundamentals.fetch_point_in_time_roe("ATD")
    assert roe.empty
    assert calls == []  # no CIK -> never even tries the network, same convention as every other fetch function


def _fake_companyfacts_bvps(equity_rows, shares_rows):
    return _FakeResponse({
        "facts": {
            "us-gaap": {"StockholdersEquity": {"units": {"USD": equity_rows}}},
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares_rows}}},
        }
    })


class _FakeYfTicker:
    def __init__(self, splits):
        self.splits = splits


def _mock_no_splits(monkeypatch):
    """No real stock splits -- _cumulative_split_factor() should return 1.0
    (no adjustment), same as every existing BVPS test's pre-fix expected
    values."""
    monkeypatch.setattr(sec_fundamentals.yf, "Ticker", lambda t: _FakeYfTicker(pd.Series(dtype=float)))


def test_fetch_point_in_time_bvps_computes_from_matched_10k_pair(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})
    _mock_no_splits(monkeypatch)

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_bvps(
            equity_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
            shares_rows=[{"accn": "A1", "end": "2021-02-10", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("TEST")
    assert len(bvps) == 1
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(10.0)  # 10,000,000 / 1,000,000
    assert bvps.iloc[0]["filed_date"] == pd.Timestamp("2021-02-15")


def test_fetch_point_in_time_bvps_ignores_non_10k_forms(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})
    _mock_no_splits(monkeypatch)

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_bvps(
            equity_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 5_000_000, "form": "10-Q", "filed": "2020-11-01"}],
            shares_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 500_000, "form": "10-Q", "filed": "2020-11-01"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("TEST")
    assert bvps.empty


def test_fetch_point_in_time_bvps_wont_pair_mismatched_accession(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})
    _mock_no_splits(monkeypatch)

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_bvps(
            equity_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
            shares_rows=[{"accn": "A2", "end": "2020-02-10", "val": 900_000, "form": "10-K", "filed": "2020-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("TEST")
    assert bvps.empty


def test_fetch_point_in_time_bvps_skips_zero_shares(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})
    _mock_no_splits(monkeypatch)

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_bvps(
            equity_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
            shares_rows=[{"accn": "A1", "end": "2021-02-10", "val": 0, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("TEST")
    assert bvps.empty


def test_fetch_point_in_time_bvps_unmapped_ticker_returns_empty_without_network_call(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {})
    calls = []
    monkeypatch.setattr(sec_fundamentals.requests, "get", lambda *a, **k: calls.append(1))
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("ATD")
    assert bvps.empty
    assert calls == []


def test_cumulative_split_factor_multiplies_only_splits_after_filed_date():
    splits = pd.Series(
        [4.0, 10.0],
        index=pd.DatetimeIndex(["2021-06-01", "2024-06-10"]),
    )
    # A filing well before BOTH splits picks up both (4x * 10x = 40x).
    assert sec_fundamentals._cumulative_split_factor(splits, "2020-01-01") == pytest.approx(40.0)
    # A filing between the two splits picks up only the later one.
    assert sec_fundamentals._cumulative_split_factor(splits, "2022-01-01") == pytest.approx(10.0)
    # A filing after both splits gets no adjustment.
    assert sec_fundamentals._cumulative_split_factor(splits, "2025-01-01") == pytest.approx(1.0)


def test_cumulative_split_factor_empty_splits_returns_one():
    assert sec_fundamentals._cumulative_split_factor(pd.Series(dtype=float), "2020-01-01") == 1.0
    assert sec_fundamentals._cumulative_split_factor(None, "2020-01-01") == 1.0


def test_fetch_point_in_time_bvps_adjusts_pre_split_filing_for_a_later_split(monkeypatch):
    """The real bug this test locks in: a 10:1 split AFTER a filing must
    scale that filing's raw book-value-per-share DOWN by 10x, so it lands
    on the same modern post-split share basis run_backtest.fetch_history()'s
    Close prices already use (2026-09-21 fix, NVDA was the real incident --
    see this module's own top-of-file writeup)."""
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})
    monkeypatch.setattr(
        sec_fundamentals.yf, "Ticker",
        lambda t: _FakeYfTicker(pd.Series([10.0], index=pd.DatetimeIndex(["2024-06-10"]))),
    )

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_bvps(
            # Pre-split filing (raw, as-filed share count) -- filed BEFORE the 10:1 split.
            equity_rows=[{"accn": "A1", "end": "2023-01-31", "val": 22_000_000_000, "form": "10-K", "filed": "2023-02-24"}],
            shares_rows=[{"accn": "A1", "end": "2023-02-20", "val": 2_470_000_000, "form": "10-K", "filed": "2023-02-24"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share("TEST")
    raw_bvps = 22_000_000_000 / 2_470_000_000
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(raw_bvps / 10.0)


def test_build_book_value_per_share_panel_forward_fills_without_look_ahead(monkeypatch):
    date_index = pd.date_range("2020-01-01", periods=10, freq="D")

    def fake_fetch(ticker):
        if ticker == "A":
            return pd.DataFrame({"filed_date": [pd.Timestamp("2020-01-05")], "book_value_per_share": [20.0]})
        return pd.DataFrame(columns=["filed_date", "book_value_per_share"])

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share", fake_fetch)
    panel = sec_fundamentals.build_book_value_per_share_panel(["A", "B"], date_index)

    assert pd.isna(panel.loc["2020-01-01", "A"])
    assert pd.isna(panel.loc["2020-01-04", "A"])
    assert panel.loc["2020-01-05", "A"] == pytest.approx(20.0)
    assert panel.loc["2020-01-10", "A"] == pytest.approx(20.0)
    assert panel["B"].isna().all()


def test_build_roe_panel_forward_fills_without_look_ahead(monkeypatch):
    date_index = pd.date_range("2020-01-01", periods=10, freq="D")

    def fake_fetch(ticker):
        if ticker == "A":
            return pd.DataFrame({"filed_date": [pd.Timestamp("2020-01-05")], "roe": [0.2]})
        return pd.DataFrame(columns=["filed_date", "roe"])

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_roe", fake_fetch)
    panel = sec_fundamentals.build_roe_panel(["A", "B"], date_index)

    # Before the filed_date, must read NaN -- NEVER borrow the future value.
    assert pd.isna(panel.loc["2020-01-01", "A"])
    assert pd.isna(panel.loc["2020-01-04", "A"])
    # On and after the filed_date, forward-filled with the real value.
    assert panel.loc["2020-01-05", "A"] == pytest.approx(0.2)
    assert panel.loc["2020-01-10", "A"] == pytest.approx(0.2)
    # A ticker with no SEC data at all (e.g. a Canadian cross-listed name) -> all-NaN column.
    assert panel["B"].isna().all()


class _FakeCollection:
    def __init__(self, docs=None):
        self._docs = docs or {}
        self.replace_calls = []

    def find_one(self, query):
        return self._docs.get(query["_id"])

    def replace_one(self, query, doc, upsert=False):
        self.replace_calls.append(doc)
        self._docs[query["_id"]] = doc


class _FakeDB:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, name):
        return self._collection


def test_fetch_point_in_time_bvps_cached_uses_fresh_cache_without_network_call(monkeypatch):
    fake_collection = _FakeCollection({
        "TEST": {
            "_id": "TEST",
            "rows": [{"filed_date": "2021-02-15", "book_value_per_share": 4.2}],
            "fetched_at": datetime.now(timezone.utc) - timedelta(days=1),  # well within the 7-day window
        },
    })
    monkeypatch.setattr(storage, "get_db", lambda: _FakeDB(fake_collection))
    calls = []
    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share", lambda t: calls.append(t))

    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share_cached("TEST")
    assert calls == []  # cache hit -- never touched the real (network-bound) fetch
    assert len(bvps) == 1
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(4.2)
    assert bvps.iloc[0]["filed_date"] == pd.Timestamp("2021-02-15")


def test_fetch_point_in_time_bvps_cached_refetches_when_stale(monkeypatch):
    fake_collection = _FakeCollection({
        "TEST": {
            "_id": "TEST",
            "rows": [{"filed_date": "2021-02-15", "book_value_per_share": 4.2}],
            "fetched_at": datetime.now(timezone.utc) - timedelta(days=30),  # past the 7-day window
        },
    })
    monkeypatch.setattr(storage, "get_db", lambda: _FakeDB(fake_collection))
    fresh_df = pd.DataFrame({"filed_date": [pd.Timestamp("2026-01-01")], "book_value_per_share": [9.9]})
    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share", lambda t: fresh_df)

    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share_cached("TEST")
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(9.9)  # got the FRESH value, not the stale cache
    assert len(fake_collection.replace_calls) == 1  # cache was refreshed


def test_fetch_point_in_time_bvps_cached_refetches_when_missing(monkeypatch):
    fake_collection = _FakeCollection({})
    monkeypatch.setattr(storage, "get_db", lambda: _FakeDB(fake_collection))
    fresh_df = pd.DataFrame({"filed_date": [pd.Timestamp("2026-01-01")], "book_value_per_share": [5.5]})
    calls = []

    def fake_fetch(t):
        calls.append(t)
        return fresh_df

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share", fake_fetch)
    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share_cached("TEST")
    assert calls == ["TEST"]
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(5.5)
    assert len(fake_collection.replace_calls) == 1


def test_fetch_point_in_time_bvps_cached_degrades_to_live_fetch_when_mongo_unavailable(monkeypatch):
    def raise_not_configured():
        raise RuntimeError("MONGODB_URI not set")

    monkeypatch.setattr(storage, "get_db", raise_not_configured)
    fresh_df = pd.DataFrame({"filed_date": [pd.Timestamp("2026-01-01")], "book_value_per_share": [7.7]})
    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share", lambda t: fresh_df)

    bvps = sec_fundamentals.fetch_point_in_time_book_value_per_share_cached("TEST")
    assert bvps.iloc[0]["book_value_per_share"] == pytest.approx(7.7)  # never crashed, just skipped the cache


def test_build_book_value_per_share_panel_cached_forward_fills_without_look_ahead(monkeypatch):
    date_index = pd.date_range("2020-01-01", periods=10, freq="D")
    monkeypatch.setattr(storage, "get_db", lambda: (_ for _ in ()).throw(RuntimeError("no mongo in this test")))

    def fake_fetch_cached(ticker, max_age_days=sec_fundamentals.BVPS_CACHE_MAX_AGE_DAYS):
        if ticker == "A":
            return pd.DataFrame({"filed_date": [pd.Timestamp("2020-01-05")], "book_value_per_share": [20.0]})
        return pd.DataFrame(columns=["filed_date", "book_value_per_share"])

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_book_value_per_share_cached", fake_fetch_cached)
    panel = sec_fundamentals.build_book_value_per_share_panel_cached(["A", "B"], date_index)

    assert pd.isna(panel.loc["2020-01-01", "A"])
    assert panel.loc["2020-01-05", "A"] == pytest.approx(20.0)
    assert panel.loc["2020-01-10", "A"] == pytest.approx(20.0)
    assert panel["B"].isna().all()


def _fake_companyfacts_accruals(ni_rows, cfo_rows, assets_rows):
    return _FakeResponse({
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": ni_rows}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": cfo_rows}},
                "Assets": {"units": {"USD": assets_rows}},
            }
        }
    })


def test_fetch_point_in_time_accruals_computes_from_matched_10k_triple(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_accruals(
            ni_rows=[{"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"}],
            cfo_rows=[{"accn": "A1", "end": "2020-12-31", "val": 1_500_000, "form": "10-K", "filed": "2021-02-15"}],
            assets_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    accruals = sec_fundamentals.fetch_point_in_time_accruals("TEST")
    assert len(accruals) == 1
    # (1,000,000 - 1,500,000) / 10,000,000 = -0.05 -- negative accruals (cash > earnings, the "good" case)
    assert accruals.iloc[0]["accruals"] == pytest.approx(-0.05)
    assert accruals.iloc[0]["filed_date"] == pd.Timestamp("2021-02-15")


def test_fetch_point_in_time_accruals_ignores_non_10k_forms(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_accruals(
            ni_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 250_000, "form": "10-Q", "filed": "2020-11-01"}],
            cfo_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 300_000, "form": "10-Q", "filed": "2020-11-01"}],
            assets_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 5_000_000, "form": "10-Q", "filed": "2020-11-01"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    accruals = sec_fundamentals.fetch_point_in_time_accruals("TEST")
    assert accruals.empty


def test_fetch_point_in_time_accruals_wont_pair_mismatched_periods(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_accruals(
            ni_rows=[{"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"}],
            cfo_rows=[{"accn": "A2", "end": "2019-12-31", "val": 900_000, "form": "10-K", "filed": "2020-02-10"}],
            assets_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    accruals = sec_fundamentals.fetch_point_in_time_accruals("TEST")
    assert accruals.empty


def test_fetch_point_in_time_accruals_skips_zero_or_missing_assets(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_accruals(
            ni_rows=[{"accn": "A1", "end": "2020-12-31", "val": 1_000_000, "form": "10-K", "filed": "2021-02-15"}],
            cfo_rows=[{"accn": "A1", "end": "2020-12-31", "val": 1_500_000, "form": "10-K", "filed": "2021-02-15"}],
            assets_rows=[{"accn": "A1", "end": "2020-12-31", "val": 0, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    accruals = sec_fundamentals.fetch_point_in_time_accruals("TEST")
    assert accruals.empty


def test_fetch_point_in_time_accruals_unmapped_ticker_returns_empty_without_network_call(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {})
    calls = []
    monkeypatch.setattr(sec_fundamentals.requests, "get", lambda *a, **k: calls.append(1))
    accruals = sec_fundamentals.fetch_point_in_time_accruals("ATD")
    assert accruals.empty
    assert calls == []


def test_build_accruals_panel_forward_fills_without_look_ahead(monkeypatch):
    date_index = pd.date_range("2020-01-01", periods=10, freq="D")

    def fake_fetch(ticker):
        if ticker == "A":
            return pd.DataFrame({"filed_date": [pd.Timestamp("2020-01-05")], "accruals": [-0.05]})
        return pd.DataFrame(columns=["filed_date", "accruals"])

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_accruals", fake_fetch)
    panel = sec_fundamentals.build_accruals_panel(["A", "B"], date_index)

    assert pd.isna(panel.loc["2020-01-01", "A"])
    assert pd.isna(panel.loc["2020-01-04", "A"])
    assert panel.loc["2020-01-05", "A"] == pytest.approx(-0.05)
    assert panel.loc["2020-01-10", "A"] == pytest.approx(-0.05)
    assert panel["B"].isna().all()


def _fake_companyfacts_cash_profitability(cfo_rows, assets_rows):
    return _FakeResponse({
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": cfo_rows}},
                "Assets": {"units": {"USD": assets_rows}},
            }
        }
    })


def test_fetch_point_in_time_cash_profitability_computes_from_matched_10k_pair(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_cash_profitability(
            cfo_rows=[{"accn": "A1", "end": "2020-12-31", "val": 2_000_000, "form": "10-K", "filed": "2021-02-15"}],
            assets_rows=[{"accn": "A1", "end": "2020-12-31", "val": 10_000_000, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    result = sec_fundamentals.fetch_point_in_time_cash_profitability("TEST")
    assert len(result) == 1
    assert result.iloc[0]["cash_profitability"] == pytest.approx(0.2)  # 2,000,000 / 10,000,000
    assert result.iloc[0]["filed_date"] == pd.Timestamp("2021-02-15")


def test_fetch_point_in_time_cash_profitability_ignores_non_10k_forms(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_cash_profitability(
            cfo_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 500_000, "form": "10-Q", "filed": "2020-11-01"}],
            assets_rows=[{"accn": "Q1", "end": "2020-09-30", "val": 5_000_000, "form": "10-Q", "filed": "2020-11-01"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    result = sec_fundamentals.fetch_point_in_time_cash_profitability("TEST")
    assert result.empty


def test_fetch_point_in_time_cash_profitability_skips_zero_assets(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {"TEST": "0000000001"})

    def fake_get(url, headers=None, timeout=None):
        return _fake_companyfacts_cash_profitability(
            cfo_rows=[{"accn": "A1", "end": "2020-12-31", "val": 2_000_000, "form": "10-K", "filed": "2021-02-15"}],
            assets_rows=[{"accn": "A1", "end": "2020-12-31", "val": 0, "form": "10-K", "filed": "2021-02-15"}],
        )

    monkeypatch.setattr(sec_fundamentals.requests, "get", fake_get)
    result = sec_fundamentals.fetch_point_in_time_cash_profitability("TEST")
    assert result.empty


def test_fetch_point_in_time_cash_profitability_unmapped_ticker_returns_empty_without_network_call(monkeypatch):
    monkeypatch.setattr(sec_fundamentals, "fetch_cik_map", lambda: {})
    calls = []
    monkeypatch.setattr(sec_fundamentals.requests, "get", lambda *a, **k: calls.append(1))
    result = sec_fundamentals.fetch_point_in_time_cash_profitability("ATD")
    assert result.empty
    assert calls == []


def test_build_cash_profitability_panel_forward_fills_without_look_ahead(monkeypatch):
    date_index = pd.date_range("2020-01-01", periods=10, freq="D")

    def fake_fetch(ticker):
        if ticker == "A":
            return pd.DataFrame({"filed_date": [pd.Timestamp("2020-01-05")], "cash_profitability": [0.25]})
        return pd.DataFrame(columns=["filed_date", "cash_profitability"])

    monkeypatch.setattr(sec_fundamentals, "fetch_point_in_time_cash_profitability", fake_fetch)
    panel = sec_fundamentals.build_cash_profitability_panel(["A", "B"], date_index)

    assert pd.isna(panel.loc["2020-01-01", "A"])
    assert panel.loc["2020-01-05", "A"] == pytest.approx(0.25)
    assert panel.loc["2020-01-10", "A"] == pytest.approx(0.25)
    assert panel["B"].isna().all()
