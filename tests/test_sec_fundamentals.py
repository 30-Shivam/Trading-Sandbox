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
import pandas as pd
import pytest

import sec_fundamentals


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
