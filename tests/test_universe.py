import json

import numpy as np
import pandas as pd

import universe
import watchlist


def _frame(price, volume, n=80):
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({"Close": [price] * n, "Volume": [volume] * n}, index=idx)


def test_normalize_ticker_share_class():
    assert universe.normalize_ticker("brk.b") == "BRK-B"


def test_parse_constituents_normalizes_sector_and_skips_junk():
    table = pd.DataFrame({
        "Symbol": ["AAPL", "BF.B", "bad ticker!"],
        "GICS Sector": ["Information Technology", "Consumer Staples", "Health Care"],
    })
    rows = universe.parse_constituents(table, "large")
    assert [r["ticker"] for r in rows] == ["AAPL", "BF-B"]
    assert rows[0]["sector"] == "Technology" and rows[0]["cap"] == "large"


def test_liquidity_verdicts():
    f = _frame(50, 1_000_000)
    assert universe.liquidity_verdict(f["Close"], f["Volume"]) == (True, "ok")
    f = _frame(2, 10_000_000)
    assert universe.liquidity_verdict(f["Close"], f["Volume"]) == (False, "low_price")
    f = _frame(50, 100)
    assert universe.liquidity_verdict(f["Close"], f["Volume"]) == (False, "low_dollar_volume")
    f = _frame(np.nan, np.nan)
    assert universe.liquidity_verdict(f["Close"], f["Volume"]) == (False, "no_data")


def test_build_universe_records_dropped_with_reason(tmp_path):
    members = [
        {"ticker": "GOOD", "sector": "Technology", "cap": "large"},
        {"ticker": "PENNY", "sector": "Energy", "cap": "small"},
        {"ticker": "GONE", "sector": "Utilities", "cap": "mid"},
    ]

    def fake(tickers):
        return pd.concat({"GOOD": _frame(100, 1_000_000), "PENNY": _frame(1, 9_999_999)}, axis=1)

    uni = universe.build_universe(members, downloader=fake)
    assert [m["ticker"] for m in uni["watchlist"]] == ["GOOD"]
    assert {m["ticker"]: m["reason"] for m in uni["dropped"]} == {"PENNY": "low_price", "GONE": "no_data"}


def test_download_failure_drops_chunk_with_reason_not_silently():
    def boom(tickers):
        raise RuntimeError("offline")

    uni = universe.build_universe([{"ticker": "AAA", "sector": "", "cap": "large"}], downloader=boom)
    assert uni["watchlist"] == [] and "download_error" in uni["dropped"][0]["reason"]


def test_universe_file_readable_by_existing_watchlist_parsers(tmp_path):
    path = tmp_path / "universe.json"
    universe.write_universe({"watchlist": [
        {"ticker": "AAA", "sector": "Technology", "cap": "large"},
        {"ticker": "BBB", "sector": "Energy", "cap": "mid"},
    ]}, path)
    assert watchlist.read_tickers(path) == ["AAA", "BBB"]
    assert watchlist.read_ticker_sectors(path) == {"AAA": "Technology", "BBB": "Energy"}
    assert [m["ticker"] for m in universe.load_universe(path, {"mid"})] == ["BBB"]


def test_coverage_summary_flags_degraded_and_groups_reasons():
    skipped = [("A", "possibly delisted"), ("B", "Read timed out"), ("C", "HTTP 429")] + [("D", "x")] * 7
    s = universe.coverage_summary(100, skipped)
    assert s["analyzed"] == 90 and not s["degraded"]
    s = universe.coverage_summary(100, skipped + [("E", "x")])
    assert s["degraded"]
    assert s["reasons"] == {"no_data_or_delisted": 1, "timeout": 1, "rate_limited": 1, "other": 8}
    assert universe.format_coverage_line(s).startswith("COVERAGE 89/100 (89.0%) DEGRADED")


def test_coverage_summary_empty_request():
    assert universe.coverage_summary(0, [])["degraded"]


def test_daily_run_extracts_coverage_line():
    import daily_run
    out = "noise\nCOVERAGE 380/407 (93.4%) skipped: other=27\nmore"
    assert daily_run.extract_coverage(out) == "COVERAGE 380/407 (93.4%) skipped: other=27"
    assert daily_run.extract_coverage("nothing here") is None
