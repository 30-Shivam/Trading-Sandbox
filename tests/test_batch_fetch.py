import numpy as np
import pandas as pd

import market_data


def _ohlcv(n=30, close=10.0):
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Adj Close": close, "Volume": 1000}, index=idx)


def _multi(frames):
    return pd.concat(frames, axis=1)  # columns (ticker, field), like group_by="ticker"


def test_batch_returns_frames_and_skips_missing(monkeypatch):
    good = _ohlcv()
    bad = _ohlcv() * np.nan
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: _multi({"AAA": good, "BBB": bad}))
    frames, skipped = market_data.fetch_ohlcv_batch(("AAA", "BBB"))
    assert list(frames) == ["AAA"] and [s[0] for s in skipped] == ["BBB"]


def test_batch_chunk_failure_skips_only_that_chunk(monkeypatch):
    monkeypatch.setattr(market_data, "BATCH_DOWNLOAD_CHUNK", 2)
    calls = []

    def fake(chunk, **k):
        calls.append(list(chunk))
        if "CCC" in chunk:
            raise RuntimeError("boom")
        return _multi({t: _ohlcv() for t in chunk})

    monkeypatch.setattr(market_data.yf, "download", fake)
    frames, skipped = market_data.fetch_ohlcv_batch(("AAA", "BBB", "CCC", "DDD"))
    assert sorted(frames) == ["AAA", "BBB"]
    assert sorted(s[0] for s in skipped) == ["CCC", "DDD"] and "boom" in skipped[0][1]


def test_batch_single_ticker_chunk_is_flat_frame(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: _ohlcv())
    frames, skipped = market_data.fetch_ohlcv_batch(("AAA",))
    assert list(frames) == ["AAA"] and not skipped


def test_extras_failure_does_not_drop_a_ticker(monkeypatch):
    def boom(ticker, now):
        raise RuntimeError("yahoo 429")

    monkeypatch.setattr(market_data, "_fetch_extras", boom)
    assert market_data._safe_extras("AAA", pd.Timestamp.now(tz="UTC")) == (None, "")


def test_with_retry_recovers_then_gives_up(monkeypatch):
    monkeypatch.setattr(market_data.time, "sleep", lambda s: None)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("rate limited")
        return "ok"

    assert market_data._with_retry(flaky) == "ok" and state["n"] == 3

    def dead():
        raise RuntimeError("down")

    import pytest
    with pytest.raises(RuntimeError):
        market_data._with_retry(dead)
