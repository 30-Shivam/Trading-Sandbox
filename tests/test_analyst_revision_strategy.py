"""Analyst Revision Momentum strategy (2026-09-21) -- buys a ticker when
real analyst rating CHANGES (not reiterations) cluster net-positive within
a recent window, in a confirmed macro uptrend. A classic, well-documented
academic factor never tried in this codebase before -- see
run_backtest.fetch_analyst_revisions() for the real, confirmed-deep
(multi-year, not a 5-7-quarter snapshot) yfinance data source. Structural
mirror of test_insider_buying_strategy.py.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import run_backtest
import swingtrade
from swingtrade.levels import precompute_analyst_revision_frame

CONFIG = swingtrade.DEFAULT_CONFIG


def _uptrend_ohlcv(n: int, start_close: float, seed: int) -> pd.DataFrame:
    # Larger drift, smaller noise than test_insider_buying_strategy.py's own
    # helper -- that fixture's 0.001 drift / 0.01 stddev combo let Last_Close
    # dip below its own trailing SMA200 by chance right at the end of a
    # 260-day window for this test file's specific seeds. Matches the
    # steadier _trending_ohlcv() pattern test_value_rank_strategy.py/
    # test_quality_rank_strategy.py already use reliably.
    rng = np.random.default_rng(seed)
    returns = 0.001 + rng.normal(0, 0.002, n)
    close = start_close * np.cumprod(1 + returns)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def _revisions(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date_str, direction), ...] -- builds the same tz-aware-UTC
    "effective_date"/"direction" shape run_backtest.fetch_analyst_revisions()
    returns."""
    return pd.DataFrame({
        "effective_date": pd.to_datetime([r[0] for r in rows], utc=True),
        "direction": [r[1] for r in rows],
    })


# --- fetch_analyst_revisions(): real network call mocked, verifying the
# Action-filtering/direction-mapping/reporting-lag logic.

def _fake_upgrades_downgrades(rows):
    """rows: [(grade_date, action), ...] -- shape of yfinance's real
    Ticker.upgrades_downgrades (GradeDate-indexed)."""
    return pd.DataFrame(
        {"Action": [r[1] for r in rows]},
        index=pd.DatetimeIndex([r[0] for r in rows], name="GradeDate"),
    )


def test_fetch_analyst_revisions_keeps_only_real_rating_changes(monkeypatch):
    raw = _fake_upgrades_downgrades([
        ("2026-01-01", "up"), ("2026-01-05", "down"), ("2026-01-10", "main"),
        ("2026-01-15", "reit"), ("2026-01-20", "init"),
    ])
    fake_ticker = MagicMock()
    fake_ticker.upgrades_downgrades = raw
    monkeypatch.setattr(run_backtest.yf, "Ticker", lambda t: fake_ticker)

    revisions = run_backtest.fetch_analyst_revisions("TEST")
    assert len(revisions) == 2  # only the real "up"/"down" rows survive
    assert set(revisions["direction"]) == {1.0, -1.0}


def test_fetch_analyst_revisions_applies_reporting_lag(monkeypatch):
    raw = _fake_upgrades_downgrades([("2026-01-01 11:30:00", "up")])
    fake_ticker = MagicMock()
    fake_ticker.upgrades_downgrades = raw
    monkeypatch.setattr(run_backtest.yf, "Ticker", lambda t: fake_ticker)

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "analyst_revision_reporting_lag_days": 1})
    revisions = run_backtest.fetch_analyst_revisions("TEST", config)
    # .values (used to build the returned DataFrame, same convention
    # fetch_insider_purchases() already established) strips the tz LABEL
    # but the underlying instant is already correctly UTC-converted --
    # compare via tz_localize rather than requiring an exact tz-aware match.
    got = pd.Timestamp(revisions.iloc[0]["effective_date"])
    assert got.tz_localize("UTC") == pd.Timestamp("2026-01-02 11:30:00", tz="UTC")


def test_fetch_analyst_revisions_no_data_returns_empty(monkeypatch):
    fake_ticker = MagicMock()
    fake_ticker.upgrades_downgrades = None
    monkeypatch.setattr(run_backtest.yf, "Ticker", lambda t: fake_ticker)
    revisions = run_backtest.fetch_analyst_revisions("TEST")
    assert revisions.empty
    assert list(revisions.columns) == ["effective_date", "direction"]


def test_fetch_analyst_revisions_network_failure_degrades_to_empty(monkeypatch):
    def raise_error(t):
        raise ConnectionError("network down")
    monkeypatch.setattr(run_backtest.yf, "Ticker", raise_error)
    revisions = run_backtest.fetch_analyst_revisions("TEST")
    assert revisions.empty


# --- precompute_analyst_revision_frame(): synthetic OHLCV + synthetic
# revision events, hand-verifying the lookback-window net-upgrade sum.

def test_precompute_analyst_revision_frame_absent_without_revisions():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    frame = precompute_analyst_revision_frame(df, None, CONFIG)
    assert (frame["Net_Upgrades"] == 0.0).all()


def test_precompute_analyst_revision_frame_sums_within_lookback_window():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    # 2 upgrades, 1 downgrade -> net +1, all within the default 90-day lookback.
    revisions = _revisions([
        ("2025-01-10", 1.0), ("2025-01-15", 1.0), ("2025-01-20", -1.0),
    ])
    frame = precompute_analyst_revision_frame(df, revisions, CONFIG)
    assert frame.loc["2025-01-25", "Net_Upgrades"] == pytest.approx(1.0)
    # Before ANY event, net upgrades must read 0 -- never a future value.
    assert frame.loc["2025-01-05", "Net_Upgrades"] == pytest.approx(0.0)


def test_precompute_analyst_revision_frame_drops_out_of_window_events():
    n = 200
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    revisions = _revisions([("2025-01-10", 1.0)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "analyst_revision_lookback_days": 30})
    frame = precompute_analyst_revision_frame(df, revisions, config)
    # Still within the 30-day window.
    assert frame.loc["2025-02-05", "Net_Upgrades"] == pytest.approx(1.0)
    # Past the 30-day window -- the event has aged out.
    assert frame.loc["2025-03-01", "Net_Upgrades"] == pytest.approx(0.0)


def test_analyst_revision_levels_from_frame_fires_when_net_upgrades_clears_threshold():
    n = 260
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    revisions = _revisions([
        ("2025-08-01", 1.0), ("2025-08-05", 1.0), ("2025-08-10", 1.0),
    ])
    frame = precompute_analyst_revision_frame(df, revisions, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.analyst_revision_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Net_Upgrades"] == pytest.approx(3.0)
    assert levels["Analyst_Revision_Signal"] is True
    assert levels["Signal_Strength_Pct"] == pytest.approx(1.0)  # 3.0 - min(2.0)
    assert levels["Buy_Price"] == levels["Last_Close"]
    assert levels["Distance_to_Buy_Pct"] == 0.0


def test_analyst_revision_levels_from_frame_no_signal_below_threshold():
    n = 260
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    revisions = _revisions([("2025-08-01", 1.0)])  # net +1, below the default min of 2.0
    frame = precompute_analyst_revision_frame(df, revisions, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.analyst_revision_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Analyst_Revision_Signal"] is False
    assert levels["Signal_Strength_Pct"] == 0.0


def test_analyst_revision_levels_from_frame_net_downgrades_never_fires():
    n = 260
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    revisions = _revisions([("2025-08-01", -1.0), ("2025-08-05", -1.0)])
    frame = precompute_analyst_revision_frame(df, revisions, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.analyst_revision_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Net_Upgrades"] == pytest.approx(-2.0)
    assert levels["Analyst_Revision_Signal"] is False


def test_analyst_revision_levels_from_frame_raises_on_macro_downtrend():
    n = 260
    rng = np.random.default_rng(1)
    close = 100.0 * np.cumprod(1 + rng.normal(-0.003, 0.01, n))
    df = pd.DataFrame({
        "Open": close, "High": close + 0.2, "Low": close - 0.2, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    revisions = _revisions([("2025-08-01", 1.0), ("2025-08-05", 1.0), ("2025-08-10", 1.0)])
    frame = precompute_analyst_revision_frame(df, revisions, CONFIG)
    as_of = frame.index[-1]
    try:
        swingtrade.levels.analyst_revision_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)
        assert False, "expected RuntimeError for macro downtrend"
    except RuntimeError as exc:
        assert "macro downtrend" in str(exc)


def _analyst_row(signal, signal_strength_pct=1.0):
    return {"Analyst_Revision_Signal": signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct}


def test_add_analyst_revision_trade_score_ineligible_when_no_signal():
    df = pd.DataFrame([_analyst_row(False)])
    result = swingtrade.add_analyst_revision_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_analyst_revision_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_analyst_row(True, signal_strength_pct=2.0)])
    result = swingtrade.add_analyst_revision_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_analyst_revision_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _analyst_row(True, signal_strength_pct=CONFIG.analyst_revision_strength_cap),
        _analyst_row(True, signal_strength_pct=CONFIG.analyst_revision_strength_cap * 10),
    ])
    result = swingtrade.add_analyst_revision_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_run_backtest_dispatches_analyst_revision_and_resolves_own_ticker():
    n = 260
    market_ohlcv = _uptrend_ohlcv(n, 400.0, seed=99)
    ticker_data = {
        "UPGRADED": _uptrend_ohlcv(n, 100.0, seed=1),
        "UNCOVERED": _uptrend_ohlcv(n, 100.0, seed=2),
    }
    analyst_revision_data = {
        "UPGRADED": _revisions([("2025-06-01", 1.0), ("2025-06-05", 1.0), ("2025-06-10", 1.0)]),
        # UNCOVERED has no entry at all -- should never fire.
    }

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "analyst_revision"})
    window_start = ticker_data["UPGRADED"].index[210]
    window_end = ticker_data["UPGRADED"].index[-1]

    trades = swingtrade.run_backtest(
        ticker_data, market_ohlcv, window_start, window_end, config,
        strategy="analyst_revision", analyst_revision_data=analyst_revision_data,
    )
    for trade in trades:
        assert trade["signal"] == "Analyst_Revision"
        if trade["status"] != "OPEN":
            assert "pnl_pct" in trade
    assert all(t["ticker"] == "UPGRADED" for t in trades)

    try:
        swingtrade.run_backtest(ticker_data, market_ohlcv, window_start, window_end, config, strategy="nonexistent")
        assert False, "expected ValueError for unknown strategy"
    except ValueError as exc:
        assert "analyst_revision" in str(exc)


def test_run_random_backtest_dispatches_analyst_revision():
    n = 260
    market_ohlcv = _uptrend_ohlcv(n, 400.0, seed=99)
    ticker_data = {"UPGRADED": _uptrend_ohlcv(n, 100.0, seed=1)}
    window_start = ticker_data["UPGRADED"].index[210]
    window_end = ticker_data["UPGRADED"].index[-1]

    import random
    rng = random.Random(42)
    trades = swingtrade.run_random_backtest(
        ticker_data, market_ohlcv, window_start, window_end,
        real_trade_counts={"UPGRADED": 3}, rng=rng, config=CONFIG, strategy="analyst_revision",
    )
    for trade in trades:
        assert trade["signal"] == "Random_Analyst_Revision"
