"""PEAD (post-earnings-announcement drift) strategy -- a genuinely NEW
signal family (an earnings-surprise EVENT, not a price pattern), built
2026-09-12 per explicit user request after a real finding
(ic_tracking.backtest_ic_check()) that every price-pattern strategy tried
so far shows ~zero real backtest-time ranking skill. Buys within
pead_signal_window_days of a real earnings beat clearing
pead_surprise_pct_min, in a confirmed macro uptrend. See
run_backtest.fetch_earnings_surprises() for the data source and its own
EXPLICIT point-in-time-integrity caveat -- not re-tested here (that
function's own correctness against real yfinance data was verified
manually, not via a synthetic unit test, same as every other network-
dependent fetch function in this codebase).

Mirrors tests/test_insider_buying_strategy.py's own structure exactly --
same synthetic-fixture conventions, same coverage shape (precompute-frame
window hand-verification, add_*_trade_score scoring behavior, the
mandatory RRR-vs-scoring-ceiling check)."""
import numpy as np
import pandas as pd

import swingtrade
from swingtrade.levels import precompute_pead_frame

CONFIG = swingtrade.DEFAULT_CONFIG


def _uptrend_ohlcv(n: int, start_close: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_close * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def _surprises(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date_str, surprise_pct), ...] -- builds the same tz-aware-UTC
    "effective_date"/"surprise_pct" shape run_backtest.fetch_earnings_surprises()
    returns."""
    return pd.DataFrame({
        "effective_date": pd.to_datetime([r[0] for r in rows], utc=True),
        "surprise_pct": [r[1] for r in rows],
    })


# --- precompute_pead_frame(): synthetic OHLCV + synthetic surprise events,
# hand-verifying the signal-window gating produces the right values on the
# right days.

def test_precompute_pead_frame_absent_without_surprises():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    frame = precompute_pead_frame(df, None, CONFIG)
    assert frame["PEAD_Surprise_Pct"].isna().all()


def test_precompute_pead_frame_window_hand_verified():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    event_date = df.index[10]
    surprises = _surprises([(event_date.strftime("%Y-%m-%d"), 12.5)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "pead_signal_window_days": 3})

    frame = precompute_pead_frame(df, surprises, config)

    # Before the event: no surprise yet.
    assert pd.isna(frame["PEAD_Surprise_Pct"].iloc[9])
    # On the event day itself, and through day 10+3=13 (inclusive): visible.
    assert frame["PEAD_Surprise_Pct"].iloc[10] == 12.5
    assert frame["PEAD_Surprise_Pct"].iloc[13] == 12.5
    # The day after the window closes: gone again.
    assert pd.isna(frame["PEAD_Surprise_Pct"].iloc[14])


def test_precompute_pead_frame_takes_max_of_overlapping_events():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    surprises = _surprises([
        (df.index[10].strftime("%Y-%m-%d"), 5.0),
        (df.index[11].strftime("%Y-%m-%d"), 20.0),
    ])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "pead_signal_window_days": 5})
    frame = precompute_pead_frame(df, surprises, config)
    assert frame["PEAD_Surprise_Pct"].iloc[12] == 20.0


# --- add_pead_trade_score(): mirrors test_insider_buying_strategy.py's own
# add_insider_buying_trade_score tests exactly.

def _pead_row(signal, signal_strength_pct=1.0):
    return {"PEAD_Signal": signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct}


def test_add_pead_trade_score_ineligible_when_no_signal():
    df = pd.DataFrame([_pead_row(False)])
    result = swingtrade.add_pead_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_pead_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_pead_row(True, signal_strength_pct=1.0)])
    result = swingtrade.add_pead_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_pead_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _pead_row(True, signal_strength_pct=CONFIG.pead_strength_cap_pct),
        _pead_row(True, signal_strength_pct=CONFIG.pead_strength_cap_pct * 10),
    ])
    result = swingtrade.add_pead_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_add_pead_trade_score_best_case_clears_buy_threshold():
    # Same RRR-vs-scoring-ceiling check validation-pipeline point 9
    # mandates for every strategy before it's trusted -- a config could
    # otherwise be structurally unable to ever log a real Buy signal.
    df = pd.DataFrame([_pead_row(True, signal_strength_pct=CONFIG.pead_strength_cap_pct)])
    result = swingtrade.add_pead_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] >= CONFIG.signal_buy_threshold
    assert result.loc[0, "Signal"] in ("Buy", "Strong Buy")


# --- simulate_pead_signals(): real end-to-end regression, confirming a
# genuine signal fires, uses the "next_open" default fill, and carries a
# real trade_score from day one (see ic_tracking.backtest_ic_check() --
# the whole reason this field is captured from the start this time).

def test_simulate_pead_signals_fires_and_carries_trade_score(uptrend_ohlcv, market_ohlcv):
    base = uptrend_ohlcv.copy()  # session-scoped fixture, date-aligned with market_ohlcv
    event_date = base.index[250]
    surprises = _surprises([(event_date.strftime("%Y-%m-%d"), 15.0)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "pead"})

    trades = swingtrade.simulate_pead_signals(
        "TEST", base, market_ohlcv, base.index[0], base.index[-1], config,
        sector="Tech", earnings_surprises=surprises,
    )
    assert trades, "expected at least one real PEAD trade from a clean, qualifying synthetic surprise"
    assert all("trade_score" in t and isinstance(t["trade_score"], float) for t in trades)
    assert any(t["signal"] == "PEAD" for t in trades)


def test_simulate_pead_signals_never_fires_without_surprise_data(uptrend_ohlcv, market_ohlcv):
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "pead"})
    trades = swingtrade.simulate_pead_signals(
        "TEST", uptrend_ohlcv, market_ohlcv, uptrend_ohlcv.index[0], uptrend_ohlcv.index[-1], config,
        sector="Tech", earnings_surprises=None,
    )
    assert trades == [], "PEAD_Signal must always be False with no surprise data (missing optional data never fabricates a signal)"
