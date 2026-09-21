"""regime_switcher.py -- EXPLICITLY prospective-only, never backtested (see
its own module docstring for why). These tests cover only the mechanical
correctness of classify_regime()/select_regime_pick() -- whether the
hypothesis itself has real predictive value can only be judged by real
settled trades over real calendar time, never by a unit test.
"""
import math

import numpy as np
import pandas as pd

import regime_switcher as rs
import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def test_classify_regime_trending_at_and_above_threshold():
    assert rs.classify_regime(25.0) == "trending"
    assert rs.classify_regime(40.0) == "trending"


def test_classify_regime_choppy_below_threshold():
    assert rs.classify_regime(24.9) == "choppy"
    assert rs.classify_regime(5.0) == "choppy"


def test_classify_regime_none_when_adx_unavailable():
    assert rs.classify_regime(None) is None
    assert rs.classify_regime(float("nan")) is None


def _row(adx, signal="Buy"):
    return {"Ticker": "TEST", "Signal": signal, "ADX": adx, "Trade_Score": 70.0}


def test_select_regime_pick_single_strategy_fires_and_matches_regime():
    # Trending regime (ADX=30) prefers ma_crossover -- only ma_crossover fired.
    rows = {"ma_crossover": _row(30.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "ma_crossover"
    assert pick["Regime"] == "trending"


def test_select_regime_pick_single_strategy_fires_but_does_not_match_regime():
    # Trending regime (ADX=30) prefers ma_crossover -- only pairs fired,
    # which isn't in the trending preference list.
    rows = {"pairs": _row(30.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is None


def test_select_regime_pick_choppy_regime_prefers_pairs():
    rows = {"pairs": _row(10.0), "ma_crossover": _row(10.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "pairs"
    assert pick["Regime"] == "choppy"


def test_select_regime_pick_preferred_one_absent():
    # Choppy regime (ADX=10) prefers only pairs -- ma_crossover fired
    # instead, which isn't preferred for this regime.
    rows = {"ma_crossover": _row(10.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is None


def test_select_regime_pick_retired_strategies_never_match():
    # breakout/squeeze_breakout were both retired from live scanning and
    # dropped from REGIME_STRATEGY_PREFERENCE (2026-08-31 fix) -- even if a
    # stale row somehow showed up under one of these keys, it must never be
    # picked, in either regime.
    assert rs.select_regime_pick("TEST", {"breakout": _row(30.0)}) is None
    assert rs.select_regime_pick("TEST", {"squeeze_breakout": _row(10.0)}) is None


def test_select_regime_pick_no_strategies_fired():
    assert rs.select_regime_pick("TEST", {}) is None


def test_select_regime_pick_no_adx_available_anywhere():
    rows = {"ma_crossover": _row(None)}
    assert rs.select_regime_pick("TEST", rows) is None


def _correlated_ohlcv(n: int, start_close: float, shared_returns: np.ndarray, noise_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idiosyncratic = rng.normal(0, noise_scale, n)
    close = start_close * np.cumprod(1 + shared_returns + idiosyncratic)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_pairs_levels_from_frame_real_dict_includes_adx_and_reaches_choppy_pick():
    """Real regression test for the 2026-09-21 bug: pairs_levels_from_frame()
    never exposed ADX in its returned dict (even though precompute_pairs_frame()
    -- built on precompute_breakout_frame() -- already computes the column
    internally), so select_regime_pick() could never classify a regime from
    a pairs-only candidate. Every test above this one uses a hand-typed
    `_row()` fixture that ALREADY includes "ADX" -- exactly the kind of gap
    test_strategy_dispatch_parity.py's own docstring warns about (a synthetic
    fixture can't catch a real dict failing to carry a field). This test
    uses the REAL swingtrade.levels.pairs_levels_from_frame() output instead,
    so it would have caught the real incident (regime_switcher silently
    logging zero signals for 18 real days) before it happened."""
    n = 260
    shared = 0.001 + np.random.default_rng(0).normal(0, 0.01, n)  # slight upward drift so the
                                           # fixture clears the macro-uptrend gate every
                                           # strategy in this codebase shares, same as the
                                           # value_rank/quality_rank tests' own _trending_ohlcv()
    ticker_df = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=1)
    peer_df = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=2)
    peer_prices = pd.DataFrame({"PEER": peer_df["Close"]})

    frame = swingtrade.levels.precompute_pairs_frame(ticker_df, peer_prices, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.pairs_levels_from_frame("TEST", frame, as_of, CONFIG)

    assert "ADX" in levels
    assert levels["ADX"] is not None  # the real bug: this used to be absent from the dict entirely

    # A real pairs-only row must now be able to reach a choppy pick end to
    # end, using ONLY its own ADX -- no ma_crossover row needed to supply it.
    rows = {"pairs": {**levels, "Signal": "Buy", "Trade_Score": 70.0, "ADX": 10.0}}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "pairs"
    assert pick["Regime"] == "choppy"
