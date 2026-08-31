"""regime_switcher.py -- EXPLICITLY prospective-only, never backtested (see
its own module docstring for why). These tests cover only the mechanical
correctness of classify_regime()/select_regime_pick() -- whether the
hypothesis itself has real predictive value can only be judged by real
settled trades over real calendar time, never by a unit test.
"""
import math

import regime_switcher as rs


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
