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
    # Trending regime (ADX=30) prefers breakout first -- only breakout fired.
    rows = {"breakout": _row(30.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "breakout"
    assert pick["Regime"] == "trending"


def test_select_regime_pick_single_strategy_fires_but_does_not_match_regime():
    # Trending regime (ADX=30) prefers breakout/ma_crossover -- only
    # squeeze_breakout fired, which isn't in the trending preference list.
    rows = {"squeeze_breakout": _row(30.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is None


def test_select_regime_pick_multiple_fire_preferred_one_among_them():
    # Trending regime prefers breakout over ma_crossover -- both fired,
    # breakout should win.
    rows = {"ma_crossover": _row(30.0), "breakout": _row(30.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "breakout"


def test_select_regime_pick_multiple_fire_preferred_one_absent():
    # Choppy regime (ADX=10) prefers only squeeze_breakout -- breakout and
    # ma_crossover both fired instead, neither is preferred here.
    rows = {"breakout": _row(10.0), "ma_crossover": _row(10.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is None


def test_select_regime_pick_choppy_regime_prefers_squeeze_breakout():
    rows = {"squeeze_breakout": _row(10.0), "breakout": _row(10.0)}
    pick = rs.select_regime_pick("TEST", rows)
    assert pick is not None
    assert pick["Source_Strategy"] == "squeeze_breakout"


def test_select_regime_pick_no_strategies_fired():
    assert rs.select_regime_pick("TEST", {}) is None


def test_select_regime_pick_no_adx_available_anywhere():
    rows = {"breakout": _row(None)}
    assert rs.select_regime_pick("TEST", rows) is None
