"""Sector relative-strength optional filter (backtest/Optuna-only), added
after item 68's rejection was corrected to a clean PASS on all three cuts
once re-checked with a proper multi-seed averaged holdout (item 69) --
tailwind (sector beating SPY) beats headwind by ~3-9x sharpe_like. Mirrors
breakout_relative_strength_min exactly, applied to breakout/squeeze_breakout/
ma_crossover's own new *_sector_relative_strength_min fields.

Deliberately BACKTEST/OPTUNA-ONLY: market_data.py's live scan path never
supplies sector_df, so every gate here must degrade to "don't exclude" when
Sector_Relative_Strength is missing/NaN -- the same convention every other
optional filter in this codebase already follows.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _synthetic_ohlcv(n: int, start_close: float, daily_drift_pct: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_close * np.cumprod(1 + daily_drift_pct / 100 + rng.normal(0, 0.001, n))
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_sector_relative_strength_computes_from_known_series():
    n = 260
    # Sector clearly outpacing the market -- sector drifts up 0.3%/day,
    # market flat, so Sector_Relative_Strength should be reliably positive.
    market_df = _synthetic_ohlcv(n, 100.0, 0.0, seed=1)
    sector_df = _synthetic_ohlcv(n, 100.0, 0.3, seed=2)
    ticker_df = _synthetic_ohlcv(n, 100.0, 0.1, seed=3)

    frame = swingtrade.levels.precompute_breakout_frame(
        ticker_df, CONFIG, market_df=market_df, sector_df=sector_df
    )

    assert "Sector_Relative_Strength" in frame.columns
    tail = frame["Sector_Relative_Strength"].dropna()
    assert len(tail) > 0
    assert (tail > 0).mean() > 0.9  # sector reliably beats a flat market


def test_sector_relative_strength_absent_without_sector_df():
    n = 260
    market_df = _synthetic_ohlcv(n, 100.0, 0.0, seed=1)
    ticker_df = _synthetic_ohlcv(n, 100.0, 0.1, seed=3)

    frame = swingtrade.levels.precompute_breakout_frame(ticker_df, CONFIG, market_df=market_df)

    assert "Sector_Relative_Strength" not in frame.columns


def test_compute_breakout_levels_returns_none_without_sector_df():
    n = 260
    market_df = _synthetic_ohlcv(n, 100.0, 0.0, seed=1)
    ticker_df = _synthetic_ohlcv(n, 100.0, 0.3, seed=3)  # trending up, clears macro-uptrend gate

    levels = swingtrade.levels.compute_breakout_levels(
        "TEST", ticker_df, CONFIG, market_df=market_df,
    )

    assert levels["Sector_Relative_Strength"] is None


def _breakout_row(sector_relative_strength):
    return {
        "Breakout_Signal": True, "RSI": 50.0, "Relative_Strength": 0.0,
        "Volume_Ratio": 1.0, "ADX": 30.0, "OBV_Zscore": 0.0, "Squeeze_Zscore": 0.0,
        "Sector_Relative_Strength": sector_relative_strength,
        "RRR": 2.0, "Distance_to_Buy_Pct": 0.0,
    }


def test_breakout_sector_filter_excludes_below_threshold():
    df = pd.DataFrame([_breakout_row(-0.20)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "breakout_sector_relative_strength_min": -0.10})
    result = swingtrade.add_breakout_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] == 0.0


def test_breakout_sector_filter_admits_above_threshold():
    df = pd.DataFrame([_breakout_row(0.05)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "breakout_sector_relative_strength_min": -0.10})
    result = swingtrade.add_breakout_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_breakout_sector_filter_none_never_excludes():
    df = pd.DataFrame([_breakout_row(None)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "breakout_sector_relative_strength_min": 0.50})
    result = swingtrade.add_breakout_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_breakout_sector_filter_is_noop_at_default():
    df = pd.DataFrame([_breakout_row(-99.0)])  # extreme, would fail almost any real threshold
    result = swingtrade.add_breakout_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def _squeeze_row(sector_relative_strength):
    return {
        "Squeeze_Signal": True, "RSI": 50.0, "Relative_Strength": 0.0,
        "Volume_Ratio": 1.0, "ADX": 30.0, "OBV_Zscore": 0.0,
        "Sector_Relative_Strength": sector_relative_strength,
        "Catalyst_Warning": False, "RRR": 2.0, "Signal_Strength_Pct": 1.0,
    }


def test_squeeze_breakout_sector_filter_excludes_below_threshold():
    df = pd.DataFrame([_squeeze_row(-0.20)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "squeeze_breakout_sector_relative_strength_min": -0.10})
    result = swingtrade.add_squeeze_breakout_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] == 0.0


def test_squeeze_breakout_sector_filter_is_noop_at_default():
    df = pd.DataFrame([_squeeze_row(-99.0)])
    result = swingtrade.add_squeeze_breakout_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def _ma_crossover_row(sector_relative_strength, yield_curve_spread=None, skew_regime_diff=None):
    return {
        "MA_Crossover_Signal": True, "Catalyst_Warning": False,
        "Sector_Relative_Strength": sector_relative_strength,
        "Yield_Curve_Spread": yield_curve_spread,
        "Skew_Regime_Diff": skew_regime_diff,
        "RRR": 2.0, "Signal_Strength_Pct": 1.0,
    }


def test_ma_crossover_sector_filter_excludes_below_threshold():
    df = pd.DataFrame([_ma_crossover_row(-0.20)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "ma_crossover_sector_relative_strength_min": -0.10})
    result = swingtrade.add_ma_crossover_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] == 0.0


def test_ma_crossover_sector_filter_is_noop_at_default():
    df = pd.DataFrame([_ma_crossover_row(-99.0)])
    result = swingtrade.add_ma_crossover_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


# --- ma_crossover_yield_curve_spread_max (2026-08-31, benchmark_macro_regime.py) ---

def test_ma_crossover_yield_curve_filter_excludes_above_threshold():
    # Real finding: ma_crossover's edge is stronger when the curve is
    # INVERTED (low/negative) -- the field is a CEILING, opposite polarity
    # from the sector filter's floor. A spread of 2.0 exceeding a max of
    # 1.0 must exclude.
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, yield_curve_spread=2.0)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "ma_crossover_yield_curve_spread_max": 1.0})
    result = swingtrade.add_ma_crossover_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] == 0.0


def test_ma_crossover_yield_curve_filter_is_noop_at_default():
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, yield_curve_spread=50.0)])
    result = swingtrade.add_ma_crossover_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_ma_crossover_yield_curve_filter_missing_value_never_excludes():
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, yield_curve_spread=None)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "ma_crossover_yield_curve_spread_max": -3.0})
    result = swingtrade.add_ma_crossover_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] > 0.0


# --- ma_crossover_skew_regime_min (2026-09-03, benchmark_skew_regime.py) ---

def test_ma_crossover_skew_regime_filter_excludes_below_threshold():
    # Real finding: ma_crossover's edge is stronger when ^SKEW is ELEVATED
    # relative to its own trailing-year median -- the field is a FLOOR, same
    # polarity as the sector filter, opposite polarity from yield curve's
    # ceiling. A diff of -5.0 below a min of 0.0 must exclude.
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, skew_regime_diff=-5.0)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "ma_crossover_skew_regime_min": 0.0})
    result = swingtrade.add_ma_crossover_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] == 0.0


def test_ma_crossover_skew_regime_filter_is_noop_at_default():
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, skew_regime_diff=-99.0)])
    result = swingtrade.add_ma_crossover_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_ma_crossover_skew_regime_filter_missing_value_never_excludes():
    df = pd.DataFrame([_ma_crossover_row(sector_relative_strength=None, skew_regime_diff=None)])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "ma_crossover_skew_regime_min": 30.0})
    result = swingtrade.add_ma_crossover_trade_score(df, config)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_skew_regime_diff_uses_rolling_not_expanding_window():
    # 2026-09-03 real bug this guards against: an EXPANDING-since-1990
    # median stays anchored to a stale baseline given ^SKEW's own real
    # secular drift, misclassifying almost everything recent as "elevated"
    # (a 5-ticker smoke test split 38 elevated vs 3 normal before this was
    # fixed -- see benchmark_skew_regime.py's own docstring). Construct a
    # synthetic SKEW series that drifts from 100 up to 200 over 800 days,
    # then holds flat at 200 -- an EXPANDING median would keep reading the
    # flat tail as wildly elevated relative to the early, much-lower
    # history; a ROLLING (bounded, ~1yr) median should instead read the
    # flat tail as roughly Skew_Regime_Diff ~= 0 (today's value matches its
    # own recent past), since the window no longer reaches the low, early
    # values at all.
    # Rise for 800 days, THEN a further 600 flat days before the ticker's own
    # 100-day sample window -- the sample must sit comfortably more than
    # SKEW_REGIME_ROLLING_WINDOW_DAYS (365) past the end of the rise, or its
    # own rolling window would still partly reach back into the still-rising
    # period and legitimately show a real (not spurious) positive diff.
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    drift = pd.Series(range(n), index=dates).clip(upper=800) / 800 * 100 + 100  # 100 -> 200 over 800 days, then flat
    skew_series = pd.Series(drift.values, index=dates)

    ticker_dates = dates[-100:]  # the flat tail, ~600 days after the rise ended
    ticker_df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000,
    }, index=ticker_dates)

    frame = swingtrade.levels.precompute_ma_crossover_frame(ticker_df, CONFIG, skew_regime=skew_series)
    tail_diff = frame["Skew_Regime_Diff"].dropna()
    assert len(tail_diff) > 0
    # Rolling: flat tail vs its own recent (also-flat) history -> near zero,
    # NOT the ~50-100 an expanding-since-day-1 median would have shown.
    assert tail_diff.abs().max() < 5.0


# --- compute_ma_crossover_levels() live-wiring (2026-08-31) -- the live
# single-ticker wrapper market_data.score_bundle_for_strategy() actually
# calls for every real "ma_crossover" scan, distinct from
# precompute_ma_crossover_frame()'s own already-tested yield_curve
# threading (test_yield_curve_multiprocessing_threading.py covers the
# backtest/Optuna side only) -- mirrors
# test_compute_breakout_levels_returns_none_without_sector_df()'s exact
# shape for the live-wrapper-level graceful-degradation check.

def test_compute_ma_crossover_levels_yield_curve_spread_populated_with_data():
    n = 260
    ticker_df = _synthetic_ohlcv(n, 100.0, 0.1, seed=3)
    yield_curve = pd.Series(1.5, index=ticker_df.index)

    levels = swingtrade.levels.compute_ma_crossover_levels(
        "TEST", ticker_df, CONFIG, yield_curve=yield_curve,
    )

    assert levels["Yield_Curve_Spread"] is not None


def test_compute_ma_crossover_levels_yield_curve_spread_none_without_data():
    n = 260
    ticker_df = _synthetic_ohlcv(n, 100.0, 0.1, seed=3)

    levels = swingtrade.levels.compute_ma_crossover_levels("TEST", ticker_df, CONFIG)

    assert levels["Yield_Curve_Spread"] is None
