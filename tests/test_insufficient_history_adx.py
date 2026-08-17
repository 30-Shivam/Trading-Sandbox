"""Regression test for a real bug: ta.adx() returns None (not a NaN-filled
DataFrame) when given too little history to compute even one window --
crashed precompute_breakout_frame() with `TypeError: 'NoneType' object is
not subscriptable` for any ticker with genuinely tiny row counts (confirmed
for EA, a real S&P 500 name that went take-private in 2025 and left only a
handful of trailing bars under its old ticker via yfinance -- see
improvements.txt). Hit optimize.py/benchmark_random_entry.py directly,
which -- unlike market_data.score_bundle_for_strategy()'s per-ticker
try/except -- had no protection against it.
"""
import pandas as pd

import swingtrade


def test_precompute_breakout_frame_survives_too_little_history_for_adx():
    df = pd.DataFrame({
        "Open": [100.0] * 6, "High": [101.0] * 6, "Low": [99.0] * 6,
        "Close": [100.0] * 6, "Volume": [1_000_000] * 6,
    }, index=pd.date_range("2026-08-01", periods=6, freq="D"))

    frame = swingtrade.levels.precompute_breakout_frame(df, swingtrade.DEFAULT_CONFIG)

    assert frame["ADX"].isna().all()


def test_precompute_breakout_frame_normal_history_still_computes_adx():
    # Non-regression: a real, sufficiently long series must still get a
    # genuine (non-NaN) ADX reading -- the fix must not silently disable
    # ADX for every ticker, only the genuinely-too-short ones.
    import numpy as np
    rng = np.random.default_rng(1)
    n = 260
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    df = pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))

    frame = swingtrade.levels.precompute_breakout_frame(df, swingtrade.DEFAULT_CONFIG)

    assert frame["ADX"].notna().any()
