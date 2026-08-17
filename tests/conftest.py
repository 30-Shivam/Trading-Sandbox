"""Shared pytest fixtures. Deterministic, offline (no network/Mongo) OHLCV
fixture used by tests that need REAL, schema-complete levels rows without
depending on yfinance being reachable from CI -- reuses the same "steady
uptrend, deterministic RNG" approach this project's own ad-hoc strategy
unit tests (e.g. ma_crossover's) have used throughout, so every
compute_*_levels() function gets a long enough, liquid enough, macro-
uptrending history to return a real dict rather than raising
"insufficient history"/"macro downtrend"/"illiquid" -- whether or not any
particular strategy's own trigger condition fires on this data is
irrelevant to what these tests check.
"""
import numpy as np
import pandas as pd
import pytest


def _make_uptrend_ohlcv(seed: int = 7, n_days: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes = [200.0]
    for _ in range(n_days - 1):
        closes.append(closes[-1] * (1 + rng.normal(0.0012, 0.006)))
    closes = np.array(closes)
    highs = closes * (1 + np.abs(rng.normal(0.003, 0.0015, n_days)))
    lows = closes * (1 - np.abs(rng.normal(0.003, 0.0015, n_days)))
    opens = closes * (1 + rng.normal(0, 0.0015, n_days))
    volumes = rng.integers(3_000_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=dates,
    )


@pytest.fixture(scope="session")
def uptrend_ohlcv() -> pd.DataFrame:
    """420 trading days of a steady, deterministic synthetic uptrend --
    enough for SMA200/ADX/OBV/Squeeze warmup, comfortably liquid, no
    macro-downtrend exclusion. Session-scoped since it's pure, expensive-
    ish to regenerate, and every consumer only reads it."""
    return _make_uptrend_ohlcv()


@pytest.fixture(scope="session")
def market_ohlcv() -> pd.DataFrame:
    """A second, independent synthetic uptrend standing in for SPY/the
    market-uptrend proxy -- same shape, different seed."""
    return _make_uptrend_ohlcv(seed=99)


@pytest.fixture(scope="session")
def sector_ohlcv() -> pd.DataFrame:
    """A third, independent synthetic uptrend standing in for a sector ETF
    (e.g. XLK) -- same shape, different seed, for the backtest/Optuna-only
    Sector_Relative_Strength filter (improvements.txt items 68/70)."""
    return _make_uptrend_ohlcv(seed=55)
