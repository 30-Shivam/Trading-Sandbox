"""swingtrade.indicators must stay numerically identical to pandas_ta (dev-only
oracle; skipped if pandas_ta isn't installed)."""

import numpy as np
import pandas as pd
import pytest

from swingtrade import indicators as ind

ta = pytest.importorskip("pandas_ta")


def _ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=pd.bdate_range("2020-01-01", periods=n))
    high = close + rng.uniform(0.1, 2, n)
    low = close - rng.uniform(0.1, 2, n)
    return high, low, close


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("length", [5, 14, 21])
def test_matches_pandas_ta(seed, length):
    h, l, c = _ohlc(seed=seed)
    pd.testing.assert_series_equal(ind.rsi(c, length), ta.rsi(c, length=length), check_names=False)
    pd.testing.assert_series_equal(ind.atr(h, l, c, length), ta.atr(h, l, c, length=length), check_names=False)
    ours, theirs = ind.adx(h, l, c, length), ta.adx(h, l, c, length=length)
    pd.testing.assert_series_equal(ours[f"ADX_{length}"], theirs[f"ADX_{length}"], check_names=False)
    pd.testing.assert_series_equal(ours[f"DMP_{length}"], theirs[f"DMP_{length}"], check_names=False)


def test_too_short_is_all_nan():
    h, l, c = _ohlc(n=10)
    assert ind.rsi(c, 14).isna().all()
    assert ind.atr(h, l, c, 14).isna().all()
    assert ind.adx(h, l, c, 14)["ADX_14"].isna().all()
