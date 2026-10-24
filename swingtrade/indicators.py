"""Wilder-smoothed RSI / ATR / ADX, numerically identical to pandas_ta 0.4.71b0
(the no-TA-Lib code path) so the library dependency can be dropped without
changing any backtest or live signal. Pure pandas/numpy; no I/O.

Too-short inputs return all-NaN output (pandas_ta returned None), which every
caller already treats as "indicator unavailable".
"""

import numpy as np
import pandas as pd


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average: EMA with alpha = 1/length, seeded from the first value."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _nan_like(series: pd.Series) -> pd.Series:
    return pd.Series(np.nan, index=series.index, dtype=float)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if len(close) < length + 1:
        return _nan_like(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0)
    # NaN diff on row 0 must stay NaN (clip preserves it).
    gain_avg = _rma(gain, length)
    loss_avg = _rma(loss, length)
    return 100 * gain_avg / (gain_avg + loss_avg.abs())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, high - prev_close, prev_close - low], axis=1)
    return ranges.abs().max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, prenan: bool = False) -> pd.Series:
    if len(close) < length + 1:
        return _nan_like(close)
    tr = true_range(high, low, close)
    if prenan:
        tr.iloc[:1] = np.nan
    if tr.isna().all():
        return _nan_like(close)
    # pandas_ta seeds Wilder smoothing with the SMA of the first `length` TRs.
    seed = tr.iloc[:length].mean()
    tr.iloc[: length - 1] = np.nan
    tr.iloc[length - 1] = seed
    out = _rma(tr, length)
    return _nan_like(close) if out.isna().all() else out


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Returns a frame with ADX_{length}, DMP_{length}, DMN_{length} columns."""
    cols = [f"ADX_{length}", f"DMP_{length}", f"DMN_{length}"]
    empty = pd.DataFrame({c: _nan_like(close) for c in cols}, index=close.index)
    if len(close) < length:
        return empty
    atr_ = atr(high, low, close, length=length, prenan=True)
    if atr_.isna().all():
        return empty
    k = 100 / atr_
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = (((up > dn) & (up > 0)) * up).replace(0, 0.0)
    neg = (((dn > up) & (dn > 0)) * dn).replace(0, 0.0)
    dmp = k * _rma(pos, length)
    dmn = k * _rma(neg, length)
    dx = 100 * (dmp - dmn).abs() / (dmp + dmn)
    adx_ = _rma(dx, length)
    return pd.DataFrame({cols[0]: adx_, cols[1]: dmp, cols[2]: dmn}, index=close.index)
