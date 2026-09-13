"""compute_volatility_regime_series()/summarize_by_volatility_regime()
(2026-09-13, improvements.txt item 134) -- a DIAGNOSTIC backtest-reporting
axis, not a new strategy filter/gate. Built after recognizing every
simulate_*_signals() function already hard-gates on market_uptrend_from_frame()
(SPY Close >= its own SMA_TREND), making a bull-vs-bear regime breakdown moot
here -- bear days are excluded by construction for every strategy already.
Volatility regime is NOT excluded by that gate, making it the informative
axis for "does this edge hold in calm vs turbulent markets" within the
population every strategy actually trades in.
"""
import numpy as np
import pandas as pd

import swingtrade


def _make_market_ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 400 * np.cumprod(1 + rng.normal(0.0005, 0.005, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
        "Volume": rng.integers(50_000_000, 100_000_000, n),
    }, index=dates)


def test_compute_volatility_regime_series_flags_a_real_spike_as_more_elevated_than_calm():
    n = 500
    df = _make_market_ohlcv(n)
    # Inject a real, sharp volatility spike over a real stretch, well past
    # the point where both the realized-vol window and the rolling-median
    # baseline window have enough history -- mirrors the SKEW regime
    # filter's own real "genuine spike vs stale baseline" test shape.
    spike_start, spike_end = 400, 420
    close = df["Close"].to_numpy().copy()
    rng = np.random.default_rng(99)
    close[spike_start:spike_end] *= np.cumprod(1 + rng.normal(0, 0.06, spike_end - spike_start))
    df["Close"] = close

    regime = swingtrade.compute_volatility_regime_series(df)
    # Skip the first few spike days while the 20d realized-vol window is
    # still partly filled with pre-spike (calm) returns; skip the first 30
    # days of the whole series so both windows have real history to work with.
    spike_dates = df.index[spike_start + 5:spike_end]
    calm_dates = df.index[30:spike_start]

    spike_share_elevated = (regime.loc[spike_dates] == "elevated").mean()
    calm_share_elevated = (regime.loc[calm_dates] == "elevated").mean()
    assert spike_share_elevated > calm_share_elevated, (
        "a real, sharp volatility spike should read 'elevated' meaningfully more often than an "
        "ordinary calm stretch, relative to its own trailing rolling median"
    )


def test_compute_volatility_regime_series_defaults_to_normal_before_any_history_exists():
    n = 15  # shorter than VOLATILITY_REALIZED_WINDOW_DAYS (20) -- realized_vol is NaN throughout
    df = _make_market_ohlcv(n, seed=2)
    regime = swingtrade.compute_volatility_regime_series(df)
    assert (regime == "normal").all(), (
        "with not even enough history for the realized-vol window itself, every day must default "
        "to 'normal' rather than fabricate 'elevated' from insufficient data"
    )
    assert set(regime.unique()) <= {"elevated", "normal"}


def test_summarize_by_volatility_regime_buckets_trades_correctly():
    df = _make_market_ohlcv(50, seed=3)
    # Force a known, hand-picked regime series rather than relying on the
    # real computation -- isolates the BUCKETING logic from the REGIME
    # CLASSIFICATION logic (already covered by the two tests above).
    regime_series = pd.Series("normal", index=df.index)
    regime_series.iloc[10:15] = "elevated"

    trades = [
        {"entry_date": df.index[11].date(), "status": "WIN", "pnl_pct": 5.0},
        {"entry_date": df.index[12].date(), "status": "LOSS", "pnl_pct": -3.0},
        {"entry_date": df.index[30].date(), "status": "WIN", "pnl_pct": 2.0},
    ]
    result = swingtrade.summarize_by_volatility_regime(trades, regime_series)
    assert result["elevated"]["trade_count"] == 2
    assert result["normal"]["trade_count"] == 1


def test_summarize_by_volatility_regime_defaults_missing_date_to_normal():
    regime_series = pd.Series(["elevated"], index=[pd.Timestamp("2024-01-02")])
    trades = [{"entry_date": pd.Timestamp("2030-01-01").date(), "status": "WIN", "pnl_pct": 1.0}]
    result = swingtrade.summarize_by_volatility_regime(trades, regime_series)
    assert result["normal"]["trade_count"] == 1
    assert result["elevated"]["trade_count"] == 0
