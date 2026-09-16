"""Cross-sectional LOW-VOLATILITY RANK strategy -- ranks every ticker in the
watchlist by trailing REALIZED VOLATILITY and buys the bottom-vol decile
(Ang/Hodges/Xing/Zhang low-volatility anomaly). Exact structural mirror of
test_momentum_rank_strategy.py, adapted for the inverted convention: LOW
volatility maps to a HIGH percentile, so ">= lowvol_top_percentile_min
fires" reads the same as momentum_rank's own gate despite ranking the
opposite underlying quantity.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _trending_ohlcv(n: int, start_close: float, daily_drift: float, seed: int, daily_noise: float = 0.002) -> pd.DataFrame:
    """Same synthetic-ticker generator as test_momentum_rank_strategy.py's
    own helper, with an added `daily_noise` knob -- this strategy's ranking
    quantity IS return noise (realized volatility), so tests need to vary
    it directly, unlike momentum_rank's fixed-noise helper."""
    rng = np.random.default_rng(seed)
    returns = daily_drift + rng.normal(0, daily_noise, n)
    close = start_close * np.cumprod(1 + returns)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_compute_lowvol_rank_frame_ranks_least_volatile_at_100th_percentile():
    n = 260
    # 20 tickers, each with a distinctly different, deterministic daily
    # return MAGNITUDE (alternating sign so price still moves, but the
    # realized stdev is exactly controlled) -- ticker "T00" has the
    # smallest-magnitude daily moves (least volatile, should rank #1/highest
    # percentile), "T19" the largest (most volatile, lowest percentile).
    n_tickers = 20
    panel = {}
    for i in range(n_tickers):
        magnitude = 0.001 * (i + 1)
        alternating = np.array([1 if d % 2 == 0 else -1 for d in range(n)])
        returns = magnitude * alternating
        panel[f"T{i:02d}"] = 100.0 * np.cumprod(1 + returns)
    panel = pd.DataFrame(panel, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    rank_frame = swingtrade.compute_lowvol_rank_frame(panel, CONFIG.lowvol_lookback_days)

    last_row = rank_frame.iloc[-1]
    assert last_row.notna().all()
    # Smallest daily-move magnitude -> lowest realized vol -> 100th percentile
    # (same exact floor/ceiling shape as compute_momentum_rank_frame()'s own
    # test: rank(pct=True) of 20 unique values spans 1/20..20/20, so the
    # "best" ticker reads exactly 100.0 and the "worst" reads 100/20=5.0).
    assert last_row["T00"] == 100.0
    # Largest daily-move magnitude -> highest realized vol -> 1/20th percentile.
    assert last_row["T19"] == 5.0
    # Monotonic: higher magnitude index -> lower (or equal) percentile.
    values = [last_row[f"T{i:02d}"] for i in range(n_tickers)]
    assert values == sorted(values, reverse=True)


def test_compute_lowvol_rank_frame_early_rows_are_nan_before_lookback_fills():
    n = 260
    panel = pd.DataFrame({
        "A": _trending_ohlcv(n, 100.0, 0.0005, seed=1)["Close"],
        "B": _trending_ohlcv(n, 100.0, 0.001, seed=2)["Close"],
    })
    rank_frame = swingtrade.compute_lowvol_rank_frame(panel, lookback_days=63)
    # Before 63 trading days of RETURNS (64 prices), rolling(63).std() is NaN.
    assert rank_frame.iloc[0].isna().all()
    assert rank_frame.iloc[61].isna().all()
    assert rank_frame.iloc[-1].notna().all()


def test_lowvol_levels_from_frame_fires_when_percentile_clears_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    # Hand-construct the rank column directly (bypassing compute_lowvol_rank_frame
    # entirely) so this test isolates lowvol_levels_from_frame()'s own gating
    # logic from the ranking math already covered above.
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 95.0  # bottom-vol-decile on the final (as_of) day only

    frame = swingtrade.levels.precompute_lowvol_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.lowvol_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["LowVol_Percentile"] == 95.0
    assert levels["LowVol_Signal"] is True
    # 95 - lowvol_top_percentile_min (90.0) = 5.0
    assert levels["Signal_Strength_Pct"] == 5.0
    assert levels["Buy_Price"] == levels["Last_Close"]
    assert levels["Distance_to_Buy_Pct"] == 0.0


def test_lowvol_levels_from_frame_no_signal_just_below_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 89.9  # just under lowvol_top_percentile_min (90.0)

    frame = swingtrade.levels.precompute_lowvol_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.lowvol_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["LowVol_Signal"] is False
    assert levels["Signal_Strength_Pct"] == 0.0


def test_lowvol_levels_from_frame_no_signal_without_rank_column():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    frame = swingtrade.levels.precompute_lowvol_frame(df, None, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.lowvol_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["LowVol_Percentile"] is None
    assert levels["LowVol_Signal"] is False


def test_lowvol_levels_from_frame_raises_on_macro_downtrend():
    n = 260
    # A steady DECLINE -- Last_Close should end up below its own SMA_TREND.
    df = _trending_ohlcv(n, 100.0, daily_drift=-0.003, seed=1)
    rank_column = pd.Series(95.0, index=df.index)  # would otherwise fire

    frame = swingtrade.levels.precompute_lowvol_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    try:
        swingtrade.levels.lowvol_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)
        assert False, "expected RuntimeError for macro downtrend"
    except RuntimeError as exc:
        assert "macro downtrend" in str(exc)


def _lowvol_row(lowvol_signal, signal_strength_pct=1.0):
    return {
        "LowVol_Signal": lowvol_signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct,
    }


def test_add_lowvol_trade_score_ineligible_when_no_lowvol_signal():
    df = pd.DataFrame([_lowvol_row(False)])
    result = swingtrade.add_lowvol_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_lowvol_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_lowvol_row(True, signal_strength_pct=5.0)])
    result = swingtrade.add_lowvol_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_lowvol_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _lowvol_row(True, signal_strength_pct=CONFIG.lowvol_strength_cap_pct),
        _lowvol_row(True, signal_strength_pct=CONFIG.lowvol_strength_cap_pct * 10),
    ])
    result = swingtrade.add_lowvol_trade_score(df, CONFIG)
    # Strength beyond the cap shouldn't earn extra score -- same clip-then-scale
    # shape every other z-score/pct-based strategy uses.
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_run_backtest_dispatches_lowvol_rank_and_resolves_own_column():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {
        "CALM": _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1, daily_noise=0.0005),
        "WILD": _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=2, daily_noise=0.02),
    }
    panel = pd.DataFrame({t: df["Close"] for t, df in ticker_data.items()})
    lowvol_rank_frame = swingtrade.compute_lowvol_rank_frame(panel, CONFIG.lowvol_lookback_days)

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "lowvol_rank"})
    window_start = ticker_data["CALM"].index[CONFIG.lowvol_lookback_days + 5]
    window_end = ticker_data["CALM"].index[-1]

    trades = swingtrade.run_backtest(
        ticker_data, market_ohlcv, window_start, window_end, config,
        strategy="lowvol_rank", lowvol_rank_frame=lowvol_rank_frame,
    )
    # Not asserting a specific count (depends on noise-driven fills), just
    # that dispatch works end-to-end with no exception and produces a
    # schema-compatible trade list when it does fire.
    for trade in trades:
        assert trade["signal"] == "LowVol_Rank"
        if trade["status"] != "OPEN":
            assert "pnl_pct" in trade

    # Unknown strategy string still raises (regression check on the error
    # message listing every valid strategy, including the new one).
    try:
        swingtrade.run_backtest(ticker_data, market_ohlcv, window_start, window_end, config, strategy="nonexistent")
        assert False, "expected ValueError for unknown strategy"
    except ValueError as exc:
        assert "lowvol_rank" in str(exc)


def test_run_random_backtest_dispatches_lowvol_rank():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {"CALM": _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1, daily_noise=0.0005)}
    window_start = ticker_data["CALM"].index[CONFIG.lowvol_lookback_days + 5]
    window_end = ticker_data["CALM"].index[-1]

    import random
    rng = random.Random(42)
    trades = swingtrade.run_random_backtest(
        ticker_data, market_ohlcv, window_start, window_end,
        real_trade_counts={"CALM": 3}, rng=rng, config=CONFIG, strategy="lowvol_rank",
    )
    for trade in trades:
        assert trade["signal"] == "Random_LowVol_Rank"
