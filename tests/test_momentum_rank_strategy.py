"""Cross-sectional MOMENTUM RANK strategy -- ranks every ticker in the
watchlist by trailing return and buys the top decile (Jegadeesh & Titman
cross-sectional momentum). The first strategy in this project whose signal
for one ticker depends on every OTHER ticker's own return the same day --
every prior strategy scores a ticker from its own price history alone.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _trending_ohlcv(n: int, start_close: float, daily_drift: float, seed: int) -> pd.DataFrame:
    """A synthetic ticker with a steady daily drift (small per-day return)
    plus tiny noise -- enough real history for SMA_TREND/ATR/AvgVolume to
    all be valid, and (for a positive drift) to clear the macro-uptrend gate."""
    rng = np.random.default_rng(seed)
    returns = daily_drift + rng.normal(0, 0.002, n)
    close = start_close * np.cumprod(1 + returns)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_compute_momentum_rank_frame_ranks_highest_return_at_100th_percentile():
    n = 260
    # 20 tickers, each with a distinctly different, NOISE-FREE constant
    # drift -- ticker "T19" has the highest drift (should rank #1/highest
    # percentile on the last day), "T00" the lowest. Deliberately
    # deterministic (no per-ticker noise): this test verifies the ranking
    # MATH itself, not realistic price behavior -- noise large enough to be
    # realistic can also be large enough to swap adjacent, closely-spaced
    # drift ranks over a 63-day compounding window, which would make this
    # test flaky for the wrong reason.
    panel = pd.DataFrame({
        f"T{i:02d}": 100.0 * np.cumprod(np.full(n, 1.0 + 0.0001 * i))
        for i in range(20)
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    rank_frame = swingtrade.compute_momentum_rank_frame(panel, CONFIG.momentum_lookback_days)

    last_row = rank_frame.iloc[-1]
    assert last_row.notna().all()
    # Highest drift -> highest trailing return -> 100th percentile (last of 20).
    assert last_row["T19"] == 100.0
    # Lowest drift -> lowest trailing return -> 1/20th percentile.
    assert last_row["T00"] == 5.0
    # Monotonic: higher drift index -> higher (or equal) percentile.
    values = [last_row[f"T{i:02d}"] for i in range(20)]
    assert values == sorted(values)


def test_compute_momentum_rank_frame_early_rows_are_nan_before_lookback_fills():
    n = 260
    panel = pd.DataFrame({
        "A": _trending_ohlcv(n, 100.0, 0.0005, seed=1)["Close"],
        "B": _trending_ohlcv(n, 100.0, 0.001, seed=2)["Close"],
    })
    rank_frame = swingtrade.compute_momentum_rank_frame(panel, lookback_days=63)
    # Before 63 trading days of history, pct_change(63) is NaN for every ticker.
    assert rank_frame.iloc[0].isna().all()
    assert rank_frame.iloc[62].isna().all()
    assert rank_frame.iloc[-1].notna().all()


def test_momentum_levels_from_frame_fires_when_percentile_clears_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    # Hand-construct the rank column directly (bypassing compute_momentum_rank_frame
    # entirely) so this test isolates momentum_levels_from_frame()'s own gating
    # logic from the ranking math already covered above.
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 95.0  # top-decile on the final (as_of) day only

    frame = swingtrade.levels.precompute_momentum_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.momentum_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Momentum_Percentile"] == 95.0
    assert levels["Momentum_Signal"] is True
    # 95 - momentum_top_percentile_min (90.0) = 5.0
    assert levels["Signal_Strength_Pct"] == 5.0
    assert levels["Buy_Price"] == levels["Last_Close"]
    assert levels["Distance_to_Buy_Pct"] == 0.0


def test_momentum_levels_from_frame_no_signal_just_below_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 89.9  # just under momentum_top_percentile_min (90.0)

    frame = swingtrade.levels.precompute_momentum_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.momentum_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Momentum_Signal"] is False
    assert levels["Signal_Strength_Pct"] == 0.0


def test_momentum_levels_from_frame_no_signal_without_rank_column():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    frame = swingtrade.levels.precompute_momentum_frame(df, None, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.momentum_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Momentum_Percentile"] is None
    assert levels["Momentum_Signal"] is False


def test_momentum_levels_from_frame_raises_on_macro_downtrend():
    n = 260
    # A steady DECLINE -- Last_Close should end up below its own SMA_TREND.
    df = _trending_ohlcv(n, 100.0, daily_drift=-0.003, seed=1)
    rank_column = pd.Series(95.0, index=df.index)  # would otherwise fire

    frame = swingtrade.levels.precompute_momentum_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    try:
        swingtrade.levels.momentum_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)
        assert False, "expected RuntimeError for macro downtrend"
    except RuntimeError as exc:
        assert "macro downtrend" in str(exc)


def _momentum_row(momentum_signal, signal_strength_pct=1.0):
    return {
        "Momentum_Signal": momentum_signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct,
    }


def test_add_momentum_trade_score_ineligible_when_no_momentum_signal():
    df = pd.DataFrame([_momentum_row(False)])
    result = swingtrade.add_momentum_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_momentum_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_momentum_row(True, signal_strength_pct=5.0)])
    result = swingtrade.add_momentum_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_momentum_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _momentum_row(True, signal_strength_pct=CONFIG.momentum_strength_cap_pct),
        _momentum_row(True, signal_strength_pct=CONFIG.momentum_strength_cap_pct * 10),
    ])
    result = swingtrade.add_momentum_trade_score(df, CONFIG)
    # Strength beyond the cap shouldn't earn extra score -- same clip-then-scale
    # shape every other z-score/pct-based strategy uses.
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_run_backtest_dispatches_momentum_rank_and_resolves_own_column():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {
        "HIGH": _trending_ohlcv(n, 100.0, daily_drift=0.002, seed=1),
        "LOW": _trending_ohlcv(n, 100.0, daily_drift=0.0001, seed=2),
    }
    panel = pd.DataFrame({t: df["Close"] for t, df in ticker_data.items()})
    momentum_rank_frame = swingtrade.compute_momentum_rank_frame(panel, CONFIG.momentum_lookback_days)

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "momentum_rank"})
    window_start = ticker_data["HIGH"].index[CONFIG.momentum_lookback_days + 5]
    window_end = ticker_data["HIGH"].index[-1]

    trades = swingtrade.run_backtest(
        ticker_data, market_ohlcv, window_start, window_end, config,
        strategy="momentum_rank", momentum_rank_frame=momentum_rank_frame,
    )
    # Not asserting a specific count (depends on noise-driven fills), just
    # that dispatch works end-to-end with no exception and produces a
    # schema-compatible trade list when it does fire.
    for trade in trades:
        assert trade["signal"] == "Momentum_Rank"
        # A trade still OPEN at window_end has no resolved pnl_pct yet --
        # same "resolved = [t for t in trades if t['status'] != 'OPEN']"
        # convention every summarizer in this codebase follows.
        if trade["status"] != "OPEN":
            assert "pnl_pct" in trade

    # Unknown strategy string still raises (regression check on the error
    # message listing every valid strategy, including the new one).
    try:
        swingtrade.run_backtest(ticker_data, market_ohlcv, window_start, window_end, config, strategy="nonexistent")
        assert False, "expected ValueError for unknown strategy"
    except ValueError as exc:
        assert "momentum_rank" in str(exc)
