"""Cross-sectional VALUE RANK strategy -- ranks every ticker by its own
Price-to-Book ratio (Close / point-in-time Book Value Per Share, via SEC
EDGAR, see sec_fundamentals.py) and buys the CHEAPEST decile -- the classic
Fama-French value premium, opposite in spirit to quality_rank's "buy the
most PROFITABLE decile." Structural mirror of test_quality_rank_strategy.py.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _trending_ohlcv(n: int, start_close: float, daily_drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = daily_drift + rng.normal(0, 0.002, n)
    close = start_close * np.cumprod(1 + returns)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_compute_value_rank_frame_ranks_cheapest_price_to_book_at_100th_percentile():
    n = 30
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    # Every ticker has the SAME price (100) but DIFFERENT book value per
    # share -- T00 has the smallest book value (most expensive, highest
    # P/B), T19 has the largest book value (cheapest, lowest P/B).
    price_panel = pd.DataFrame({f"T{i:02d}": np.full(n, 100.0) for i in range(20)}, index=dates)
    bvps_panel = pd.DataFrame({f"T{i:02d}": np.full(n, 1.0 + i) for i in range(20)}, index=dates)

    rank_frame = swingtrade.compute_value_rank_frame(price_panel, bvps_panel)
    last_row = rank_frame.iloc[-1]

    assert last_row.notna().all()
    assert last_row["T19"] == 100.0  # highest book value -> lowest P/B -> cheapest -> best
    assert last_row["T00"] == 5.0    # lowest book value -> highest P/B -> most expensive -> worst
    # Percentile should be monotonically INCREASING as book value increases (P/B decreases).
    values = [last_row[f"T{i:02d}"] for i in range(20)]
    assert values == sorted(values)


def test_compute_value_rank_frame_missing_ticker_reads_nan():
    n = 10
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    price_panel = pd.DataFrame({"A": [100.0] * n, "B": [100.0] * n}, index=dates)
    bvps_panel = pd.DataFrame({"A": [10.0] * n, "B": [np.nan] * n}, index=dates)  # B has no SEC data
    rank_frame = swingtrade.compute_value_rank_frame(price_panel, bvps_panel)
    assert rank_frame["B"].isna().all()
    assert (rank_frame["A"] == 100.0).all()  # sole remaining ranked ticker


def test_value_levels_from_frame_fires_when_percentile_clears_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 95.0  # cheapest-decile P/B on the final (as_of) day only

    frame = swingtrade.levels.precompute_value_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.value_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Value_Percentile"] == 95.0
    assert levels["Value_Signal"] is True
    assert levels["Signal_Strength_Pct"] == 5.0
    assert levels["Buy_Price"] == levels["Last_Close"]
    assert levels["Distance_to_Buy_Pct"] == 0.0


def test_value_levels_from_frame_no_signal_just_below_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 89.9

    frame = swingtrade.levels.precompute_value_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.value_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Value_Signal"] is False
    assert levels["Signal_Strength_Pct"] == 0.0


def test_value_levels_from_frame_no_signal_without_rank_column():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    frame = swingtrade.levels.precompute_value_frame(df, None, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.value_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["Value_Percentile"] is None
    assert levels["Value_Signal"] is False


def test_value_levels_from_frame_raises_on_macro_downtrend():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=-0.003, seed=1)
    rank_column = pd.Series(95.0, index=df.index)

    frame = swingtrade.levels.precompute_value_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    try:
        swingtrade.levels.value_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)
        assert False, "expected RuntimeError for macro downtrend"
    except RuntimeError as exc:
        assert "macro downtrend" in str(exc)


def _value_row(signal, signal_strength_pct=1.0):
    return {"Value_Signal": signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct}


def test_add_value_trade_score_ineligible_when_no_signal():
    df = pd.DataFrame([_value_row(False)])
    result = swingtrade.add_value_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_value_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_value_row(True, signal_strength_pct=5.0)])
    result = swingtrade.add_value_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_value_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _value_row(True, signal_strength_pct=CONFIG.value_strength_cap_pct),
        _value_row(True, signal_strength_pct=CONFIG.value_strength_cap_pct * 10),
    ])
    result = swingtrade.add_value_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_run_backtest_dispatches_value_rank_and_resolves_own_column():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {
        "CHEAP": _trending_ohlcv(n, 100.0, daily_drift=0.002, seed=1),
        "EXPENSIVE": _trending_ohlcv(n, 100.0, daily_drift=0.0001, seed=2),
    }
    value_rank_frame = pd.DataFrame({
        "CHEAP": [95.0] * n,
        "EXPENSIVE": [10.0] * n,
    }, index=ticker_data["CHEAP"].index)

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "value_rank"})
    window_start = ticker_data["CHEAP"].index[210]
    window_end = ticker_data["CHEAP"].index[-1]

    trades = swingtrade.run_backtest(
        ticker_data, market_ohlcv, window_start, window_end, config,
        strategy="value_rank", value_rank_frame=value_rank_frame,
    )
    for trade in trades:
        assert trade["signal"] == "Value_Rank"
        if trade["status"] != "OPEN":
            assert "pnl_pct" in trade
    assert all(t["ticker"] == "CHEAP" for t in trades)

    try:
        swingtrade.run_backtest(ticker_data, market_ohlcv, window_start, window_end, config, strategy="nonexistent")
        assert False, "expected ValueError for unknown strategy"
    except ValueError as exc:
        assert "value_rank" in str(exc)


def test_run_random_backtest_dispatches_value_rank():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {"CHEAP": _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)}
    window_start = ticker_data["CHEAP"].index[210]
    window_end = ticker_data["CHEAP"].index[-1]

    import random
    rng = random.Random(42)
    trades = swingtrade.run_random_backtest(
        ticker_data, market_ohlcv, window_start, window_end,
        real_trade_counts={"CHEAP": 3}, rng=rng, config=CONFIG, strategy="value_rank",
    )
    for trade in trades:
        assert trade["signal"] == "Random_Value_Rank"
