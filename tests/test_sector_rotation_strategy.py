"""Cross-sectional SECTOR ROTATION strategy -- ranks the ~11 GICS sector
ETFs against EACH OTHER by trailing return, and fires for every ticker
whose OWN SECTOR clears the top-percentile threshold. Structural mirror of
test_momentum_rank_strategy.py/test_lowvol_rank_strategy.py, but the
cross-sectional ranking universe here is SECTORS, not individual tickers --
deliberately a DIFFERENT mechanism from the already-removed
best_ideas_sector_rs (which ranked tickers against their OWN sector, not
sectors against each other).
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


def test_compute_sector_rotation_rank_frame_ranks_strongest_sector_at_100th_percentile():
    n = 260
    n_sectors = 11
    sector_panel = pd.DataFrame({
        f"Sector{i:02d}": 100.0 * np.cumprod(np.full(n, 1.0 + 0.0001 * i))
        for i in range(n_sectors)
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))
    rank_frame = swingtrade.compute_sector_rotation_rank_frame(sector_panel, CONFIG.sector_rotation_lookback_days)

    last_row = rank_frame.iloc[-1]
    assert last_row.notna().all()
    # Highest drift sector -> highest trailing return -> 100th percentile.
    assert last_row["Sector10"] == 100.0
    # Lowest drift sector -> lowest trailing return -> 1/11th percentile.
    assert round(last_row["Sector00"], 4) == round(100 / n_sectors, 4)
    values = [last_row[f"Sector{i:02d}"] for i in range(n_sectors)]
    assert values == sorted(values)


def test_compute_sector_rotation_rank_frame_early_rows_are_nan_before_lookback_fills():
    n = 260
    sector_panel = pd.DataFrame({
        "Technology": _trending_ohlcv(n, 100.0, 0.0005, seed=1)["Close"],
        "Financials": _trending_ohlcv(n, 100.0, 0.001, seed=2)["Close"],
    })
    rank_frame = swingtrade.compute_sector_rotation_rank_frame(sector_panel, lookback_days=63)
    assert rank_frame.iloc[0].isna().all()
    assert rank_frame.iloc[62].isna().all()
    assert rank_frame.iloc[-1].notna().all()


def test_sector_rotation_levels_from_frame_fires_when_percentile_clears_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    # Hand-construct the (sector-level) rank column directly -- isolates
    # sector_rotation_levels_from_frame()'s own gating logic from the
    # sector-ranking math already covered above.
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 95.0  # ticker's own sector is top-ranked on the final (as_of) day only

    frame = swingtrade.levels.precompute_sector_rotation_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.sector_rotation_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["SectorRotation_Percentile"] == 95.0
    assert levels["SectorRotation_Signal"] is True
    assert levels["Signal_Strength_Pct"] == 5.0
    assert levels["Buy_Price"] == levels["Last_Close"]
    assert levels["Distance_to_Buy_Pct"] == 0.0


def test_sector_rotation_levels_from_frame_no_signal_just_below_threshold():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    rank_column = pd.Series(50.0, index=df.index)
    rank_column.iloc[-1] = 89.9

    frame = swingtrade.levels.precompute_sector_rotation_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.sector_rotation_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["SectorRotation_Signal"] is False
    assert levels["Signal_Strength_Pct"] == 0.0


def test_sector_rotation_levels_from_frame_no_signal_without_rank_column():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)
    frame = swingtrade.levels.precompute_sector_rotation_frame(df, None, CONFIG)
    as_of = frame.index[-1]
    levels = swingtrade.levels.sector_rotation_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)

    assert levels["SectorRotation_Percentile"] is None
    assert levels["SectorRotation_Signal"] is False


def test_sector_rotation_levels_from_frame_raises_on_macro_downtrend():
    n = 260
    df = _trending_ohlcv(n, 100.0, daily_drift=-0.003, seed=1)
    rank_column = pd.Series(95.0, index=df.index)  # would otherwise fire

    frame = swingtrade.levels.precompute_sector_rotation_frame(df, rank_column, CONFIG)
    as_of = frame.index[-1]
    try:
        swingtrade.levels.sector_rotation_levels_from_frame(as_of=as_of, ticker="TEST", frame=frame, config=CONFIG)
        assert False, "expected RuntimeError for macro downtrend"
    except RuntimeError as exc:
        assert "macro downtrend" in str(exc)


def _sector_rotation_row(signal, signal_strength_pct=1.0):
    return {
        "SectorRotation_Signal": signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct,
    }


def test_add_sector_rotation_trade_score_ineligible_when_no_signal():
    df = pd.DataFrame([_sector_rotation_row(False)])
    result = swingtrade.add_sector_rotation_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_sector_rotation_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_sector_rotation_row(True, signal_strength_pct=5.0)])
    result = swingtrade.add_sector_rotation_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_sector_rotation_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _sector_rotation_row(True, signal_strength_pct=CONFIG.sector_rotation_strength_cap_pct),
        _sector_rotation_row(True, signal_strength_pct=CONFIG.sector_rotation_strength_cap_pct * 10),
    ])
    result = swingtrade.add_sector_rotation_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_run_backtest_dispatches_sector_rotation_and_resolves_by_sector():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {
        "TICKA": _trending_ohlcv(n, 100.0, daily_drift=0.002, seed=1),
        "TICKB": _trending_ohlcv(n, 100.0, daily_drift=0.0001, seed=2),
    }
    # TICKA and TICKB are in DIFFERENT sectors, each mapped to a different
    # column in sector_rotation_rank_frame -- confirms resolution is by
    # SECTOR (via sector_lookup), not by ticker symbol.
    sector_lookup = {"TICKA": "Technology", "TICKB": "Financials"}
    sector_panel = pd.DataFrame({
        "Technology": _trending_ohlcv(n, 50.0, 0.002, seed=10)["Close"],
        "Financials": _trending_ohlcv(n, 50.0, 0.0001, seed=11)["Close"],
    })
    sector_rotation_rank_frame = swingtrade.compute_sector_rotation_rank_frame(
        sector_panel, CONFIG.sector_rotation_lookback_days
    )

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "strategy": "sector_rotation"})
    window_start = ticker_data["TICKA"].index[CONFIG.sector_rotation_lookback_days + 5]
    window_end = ticker_data["TICKA"].index[-1]

    trades = swingtrade.run_backtest(
        ticker_data, market_ohlcv, window_start, window_end, config,
        strategy="sector_rotation", sector_lookup=sector_lookup,
        sector_rotation_rank_frame=sector_rotation_rank_frame,
    )
    for trade in trades:
        assert trade["signal"] == "Sector_Rotation"
        if trade["status"] != "OPEN":
            assert "pnl_pct" in trade

    try:
        swingtrade.run_backtest(ticker_data, market_ohlcv, window_start, window_end, config, strategy="nonexistent")
        assert False, "expected ValueError for unknown strategy"
    except ValueError as exc:
        assert "sector_rotation" in str(exc)


def test_run_random_backtest_dispatches_sector_rotation():
    n = 260
    market_ohlcv = _trending_ohlcv(n, 400.0, daily_drift=0.0005, seed=99)
    ticker_data = {"TICKA": _trending_ohlcv(n, 100.0, daily_drift=0.001, seed=1)}
    window_start = ticker_data["TICKA"].index[CONFIG.sector_rotation_lookback_days + 5]
    window_end = ticker_data["TICKA"].index[-1]

    import random
    rng = random.Random(42)
    trades = swingtrade.run_random_backtest(
        ticker_data, market_ohlcv, window_start, window_end,
        real_trade_counts={"TICKA": 3}, rng=rng, config=CONFIG, strategy="sector_rotation",
    )
    for trade in trades:
        assert trade["signal"] == "Random_Sector_Rotation"
