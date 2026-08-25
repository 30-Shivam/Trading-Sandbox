"""run_random_backtest() -- the shared matched-count random-entry dispatch
table (2026-08-24), factored out so it's never duplicated a third time
across benchmark_random_entry.py/optimize.py (this codebase has already
been bitten twice by exactly this class of drift -- see ingest.py's and
dip_buy_analyzer.py's independently hand-maintained _score_for_strategy()
copies)."""
import random

import numpy as np
import pandas as pd

import swingtrade


def _trending_ohlcv(n: int, start_close: float, daily_drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = daily_drift + rng.normal(0, 0.01, n)
    close = start_close * np.cumprod(1 + returns)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2023-01-01", periods=n, freq="D"))


def test_run_random_backtest_matches_requested_trade_count_upper_bound():
    n = 500
    ticker_data = {
        "A": _trending_ohlcv(n, 100.0, 0.0005, seed=1),
        "B": _trending_ohlcv(n, 100.0, 0.0005, seed=2),
    }
    market_ohlcv = _trending_ohlcv(n, 400.0, 0.0003, seed=99)
    window_start = ticker_data["A"].index[100]
    window_end = ticker_data["A"].index[-1]
    real_trade_counts = {"A": 5, "B": 0}

    trades = swingtrade.run_random_backtest(
        ticker_data, market_ohlcv, window_start, window_end, real_trade_counts,
        rng=random.Random(1), config=swingtrade.DEFAULT_CONFIG, strategy="rsi",
    )
    a_trades = [t for t in trades if t["ticker"] == "A"]
    b_trades = [t for t in trades if t["ticker"] == "B"]
    # Never MORE than requested (can be fewer if not enough eligible
    # candidate days exist, or if a chosen day's fill never gets touched --
    # same "at most min(n_trades, len(candidates))" contract every
    # simulate_random_*_entries() function documents).
    assert len(a_trades) <= 5
    assert len(b_trades) == 0


def test_run_random_backtest_dispatches_correctly_per_strategy():
    n = 500
    ticker_data = {"A": _trending_ohlcv(n, 100.0, 0.0008, seed=1)}
    market_ohlcv = _trending_ohlcv(n, 400.0, 0.0003, seed=99)
    window_start = ticker_data["A"].index[100]
    window_end = ticker_data["A"].index[-1]

    for strategy in ["rsi", "breakout", "squeeze_breakout", "ma_crossover", "momentum_rank"]:
        config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy})
        trades = swingtrade.run_random_backtest(
            ticker_data, market_ohlcv, window_start, window_end, {"A": 3},
            rng=random.Random(1), config=config, strategy=strategy,
        )
        # No exception, schema-compatible output (or empty, both are fine --
        # this is purely a dispatch-correctness check, not a signal-quality one).
        for t in trades:
            assert t["ticker"] == "A"
            assert "signal" in t


def test_run_random_backtest_raises_on_unknown_strategy():
    ticker_data = {"A": _trending_ohlcv(300, 100.0, 0.0005, seed=1)}
    market_ohlcv = _trending_ohlcv(300, 400.0, 0.0003, seed=99)
    try:
        swingtrade.run_random_backtest(
            ticker_data, market_ohlcv, ticker_data["A"].index[50], ticker_data["A"].index[-1],
            {"A": 1}, rng=random.Random(1), strategy="nonexistent",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "momentum_rank" in str(exc)
