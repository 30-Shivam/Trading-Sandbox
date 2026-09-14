"""simulate_signals() (RSI, real) -- its own trade dicts' "signal" field
used to carry the LIVE, per-trade Buy/Strong Buy value (`scored["Signal"]`)
instead of a stable per-strategy label, unlike every other real
simulate_*_signals() function in this codebase ("MA_Crossover", "Pairs",
"Breakout", etc.) -- a real, previously-unnoticed inconsistency found
2026-09-14 (improvements.txt item 150) while extending
benchmark_multi_strategy_portfolio.py to a third live strategy: pooled
group-by-"signal" attribution silently fragmented RSI's own real trades
across "Buy"/"Strong Buy" keys instead of a single "RSI" bucket, since
nothing before this session ever grouped real trades by "signal" across
strategies.

Fixed by tagging "signal": "RSI" (matching every sibling strategy's own
convention) while preserving the original live value under a NEW
"live_signal" key -- no information lost, just no longer silently
misused as a strategy identifier.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _uptrend_with_dips(n: int, seed: int) -> pd.DataFrame:
    """A steady uptrend with periodic pullbacks -- reliably fires RSI's
    own oversold-mean-reversion trigger multiple times, unlike a smooth
    monotonic uptrend."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 100 + t * 0.15
    dips = 8 * np.sin(t / 12.0)
    close = trend + dips + rng.normal(0, 0.5, n)
    high = close + rng.uniform(0.2, 0.5, n)
    low = close - rng.uniform(0.2, 0.5, n)
    open_ = close + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(3_000_000, 5_000_000, n).astype(float),
    }, index=pd.bdate_range("2023-01-02", periods=n))


def test_rsi_real_trades_carry_a_stable_strategy_label():
    df = _uptrend_with_dips(420, seed=11)
    market = _uptrend_with_dips(420, seed=99)
    trades = swingtrade.simulate_signals(
        "TEST", df, market, df.index[0], df.index[-1], CONFIG, sector="Tech",
    )
    if not trades:
        import pytest
        pytest.skip("synthetic fixture produced no real RSI signals this run -- not what's under test here")

    assert all(t["signal"] == "RSI" for t in trades), (
        "every real RSI backtest trade must carry a STABLE 'RSI' strategy label in 'signal' -- "
        "matching every sibling simulate_*_signals() function's own convention -- so anything "
        "that pools/groups trades by 'signal' across strategies (see "
        "swingtrade.compute_strategy_correlation()/simulate_portfolio_constrained(group_key=...)) "
        "attributes RSI's own trades correctly instead of fragmenting them across live Buy/Strong "
        "Buy values"
    )
    assert all(t.get("live_signal") in ("Buy", "Strong Buy") for t in trades), (
        "the original live Buy/Strong Buy value must still be preserved under 'live_signal' -- "
        "fixing the strategy-label bug must not silently discard this information"
    )
