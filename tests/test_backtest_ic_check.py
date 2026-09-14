"""ic_tracking.backtest_ic_check() -- the new backtest-time counterpart to
methodology_report()'s own live overall_ic (improvements.txt, per explicit
user request to "enhance backtesting" after a real finding: rsi_mean_reversion
showed backtest_ic=-0.013 over 9,256 trades despite a live pooled overall_ic
of +0.605, a question this project's offline validation pipeline had never
actually asked before). Also covers the real, previously-missing `trade_score`
field now recorded by simulate_pairs_signals() (and its 3 siblings --
ma_crossover/squeeze_breakout/momentum_rank -- confirmed separately via a
real backtest run, not duplicated here) via the same synthetic pair-panel
fixture tests/test_pairs_multiprocessing_threading.py already established.
"""
import numpy as np
import pandas as pd
import pytest

import ic_tracking
import swingtrade


def test_backtest_ic_check_below_min_trades_returns_none():
    trades = [
        {"status": "WIN", "trade_score": 70.0, "pnl_pct": 2.0, "entry_date": pd.Timestamp("2024-01-01"), "sector": "Tech"},
        {"status": "LOSS", "trade_score": 30.0, "pnl_pct": -1.0, "entry_date": pd.Timestamp("2024-01-02"), "sector": "Tech"},
    ]
    result = ic_tracking.backtest_ic_check(trades, min_trades=10)
    assert result == {"n": 2, "ic": None}


def test_backtest_ic_check_excludes_open_trades():
    resolved = [
        {"status": "WIN", "trade_score": float(i), "pnl_pct": float(i),
         "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "sector": "Tech"}
        for i in range(10)
    ]
    open_trades = [
        {"status": "OPEN", "trade_score": 999.0, "pnl_pct": None,
         "entry_date": pd.Timestamp("2024-06-01"), "sector": "Tech"}
        for _ in range(5)
    ]
    result = ic_tracking.backtest_ic_check(resolved + open_trades, min_trades=10)
    assert result["n"] == 10, "OPEN trades must never count toward n or feed the IC"


def test_backtest_ic_check_excludes_trades_missing_trade_score():
    resolved = [
        {"status": "WIN", "trade_score": float(i), "pnl_pct": float(i),
         "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "sector": "Tech"}
        for i in range(10)
    ]
    no_score = [
        {"status": "WIN", "pnl_pct": 5.0, "entry_date": pd.Timestamp("2024-06-01"), "sector": "Tech"}
        for _ in range(5)
    ]
    result = ic_tracking.backtest_ic_check(resolved + no_score, min_trades=10)
    assert result["n"] == 10, "trades without a trade_score key must never count toward n or feed the IC"


def test_backtest_ic_check_detects_real_perfect_rank_correlation():
    # Spread across many distinct dates/sectors so cluster weighting doesn't
    # collapse this into a thin effective sample -- isolates the rank-IC
    # math itself, not the clustering correction (already covered by
    # swingtrade's own compute_cluster_weights tests).
    trades = [
        {"status": "WIN" if i % 2 == 0 else "LOSS", "trade_score": float(i), "pnl_pct": float(i) * 0.1,
         "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "sector": f"Sector{i % 5}"}
        for i in range(60)
    ]
    result = ic_tracking.backtest_ic_check(trades, min_trades=10)
    assert result["n"] == 60
    assert result["ic"] > 0.95, f"perfectly rank-correlated synthetic data should show IC near 1.0, got {result['ic']}"


def test_backtest_ic_check_detects_real_zero_skill():
    rng = np.random.default_rng(42)
    trades = [
        {"status": "WIN" if rng.random() > 0.5 else "LOSS", "trade_score": float(rng.uniform(0, 100)),
         "pnl_pct": float(rng.normal(0, 1)),
         "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "sector": f"Sector{i % 5}"}
        for i in range(200)
    ]
    result = ic_tracking.backtest_ic_check(trades, min_trades=10)
    assert result["n"] == 200
    assert abs(result["ic"]) < 0.2, f"independently-random score/pnl should show ~zero IC, got {result['ic']}"


def _make_pairs_config(entry_fill: str = "next_open") -> swingtrade.TradingConfig:
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "pairs", "pairs_entry_fill": entry_fill,
    })


def test_pairs_backtest_trades_carry_trade_score(uptrend_ohlcv, market_ohlcv):
    """Real regression coverage for the trade_score field
    simulate_pairs_signals() gained (this session, per the "enhance
    backtesting" request) -- built on the same session-scoped synthetic
    fixtures every other test file uses (so market_uptrend_from_frame's own
    date-alignment requirement between `ohlcv` and `market_ohlcv` is
    satisfied for free), with a PEER series diverging enough from TEST's
    own Close to reliably cross pairs_zscore_entry_max."""
    base = uptrend_ohlcv.copy()
    # A persistent downward-diverging spread: TEST drifts below PEER enough,
    # for long enough, to cross pairs_zscore_entry_max under default config.
    base["Close"] = base["Close"] - np.linspace(0, 40, len(base))
    base["Open"] = base["Close"]
    base["High"] = base["Close"] * 1.01
    base["Low"] = base["Close"] * 0.99
    peer_close = uptrend_ohlcv["Close"] + np.random.default_rng(2).normal(0, 0.5, len(uptrend_ohlcv))

    panel = pd.DataFrame({"PEER": peer_close}, index=base.index)
    config = _make_pairs_config()
    trades = swingtrade.simulate_pairs_signals(
        "TEST", base, market_ohlcv, base.index[0], base.index[-1], config,
        sector="Tech", peer_prices=panel,
    )
    if not trades:
        pytest.skip("synthetic fixture didn't cross pairs_zscore_entry_max this run -- not what's under test here")
    assert all("trade_score" in t and isinstance(t["trade_score"], float) for t in trades), (
        "every real pairs backtest trade must carry a real trade_score -- see "
        "ic_tracking.backtest_ic_check(), which silently reports n=0 without this"
    )
    assert all("signal_strength_pct" in t and isinstance(t["signal_strength_pct"], float) for t in trades), (
        "every real pairs backtest trade must also carry the raw signal_strength_pct component "
        "(2026-09-13, improvements.txt item 145) -- without it, swingtrade.audit_cap_calibration() "
        "can't be run directly against a real backtest's own trades, only via a separate dedicated "
        "collector script (audit_strength_cap_calibration.py)"
    )
