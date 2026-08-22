"""Mean-reversion PAIRS strategy (long-only laggard-convergence, added per
improvements.txt item 82) -- buys a ticker when it has diverged unusually
far BELOW its most-correlated same-sector peer over a recent window,
betting on convergence. No short leg (this codebase has no short-position
support anywhere) -- this is the long-only variant strategy_selection.txt
itself flagged.
"""
import numpy as np
import pandas as pd

import swingtrade

CONFIG = swingtrade.DEFAULT_CONFIG


def _correlated_ohlcv(n: int, start_close: float, shared_returns: np.ndarray, noise_scale: float, seed: int) -> pd.DataFrame:
    """A synthetic ticker whose daily returns = shared_returns (a common
    factor, driving correlation with any other ticker built from the same
    shared_returns array) plus small independent noise."""
    rng = np.random.default_rng(seed)
    idiosyncratic = rng.normal(0, noise_scale, n)
    close = start_close * np.cumprod(1 + shared_returns + idiosyncratic)
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def _independent_ohlcv(n: int, start_close: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_close * np.cumprod(1 + rng.normal(0, 0.015, n))
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_precompute_pairs_frame_picks_the_correlated_peer_not_the_independent_one():
    n = 260
    shared = np.random.default_rng(0).normal(0, 0.01, n)
    ticker_df = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=1)
    correlated_peer = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=2)
    independent_peer = _independent_ohlcv(n, 100.0, seed=3)

    peer_prices = pd.DataFrame({"CORR": correlated_peer["Close"], "INDEP": independent_peer["Close"]})
    frame = swingtrade.levels.precompute_pairs_frame(ticker_df, peer_prices, CONFIG)

    tail = frame[["Pair_Partner", "Pair_Correlation"]].dropna().tail(20)
    assert len(tail) > 0
    assert (tail["Pair_Partner"] == "CORR").all()
    assert (tail["Pair_Correlation"] > CONFIG.pairs_min_correlation).all()


def test_precompute_pairs_frame_no_partner_when_no_peer_clears_min_correlation():
    n = 260
    ticker_df = _independent_ohlcv(n, 100.0, seed=1)
    peer_a = _independent_ohlcv(n, 100.0, seed=2)
    peer_b = _independent_ohlcv(n, 100.0, seed=3)
    peer_prices = pd.DataFrame({"A": peer_a["Close"], "B": peer_b["Close"]})

    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "pairs_min_correlation": 0.95})
    frame = swingtrade.levels.precompute_pairs_frame(ticker_df, peer_prices, config)

    assert frame["Pair_Partner"].notna().sum() == 0
    assert frame["Pair_Spread_Zscore"].notna().sum() == 0


def test_precompute_pairs_frame_absent_without_peer_prices():
    n = 260
    ticker_df = _independent_ohlcv(n, 100.0, seed=1)
    frame = swingtrade.levels.precompute_pairs_frame(ticker_df, None, CONFIG)
    assert frame["Pair_Partner"].notna().sum() == 0
    assert frame["Pair_Spread_Zscore"].notna().sum() == 0


def test_precompute_pairs_frame_zscore_goes_negative_when_ticker_underperforms_partner():
    n = 260
    shared = np.random.default_rng(0).normal(0, 0.01, n)
    ticker_df = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=1)
    partner_df = _correlated_ohlcv(n, 100.0, shared, noise_scale=0.001, seed=2)

    # Force a real, recent divergence: crash the ticker's last 15 closes
    # relative to the partner (partner untouched), well past pairs_spread_window_days.
    ticker_df = ticker_df.copy()
    ticker_df.loc[ticker_df.index[-15:], "Close"] *= np.linspace(1.0, 0.85, 15)

    peer_prices = pd.DataFrame({"PARTNER": partner_df["Close"]})
    frame = swingtrade.levels.precompute_pairs_frame(ticker_df, peer_prices, CONFIG)

    last_zscore = frame["Pair_Spread_Zscore"].iloc[-1]
    assert pd.notna(last_zscore)
    assert last_zscore < -1.0  # a real, unusual divergence, not just noise


def _pairs_row(pair_signal, signal_strength_pct=1.0):
    return {
        "Pair_Signal": pair_signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct,
    }


def test_add_pairs_trade_score_ineligible_when_no_pair_signal():
    df = pd.DataFrame([_pairs_row(False)])
    result = swingtrade.add_pairs_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_pairs_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_pairs_row(True, signal_strength_pct=1.5)])
    result = swingtrade.add_pairs_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_pairs_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _pairs_row(True, signal_strength_pct=CONFIG.pairs_zscore_strength_cap),
        _pairs_row(True, signal_strength_pct=CONFIG.pairs_zscore_strength_cap * 10),
    ])
    result = swingtrade.add_pairs_trade_score(df, CONFIG)
    # Strength beyond the cap shouldn't earn extra score -- same clip-then-scale
    # shape every other z-score/pct-based strategy uses.
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]
