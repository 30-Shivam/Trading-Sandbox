import numpy as np
import pandas as pd

import factor_portfolios as fp


def _panel(n_tickers=60, n_months=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    drift = np.linspace(-0.01, 0.02, n_tickers)           # ticker i has a persistent drift
    rets = rng.normal(0, 0.02, (n_months, n_tickers)) + drift
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"T{i}" for i in range(n_tickers)])


def test_top_quintile_of_a_perfect_score_beats_the_average():
    me = _panel()
    perfect = pd.DataFrame(np.tile(np.linspace(-1, 1, me.shape[1]), (len(me), 1)), index=me.index, columns=me.columns)
    top = fp.run_portfolio(me, perfect, 0.2, cost_bps=0)
    ew = me.pct_change().mean(axis=1)
    assert top.mean() > ew.reindex(top.index).mean()


def test_costs_reduce_returns_and_no_lookahead_on_last_month():
    me = _panel()
    score = pd.DataFrame(np.random.default_rng(1).normal(size=me.shape), index=me.index, columns=me.columns)
    free, costly = fp.run_portfolio(me, score, 0.2, 0), fp.run_portfolio(me, score, 0.2, 50)
    assert costly.sum() < free.sum()
    assert free.index.max() == me.index[-1] and free.index.min() == me.index[1]  # decide at t, earn t+1


def test_stats_alpha_beta_against_benchmark():
    idx = pd.date_range("2020-01-31", periods=48, freq="ME")
    bench = pd.Series(np.random.default_rng(2).normal(0.01, 0.04, 48), index=idx)
    s = fp.stats(1.0 * bench + 0.005, bench)
    assert abs(s["beta"] - 1.0) < 1e-6 and abs(s["excess%/yr"] - 6.0) < 1e-6
