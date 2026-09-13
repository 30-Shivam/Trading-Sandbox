"""real_vs_random_ratio_check()'s multi-seed RANDOM baseline (2026-09-13,
improvements.txt item 132) -- the same sampling-noise problem
[[feedback-strategy-validation-pipeline]] point 14 already found and fixed
for ticker-holdout splits, applied here for the first time: a single fixed
random-entry draw (the old `seed=1` default) could itself be unusually
lucky/unlucky, potentially flipping whether a real edge looks like it beats
random at all. REAL is deterministic (simulated once); only RANDOM is
re-drawn per seed and averaged via optimize.average_summaries(), which also
reports the min/max spread rather than hiding seed-to-seed instability.
"""
import pandas as pd
import pytest

import optimize
import swingtrade


def _resolved_trade(pnl_pct: float, day: int) -> dict:
    return {
        "ticker": "TEST", "status": "WIN" if pnl_pct > 0 else "LOSS", "pnl_pct": pnl_pct,
        "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), "sector": "Tech",
    }


@pytest.fixture
def config():
    return swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "ma_crossover"})


def test_real_vs_random_ratio_check_averages_random_across_seeds(monkeypatch, config):
    # Some real spread within each trade set -- a constant pnl_pct across
    # every trade gives std=0, an undefined (None) sharpe_like, same
    # "don't fabricate a ratio from zero variance" convention this
    # project's own rank_ic()/sharpe math already follows elsewhere.
    real_trades = [_resolved_trade(2.0 + (i % 3) * 0.1, i) for i in range(20)]
    # Three distinct per-seed random-baseline outcomes -- a real average
    # should land strictly between the min and max, not equal either one.
    per_seed_random_trades = {
        1: [_resolved_trade(-3.0 + (i % 3) * 0.1, i) for i in range(20)],
        42: [_resolved_trade(1.0 + (i % 3) * 0.1, i) for i in range(20)],
        7: [_resolved_trade(-1.0 + (i % 3) * 0.1, i) for i in range(20)],
    }
    calls = []

    def fake_run_backtest(*args, **kwargs):
        return real_trades

    def fake_run_random_backtest(ticker_data, market_data, start, end, real_counts, rng, config, **kwargs):
        assert hasattr(rng, "sample"), "expected a real random.Random instance, one per seed"
        calls.append(rng)
        # Relies on call order matching the exact `seeds` list order this
        # test passes below ([1, 42, 7]) -- real_vs_random_ratio_check()
        # iterates seeds in order, one run_random_backtest call each.
        idx = len(calls) - 1
        ordered = [per_seed_random_trades[1], per_seed_random_trades[42], per_seed_random_trades[7]]
        return ordered[idx]

    monkeypatch.setattr(optimize.swingtrade, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(optimize.swingtrade, "run_random_backtest", fake_run_random_backtest)

    result = optimize.real_vs_random_ratio_check(
        "ma_crossover", config, {"TEST": pd.DataFrame()}, pd.DataFrame(), pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-06-01"), {"TEST": "Tech"}, seeds=[1, 42, 7],
    )

    assert len(calls) == 3, "expected exactly one run_random_backtest call per seed"
    assert result["random"]["_seeds_used"] == 3
    # Averaged random sharpe_like should sit strictly between the per-seed
    # extremes -- not collapse to a single draw.
    random_sharpes = [
        optimize.swingtrade.summarize_trades_weighted(
            t, optimize.swingtrade.compute_cluster_weights(t)
        )["sharpe_like"]
        for t in per_seed_random_trades.values()
    ]
    assert min(random_sharpes) < result["random"]["sharpe_like"] < max(random_sharpes)
    # gap min/max should reflect the SAME real_sharpe against each seed's
    # own random draw, not the averaged one.
    assert result["_gap_min"] < result["gap"] < result["_gap_max"] or result["_gap_min"] == result["_gap_max"]


def test_real_vs_random_ratio_check_single_seed_list_reproduces_old_behavior(monkeypatch, config):
    """Passing seeds=[seed] (a single-element list) must reproduce the
    pre-2026-09-13 one-draw behavior exactly -- backward compatible for a
    fast smoke test where extra draws aren't worth the cost."""
    real_trades = [_resolved_trade(2.0, i) for i in range(20)]
    random_trades = [_resolved_trade(-1.0, i) for i in range(20)]

    monkeypatch.setattr(optimize.swingtrade, "run_backtest", lambda *a, **k: real_trades)
    monkeypatch.setattr(optimize.swingtrade, "run_random_backtest", lambda *a, **k: random_trades)

    result = optimize.real_vs_random_ratio_check(
        "ma_crossover", config, {"TEST": pd.DataFrame()}, pd.DataFrame(), pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-06-01"), {"TEST": "Tech"}, seeds=[1],
    )
    assert result["random"]["_seeds_used"] == 1
    assert result["_gap_min"] == result["_gap_max"] == result["gap"]


def test_research_holdout_cutoff_is_a_real_timestamp():
    from run_backtest import RESEARCH_HOLDOUT_CUTOFF
    assert isinstance(RESEARCH_HOLDOUT_CUTOFF, pd.Timestamp)
