"""swingtrade.permutation_test_gap() (2026-09-13, improvements.txt item 135)
-- a real, quantified confidence figure on the real-vs-random GAP itself,
distinct from DSR (selection bias across many Optuna TRIALS) and from
backtest_ic_check() (does the score rank outcomes, not just beat random on
aggregate). A standard permutation/shuffle test: pools REAL + RANDOM trades,
repeatedly reshuffles into groups of the original sizes, and reports what
fraction of those null shuffles produce a gap at least as large as the one
actually observed.
"""
import pandas as pd

import swingtrade


def _trade(pnl_pct: float, day: int) -> dict:
    return {
        "status": "WIN" if pnl_pct > 0 else "LOSS", "pnl_pct": pnl_pct,
        "entry_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
    }


def test_permutation_test_gap_low_p_value_for_a_real_clear_edge():
    # REAL consistently positive, RANDOM consistently negative -- a real,
    # unambiguous edge should be very unlikely under the shuffle null.
    real_trades = [_trade(3.0 + (i % 3) * 0.1, i) for i in range(60)]
    random_trades = [_trade(-3.0 + (i % 3) * 0.1, i) for i in range(60)]
    result = swingtrade.permutation_test_gap(real_trades, random_trades, n_permutations=500, seed=1)
    assert result["p_value"] is not None
    assert result["p_value"] < 0.05, f"expected a low p_value for a clear edge, got {result['p_value']}"
    assert result["observed_gap"] > 0


def test_permutation_test_gap_high_p_value_when_real_and_random_are_indistinguishable():
    # REAL and RANDOM drawn from the SAME distribution -- no real edge, the
    # observed gap should look like a typical, unremarkable draw from the
    # null (a high, not low, p_value -- no false positive).
    import random as _random
    rng = _random.Random(7)
    pool = [_trade(rng.uniform(-2, 2), i) for i in range(120)]
    real_trades = pool[:60]
    random_trades = pool[60:]
    result = swingtrade.permutation_test_gap(real_trades, random_trades, n_permutations=500, seed=2)
    assert result["p_value"] is not None
    assert result["p_value"] > 0.05, (
        f"expected a high p_value when real/random are genuinely interchangeable, got {result['p_value']}"
    )


def test_permutation_test_gap_degrades_gracefully_on_undefined_sharpe():
    # A single trade (or constant pnl_pct) gives an undefined (None)
    # sharpe_like -- must not crash, must report p_value=None honestly.
    real_trades = [_trade(1.0, 0)]
    random_trades = [_trade(1.0, 0)]
    result = swingtrade.permutation_test_gap(real_trades, random_trades, n_permutations=50, seed=3)
    assert result["p_value"] is None
    assert result["n_permutations"] == 0


def test_permutation_test_gap_is_deterministic_given_the_same_seed():
    real_trades = [_trade(2.0 + (i % 4) * 0.2, i) for i in range(40)]
    random_trades = [_trade(-1.0 + (i % 4) * 0.2, i) for i in range(40)]
    r1 = swingtrade.permutation_test_gap(real_trades, random_trades, n_permutations=200, seed=11)
    r2 = swingtrade.permutation_test_gap(real_trades, random_trades, n_permutations=200, seed=11)
    assert r1 == r2


def test_permutation_test_gap_uses_custom_summarize_fn():
    real_trades = [_trade(2.0 + (i % 3) * 0.1, i) for i in range(30)]
    random_trades = [_trade(-1.0 + (i % 3) * 0.1, i) for i in range(30)]
    calls = {"n": 0}

    def counting_summarize(trades):
        calls["n"] += 1
        return swingtrade.summarize_trades(trades)

    result = swingtrade.permutation_test_gap(real_trades, random_trades, counting_summarize, n_permutations=20, seed=5)
    assert calls["n"] > 20, "custom summarize_fn must actually be used for both the observed gap and every shuffle"
    assert result["p_value"] is not None
