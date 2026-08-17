"""average_holdout_summary() -- averages TUNE/HOLDOUT summary stats across
multiple ticker-holdout seeds instead of trusting one. Built after a real
2026-08-15 investigation (improvements.txt item 69) found a single
fixed-seed 102-ticker HOLDOUT split carries substantial sampling noise: two
real strategy checks (breakout v43, sector relative-strength) both had
their HOLDOUT verdict flip direction across just 3 different seeds, while
their ALL/TUNE cuts stayed stable -- confirmed independently by a neutral,
no-strategy buy-and-hold sweep showing -44pp to +54pp swings in the
TUNE-vs-HOLDOUT return gap purely from which tickers land where.
"""
import optimize

SECTOR_LOOKUP = {
    "AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "DDD": "Tech",
    "EEE": "Energy", "FFF": "Energy", "GGG": "Energy", "HHH": "Energy",
}
TICKER_VALUE = {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0, "EEE": 5.0, "FFF": 6.0, "GGG": 7.0, "HHH": 8.0}
TRADES = [{"ticker": t, "value": v} for t, v in TICKER_VALUE.items()]


def fake_summarize(trades: list[dict]) -> dict:
    """Deterministic stand-in for the real summarize() -- mean of each
    trade's own 'value' field, so expected results can be hand-computed."""
    if not trades:
        return {"trade_count": 0, "sharpe_like": None, "annualized_sharpe_like": None}
    values = [t["value"] for t in trades]
    return {
        "trade_count": len(trades),
        "sharpe_like": sum(values) / len(values),
        "annualized_sharpe_like": None,  # always None -- tests the None-skipping path
    }


def _expected_bucket(seeds: list[int], frac: float, want_holdout: bool) -> list[dict]:
    summaries = []
    for seed in seeds:
        tune, holdout = optimize.split_tickers_holdout(list(TICKER_VALUE.keys()), SECTOR_LOOKUP, frac, seed)
        group = holdout if want_holdout else tune
        group_trades = [t for t in TRADES if t["ticker"] in group]
        summaries.append(fake_summarize(group_trades))
    return summaries


def test_averages_sharpe_like_correctly_across_seeds():
    seeds = [1, 2, 3]
    _, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.25, seeds, fake_summarize)
    expected = _expected_bucket(seeds, 0.25, want_holdout=True)
    expected_mean = round(sum(s["sharpe_like"] for s in expected) / len(expected), 4)
    assert holdout_avg["sharpe_like"] == expected_mean


def test_tune_and_holdout_are_independently_averaged():
    seeds = [1, 2, 3]
    tune_avg, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.25, seeds, fake_summarize)
    expected_tune = _expected_bucket(seeds, 0.25, want_holdout=False)
    expected_tune_mean = round(sum(s["sharpe_like"] for s in expected_tune) / len(expected_tune), 4)
    assert tune_avg["sharpe_like"] == expected_tune_mean
    # TUNE (6 tickers) should never equal HOLDOUT (2 tickers) here -- sanity check
    # the two buckets aren't accidentally computed from the same trades.
    assert tune_avg["trade_count"] != holdout_avg["trade_count"]


def test_none_values_are_skipped_not_treated_as_zero():
    # annualized_sharpe_like is always None from fake_summarize() -- averaging
    # must not crash, and must report None (not 0.0) when every value is None.
    seeds = [1, 2, 3]
    tune_avg, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.25, seeds, fake_summarize)
    assert tune_avg["annualized_sharpe_like"] is None
    assert holdout_avg["annualized_sharpe_like"] is None


def test_min_max_spread_reported():
    seeds = [1, 2, 3]
    _, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.25, seeds, fake_summarize)
    expected = _expected_bucket(seeds, 0.25, want_holdout=True)
    expected_values = [s["sharpe_like"] for s in expected]
    assert holdout_avg["_sharpe_like_min"] == round(min(expected_values), 4)
    assert holdout_avg["_sharpe_like_max"] == round(max(expected_values), 4)
    assert holdout_avg["_seeds_used"] == 3


def test_single_seed_min_equals_max():
    _, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.25, [42], fake_summarize)
    assert holdout_avg["_sharpe_like_min"] == holdout_avg["_sharpe_like_max"] == holdout_avg["sharpe_like"]


def test_zero_holdout_frac_gives_empty_holdout_every_seed():
    _, holdout_avg = optimize.average_holdout_summary(TRADES, SECTOR_LOOKUP, 0.0, [1, 2, 3], fake_summarize)
    assert holdout_avg["trade_count"] == 0.0
    assert holdout_avg["sharpe_like"] is None
