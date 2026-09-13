"""benchmark_random_entry._average_by_key() -- the BY YEAR/BY VOLATILITY
REGIME counterpart to optimize.average_summaries() (2026-09-13,
improvements.txt item 141), extending multi-seed RANDOM-baseline
averaging (previously only the ALL-TICKERS headline, per item 132) to
every other report section. Averages a list of {key: summary} dicts (one
per random-baseline seed) into one {key: averaged_summary}, handling a
key missing from some seeds gracefully (e.g. a calendar year with zero
random trades under an unlucky seed).
"""
from benchmark_random_entry import _average_by_key


def _summary(sharpe, trade_count):
    return {"sharpe_like": sharpe, "trade_count": trade_count, "k_ratio": None}


def test_averages_a_key_present_in_every_seed():
    per_seed = [
        {2023: _summary(0.1, 10), 2024: _summary(0.3, 20)},
        {2023: _summary(0.3, 12), 2024: _summary(0.5, 18)},
    ]
    result = _average_by_key(per_seed)
    assert result[2023]["sharpe_like"] == 0.2
    assert result[2023]["trade_count"] == 11.0
    assert result[2024]["sharpe_like"] == 0.4
    assert result[2023]["_seeds_used"] == 2


def test_key_missing_from_some_seeds_averages_over_whichever_have_it():
    # 2022 only appears in the first seed's dict (e.g. the second seed's
    # random draw happened to produce zero trades that year) -- should
    # average over just the 1 seed that has it, not be excluded or
    # treated as a 0.
    per_seed = [
        {2022: _summary(0.5, 5), 2023: _summary(0.1, 10)},
        {2023: _summary(0.3, 12)},
    ]
    result = _average_by_key(per_seed)
    assert result[2022]["sharpe_like"] == 0.5
    assert result[2022]["_seeds_used"] == 1
    assert result[2023]["sharpe_like"] == 0.2
    assert result[2023]["_seeds_used"] == 2


def test_empty_list_returns_empty_dict():
    assert _average_by_key([]) == {}


def test_works_for_volatility_regime_style_string_keys_too():
    per_seed = [
        {"elevated": _summary(0.2, 10), "normal": _summary(0.1, 15)},
        {"elevated": _summary(0.4, 8), "normal": _summary(0.3, 20)},
    ]
    result = _average_by_key(per_seed)
    assert set(result.keys()) == {"elevated", "normal"}
    assert result["elevated"]["sharpe_like"] == 0.3
