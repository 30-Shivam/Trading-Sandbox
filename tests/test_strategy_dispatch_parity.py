"""Guards against the exact bug class that silently broke ingest.py's
"Squeeze Breakout"/"MA Crossover" secondary scans: dip_buy_analyzer.py and
ingest.py each maintain their OWN copy of the strategy -> add_*_trade_score
dispatch function, and ingest.py's copy was never updated when
squeeze_breakout/ma_crossover were promoted into
config_loader.SECONDARY_STRATEGY_VERSIONS -- any day either strategy had a
real signal, it crashed with KeyError: 'Oversold_Streak_Days' (falling
through to the RSI-only scorer) before ever reaching MongoDB. See
improvements.txt for the incident.

For every strategy this codebase supports, builds one REAL, schema-complete
levels row (via the actual swingtrade.compute_*_levels() functions against
a synthetic-but-realistic OHLCV fixture -- see conftest.py -- not a
hand-typed dict, so the row's columns can't silently drift from what
production actually produces) and asserts BOTH dispatch copies (a) don't
raise and (b) return the identical Trade_Score for it.

2026-08-29: this test's own STRATEGIES list was ITSELF found stale --
"pairs" and "momentum_rank" both had real dispatch code in both
_score_for_strategy() copies (ingest.py's momentum_rank branch was in fact
MISSING and got fixed the same session, see improvements.txt item 98) but
neither was covered here, because STRATEGIES was a second hand-maintained
list that could silently drift from COMPUTE_LEVELS_FN just like the two
real dispatch copies drifted from each other. Fixed at the root:
STRATEGIES is now DERIVED from COMPUTE_LEVELS_FN's own keys (one list, not
two to keep in sync) -- see test_every_live_strategy_has_dispatch_coverage()
below for the second half of the fix: a live check that anything actually
wired into config_loader's real strategy-version dicts has a
COMPUTE_LEVELS_FN entry at all, so a FUTURE new strategy can't silently
fall through this exact gap again the way momentum_rank just did.
"""
import pandas as pd
import pytest

import config_loader
import storage
import swingtrade
import dip_buy_analyzer
import ingest

COMPUTE_LEVELS_FN = {
    "rsi": swingtrade.compute_levels,
    "breakout": swingtrade.compute_breakout_levels,
    "pullback": swingtrade.compute_pullback_levels,
    "breakout_retest": swingtrade.compute_breakout_retest_levels,
    "week52_high": swingtrade.compute_week52_levels,
    "momentum_burst": swingtrade.compute_momentum_burst_levels,
    "squeeze_breakout": swingtrade.compute_squeeze_breakout_levels,
    "adx_trend_entry": swingtrade.compute_adx_trend_entry_levels,
    "ma_crossover": swingtrade.compute_ma_crossover_levels,
    "pairs": swingtrade.compute_pairs_levels,
    "momentum_rank": swingtrade.compute_momentum_levels,
}
STRATEGIES = list(COMPUTE_LEVELS_FN.keys())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_dispatch_parity(strategy, uptrend_ohlcv):
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy})
    levels = COMPUTE_LEVELS_FN[strategy]("TEST", uptrend_ohlcv, config)
    df = pd.DataFrame([levels])

    dashboard_result = dip_buy_analyzer._score_for_strategy(df.copy(), config)
    ingest_result = ingest._score_for_strategy(df.copy(), config)

    assert "Trade_Score" in dashboard_result.columns, f"{strategy}: dashboard dispatch produced no Trade_Score column"
    assert "Trade_Score" in ingest_result.columns, f"{strategy}: ingest dispatch produced no Trade_Score column"
    assert dashboard_result["Trade_Score"].iloc[0] == ingest_result["Trade_Score"].iloc[0], (
        f"{strategy}: dip_buy_analyzer._score_for_strategy and ingest._score_for_strategy disagree -- "
        f"dashboard={dashboard_result['Trade_Score'].iloc[0]}, ingest={ingest_result['Trade_Score'].iloc[0]} "
        "(the two dispatch copies have drifted out of sync -- see this module's own docstring)"
    )


def _mongo_available() -> bool:
    try:
        storage.get_db()
        return True
    except Exception:
        return False


def _live_strategy_labels_and_versions() -> list[tuple[str, int]]:
    """Every strategy actually wired live right now, as (label, version)
    pairs -- config_loader.SECONDARY_STRATEGY_VERSIONS +
    EXPERIMENTAL_STRATEGY_VERSIONS, the authoritative "what's actually
    scanned/displayed today" source (same dicts ingest.py/dip_buy_analyzer.py
    themselves iterate). Deliberately does NOT include the primary slot --
    load_active_config() always succeeds (falls back to DEFAULT_CONFIG,
    strategy="rsi") even with no real active document, so it can't signal
    "nothing is live" the way a missing secondary/experimental version can."""
    return [
        (label, version)
        for label, version in {**config_loader.SECONDARY_STRATEGY_VERSIONS, **config_loader.EXPERIMENTAL_STRATEGY_VERSIONS}.items()
    ]


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_every_live_strategy_has_dispatch_coverage():
    """The other half of this module's own 2026-08-29 fix (see its
    docstring): STRATEGIES/COMPUTE_LEVELS_FN being internally consistent
    with EACH OTHER (fixed above) doesn't guarantee either of them stays in
    sync with what's actually LIVE -- a future strategy could get added to
    config_loader.SECONDARY_STRATEGY_VERSIONS/EXPERIMENTAL_STRATEGY_VERSIONS
    (wiring it into real daily scans) without anyone remembering to also add
    it to COMPUTE_LEVELS_FN here, silently reproducing the exact
    momentum_rank gap this session found and fixed. Resolves each live
    (label, version) to its REAL config.strategy field (not the display
    label -- e.g. "RSI Mean-Reversion" resolves to "rsi") via
    config_loader.load_config_by_version(), the same real Mongo documents
    production actually loads, then asserts each one is a COMPUTE_LEVELS_FN
    key. Skips (not fails) if MONGODB_URI isn't available, same convention
    as test_mongo_indexes.py."""
    missing = []
    for label, version in _live_strategy_labels_and_versions():
        config, source = config_loader.load_config_by_version(version)
        if config is None:
            pytest.fail(f"{label} (v{version}): could not load its own System_Config -- {source}")
        if config.strategy not in COMPUTE_LEVELS_FN:
            missing.append(f"{label} (v{version}) -> strategy={config.strategy!r}")
    assert not missing, (
        "Live strategy/strategies with no dispatch-parity test coverage at all: "
        f"{missing} -- add an entry to this module's own COMPUTE_LEVELS_FN, the same gap "
        "momentum_rank fell through before this test existed (see improvements.txt item 98)."
    )
