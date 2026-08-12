"""Load the active TradingConfig from MongoDB's System_Config, falling back
to swingtrade.DEFAULT_CONFIG when Mongo is unreachable, unconfigured, or
nothing has been promoted yet. Shared by the interactive dashboard
(dip_buy_analyzer.py, which wraps this in st.cache_data on a TTL) and the
standalone scheduled scan (ingest.py, which calls it once per run) so both
always agree on which config is "live" -- see ARCHITECTURE_PLAN.md Phase 6/7.
"""

import storage
import swingtrade

# Fixed, immutable candidate System_Config versions for the validated
# secondary strategies shown alongside the active config -- see
# improvements.txt items 27/28. Single source of truth shared by
# dip_buy_analyzer.py (the dashboard's secondary sections) and ingest.py
# (the headless scheduled scan) so the two can never silently drift apart
# on which candidate versions are "the" secondary strategies.
#
# Breakout Retest (v27) and 52-Week High (v28) removed 2026-08-11 per
# explicit user request ("remove all the losing strategies off the
# dashboard") -- both strategies' apparent edges evaporated once forced
# into any RRR shape that could actually clear the live scoring ceiling
# (v44/v45, improvements.txt item 42/43): win rates collapsed to
# statistically indistinguishable from DEFAULT_CONFIG's own unfiltered
# baseline, regardless of which end of the RRR moved. v27/v28 themselves
# (at their original, non-fireable RRR) are UNCHANGED in Mongo -- this is
# purely a dashboard/capital-eligibility removal, not a data deletion; the
# strategies could be re-added here if a genuine refinement is ever found.
#
# MA Crossover (v49) promoted 2026-08-11 per explicit user request
# ("promote this to capital eligible as well") -- the first strategy since
# squeeze_breakout to clear the full validation pipeline: real-vs-random
# benchmark passed on all 3 cuts at untuned defaults (item 49), its own
# core-trigger parameters (short/long window) refined with nothing beating
# baseline (item 50), entry-fill sensitivity checked under both limit/
# next_open (edge holds direction under both, doesn't reverse like
# momentum_burst's did), RRR-vs-scoring-ceiling confirmed compatible
# (clears signal_buy_threshold, same ceiling shape as active v43), and a
# live smoke test against the real watchlist confirmed clean dispatch +
# real per-ticker score differentiation. See improvements.txt item 51.
SECONDARY_STRATEGY_VERSIONS = {
    "Squeeze Breakout": 39,
    "MA Crossover": 49,
}

# Fixed candidate System_Config versions for EXPERIMENTAL strategies --
# deliberately a SEPARATE dict from SECONDARY_STRATEGY_VERSIONS above, not
# merged into it. This separation is the actual mechanism that keeps an
# experimental strategy out of ingest.py's automation loop (which only
# ever iterates SECONDARY_STRATEGY_VERSIONS) and out of any capital-
# allocation path -- see dip_buy_analyzer.py's "Daily Signals" tab and
# improvements.txt item 34/35. A strategy only moves from here to
# SECONDARY_STRATEGY_VERSIONS once it's been promoted after passing
# validation, the same graduation breakout_retest/week52_high went
# through -- squeeze_breakout (v39) graduated 2026-08-10 per explicit user
# request (improvements.txt item 43/44): the only Daily Signals candidate
# that beat its own random-entry baseline on ALL/TUNE/HOLDOUT, now made
# capital-eligible with its own secondary dashboard section.
#
# Momentum Burst (v38) and ADX Trend Entry (v40) removed 2026-08-11, same
# "remove all the losing strategies" request -- both lost to their own
# random-entry baseline on HOLDOUT in the real 5-year benchmark comparison
# run 2026-08-09/10 (momentum_burst additionally flat/negative on every
# cut). Empty for now -- nothing currently in the Daily Signals tab until
# a genuine replacement is found; the tab itself still renders (just shows
# nothing experimental), and the underlying strategy code/Mongo candidates
# are untouched, only removed from this dict.
EXPERIMENTAL_STRATEGY_VERSIONS = {}


def load_active_config() -> tuple[swingtrade.TradingConfig, str]:
    try:
        doc = storage.get_active_config_doc()
    except storage.MongoNotConfigured:
        return swingtrade.DEFAULT_CONFIG, "MongoDB not configured -- using built-in defaults."
    except Exception as exc:
        return swingtrade.DEFAULT_CONFIG, f"Could not reach MongoDB ({exc}) -- using built-in defaults."

    if doc is None:
        return swingtrade.DEFAULT_CONFIG, "No active System_Config yet -- using built-in defaults."

    try:
        config = swingtrade.TradingConfig.from_dict(doc["params"])
    except Exception as exc:
        return swingtrade.DEFAULT_CONFIG, f"Active System_Config failed to parse ({exc}) -- using built-in defaults."

    return config, f"Using System_Config v{doc['version']} (active)."


def load_config_by_version(version: int) -> tuple[swingtrade.TradingConfig | None, str]:
    """Load one specific, fixed System_Config document by version number --
    for secondary/candidate strategies shown alongside the active config
    (see dip_buy_analyzer.py's breakout_retest/week52_high sections), NOT
    a substitute for load_active_config(). Unlike load_active_config(),
    which always falls back to a sensible default (no active config is a
    normal, expected state), a missing/unparseable SPECIFIC version is a
    genuine error worth surfacing distinctly -- silently substituting
    DEFAULT_CONFIG (strategy="rsi") for, say, a missing breakout_retest
    candidate would display a nonsensical "Breakout Retest" section
    actually running RSI logic. Returns `(None, reason)` on any failure;
    callers should skip rendering that section rather than fall back."""
    try:
        doc = storage.get_config_by_version(version)
    except storage.MongoNotConfigured:
        return None, "MongoDB not configured."
    except Exception as exc:
        return None, f"Could not reach MongoDB ({exc})."

    if doc is None:
        return None, f"No System_Config document with version={version}."

    try:
        config = swingtrade.TradingConfig.from_dict(doc["params"])
    except Exception as exc:
        return None, f"System_Config v{version} failed to parse ({exc})."

    return config, f"Using System_Config v{version} ({doc.get('status', 'unknown status')})."
