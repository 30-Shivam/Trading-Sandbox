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
SECONDARY_STRATEGY_VERSIONS = {
    "Breakout Retest": 27,
    "52-Week High": 28,
    "Squeeze Breakout": 39,
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
EXPERIMENTAL_STRATEGY_VERSIONS = {
    "Momentum Burst": 38,
    "ADX Trend Entry": 40,
}


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
