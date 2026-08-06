"""Load the active TradingConfig from MongoDB's System_Config, falling back
to swingtrade.DEFAULT_CONFIG when Mongo is unreachable, unconfigured, or
nothing has been promoted yet. Shared by the interactive dashboard
(dip_buy_analyzer.py, which wraps this in st.cache_data on a TTL) and the
standalone scheduled scan (ingest.py, which calls it once per run) so both
always agree on which config is "live" -- see ARCHITECTURE_PLAN.md Phase 6/7.
"""

import storage
import swingtrade


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
