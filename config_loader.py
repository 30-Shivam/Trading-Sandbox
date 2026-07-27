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
