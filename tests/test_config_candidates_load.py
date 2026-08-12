"""Read-only: every version referenced in config_loader.SECONDARY_STRATEGY_VERSIONS
(plus the active config) actually loads into a valid TradingConfig from
Mongo -- catches a candidate document being deleted/corrupted, or a
version number pointing at nothing, before a scheduled run discovers it
the hard way. Skips (not fails) if MONGODB_URI isn't available.
"""
import pytest

import config_loader


def _mongo_available() -> bool:
    try:
        import storage
        storage.get_db()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")


def test_active_config_loads():
    config, source = config_loader.load_active_config()
    assert config is not None
    assert config.strategy, f"active config loaded with an empty strategy field ({source})"


@pytest.mark.parametrize("label,version", list(config_loader.SECONDARY_STRATEGY_VERSIONS.items()))
def test_secondary_config_loads(label, version):
    config, source = config_loader.load_config_by_version(version)
    assert config is not None, f"{label} (v{version}) failed to load: {source}"
    assert config.strategy, f"{label} (v{version}) loaded with an empty strategy field"


@pytest.mark.parametrize("label,version", list(config_loader.EXPERIMENTAL_STRATEGY_VERSIONS.items()))
def test_experimental_config_loads(label, version):
    config, source = config_loader.load_config_by_version(version)
    assert config is not None, f"{label} (v{version}) failed to load: {source}"
    assert config.strategy, f"{label} (v{version}) loaded with an empty strategy field"
