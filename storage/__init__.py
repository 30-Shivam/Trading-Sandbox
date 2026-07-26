"""MongoDB persistence layer: Trade_Signals (read/write), Trade_Outcomes and
System_Config (schema/indexes defined, read/write logic lands in later
phases). Kept separate from `swingtrade` so the pure calculation library
never needs pymongo installed to be reused elsewhere (e.g. inside an Optuna
worker sandbox).
"""

from . import outcomes, system_config
from .mongo import MongoNotConfigured, get_client, get_db
from .signals import log_trade_signal, log_trade_signals

__all__ = [
    "MongoNotConfigured",
    "get_client",
    "get_db",
    "log_trade_signal",
    "log_trade_signals",
    "ensure_indexes",
]


def ensure_indexes() -> None:
    """Create every collection's indexes. Idempotent -- safe to call on
    every process startup."""
    from . import signals

    signals.ensure_indexes()
    outcomes.ensure_indexes()
    system_config.ensure_indexes()
