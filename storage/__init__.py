"""MongoDB persistence layer: Trade_Signals, Trade_Outcomes, and
System_Config, all with read/write logic now. Kept separate from
`swingtrade` so the pure calculation library never needs pymongo installed
to be reused elsewhere (e.g. inside an Optuna worker sandbox).
"""

from . import outcomes, signals, system_config
from .mongo import MongoNotConfigured, get_client, get_db
from .outcomes import log_trade_outcome
from .signals import get_unsettled_signals, log_trade_signal, log_trade_signals, mark_settled
from .system_config import (
    get_active_config,
    get_config_by_version,
    get_next_version,
    list_configs,
    promote_candidate,
    write_candidate,
)

__all__ = [
    "MongoNotConfigured",
    "get_client",
    "get_db",
    "log_trade_signal",
    "log_trade_signals",
    "get_unsettled_signals",
    "mark_settled",
    "log_trade_outcome",
    "get_active_config",
    "get_next_version",
    "write_candidate",
    "list_configs",
    "get_config_by_version",
    "promote_candidate",
    "ensure_indexes",
]


def ensure_indexes() -> None:
    """Create every collection's indexes. Idempotent -- safe to call on
    every process startup."""
    signals.ensure_indexes()
    outcomes.ensure_indexes()
    system_config.ensure_indexes()
