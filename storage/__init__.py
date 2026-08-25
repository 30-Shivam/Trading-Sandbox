"""MongoDB persistence layer: Trade_Signals, Trade_Outcomes, and
System_Config, all with read/write logic now. Kept separate from
`swingtrade` so the pure calculation library never needs pymongo installed
to be reused elsewhere (e.g. inside an Optuna worker sandbox).
"""

from . import holdings, outcomes, position_review_state, research_journal, signals, system_config
from .holdings import get_holdings, set_holdings
from .mongo import MongoNotConfigured, get_client, get_db
from .outcomes import log_trade_outcome
from .position_review_state import (
    get_position_review_state,
    prune_position_review_state,
    set_position_review_state,
)
from .research_journal import get_recent_cycles, write_cycle
from .signals import (
    LOOSENED_RESEARCH_TIER,
    confirm_fill,
    get_signals_pending_confirmation,
    get_unsettled_signals,
    log_trade_signal,
    log_trade_signals,
    mark_settled,
    record_user_decision,
    unconfirm_fill,
)
from .system_config import (
    get_active_config,
    get_active_config_doc,
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
    "LOOSENED_RESEARCH_TIER",
    "log_trade_signal",
    "log_trade_signals",
    "get_unsettled_signals",
    "mark_settled",
    "confirm_fill",
    "unconfirm_fill",
    "get_signals_pending_confirmation",
    "record_user_decision",
    "log_trade_outcome",
    "get_active_config",
    "get_active_config_doc",
    "get_next_version",
    "write_candidate",
    "list_configs",
    "get_config_by_version",
    "promote_candidate",
    "get_holdings",
    "set_holdings",
    "write_cycle",
    "get_recent_cycles",
    "get_position_review_state",
    "set_position_review_state",
    "prune_position_review_state",
    "ensure_indexes",
]


def ensure_indexes() -> None:
    """Create every collection's indexes. Idempotent -- safe to call on
    every process startup."""
    signals.ensure_indexes()
    outcomes.ensure_indexes()
    system_config.ensure_indexes()
    holdings.ensure_indexes()
    research_journal.ensure_indexes()
    position_review_state.ensure_indexes()
