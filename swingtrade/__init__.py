"""Pure trading calculation library: no network calls, no UI. Shared by the
Streamlit dashboard, the nightly settlement job, the backtester, and the
Optuna learning engine so they all score trades identically.
"""

from .allocation import allocate_capital
from .config import DEFAULT_CONFIG, TradingConfig
from .levels import compute_levels, is_market_uptrend
from .scoring import add_trade_score, signal_for_score
from .settlement import settle_trade

__all__ = [
    "TradingConfig",
    "DEFAULT_CONFIG",
    "compute_levels",
    "is_market_uptrend",
    "add_trade_score",
    "signal_for_score",
    "allocate_capital",
    "settle_trade",
]
