"""Pure trading calculation library: no network calls, no UI. Shared by the
Streamlit dashboard, the nightly settlement job, the backtester, and the
Optuna learning engine so they all score trades identically.
"""

from .allocation import allocate_capital
from .backtest import (
    Fold,
    FoldResult,
    compute_cluster_weights,
    generate_folds,
    run_backtest,
    run_walk_forward,
    simulate_signals,
    summarize_by_catalyst,
    summarize_trades,
    summarize_trades_weighted,
)
from .config import DEFAULT_CONFIG, TradingConfig
from .levels import compute_levels, is_market_uptrend, review_holding
from .scoring import add_trade_score, signal_for_score
from .settlement import settle_trade

__all__ = [
    "TradingConfig",
    "DEFAULT_CONFIG",
    "compute_levels",
    "is_market_uptrend",
    "review_holding",
    "add_trade_score",
    "signal_for_score",
    "allocate_capital",
    "settle_trade",
    "Fold",
    "FoldResult",
    "generate_folds",
    "simulate_signals",
    "run_backtest",
    "summarize_trades",
    "summarize_trades_weighted",
    "summarize_by_catalyst",
    "compute_cluster_weights",
    "run_walk_forward",
]
