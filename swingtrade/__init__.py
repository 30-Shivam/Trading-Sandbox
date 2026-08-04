"""Pure trading calculation library: no network calls, no UI. Shared by the
Streamlit dashboard, the nightly settlement job, the backtester, and the
Optuna learning engine so they all score trades identically.
"""

from .allocation import allocate_capital, size_by_risk
from .backtest import (
    Fold,
    FoldResult,
    compute_cluster_weights,
    generate_folds,
    run_backtest,
    run_walk_forward,
    simulate_breakout_signals,
    simulate_random_breakout_entries,
    simulate_random_entries,
    simulate_signals,
    summarize_by_catalyst,
    summarize_trades,
    summarize_trades_weighted,
)
from .config import DEFAULT_CONFIG, TradingConfig
from .levels import compute_breakout_levels, compute_levels, is_market_uptrend, review_holding
from .scoring import add_breakout_trade_score, add_trade_score, signal_for_score
from .settlement import settle_trade

__all__ = [
    "TradingConfig",
    "DEFAULT_CONFIG",
    "compute_levels",
    "compute_breakout_levels",
    "is_market_uptrend",
    "review_holding",
    "add_trade_score",
    "add_breakout_trade_score",
    "signal_for_score",
    "allocate_capital",
    "size_by_risk",
    "settle_trade",
    "Fold",
    "FoldResult",
    "generate_folds",
    "simulate_signals",
    "simulate_random_entries",
    "simulate_breakout_signals",
    "simulate_random_breakout_entries",
    "run_backtest",
    "summarize_trades",
    "summarize_trades_weighted",
    "summarize_by_catalyst",
    "compute_cluster_weights",
    "run_walk_forward",
]
