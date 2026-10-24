"""Walk-Forward backtesting harness (Phase 4). Pure -- takes pre-fetched
historical OHLCV DataFrames in, returns simulated trades and fold-level
metrics out. Reuses compute_levels / add_trade_score / settle_trade /
is_market_uptrend exactly as the live dashboard and nightly settlement job
do, so a backtested TradingConfig is scored by identical rules to a live
signal -- that's the whole point of Phase 1 having pulled this math out of
the Streamlit app in the first place.

Walk-Forward Optimization, not a single static window: a backtest over one
fixed historical range and calling the winner "optimal" is a curve-fitting
trap -- it rewards parameters that memorized that period's noise. Instead,
generate_folds() produces a sequence of rolling in-sample/out-of-sample
windows; Optuna (Phase 5) will score a candidate config on the AGGREGATE
out-of-sample performance across all folds, never a single fold's in-sample
fit. See ARCHITECTURE_PLAN.md amendment #3 for the full rationale.

No look-ahead: on each simulated day `as_of`, only bars up to and including
`as_of` are used to decide whether a signal fires (mirrors the live app,
where the last row of fetched data IS "today"). Settling a trade started on
`as_of` is free to use bars strictly after `as_of` -- grading a past
decision against realized future prices is not look-ahead bias, it's just
how you find out whether the trade won.

Entry timing is realistic, not instantaneous: a signal computed from
`as_of`'s close can't be acted on until the NEXT session at the earliest,
so simulate_signals() no longer assumes entry on the same bar the signal
fired. _find_entry_fill() walks forward looking for the resting limit order
at Buy_Price to actually get touched (gap-aware, same rules as stop/target
fills), within config.max_entry_wait_days -- a signal that's never touched
produces no trade at all, which is the honest outcome for a limit order
nobody would leave open forever.

Catalyst awareness IS simulated, given an optional per-ticker
`earnings_dates` (see run_backtest.fetch_earnings_dates): yfinance's
get_earnings_dates(limit=40) returns roughly a decade of REPORTED earnings
dates, not just upcoming estimates. Only the calendar date is used, never
the EPS/Surprise columns -- once a company schedules a report, the date
itself is a fixed fact that doesn't get revised after the fact the way an
EPS estimate does, so looking up "the next earnings date after `as_of`"
from that static list introduces no look-ahead leakage even for a backtest
day years in the past. Without `earnings_dates` passed in, Catalyst_Warning
is always False (the old behavior) rather than raising -- callers that
don't need catalyst simulation aren't forced to fetch it.

Pooled trades are NOT independent samples, and summarize_trades()'s
mean/std treats them as if they were. A real, observed failure mode: 30+
correlated Technology signals firing the same day during one sector move
aren't 30 independent bets, they're one correlated event counted 30 times,
which understates the true variance of pooled pnl_pct and overstates
sharpe_like's apparent confidence. compute_cluster_weights() (given a
sector_lookup) gives each (entry_date, sector) cluster a combined weight of
1.0 split evenly across its members -- a conservative correction (assumes
perfect intra-cluster correlation, zero inter-cluster), not an exact one,
but a real one. See optimize.py's summarize_weighted() for where this
combines with the existing recency weighting.
"""

from .core import (
    ENTRY_SIGNALS,
    LOOKBACK_BUFFER_BARS,
    Fold,
    generate_folds,
    _next_earnings_date,
    _find_entry_fill,
    _find_breakout_fill,
    _find_next_open_fill,
    _settle,
)  # noqa: F401
from .signals_technical import (
    simulate_signals,
    simulate_random_entries,
    simulate_breakout_signals,
    simulate_random_breakout_entries,
    simulate_pullback_signals,
    simulate_random_pullback_entries,
    simulate_breakout_retest_signals,
    simulate_random_breakout_retest_entries,
    simulate_week52_signals,
    simulate_random_week52_entries,
    simulate_momentum_burst_signals,
    simulate_random_momentum_burst_entries,
    simulate_squeeze_breakout_signals,
    simulate_random_squeeze_breakout_entries,
)  # noqa: F401
from .signals_cross_sectional import (
    simulate_pairs_signals,
    simulate_random_pairs_entries,
    simulate_momentum_signals,
    simulate_random_momentum_entries,
    simulate_lowvol_signals,
    simulate_random_lowvol_entries,
    simulate_sector_rotation_signals,
    simulate_random_sector_rotation_entries,
)  # noqa: F401
from .signals_fundamental import (
    simulate_quality_signals,
    simulate_value_signals,
    simulate_random_value_entries,
    simulate_accruals_signals,
    simulate_random_accruals_entries,
    simulate_cash_profitability_signals,
    simulate_random_cash_profitability_entries,
    simulate_random_quality_entries,
    simulate_insider_buying_signals,
    simulate_random_insider_buying_entries,
    simulate_analyst_revision_signals,
    simulate_random_analyst_revision_entries,
    simulate_pead_signals,
    simulate_random_pead_entries,
)  # noqa: F401
from .signals_other import (
    simulate_llm_strategy_signals,
    simulate_random_llm_strategy_entries,
    simulate_adx_trend_entry_signals,
    simulate_random_adx_trend_entry_entries,
    simulate_ma_crossover_signals,
    simulate_random_ma_crossover_entries,
)  # noqa: F401
from .engine import (
    run_backtest,
    run_random_backtest,
)  # noqa: F401
from .stats import (
    _annualize_sharpe,
    summarize_trades,
    compute_cluster_weights,
    summarize_trades_weighted,
    summarize_by_catalyst,
    summarize_by_period,
    VOLATILITY_REGIME_ROLLING_WINDOW_DAYS,
    VOLATILITY_REALIZED_WINDOW_DAYS,
    compute_volatility_regime_series,
    summarize_by_volatility_regime,
    _concurrency_at_entry,
    _batched_weighted_returns,
    compute_max_drawdown,
    _exit_date_batch_layout,
    monte_carlo_drawdown,
    permutation_test_gap,
    compute_k_ratio,
)  # noqa: F401
from .portfolio_audits import (
    simulate_sector_rotation_rebalanced,
    _close_rotation_trade,
    compute_equal_weight_buy_and_hold,
    simulate_portfolio_constrained,
    audit_dividend_drag,
    compute_strategy_correlation,
    audit_same_ticker_overlap,
    audit_capital_capacity,
    flatten_out_sample_trades,
)  # noqa: F401
from .walkforward import (
    FoldResult,
    _run_fold_sequential,
    _worker_ticker_data,
    _worker_market_data,
    _worker_sector_data,
    _worker_pair_price_panels,
    _worker_momentum_rank_frame,
    _worker_yield_curve,
    _worker_skew_regime,
    _init_worker,
    _run_fold_worker,
    run_walk_forward,
)  # noqa: F401
from .index_gate import (
    audit_vs_index,
    INDEX_GATE_MIN_EXCESS_PCT,
    INDEX_GATE_MIN_T,
    index_gate_verdict,
)  # noqa: F401
