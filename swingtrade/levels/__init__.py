"""Pure technical-level calculations. No network calls, no UI -- takes
already-fetched OHLCV data (and already-resolved catalyst info) in, returns
plain dicts/tuples out. Safe to call from Streamlit, the settlement job, or a
backtest loop replaying years of historical bars.
"""

from .base import (
    is_market_uptrend,
    market_uptrend_from_frame,
    precompute_rsi_frame,
    levels_from_rsi_frame,
    compute_levels,
    compute_relative_strength,
    review_holding,
)  # noqa: F401
from .technical import (
    precompute_breakout_frame,
    breakout_levels_from_frame,
    compute_breakout_levels,
    precompute_pullback_frame,
    pullback_levels_from_frame,
    compute_pullback_levels,
    precompute_breakout_retest_frame,
    breakout_retest_levels_from_frame,
    compute_breakout_retest_levels,
    precompute_week52_frame,
    week52_levels_from_frame,
    compute_week52_levels,
    precompute_momentum_burst_frame,
    momentum_burst_levels_from_frame,
    compute_momentum_burst_levels,
    precompute_squeeze_breakout_frame,
    squeeze_breakout_levels_from_frame,
    compute_squeeze_breakout_levels,
)  # noqa: F401
from .cross_sectional import (
    precompute_pairs_frame,
    pairs_levels_from_frame,
    compute_pairs_levels,
    compute_momentum_rank_frame,
    precompute_momentum_frame,
    momentum_levels_from_frame,
    compute_momentum_levels,
    compute_lowvol_rank_frame,
    precompute_lowvol_frame,
    lowvol_levels_from_frame,
    compute_lowvol_levels,
    compute_sector_rotation_rank_frame,
    precompute_sector_rotation_frame,
    sector_rotation_levels_from_frame,
    compute_sector_rotation_levels,
)  # noqa: F401
from .fundamental import (
    compute_quality_rank_frame,
    precompute_quality_frame,
    quality_levels_from_frame,
    compute_quality_levels,
    compute_value_rank_frame,
    precompute_value_frame,
    value_levels_from_frame,
    compute_value_levels,
    compute_accruals_rank_frame,
    precompute_accruals_frame,
    accruals_levels_from_frame,
    compute_accruals_levels,
    compute_cash_profitability_rank_frame,
    precompute_cash_profitability_frame,
    cash_profitability_levels_from_frame,
    compute_cash_profitability_levels,
    classify_insider_transaction,
    precompute_insider_buying_frame,
    insider_buying_levels_from_frame,
    compute_insider_buying_levels,
    precompute_analyst_revision_frame,
    analyst_revision_levels_from_frame,
    compute_analyst_revision_levels,
    precompute_pead_frame,
    pead_levels_from_frame,
    compute_pead_levels,
)  # noqa: F401
from .llm_rules import (
    KNOWN_INDICATOR_FIELDS,
    _ALLOWED_CONDITION_OPS,
    evaluate_llm_rule_conditions,
    rule_exit_to_config,
    precompute_llm_strategy_frame,
    llm_strategy_levels_from_frame,
    compute_llm_strategy_levels,
)  # noqa: F401
from .trend import (
    precompute_adx_trend_entry_frame,
    adx_trend_entry_levels_from_frame,
    compute_adx_trend_entry_levels,
    SKEW_REGIME_ROLLING_WINDOW_DAYS,
    precompute_ma_crossover_frame,
    ma_crossover_levels_from_frame,
    compute_ma_crossover_levels,
)  # noqa: F401
from .audits import (
    audit_no_lookahead,
    audit_data_quality,
    sanitize_ohlcv,
)  # noqa: F401
