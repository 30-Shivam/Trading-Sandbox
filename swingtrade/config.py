"""Tunable trading parameters, decoupled from any fetch/UI/storage concern.

`TradingConfig` is the single object every pure calculation function in this
package takes as a parameter instead of reading module-level globals. That's
what lets the same math run identically from Streamlit, the nightly settlement
job, the backtester, and the Optuna objective function -- and it's the shape
Phase 6 will eventually load from MongoDB's System_Config collection (hence
`to_dict`/`from_dict` below).
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TradingConfig:
    # Structural support / moving-average context
    ma_window: int = 20                    # SMA window (trading days), context only
    support_lookback_days: int = 20        # window to scan for the structural swing low
    ma_discount_pct: float = 0.05          # 5% below the SMA, context only

    # RSI / ATR
    rsi_window: int = 14                   # RSI lookback (trading days)
    atr_window: int = 14                   # ATR lookback (trading days)
    rsi_oversold_threshold: float = 45     # buy signal requires RSI below this
    atr_take_profit_multiplier: float = 1.5  # sell_price = buy_price + multiplier * ATR
    stop_loss_atr_multiplier: float = 1.0    # stop_loss = buy_price - multiplier * ATR

    # Macro trend / liquidity gates
    sma_trend_window: int = 200            # macro trend filter window (trading days)
    volume_lookback_days: int = 20         # window for average volume / liquidity check
    min_dollar_volume: float = 5_000_000   # exclude tickers below this 20d $ volume

    # Catalyst awareness
    earnings_warning_days: int = 14        # flag Catalyst_Warning if earnings within N days

    # Trade_Score weights (should sum to 100)
    rrr_score_weight: float = 40           # points for Risk-to-Reward Ratio
    rrr_score_cap: float = 4.0             # RRR at/above this earns full RRR points
    rsi_score_weight: float = 40           # points for RSI (oversold-ness)
    rsi_score_floor: float = 30            # RSI at/below this earns full RSI points
    rsi_score_ceiling: float = 60          # RSI at/above this earns zero RSI points
    distance_score_weight: float = 20      # points for proximity to the buy trigger
    distance_score_cap_pct: float = 20     # distance at/above this earns zero points

    # Signal thresholds
    signal_strong_buy_threshold: float = 80  # score > this -> "Strong Buy"
    signal_buy_threshold: float = 60         # score >= this -> "Buy"
    signal_watch_threshold: float = 40       # score >= this -> "Watch"; else "Ignore"

    # Position sizing
    fractional_share_decimals: int = 4     # precision for fractional-share sizing

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TradingConfig":
        return cls(**data)


DEFAULT_CONFIG = TradingConfig()
