"""Tunable trading parameters, decoupled from any fetch/UI/storage concern.

`TradingConfig` is the single object every pure calculation function in this
package takes as a parameter instead of reading module-level globals. That's
what lets the same math run identically from Streamlit, the nightly settlement
job, the backtester, and the Optuna objective function -- and it's the shape
Phase 6 will eventually load from MongoDB's System_Config collection (hence
`to_dict`/`from_dict` below).
"""

from dataclasses import asdict, dataclass, replace


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

    # Falling-knife awareness -- a loose rsi_oversold_threshold (this system's
    # tuned active value has landed as high as ~52, well above a "classic"
    # RSI<30 reading) can stay satisfied for weeks during a genuine sustained
    # decline, not just a brief dip. Combined with support_lookback_days
    # recalculating the structural low on a rolling window, that can produce
    # repeated fresh Buy/Strong Buy signals on a ticker that's simply making
    # new lows day after day, not reversing. Purely informational -- does NOT
    # affect Trade_Score/Signal, just flags it for you to see.
    extended_decline_warning_days: int = 5  # flag if RSI has stayed below
                                             # rsi_oversold_threshold for at
                                             # least this many consecutive
                                             # trading days
    extended_decline_penalty_per_day: float = 1.5  # Trade_Score points
                                             # subtracted per day the streak
                                             # exceeds extended_decline_warning_days
    extended_decline_penalty_cap: float = 30.0  # max total points subtracted

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

    # Portfolio-level risk (capital allocation, not signal generation --
    # never part of optimize.py's search space: a per-ticker walk-forward
    # backtest doesn't model simultaneous cross-ticker correlation, so this
    # is a personal risk preference, not something to be tuned against
    # backtested Sharpe)
    max_sector_allocation_pct: float = 0.40  # cap any one sector at this fraction
                                              # of total_cash in allocate_capital;
                                              # 0 (or negative) disables the cap

    # Settlement (Phase 3)
    max_holding_days: int = 15             # mark EXPIRED if neither stop nor target
                                            # hit within N trading days of entry

    # Execution realism (backtest optimism correction). Only applied to
    # stop_hit_intraday fills -- gap fills already use the real traded Open,
    # and target_hit/gap_up_target are limit fills that by definition can't
    # legitimately execute worse than their limit. commission is a flat %
    # of trade value per round-trip (not a $ amount -- settle_trade has no
    # notion of position size/share count, so a %-of-value cost is the only
    # form that stays consistent regardless of how big a position was).
    slippage_pct: float = 0.001            # 0.1% haircut on triggered-stop fills
    commission_pct_per_trade: float = 0.0  # round-trip cost as % of trade value;
                                            # 0 reflects most modern zero-commission
                                            # US-equity brokers -- override if yours
                                            # charges per-share/flat fees

    # Entry-fill timing realism (backtest only -- see swingtrade/backtest.py's
    # simulate_signals). A signal can only be known AFTER the day's close it
    # was computed from, so the earliest a real limit order could possibly
    # fill is the NEXT session, not the same bar the signal fired on.
    max_entry_wait_days: int = 5           # a resting limit order at Buy_Price
                                            # is abandoned (no trade) if never
                                            # touched within this many trading
                                            # days after the signal

    # Breakout/trend-following strategy (swingtrade/levels.compute_breakout_levels,
    # swingtrade/backtest.simulate_breakout_signals) -- a second, independent
    # signal separate from the RSI-oversold mean-reversion one above. Buys
    # strength (a new N-day closing high in a confirmed uptrend) instead of
    # weakness. Built after benchmark_random_entry.py showed RSI-oversold
    # TIMING carries no real predictive value over random entry days -- see
    # improvements.txt's STRATEGIC PIVOT section.
    breakout_lookback_days: int = 20       # signal fires when today's Close
                                            # exceeds the highest High of the
                                            # PRIOR N trading days (excludes
                                            # today itself -- no look-ahead)
    breakout_rsi_overbought_threshold: float = 100.0  # skip a breakout signal
                                            # if RSI is already at/above this
                                            # (over-extended/exhausted, more
                                            # likely to fail or reverse than
                                            # a breakout fresh off quiet
                                            # consolidation). 100.0 is the
                                            # practical "disabled" value --
                                            # RSI essentially never reaches
                                            # exactly 100
    breakout_relative_strength_min: float = -100.0  # skip a breakout signal
                                            # if the ticker's return over the
                                            # trailing breakout_lookback_days
                                            # window, minus SPY's return over
                                            # the same window, is below this
                                            # (a stock breaking out only
                                            # because the whole market is
                                            # ripping isn't the same as one
                                            # genuinely beating the market).
                                            # -100.0 is the practical
                                            # "disabled" value -- relative
                                            # returns essentially never fall
                                            # that far
    breakout_volume_ratio_min: float = 0.0  # skip a breakout signal unless
                                            # today's Volume is at least this
                                            # many times the PRIOR
                                            # volume_lookback_days average
                                            # (excludes today itself, same
                                            # no-look-ahead convention as
                                            # Highest_High) -- a genuine
                                            # breakout on high volume vs. a
                                            # low-volume drift above an old
                                            # high are different events.
                                            # 0.0 is the practical "disabled"
                                            # value -- a real ratio is always
                                            # >= 0
    adx_window: int = 14                   # ADX (trend-strength) lookback
                                            # (trading days)
    breakout_adx_min: float = 0.0          # skip a breakout signal unless
                                            # ADX is at least this. ADX
                                            # measures how STRONG the
                                            # current trend is, independent
                                            # of direction -- a different
                                            # dimension than RSI (momentum
                                            # level) or Relative_Strength
                                            # (direction vs. market). A
                                            # breakout during a weak/choppy
                                            # trend and one during a
                                            # genuinely strong trend look
                                            # identical to every other
                                            # filter in this system, but
                                            # aren't the same event. 0.0 is
                                            # the practical "disabled" value
                                            # -- ADX is always >= 0
    obv_window: int = 20                   # rolling window for On-Balance
                                            # Volume's own z-score baseline
                                            # (trading days)
    breakout_obv_zscore_min: float = -100.0  # skip a breakout signal
                                            # unless On-Balance Volume's
                                            # z-score against its own
                                            # trailing obv_window
                                            # mean/stdev is at least this.
                                            # OBV (cumulative signed volume
                                            # -- up days add Volume, down
                                            # days subtract it) rising
                                            # relative to its own recent
                                            # baseline reflects sustained
                                            # buying pressure BUILDING UP
                                            # over time, a deeper signal
                                            # than Volume_Ratio's single-day
                                            # spike check. Z-scored (not
                                            # used raw) because OBV's
                                            # absolute magnitude is
                                            # arbitrary -- it depends on how
                                            # much leading history precedes
                                            # the window -- but the z-score
                                            # is NOT: OBV and its rolling
                                            # mean shift by the same
                                            # constant for any amount of
                                            # extra leading history, so
                                            # their difference (and this
                                            # z-score) is invariant to it.
                                            # -100.0 is the practical
                                            # "disabled" value -- a real
                                            # z-score essentially never
                                            # gets that extreme
    bb_window: int = 20                    # window for the raw
                                            # price-volatility measure
                                            # feeding the squeeze z-score
                                            # (rolling stdev/mean of
                                            # Close, trading days)
    bb_squeeze_window: int = 60            # window over which that
                                            # volatility measure is itself
                                            # z-scored, to find "is
                                            # volatility unusually
                                            # CONTRACTED relative to its
                                            # own recent history" (trading
                                            # days)
    breakout_squeeze_zscore_max: float = 100.0  # skip a breakout signal
                                            # unless the PRIOR day's
                                            # (.shift(1) -- so today's own
                                            # breakout move doesn't
                                            # contaminate the reading)
                                            # volatility z-score was at/below
                                            # this. Classic "squeeze"
                                            # pattern: a breakout emerging
                                            # from a period of volatility
                                            # CONTRACTION ("coiled spring")
                                            # is a different, often more
                                            # reliable event than one that
                                            # isn't -- a volatility-regime
                                            # signal, distinct from every
                                            # other filter here. 100.0 is
                                            # the practical "disabled"
                                            # value -- a real z-score
                                            # essentially never gets that
                                            # extreme

    # Which signal this config represents -- "rsi" (simulate_signals,
    # mean-reversion) or "breakout" (simulate_breakout_signals,
    # trend-following). Purely a tag for downstream dispatch (optimize.py,
    # run_backtest.py, eventually live signal generation); doesn't affect
    # any calculation itself. Old configs written before this field existed
    # default to "rsi" via from_dict(), which is correct -- every config
    # before the breakout strategy was added WAS an RSI config.
    strategy: str = "rsi"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TradingConfig":
        return cls(**data)


DEFAULT_CONFIG = TradingConfig()


def loosened_breakout_config(config: TradingConfig) -> TradingConfig:
    """Same core breakout definition as `config` (breakout_lookback_days,
    ATR stop/target multiples, everything that defines WHAT a signal is)
    but the six "sharpening" filters -- breakout_rsi_overbought_threshold,
    breakout_relative_strength_min, breakout_volume_ratio_min,
    breakout_adx_min, breakout_obv_zscore_min, breakout_squeeze_zscore_max
    -- reset to their practical-disabled defaults. Shows what would have
    scored a real signal under just the base strategy, without the extra
    selectivity that's WHY the real config is validated to be as selective
    as it is (see improvements.txt items 17/18/23 -- each filter earned its
    place, or didn't, through actual holdout testing).

    Purely a display/research convenience (see dip_buy_analyzer.py's
    "Loosened Filters" section) -- never used for capital allocation,
    never logged to Trade_Signals, never touches the active System_Config.
    Only meaningful for strategy="breakout" configs; harmless no-op shape
    for "rsi" (those six fields aren't read by RSI scoring at all)."""
    return replace(
        config,
        breakout_rsi_overbought_threshold=100.0,
        breakout_relative_strength_min=-100.0,
        breakout_volume_ratio_min=0.0,
        breakout_adx_min=0.0,
        breakout_obv_zscore_min=-100.0,
        breakout_squeeze_zscore_max=100.0,
    )
