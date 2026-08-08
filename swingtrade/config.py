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
    max_total_deployed_pct: float = 0.0    # cap total spend (per allocate_capital
                                            # call) at this fraction of portfolio
                                            # value; 0 (or negative) disables the
                                            # cap -- ships disabled so existing
                                            # active configs are unaffected until
                                            # explicitly opted in

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

    # Pullback-in-uptrend strategy (swingtrade/levels.compute_pullback_levels,
    # swingtrade/backtest.simulate_pullback_signals) -- a third, independent
    # signal, distinct from both RSI-oversold mean-reversion (any weakness,
    # no trend context -- already shown to carry no real timing edge, see
    # benchmark_random_entry.py) and breakout (requires a fresh N-day
    # closing high THE SAME DAY, which is rare). Buys a shallow dip toward
    # a rising short-term moving average WITHIN a confirmed macro uptrend --
    # fires far more often than breakout because "price near its own rising
    # 20-day SMA" happens on many more days per ticker than "fresh 45-day
    # high", while still being trend-following (buying in a real uptrend),
    # not blind mean-reversion.
    pullback_ma_window: int = 20           # short SMA price pulls back
                                            # toward (the "pullback anchor")
    pullback_ma_slope_window: int = 10     # lookback to confirm that SMA
                                            # is itself rising
                                            # (MA > MA.shift(this)) -- a
                                            # topping/rolling-over MA is a
                                            # different, less trustworthy
                                            # event than a genuine uptrend
                                            # pullback and shouldn't count
    pullback_band_pct: float = 3.0         # symmetric band (both above and
                                            # below the MA) counted as
                                            # "close enough" to call it a
                                            # pullback/support test.
                                            # Deliberately symmetric (not
                                            # direction-aware) for a lean
                                            # v1 -- tunable by Optuna, which
                                            # can reveal whether a
                                            # tighter/looser or asymmetric
                                            # band would help

    # Breakout-retest strategy (swingtrade/levels.compute_breakout_retest_levels,
    # swingtrade/backtest.simulate_breakout_retest_signals) -- a fourth
    # signal, built after BOTH RSI-oversold and pullback-in-uptrend lost to
    # matched-count random-entry timing on held-out tickers (see
    # benchmark_random_entry.py) while breakout (v19) was the one signal
    # that beat it. Keeps breakout's validated ingredient -- a genuine
    # fresh breakout_lookback_days-day closing high -- but relaxes its most
    # restrictive property (must fire THE SAME DAY) by allowing entry on a
    # pullback BACK TO that breakout's level within a following window,
    # instead of requiring the chase on day one. Reuses breakout_lookback_days
    # for what counts as "a breakout" in the first place; no separate field.
    retest_window_days: int = 10           # how many days after a
                                            # confirmed breakout the retest
                                            # is still considered valid
    retest_band_pct: float = 3.0           # symmetric band (both
                                            # above/below the original
                                            # breakout trigger level)
                                            # counted as "close enough" to
                                            # call it a genuine retest --
                                            # same lean-v1, Optuna-tunable
                                            # pattern as pullback_band_pct

    # 52-week-high momentum strategy (swingtrade/levels.compute_week52_levels,
    # swingtrade/backtest.simulate_week52_signals) -- a fifth signal, a
    # well-documented academic factor (George & Hwang 2004) distinct from
    # every prior attempt: unlike breakout (a discrete "new high TODAY"
    # event) or breakout_retest (a bounded window after one specific
    # event), this is a continuous STATE -- how close is price, right now,
    # to its own trailing week52_lookback_days high -- so it can stay true
    # for many consecutive days while a stock consolidates near its highs,
    # likely the most frequent-firing strength-anchored signal tried yet.
    # Still anchored to genuine strength, not weakness or mere MA proximity.
    week52_lookback_days: int = 252        # trailing window (trading days,
                                            # ~52 weeks) for the high
    week52_nearness_pct: float = 5.0       # max % BELOW that trailing high
                                            # still counted as "near" (0% =
                                            # at/above the high itself)

    # Momentum-burst strategy (swingtrade/levels.compute_momentum_burst_levels,
    # swingtrade/backtest.simulate_momentum_burst_signals) -- a sixth signal,
    # built for faster firing than any prior strategy: fires on a single
    # day's strong price gain CONFIRMED by unusually high volume, rather
    # than requiring a fresh N-day high (breakout) or proximity to one
    # (breakout_retest/week52_high). Reuses Volume_Ratio (today's Volume /
    # prior volume_lookback_days average -- already computed for breakout's
    # own optional filter, see breakout_volume_ratio_min above) as the
    # volume-confirmation leg; these two fields are this strategy's own
    # independent thresholds, deliberately NOT shared with
    # breakout_volume_ratio_min (a different strategy's optional gate).
    # Unlike every "0/disabled by default" filter field above, these two
    # DEFINE the trigger itself, so they get real, non-disabled defaults.
    momentum_burst_gain_pct_min: float = 3.0  # today's Close vs. prior
                                            # Close % gain must be at least
                                            # this
    momentum_burst_volume_ratio_min: float = 2.0  # today's Volume must be
                                            # at least this many times the
                                            # PRIOR volume_lookback_days
                                            # average (same no-look-ahead
                                            # convention as
                                            # breakout_volume_ratio_min)
    momentum_burst_entry_fill: str = "limit"  # backtest-only (see
                                            # swingtrade/backtest.py's
                                            # simulate_momentum_burst_signals) --
                                            # "limit" = resting limit order
                                            # waiting for a downside touch
                                            # (same convention week52_high
                                            # uses); "next_open" = buy the
                                            # very next session's Open
                                            # unconditionally, no waiting.
                                            # Default "limit" preserves
                                            # today's exact behavior; see
                                            # improvements.txt for why
                                            # "next_open" may better model
                                            # a genuine momentum-chase entry

    # Which signal this config represents -- "rsi" (simulate_signals,
    # mean-reversion), "breakout" (simulate_breakout_signals,
    # trend-following), "pullback" (simulate_pullback_signals,
    # trend-following pullback entry), "breakout_retest"
    # (simulate_breakout_retest_signals, pullback to a recent breakout's
    # level), or "week52_high" (simulate_week52_signals, near a trailing
    # 52-week high). Purely a tag for downstream dispatch (optimize.py,
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
