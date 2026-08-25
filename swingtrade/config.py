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
    atr_take_profit_multiplier: float = 2.0  # sell_price = buy_price + multiplier * ATR
                                            # (RRR = this / stop_loss_atr_multiplier
                                            # = 2.0 -- kept comfortably above
                                            # optimize.py's RRR_FLOOR=1.6 so any
                                            # untuned config starts able to
                                            # actually clear signal_buy_threshold;
                                            # see improvements.txt for the
                                            # v19/v27/v28 incident this avoids
                                            # repeating for brand-new configs)
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

    # Trailing-stop exit (swingtrade/settlement.settle_trade_with_trailing) --
    # shared exit infrastructure, not a per-strategy filter, so ONE pair of
    # fields applies uniformly to whichever strategy's simulate_*_signals()
    # honors trailing_stop_enabled (currently breakout/squeeze_breakout/
    # ma_crossover, the 3 active strategies -- see backtest.py). A trade
    # behaves exactly as it does today (fixed stop_loss/sell_price) until
    # sell_price is first reached; only then does it start trailing instead
    # of exiting automatically -- see settle_trade_with_trailing()'s own
    # docstring for the full design rationale. False (disabled) is the
    # practical no-op default, same convention as every other optional
    # exit/filter field in this codebase.
    trailing_stop_enabled: bool = False
    trailing_stop_atr_multiplier: float = 1.5  # how far below the running
                                            # post-target high the trailing
                                            # stop sits, in ATR multiples

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
    sector_relative_strength_lookback_days: int = 63  # ~3 months, standard
                                            # sector-momentum convention --
                                            # deliberately its OWN shared
                                            # window (like adx_window/
                                            # obv_window/bb_window above),
                                            # not tied to any one strategy's
                                            # own trigger window, since this
                                            # measures broader market
                                            # context rather than re-scoring
                                            # the same-window ticker-level
                                            # Relative_Strength above. See
                                            # improvements.txt items 68/69 --
                                            # this exact lookback is what
                                            # was backtested and validated
                                            # (tailwind beats headwind on
                                            # ALL/TUNE/HOLDOUT once checked
                                            # with a proper multi-seed
                                            # holdout average).
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
    breakout_sector_relative_strength_min: float = -100.0  # skip a
                                            # breakout signal if the
                                            # ticker's own SECTOR (a real
                                            # SPDR sector ETF, e.g. XLK for
                                            # Technology) return over the
                                            # trailing
                                            # sector_relative_strength_lookback_days
                                            # window, minus SPY's return
                                            # over the same window, is
                                            # below this -- a broader-
                                            # context sibling to
                                            # breakout_relative_strength_min
                                            # above (that one compares the
                                            # TICKER to the market; this one
                                            # compares the ticker's whole
                                            # SECTOR to the market).
                                            # BACKTEST/OPTUNA-ONLY as of
                                            # this field's introduction --
                                            # market_data.py's live scan
                                            # path does not fetch sector ETF
                                            # data, so this reads None/NaN
                                            # (never excludes on its own,
                                            # same convention as every other
                                            # filter here) in live
                                            # production regardless of this
                                            # value, until that live-wiring
                                            # is deliberately added as its
                                            # own separate step. -100.0 is
                                            # the practical "disabled"
                                            # value, same convention as
                                            # breakout_relative_strength_min.

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
    momentum_burst_strength_cap_pct: float = 5.0  # add_momentum_burst_trade_score's
                                            # Signal_Strength_Pct (Day_Gain_Pct
                                            # minus momentum_burst_gain_pct_min,
                                            # i.e. how far today's gain clears
                                            # its own minimum bar) earns full
                                            # score points at/above this excess
                                            # -- an 8%+ day (3% min + 5pp) reads
                                            # as maximally strong. Replaces
                                            # Distance_to_Buy_Pct, which is
                                            # always 0 for this strategy (see
                                            # momentum_burst_levels_from_frame)
                                            # and so could never differentiate
                                            # tickers -- see improvements.txt.

    # Squeeze-breakout strategy (swingtrade/levels.compute_squeeze_breakout_levels,
    # swingtrade/backtest.simulate_squeeze_breakout_signals) -- a seventh
    # signal, a materially different trigger from every prior one: fires
    # when volatility was recently CONTRACTED (a squeeze, reusing
    # Squeeze_Zscore -- already computed for breakout's own optional
    # breakout_squeeze_zscore_max filter, see above) and today shows a
    # real directional EXPANSION (a meaningful same-day gain, reusing the
    # Day_Gain_Pct concept momentum_burst introduced). Deliberately does
    # NOT also require a fresh high over any window (an earlier design
    # draft did -- rejected because requiring BOTH a squeeze AND a fresh
    # high is the intersection of two conditions, necessarily rarer than
    # either alone, defeating the point of a faster-firing signal) and
    # does NOT require volume confirmation (unlike momentum_burst) -- kept
    # deliberately distinct rather than a near-duplicate of the existing
    # fast-firing candidate. Same "real, non-disabled default" treatment
    # as momentum_burst's fields -- these DEFINE the trigger.
    squeeze_breakout_zscore_max: float = -1.0  # the trailing MINIMUM
                                            # Squeeze_Zscore over
                                            # squeeze_breakout_lookback_days
                                            # must be at/below this --
                                            # bottom ~16% of the ticker's
                                            # own trailing volatility
                                            # distribution, a real
                                            # contraction
    squeeze_breakout_lookback_days: int = 5  # how many trailing days to
                                            # check for a recent squeeze
                                            # (squeezes often persist
                                            # several days before
                                            # releasing -- the breakout
                                            # day itself need not be the
                                            # single tightest day)
    squeeze_breakout_gain_pct_min: float = 2.0  # today's Close vs. prior
                                            # Close % gain must be at
                                            # least this -- lower bar than
                                            # momentum_burst_gain_pct_min
                                            # since there's no volume
                                            # co-requirement here
    squeeze_breakout_entry_fill: str = "limit"  # same "limit" vs.
                                            # "next_open" toggle as
                                            # momentum_burst_entry_fill --
                                            # see that field's comment.
                                            # Built in from the start here
                                            # (not retrofitted) since this
                                            # is the same "chase a same-day
                                            # expansion" signal shape that
                                            # made the fill choice matter
                                            # for momentum_burst
    squeeze_breakout_strength_cap_pct: float = 5.0  # same role as
                                            # momentum_burst_strength_cap_pct
                                            # -- add_squeeze_breakout_trade_score's
                                            # Signal_Strength_Pct (Day_Gain_Pct
                                            # minus squeeze_breakout_gain_pct_min)
                                            # earns full score points at/above
                                            # this excess (a 7%+ day, 2% min +
                                            # 5pp). Replaces Distance_to_Buy_Pct,
                                            # always 0 for this strategy.

    # squeeze_breakout's own "sharpening" filters -- same five dimensions
    # adx_trend_entry's Phase 2 added (breakout's own six minus whichever
    # one is already THIS strategy's core trigger -- here, Squeeze_Zscore
    # itself, not an add-on filter). Reuses the identical already-computed
    # columns (RSI/Relative_Strength/Volume_Ratio/ADX/OBV_Zscore), applied
    # the same way breakout's/adx_trend_entry's filters are: as additional
    # gates in add_squeeze_breakout_trade_score() AND
    # simulate_squeeze_breakout_signals() (kept in sync so live and
    # backtested definitions can't disagree), NOT baked into Squeeze_Signal
    # itself. Every default below is the identical "practical no-op" value
    # breakout's own filters use -- 0/off changes nothing until explicitly
    # tuned. Added specifically to be Optuna-searchable under the
    # RRR_FLOOR-safe, tp/sl-PINNED regime (see optimize.py's
    # --pin-atr-take-profit-multiplier) -- see improvements.txt item 42 for
    # why every prior filter/tune on this strategy needed re-doing.
    squeeze_breakout_rsi_overbought_threshold: float = 100.0
    squeeze_breakout_relative_strength_min: float = -100.0
    squeeze_breakout_volume_ratio_min: float = 0.0
    squeeze_breakout_adx_min: float = 0.0
    squeeze_breakout_obv_zscore_min: float = -100.0
    squeeze_breakout_sector_relative_strength_min: float = -100.0  # sibling
                                                   # to breakout_sector_relative_strength_min
                                                   # above -- see that field's own comment
                                                   # for the full rationale. Same
                                                   # BACKTEST/OPTUNA-ONLY caveat: reads
                                                   # None/NaN (never excludes on its own)
                                                   # in live production until sector-ETF
                                                   # fetching is separately wired into
                                                   # market_data.py.
    squeeze_breakout_earnings_gate: bool = False  # excludes a ticker whose Catalyst_Warning
                                                   # is True (within earnings_warning_days of
                                                   # its next earnings report -- reuses that
                                                   # shared field/machinery directly instead of
                                                   # inventing a second earnings-window concept).
                                                   # Boolean, not a numeric threshold, since
                                                   # unlike the other filters above there's no
                                                   # meaningful "how much" to tune -- either the
                                                   # gate applies or it doesn't. False (disabled)
                                                   # is the practical no-op default, same
                                                   # treatment as every other optional filter.
                                                   # See improvements.txt for the validation
                                                   # result before ever setting this True on a
                                                   # live config.

    # ADX-trend-entry strategy (swingtrade/levels.compute_adx_trend_entry_levels,
    # swingtrade/backtest.simulate_adx_trend_entry_signals) -- a ninth
    # signal, a continuous STATE (like week52_high/squeeze_breakout, not a
    # discrete event): fires whenever ADX (already computed, config.adx_window)
    # is at/above a "genuinely trending" threshold AND price is above a
    # short-term MA for direction (ADX alone measures trend STRENGTH,
    # independent of direction). Deliberately lean v1, mirroring every
    # other strategy's own launch -- reuses breakout's OWN eventual history
    # as the template: v19 didn't launch with its six optional "sharpening"
    # filters below (breakout_rsi_overbought_threshold etc.) either, they
    # were added incrementally AFTER it was already a trusted baseline.
    # Real, non-disabled defaults -- these DEFINE the trigger.
    adx_trend_entry_threshold: float = 25.0  # ADX must be at/above this --
                                            # the classic "trending" reading
                                            # in technical-analysis terms
                                            # (25 = trending, 40+ = strong
                                            # trend)
    adx_trend_entry_ma_window: int = 10    # short-term SMA window for
                                            # directional confirmation
                                            # (deliberately its own field,
                                            # not shared with ma_window/
                                            # pullback_ma_window -- every
                                            # strategy owns its own window)
    adx_trend_entry_entry_fill: str = "limit"  # same "limit" vs.
                                            # "next_open" toggle as
                                            # momentum_burst_entry_fill/
                                            # squeeze_breakout_entry_fill,
                                            # built in from the start here
                                            # too
    adx_trend_entry_strength_cap: float = 15.0  # same role as
                                            # momentum_burst_strength_cap_pct
                                            # -- add_adx_trend_entry_trade_score's
                                            # Signal_Strength_Pct (ADX minus
                                            # adx_trend_entry_threshold) earns
                                            # full score points at/above this
                                            # excess (ADX 40+, a "very strong
                                            # trend" TA reading, off a 25
                                            # threshold). Replaces
                                            # Distance_to_Buy_Pct, always 0
                                            # for this strategy.

    # adx_trend_entry's own "sharpening" filters -- Phase 2, added only
    # after the lean v1 above cleared the random-entry-timing bar (see
    # improvements.txt item 40). Deliberately the SAME five dimensions
    # breakout's own filters cover (relative strength, volume, OBV,
    # squeeze -- ADX itself is already this strategy's core trigger, not
    # an add-on filter here), reusing the identical already-computed
    # columns (Relative_Strength/Volume_Ratio/OBV_Zscore/Squeeze_Zscore),
    # applied the SAME way breakout's filters are: as additional gates in
    # add_adx_trend_entry_trade_score() AND simulate_adx_trend_entry_signals()
    # (kept in sync so live and backtested definitions can't disagree),
    # NOT baked into ADX_Trend_Signal itself. Every default below is the
    # identical "practical no-op" value breakout's own filters use -- 0/off
    # changes nothing until explicitly tuned.
    adx_trend_entry_rsi_overbought_threshold: float = 100.0
    adx_trend_entry_relative_strength_min: float = -100.0
    adx_trend_entry_volume_ratio_min: float = 0.0
    adx_trend_entry_obv_zscore_min: float = -100.0
    adx_trend_entry_squeeze_zscore_max: float = 100.0

    # Moving-average-crossover strategy (swingtrade/levels.compute_ma_crossover_levels,
    # swingtrade/backtest.simulate_ma_crossover_signals) -- a genuinely
    # different mechanical trigger from every strategy tried in this
    # project so far: fires the day a short-term SMA crosses ABOVE a
    # long-term SMA (trend CONFIRMATION via relative moving-average
    # positioning), not a price-level breakout (breakout/breakout_retest/
    # week52_high), not RSI-based mean-reversion (rsi/pullback), not a
    # volatility-regime shift (squeeze_breakout), not a raw trend-strength
    # threshold (adx_trend_entry). Built specifically because every
    # momentum/strength-family signal tried this session lost to
    # random-entry timing once properly checked (see improvements.txt item
    # 47) -- squeeze_breakout, the one strategy that DOES beat random, is
    # a volatility-regime signal, not a momentum one, which is the whole
    # motivation for trying something mechanically unrelated here.
    #
    # RRR fields set explicitly floor-compliant (2.0/1.0, matching
    # DEFAULT_CONFIG) from this candidate's very first version -- unlike
    # breakout's v43, which inherited v19's non-compliant ratio and had to
    # be fixed after promotion (improvements.txt item 44), this strategy
    # never gets an RRR that can't clear signal_buy_threshold in the first
    # place. See optimize.py's RRR_FLOOR for why 1.6 is the hard minimum;
    # 2.0 here matches the project's own established safe default.
    ma_crossover_short_window: int = 20   # short SMA lookback (trading days)
    ma_crossover_long_window: int = 50    # long SMA lookback (trading days) --
                                           # the cross itself is short crossing
                                           # ABOVE long, checked against
                                           # yesterday's relative position so
                                           # it only fires on the actual
                                           # crossover day, not every day the
                                           # short SMA merely stays above
    ma_crossover_entry_fill: str = "limit"  # same "limit" vs. "next_open"
                                           # toggle every same-day-trigger
                                           # strategy gets from the start --
                                           # see momentum_burst_entry_fill's
                                           # own comment for the full
                                           # rationale
    ma_crossover_strength_cap_pct: float = 0.5  # add_ma_crossover_trade_score's
                                           # Signal_Strength_Pct (the
                                           # crossover's own gap, short SMA
                                           # minus long SMA as a % of price)
                                           # earns full score points at/above
                                           # this excess -- replaces
                                           # Distance_to_Buy_Pct, always 0 for
                                           # this strategy (Buy_Price IS
                                           # Last_Close, same convention
                                           # momentum_burst/squeeze_breakout/
                                           # adx_trend_entry already use).
                                           # Was 2.0 -- found 2026-08-23 to be
                                           # structurally unreachable: real
                                           # data (584 crossovers, 70 tickers,
                                           # 5 years) shows this gap NEVER
                                           # exceeds ~1.24% at the crossover
                                           # moment (median 0.14%, p90 0.52%,
                                           # p99 1.08%) since a crossover is
                                           # measured right when the gap just
                                           # turned positive, by construction
                                           # near zero -- the old 2.0% cap
                                           # made the Buy/Strong Buy tier
                                           # unreachable for every real
                                           # ma_crossover signal ever seen,
                                           # so allocate_capital() could never
                                           # size a real position no matter
                                           # how long it ran. Recalibrated to
                                           # just under the real p90 (0.52%)
                                           # so a genuinely above-average
                                           # crossover (top ~10%) is Buy-
                                           # eligible. This field is never an
                                           # Optuna search dimension (always
                                           # just carried through from the
                                           # config default/candidate), so
                                           # this default matters for every
                                           # future re-tune too, not just the
                                           # live promoted config. See
                                           # improvements.txt.
    ma_crossover_earnings_gate: bool = False  # sibling to
                                           # squeeze_breakout_earnings_gate --
                                           # see that field's own comment for
                                           # the full rationale. False
                                           # (disabled) is the practical no-op
                                           # default.
    ma_crossover_sector_relative_strength_min: float = -100.0  # sibling to
                                           # breakout_sector_relative_strength_min --
                                           # see that field's own comment for the
                                           # full rationale. This is ma_crossover's
                                           # FIRST optional numeric filter (unlike
                                           # breakout/squeeze_breakout, it launched
                                           # with no "sharpening" filter family at
                                           # all). Same BACKTEST/OPTUNA-ONLY caveat:
                                           # reads None/NaN (never excludes on its
                                           # own) in live production until sector-ETF
                                           # fetching is separately wired into
                                           # market_data.py.

    # Mean-reversion PAIRS strategy (swingtrade/levels.compute_pairs_levels,
    # swingtrade/backtest.simulate_pairs_signals) -- LONG-ONLY laggard-
    # convergence: buy a ticker when it has diverged unusually far BELOW its
    # most-correlated same-sector peer over a recent window, betting on
    # reversion. Genuinely different mechanism from every trend/volatility-
    # following strategy tried so far (ticker-vs-peer divergence, not
    # ticker/sector-vs-market momentum) -- no short leg (this codebase has
    # no short-position support anywhere), so this is the long-only variant
    # strategy_selection.txt itself flagged as an option. Deliberately lean
    # v1, mirroring every other strategy's own launch -- reuses the shared
    # atr_take_profit_multiplier/stop_loss_atr_multiplier bracket, no new
    # payoff-geometry fields, no optional "sharpening" filters yet.
    pairs_lookback_days: int = 90  # trailing window (trading days) used to
                                           # compute rolling correlation between a
                                           # ticker and each same-sector peer, to
                                           # pick its "partner" -- long enough to
                                           # reflect a real, stable relationship,
                                           # not short-term coincidence
    pairs_min_correlation: float = 0.6  # a same-sector peer must clear this
                                           # rolling correlation to be accepted as
                                           # a partner at all -- below this, no
                                           # partner is assigned (missing data,
                                           # not a fabricated pairing), same
                                           # "don't exclude on missing/insufficient
                                           # data" convention as every other
                                           # optional signal in this codebase
    pairs_spread_window_days: int = 10  # window (trading days) over which the
                                           # ticker's own cumulative return minus
                                           # its partner's is measured -- "how far
                                           # they've diverged recently"
    pairs_zscore_window_days: int = 60  # trailing window the spread's own
                                           # rolling mean/stdev (the z-score's
                                           # baseline "what's normal for this
                                           # pair") is computed over
    pairs_zscore_entry_max: float = -2.0  # Pair_Spread_Zscore must be at or
                                           # below this to fire -- the ticker has
                                           # underperformed its partner unusually
                                           # much (2+ std devs) relative to its own
                                           # recent history
    pairs_zscore_strength_cap: float = 2.0  # extra z-score points below
                                           # pairs_zscore_entry_max that earn full
                                           # Signal_Strength_Pct credit (same
                                           # "distance past the trigger"
                                           # differentiating-term role
                                           # squeeze_breakout_strength_cap_pct
                                           # plays, just in z-score units here
                                           # instead of a %)
    pairs_entry_fill: str = "limit"  # same "limit" vs. "next_open" toggle every
                                           # other strategy has -- see
                                           # squeeze_breakout_entry_fill's comment

    # Cross-sectional MOMENTUM RANK strategy (swingtrade/levels.compute_momentum_levels,
    # swingtrade/backtest.simulate_momentum_signals) -- a genuinely different
    # mechanism from every strategy tried so far: every prior signal scores
    # ONE ticker from its own price history alone; this one ranks a ticker's
    # trailing return against every OTHER ticker in the watchlist on the
    # SAME day and buys the top decile (Jegadeesh & Titman cross-sectional
    # momentum). Continuous STATE, not a discrete event (like week52_high,
    # not like breakout) -- fires every day a ticker's percentile clears the
    # threshold, using the same stop/target/max-holding-day exit machinery
    # every strategy shares; no new "exit because you fell out of the top
    # decile" mechanism was built (this codebase has no rebalance-portfolio
    # concept anywhere else). RRR pinned >= RRR_FLOOR (1.6) from this very
    # first default -- applying the item-97/RRR-ceiling lesson proactively
    # instead of discovering post-hoc the Buy tier is unreachable.
    momentum_lookback_days: int = 63  # trailing-return formation window
                                           # (trading days, ~3 months) -- matches
                                           # this project's own existing
                                           # sector_relative_strength_lookback_days
                                           # default, a "current" (not multi-year)
                                           # momentum read
    momentum_top_percentile_min: float = 90.0  # a ticker's trailing-return
                                           # percentile rank (0-100 scale, see
                                           # best_ideas.compute_sector_rs_scores()'s
                                           # identical rank(pct=True)*100
                                           # convention) must clear this to fire --
                                           # 90 = top decile, matching this
                                           # project's own "buy the top decile"
                                           # framing
    momentum_strength_cap_pct: float = 10.0  # extra percentile points past
                                           # momentum_top_percentile_min that earn
                                           # full Signal_Strength_Pct credit --
                                           # same "reused field name, different
                                           # units" precedent as
                                           # pairs_zscore_strength_cap/
                                           # squeeze_breakout_strength_cap_pct
    momentum_entry_fill: str = "limit"  # same "limit" vs. "next_open" toggle
                                           # every other strategy has, built in
                                           # from day one per the item-37 lesson
                                           # (check fill-model sensitivity
                                           # proactively, not after promotion)

    # Insider-buying (2026-08-21) -- buy when recent, real-dollar insider
    # Form-4 purchases cluster within a lookback window, in a confirmed
    # macro uptrend. See run_backtest.fetch_insider_purchases() for the
    # data source (yfinance's insider_transactions, ~12-22mo of real
    # history only -- NOT this project's usual 5y window) and its
    # reporting-lag no-look-ahead handling.
    insider_lookback_days: int = 14  # matches the source idea's own convention
    insider_min_purchase_value: float = 50_000.0  # filters trivial/option-exercise-
                                           # adjacent buys, pooled over the window
    insider_min_distinct_buyers: int = 1  # >1 requires broader conviction, not
                                           # just one insider repeatedly buying
    insider_strength_cap_buyers: float = 2.0  # same role as
                                           # squeeze_breakout_strength_cap_pct --
                                           # add_insider_buying_trade_score's
                                           # Signal_Strength_Pct (distinct buyers
                                           # beyond insider_min_distinct_buyers,
                                           # NOT a % -- see pairs_zscore_strength_cap's
                                           # identical "reused field name, different
                                           # units" precedent) earns full credit at
                                           # this many EXTRA distinct buyers; raw
                                           # dollar value isn't used here since real
                                           # purchase sizes ($1M-$10M+) would blow
                                           # past any sane cap almost immediately
                                           # and stop differentiating anything
    insider_reporting_lag_days: int = 3  # conservative guard against Start-Date's
                                           # transaction-vs-filing-date ambiguity --
                                           # see fetch_insider_purchases()
    insider_entry_fill: str = "limit"  # same "limit" vs. "next_open" toggle every
                                           # other strategy has

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
