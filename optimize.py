"""
Optuna learning engine (Phase 5).

Searches the RSI/ATR parameters named in the original architecture goal --
rsi_oversold_threshold, atr_take_profit_multiplier, stop_loss_atr_multiplier
-- via Bayesian optimization (Optuna's TPE sampler), scoring each candidate
on its AGGREGATE out-of-sample performance pooled across every Phase 4
walk-forward fold. Never a single fold's in-sample fit -- see
swingtrade/backtest.py for why that would be curve-fitting.

Historical OHLCV is fetched once up front and reused across every trial;
each trial then only re-runs the in-memory backtest (fast, pure
computation), which is what makes a 50+ trial search practical.

Objective: sharpe_like (mean / stdev of pooled out-of-sample pnl_pct) --
rewards consistency, not just raw returns, matching the risk-management
focus already baked into the rest of this system (Stop_Loss, RRR, capital
allocation). Candidates that produce too few pooled trades to trust
(< MIN_TRADES_FOR_SCORE) are penalized rather than scored on a lucky small
sample -- the same anti-curve-fitting spirit as walk-forward itself, just
applied within a single trial too.

The winning trial is written to MongoDB's System_Config collection as a new
`candidate` document -- it is NEVER auto-promoted to `active`. A human
reviews it and runs promote_config.py to actually put it live
(champion/challenger pattern): a noisy optimization run should not be able
to silently degrade live trading.

Recency-weighted scoring: out-of-sample folds closer to --end (i.e. closer
to today, the period your actual live trades fall in) count exponentially
more toward the pooled score than older folds, via --recency-half-life-days
(default 180). This is deliberately NOT the same thing as mixing live
Trade_Outcomes documents directly into the score -- those reflect one
specific historical config, not each trial's varying candidate, so grafting
them in would conflate two different questions (see summarize_weighted()).
Instead, every trial gets re-simulated fairly over the SAME recent calendar
window under ITS OWN candidate config, and that recent window is simply
weighted more heavily -- so what's been working lately (the regime your
live trades are actually part of) has outsized influence on the winner,
without corrupting the cross-candidate comparison. Pass --recency-half-life-days
0 to fall back to old-style uniform pooling.

Correlation-adjusted scoring: pooled trades also aren't independent of each
other -- a real, observed failure mode is dozens of correlated Technology
signals firing the same day during one sector move, which a naive pooled
mean/std would count as dozens of independent data points instead of one
correlated event. summarize_weighted() combines the recency weight above
with swingtrade.compute_cluster_weights() (same-day/same-sector clusters
split a combined weight of 1.0), via watchlist.read_ticker_sectors. Both
weighting effects flow into the same Kish effective-sample-size gate for
MIN_TRADES_FOR_SCORE, so a candidate can't look well-sampled just because it
produced a large trade_count that's secretly a handful of correlated events
repeated many times.

Live Trade_Outcomes are still reported separately for context (how many
exist, their aggregate performance) -- useful as a sanity check against what
the backtest predicts, but not an input to the score.

Ticker-universe holdout validation: walk-forward already holds out TIME
rigorously (in-sample/out-of-sample folds), but until now every search also
tuned AND "validated" on the exact same ticker universe -- a config could be
quietly overfit to the specific names in the watchlist without anyone
catching it (the walk-forward folds only ever ask "does this generalize to
a later date for these same tickers," never "does this generalize to a
ticker the search never saw at all"). split_tickers_holdout() partitions the
fetched tickers into a TUNE set (all trials -- objective, baseline, Optuna
search -- only ever see this) and a HOLDOUT set that Optuna never touches
during the search, stratified by sector so neither set accidentally
concentrates in one sector the way the original live-vs-backtest gap did.
After the search picks a winner, that winning config (and DEFAULT_CONFIG,
for reference) get re-run ONE extra time against the holdout tickers over
the same folds -- reported alongside the tune-set numbers, and stored on
the candidate doc as `holdout_metrics`, so whoever reviews the candidate
before promoting can see whether the edge generalizes to unseen names or
was quietly specific to the tuning watchlist. Pass --holdout-frac 0 to
disable (old behavior: tune and "validate" on every ticker).

Strategy-agnostic since the breakout/trend-following signal was added
(improvements.txt's STRATEGIC PIVOT section, built after benchmark_random_entry.py
showed RSI-oversold timing carries no real predictive value over random entry
days). --strategy rsi (default) searches the original RSI/ATR/stop-loss/
streak-penalty space; --strategy breakout searches breakout_lookback_days/
atr_take_profit_multiplier/stop_loss_atr_multiplier/
breakout_rsi_overbought_threshold/breakout_relative_strength_min/
breakout_volume_ratio_min instead; --strategy pullback searches
pullback_ma_window/pullback_ma_slope_window/pullback_band_pct/
atr_take_profit_multiplier/stop_loss_atr_multiplier (a genuinely different,
more frequent trend-following signal -- buys a shallow dip toward a rising
short-term MA instead of requiring a fresh N-day high, see
swingtrade.compute_pullback_levels); --strategy breakout_retest searches
breakout_lookback_days/retest_window_days/retest_band_pct/
atr_take_profit_multiplier/stop_loss_atr_multiplier (buys a pullback BACK
TO a recent genuine breakout's own trigger level, instead of requiring the
chase on the breakout day itself -- see
swingtrade.compute_breakout_retest_levels, built after pullback lost to
random-entry timing while breakout itself didn't); --strategy week52_high
searches week52_lookback_days/week52_nearness_pct/atr_take_profit_multiplier/
stop_loss_atr_multiplier (a continuous-STATE signal -- how close is price to
its own trailing 52-week high, right now -- rather than a discrete event,
so it can fire on many consecutive days; a well-documented academic factor,
see swingtrade.compute_week52_levels); --strategy momentum_burst searches
momentum_burst_gain_pct_min/momentum_burst_volume_ratio_min/
atr_take_profit_multiplier/stop_loss_atr_multiplier (fires on a single
day's price gain CONFIRMED by unusually high volume, no fresh-high
requirement at all -- built specifically to fire more often than any prior
strategy, see swingtrade.compute_momentum_burst_levels; shipped
experimental with a mixed untuned-defaults result -- beats random-entry
timing on holdout/aggregate, loses on tune -- see improvements.txt item 35;
tuning made it WORSE, not better -- holdout sharpe went net negative, the
textbook overfitting signature -- see item 36; deprioritized); --strategy
squeeze_breakout searches squeeze_breakout_zscore_max/
squeeze_breakout_lookback_days/squeeze_breakout_gain_pct_min/
atr_take_profit_multiplier/stop_loss_atr_multiplier (fires on a real price
expansion following a recent volatility CONTRACTION -- a squeeze, reusing
Squeeze_Zscore already computed for breakout's own optional filter -- no
fresh-high requirement, no volume confirmation, see
swingtrade.compute_squeeze_breakout_levels; shipped experimental with a
CLEAN untuned-defaults result -- beats random-entry timing on ALL cuts
under BOTH entry-fill models, see improvements.txt item 38, this search
tests whether tuning can improve an already-solid candidate rather than
rescue a fragile one); --strategy ma_crossover searches
ma_crossover_short_window/ma_crossover_long_window/
atr_take_profit_multiplier/stop_loss_atr_multiplier (short SMA crosses
above long SMA -- trend confirmation, see
swingtrade.compute_ma_crossover_levels; previously refined only via manual
one-at-a-time grid search on the two window params with tp/sl held fixed,
see improvements.txt item 50 -- this is the first search of all 4
dimensions jointly) -- everything above (WFO, recency weighting,
correlation-adjustment, ticker-holdout, champion/challenger) applies
identically to all eight. Uses config.squeeze_breakout_entry_fill's/
config.ma_crossover_entry_fill's own default ("limit") -- entry-fill mode
itself is not part of this search space, same as every other strategy's
fixed structural choices.

Usage:
    python optimize.py --trials 50 --start 2023-01-01 --end 2026-07-01
    python optimize.py --trials 50 --recency-half-life-days 90   # weight recent regime more heavily
    python optimize.py --trials 50 --holdout-frac 0.3            # hold out 30% of tickers by sector
    python optimize.py --trials 30 --strategy breakout           # search the breakout signal instead
    python optimize.py --trials 30 --strategy pullback           # search the pullback signal instead
    python optimize.py --trials 30 --strategy breakout_retest    # search the breakout-retest signal instead
    python optimize.py --trials 30 --strategy week52_high        # search the 52-week-high signal instead
    python optimize.py --trials 30 --strategy squeeze_breakout   # search the squeeze-breakout signal instead
    python optimize.py --trials 30 --strategy momentum_burst     # search the momentum-burst signal instead
    python optimize.py --trials 30 --strategy ma_crossover       # search the MA-crossover signal instead
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import optuna
import pandas as pd

import storage
import swingtrade
from run_backtest import LOOKBACK_BUFFER_DAYS, MARKET_INDEX_TICKER, fetch_earnings_dates, fetch_history
from watchlist import SECTOR_ETF, read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5

MIN_TRADES_FOR_SCORE = 50    # penalize candidates with too few pooled trades to trust --
                              # raised from 15 (2026-08-10, see improvements.txt item 43)
                              # after discovering EVERY winning trial across 5 separate
                              # tuning attempts this session (breakout/squeeze_breakout,
                              # with and without tp/sl pinned) converged to a TUNE
                              # effective_trade_count clustered in the high-teens/twenties
                              # -- just above the old 15-trade floor. With 100 trials all
                              # competing to maximize sharpe_like and no penalty for merely
                              # clearing the floor, Optuna was structurally incentivized to
                              # find the smallest sample that dodged the penalty (small
                              # samples have the highest sharpe variance, so that's where
                              # lucky-looking spikes live) -- and holdout caught every one
                              # of them (sharpe went negative on 3 of 5). 50 is still well
                              # below what a genuinely non-overfit config achieves (v43's
                              # breakout got 157.9 tune-effective, v39's squeeze_breakout
                              # got 2491.2) but high enough to rule out the observed
                              # 17-29 overfit zone outright.
UNDER_SAMPLED_PENALTY = -10.0  # well below any realistic sharpe_like, but finite (no inf/NaN into Optuna)
UNDER_SAMPLED_DRAWDOWN_PENALTY = 100.0  # --multi-objective only: worst-possible (100%) drawdown for an
                                         # under-sampled trial, so it's dominated on both axes rather than
                                         # accidentally looking "safe" by having no trades to draw down from
MIN_TRADES_FOR_TRUSTED_DRAWDOWN = 2 * MIN_TRADES_FOR_SCORE  # --multi-objective only. A trial that clears
                                         # MIN_TRADES_FOR_SCORE but is still thin (e.g. 55 vs. another
                                         # trial's 300) reported its raw max_drawdown with no tapering --
                                         # fewer trades mechanically means less time for a bad losing
                                         # streak to compound, so the drawdown axis alone could look
                                         # artificially "safe" purely from selectivity, independent of
                                         # MIN_TRADES_FOR_SCORE already gating the sharpe axis (see
                                         # improvements.txt item 23's own flagged follow-up, and
                                         # taper_drawdown_for_sample_size() below for the fix).

# Search space bounds for the tunables this study is allowed to move.
# A prior run pinned rsi_oversold_threshold at 54.9/55 and stop_loss_atr_multiplier
# at 0.53/0.5 -- both right against their old boundaries, which usually means the
# bound was cutting off the true optimum rather than the search finding an interior
# sweet spot. Widened accordingly; atr_take_profit_multiplier landed interior
# (2.13 of 1.0-3.0) so it's untouched.
RSI_OVERSOLD_RANGE = (15.0, 70.0)
# RRR_FLOOR + these two ranges replaced a version that let
# atr_take_profit_multiplier/stop_loss_atr_multiplier be sampled
# independently -- discovered (2026-08-09) that this let Optuna land on a
# config that wins on backtested $ P&L while being structurally unable to
# ever log a live signal or allocate capital: swingtrade/scoring.py's
# Trade_Score formula treats RRR (= atr_take_profit_multiplier /
# stop_loss_atr_multiplier, a config constant, not ticker-specific) as a
# 0-40-point term, and signal_buy_threshold=60 is mathematically
# unreachable below RRR=1.6 no matter how good Distance_to_Buy_Pct is.
# v19/v27/v28 all landed below 1.6 (0.29-0.41) from independent sampling
# -- three real, capital-allocating strategies that had silently never
# logged a signal since being tuned. See improvements.txt for the full
# incident.
RRR_FLOOR = 1.6
ATR_TAKE_PROFIT_RANGE = (1.0, 5.0)  # widened from (1.0, 3.0) so there's
                                     # real search room above stop_loss_atr_multiplier*RRR_FLOOR
                                     # across the whole STOP_LOSS_ATR_RANGE below
STOP_LOSS_ATR_RANGE = (0.5, 2.5)    # narrowed from (0.25, 5.0) -- at the
                                     # old 5.0 upper bound, RRR_FLOOR would
                                     # require atr_take_profit_multiplier >= 8.0,
                                     # an unrealistically distant target that
                                     # would rarely ever get hit before
                                     # max_holding_days
EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE = (0.0, 4.0)
EXTENDED_DECLINE_PENALTY_CAP_RANGE = (0.0, 50.0)

# Breakout strategy's own search space. A single-window benchmark_random_entry.py
# comparison already showed 20 has no value and 55 shows a real (if modest) edge
# on held-out tickers (see improvements.txt) -- this range lets a real WFO search
# find the actual optimum instead of guessing between two hand-picked points.
BREAKOUT_LOOKBACK_RANGE = (10, 80)
# 100 = practical "disabled" (RSI essentially never reaches it); 50 is a fairly
# restrictive cutoff. Lets Optuna find out whether skipping over-extended
# breakouts (see simulate_breakout_signals) helps at all, and by how much,
# rather than assuming a hand-picked threshold -- this is itself an unvalidated
# hypothesis (improvements.txt item 7a) until a real search says otherwise.
BREAKOUT_RSI_OVERBOUGHT_RANGE = (50.0, 100.0)
# -100 = practical "disabled"; 0 requires the ticker to at least match the
# market over the breakout window, positive values require genuine
# outperformance. Lower bound (-50) is intentionally permissive so the
# search can effectively find "no meaningful filtering" if that's actually
# best, rather than being forced to always filter -- same "let a real
# search decide" reasoning as the overbought threshold above
# (improvements.txt item 5).
BREAKOUT_RELATIVE_STRENGTH_RANGE = (-50.0, 15.0)
# Same range/reasoning as BREAKOUT_RELATIVE_STRENGTH_RANGE, applied to the
# backtest/Optuna-only Sector_Relative_Strength filter (improvements.txt
# items 68/70/71) -- shared across breakout/squeeze_breakout/ma_crossover,
# same permissive lower bound so a search can find "no meaningful
# filtering" if that's actually best rather than being forced to filter.
SECTOR_RELATIVE_STRENGTH_RANGE = (-50.0, 15.0)
# 0.0 = practical "disabled" (a real ratio is always >= 0); upper bound (3.0)
# requires today's volume to be 3x the prior average -- a genuinely
# demanding confirmation threshold. Same "let a real search decide"
# reasoning as the other two breakout filters (improvements.txt item 6).
BREAKOUT_VOLUME_RATIO_RANGE = (0.0, 3.0)
# 0.0 = practical "disabled" (ADX is always >= 0); upper bound (40.0) is
# already a demanding bar in classical TA terms (ADX > 25 is commonly read
# as "trending", > 40 "strong trend"). ADX measures trend STRENGTH,
# independent of direction -- a different dimension than RSI/Relative_Strength/
# Volume_Ratio. Same "let a real search decide" reasoning as the other
# breakout filters (improvements.txt item 17).
BREAKOUT_ADX_MIN_RANGE = (0.0, 40.0)
# Both OBV_Zscore and Squeeze_Zscore (see levels.precompute_breakout_frame)
# are genuine z-scores -- realistically span roughly -3 to +3, so that range
# covers everywhere from "essentially always passes" to "essentially always
# blocks" at either end, same "let a real search decide" reasoning as every
# other breakout filter (improvements.txt item 18).
BREAKOUT_OBV_ZSCORE_MIN_RANGE = (-3.0, 3.0)
BREAKOUT_SQUEEZE_ZSCORE_MAX_RANGE = (-3.0, 3.0)

# Pullback-in-uptrend strategy's own search space. Deliberately compact
# (3 new dimensions, vs. breakout's 6) -- this session's breakout searches
# showed a 6-9D space can under-explore on a 15-45 trial budget (see item
# 17/18's discussion of v22/v24's thin holdout samples); starting smaller
# for a brand-new, unvalidated strategy family is the more conservative
# choice, not a shortcut.
PULLBACK_MA_WINDOW_RANGE = (10, 50)
# Lower bound (3) lets Optuna find "barely needs to be rising" almost-disabled;
# upper bound (30) requires a genuinely sustained uptrend in the MA itself,
# not just a brief tick up -- same "let a real search decide" reasoning as
# every breakout filter range above.
PULLBACK_MA_SLOPE_WINDOW_RANGE = (3, 30)
# Lower bound (0.5) is a tight, demanding band (barely any room around the
# MA); upper bound (10.0) is quite loose (a wide berth still counts as "a
# pullback"). 3.0's default sits in the middle of this range.
PULLBACK_BAND_PCT_RANGE = (0.5, 10.0)

# Breakout-retest strategy's own search space. Reuses BREAKOUT_LOOKBACK_RANGE
# (same "what counts as a breakout" question breakout's own search already
# answers) plus 2 new retest-specific dimensions -- deliberately compact
# (5D total incl. the two shared ATR/stop fields), same conservative
# reasoning as pullback's search space above.
RETEST_WINDOW_DAYS_RANGE = (3, 30)
# Lower bound (3) is a tight window (barely any time to retest); upper
# bound (30) is quite generous (over a month to pull back). 10's default
# sits well within this range.
RETEST_BAND_PCT_RANGE = (0.5, 10.0)
# Same range/reasoning as PULLBACK_BAND_PCT_RANGE -- both are symmetric
# proximity-to-a-level bands with the same "how demanding should this be"
# question.

# 52-week-high momentum strategy's own search space. Even leaner than
# pullback/retest (4D total incl. the two shared ATR/stop fields) -- no
# third "slope"/"underlying lookback" dimension needed, just the window
# and the nearness band.
WEEK52_LOOKBACK_DAYS_RANGE = (100, 252)
# Lower bound (100) is roughly a 5-month high (much shorter than the
# classic "52-week" framing, letting Optuna find out whether a shorter
# window works better); upper bound (252) is the standard full 52-week
# definition -- the search can't go LONGER than the textbook definition,
# only shorter, since going longer has no real academic grounding to test
# against.
WEEK52_NEARNESS_PCT_RANGE = (0.5, 15.0)
# Lower bound (0.5) is a very tight band (must be almost exactly at the
# high); upper bound (15.0) is quite loose (a stock 15% off its highs
# still counts as "near"). 5.0's default sits well within this range.

# Momentum-burst strategy's own search space. Same lean 4D shape as
# pullback/retest/week52 (2 new dimensions + the two shared ATR/stop
# fields) -- built specifically to answer whether tuning resolves this
# strategy's mixed untuned-defaults result (see improvements.txt item 35:
# beats random-entry timing on holdout/aggregate, loses on tune).
MOMENTUM_BURST_GAIN_PCT_RANGE = (1.0, 10.0)
# Lower bound (1.0%) lets Optuna find "barely needs to be a burst"
# almost-loose; upper bound (10.0%) requires a genuinely large single-day
# move. 3.0's default sits well within this range.
MOMENTUM_BURST_VOLUME_RATIO_RANGE = (1.0, 5.0)
# Lower bound (1.0x, merely at-or-above-average volume) is the loosest
# this can go and still mean "confirmation" at all -- unlike breakout's
# own OPTIONAL breakout_volume_ratio_min filter (which can go to 0.0,
# fully disabled), this field IS the trigger's volume-confirmation leg,
# not an add-on gate, so it can't sensibly go below "at least average."
# Upper bound (5.0x) is a genuinely demanding spike. 2.0's default sits
# within this range.

# Squeeze-breakout strategy's own search space. Same lean shape as
# momentum_burst (3 new dimensions + the two shared ATR/stop fields) --
# unlike momentum_burst, this strategy's untuned defaults already showed a
# CLEAN win on every cut under both entry-fill models (see improvements.txt
# item 38), so this search tests whether tuning can improve on an
# already-solid candidate, not rescue a fragile one.
SQUEEZE_BREAKOUT_ZSCORE_MAX_RANGE = (-3.0, -0.25)
# Squeeze_Zscore is a genuine z-score (realistically spans roughly -3 to
# +3, same reasoning as BREAKOUT_SQUEEZE_ZSCORE_MAX_RANGE). Constrained to
# the negative half only -- unlike breakout's own OPTIONAL
# breakout_squeeze_zscore_max filter (which can go all the way to +100,
# fully disabled), this field IS the trigger's contraction-detection leg,
# so a positive/near-zero value wouldn't represent a genuine squeeze at
# all. Upper bound (-0.25) is a mild-but-real contraction; lower bound
# (-3.0) an extremely tight one. -1.0's default sits well within this range.
SQUEEZE_BREAKOUT_LOOKBACK_DAYS_RANGE = (2, 15)
# Lower bound (2) requires the squeeze to be very recent; upper bound (15,
# three trading weeks) is generous -- squeezes can persist a while before
# releasing. 5's default sits well within this range.
SQUEEZE_BREAKOUT_GAIN_PCT_RANGE = (0.5, 6.0)
# Lower bound (0.5%) lets Optuna find "barely needs to be an expansion"
# almost-loose; upper bound (6.0%) requires a genuinely large single-day
# move -- lower ceiling than MOMENTUM_BURST_GAIN_PCT_RANGE's 10.0% since
# this strategy has no volume co-requirement to also demand a big move.
# 2.0's default sits well within this range.

# MA-crossover strategy's own search space -- the leanest of any strategy
# here (2 trigger dimensions + the two shared ATR/stop fields). Unlike
# every other --strategy branch, this one has never been run through
# Optuna at all: it was refined only via manual one-at-a-time grid search
# on short/long window (improvements.txt item 50), with tp/sl held fixed
# throughout at their promoted values. This is the first JOINT search of
# all 4 dimensions together, which is why it's included even though the
# 2-window grid search already found its own local optimum on those 2
# knobs in isolation.
MA_CROSSOVER_SHORT_WINDOW_RANGE = (5, 30)
MA_CROSSOVER_LONG_WINDOW_RANGE = (30, 100)
# Identical bounds to item 50's own grid-search sweep -- that sweep already
# covered this exact range one dimension at a time and found v49's
# untuned baseline (short=20/long=50) was already the best of everything
# tested on TUNE+HOLD together, so these bounds are known-reasonable, not
# a fresh guess.

# slippage_pct / commission_pct_per_trade are deliberately NEVER in this search
# space: they model execution friction, not strategy behavior. Letting Optuna
# tune them would just teach it to zero out the very realism they exist to add.


def build_candidate_config(
    strategy: str, best_params: dict, pin_atr_take_profit_multiplier: float | None,
    pin_stop_loss_atr_multiplier: float | None,
) -> "swingtrade.TradingConfig":
    """Builds the winning trial's full TradingConfig -- factored out of
    main() specifically so this is unit-testable in isolation. `best_params`
    is Optuna's own trial.params dict, which NEVER contains tp/sl when both
    pin args are given (pinned values are hardcoded constants inside
    objective(), never passed through trial.suggest_*(), so they're
    invisible to Optuna's own params dict). Without the explicit override
    below, the candidate would silently fall back to DEFAULT_CONFIG's tp/sl
    instead of the pinned values every trial in the search was actually
    scored under -- a real bug found while re-tuning squeeze_breakout with
    pinning for the first time via this CLI path (see improvements.txt)."""
    pinned_tp_sl = (
        {"atr_take_profit_multiplier": pin_atr_take_profit_multiplier,
         "stop_loss_atr_multiplier": pin_stop_loss_atr_multiplier}
        if pin_atr_take_profit_multiplier is not None and pin_stop_loss_atr_multiplier is not None
        else {}
    )
    return swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy, **best_params, **pinned_tp_sl,
    })


def is_below_frequency_floor(effective_trade_count: float, min_effective_trade_count: float | None) -> bool:
    """Extracted from objective() purely for unit-testability. `None` (the
    default -- --min-frequency-fraction omitted or 0) always returns False,
    preserving every existing search's exact behavior unless explicitly
    opted in. See --min-frequency-fraction's own help text for why this
    exists."""
    return min_effective_trade_count is not None and effective_trade_count < min_effective_trade_count


def is_below_win_rate_floor(win_rate: float | None, min_win_rate: float | None) -> bool:
    """Extracted from objective() purely for unit-testability, same shape
    as is_below_frequency_floor(). `None` (the default -- --min-win-rate
    omitted or 0) always returns False, preserving every existing search's
    exact behavior unless explicitly opted in. `win_rate is None` (the
    under-sampled-to-zero-trades case) is treated as failing the floor --
    a candidate with no measurable win rate can't be trusted to clear one."""
    if min_win_rate is None:
        return False
    return win_rate is None or win_rate < min_win_rate


def taper_drawdown_for_sample_size(max_dd: float, effective_trade_count: float) -> float:
    """--multi-objective only. Linearly tapers a trial's raw max_drawdown
    toward UNDER_SAMPLED_DRAWDOWN_PENALTY as effective_trade_count
    approaches MIN_TRADES_FOR_SCORE from above -- extends the existing
    binary under-sampled gate (which already fully penalizes anything
    BELOW MIN_TRADES_FOR_SCORE) to also distrust anything only barely
    above it, rather than trusting the raw number the moment it clears
    the floor. At MIN_TRADES_FOR_TRUSTED_DRAWDOWN (2x the floor) and
    above, returns max_dd completely unchanged. Below MIN_TRADES_FOR_SCORE
    this is never called -- that case is still handled by the existing
    binary `under_sampled` check in objective()."""
    if effective_trade_count >= MIN_TRADES_FOR_TRUSTED_DRAWDOWN:
        return max_dd
    span = MIN_TRADES_FOR_TRUSTED_DRAWDOWN - MIN_TRADES_FOR_SCORE
    fraction = (effective_trade_count - MIN_TRADES_FOR_SCORE) / span
    fraction = min(max(fraction, 0.0), 1.0)
    return UNDER_SAMPLED_DRAWDOWN_PENALTY + fraction * (max_dd - UNDER_SAMPLED_DRAWDOWN_PENALTY)


def split_tickers_holdout(
    tickers: list[str], sector_lookup: dict[str, str], holdout_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Partition `tickers` into (tune, holdout) sets, stratified by sector so
    neither set accidentally over-concentrates in one sector (the exact
    mechanism behind the original live-vs-backtest gap -- see
    backtest_diagnostic.txt). Deterministic given the same seed. A sector
    with only 1-2 tickers may contribute 0 to holdout (round() can floor to
    0) -- expected, not a bug, with small watchlists or small sectors."""
    if holdout_frac <= 0:
        return list(tickers), []

    rng = random.Random(seed)
    by_sector: dict[str, list[str]] = defaultdict(list)
    for t in tickers:
        by_sector[sector_lookup.get(t, "Unknown")].append(t)

    tune, holdout = [], []
    for sector in sorted(by_sector):
        group = sorted(by_sector[sector])
        rng.shuffle(group)
        n_holdout = round(len(group) * holdout_frac)
        holdout.extend(group[:n_holdout])
        tune.extend(group[n_holdout:])
    return tune, holdout


# Shared default seed set for average_holdout_summary() -- the same 10 seeds
# used in the 2026-08-15 holdout-noise investigation (improvements.txt item
# 69), so results stay comparable to that investigation's own numbers.
DEFAULT_HOLDOUT_SEEDS = [42, 1, 5, 7, 13, 99, 123, 2024, 2025, 777]


def average_holdout_summary(
    trades: list[dict], sector_lookup: dict, holdout_frac: float, seeds: list[int], summarize_fn,
) -> tuple[dict, dict]:
    """Average TUNE/HOLDOUT summary stats across multiple ticker-holdout
    seeds instead of trusting one -- a single fixed-seed split_tickers_holdout()
    draw carries real, substantial sampling noise at typical holdout sizes
    (~100 tickers): a companion neutral no-strategy buy-and-hold check found
    the SAME 102-ticker sample swinging total-return gaps between TUNE and
    HOLDOUT by -44pp to +54pp purely from which tickers land where, and two
    real strategy checks (breakout v43, sector relative-strength) both had
    their HOLDOUT verdict flip direction across just 3 different seeds while
    their ALL/TUNE cuts stayed stable. See improvements.txt item 69.

    `trades` should already be the FULL, already-computed trade list (every
    ticker, one simulation pass) -- this function only re-partitions and
    re-summarizes it under each seed, no re-simulation, so averaging over
    many seeds is cheap. `summarize_fn` is the caller's own summarize()
    (wrapping swingtrade.compute_cluster_weights + summarize_trades_weighted).

    Returns (tune_avg, holdout_avg), each a dict with every numeric field
    from summarize_fn averaged across seeds (None values skipped, not
    treated as 0 -- annualized_sharpe_like/trades_per_year can legitimately
    be None on a thin per-seed split), PLUS `_sharpe_like_min`/
    `_sharpe_like_max`/`_seeds_used` so the spread itself stays visible --
    hiding the instability behind one averaged number would repeat today's
    mistake at one more level of abstraction."""

    def _average_bucket(per_seed_summaries: list[dict]) -> dict:
        # A key is "numeric" if every value seen for it, across every seed,
        # is either None or a real number -- so a field that's legitimately
        # None in EVERY seed (e.g. sharpe_like on an empty holdout) still
        # gets included and averages to None, while a genuinely non-numeric
        # field (a label string, say) is excluded entirely even if it's
        # None in some seeds.
        candidate_keys, excluded_keys = set(), set()
        for s in per_seed_summaries:
            for k, v in s.items():
                if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)):
                    candidate_keys.add(k)
                else:
                    excluded_keys.add(k)
        numeric_keys = candidate_keys - excluded_keys

        averaged = {}
        for key in numeric_keys:
            values = [s[key] for s in per_seed_summaries if s.get(key) is not None]
            averaged[key] = round(sum(values) / len(values), 4) if values else None
        sharpe_values = [s["sharpe_like"] for s in per_seed_summaries if s.get("sharpe_like") is not None]
        averaged["_sharpe_like_min"] = round(min(sharpe_values), 4) if sharpe_values else None
        averaged["_sharpe_like_max"] = round(max(sharpe_values), 4) if sharpe_values else None
        averaged["_seeds_used"] = len(per_seed_summaries)
        return averaged

    tune_summaries = []
    holdout_summaries = []
    for seed in seeds:
        _, holdout_tickers = split_tickers_holdout(
            sorted({t["ticker"] for t in trades}), sector_lookup, holdout_frac, seed
        )
        holdout_set = set(holdout_tickers)
        tune_trades = [t for t in trades if t["ticker"] not in holdout_set]
        holdout_trades = [t for t in trades if t["ticker"] in holdout_set]
        tune_summaries.append(summarize_fn(tune_trades))
        holdout_summaries.append(summarize_fn(holdout_trades))

    return _average_bucket(tune_summaries), _average_bucket(holdout_summaries)


def fold_weight(fold: swingtrade.Fold, end: pd.Timestamp, half_life_days: float) -> float:
    """Exponential recency weight for one fold's out-of-sample window: 1.0
    at `end`, halving every `half_life_days`. half_life_days <= 0 disables
    weighting (every fold counts equally, matching the old behavior)."""
    if half_life_days <= 0:
        return 1.0
    age_days = max((end - fold.out_sample_end).days, 0)
    return 0.5 ** (age_days / half_life_days)


def summarize_weighted(fold_results: list, end: pd.Timestamp, half_life_days: float) -> dict:
    """Like swingtrade.summarize_trades, but pools out-of-sample trades
    weighted by BOTH fold recency (the current market regime counts for
    more than one from a year ago) AND same-day/same-sector correlation
    (see swingtrade.compute_cluster_weights -- 30 correlated Technology
    signals firing one day are one effective observation, not 30
    independent ones). Folds don't overlap in calendar time, so clustering
    across the whole pooled set is equivalent to clustering per-fold and
    concatenating. Uses Kish's effective sample size (not raw trade_count)
    for MIN_TRADES_FOR_SCORE, so heavy weighting -- from either source --
    can't let a big pile of low-weight trades masquerade as a trustworthy
    sample."""
    flat_trades, recency_weights = [], []
    for fr in fold_results:
        w = fold_weight(fr.fold, end, half_life_days)
        for t in fr.out_sample_trades:
            if t["status"] == "OPEN":
                continue
            flat_trades.append(t)
            recency_weights.append(w)

    if not flat_trades:
        return swingtrade.summarize_trades_weighted([], [])

    cluster_weights = swingtrade.compute_cluster_weights(flat_trades)
    combined_weights = [r * c for r, c in zip(recency_weights, cluster_weights)]
    return swingtrade.summarize_trades_weighted(flat_trades, combined_weights)


def build_objective(
    ticker_data: dict, market_data: pd.DataFrame, folds: list, end: pd.Timestamp, half_life_days: float,
    sector_lookup: dict[str, str], strategy: str = "rsi", max_workers: int | None = None,
    multi_objective: bool = False,
    pin_atr_take_profit_multiplier: float | None = None,
    pin_stop_loss_atr_multiplier: float | None = None,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
    min_effective_trade_count: float | None = None,
    min_win_rate: float | None = None,
    sector_data: dict[str, pd.DataFrame] | None = None,
):
    """`multi_objective=False` (default) is byte-for-byte the original
    single-scalar-objective behavior -- unchanged, so every existing script/
    workflow that assumes a single-objective study (study.best_trial, the
    candidate-writing flow in main()) keeps working exactly as before.

    `multi_objective=True` makes the objective return a 2-tuple instead:
    (sharpe_like, max_drawdown) -- see swingtrade.compute_max_drawdown()'s
    docstring for why sharpe_like alone is blind to tail risk. Optuna then
    searches for the Pareto front (configs where no other trial is BOTH a
    higher sharpe_like AND a lower drawdown) instead of a single winner --
    see main()'s post-search handling for how one candidate still gets
    selected from that front to actually write to System_Config.

    `pin_atr_take_profit_multiplier`/`pin_stop_loss_atr_multiplier`: when
    BOTH are given, tp/sl stop being search dimensions entirely -- every
    trial gets this exact, identical payoff shape, and Optuna only tunes
    the strategy's other (filter/trigger) params. Added after breakout's
    own re-tune (2026-08-09/10, see improvements.txt item 42) proved that
    letting Optuna freely search tp/sl (even RRR_FLOOR-constrained)
    simultaneously with every filter threshold gives it enough freedom to
    cherry-pick a tiny, overfit, sometimes holdout-negative sample -- the
    fix that actually worked was pinning tp/sl at a separately-verified
    value and searching nothing else in that dimension. If only one or
    neither is given, falls back to the RRR_FLOOR-respecting sampling
    below unchanged.

    `min_effective_trade_count`: if given, any trial whose own TUNE-set
    effective_trade_count falls below this value is treated as under-
    sampled (same UNDER_SAMPLED_PENALTY/UNDER_SAMPLED_DRAWDOWN_PENALTY
    path as failing MIN_TRADES_FOR_SCORE) regardless of how good its
    sharpe_like/drawdown looks. None (default) disables this entirely --
    see --min-frequency-fraction's own help text for why this exists:
    without it, a free multi-filter search has a structural incentive to
    stack tight filters together and collapse trade frequency, since a
    smaller sample naturally looks better on risk-adjusted metrics alone.

    `min_win_rate`: same shape, but on win_rate directly instead of
    frequency -- sharpe_like/drawdown alone can favor a config with a low
    win rate but rare large winners; if the user wants win_rate to
    actually gate selection (not just be reported), any trial below this
    threshold is rejected the same way. None (default) disables this
    entirely -- see --min-win-rate's own help text.

    `sector_data` (optional, sector NAME -> OHLCV, see watchlist.SECTOR_ETF)
    backs each trial's backtest/Optuna-only *_sector_relative_strength_min
    search dimension for breakout/squeeze_breakout/ma_crossover
    (improvements.txt items 68/70/71) -- threaded straight through to
    run_walk_forward(). None (default) means every trial's
    Sector_Relative_Strength reads None/NaN, same as before this
    parameter existed."""
    def objective(trial: optuna.Trial):
        # Shared by every strategy branch below -- hoisted out of the
        # per-strategy dicts (used to be 7 duplicated independent-sampling
        # call-site pairs) so RRR_FLOOR is enforced by construction in
        # exactly one place. See RRR_FLOOR's own comment above for why:
        # sampling these two independently let Optuna land on configs that
        # win on backtested $ P&L while being structurally unable to ever
        # clear swingtrade/scoring.py's signal_buy_threshold live.
        if pin_atr_take_profit_multiplier is not None and pin_stop_loss_atr_multiplier is not None:
            stop_loss_atr_multiplier = pin_stop_loss_atr_multiplier
            atr_take_profit_multiplier = pin_atr_take_profit_multiplier
            trial.set_user_attr("pinned_tp_sl", True)
        else:
            stop_loss_atr_multiplier = trial.suggest_float("stop_loss_atr_multiplier", *STOP_LOSS_ATR_RANGE)
            atr_take_profit_multiplier = trial.suggest_float(
                "atr_take_profit_multiplier",
                max(ATR_TAKE_PROFIT_RANGE[0], stop_loss_atr_multiplier * RRR_FLOOR),
                ATR_TAKE_PROFIT_RANGE[1],
            )

        if strategy == "rsi":
            params = {
                "rsi_oversold_threshold": trial.suggest_float("rsi_oversold_threshold", *RSI_OVERSOLD_RANGE),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
                "extended_decline_penalty_per_day": trial.suggest_float(
                    "extended_decline_penalty_per_day", *EXTENDED_DECLINE_PENALTY_PER_DAY_RANGE
                ),
                "extended_decline_penalty_cap": trial.suggest_float(
                    "extended_decline_penalty_cap", *EXTENDED_DECLINE_PENALTY_CAP_RANGE
                ),
            }
        elif strategy == "breakout":
            params = {
                "breakout_lookback_days": trial.suggest_int("breakout_lookback_days", *BREAKOUT_LOOKBACK_RANGE),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
                "breakout_rsi_overbought_threshold": trial.suggest_float(
                    "breakout_rsi_overbought_threshold", *BREAKOUT_RSI_OVERBOUGHT_RANGE
                ),
                "breakout_relative_strength_min": trial.suggest_float(
                    "breakout_relative_strength_min", *BREAKOUT_RELATIVE_STRENGTH_RANGE
                ),
                "breakout_volume_ratio_min": trial.suggest_float(
                    "breakout_volume_ratio_min", *BREAKOUT_VOLUME_RATIO_RANGE
                ),
                "breakout_adx_min": trial.suggest_float("breakout_adx_min", *BREAKOUT_ADX_MIN_RANGE),
                "breakout_obv_zscore_min": trial.suggest_float(
                    "breakout_obv_zscore_min", *BREAKOUT_OBV_ZSCORE_MIN_RANGE
                ),
                "breakout_squeeze_zscore_max": trial.suggest_float(
                    "breakout_squeeze_zscore_max", *BREAKOUT_SQUEEZE_ZSCORE_MAX_RANGE
                ),
                "breakout_sector_relative_strength_min": trial.suggest_float(
                    "breakout_sector_relative_strength_min", *SECTOR_RELATIVE_STRENGTH_RANGE
                ),
            }
        elif strategy == "pullback":
            params = {
                "pullback_ma_window": trial.suggest_int("pullback_ma_window", *PULLBACK_MA_WINDOW_RANGE),
                "pullback_ma_slope_window": trial.suggest_int(
                    "pullback_ma_slope_window", *PULLBACK_MA_SLOPE_WINDOW_RANGE
                ),
                "pullback_band_pct": trial.suggest_float("pullback_band_pct", *PULLBACK_BAND_PCT_RANGE),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        elif strategy == "breakout_retest":
            params = {
                "breakout_lookback_days": trial.suggest_int("breakout_lookback_days", *BREAKOUT_LOOKBACK_RANGE),
                "retest_window_days": trial.suggest_int("retest_window_days", *RETEST_WINDOW_DAYS_RANGE),
                "retest_band_pct": trial.suggest_float("retest_band_pct", *RETEST_BAND_PCT_RANGE),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        elif strategy == "week52_high":
            params = {
                "week52_lookback_days": trial.suggest_int("week52_lookback_days", *WEEK52_LOOKBACK_DAYS_RANGE),
                "week52_nearness_pct": trial.suggest_float("week52_nearness_pct", *WEEK52_NEARNESS_PCT_RANGE),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        elif strategy == "momentum_burst":
            params = {
                "momentum_burst_gain_pct_min": trial.suggest_float(
                    "momentum_burst_gain_pct_min", *MOMENTUM_BURST_GAIN_PCT_RANGE
                ),
                "momentum_burst_volume_ratio_min": trial.suggest_float(
                    "momentum_burst_volume_ratio_min", *MOMENTUM_BURST_VOLUME_RATIO_RANGE
                ),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        elif strategy == "squeeze_breakout":
            params = {
                "squeeze_breakout_zscore_max": trial.suggest_float(
                    "squeeze_breakout_zscore_max", *SQUEEZE_BREAKOUT_ZSCORE_MAX_RANGE
                ),
                "squeeze_breakout_lookback_days": trial.suggest_int(
                    "squeeze_breakout_lookback_days", *SQUEEZE_BREAKOUT_LOOKBACK_DAYS_RANGE
                ),
                "squeeze_breakout_gain_pct_min": trial.suggest_float(
                    "squeeze_breakout_gain_pct_min", *SQUEEZE_BREAKOUT_GAIN_PCT_RANGE
                ),
                # Phase 2 sharpening filters (improvements.txt item 42/43) --
                # reuse breakout's own range constants directly, same
                # underlying indicators, no need for per-strategy duplicates.
                "squeeze_breakout_rsi_overbought_threshold": trial.suggest_float(
                    "squeeze_breakout_rsi_overbought_threshold", *BREAKOUT_RSI_OVERBOUGHT_RANGE
                ),
                "squeeze_breakout_relative_strength_min": trial.suggest_float(
                    "squeeze_breakout_relative_strength_min", *BREAKOUT_RELATIVE_STRENGTH_RANGE
                ),
                "squeeze_breakout_volume_ratio_min": trial.suggest_float(
                    "squeeze_breakout_volume_ratio_min", *BREAKOUT_VOLUME_RATIO_RANGE
                ),
                "squeeze_breakout_adx_min": trial.suggest_float(
                    "squeeze_breakout_adx_min", *BREAKOUT_ADX_MIN_RANGE
                ),
                "squeeze_breakout_obv_zscore_min": trial.suggest_float(
                    "squeeze_breakout_obv_zscore_min", *BREAKOUT_OBV_ZSCORE_MIN_RANGE
                ),
                "squeeze_breakout_sector_relative_strength_min": trial.suggest_float(
                    "squeeze_breakout_sector_relative_strength_min", *SECTOR_RELATIVE_STRENGTH_RANGE
                ),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        else:
            params = {
                "ma_crossover_short_window": trial.suggest_int(
                    "ma_crossover_short_window", *MA_CROSSOVER_SHORT_WINDOW_RANGE
                ),
                "ma_crossover_long_window": trial.suggest_int(
                    "ma_crossover_long_window", *MA_CROSSOVER_LONG_WINDOW_RANGE
                ),
                "ma_crossover_sector_relative_strength_min": trial.suggest_float(
                    "ma_crossover_sector_relative_strength_min", *SECTOR_RELATIVE_STRENGTH_RANGE
                ),
                "atr_take_profit_multiplier": atr_take_profit_multiplier,
                "stop_loss_atr_multiplier": stop_loss_atr_multiplier,
            }
        candidate = swingtrade.TradingConfig(**{
            **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy, **params,
        })

        fold_results = swingtrade.run_walk_forward(
            ticker_data, market_data, folds, candidate, earnings_data=earnings_data,
            sector_lookup=sector_lookup, strategy=strategy, max_workers=max_workers,
            sector_data=sector_data,
        )
        metrics = summarize_weighted(fold_results, end, half_life_days)
        trial.set_user_attr("metrics", metrics)

        under_sampled = metrics["effective_trade_count"] < MIN_TRADES_FOR_SCORE or metrics["sharpe_like"] is None
        under_sampled = under_sampled or is_below_frequency_floor(
            metrics["effective_trade_count"], min_effective_trade_count
        )
        under_sampled = under_sampled or is_below_win_rate_floor(metrics["win_rate"], min_win_rate)

        if not multi_objective:
            return UNDER_SAMPLED_PENALTY if under_sampled else metrics["sharpe_like"]

        max_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(fold_results))
        trial.set_user_attr("max_drawdown", max_dd)
        if under_sampled:
            return UNDER_SAMPLED_PENALTY, UNDER_SAMPLED_DRAWDOWN_PENALTY
        if max_dd is None:
            return metrics["sharpe_like"], UNDER_SAMPLED_DRAWDOWN_PENALTY
        tapered_dd = taper_drawdown_for_sample_size(max_dd, metrics["effective_trade_count"])
        trial.set_user_attr("tapered_max_drawdown", tapered_dd)
        return metrics["sharpe_like"], tapered_dd

    return objective


def report_live_outcomes_context() -> None:
    """Informational only -- see module docstring for why live outcomes are
    kept out of the per-trial score itself (recency-weighting is the
    mechanism that lets the score respond to what's working lately). Splits
    out confirmed fills (see confirm_fill.py) from every mechanical signal's
    hypothetical outcome, AND splits actionable (Strong Buy/Buy) from
    research (Watch) tier (see storage/signals.py) -- most logged signals
    were never actually traded, and the research tier specifically was
    never meant to be, so the pooled-everything number overstates both the
    sample of real trades and the sample of signals you'd actually act on."""
    try:
        db = storage.get_db()
    except storage.MongoNotConfigured:
        return
    docs = list(db[storage.outcomes.COLLECTION_NAME].find({}))
    if not docs:
        print("Live Trade_Outcomes so far: 0 (too early to factor into scoring).")
        return
    as_trades = lambda ds: [{"status": d["status"], "pnl_pct": d["pnl_pct"]} for d in ds]  # noqa: E731
    print(f"Live Trade_Outcomes so far (every signal, all tiers): {len(docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(docs))}")

    actionable_docs = [d for d in docs if d.get("tier", "actionable") == "actionable"]
    research_docs = [d for d in docs if d.get("tier") == "research"]
    loosened_docs = [d for d in docs if d.get("tier") == "research_loosened"]
    print(f"  Actionable tier (Strong Buy/Buy) only: {len(actionable_docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(actionable_docs))}")

    confirmed_docs = [d for d in actionable_docs if d.get("confirmed_filled")]
    if confirmed_docs:
        print(f"    ...of which CONFIRMED real fills: {len(confirmed_docs)} -- "
              f"pooled metrics: {swingtrade.summarize_trades(as_trades(confirmed_docs))}")
    else:
        print("    ...of which CONFIRMED real fills: 0 -- see confirm_fill.py.")

    print(f"  Research tier (Watch, never traded/tradeable) only: {len(research_docs)} -- "
          f"pooled metrics: {swingtrade.summarize_trades(as_trades(research_docs))}")

    if loosened_docs:
        print(f"  Research_loosened tier (active config scored Ignore, loosened config didn't, "
              f"never traded/tradeable) only: {len(loosened_docs)} -- "
              f"pooled metrics: {swingtrade.summarize_trades(as_trades(loosened_docs))}. "
              "Reflects the loosened config, NOT the active one -- a sanity check on whether "
              "loosening is worth considering, not a report card on v19 itself.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strategy",
        choices=["rsi", "breakout", "pullback", "breakout_retest", "week52_high", "momentum_burst", "squeeze_breakout", "ma_crossover"],
        default="rsi",
        help="Which signal to search parameters for. Default: rsi.",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 1y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--in-sample-days", type=int, default=182)
    parser.add_argument("--out-sample-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed, for reproducible searches.")
    parser.add_argument(
        "--recency-half-life-days", type=float, default=180.0,
        help="Out-of-sample folds this many days before --end count half as much as the most "
             "recent fold (exponential decay). 0 disables weighting (uniform pooling).",
    )
    parser.add_argument(
        "--holdout-frac", type=float, default=0.25,
        help="Fraction of tickers (stratified by sector) held out from tuning entirely, "
             "then used once at the end to validate the winning config on names Optuna "
             "never saw. 0 disables (tune and validate on every ticker, old behavior).",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=42,
        help="Seed for the ticker tune/holdout split, so the same split is reproducible run to run.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
        help="CPU cores used to parallelize walk-forward folds (see swingtrade.run_walk_forward). "
             "Defaults to cpu_count()-4 (leaves headroom for the OS/other apps and keeps sustained "
             "thermal load down) rather than using every core -- pass a higher number (or your full "
             "core count) for maximum speed, or 1 to fall back to the old sequential behavior.",
    )
    parser.add_argument(
        "--multi-objective", action="store_true",
        help="Also optimize for max_drawdown (see swingtrade.compute_max_drawdown), not just "
             "sharpe_like -- finds the Pareto front (configs where no trial is both a higher "
             "sharpe_like AND a lower drawdown) instead of a single scalar winner, then picks "
             "the highest-sharpe_like trial on that front (still clearing MIN_TRADES_FOR_SCORE) "
             "to actually write as a candidate. Off by default -- the original single-objective "
             "search is unchanged unless you pass this.",
    )
    parser.add_argument(
        "--pin-atr-take-profit-multiplier", type=float, default=None,
        help="Pin atr_take_profit_multiplier to this exact value for every trial (must be passed "
             "together with --pin-stop-loss-atr-multiplier) -- tp/sl stops being searched at all, "
             "Optuna only tunes the strategy's other params. See build_objective()'s docstring for "
             "why: letting Optuna search tp/sl AND every filter simultaneously can cherry-pick a "
             "tiny overfit sample even within the RRR_FLOOR-constrained space.",
    )
    parser.add_argument(
        "--pin-stop-loss-atr-multiplier", type=float, default=None,
        help="Pin stop_loss_atr_multiplier to this exact value -- see --pin-atr-take-profit-multiplier.",
    )
    parser.add_argument(
        "--min-frequency-fraction", type=float, default=0.0,
        help="Reject any trial whose own TUNE-set effective_trade_count falls below this "
             "fraction of DEFAULT_CONFIG's baseline effective_trade_count on the same tickers "
             "-- 0.0 (default) disables this gate entirely, preserving every existing search's "
             "exact behavior. Added after a real free-filter search (squeeze_breakout, see "
             "improvements.txt) found a candidate with a genuine timing edge that also fired "
             "~11x less often than baseline: with 8 free filter dimensions and no cost for "
             "reduced frequency, Optuna has a structural incentive to stack tight filters "
             "together (a smaller, more selective sample naturally looks better on risk-"
             "adjusted metrics alone). A trial that fails this gate is treated exactly like "
             "the existing under-sampled case (UNDER_SAMPLED_PENALTY / "
             "UNDER_SAMPLED_DRAWDOWN_PENALTY), not a new penalty scale.",
    )
    parser.add_argument(
        "--min-win-rate", type=float, default=0.0,
        help="Reject any trial whose own win_rate (0-100 scale) falls below this value -- "
             "0.0 (default) disables this gate entirely, preserving every existing search's "
             "exact behavior. win_rate is already reported in every metrics dict but plays no "
             "role in what Optuna actually selects (only sharpe_like, and optionally "
             "max_drawdown, do) -- this makes it an explicit, opt-in selection criterion "
             "instead of just a number to read afterward. A trial that fails this gate is "
             "treated exactly like the existing under-sampled case (UNDER_SAMPLED_PENALTY / "
             "UNDER_SAMPLED_DRAWDOWN_PENALTY), not a new penalty scale or a 3rd Optuna "
             "objective -- see is_below_win_rate_floor().",
    )
    parser.add_argument(
        "--with-catalyst", action="store_true",
        help="Fetch historical earnings dates (one extra yfinance call per ticker, see "
             "run_backtest.fetch_earnings_dates) so Catalyst_Warning is computed honestly "
             "instead of always False during this search. Only affects --strategy "
             "squeeze_breakout currently (the only strategy here whose simulate_*_signals() "
             "accepts earnings_dates) -- a no-op flag for every other --strategy value. Off "
             "by default, same convention as run_backtest.py's own --with-catalyst.",
    )
    args = parser.parse_args()

    if (args.pin_atr_take_profit_multiplier is None) != (args.pin_stop_loss_atr_multiplier is None):
        print("[ERROR] --pin-atr-take-profit-multiplier and --pin-stop-loss-atr-multiplier "
              "must be passed together, or not at all.", file=sys.stderr)
        sys.exit(1)

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Fetching history for {len(tickers)} ticker(s) + {MARKET_INDEX_TICKER}, "
          f"{start.date()}..{end.date()} (once, reused across all {args.trials} trials)...")
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_data.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    # Sector-ETF data (backtest/Optuna-only Sector_Relative_Strength filter,
    # improvements.txt items 68/70/71) -- only the sectors actually present
    # in this run's own ticker universe, fetched once regardless of how
    # many tickers share a sector, same "fetch once, reuse" discipline as
    # market_data above.
    sector_data: dict[str, pd.DataFrame] = {}
    present_sectors = sorted({sector_lookup[t] for t in tickers if t in sector_lookup} & set(SECTOR_ETF))
    if present_sectors:
        print(f"Fetching {len(present_sectors)} sector ETF(s) for Sector_Relative_Strength...")
        for i, sector in enumerate(present_sectors):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            etf_df = fetch_history(SECTOR_ETF[sector], start, end)
            if not etf_df.empty:
                sector_data[sector] = etf_df

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_history(ticker, start, end)
            if not df.empty:
                ticker_data[ticker] = df
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}", file=sys.stderr)
    print(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s).")
    if not ticker_data:
        print("[ERROR] No ticker data available.", file=sys.stderr)
        sys.exit(1)

    earnings_data = {}
    if args.with_catalyst and args.strategy == "squeeze_breakout":
        print(f"\nFetching historical earnings dates for {len(ticker_data)} ticker(s)...")
        for i, ticker in enumerate(ticker_data):
            if i > 0:
                time.sleep(REQUEST_DELAY_SEC)
            earnings_data[ticker] = fetch_earnings_dates(ticker)
        found = sum(1 for d in earnings_data.values() if len(d) > 0)
        print(f"Got earnings history for {found}/{len(ticker_data)} ticker(s).")

    folds = swingtrade.generate_folds(start, end, args.in_sample_days, args.out_sample_days, args.step_days)
    print(f"Generated {len(folds)} walk-forward fold(s).")
    if not folds:
        print("[ERROR] Date range too short for even one fold.", file=sys.stderr)
        sys.exit(1)

    tune_tickers, holdout_tickers = split_tickers_holdout(
        list(ticker_data.keys()), sector_lookup, args.holdout_frac, args.holdout_seed
    )
    tune_ticker_data = {t: ticker_data[t] for t in tune_tickers}
    holdout_ticker_data = {t: ticker_data[t] for t in holdout_tickers}
    if holdout_tickers:
        print(f"Ticker holdout split (seed={args.holdout_seed}, frac={args.holdout_frac}): "
              f"{len(tune_tickers)} tune / {len(holdout_tickers)} holdout. "
              f"Holdout tickers never seen during tuning: {sorted(holdout_tickers)}")
    else:
        print("Ticker holdout validation disabled (--holdout-frac 0 or too few tickers) -- "
              "tuning and reporting on every ticker, old behavior.")

    print(f"Using up to {args.max_workers} CPU core(s) for walk-forward parallelization "
          f"(--max-workers to change; {os.cpu_count()} available).")

    baseline_config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": args.strategy})
    baseline_results = swingtrade.run_walk_forward(
        tune_ticker_data, market_data, folds, baseline_config, earnings_data=earnings_data,
        sector_lookup=sector_lookup, strategy=args.strategy, max_workers=args.max_workers,
        sector_data=sector_data,
    )
    baseline_metrics = summarize_weighted(baseline_results, end, args.recency_half_life_days)
    weight_note = (
        f"(recency-weighted, half-life={args.recency_half_life_days:.0f}d)"
        if args.recency_half_life_days > 0 else "(uniform pooling)"
    )
    print(f"\nBaseline (DEFAULT_CONFIG, strategy={args.strategy}) pooled out-of-sample metrics on TUNE tickers {weight_note}: {baseline_metrics}")
    if args.multi_objective:
        baseline_drawdown = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(baseline_results))
        print(f"  Baseline max_drawdown: {baseline_drawdown}%")
    report_live_outcomes_context()

    min_effective_trade_count = None
    if args.min_frequency_fraction > 0:
        min_effective_trade_count = args.min_frequency_fraction * (baseline_metrics["effective_trade_count"] or 0)
        print(f"\nRejecting any trial with TUNE effective_trade_count below "
              f"{min_effective_trade_count:.1f} ({args.min_frequency_fraction:.0%} of baseline's "
              f"{baseline_metrics['effective_trade_count']:.1f}).")

    min_win_rate = args.min_win_rate if args.min_win_rate > 0 else None
    if min_win_rate is not None:
        print(f"Rejecting any trial with TUNE win_rate below {min_win_rate:.1f}%.")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.multi_objective:
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        build_objective(
            tune_ticker_data, market_data, folds, end, args.recency_half_life_days, sector_lookup, args.strategy,
            max_workers=args.max_workers, multi_objective=args.multi_objective,
            pin_atr_take_profit_multiplier=args.pin_atr_take_profit_multiplier,
            pin_stop_loss_atr_multiplier=args.pin_stop_loss_atr_multiplier,
            earnings_data=earnings_data,
            min_effective_trade_count=min_effective_trade_count,
            min_win_rate=min_win_rate,
            sector_data=sector_data,
        ),
        n_trials=args.trials, show_progress_bar=False,
    )

    def _annualized_and_win_rate(trial) -> str:
        # Pareto/best-trial listings only carry the raw objective value(s)
        # (t.values), not the full metrics dict -- pull annualized_sharpe_like/
        # win_rate from the trial's own stored metrics (set via
        # trial.set_user_attr("metrics", ...) inside objective()) so these
        # are visible inline without hunting through the full dict below.
        m = trial.user_attrs.get("metrics", {})
        ann = m.get("annualized_sharpe_like")
        wr = m.get("win_rate")
        ann_str = f"{ann:.3f}" if ann is not None else "n/a"
        wr_str = f"{wr:.1f}%" if wr is not None else "n/a"
        return f"annualized_sharpe~{ann_str}, win_rate={wr_str}"

    if args.multi_objective:
        pareto = study.best_trials
        print()
        print(f"Pareto front: {len(pareto)} non-dominated trial(s) (sharpe_like, max_drawdown%)")
        for t in sorted(pareto, key=lambda t: -t.values[0]):
            print(f"  Trial #{t.number}: sharpe_like={t.values[0]:.4f}, max_drawdown={t.values[1]:.2f}%  "
                  f"({_annualized_and_win_rate(t)})  params={t.params}")

        eligible = [t for t in pareto if t.values[0] > UNDER_SAMPLED_PENALTY]
        if not eligible:
            print()
            print("[WARN] No Pareto-optimal trial cleared the under-sampled floor -- not writing "
                  "a candidate. Try a wider date range, more tickers, or a longer in-sample window.")
            return
        # Selection heuristic, stated plainly: highest sharpe_like among the
        # Pareto-optimal trials (i.e. among configs where no OTHER trial beat
        # them on both axes simultaneously) -- keeps the existing "one
        # candidate gets written" champion/challenger flow intact rather than
        # asking a human to pick blindly, while the full front (printed
        # above) still shows what was traded off to get there.
        best = max(eligible, key=lambda t: t.values[0])
        best_metrics = best.user_attrs.get("metrics", {})
        best_drawdown = best.user_attrs.get("max_drawdown")
        print()
        print(f"Selected from Pareto front, trial #{best.number}: sharpe_like={best.values[0]:.4f}, "
              f"max_drawdown={best.values[1]:.2f}%  ({_annualized_and_win_rate(best)})")
        print(f"  params: {best.params}")
        print(f"  metrics (TUNE tickers): {best_metrics}")
    else:
        best = study.best_trial
        best_metrics = best.user_attrs.get("metrics", {})
        best_drawdown = None
        print()
        print(f"Best trial #{best.number}: score(sharpe_like or penalty)={best.value:.4f}  "
              f"({_annualized_and_win_rate(best)})")
        print(f"  params: {best.params}")
        print(f"  metrics (TUNE tickers): {best_metrics}")

        if best.value <= UNDER_SAMPLED_PENALTY:
            print()
            print("[WARN] Even the best trial was under-sampled or had no valid sharpe_like -- "
                  "not writing a candidate. Try a wider date range, more tickers, or a longer "
                  "in-sample window.")
            return

    candidate_config = build_candidate_config(
        args.strategy, best.params, args.pin_atr_take_profit_multiplier, args.pin_stop_loss_atr_multiplier,
    )

    holdout_metrics = {}
    if holdout_ticker_data:
        holdout_baseline_results = swingtrade.run_walk_forward(
            holdout_ticker_data, market_data, folds, baseline_config, earnings_data=earnings_data,
            sector_lookup=sector_lookup, strategy=args.strategy, max_workers=args.max_workers,
            sector_data=sector_data,
        )
        holdout_baseline_metrics = summarize_weighted(holdout_baseline_results, end, args.recency_half_life_days)

        holdout_candidate_results = swingtrade.run_walk_forward(
            holdout_ticker_data, market_data, folds, candidate_config, earnings_data=earnings_data,
            sector_lookup=sector_lookup, strategy=args.strategy, max_workers=args.max_workers,
            sector_data=sector_data,
        )
        holdout_candidate_metrics = summarize_weighted(holdout_candidate_results, end, args.recency_half_life_days)

        print()
        print(f"=== Ticker-universe holdout validation ({len(holdout_tickers)} tickers Optuna never saw) ===")
        print(f"  baseline (DEFAULT_CONFIG, strategy={args.strategy}) on holdout: {holdout_baseline_metrics}")
        print(f"  candidate (winning trial) on holdout: {holdout_candidate_metrics}")
        if args.multi_objective:
            holdout_baseline_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(holdout_baseline_results))
            holdout_candidate_dd = swingtrade.compute_max_drawdown(swingtrade.flatten_out_sample_trades(holdout_candidate_results))
            print(f"  baseline max_drawdown on holdout: {holdout_baseline_dd}%")
            print(f"  candidate max_drawdown on holdout: {holdout_candidate_dd}%")
        if (holdout_candidate_metrics.get("effective_trade_count") or 0) < MIN_TRADES_FOR_SCORE:
            print("  [WARN] Holdout effective_trade_count is thin -- treat this validation as a weak "
                  "signal, not proof either way. Consider a lower --holdout-frac or more tickers.")

        holdout_metrics = {
            "holdout_tickers": sorted(holdout_tickers),
            "tune_tickers": sorted(tune_tickers),
            "holdout_frac": args.holdout_frac,
            "holdout_seed": args.holdout_seed,
            "baseline": holdout_baseline_metrics,
            "candidate": holdout_candidate_metrics,
        }
        if args.multi_objective:
            holdout_metrics["baseline_max_drawdown"] = holdout_baseline_dd
            holdout_metrics["candidate_max_drawdown"] = holdout_candidate_dd

    best_sharpe = best.values[0] if args.multi_objective else best.value
    notes = (
        f"Optuna search (strategy={args.strategy}{'  multi-objective' if args.multi_objective else ''}): "
        f"{args.trials} trials, {start.date()}..{end.date()}, "
        f"{len(tune_ticker_data)} tune / {len(holdout_ticker_data)} holdout ticker(s), {len(folds)} fold(s), "
        f"recency_half_life_days={args.recency_half_life_days:.0f}. "
        f"Baseline sharpe_like={baseline_metrics.get('sharpe_like')}, best sharpe_like={best_sharpe:.4f}."
    )
    if args.multi_objective:
        notes += f" Selected trial max_drawdown={best_drawdown}% (Pareto front had {len(pareto)} trial(s))."
    if holdout_metrics:
        notes += (
            f" Holdout ({len(holdout_tickers)} tickers): baseline sharpe_like="
            f"{holdout_metrics['baseline'].get('sharpe_like')}, candidate sharpe_like="
            f"{holdout_metrics['candidate'].get('sharpe_like')}."
        )
    version = storage.write_candidate(
        candidate_config.to_dict(), notes=notes, metrics=best_metrics, holdout_metrics=holdout_metrics
    )

    print()
    print(f"Wrote candidate System_Config version={version} (status=candidate, NOT active).")
    print(f"Review it, then run: python promote_config.py --promote {version}")


if __name__ == "__main__":
    main()
