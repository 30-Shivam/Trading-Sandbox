"""Pure Trade_Score / Signal scoring. Operates on a DataFrame produced by
`swingtrade.levels.compute_levels` -- no fetching, no UI.
"""

import pandas as pd

from .config import DEFAULT_CONFIG, TradingConfig


def signal_for_score(score: float, config: TradingConfig = DEFAULT_CONFIG) -> str:
    if score > config.signal_strong_buy_threshold:
        return "Strong Buy"
    if score >= config.signal_buy_threshold:
        return "Buy"
    if score >= config.signal_watch_threshold:
        return "Watch"
    return "Ignore"


def add_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Blend RRR, RSI, and Distance_to_Buy_Pct into a 0-100 Trade_Score and
    map it to a Strong Buy / Buy / Watch / Ignore Signal.

    A streak penalty is subtracted for tickers whose Oversold_Streak_Days
    (see compute_levels) exceeds extended_decline_warning_days -- without
    it, the RSI component alone rewards "lower RSI" as unconditionally
    better, which can't distinguish a stock that just dipped from one that's
    been falling for a month (a real incident: two tickers with 19-day
    oversold streaks were the top-scored Strong Buys two days running, then
    both breached stop). The penalty is capped so a long streak can push a
    ticker down the ranking but can't single-handedly zero it out."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    rsi_clipped = df["RSI"].clip(lower=config.rsi_score_floor, upper=config.rsi_score_ceiling)
    rsi_score = (
        (config.rsi_score_ceiling - rsi_clipped) / (config.rsi_score_ceiling - config.rsi_score_floor)
    ) * config.rsi_score_weight

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=config.distance_score_cap_pct)
    distance_score = (1 - distance_clipped / config.distance_score_cap_pct) * config.distance_score_weight

    streak_excess = (df["Oversold_Streak_Days"] - config.extended_decline_warning_days).clip(lower=0)
    streak_penalty = (streak_excess * config.extended_decline_penalty_per_day).clip(upper=config.extended_decline_penalty_cap)

    df["Trade_Score"] = (rrr_score + rsi_score + distance_score - streak_penalty).clip(lower=0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_breakout_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Trend-following counterpart to add_trade_score() -- blends RRR and
    Distance_to_Buy_Pct into a 0-100 Trade_Score for rows produced by
    compute_breakout_levels(), reusing signal_for_score()'s thresholds so a
    breakout Trade_Score means the same thing on the same 0-100 scale as an
    RSI one (comparable when both appear in the same allocate_capital() run).

    No RSI component: "lower RSI is better" is RSI-oversold's whole premise,
    and would be backwards here -- a fresh breakout WANTS elevated RSI, that's
    what "buying strength" means. Rather than penalize strong breakouts for
    looking like strong breakouts, the RSI dimension is dropped entirely and
    rrr_score_weight/distance_score_weight are rescaled to still sum to 100,
    preserving their relative emphasis from the active config. No streak
    penalty either -- Oversold_Streak_Days is an RSI-mean-reversion concept
    with no breakout equivalent.

    For breakout, Distance_to_Buy_Pct (see compute_breakout_levels) means
    "how far ABOVE the trigger has price already moved" -- smaller is a
    fresher, less-extended breakout, which is why the same "smaller distance
    scores higher" formula as add_trade_score() applies unchanged.

    Hard gate, not just a scoring input: a ticker whose Breakout_Signal is
    False (hasn't actually broken out yet) gets Trade_Score=0/Ignore, full
    stop -- NOT scored via the formula below. This matters because a ticker
    still below its trigger has a NEGATIVE Distance_to_Buy_Pct, and clipping
    that to 0 (as the formula does for a genuine fresh breakout) would
    otherwise make "hasn't broken out" score identically to "just broke
    out perfectly" -- a real bug caught before this shipped. Also gates out
    breakouts already at/above config.breakout_rsi_overbought_threshold
    (see simulate_breakout_signals -- kept in sync with the backtested
    definition so live and backtested signals can never silently disagree).
    Missing/NaN RSI (insufficient warmup) is treated as NOT overbought --
    RSI is informational for breakout, not a reason to suppress a signal on
    its own. Same treatment for missing/NaN Relative_Strength (see
    compute_relative_strength -- None when market_df wasn't supplied or
    there's insufficient history): not a reason to suppress on its own.
    Also gates out breakouts whose Relative_Strength (ticker return minus
    market return over the same breakout_lookback_days window) is below
    config.breakout_relative_strength_min, whose Volume_Ratio (today's
    Volume over the PRIOR volume_lookback_days average) is below
    config.breakout_volume_ratio_min, whose ADX (trend strength,
    independent of direction) is below config.breakout_adx_min, whose
    OBV_Zscore (On-Balance Volume vs. its own recent baseline) is below
    config.breakout_obv_zscore_min, and whose Squeeze_Zscore (prior-day
    volatility vs. its own recent baseline) is above
    config.breakout_squeeze_zscore_max -- all five kept in sync with
    simulate_breakout_signals so live and backtested definitions can't
    silently disagree."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=config.distance_score_cap_pct)
    distance_score = (1 - distance_clipped / config.distance_score_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + distance_score) * rescale).clip(lower=0)

    not_overbought = df["RSI"].isna() | (df["RSI"] < config.breakout_rsi_overbought_threshold)
    strong_enough = df["Relative_Strength"].isna() | (df["Relative_Strength"] >= config.breakout_relative_strength_min)
    enough_volume = df["Volume_Ratio"].isna() | (df["Volume_Ratio"] >= config.breakout_volume_ratio_min)
    strong_trend = df["ADX"].isna() | (df["ADX"] >= config.breakout_adx_min)
    obv_ok = df["OBV_Zscore"].isna() | (df["OBV_Zscore"] >= config.breakout_obv_zscore_min)
    squeezed = df["Squeeze_Zscore"].isna() | (df["Squeeze_Zscore"] <= config.breakout_squeeze_zscore_max)
    # Sector_Relative_Strength (backtest/Optuna-only, see precompute_breakout_frame) --
    # None/NaN in live production (no sector_df supplied there), so this
    # never excludes a live ticker regardless of the config value.
    sector_strong_enough = (
        df["Sector_Relative_Strength"].isna()
        | (df["Sector_Relative_Strength"] >= config.breakout_sector_relative_strength_min)
    )
    eligible = (
        df["Breakout_Signal"] & not_overbought & strong_enough & enough_volume
        & strong_trend & obv_ok & squeezed & sector_strong_enough
    )

    df["Trade_Score"] = raw_score.where(eligible, 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_pullback_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Pullback-in-uptrend counterpart to add_trade_score()/add_breakout_trade_score()
    -- blends RRR and Distance_to_Buy_Pct into a 0-100 Trade_Score for rows
    produced by compute_pullback_levels(), reusing signal_for_score()'s
    thresholds so a pullback Trade_Score means the same thing on the same
    0-100 scale as the other two strategies (comparable when shown/allocated
    side by side).

    No RSI component, same reasoning as breakout: this strategy isn't
    RSI-gated by design (kept structurally distinct from the already-
    disproven RSI-oversold timing signal -- see benchmark_random_entry.py),
    so there's no RSI dimension to score. rrr_score_weight/distance_score_weight
    are rescaled to still sum to 100, same as add_breakout_trade_score().

    Distance_to_Buy_Pct here means "how far price is from the pullback MA,
    signed" -- can be negative (at/below the MA, within the allowed band)
    or positive (still above it, approaching from strength). The same
    "clip at 0, then smaller distance scores higher" formula as the other
    two strategies applies: at-or-below the MA scores the max distance
    points uniformly (a real, gated pullback is a real, gated pullback,
    however deep within the allowed band), while still being above the MA
    is progressively penalized as it approaches the band's edge.

    Hard gate, not just a scoring input: a ticker whose Pullback_Signal is
    False (not within pullback_band_pct of a rising pullback_ma_window-day
    SMA, in a confirmed macro uptrend -- see compute_pullback_levels) gets
    Trade_Score=0/Ignore, full stop -- same "not eligible at all" semantics
    as add_breakout_trade_score()'s Breakout_Signal gate, not merely a low
    score."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=config.distance_score_cap_pct)
    distance_score = (1 - distance_clipped / config.distance_score_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + distance_score) * rescale).clip(lower=0)

    df["Trade_Score"] = raw_score.where(df["Pullback_Signal"], 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_breakout_retest_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Breakout-retest counterpart to add_trade_score()/add_breakout_trade_score()/
    add_pullback_trade_score() -- blends RRR and Distance_to_Buy_Pct into a
    0-100 Trade_Score for rows produced by compute_breakout_retest_levels(),
    reusing signal_for_score()'s thresholds so a breakout-retest Trade_Score
    means the same thing on the same 0-100 scale as the other three
    strategies.

    No RSI component, same reasoning as breakout/pullback: this strategy
    isn't RSI-gated by design. rrr_score_weight/distance_score_weight are
    rescaled to still sum to 100, same as the other two trend-following
    strategies.

    Distance_to_Buy_Pct here means "how far price is from the original
    breakout's trigger level, signed" -- same "clip at 0, then smaller
    distance scores higher" formula as add_pullback_trade_score(): at-or-
    below the level scores the max distance points uniformly, while still
    being above it is progressively penalized as it approaches the band's
    edge.

    Hard gate, not just a scoring input: a ticker whose Retest_Signal is
    False (no genuine breakout within retest_window_days, or price outside
    retest_band_pct of that breakout's level -- see
    compute_breakout_retest_levels) gets Trade_Score=0/Ignore, full stop --
    same "not eligible at all" semantics as the other two trend-following
    strategies' hard gates, not merely a low score."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=config.distance_score_cap_pct)
    distance_score = (1 - distance_clipped / config.distance_score_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + distance_score) * rescale).clip(lower=0)

    df["Trade_Score"] = raw_score.where(df["Retest_Signal"], 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_week52_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """52-week-high-momentum counterpart to add_trade_score()/add_breakout_trade_score()/
    add_pullback_trade_score()/add_breakout_retest_trade_score() -- blends
    RRR and Distance_to_Buy_Pct into a 0-100 Trade_Score for rows produced
    by compute_week52_levels(), reusing signal_for_score()'s thresholds so
    a week52_high Trade_Score means the same thing on the same 0-100 scale
    as the other four strategies.

    No RSI component, same reasoning as the other trend-following
    strategies. rrr_score_weight/distance_score_weight are rescaled to
    still sum to 100.

    Distance_to_Buy_Pct here means "how far price is BELOW the trailing
    52-week high" (can go slightly negative on a fresh high) -- same "clip
    at 0, then smaller distance scores higher" formula as the other three
    trend-following strategies: at-or-above the high scores the max
    distance points uniformly, while further below is progressively
    penalized as it approaches the band's edge.

    Hard gate, not just a scoring input: a ticker whose Week52_Signal is
    False (further than week52_nearness_pct below its own trailing
    week52_lookback_days high, or outside the macro-uptrend/liquidity
    gates -- see compute_week52_levels) gets Trade_Score=0/Ignore, full
    stop -- same "not eligible at all" semantics as every other
    strategy's hard gate, not merely a low score."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    distance_clipped = df["Distance_to_Buy_Pct"].clip(lower=0, upper=config.distance_score_cap_pct)
    distance_score = (1 - distance_clipped / config.distance_score_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + distance_score) * rescale).clip(lower=0)

    df["Trade_Score"] = raw_score.where(df["Week52_Signal"], 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_momentum_burst_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Momentum-burst counterpart to add_trade_score()/add_breakout_trade_score()/
    add_pullback_trade_score()/add_breakout_retest_trade_score()/add_week52_trade_score()
    -- blends RRR and Distance_to_Buy_Pct into a 0-100 Trade_Score for rows
    produced by compute_momentum_burst_levels(), reusing signal_for_score()'s
    thresholds so a momentum_burst Trade_Score means the same thing on the
    same 0-100 scale as every other strategy.

    No RSI component, same reasoning as the other trend-following
    strategies. rrr_score_weight/distance_score_weight are rescaled to
    still sum to 100.

    Distance_to_Buy_Pct is always 0 for this strategy (Buy_Price IS
    Last_Close, see momentum_burst_levels_from_frame) -- it stays in the
    returned dict for schema compatibility but is NOT used in this
    formula. In its place, Signal_Strength_Pct (how far today's
    Day_Gain_Pct clears momentum_burst_gain_pct_min -- see
    momentum_burst_levels_from_frame) fills the second term, using the
    same clip-then-scale shape rrr_score already uses (bigger is better,
    capped at momentum_burst_strength_cap_pct) so every eligible ticker's
    score actually differentiates by real signal strength instead of
    landing on one fixed value (see improvements.txt for the incident that
    prompted this -- the old Distance_to_Buy_Pct-based formula gave every
    eligible ticker the exact same score, RRR being a config constant
    too).

    Hard gate, not just a scoring input: a ticker whose Momentum_Signal is
    False (didn't clear BOTH the gain and volume thresholds, or is outside
    the macro-uptrend/liquidity gates -- see compute_momentum_burst_levels)
    gets Trade_Score=0/Ignore, full stop -- same "not eligible at all"
    semantics as every other strategy's hard gate, not merely a low score."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    strength_clipped = df["Signal_Strength_Pct"].clip(lower=0, upper=config.momentum_burst_strength_cap_pct)
    strength_score = (strength_clipped / config.momentum_burst_strength_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + strength_score) * rescale).clip(lower=0)

    df["Trade_Score"] = raw_score.where(df["Momentum_Signal"], 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_pairs_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Mean-reversion PAIRS counterpart to add_trade_score()/add_breakout_trade_score()/
    add_squeeze_breakout_trade_score()/etc. -- blends RRR and
    Signal_Strength_Pct (z-score points past pairs_zscore_entry_max) into a
    0-100 Trade_Score for rows produced by pairs_levels_from_frame(),
    reusing signal_for_score()'s thresholds so a pairs Trade_Score means
    the same thing on the same 0-100 scale as every other strategy.

    Distance_to_Buy_Pct is always 0 for this strategy (Buy_Price IS
    Last_Close) -- Signal_Strength_Pct (in z-score units here, not a %)
    fills that role instead, same "differentiating term" pattern
    squeeze_breakout/momentum_burst use.

    Hard gate, not just a scoring input: a ticker whose Pair_Signal is
    False (no partner found/cleared pairs_min_correlation, or the spread
    z-score didn't clear pairs_zscore_entry_max, or outside the shared
    macro-uptrend/liquidity gates) gets Trade_Score=0/Ignore, full stop --
    same "not eligible at all" semantics as every other strategy's hard
    gate, not merely a low score.

    Lean v1, no optional "sharpening" filters yet (unlike breakout/
    squeeze_breakout/adx_trend_entry's own Phase 2 additions) -- mirrors
    how every other strategy in this project launched before any filters
    were added, per this project's own established discipline."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    strength_clipped = df["Signal_Strength_Pct"].clip(lower=0, upper=config.pairs_zscore_strength_cap)
    strength_score = (strength_clipped / config.pairs_zscore_strength_cap) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + strength_score) * rescale).clip(lower=0)

    df["Trade_Score"] = raw_score.where(df["Pair_Signal"], 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_squeeze_breakout_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Squeeze-breakout counterpart to add_trade_score()/add_breakout_trade_score()/
    add_pullback_trade_score()/add_breakout_retest_trade_score()/add_week52_trade_score()/
    add_momentum_burst_trade_score() -- blends RRR and Distance_to_Buy_Pct
    into a 0-100 Trade_Score for rows produced by compute_squeeze_breakout_levels(),
    reusing signal_for_score()'s thresholds so a squeeze_breakout Trade_Score
    means the same thing on the same 0-100 scale as every other strategy.

    No RSI component, same reasoning as the other trend-following
    strategies. rrr_score_weight/distance_score_weight are rescaled to
    still sum to 100.

    Distance_to_Buy_Pct is always 0 for this strategy (Buy_Price IS
    Last_Close, see squeeze_breakout_levels_from_frame) -- it stays in the
    returned dict for schema compatibility but is NOT used in this
    formula. In its place, Signal_Strength_Pct (how far today's
    Day_Gain_Pct clears squeeze_breakout_gain_pct_min -- see
    squeeze_breakout_levels_from_frame) fills the second term, same
    clip-then-scale shape as rrr_score (bigger is better, capped at
    squeeze_breakout_strength_cap_pct) -- see improvements.txt for why:
    the old Distance_to_Buy_Pct-based formula gave every eligible ticker
    the exact same fixed score.

    Hard gate, not just a scoring input: a ticker whose Squeeze_Signal is
    False (no genuine recent squeeze, or today's gain didn't clear the
    bar, or outside the macro-uptrend/liquidity gates -- see
    compute_squeeze_breakout_levels) gets Trade_Score=0/Ignore, full stop
    -- same "not eligible at all" semantics as every other strategy's
    hard gate, not merely a low score.

    Phase 2 (improvements.txt item 42/43): also gates out tickers whose RSI
    is at/above config.squeeze_breakout_rsi_overbought_threshold, whose
    Relative_Strength is below config.squeeze_breakout_relative_strength_min,
    whose Volume_Ratio is below config.squeeze_breakout_volume_ratio_min,
    whose ADX is below config.squeeze_breakout_adx_min, or whose OBV_Zscore
    is below config.squeeze_breakout_obv_zscore_min -- all five kept in
    sync with simulate_squeeze_breakout_signals so live and backtested
    definitions can't silently disagree, and all five default to a
    practical no-op (missing/NaN values also never exclude a ticker on
    their own), same "disabled until explicitly tuned" treatment as
    breakout's own six filters / adx_trend_entry's five.

    A sixth optional gate, config.squeeze_breakout_earnings_gate (boolean,
    default False), excludes a ticker whose Catalyst_Warning is True --
    kept in sync with simulate_squeeze_breakout_signals the same way."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    strength_clipped = df["Signal_Strength_Pct"].clip(lower=0, upper=config.squeeze_breakout_strength_cap_pct)
    strength_score = (strength_clipped / config.squeeze_breakout_strength_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + strength_score) * rescale).clip(lower=0)

    not_overbought = df["RSI"].isna() | (df["RSI"] < config.squeeze_breakout_rsi_overbought_threshold)
    strong_enough = df["Relative_Strength"].isna() | (df["Relative_Strength"] >= config.squeeze_breakout_relative_strength_min)
    enough_volume = df["Volume_Ratio"].isna() | (df["Volume_Ratio"] >= config.squeeze_breakout_volume_ratio_min)
    strong_trend = df["ADX"].isna() | (df["ADX"] >= config.squeeze_breakout_adx_min)
    obv_ok = df["OBV_Zscore"].isna() | (df["OBV_Zscore"] >= config.squeeze_breakout_obv_zscore_min)
    # Sector_Relative_Strength (backtest/Optuna-only) -- same graceful
    # None/NaN-never-excludes treatment as breakout's own.
    sector_strong_enough = (
        df["Sector_Relative_Strength"].isna()
        | (df["Sector_Relative_Strength"] >= config.squeeze_breakout_sector_relative_strength_min)
    )
    # earnings_gate is a boolean on/off toggle, not a numeric threshold (see
    # config.squeeze_breakout_earnings_gate) -- disabled (default) means no
    # exclusion at all, same "missing/NaN never excludes on its own"
    # treatment as every other filter above when a threshold isn't binding.
    not_near_earnings = (
        (df["Catalyst_Warning"].isna() | ~df["Catalyst_Warning"])
        if config.squeeze_breakout_earnings_gate else True
    )
    eligible = (
        df["Squeeze_Signal"] & not_overbought & strong_enough & enough_volume & strong_trend & obv_ok
        & sector_strong_enough & not_near_earnings
    )

    df["Trade_Score"] = raw_score.where(eligible, 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_adx_trend_entry_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """ADX-trend-entry counterpart to add_trade_score()/add_breakout_trade_score()/
    add_pullback_trade_score()/add_breakout_retest_trade_score()/add_week52_trade_score()/
    add_momentum_burst_trade_score()/add_squeeze_breakout_trade_score() -- blends
    RRR and Distance_to_Buy_Pct into a 0-100 Trade_Score for rows produced
    by compute_adx_trend_entry_levels(), reusing signal_for_score()'s
    thresholds so an adx_trend_entry Trade_Score means the same thing on
    the same 0-100 scale as every other strategy.

    No RSI component, same reasoning as the other trend-following
    strategies. rrr_score_weight/distance_score_weight are rescaled to
    still sum to 100.

    Distance_to_Buy_Pct is always 0 for this strategy (Buy_Price IS
    Last_Close, see adx_trend_entry_levels_from_frame) -- it stays in the
    returned dict for schema compatibility but is NOT used in this
    formula. In its place, Signal_Strength_Pct (how far ADX clears
    adx_trend_entry_threshold -- see adx_trend_entry_levels_from_frame)
    fills the second term, same clip-then-scale shape as rrr_score
    (bigger is better, capped at adx_trend_entry_strength_cap) -- see
    improvements.txt for why: the old Distance_to_Buy_Pct-based formula
    gave every eligible ticker the exact same fixed score, discovered
    during live dashboard testing (a stock's actual trend strength never
    reached the ranking at all).

    Hard gate, not just a scoring input: a ticker whose ADX_Trend_Signal
    is False (ADX below threshold, price below the short MA, or outside
    the macro-uptrend/liquidity gates -- see compute_adx_trend_entry_levels)
    gets Trade_Score=0/Ignore, full stop -- same "not eligible at all"
    semantics as every other strategy's hard gate, not merely a low score.

    Phase 2 (improvements.txt item 40): also gates out tickers whose RSI is
    at/above config.adx_trend_entry_rsi_overbought_threshold, whose
    Relative_Strength is below config.adx_trend_entry_relative_strength_min,
    whose Volume_Ratio is below config.adx_trend_entry_volume_ratio_min,
    whose OBV_Zscore is below config.adx_trend_entry_obv_zscore_min, or
    whose Squeeze_Zscore is above config.adx_trend_entry_squeeze_zscore_max
    -- all five kept in sync with simulate_adx_trend_entry_signals so live
    and backtested definitions can't silently disagree, and all five
    default to a practical no-op (missing/NaN values also never exclude a
    ticker on their own), same "disabled until explicitly tuned" treatment
    as breakout's own six filters (add_breakout_trade_score)."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    strength_clipped = df["Signal_Strength_Pct"].clip(lower=0, upper=config.adx_trend_entry_strength_cap)
    strength_score = (strength_clipped / config.adx_trend_entry_strength_cap) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + strength_score) * rescale).clip(lower=0)

    not_overbought = df["RSI"].isna() | (df["RSI"] < config.adx_trend_entry_rsi_overbought_threshold)
    strong_enough = df["Relative_Strength"].isna() | (df["Relative_Strength"] >= config.adx_trend_entry_relative_strength_min)
    enough_volume = df["Volume_Ratio"].isna() | (df["Volume_Ratio"] >= config.adx_trend_entry_volume_ratio_min)
    obv_ok = df["OBV_Zscore"].isna() | (df["OBV_Zscore"] >= config.adx_trend_entry_obv_zscore_min)
    squeezed = df["Squeeze_Zscore"].isna() | (df["Squeeze_Zscore"] <= config.adx_trend_entry_squeeze_zscore_max)
    eligible = (
        df["ADX_Trend_Signal"] & not_overbought & strong_enough & enough_volume & obv_ok & squeezed
    )

    df["Trade_Score"] = raw_score.where(eligible, 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df


def add_ma_crossover_trade_score(df: pd.DataFrame, config: TradingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Moving-average-crossover counterpart to every other add_*_trade_score()
    this session -- blends RRR and Distance_to_Buy_Pct into a 0-100
    Trade_Score for rows produced by compute_ma_crossover_levels(), reusing
    signal_for_score()'s thresholds so this Trade_Score means the same
    thing on the same 0-100 scale as every other strategy.

    No RSI component, same reasoning as the other trend-following
    strategies. rrr_score_weight/distance_score_weight are rescaled to
    still sum to 100.

    Distance_to_Buy_Pct is always 0 for this strategy (Buy_Price IS
    Last_Close, see ma_crossover_levels_from_frame) -- it stays in the
    returned dict for schema compatibility but is NOT used in this
    formula. In its place, Signal_Strength_Pct (the crossover's own gap,
    short MA minus long MA as a % of price) fills the second term, same
    clip-then-scale shape as rrr_score (bigger is better, capped at
    ma_crossover_strength_cap_pct).

    Hard gate, not just a scoring input: a ticker whose
    MA_Crossover_Signal is False (no crossover today, or outside the
    macro-uptrend/liquidity gates -- see compute_ma_crossover_levels)
    gets Trade_Score=0/Ignore, full stop -- same "not eligible at all"
    semantics as every other strategy's hard gate, not merely a low
    score. Launched lean v1 with no optional sharpening filters (mirrors
    every other strategy's own launch pattern: build lean, validate via
    the random-entry benchmark FIRST, add filters only if that clears the
    bar) -- one optional gate since added, config.ma_crossover_earnings_gate
    (boolean, default False), excludes a ticker whose Catalyst_Warning is
    True, kept in sync with simulate_ma_crossover_signals. See
    improvements.txt for the validation result before ever setting it
    True on a live config."""
    df = df.copy()

    rrr_score = (df["RRR"].clip(lower=0, upper=config.rrr_score_cap) / config.rrr_score_cap) * config.rrr_score_weight

    strength_clipped = df["Signal_Strength_Pct"].clip(lower=0, upper=config.ma_crossover_strength_cap_pct)
    strength_score = (strength_clipped / config.ma_crossover_strength_cap_pct) * config.distance_score_weight

    total_weight = config.rrr_score_weight + config.distance_score_weight
    rescale = (100 / total_weight) if total_weight > 0 else 0.0

    raw_score = ((rrr_score + strength_score) * rescale).clip(lower=0)

    # earnings_gate is a boolean on/off toggle, not a numeric threshold (see
    # config.ma_crossover_earnings_gate) -- disabled (default) means no
    # exclusion at all.
    not_near_earnings = (
        (df["Catalyst_Warning"].isna() | ~df["Catalyst_Warning"])
        if config.ma_crossover_earnings_gate else True
    )
    # Sector_Relative_Strength (backtest/Optuna-only) -- ma_crossover's
    # first optional numeric filter; same graceful None/NaN-never-excludes
    # treatment as breakout's/squeeze_breakout's own.
    sector_strong_enough = (
        df["Sector_Relative_Strength"].isna()
        | (df["Sector_Relative_Strength"] >= config.ma_crossover_sector_relative_strength_min)
    )
    eligible = df["MA_Crossover_Signal"] & not_near_earnings & sector_strong_enough

    df["Trade_Score"] = raw_score.where(eligible, 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df
