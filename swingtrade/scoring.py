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
    config.breakout_relative_strength_min, and whose Volume_Ratio (today's
    Volume over the PRIOR volume_lookback_days average) is below
    config.breakout_volume_ratio_min -- both kept in sync with
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
    eligible = df["Breakout_Signal"] & not_overbought & strong_enough & enough_volume

    df["Trade_Score"] = raw_score.where(eligible, 0.0).round(1)
    df["Signal"] = df["Trade_Score"].apply(lambda score: signal_for_score(score, config))
    return df
