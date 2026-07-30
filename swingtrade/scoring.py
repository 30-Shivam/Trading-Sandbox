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
