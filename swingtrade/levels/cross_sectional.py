"""Cross-sectional level calculators: pairs, momentum, low-vol, sector rotation."""

import numpy as np
import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .technical import precompute_breakout_frame


def precompute_pairs_frame(
    df: pd.DataFrame,
    peer_prices: pd.DataFrame | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the mean-reversion PAIRS strategy's columns
    -- built on top of precompute_breakout_frame() (reused wholesale for
    SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/liquidity gates
    every strategy shares).

    `peer_prices`, if given, should be a wide DataFrame of Close prices for
    every OTHER ticker in this ticker's own sector (this ticker's own
    column must NOT be included -- the caller's job to pre-filter by
    sector_lookup, keeping this function a pure DataFrame-in computation
    with no fetching/lookup of its own), reindexed to (at least cover)
    `df`'s own date range.

    For each trading day, picks whichever peer has the highest rolling
    correlation (pairs_lookback_days window) to this ticker's own daily
    returns, PROVIDED it clears pairs_min_correlation -- Pair_Partner (the
    winning peer's ticker) and Pair_Correlation are both fully vectorized
    via pandas' own rolling().corr(), genuinely causal (each day's value
    uses only the trailing window ending that day, never a future value) --
    not a per-day Python loop.

    Pair_Spread_Zscore measures how unusual TODAY's divergence from the
    winning partner is, relative to what's been NORMAL for this specific
    pair recently: spread = this ticker's cumulative return over
    pairs_spread_window_days minus the partner's own cumulative return over
    the identical window; the z-score baseline (rolling mean/stdev of that
    spread over pairs_zscore_window_days) is itself .shift(1)'d before use
    -- same "don't let today's own extreme value inflate the very baseline
    it's being compared against" discipline Squeeze_Zscore already
    established (see precompute_breakout_frame's own comment) -- otherwise
    a big divergence would partly absorb into its own rolling mean/stdev
    and dampen the very signal it's supposed to flag.

    Every same-sector peer's spread/z-score is computed (not just the
    eventual daily winner's), since that's cheap/vectorized regardless;
    only the winning partner's own value is kept per day via a row-wise
    select against that day's own Pair_Partner column."""
    df = precompute_breakout_frame(df, config)
    df["Pair_Partner"] = None
    df["Pair_Correlation"] = np.nan
    df["Pair_Spread_Zscore"] = np.nan

    if peer_prices is None or peer_prices.empty:
        return df

    peer_prices = peer_prices.reindex(df.index)
    ticker_returns = df["Close"].pct_change()
    ticker_cum_spread = df["Close"].pct_change(periods=config.pairs_spread_window_days)

    corr_by_peer: dict[str, pd.Series] = {}
    zscore_by_peer: dict[str, pd.Series] = {}
    for peer in peer_prices.columns:
        peer_close = peer_prices[peer]
        if peer_close.isna().all():
            continue
        peer_returns = peer_close.pct_change()
        corr_by_peer[peer] = ticker_returns.rolling(window=config.pairs_lookback_days).corr(peer_returns)

        peer_cum_spread = peer_close.pct_change(periods=config.pairs_spread_window_days)
        spread = (ticker_cum_spread - peer_cum_spread).replace([float("inf"), float("-inf")], pd.NA)

        spread_mean = spread.shift(1).rolling(window=config.pairs_zscore_window_days).mean()
        spread_std = spread.shift(1).rolling(window=config.pairs_zscore_window_days).std()
        zscore_by_peer[peer] = (spread - spread_mean) / spread_std

    if not corr_by_peer:
        return df

    corr_df = pd.DataFrame(corr_by_peer)
    zscore_df = pd.DataFrame(zscore_by_peer)

    # DataFrame.idxmax(axis=1) raises ValueError on a row that's entirely
    # NaN (every day before the pairs_lookback_days rolling window has
    # filled is exactly this, across every peer column simultaneously) --
    # compute idxmax/max only over rows with at least one real value, and
    # leave the rest as NaN rather than crashing.
    best_corr = pd.Series(np.nan, index=corr_df.index)
    best_partner = pd.Series(np.nan, index=corr_df.index, dtype=object)
    has_any_value = corr_df.notna().any(axis=1)
    if has_any_value.any():
        valid_subset = corr_df.loc[has_any_value]
        best_corr.loc[has_any_value] = valid_subset.max(axis=1)
        best_partner.loc[has_any_value] = valid_subset.idxmax(axis=1)
    # Explicitly null out days where the winning correlation is missing or
    # below the minimum threshold, rather than fabricating a partner.
    valid = best_corr.notna() & (best_corr >= config.pairs_min_correlation)
    best_partner = best_partner.where(valid)
    best_corr = best_corr.where(valid)

    best_zscore = pd.Series(np.nan, index=df.index)
    for peer in zscore_df.columns:
        is_winner = best_partner == peer
        best_zscore = best_zscore.where(~is_winner, zscore_df[peer])

    df["Pair_Partner"] = best_partner
    df["Pair_Correlation"] = best_corr.round(4)
    df["Pair_Spread_Zscore"] = best_zscore
    return df


def pairs_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the mean-reversion PAIRS strategy's dict for one row of a
    frame already built by precompute_pairs_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi, adx = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
        last_row["ADX"],
    )
    pair_partner, pair_correlation, pair_spread_zscore = (
        last_row["Pair_Partner"], last_row["Pair_Correlation"], last_row["Pair_Spread_Zscore"],
    )
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating (this strategy's trigger is
    # the peer-spread z-score, not RSI) -- same treatment every other
    # non-RSI strategy (squeeze_breakout, breakout, ma_crossover) gives it.
    # Including it (unlike the first draft, which omitted it entirely) --
    # storage/signals.py's _build_document() unconditionally read row["RSI"]
    # for every strategy ever built before this one, a real crash caught
    # only once a real full-watchlist signal actually reached logging (see
    # improvements.txt item 82's live-wiring follow-up); fixing that one
    # call site AND restoring RSI here closes the whole gap, not just the
    # one instance found.
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)
    # ADX (2026-09-21 real bug fix): also informational only, not used for
    # gating -- but regime_switcher.select_regime_pick() reads it from
    # WHATEVER strategy row is available to classify that ticker's regime,
    # and "pairs" is choppy's own preferred strategy. Never exposing ADX
    # here meant regime_switcher could only ever resolve a regime on a day
    # ma_crossover ALSO fired for the identical ticker (supplying ADX from
    # its own row) -- given ma_crossover/pairs' near-zero same-ticker/day
    # overlap (confirmed ~0.23%, see improvements.txt item 149), this left
    # the "choppy" regime pick effectively dead since the 2026-08-31 fix
    # that pointed it at "pairs" (previously squeeze_breakout, which DID
    # carry ADX) -- confirmed via a real diagnostic run finding regime_switcher
    # had logged zero signals in 18 days despite ma_crossover/pairs both
    # producing real non-Ignore candidates throughout that window.
    adx = None if pd.isna(adx) else round(float(adx), 2)

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    # Missing partner/z-score (no peer_prices supplied, or no same-sector
    # peer cleared pairs_min_correlation) degrades to "never fires" rather
    # than excluding the ticker outright -- same "optional data missing
    # doesn't crash, just no signal" convention every other strategy uses.
    if pd.isna(pair_partner) or pd.isna(pair_spread_zscore):
        pair_signal = False
        pair_spread_zscore_out = None
        signal_strength_pct = 0.0
    else:
        pair_spread_zscore = float(pair_spread_zscore)
        pair_signal = bool(pair_spread_zscore <= config.pairs_zscore_entry_max)
        pair_spread_zscore_out = round(pair_spread_zscore, 3)
        # How far below the trigger the z-score fell -- same "distance past
        # the trigger" differentiating-term role Signal_Strength_Pct plays
        # for squeeze_breakout/momentum_burst, just in z-score units here.
        signal_strength_pct = (
            round(config.pairs_zscore_entry_max - pair_spread_zscore, 3) if pair_signal else 0.0
        )

    # Buy_Price = today's own Close -- same "confirm and enter near market"
    # convention squeeze_breakout/ma_crossover use, not a discount-limit
    # like RSI's structural-support wait. Distance_to_Buy_Pct is therefore
    # always 0 by construction, same schema-compatible treatment those
    # strategies already established.
    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "ADX": adx,
        "Pair_Partner": None if pd.isna(pair_partner) else str(pair_partner),
        "Pair_Correlation": None if pd.isna(pair_correlation) else round(float(pair_correlation), 4),
        "Pair_Spread_Zscore": pair_spread_zscore_out,
        "Pair_Signal": pair_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Dollar_Volume": round(dollar_volume, 2),
    }


def compute_pairs_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    peer_prices: pd.DataFrame | None = None,
) -> dict:
    """Compute mean-reversion PAIRS levels for one ticker's OHLCV history --
    see precompute_pairs_frame()'s own docstring for the full mechanism
    (partner selection via rolling correlation, spread z-score). Thin
    wrapper over precompute_pairs_frame()/pairs_levels_from_frame() -- kept
    as a single-call convenience for the live dashboard, matching every
    other strategy's same rationale (see compute_squeeze_breakout_levels()).

    `peer_prices` should be a wide DataFrame of Close prices for every
    OTHER ticker in this ticker's own sector (this ticker's own column
    excluded) -- caller's job to build from the already-fetched watchlist
    bundle, same as market_data.py's own sector_data resolution. Without
    it, Pair_Signal always reads False (no partner data)."""
    frame = precompute_pairs_frame(df, peer_prices, config)
    as_of = frame.index[-1]
    return pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_momentum_rank_frame(panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own trailing return,
    computed ONCE for the whole universe -- the key structural difference
    from pairs' precompute_pairs_frame(), which recomputes its cross-ticker
    correlation math separately inside every single ticker's own call.
    Percentile rank is inherently a whole-universe operation (a ticker's
    rank today depends on every OTHER ticker's return that same day), so
    doing it once here and having every ticker's own precompute_momentum_frame()
    just look up its own already-computed column is both cheaper and more
    correct than recomputing it per ticker.

    `panel` should be a wide DataFrame of Close prices, one column per
    ticker in the ranking universe (see market_data.build_momentum_panel()
    for the live-path caller, or the equivalent inline construction in
    optimize.py/benchmark_random_entry.py for backtest callers) -- no
    sector partitioning, unlike pairs' own per-sector panels, since ranking
    is universe-wide by design.

    Returns a DataFrame the same shape as `panel` (dates x tickers), each
    cell = that ticker's trailing-return percentile (0-100 scale, matching
    best_ideas.compute_sector_rs_scores()'s identical rank(pct=True)*100
    convention) on that day. A ticker with fewer than `lookback_days` of
    prior history reads NaN until its trailing return itself becomes
    computable, same "missing data doesn't fabricate a signal" convention
    as every other optional field in this codebase."""
    trailing_return = panel.pct_change(periods=lookback_days)
    return trailing_return.rank(axis=1, pct=True) * 100


def precompute_momentum_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional MOMENTUM RANK strategy's
    columns -- built on top of precompute_breakout_frame() (reused
    wholesale for SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/
    liquidity gates every strategy shares).

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_momentum_rank_frame() over every ticker's Close prices --
    unlike pairs' `peer_prices` (a DataFrame of every OTHER ticker, with
    correlation computed per-ticker inside precompute_pairs_frame()), the
    rank computation itself is inherently a whole-universe operation done
    ONCE by the caller; this function only reads this ticker's own
    already-computed percentile, reindexed to df's own date range. Missing
    (None) degrades to "never fires" rather than excluding the ticker
    outright, same convention as pairs' missing peer_prices."""
    df = precompute_breakout_frame(df, config)
    df["Momentum_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def momentum_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional MOMENTUM RANK strategy's dict for one row
    of a frame already built by precompute_momentum_frame() -- the
    O(1)-per-row counterpart a walk-forward loop calls once per `as_of`.
    Same macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    momentum_percentile = last_row["Momentum_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating (this strategy's trigger is
    # the universe-wide percentile rank, not RSI) -- same treatment every
    # other non-RSI strategy (squeeze_breakout, breakout, ma_crossover,
    # pairs) gives it, per the real storage/signals.py bug pairs' own
    # rollout caught (row["RSI"] bracket access assumed every strategy
    # carries this key -- see pairs_levels_from_frame()'s identical comment).
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    # Missing percentile (no rank_column supplied, or too little history
    # for this ticker's own trailing return yet) degrades to "never fires"
    # rather than excluding the ticker outright -- same "optional data
    # missing doesn't crash, just no signal" convention every other
    # strategy uses.
    if pd.isna(momentum_percentile):
        momentum_signal = False
        momentum_percentile_out = None
        signal_strength_pct = 0.0
    else:
        momentum_percentile = float(momentum_percentile)
        momentum_signal = bool(momentum_percentile >= config.momentum_top_percentile_min)
        momentum_percentile_out = round(momentum_percentile, 2)
        # How far past the trigger the percentile cleared -- same
        # "distance past the trigger" differentiating-term role
        # Signal_Strength_Pct plays for squeeze_breakout/pairs, just in
        # percentile points here.
        signal_strength_pct = (
            round(momentum_percentile - config.momentum_top_percentile_min, 2) if momentum_signal else 0.0
        )

    # Buy_Price = today's own Close -- same "confirm and enter near market"
    # convention squeeze_breakout/ma_crossover/pairs use, not a discount-
    # limit like RSI's structural-support wait. Distance_to_Buy_Pct is
    # therefore always 0 by construction, same schema-compatible treatment
    # those strategies already established.
    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "Momentum_Percentile": momentum_percentile_out,
        "Momentum_Signal": momentum_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Dollar_Volume": round(dollar_volume, 2),
    }


def compute_momentum_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional MOMENTUM RANK levels for one ticker's OHLCV
    history -- see precompute_momentum_frame()'s own docstring for the full
    mechanism. Thin wrapper over precompute_momentum_frame()/
    momentum_levels_from_frame() -- kept as a single-call convenience for
    the live dashboard, matching every other strategy's same rationale (see
    compute_pairs_levels()).

    `rank_column` should be this ticker's own column already sliced from a
    shared universe-wide rank frame (see compute_momentum_rank_frame()) --
    caller's job to build from the already-fetched watchlist bundle, same
    as market_data.py's own pair_price_panels resolution. Without it,
    Momentum_Signal always reads False (no percentile data)."""
    frame = precompute_momentum_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return momentum_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_lowvol_rank_frame(panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of every ticker's own trailing
    REALIZED VOLATILITY (stdev of daily returns), computed ONCE for the
    whole universe -- the low-volatility-anomaly mirror of
    compute_momentum_rank_frame(), same "whole-universe operation done once,
    not recomputed per ticker" rationale.

    `panel` should be a wide DataFrame of Close prices, one column per
    ticker in the ranking universe -- identical shape/convention to
    compute_momentum_rank_frame()'s own `panel` argument (same caller-side
    construction: pd.DataFrame({t: ticker_data[t]["Close"] for t in ...})).

    Returns a DataFrame the same shape as `panel` (dates x tickers), each
    cell = that ticker's trailing-volatility percentile on that day, but
    INVERTED relative to a plain rank(pct=True) -- the LEAST volatile
    ticker that day reads 100.0 (not the most volatile), so
    lowvol_levels_from_frame()'s ">= lowvol_top_percentile_min fires" gate
    reads identically in spirit to momentum_rank's ">= momentum_top_percentile_min"
    gate despite ranking the opposite underlying direction. Uses
    `ascending=False` on the raw volatility (not `1 - rank(ascending=True)`)
    so the scale's floor/ceiling exactly mirror compute_momentum_rank_frame()'s
    own (100/n to 100, not 0 to 100 - 100/n) -- the two off-by-one-normalization
    shapes are NOT interchangeable, confirmed the hard way by this function's
    own test suite catching a real bug in an earlier `1 - rank` draft. A
    ticker with fewer than `lookback_days`+1 prior daily returns reads NaN
    until its own trailing volatility becomes computable, same "missing data
    doesn't fabricate a signal" convention as compute_momentum_rank_frame()."""
    daily_returns = panel.pct_change()
    trailing_vol = daily_returns.rolling(window=lookback_days).std()
    return trailing_vol.rank(axis=1, pct=True, ascending=False) * 100


def precompute_lowvol_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional LOW-VOLATILITY RANK
    strategy's columns -- built on top of precompute_breakout_frame() (reused
    wholesale for SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/
    liquidity gates every strategy shares), exact structural mirror of
    precompute_momentum_frame().

    `rank_column`, if given, should be THIS ticker's own column already
    sliced from a shared universe-wide rank frame built once via
    compute_lowvol_rank_frame() over every ticker's Close prices. Missing
    (None) degrades to "never fires" rather than excluding the ticker
    outright, same convention as precompute_momentum_frame()."""
    df = precompute_breakout_frame(df, config)
    df["LowVol_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def lowvol_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional LOW-VOLATILITY RANK strategy's dict for
    one row of a frame already built by precompute_lowvol_frame() -- exact
    structural mirror of momentum_levels_from_frame(), differing only in
    which precomputed percentile column gates the signal. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    lowvol_percentile = last_row["LowVol_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    # Informational only, not used for gating -- same treatment momentum_rank
    # gives it (this strategy's trigger is the universe-wide volatility
    # percentile, not RSI).
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    # Missing percentile (no rank_column supplied, or too little history for
    # this ticker's own trailing volatility yet) degrades to "never fires"
    # rather than excluding the ticker outright -- same convention
    # momentum_levels_from_frame() uses.
    if pd.isna(lowvol_percentile):
        lowvol_signal = False
        lowvol_percentile_out = None
        signal_strength_pct = 0.0
    else:
        lowvol_percentile = float(lowvol_percentile)
        lowvol_signal = bool(lowvol_percentile >= config.lowvol_top_percentile_min)
        lowvol_percentile_out = round(lowvol_percentile, 2)
        signal_strength_pct = (
            round(lowvol_percentile - config.lowvol_top_percentile_min, 2) if lowvol_signal else 0.0
        )

    # Buy_Price = today's own Close -- same "confirm and enter near market"
    # convention momentum_rank/squeeze_breakout/ma_crossover/pairs use.
    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "LowVol_Percentile": lowvol_percentile_out,
        "LowVol_Signal": lowvol_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Dollar_Volume": round(dollar_volume, 2),
    }


def compute_lowvol_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional LOW-VOLATILITY RANK levels for one ticker's
    OHLCV history -- see precompute_lowvol_frame()'s own docstring for the
    full mechanism. Thin wrapper over precompute_lowvol_frame()/
    lowvol_levels_from_frame(), exact mirror of compute_momentum_levels()."""
    frame = precompute_lowvol_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return lowvol_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)


def compute_sector_rotation_rank_frame(sector_panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of every SECTOR ETF's own trailing
    RETURN against every OTHER sector (not every other ticker) -- the
    inter-sector-rotation counterpart to compute_momentum_rank_frame(),
    same "whole-universe operation done once" rationale but the "universe"
    here is the ~11 GICS sector ETFs (see watchlist.SECTOR_ETF), not the
    ticker watchlist.

    `sector_panel` should be a wide DataFrame of Close prices, one column
    per SECTOR NAME (e.g. "Technology", "Financials", ...) -- built from
    the same sector ETF OHLCV every SECTOR_AWARE_STRATEGIES caller already
    fetches (see watchlist.SECTOR_ETF), NOT one column per ticker.

    Returns a DataFrame the same shape as `sector_panel` (dates x sectors),
    each cell = that SECTOR's trailing-return percentile among all sectors
    on that day (100 = strongest sector, matching compute_momentum_rank_frame()'s
    own "highest value reads 100" convention exactly, unlike
    compute_lowvol_rank_frame()'s deliberately-inverted one). A sector with
    fewer than `lookback_days` of prior history reads NaN, same "missing
    data doesn't fabricate a signal" convention as compute_momentum_rank_frame()."""
    trailing_return = sector_panel.pct_change(periods=lookback_days)
    return trailing_return.rank(axis=1, pct=True) * 100


def precompute_sector_rotation_frame(
    df: pd.DataFrame,
    rank_column: pd.Series | None = None,
    config: TradingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Vectorized precompute of the cross-sectional SECTOR ROTATION
    strategy's columns -- built on top of precompute_breakout_frame() (reused
    wholesale for SMA_TREND/ATR/AvgVolume/etc., the same macro-uptrend/
    liquidity gates every strategy shares), exact structural mirror of
    precompute_momentum_frame()/precompute_lowvol_frame().

    `rank_column`, if given, should be THIS TICKER'S OWN SECTOR'S column
    already sliced from a shared sector-wide rank frame built once via
    compute_sector_rotation_rank_frame() -- unlike momentum_rank_frame/
    lowvol_rank_frame (sliced by TICKER, one column per ticker), this is
    sliced by the ticker's SECTOR (every ticker in the same sector shares
    the identical column) -- caller's job to resolve `sector_lookup[ticker]`
    into the right column. Missing (None) degrades to "never fires" rather
    than excluding the ticker outright, same convention as
    precompute_momentum_frame()."""
    df = precompute_breakout_frame(df, config)
    df["SectorRotation_Percentile"] = rank_column.reindex(df.index) if rank_column is not None else np.nan
    return df


def sector_rotation_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract the cross-sectional SECTOR ROTATION strategy's dict for one
    row of a frame already built by precompute_sector_rotation_frame() --
    exact structural mirror of momentum_levels_from_frame()/
    lowvol_levels_from_frame(), differing only in which precomputed
    percentile column gates the signal. Same macro-uptrend/liquidity gates
    every strategy shares (via precompute_breakout_frame(), reused
    wholesale)."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    sector_rotation_percentile = last_row["SectorRotation_Percentile"]
    if pd.isna(last_close):
        raise RuntimeError("insufficient history: no Close price for the most recent bar")
    if pd.isna(sma_trend):
        raise RuntimeError(f"insufficient history to compute {config.sma_trend_window}-day SMA")
    if pd.isna(atr):
        raise RuntimeError(f"insufficient history to compute {config.atr_window}-day ATR")
    if pd.isna(avg_volume):
        raise RuntimeError(f"insufficient history to compute {config.volume_lookback_days}-day average volume")

    last_close, sma_trend, atr, avg_volume = float(last_close), float(sma_trend), float(atr), float(avg_volume)
    rsi = None if pd.isna(rsi) else round(float(rsi), 2)

    if last_close < sma_trend:
        raise RuntimeError(
            f"excluded: macro downtrend (Last_Close {last_close:.2f} < SMA{config.sma_trend_window} {sma_trend:.2f})"
        )

    dollar_volume = avg_volume * last_close
    if dollar_volume < config.min_dollar_volume:
        raise RuntimeError(
            f"excluded: insufficient liquidity (20d $ volume ${dollar_volume:,.0f} "
            f"< ${config.min_dollar_volume:,.0f})"
        )

    if pd.isna(sector_rotation_percentile):
        sector_rotation_signal = False
        sector_rotation_percentile_out = None
        signal_strength_pct = 0.0
    else:
        sector_rotation_percentile = float(sector_rotation_percentile)
        sector_rotation_signal = bool(sector_rotation_percentile >= config.sector_rotation_top_percentile_min)
        sector_rotation_percentile_out = round(sector_rotation_percentile, 2)
        signal_strength_pct = (
            round(sector_rotation_percentile - config.sector_rotation_top_percentile_min, 2)
            if sector_rotation_signal else 0.0
        )

    buy_price = round(last_close, 2)
    distance_to_buy_pct = 0.0

    sell_price = round(buy_price + (config.atr_take_profit_multiplier * atr), 2)
    stop_loss = round(buy_price - (config.stop_loss_atr_multiplier * atr), 2)
    risk = buy_price - stop_loss
    rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0

    as_of_ts = pd.Timestamp(last_date)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    if next_earnings_date is not None:
        days_to_earnings = (next_earnings_date - as_of_ts).total_seconds() / 86400
        catalyst_warning = days_to_earnings <= config.earnings_warning_days
        next_earnings_date_out = next_earnings_date.date()
    else:
        catalyst_warning = False
        next_earnings_date_out = None

    return {
        "Ticker": ticker,
        "As_Of": last_date.date(),
        "Last_Close": round(last_close, 2),
        "RSI": rsi,
        "ATR": round(atr, 2),
        "SectorRotation_Percentile": sector_rotation_percentile_out,
        "SectorRotation_Signal": sector_rotation_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Signal_Strength_Pct": signal_strength_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
        "Dollar_Volume": round(dollar_volume, 2),
    }


def compute_sector_rotation_levels(
    ticker: str,
    df: pd.DataFrame,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    rank_column: pd.Series | None = None,
) -> dict:
    """Compute cross-sectional SECTOR ROTATION levels for one ticker's
    OHLCV history -- see precompute_sector_rotation_frame()'s own docstring
    for the full mechanism (`rank_column` should be THIS TICKER'S SECTOR'S
    own column, not the ticker's own). Thin wrapper, exact mirror of
    compute_momentum_levels()/compute_lowvol_levels()."""
    frame = precompute_sector_rotation_frame(df, rank_column, config)
    as_of = frame.index[-1]
    return sector_rotation_levels_from_frame(ticker, frame, as_of, config, next_earnings_date, top_headline)
