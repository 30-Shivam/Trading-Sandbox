"""Cross-sectional strategy simulators (pairs, momentum, low-vol, sector rotation) and their random-entry baselines."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from ..levels import (
    lowvol_levels_from_frame,
    market_uptrend_from_frame,
    momentum_levels_from_frame,
    precompute_lowvol_frame,
    precompute_momentum_frame,
    precompute_pairs_frame,
    precompute_rsi_frame,
    precompute_sector_rotation_frame,
    pairs_levels_from_frame,
    sector_rotation_levels_from_frame,
)
from ..scoring import (
    add_lowvol_trade_score,
    add_momentum_trade_score,
    add_pairs_trade_score,
    add_sector_rotation_trade_score,
)
from .core import (
    _find_entry_fill,
    _find_next_open_fill,
    _next_earnings_date,
    _settle,
)


def simulate_pairs_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    peer_prices: pd.DataFrame | None = None,
) -> list[dict]:
    """Mean-reversion PAIRS counterpart to simulate_signals()/simulate_breakout_signals()/
    .../simulate_squeeze_breakout_signals() -- buys a ticker when it has
    diverged unusually far BELOW its most-correlated same-sector peer over
    a recent window (Pair_Spread_Zscore <= config.pairs_zscore_entry_max),
    in a confirmed macro uptrend. See swingtrade/config.py's pairs_* fields
    and swingtrade/levels.precompute_pairs_frame() for the full mechanism.
    LONG-ONLY (no short leg -- this codebase has no short-position support
    anywhere); this is the laggard-convergence side only.

    Same no-look-ahead discipline, same entry-timing realism.
    config.pairs_entry_fill selects the fill model (same "limit" vs.
    "next_open" toggle every other strategy has). Same ATR-multiple
    stop/target sizing.

    `peer_prices` should be a wide DataFrame of Close prices for every
    OTHER ticker in this ticker's own sector (caller's job to pre-filter by
    sector, this ticker's own column excluded) -- see
    precompute_pairs_frame()'s own docstring. Without it, Pair_Signal is
    always False (no partner data, degrades to "never fires," same
    convention as every other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pairs_frame(ohlcv, peer_prices, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Pair_Signal"]:
            continue

        # See simulate_squeeze_breakout_signals()'s identical comment --
        # never a gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_pairs_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pairs_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=levels.get("Dollar_Volume"),
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Pairs",
            "trade_score": float(scored["Trade_Score"]),
            "signal_strength_pct": float(levels["Signal_Strength_Pct"]),
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_pairs_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_pairs_signals() -- same idea as
    the other simulate_random_*_entries() functions, using this strategy's
    own gates (macro uptrend, liquidity via pairs_levels_from_frame) and
    the SAME config.pairs_entry_fill-selected fill mechanic, so it isolates
    whether pairs-divergence TIMING adds value over a random day using the
    identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_pairs_signals()'s real signal
    count for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see
    benchmark_random_entry.py.

    Deliberately does NOT take `peer_prices` -- called with peer_prices=None,
    so Pair_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless (same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows -- see
    simulate_random_squeeze_breakout_entries()'s identical reasoning)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pairs_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, pair signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pairs_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append(
            (as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"]), levels.get("Dollar_Volume"))
        )

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning, dollar_volume in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pairs_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=dollar_volume,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Pairs",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_momentum_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    rank_column: pd.Series | None = None,
) -> list[dict]:
    """Cross-sectional MOMENTUM RANK counterpart to simulate_signals()/
    simulate_breakout_signals()/.../simulate_pairs_signals() -- buys a
    ticker whose trailing-return percentile rank (across the WHOLE
    watchlist, not just its own sector) clears config.momentum_top_percentile_min,
    in a confirmed macro uptrend. See swingtrade/config.py's momentum_*
    fields and swingtrade/levels.compute_momentum_rank_frame()/
    precompute_momentum_frame() for the full mechanism.

    Same no-look-ahead discipline, same entry-timing realism.
    config.momentum_entry_fill selects the fill model (same "limit" vs.
    "next_open" toggle every other strategy has). Same ATR-multiple
    stop/target sizing.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_momentum_rank_frame()) -- caller's job to
    build ONCE for the whole universe and slice per ticker (unlike pairs'
    peer_prices, which are pre-filtered per-sector by the caller but the
    correlation math itself still runs per ticker; here the caller has
    already done ALL the cross-ticker work). Without it, Momentum_Signal is
    always False (no percentile data, degrades to "never fires," same
    convention as every other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = momentum_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Momentum_Signal"]:
            continue

        # See simulate_squeeze_breakout_signals()'s identical comment --
        # never a gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_momentum_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.momentum_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=levels.get("Dollar_Volume"),
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Momentum_Rank",
            "trade_score": float(scored["Trade_Score"]),
            "signal_strength_pct": float(levels["Signal_Strength_Pct"]),
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_momentum_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_momentum_signals() -- same idea
    as the other simulate_random_*_entries() functions, using this
    strategy's own gates (macro uptrend, liquidity via
    momentum_levels_from_frame) and the SAME config.momentum_entry_fill-
    selected fill mechanic, so it isolates whether cross-sectional
    momentum-rank TIMING adds value over a random day using the identical
    Buy_Price formula (that day's own Close) and entry/exit structure.
    `n_trades` should be simulate_momentum_signals()'s real signal count
    for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see
    benchmark_random_entry.py.

    Deliberately does NOT take `rank_column` -- called with rank_column=None,
    so Momentum_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless (same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows -- see
    simulate_random_pairs_entries()'s identical reasoning)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, momentum signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = momentum_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append(
            (as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"]), levels.get("Dollar_Volume"))
        )

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning, dollar_volume in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.momentum_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=dollar_volume,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Momentum_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_lowvol_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    rank_column: pd.Series | None = None,
) -> list[dict]:
    """Cross-sectional LOW-VOLATILITY RANK counterpart to
    simulate_momentum_signals() -- buys a ticker whose trailing-volatility
    percentile rank (across the WHOLE watchlist, inverted so LOW volatility
    reads as a HIGH percentile) clears config.lowvol_top_percentile_min, in
    a confirmed macro uptrend. Exact structural mirror of
    simulate_momentum_signals() -- see swingtrade/config.py's lowvol_*
    fields and swingtrade/levels.compute_lowvol_rank_frame()/
    precompute_lowvol_frame() for the full mechanism.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_lowvol_rank_frame()) -- caller's job to build
    ONCE for the whole universe and slice per ticker. Without it,
    LowVol_Signal is always False (no percentile data, degrades to "never
    fires," same convention as every other optional-data-dependent
    strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_lowvol_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = lowvol_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["LowVol_Signal"]:
            continue

        # See simulate_momentum_signals()'s identical comment -- never a
        # gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_lowvol_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.lowvol_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=levels.get("Dollar_Volume"),
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "LowVol_Rank",
            "trade_score": float(scored["Trade_Score"]),
            "signal_strength_pct": float(levels["Signal_Strength_Pct"]),
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_lowvol_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_lowvol_signals() -- same idea as
    simulate_random_momentum_entries(), using this strategy's own gates
    (macro uptrend, liquidity via lowvol_levels_from_frame) and the SAME
    config.lowvol_entry_fill-selected fill mechanic, so it isolates whether
    cross-sectional low-vol-rank TIMING adds value over a random day using
    the identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_lowvol_signals()'s real signal
    count for this ticker/window, so trade volume is matched.

    Deliberately does NOT take `rank_column` -- called with rank_column=None,
    so LowVol_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless (same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_lowvol_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning, dollar_volume) for every day that passed the macro/liquidity gates, lowvol signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = lowvol_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append(
            (as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"]), levels.get("Dollar_Volume"))
        )

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning, dollar_volume in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.lowvol_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=dollar_volume,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_LowVol_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_sector_rotation_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    rank_column: pd.Series | None = None,
) -> list[dict]:
    """Cross-sectional SECTOR ROTATION counterpart to simulate_momentum_signals()/
    simulate_lowvol_signals() -- buys a ticker whose OWN SECTOR's trailing-
    return percentile rank (across the ~11 GICS sector ETFs, NOT the ticker
    watchlist) clears config.sector_rotation_top_percentile_min, in a
    confirmed macro uptrend. Exact structural mirror of
    simulate_momentum_signals() -- see swingtrade/config.py's
    sector_rotation_* fields and swingtrade/levels.compute_sector_rotation_rank_frame()/
    precompute_sector_rotation_frame() for the full mechanism.

    `rank_column` should be THIS TICKER'S OWN SECTOR'S column already
    sliced from a shared sector-wide rank frame (see
    swingtrade.levels.compute_sector_rotation_rank_frame()) -- caller's job
    to build ONCE across the sector ETFs and resolve per ticker via
    sector_lookup (every ticker in the same sector shares the identical
    column, unlike momentum_rank_frame's per-ticker columns). Without it,
    SectorRotation_Signal is always False, same "missing optional data
    never fabricates a signal" convention as every other strategy."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_sector_rotation_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = sector_rotation_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["SectorRotation_Signal"]:
            continue

        # See simulate_momentum_signals()'s identical comment -- never a
        # gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_sector_rotation_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.sector_rotation_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=levels.get("Dollar_Volume"),
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Sector_Rotation",
            "trade_score": float(scored["Trade_Score"]),
            "signal_strength_pct": float(levels["Signal_Strength_Pct"]),
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_sector_rotation_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_sector_rotation_signals() -- same
    idea as simulate_random_momentum_entries(), using this strategy's own
    gates (macro uptrend, liquidity via sector_rotation_levels_from_frame)
    and the SAME config.sector_rotation_entry_fill-selected fill mechanic,
    so it isolates whether sector-rotation TIMING adds value over a random
    day. `n_trades` should be simulate_sector_rotation_signals()'s real
    signal count for this ticker/window, so trade volume is matched.

    Deliberately does NOT take `rank_column` -- called with rank_column=None,
    so SectorRotation_Signal is always False, but the candidate pool is
    every day that passed the shared macro/liquidity gates regardless, same
    "answers whether TIMING adds value, not whether this filter helps"
    precedent every other random baseline follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_sector_rotation_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning, dollar_volume) for every day that passed the macro/liquidity gates, sector-rotation signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = sector_rotation_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append(
            (as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"]), levels.get("Dollar_Volume"))
        )

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning, dollar_volume in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.sector_rotation_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=dollar_volume,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Sector_Rotation",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades
