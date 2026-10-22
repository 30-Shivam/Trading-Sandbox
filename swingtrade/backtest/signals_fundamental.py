"""Fundamentals/event-driven strategy simulators (quality, value, accruals, cash profitability, insider, analyst revisions, PEAD) and their random-entry baselines."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from ..levels import (
    accruals_levels_from_frame,
    cash_profitability_levels_from_frame,
    analyst_revision_levels_from_frame,
    insider_buying_levels_from_frame,
    pead_levels_from_frame,
    market_uptrend_from_frame,
    precompute_accruals_frame,
    precompute_cash_profitability_frame,
    precompute_analyst_revision_frame,
    precompute_insider_buying_frame,
    precompute_pead_frame,
    precompute_quality_frame,
    precompute_rsi_frame,
    precompute_value_frame,
    quality_levels_from_frame,
    value_levels_from_frame,
)
from ..scoring import (
    add_accruals_trade_score,
    add_analyst_revision_trade_score,
    add_cash_profitability_trade_score,
    add_pead_trade_score,
    add_quality_trade_score,
    add_value_trade_score,
)
from .core import (
    _find_entry_fill,
    _find_next_open_fill,
    _next_earnings_date,
    _settle,
)


def simulate_quality_signals(
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
    """Cross-sectional QUALITY/PROFITABILITY RANK counterpart to
    simulate_momentum_signals()/simulate_lowvol_signals() -- buys a ticker
    whose point-in-time ROE percentile rank (across the whole watchlist)
    clears config.quality_top_percentile_min, in a confirmed macro
    uptrend. The FIRST strategy in this codebase driven by real
    fundamentals data (SEC EDGAR, see sec_fundamentals.py) rather than
    yfinance OHLCV -- exact structural mirror of simulate_momentum_signals()
    otherwise. See swingtrade/config.py's quality_* fields and
    swingtrade/levels.compute_quality_rank_frame()/precompute_quality_frame()
    for the full mechanism.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_quality_rank_frame(), built from
    sec_fundamentals.build_roe_panel()) -- caller's job to build ONCE for
    the whole universe and slice per ticker. Without it, Quality_Signal is
    always False (no ROE data, or a ticker SEC EDGAR doesn't cover --
    Canadian cross-listed names -- degrades to "never fires," same
    convention as every other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_quality_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = quality_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Quality_Signal"]:
            continue

        # See simulate_momentum_signals()'s identical comment -- never a
        # gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_quality_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.quality_entry_fill == "next_open":
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
            "signal": "Quality_Rank",
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


def simulate_value_signals(
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
    """Cross-sectional VALUE RANK counterpart to simulate_quality_signals()
    -- buys a ticker whose Price-to-Book percentile rank (inverted so
    cheapest=highest, across the whole watchlist) clears
    config.value_top_percentile_min, in a confirmed macro uptrend. Exact
    structural mirror of simulate_quality_signals() otherwise -- see
    swingtrade/config.py's value_* fields and
    swingtrade.levels.compute_value_rank_frame()/precompute_value_frame()
    for the full mechanism.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_value_rank_frame(), built from
    sec_fundamentals.build_book_value_per_share_panel() divided into a
    Close-price panel) -- caller's job to build ONCE for the whole universe
    and slice per ticker. Without it, Value_Signal is always False."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_value_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = value_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Value_Signal"]:
            continue

        scored = add_value_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.value_entry_fill == "next_open":
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
            "signal": "Value_Rank",
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


def simulate_random_value_entries(
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
    """Random-entry benchmark for simulate_value_signals() -- exact
    structural mirror of simulate_random_quality_entries(). Deliberately
    does NOT take `rank_column` -- called with rank_column=None, so
    Value_Signal is always False, but the candidate pool is every day that
    passed the shared macro/liquidity gates regardless, same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_value_frame(ohlcv, None, config)

    candidates = []
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = value_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.value_entry_fill == "next_open":
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
            "signal": "Random_Value_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_accruals_signals(
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
    """Cross-sectional ACCRUALS RANK counterpart to simulate_value_signals()
    -- buys a ticker whose accruals percentile rank (inverted so lowest
    accruals=highest percentile, across the whole watchlist) clears
    config.accruals_top_percentile_min, in a confirmed macro uptrend.
    Implements Sloan (1996)'s classic accruals anomaly -- see
    swingtrade/config.py's accruals_* fields and
    swingtrade.levels.compute_accruals_rank_frame()/precompute_accruals_frame()
    for the full mechanism. Exact structural mirror of
    simulate_value_signals() otherwise.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_accruals_rank_frame(), built from
    sec_fundamentals.build_accruals_panel()) -- caller's job to build ONCE
    for the whole universe and slice per ticker. Without it,
    Accruals_Signal is always False."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_accruals_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = accruals_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Accruals_Signal"]:
            continue

        scored = add_accruals_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.accruals_entry_fill == "next_open":
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
            "signal": "Accruals_Rank",
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


def simulate_random_accruals_entries(
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
    """Random-entry benchmark for simulate_accruals_signals() -- exact
    structural mirror of simulate_random_value_entries(). Deliberately
    does NOT take `rank_column` -- called with rank_column=None, so
    Accruals_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless, same "answers
    whether TIMING adds value" precedent every other random baseline
    follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_accruals_frame(ohlcv, None, config)

    candidates = []
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = accruals_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.accruals_entry_fill == "next_open":
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
            "signal": "Random_Accruals_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_cash_profitability_signals(
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
    """Cross-sectional CASH PROFITABILITY RANK counterpart to
    simulate_accruals_signals() -- buys a ticker whose cash-profitability
    (CFO/Assets) percentile rank clears
    config.cash_profitability_top_percentile_min, in a confirmed macro
    uptrend. See swingtrade/config.py's cash_profitability_* fields and
    swingtrade.levels.compute_cash_profitability_rank_frame()/
    precompute_cash_profitability_frame() for the full mechanism. Exact
    structural mirror of simulate_accruals_signals()/simulate_value_signals()
    otherwise.

    `rank_column` should be THIS ticker's own column already sliced from a
    shared universe-wide rank frame (see
    swingtrade.levels.compute_cash_profitability_rank_frame(), built from
    sec_fundamentals.build_cash_profitability_panel()). Without it,
    Cash_Profitability_Signal is always False."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_cash_profitability_frame(ohlcv, rank_column, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = cash_profitability_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Cash_Profitability_Signal"]:
            continue

        scored = add_cash_profitability_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.cash_profitability_entry_fill == "next_open":
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
            "signal": "Cash_Profitability_Rank",
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


def simulate_random_cash_profitability_entries(
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
    """Random-entry benchmark for simulate_cash_profitability_signals() --
    exact structural mirror of simulate_random_accruals_entries(). Called
    with rank_column=None (Cash_Profitability_Signal always False), same
    "answers whether TIMING adds value" precedent every other random
    baseline follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_cash_profitability_frame(ohlcv, None, config)

    candidates = []
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = cash_profitability_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.cash_profitability_entry_fill == "next_open":
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
            "signal": "Random_Cash_Profitability_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_random_quality_entries(
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
    """Random-entry benchmark for simulate_quality_signals() -- same idea
    as simulate_random_momentum_entries(), using this strategy's own gates
    (macro uptrend, liquidity via quality_levels_from_frame) and the SAME
    config.quality_entry_fill-selected fill mechanic, so it isolates
    whether quality-rank TIMING adds value over a random day. `n_trades`
    should be simulate_quality_signals()'s real signal count for this
    ticker/window, so trade volume is matched.

    Deliberately does NOT take `rank_column` -- called with rank_column=None,
    so Quality_Signal is always False, but the candidate pool is every day
    that passed the shared macro/liquidity gates regardless, same "answers
    whether TIMING adds value, not whether this filter helps" precedent
    every other random baseline follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_quality_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning, dollar_volume) for every day that passed the macro/liquidity gates, quality signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = quality_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.quality_entry_fill == "next_open":
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
            "signal": "Random_Quality_Rank",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_insider_buying_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    insider_purchases: pd.DataFrame | None = None,
) -> list[dict]:
    """INSIDER-BUYING counterpart to simulate_signals()/simulate_squeeze_breakout_signals()/
    simulate_pairs_signals()/etc. -- buys a ticker when real, open-market
    insider purchases have clustered within a recent window
    (insider_lookback_days), in a confirmed macro uptrend. See
    swingtrade/config.py's insider_* fields and
    swingtrade/levels.precompute_insider_buying_frame() for the full
    mechanism, and run_backtest.fetch_insider_purchases() for the data
    source and its no-look-ahead reporting-lag handling.

    Same no-look-ahead discipline, same entry-timing realism.
    config.insider_entry_fill selects the fill model (same "limit" vs.
    "next_open" toggle every other strategy has). Same ATR-multiple
    stop/target sizing.

    `insider_purchases` should be run_backtest.fetch_insider_purchases()'s
    own output for this ticker. Without it, Insider_Buy_Signal is always
    False (no purchase data, degrades to "never fires," same convention as
    every other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_insider_buying_frame(ohlcv, insider_purchases, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = insider_buying_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Insider_Buy_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.insider_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Insider_Buying",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_insider_buying_entries(
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
    """Random-entry benchmark for simulate_insider_buying_signals() -- same
    idea as the other simulate_random_*_entries() functions, using this
    strategy's own gates (macro uptrend, liquidity via
    insider_buying_levels_from_frame) and the SAME
    config.insider_entry_fill-selected fill mechanic, so it isolates
    whether insider-purchase-cluster TIMING adds value over a random day
    using the identical Buy_Price formula (that day's own Close) and
    entry/exit structure. `n_trades` should be
    simulate_insider_buying_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py.

    Deliberately does NOT take `insider_purchases` -- called with
    insider_purchases=None, so Insider_Buy_Signal is always False, but the
    candidate pool is every day that passed the shared macro/liquidity
    gates regardless (same "answers whether TIMING adds value, not whether
    this filter helps" precedent every other random baseline follows --
    see simulate_random_pairs_entries()'s identical reasoning)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_insider_buying_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, insider signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = insider_buying_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.insider_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Insider_Buying",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_analyst_revision_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    analyst_revisions: pd.DataFrame | None = None,
) -> list[dict]:
    """ANALYST REVISION MOMENTUM counterpart to simulate_insider_buying_signals()
    -- buys a ticker when real analyst rating CHANGES have clustered net-
    positive within a recent window (analyst_revision_lookback_days), in a
    confirmed macro uptrend. See swingtrade/config.py's analyst_revision_*
    fields and swingtrade/levels.precompute_analyst_revision_frame() for
    the full mechanism, and run_backtest.fetch_analyst_revisions() for the
    data source and its no-look-ahead reporting-lag handling.

    `analyst_revisions` should be run_backtest.fetch_analyst_revisions()'s
    own output for this ticker. Without it, Analyst_Revision_Signal is
    always False (no revision data, degrades to "never fires")."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_analyst_revision_frame(ohlcv, analyst_revisions, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = analyst_revision_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Analyst_Revision_Signal"]:
            continue

        # See simulate_quality_signals()'s identical comment -- never a
        # gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_analyst_revision_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.analyst_revision_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Analyst_Revision",
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


def simulate_random_analyst_revision_entries(
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
    """Random-entry benchmark for simulate_analyst_revision_signals() --
    exact structural mirror of simulate_random_insider_buying_entries().
    Deliberately does NOT take `analyst_revisions` -- called with
    analyst_revisions=None, so Analyst_Revision_Signal is always False,
    but the candidate pool is every day that passed the shared macro/
    liquidity gates regardless, same "answers whether TIMING adds value"
    precedent every other random baseline follows."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_analyst_revision_frame(ohlcv, None, config)

    candidates = []
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = analyst_revision_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.analyst_revision_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_Analyst_Revision",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_pead_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    earnings_surprises: pd.DataFrame | None = None,
) -> list[dict]:
    """PEAD (post-earnings-announcement drift) counterpart to
    simulate_signals()/simulate_squeeze_breakout_signals()/
    simulate_insider_buying_signals()/etc. -- buys a ticker within
    pead_signal_window_days of a real, confirmed earnings SURPRISE
    (Reported EPS clearing Estimate by pead_surprise_pct_min or more), in a
    confirmed macro uptrend. See swingtrade/config.py's pead_* fields and
    swingtrade/levels.precompute_pead_frame() for the full mechanism, and
    run_backtest.fetch_earnings_surprises() for the data source and its OWN
    EXPLICIT point-in-time-integrity caveat -- read that before trusting
    any backtested result from this function.

    A genuinely different signal FAMILY from every other strategy in this
    codebase: an accounting/fundamental EVENT (does reality beat what
    analysts expected), not a price pattern -- built specifically after a
    real finding (ic_tracking.backtest_ic_check()) that every price-pattern
    strategy tried so far shows ~zero real backtest-time ranking skill.

    Same no-look-ahead discipline, same entry-timing realism.
    config.pead_entry_fill selects the fill model -- defaults to
    "next_open" (NOT "limit", unlike every sibling strategy's own default;
    see that field's own config.py comment for why a resting limit order
    is a poor fit for a "chase the continuation" premise).

    `earnings_surprises` should be run_backtest.fetch_earnings_surprises()'s
    own output for this ticker. Without it, PEAD_Signal is always False (no
    surprise data, degrades to "never fires," same convention as every
    other optional-data-dependent strategy)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pead_frame(ohlcv, earnings_surprises, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pead_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["PEAD_Signal"]:
            continue

        # Computed here purely so backtest trades carry a real trade_score
        # from day one (see ic_tracking.backtest_ic_check() -- the real
        # gap found this session that motivated building this check BEFORE
        # a new strategy shipped without one, not after). Never used as a
        # gate itself -- this strategy's own entry condition is PEAD_Signal
        # plus the shared macro/liquidity gates above, not a Trade_Score
        # threshold.
        scored = add_pead_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pead_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "PEAD",
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


def simulate_random_pead_entries(
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
    """Random-entry benchmark for simulate_pead_signals() -- same idea as
    the other simulate_random_*_entries() functions, using this strategy's
    own gates (macro uptrend, liquidity via pead_levels_from_frame) and the
    SAME config.pead_entry_fill-selected fill mechanic, so it isolates
    whether earnings-surprise TIMING adds value over a random day using the
    identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_pead_signals()'s real signal
    count for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see
    benchmark_random_entry.py.

    Deliberately does NOT take `earnings_surprises` -- called with
    earnings_surprises=None, so PEAD_Signal is always False, but the
    candidate pool is every day that passed the shared macro/liquidity
    gates regardless (same "answers whether TIMING adds value, not whether
    this filter helps" precedent every other random baseline follows -- see
    simulate_random_insider_buying_entries()'s identical reasoning)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pead_frame(ohlcv, None, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, PEAD signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = pead_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"]), bool(levels["Catalyst_Warning"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, catalyst_warning in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.pead_entry_fill == "next_open":
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
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_PEAD",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades
