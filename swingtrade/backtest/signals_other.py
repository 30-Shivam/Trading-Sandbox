"""LLM-rule, ADX-trend and MA-crossover strategy simulators and their random-entry baselines."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from ..levels import (
    adx_trend_entry_levels_from_frame,
    llm_strategy_levels_from_frame,
    ma_crossover_levels_from_frame,
    market_uptrend_from_frame,
    precompute_adx_trend_entry_frame,
    precompute_llm_strategy_frame,
    precompute_ma_crossover_frame,
    precompute_rsi_frame,
    rule_exit_to_config,
)
from ..scoring import add_ma_crossover_trade_score
from ..settlement import settle_trade
from .core import (
    _find_entry_fill,
    _find_next_open_fill,
    _next_earnings_date,
    _settle,
)


def simulate_llm_strategy_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """LLM-INVENTED-STRATEGY counterpart to simulate_signals()/simulate_insider_buying_signals()/
    etc. -- buys a ticker when `rule`'s own conditions (an LLM-proposed
    combination of already-computed indicators, see
    swingtrade/levels.evaluate_llm_rule_conditions()) fire, in a confirmed
    macro uptrend (NOT something the rule can configure around -- the
    shared gate every strategy respects). Exit mechanics (fixed ATR
    bracket, trailing stop, or a custom holding period) are selected by
    `rule["exit"]["type"]` via rule_exit_to_config() -- the LLM only picks
    among this project's already-built, already-tested settlement
    mechanics (`_settle()` already dispatches on `config.trailing_stop_enabled`
    for exactly this reason), it never invents new settlement code.

    Entry fill is always the resting-limit-order model (`_find_entry_fill()`)
    -- rule DSL v1 doesn't expose entry-fill-model selection, only entry
    conditions and exit mechanics, matching what was actually scoped."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    effective_config = rule_exit_to_config(rule, config)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, effective_config)
    frame = precompute_llm_strategy_frame(ohlcv, rule, effective_config, market_df=market_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, effective_config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = llm_strategy_levels_from_frame(
                ticker, frame, as_of, effective_config, next_earnings_date=next_earnings,
            )
        except RuntimeError:
            continue

        if not levels["LLM_Strategy_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, effective_config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        atr = float(levels["ATR"])
        stop_loss = round(entry_price - effective_config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + effective_config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=effective_config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "LLM_Strategy",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(levels["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_llm_strategy_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_llm_strategy_signals() -- same
    idea as every other simulate_random_*_entries() function, using the
    SAME rule's own macro/liquidity gates and exit mechanics (via the
    identical rule_exit_to_config()), so it isolates whether the rule's
    own entry TIMING adds value over a random day, using the identical
    Buy_Price formula and entry/exit structure. `n_trades` should be
    simulate_llm_strategy_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate every strategy in this project has needed -- see
    llm_strategy_research.py's own run_daily_cycle()."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    effective_config = rule_exit_to_config(rule, config)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, effective_config)
    frame = precompute_llm_strategy_frame(ohlcv, rule, effective_config, market_df=market_ohlcv)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, rule signal or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, effective_config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = llm_strategy_levels_from_frame(
                ticker, frame, as_of, effective_config, next_earnings_date=next_earnings,
            )
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
        fill = _find_entry_fill(buy_price, bars_after_signal, effective_config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - effective_config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + effective_config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = _settle(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            atr=atr,
            bars_since_entry=bars_since_entry,
            config=effective_config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_LLM_Strategy",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades


def simulate_adx_trend_entry_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """ADX-trend-entry counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()/
    simulate_momentum_burst_signals()/simulate_squeeze_breakout_signals() --
    buys while ADX signals a genuinely trending state AND price is above a
    short-term MA for direction, in a confirmed macro uptrend. A
    continuous STATE (like week52_high/squeeze_breakout), so it can fire
    on many consecutive days while a trend persists.

    Same no-look-ahead discipline, same entry-timing realism.
    config.adx_trend_entry_entry_fill selects the fill model, same toggle
    momentum_burst_entry_fill/squeeze_breakout_entry_fill added: "limit"
    (default) is a resting LIMIT order at today's own Close via
    _find_entry_fill(); "next_open" buys the very next session's Open
    unconditionally via _find_next_open_fill() -- built in from the start
    here too, per the lesson from momentum_burst/squeeze_breakout
    (improvements.txt items 36-38). Same ATR-multiple stop/target sizing
    as every other strategy.

    `market_ohlcv` enables the optional Phase 2 Relative_Strength filter
    (see swingtrade/config.py's adx_trend_entry_relative_strength_min) --
    same role as simulate_breakout_signals()'s own market_df threading.
    Phase 2 also gates on RSI-overbought/Volume_Ratio/OBV_Zscore/
    Squeeze_Zscore, all disabled by default (see
    add_adx_trend_entry_trade_score()'s docstring for the full list) --
    kept in sync with that function so live and backtested definitions
    can't silently disagree, same discipline breakout's own six filters use.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_adx_trend_entry_frame(ohlcv, config, market_df=market_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = adx_trend_entry_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["ADX_Trend_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.adx_trend_entry_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.adx_trend_entry_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.adx_trend_entry_volume_ratio_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.adx_trend_entry_obv_zscore_min:
            continue
        squeeze_zscore = levels.get("Squeeze_Zscore")
        if squeeze_zscore is not None and squeeze_zscore > config.adx_trend_entry_squeeze_zscore_max:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.adx_trend_entry_entry_fill == "next_open":
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
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "ADX_Trend_Entry",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_adx_trend_entry_entries(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    n_trades: int,
    rng,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Random-entry benchmark for simulate_adx_trend_entry_signals() --
    same idea as the other seven simulate_random_*_entries() functions,
    using this strategy's own gates (macro uptrend, liquidity via
    compute_adx_trend_entry_levels) and the SAME
    config.adx_trend_entry_entry_fill-selected fill mechanic, so it
    isolates whether ADX-trend-entry TIMING adds value over a random day
    using the identical Buy_Price formula (that day's own Close) and
    entry/exit structure. `n_trades` should be
    simulate_adx_trend_entry_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_adx_trend_entry_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, trending or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = adx_trend_entry_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Buy_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.adx_trend_entry_entry_fill == "next_open":
            fill = _find_next_open_fill(bars_after_signal)
        else:
            fill = _find_entry_fill(buy_price, bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random_ADX_Trend_Entry",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_ma_crossover_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
    yield_curve: pd.Series | None = None,
    skew_regime: pd.Series | None = None,
) -> list[dict]:
    """Moving-average-crossover counterpart to every other simulate_*_signals()
    this session -- buys the day a short-term SMA crosses ABOVE a
    long-term SMA, in a confirmed macro uptrend. A genuinely different
    mechanical trigger from breakout (price-level high), squeeze_breakout
    (volatility-regime shift), or adx_trend_entry (raw trend-strength
    threshold) -- see swingtrade/config.py's own comment on this strategy
    for the full motivation (every momentum/strength-family signal tried
    this session lost to random-entry timing once properly checked).

    Same no-look-ahead discipline, same entry-timing realism.
    config.ma_crossover_entry_fill selects the fill model, same toggle
    every same-day-trigger strategy gets from the start: "limit" (default)
    is a resting LIMIT order at today's own Close via _find_entry_fill();
    "next_open" buys the very next session's Open unconditionally via
    _find_next_open_fill(). Same ATR-multiple stop/target sizing as every
    other strategy.

    Deliberately lean v1, no optional sharpening filters -- this is a NEW
    signal with no prior history to inherit filters from, and this
    project's own established discipline (see improvements.txt) is to
    validate the base trigger via the random-entry benchmark BEFORE
    adding any filters, not after.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False --
    same role as simulate_signals()'s own `earnings_dates` param. Previously
    this strategy's backtested Catalyst_Warning was ALWAYS False regardless
    of real earnings proximity (see improvements.txt for the gap this
    closes, shared with simulate_squeeze_breakout_signals()).

    `yield_curve` (optional, a single date-indexed Series shared across
    every ticker -- see market_data.fetch_fred_series(), currently only
    ever T10Y2Y) backs the BACKTEST/OPTUNA-ONLY
    ma_crossover_yield_curve_spread_max filter -- see config.py's own
    comment on that field for the real finding motivating it
    (benchmark_macro_regime.py, 2026-08-31).

    `skew_regime` (optional, a single date-indexed Series of RAW CBOE SKEW
    Index closes shared across every ticker -- see
    run_backtest.fetch_history() called on ticker "^SKEW") backs the
    BACKTEST/OPTUNA-ONLY ma_crossover_skew_regime_min filter -- see
    config.py's own comment on that field for the real finding motivating
    it (benchmark_skew_regime.py, 2026-09-03).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    # market_df/sector_df threaded through here (previously omitted --
    # ma_crossover had no Relative_Strength/sector data at all) purely to
    # enable the new, backtest/Optuna-only Sector_Relative_Strength filter
    # below; nothing else in this strategy reads Relative_Strength, so this
    # is additive and changes no existing gating behavior.
    frame = precompute_ma_crossover_frame(
        ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv, yield_curve=yield_curve,
        skew_regime=skew_regime,
    )

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = ma_crossover_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["MA_Crossover_Signal"]:
            continue
        if config.ma_crossover_earnings_gate and levels["Catalyst_Warning"]:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.ma_crossover_sector_relative_strength_min:
            continue
        yield_curve_spread = levels.get("Yield_Curve_Spread")
        if yield_curve_spread is not None and yield_curve_spread > config.ma_crossover_yield_curve_spread_max:
            continue
        skew_regime_diff = levels.get("Skew_Regime_Diff")
        if skew_regime_diff is not None and skew_regime_diff < config.ma_crossover_skew_regime_min:
            continue

        # See simulate_squeeze_breakout_signals()'s identical comment --
        # never a gate here either, purely so backtest trades carry a real
        # trade_score for ic_tracking.backtest_ic_check().
        scored = add_ma_crossover_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.ma_crossover_entry_fill == "next_open":
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
            "signal": "MA_Crossover",
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


def simulate_random_ma_crossover_entries(
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
    """Random-entry benchmark for simulate_ma_crossover_signals() -- same
    idea as every other simulate_random_*_entries() this session, using
    this strategy's own gates (macro uptrend, liquidity via
    compute_ma_crossover_levels) and the SAME
    config.ma_crossover_entry_fill-selected fill mechanic, so it isolates
    whether MA-crossover TIMING adds value over a random day using the
    identical Buy_Price formula (that day's own Close) and entry/exit
    structure. `n_trades` should be simulate_ma_crossover_signals()'s real
    signal count for this ticker/window, so trade volume is matched.
    THIS IS THE CRITICAL VALIDATION GATE for this strategy -- see
    benchmark_random_entry.py. Deliberately checked BEFORE any filter or
    tuning work, not after.

    `earnings_dates` (optional, see simulate_squeeze_breakout_signals())
    only makes each trade dict's own `catalyst_warning` field honest --
    the candidate pool itself stays UNFILTERED by earnings proximity,
    same reasoning as simulate_random_squeeze_breakout_entries()."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_ma_crossover_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, crossover or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = ma_crossover_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.ma_crossover_entry_fill == "next_open":
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
            "signal": "Random_MA_Crossover",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades
