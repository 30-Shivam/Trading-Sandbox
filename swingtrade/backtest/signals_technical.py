"""Price/volume-pattern strategy simulators (RSI, breakout, pullback, retest, 52-week high, momentum burst, squeeze) and their random-entry baselines."""

import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from ..levels import (
    breakout_levels_from_frame,
    breakout_retest_levels_from_frame,
    levels_from_rsi_frame,
    market_uptrend_from_frame,
    momentum_burst_levels_from_frame,
    precompute_breakout_frame,
    precompute_breakout_retest_frame,
    precompute_momentum_burst_frame,
    precompute_pullback_frame,
    precompute_rsi_frame,
    precompute_squeeze_breakout_frame,
    precompute_week52_frame,
    pullback_levels_from_frame,
    squeeze_breakout_levels_from_frame,
    week52_levels_from_frame,
)
from ..scoring import add_squeeze_breakout_trade_score, add_trade_score
from ..settlement import settle_trade
from .core import (
    ENTRY_SIGNALS,
    _find_breakout_fill,
    _find_entry_fill,
    _find_next_open_fill,
    _next_earnings_date,
    _settle,
)


def simulate_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
) -> list[dict]:
    """Walk every trading day in [window_start, window_end) for one ticker,
    simulating a trade wherever swingtrade would fire a Strong Buy/Buy
    signal using only data available as of that day. A signal can only be
    known AFTER the close it's computed from, so entry is NOT assumed on
    that same bar -- _find_entry_fill() walks forward from the next session
    looking for the resting limit order at Buy_Price to actually get
    touched (gap-aware, same logic as stop/target fills), up to
    config.max_entry_wait_days out. A signal that's never touched within
    that window produces no trade at all -- correctly excluded, not forced.
    Each trade dict carries both `signal_date` (when the signal fired) and
    `entry_date` (when it actually filled, used for cluster-weighting --
    see compute_cluster_weights); `buy_price` is the real fill price
    (better than the signal's theoretical level on a gap), with the
    original `signal_buy_price` kept for reference. Each filled trade is
    then resolved with settle_trade() against the ticker's actual
    subsequent price history (which may extend past window_end).

    `ohlcv` and `market_ohlcv` must each contain enough leading history
    before window_start to cover config.sma_trend_window -- see
    LOOKBACK_BUFFER_BARS and run_backtest.py's fetch buffering.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False.

    `sector` (optional, see watchlist.read_ticker_sectors) is tagged onto
    every trade so compute_cluster_weights() can identify same-day,
    same-sector clusters -- unknown/untagged tickers default to "Unknown"
    and are never clustered with each other (no basis to assume they're
    correlated).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    # Precomputed ONCE per ticker/fold instead of being recomputed from
    # scratch at every as_of -- see levels.precompute_rsi_frame's docstring
    # for why this is an exact equivalence, not an approximation.
    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_rsi_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue  # insufficient market history yet
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = levels_from_rsi_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue  # insufficient history / macro downtrend / illiquid that day

        scored = add_trade_score(pd.DataFrame([levels]), config).iloc[0]
        if scored["Signal"] not in ENTRY_SIGNALS:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(scored["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue  # resting order never touched within the window -- not a real trade
        entry_date, entry_price = fill

        # Stop/target are risk-management levels relative to the ACTUAL entry
        # price, not the original signal's theoretical Buy_Price -- matters
        # when a gap-down fill gets a better (lower) entry than the signal
        # anticipated. Leaving them anchored to the stale higher Buy_Price
        # would put the stop ABOVE the real entry price on a gap fill,
        # causing a near-instant fake stop-out on what was actually a good
        # (better-than-limit) fill. Re-anchoring to entry_price keeps the
        # stop below and the target above wherever you actually got in --
        # identical to the original scored levels in the non-gap case, since
        # entry_price == signal Buy_Price there.
        atr = float(scored["ATR"])
        stop_loss = round(entry_price - config.stop_loss_atr_multiplier * atr, 2)
        sell_price = round(entry_price + config.atr_take_profit_multiplier * atr, 2)

        bars_since_entry = ohlcv[ohlcv.index > entry_date]
        result = settle_trade(
            buy_price=entry_price,
            stop_loss=stop_loss,
            sell_price=sell_price,
            bars_since_entry=bars_since_entry,
            config=config,
            dollar_volume=scored.get("Dollar_Volume"),
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            # "signal" is a STABLE per-strategy label here (2026-09-14,
            # improvements.txt item 150) -- every other simulate_*_signals()
            # tags its own trade dicts this way ("MA_Crossover", "Pairs",
            # etc.), a real, previously-unnoticed inconsistency this
            # function alone didn't follow (it stored the live, per-trade
            # Buy/Strong Buy value instead, which fragmented this
            # strategy's own attribution across multiple inconsistent keys
            # anywhere something groups trades by "signal" -- found via
            # swingtrade.compute_strategy_correlation()/
            # simulate_portfolio_constrained(group_key="signal")'s own real
            # multi-strategy use in benchmark_multi_strategy_portfolio.py).
            # The original live value is NOT discarded -- see
            # `live_signal` below.
            "signal": "RSI",
            "live_signal": scored["Signal"],
            "trade_score": float(scored["Trade_Score"]),
            "rsi": float(scored["RSI"]),
            "atr": atr,
            "rrr": float(scored["RRR"]),
            "buy_price": entry_price,
            "signal_buy_price": float(scored["Buy_Price"]),
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": bool(scored["Catalyst_Warning"]),
            **result,
        })

    return trades


def simulate_random_entries(
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
    """Diagnostic benchmark, NOT a production strategy -- see
    benchmark_random_entry.py. Identical universe gates (macro uptrend,
    liquidity/history via compute_levels) and identical entry-fill/stop/
    target mechanics as simulate_signals(), but the TRIGGER day is chosen
    uniformly at random from every day that passed the gates, instead of
    requiring an RSI-oversold Strong Buy/Buy signal. `n_trades` should be
    the real strategy's signal count for this ticker/window (see
    simulate_signals) so trade volume is matched -- isolates whether RSI
    timing itself adds value over picking days at random with the exact
    same underlying entry/exit structure and macro/liquidity filtering.
    If RSI-timed entries don't meaningfully beat this, the RSI signal
    carries little real predictive information."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_rsi_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the same gates simulate_signals uses
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = levels_from_rsi_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append(
            (as_of, float(levels["Buy_Price"]), float(levels["ATR"]), levels.get("Dollar_Volume"))
        )

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, buy_price, atr, dollar_volume in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
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
            dollar_volume=dollar_volume,
        )

        trades.append({
            "ticker": ticker,
            "signal_date": as_of.date(),
            "entry_date": entry_date.date(),
            "sector": sector,
            "signal": "Random",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_breakout_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
) -> list[dict]:
    """Trend-following counterpart to simulate_signals() -- buys strength (a
    new config.breakout_lookback_days-day closing high in a confirmed
    uptrend) instead of RSI-oversold weakness. Same no-look-ahead discipline
    (only data through `as_of` decides whether a signal fires), same
    entry-timing realism (a signal known at `as_of`'s close can't be acted
    on before the next session -- _find_breakout_fill() walks forward
    looking for the resting stop-buy order to actually get touched, gap-aware,
    within config.max_entry_wait_days), and the exact same ATR-multiple
    stop/target sizing as simulate_signals() -- deliberately kept identical
    so this is a clean test of whether breakout TIMING adds value, without
    also changing the exit model at the same time.

    A breakout already at/above config.breakout_rsi_overbought_threshold is
    skipped -- an over-extended/exhausted move is more likely to fail or
    reverse than one fresh off a quiet consolidation. Default threshold
    (100.0) is a practical no-op, since RSI essentially never reaches it;
    this filter is a documented, unvalidated hypothesis until tested (see
    improvements.txt) -- kept as a plain skip, not baked into the trigger
    definition itself, so it stays a searchable/toggleable parameter.

    A breakout whose Relative_Strength (ticker return minus market return
    over the same breakout_lookback_days window -- see
    compute_relative_strength) is below config.breakout_relative_strength_min
    is also skipped -- a stock breaking out only because the whole market is
    ripping isn't the same as one genuinely beating the market. Default
    (-100.0) is a practical no-op.

    A breakout whose Volume_Ratio (today's Volume over the PRIOR
    volume_lookback_days average -- see compute_breakout_levels) is below
    config.breakout_volume_ratio_min is also skipped -- a genuine breakout
    on high volume vs. a low-volume drift above an old high are different
    events. Default (0.0) is a practical no-op.

    A breakout whose ADX (trend strength, independent of direction -- see
    compute_breakout_levels) is below config.breakout_adx_min is also
    skipped -- a breakout during a weak/choppy trend and one during a
    genuinely strong trend look identical to every other filter here, but
    aren't the same event. Default (0.0) is a practical no-op.

    A breakout whose OBV_Zscore (On-Balance Volume z-scored against its
    own recent baseline -- see compute_breakout_levels) is below
    config.breakout_obv_zscore_min is also skipped -- sustained buying
    pressure building up over weeks is a deeper signal than a single day's
    volume spike. Default (-100.0) is a practical no-op.

    A breakout whose Squeeze_Zscore (prior-day volatility z-scored against
    its own recent baseline -- see compute_breakout_levels) is above
    config.breakout_squeeze_zscore_max is also skipped -- a breakout
    emerging from a period of volatility contraction ("coiled spring") is
    a different, often more reliable event than one that isn't. Default
    (100.0) is a practical no-op.

    `ohlcv`/`market_ohlcv` need the same leading-history buffer as
    simulate_signals() -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_frame(ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Breakout_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.breakout_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.breakout_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.breakout_volume_ratio_min:
            continue
        adx = levels.get("ADX")
        if adx is not None and adx < config.breakout_adx_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.breakout_obv_zscore_min:
            continue
        squeeze_zscore = levels.get("Squeeze_Zscore")
        if squeeze_zscore is not None and squeeze_zscore > config.breakout_squeeze_zscore_max:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.breakout_sector_relative_strength_min:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_breakout_fill(levels["Trigger_Price"], bars_after_signal, config.max_entry_wait_days)
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
            "signal": "Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Trigger_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_breakout_entries(
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
    """Random-entry benchmark for simulate_breakout_signals() -- same idea as
    simulate_random_entries(), but using the breakout strategy's own gates
    (macro uptrend, liquidity via compute_breakout_levels) and stop-buy fill
    mechanics (_find_breakout_fill), so it isolates whether breakout-day
    TIMING adds value over a random day using the identical trigger-price
    formula (that day's own N-day high) and entry/exit structure. `n_trades`
    should be simulate_breakout_signals()'s real signal count for this
    ticker/window, so trade volume is matched."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    # No market_df here -- matches the original's deliberate omission (the
    # random-entry benchmark never applies the relative-strength/volume/
    # overbought filters, only the shared macro-uptrend/liquidity gates).
    frame = precompute_breakout_frame(ohlcv, config)

    candidates = []  # (as_of, trigger_price, atr) for every day that passed the gates, breakout or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        candidates.append((as_of, levels["Trigger_Price"], float(levels["ATR"])))

    if not candidates or n_trades <= 0:
        return []

    chosen = rng.sample(candidates, k=min(n_trades, len(candidates)))
    chosen.sort(key=lambda c: c[0])

    trades = []
    for as_of, trigger_price, atr in chosen:
        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_breakout_fill(trigger_price, bars_after_signal, config.max_entry_wait_days)
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
            "signal": "Random_Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": trigger_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_pullback_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Pullback-in-uptrend counterpart to simulate_signals()/simulate_breakout_signals()
    -- buys a shallow dip toward a rising config.pullback_ma_window-day SMA
    within a confirmed macro uptrend, instead of RSI-oversold weakness
    (simulate_signals) or a fresh N-day closing high (simulate_breakout_signals).
    Same no-look-ahead discipline, same entry-timing realism -- but unlike
    breakout's resting STOP-BUY (_find_breakout_fill), Pullback_Signal's
    Buy_Price is a resting LIMIT order at the pullback MA level, filled via
    the same _find_entry_fill() mechanics as simulate_signals() (a downside
    touch, not an upside cross). Same ATR-multiple stop/target sizing as
    both other strategies, deliberately kept identical so this is a clean
    test of whether pullback TIMING adds value, without also changing the
    exit model.

    `market_ohlcv` is accepted for signature consistency with the other two
    simulate_*() functions (uniform calling convention for run_backtest()),
    but unused here -- v1 of this strategy has no Relative_Strength concept
    (see compute_pullback_levels's docstring on why this is a deliberately
    lean first version).

    `ohlcv` needs the same leading-history buffer as the other two
    strategies -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pullback_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = pullback_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Pullback_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        fill = _find_entry_fill(levels["Buy_Price"], bars_after_signal, config.max_entry_wait_days)
        if fill is None:
            continue
        entry_date, entry_price = fill

        # Same re-anchor-to-actual-fill-price reasoning as simulate_signals()
        # -- a gap-down fill gets a better (lower) entry than the signal
        # anticipated, so stop/target are relative to entry_price, not the
        # stale signal Buy_Price.
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
            "signal": "Pullback",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_pullback_entries(
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
    """Random-entry benchmark for simulate_pullback_signals() -- same idea as
    simulate_random_entries()/simulate_random_breakout_entries(), using the
    pullback strategy's own gates (macro uptrend, liquidity, rising MA via
    compute_pullback_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether pullback-day TIMING adds
    value over a random day using the identical Buy_Price formula (that
    day's own pullback MA level) and entry/exit structure. `n_trades`
    should be simulate_pullback_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_pullback_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, pullback or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = pullback_levels_from_frame(ticker, frame, as_of, config)
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
            "signal": "Random_Pullback",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_breakout_retest_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Breakout-retest counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals() -- buys a pullback BACK TO a recent genuine
    breakout's own trigger level, instead of RSI-oversold weakness, a fresh
    N-day high THE SAME DAY, or proximity to a rising MA regardless of any
    prior breakout. Built after both simulate_signals() and
    simulate_pullback_signals() lost to matched-count random-entry timing
    on held-out tickers (see benchmark_random_entry.py), keeping the one
    ingredient that DID show a real edge (simulate_breakout_signals'
    trigger definition) while relaxing its "must fire today" restriction.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at the original breakout's level, filled via
    the same _find_entry_fill() mechanics as simulate_signals()/
    simulate_pullback_signals() (a downside touch), NOT breakout's resting
    STOP-BUY (_find_breakout_fill(), an upside cross). Same ATR-multiple
    stop/target sizing as every other strategy, deliberately kept identical
    so this is a clean test of whether retest TIMING adds value.

    `market_ohlcv` is accepted for signature consistency with the other
    three simulate_*() functions but unused here, same as
    simulate_pullback_signals() -- v1 has no Relative_Strength concept.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_retest_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_retest_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Retest_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
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
            "signal": "Breakout_Retest",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_breakout_retest_entries(
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
    """Random-entry benchmark for simulate_breakout_retest_signals() -- same
    idea as the other three simulate_random_*_entries() functions, using
    this strategy's own gates (macro uptrend, liquidity, a recent breakout
    to retest via compute_breakout_retest_levels) and limit-order fill
    mechanics (_find_entry_fill), so it isolates whether retest-day TIMING
    adds value over a random day using the identical Buy_Price formula
    (that day's own eligible retest level) and entry/exit structure.
    `n_trades` should be simulate_breakout_retest_signals()'s real signal
    count for this ticker/window, so trade volume is matched. This is the
    critical validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_breakout_retest_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, retest-eligible or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = breakout_retest_levels_from_frame(ticker, frame, as_of, config)
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
            "signal": "Random_Breakout_Retest",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_week52_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """52-week-high-momentum counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals() -- buys
    when price is within config.week52_nearness_pct of its own trailing
    config.week52_lookback_days high, in a confirmed macro uptrend. Unlike
    every prior strategy, this is a continuous STATE rather than a
    discrete event, so it can fire on many consecutive days while a stock
    consolidates near its highs.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at essentially today's own Close (already
    near strength, not waiting for a specific dip level), filled via the
    same _find_entry_fill() mechanics as pullback/breakout_retest. Same
    ATR-multiple stop/target sizing as every other strategy.

    `market_ohlcv` is accepted for signature consistency with the other
    four simulate_*() functions but unused here -- no Relative_Strength
    concept in this v1, same as pullback/breakout_retest.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS (note: week52_lookback_days defaults to
    252, noticeably longer than the other strategies' windows, so callers
    fetching history for this strategy need a correspondingly larger
    buffer -- see run_backtest.py's fetch buffering).
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_week52_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = week52_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Week52_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
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
            "signal": "Week52_High",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_week52_entries(
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
    """Random-entry benchmark for simulate_week52_signals() -- same idea as
    the other four simulate_random_*_entries() functions, using this
    strategy's own gates (macro uptrend, liquidity via
    compute_week52_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether being-near-a-52-week-high
    TIMING adds value over a random day using the identical Buy_Price
    formula (that day's own Close) and entry/exit structure. `n_trades`
    should be simulate_week52_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_week52_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, near-high or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = week52_levels_from_frame(ticker, frame, as_of, config)
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
            "signal": "Random_Week52_High",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_momentum_burst_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    sector: str = "Unknown",
) -> list[dict]:
    """Momentum-burst counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()
    -- buys a single day's strong price gain CONFIRMED by unusually high
    volume, in a confirmed macro uptrend. Built to fire MORE OFTEN than any
    prior strategy: unlike breakout (a discrete "new high TODAY" event) or
    breakout_retest/week52_high (both anchored to a specific high level),
    this doesn't require a fresh high at all -- just a forceful, volume-
    backed move, which can happen on many more days than a genuine N-day
    high.

    Same no-look-ahead discipline, same entry-timing realism -- Buy_Price
    is a resting LIMIT order at essentially today's own Close (same
    convention as week52_high: already happening, not waiting for a
    specific dip level), filled via the same _find_entry_fill() mechanics
    as pullback/breakout_retest/week52_high, deliberately NOT a bespoke
    "buy at the next Open" mechanic -- keeping the identical fill mechanic
    across every "Buy_Price = today's Close" strategy keeps this an
    apples-to-apples comparison with week52_high specifically, and matches
    this codebase's one established precedent for this exact situation.
    Same ATR-multiple stop/target sizing as every other strategy.

    `market_ohlcv` is accepted for signature consistency with the other
    five simulate_*() functions but unused here -- no Relative_Strength
    concept in this v1, same as pullback/breakout_retest/week52_high.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_burst_frame(ohlcv, config)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = momentum_burst_levels_from_frame(ticker, frame, as_of, config)
        except RuntimeError:
            continue

        if not levels["Momentum_Signal"]:
            continue

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.momentum_burst_entry_fill == "next_open":
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
            "signal": "Momentum_Burst",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": levels["Buy_Price"],
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_random_momentum_burst_entries(
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
    """Random-entry benchmark for simulate_momentum_burst_signals() -- same
    idea as the other five simulate_random_*_entries() functions, using
    this strategy's own gates (macro uptrend, liquidity via
    compute_momentum_burst_levels) and limit-order fill mechanics
    (_find_entry_fill), so it isolates whether momentum-burst TIMING adds
    value over a random day using the identical Buy_Price formula (that
    day's own Close) and entry/exit structure. `n_trades` should be
    simulate_momentum_burst_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_momentum_burst_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr) for every day that passed the macro/liquidity gates, burst or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        try:
            levels = momentum_burst_levels_from_frame(ticker, frame, as_of, config)
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
        if config.momentum_burst_entry_fill == "next_open":
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
            "signal": "Random_Momentum_Burst",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": False,
            **result,
        })

    return trades


def simulate_squeeze_breakout_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector: str = "Unknown",
    sector_ohlcv: pd.DataFrame | None = None,
) -> list[dict]:
    """Squeeze-breakout counterpart to simulate_signals()/simulate_breakout_signals()/
    simulate_pullback_signals()/simulate_breakout_retest_signals()/simulate_week52_signals()/
    simulate_momentum_burst_signals() -- buys a real directional expansion
    following a recent volatility contraction (squeeze), in a confirmed
    macro uptrend. Deliberately does NOT require a fresh high over any
    window (unlike breakout/breakout_retest/week52_high) and does NOT
    require volume confirmation (unlike momentum_burst) -- a materially
    different, and typically more frequent, trigger.

    Same no-look-ahead discipline, same entry-timing realism.
    config.squeeze_breakout_entry_fill selects the fill model, same
    toggle momentum_burst_entry_fill added: "limit" (default) is a
    resting LIMIT order at today's own Close, filled via
    _find_entry_fill() (same convention week52_high/momentum_burst use);
    "next_open" buys the very next session's Open unconditionally via
    _find_next_open_fill() -- see improvements.txt items 36-37 for why
    this choice can materially move backtest conclusions for a same-day
    "chase an expansion" signal, which is why it's built in here from the
    start rather than retrofitted later. Same ATR-multiple stop/target
    sizing as every other strategy.

    `market_ohlcv` enables the optional Phase 2 Relative_Strength filter
    (see swingtrade/config.py's squeeze_breakout_relative_strength_min) --
    same role as simulate_breakout_signals()'s/simulate_adx_trend_entry_signals()'s
    own market_df threading. Phase 2 also gates on RSI-overbought/
    Volume_Ratio/ADX/OBV_Zscore, all disabled by default (see
    add_squeeze_breakout_trade_score()'s docstring for the full list) --
    kept in sync with that function so live and backtested definitions
    can't silently disagree, same discipline breakout's/adx_trend_entry's
    own filters use.

    `ohlcv` needs the same leading-history buffer as the other strategies
    -- see LOOKBACK_BUFFER_BARS.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False --
    same role as simulate_signals()'s own `earnings_dates` param. Previously
    this strategy's backtested Catalyst_Warning was ALWAYS False regardless
    of real earnings proximity (a genuine, previously-undiscovered gap
    between what the live dashboard shows and what was actually backtested
    -- see improvements.txt), since nothing threaded earnings data through
    here before this parameter was added.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_squeeze_breakout_frame(ohlcv, config, market_df=market_ohlcv, sector_df=sector_ohlcv)

    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = squeeze_breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue

        if not levels["Squeeze_Signal"]:
            continue
        rsi = levels.get("RSI")
        if rsi is not None and rsi >= config.squeeze_breakout_rsi_overbought_threshold:
            continue
        rel_strength = levels.get("Relative_Strength")
        if rel_strength is not None and rel_strength < config.squeeze_breakout_relative_strength_min:
            continue
        volume_ratio = levels.get("Volume_Ratio")
        if volume_ratio is not None and volume_ratio < config.squeeze_breakout_volume_ratio_min:
            continue
        adx = levels.get("ADX")
        if adx is not None and adx < config.squeeze_breakout_adx_min:
            continue
        obv_zscore = levels.get("OBV_Zscore")
        if obv_zscore is not None and obv_zscore < config.squeeze_breakout_obv_zscore_min:
            continue
        sector_rel_strength = levels.get("Sector_Relative_Strength")
        if sector_rel_strength is not None and sector_rel_strength < config.squeeze_breakout_sector_relative_strength_min:
            continue
        if config.squeeze_breakout_earnings_gate and levels["Catalyst_Warning"]:
            continue

        # Computed here purely so backtest trades carry a real trade_score
        # (see add_squeeze_breakout_trade_score()'s own docstring: kept in
        # sync with the live gate above by construction) -- never used as a
        # gate itself, ENTRY_SIGNALS-style, since this strategy's own entry
        # condition is the boolean Squeeze_Signal plus the optional filters
        # checked above, not a Trade_Score threshold. Backs a NEW
        # backtest-time IC check (does Trade_Score actually rank outcomes,
        # not just "does this strategy beat random on aggregate returns")
        # this project never had a way to compute before -- see
        # ic_tracking.backtest_ic_check().
        scored = add_squeeze_breakout_trade_score(pd.DataFrame([levels]), config).iloc[0]

        bars_after_signal = ohlcv[ohlcv.index > as_of]
        if config.squeeze_breakout_entry_fill == "next_open":
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
            "signal": "Squeeze_Breakout",
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


def simulate_random_squeeze_breakout_entries(
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
    """Random-entry benchmark for simulate_squeeze_breakout_signals() --
    same idea as the other six simulate_random_*_entries() functions,
    using this strategy's own gates (macro uptrend, liquidity via
    compute_squeeze_breakout_levels) and the SAME
    config.squeeze_breakout_entry_fill-selected fill mechanic, so it
    isolates whether squeeze-breakout TIMING adds value over a random day
    using the identical Buy_Price formula (that day's own Close) and
    entry/exit structure. `n_trades` should be
    simulate_squeeze_breakout_signals()'s real signal count for this
    ticker/window, so trade volume is matched. This is the critical
    validation gate for this strategy -- see benchmark_random_entry.py.

    `earnings_dates` (optional, see simulate_squeeze_breakout_signals())
    only makes each trade dict's own `catalyst_warning` field honest --
    the candidate pool itself stays UNFILTERED by earnings proximity
    (deliberately, same as every other optional filter's precedent: this
    random baseline answers "does TIMING add value," not "does this
    optional filter help," so it must not apply any of the strategy's own
    optional gates, earnings included)."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    market_frame = precompute_rsi_frame(market_ohlcv, config)
    frame = precompute_squeeze_breakout_frame(ohlcv, config)

    candidates = []  # (as_of, buy_price, atr, catalyst_warning) for every day that passed the macro/liquidity gates, squeeze or not
    for as_of in eligible_dates:
        try:
            market_uptrend, _, _ = market_uptrend_from_frame(market_frame, as_of, config)
        except RuntimeError:
            continue
        if not market_uptrend:
            continue

        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = squeeze_breakout_levels_from_frame(ticker, frame, as_of, config, next_earnings_date=next_earnings)
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
        if config.squeeze_breakout_entry_fill == "next_open":
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
            "signal": "Random_Squeeze_Breakout",
            "atr": atr,
            "buy_price": entry_price,
            "signal_buy_price": buy_price,
            "stop_loss": stop_loss,
            "sell_price": sell_price,
            "catalyst_warning": catalyst_warning,
            **result,
        })

    return trades
