"""LLM-proposed rule evaluation and its level calculator."""

import operator
import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig
from .technical import precompute_breakout_frame


# LLM-invented strategy research (2026-08-22, improvements.txt item 93) --
# the LLM composes a structured JSON rule (which already-computed
# indicators to condition on, which already-built exit mechanic to use),
# never generates or executes code. See llm_strategy_research.py for the
# proposal/validation/orchestration layer; everything here is the generic,
# human-written interpreter every rule runs through.

KNOWN_INDICATOR_FIELDS = (
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "ATR", "ADX", "SMA_TREND", "AvgVolume", "AvgVolume_Prior",
    "Highest_High", "OBV_Zscore", "Squeeze_Zscore", "Relative_Strength",
)  # every column precompute_breakout_frame() actually guarantees (raw OHLCV


   # always; Relative_Strength only when market_df is supplied, same as
   # every other caller of that function) -- deliberately NOT the full set
   # every OTHER strategy's own extension adds (e.g. Recent_Min_Squeeze_Zscore
   # is squeeze_breakout-specific, Insider_Purchase_Value needs external
   # fetched data) -- this generic engine only calls the shared base, so
   # only the shared base's own columns are honest to offer.

_ALLOWED_CONDITION_OPS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
}


def evaluate_llm_rule_conditions(frame: pd.DataFrame, rule: dict) -> pd.Series:
    """Pure, vectorized evaluation of one LLM-proposed rule's `conditions`
    against an already-precomputed frame (any frame carrying
    KNOWN_INDICATOR_FIELDS' columns -- precompute_llm_strategy_frame()'s
    own output). Each condition is `frame[field] {op} value`; combined via
    `rule["logic"]` ("AND" or "OR") across every condition.

    Raises ValueError (never silently ignores or coerces) on any
    unrecognized field/op/logic, a field not actually present in this
    specific frame (e.g. Relative_Strength requested but market_df wasn't
    supplied upstream), or a non-numeric condition value -- same
    "malformed input is never more trustworthy than no input" discipline
    llm_agent._parse_response() already applies to raw LLM JSON elsewhere
    in this codebase. Callers (llm_strategy_research.validate_rule()) are
    expected to catch and reject a malformed rule BEFORE it ever reaches
    a real backtest.

    NaN comparisons (insufficient warmup history for a given indicator)
    evaluate to False, never True -- a condition can only fire once its
    own indicator has real data, same as every other strategy's own
    NaN-during-warmup handling."""
    conditions = rule.get("conditions") or []
    if not conditions:
        raise ValueError("rule has no conditions")
    logic = rule.get("logic", "AND")
    if logic not in ("AND", "OR"):
        raise ValueError(f"unknown logic: {logic!r} (expected 'AND' or 'OR')")

    masks = []
    for cond in conditions:
        field = cond.get("field")
        op = cond.get("op")
        value = cond.get("value")
        if field not in KNOWN_INDICATOR_FIELDS:
            raise ValueError(f"unknown field: {field!r}")
        if field not in frame.columns:
            raise ValueError(f"field {field!r} not present in this frame (was market_df supplied?)")
        if op not in _ALLOWED_CONDITION_OPS:
            raise ValueError(f"unknown op: {op!r} (expected one of {sorted(_ALLOWED_CONDITION_OPS)})")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"condition value must be numeric, got {value!r}")
        masks.append(_ALLOWED_CONDITION_OPS[op](frame[field], value))

    combined = masks[0]
    for mask in masks[1:]:
        combined = (combined & mask) if logic == "AND" else (combined | mask)
    return combined.fillna(False)


def rule_exit_to_config(rule: dict, config: TradingConfig = DEFAULT_CONFIG) -> TradingConfig:
    """Builds a per-rule effective TradingConfig from `rule["exit"]`'s own
    chosen exit mechanic/params, reusing TradingConfig's EXISTING
    atr_take_profit_multiplier/stop_loss_atr_multiplier/
    trailing_stop_atr_multiplier/trailing_stop_enabled/max_holding_days
    fields -- the LLM only SELECTS and PARAMETERIZES among this project's
    already-built, already-tested exit mechanics
    (swingtrade.settle_trade()/settle_trade_with_trailing()), never
    invents new settlement code. Every numeric param is expected to
    already be validated/clamped by llm_strategy_research.validate_rule()
    before this is called -- this function trusts its input, it doesn't
    re-validate it."""
    exit_spec = rule["exit"]
    overrides = {
        "atr_take_profit_multiplier": exit_spec["take_profit_atr_multiplier"],
        "stop_loss_atr_multiplier": exit_spec["stop_loss_atr_multiplier"],
        "trailing_stop_enabled": exit_spec["type"] == "trailing_stop",
    }
    if exit_spec["type"] == "trailing_stop":
        overrides["trailing_stop_atr_multiplier"] = exit_spec["trailing_stop_atr_multiplier"]
    if exit_spec["type"] == "time_based":
        overrides["max_holding_days"] = exit_spec["max_holding_days"]
    return TradingConfig(**{**config.to_dict(), **overrides})


def precompute_llm_strategy_frame(
    df: pd.DataFrame,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized precompute for one LLM-proposed rule -- built on top of
    precompute_breakout_frame() (reused wholesale, exactly like every
    other strategy, for SMA_TREND/ATR/RSI/etc. and the shared
    macro-uptrend/liquidity gate every strategy respects unconditionally).
    `LLM_Strategy_Signal` is the rule's own conditions evaluated via
    evaluate_llm_rule_conditions() -- the ONLY thing this rule actually
    controls; the macro-uptrend/liquidity checks stay enforced identically
    for every rule, same as every other strategy family, in
    llm_strategy_levels_from_frame() below."""
    df = precompute_breakout_frame(df, config, market_df=market_df)
    df["LLM_Strategy_Signal"] = evaluate_llm_rule_conditions(df, rule)
    return df


def llm_strategy_levels_from_frame(
    ticker: str,
    frame: pd.DataFrame,
    as_of,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
) -> dict:
    """Extract one LLM-proposed rule's dict for one row of a frame already
    built by precompute_llm_strategy_frame() -- the O(1)-per-row
    counterpart a walk-forward loop calls once per `as_of`. Same
    macro-uptrend/liquidity gates every strategy shares (via
    precompute_breakout_frame(), reused wholesale) -- NOT something any
    rule can configure around.

    `config` is expected to already be the RULE's own effective config
    (see rule_exit_to_config()) -- Sell_Price/Stop_Loss below read
    config.atr_take_profit_multiplier/stop_loss_atr_multiplier directly,
    same as every other strategy's own levels_from_frame()."""
    last_row = frame.loc[as_of]
    last_date = as_of
    last_close, sma_trend, atr, avg_volume, rsi = (
        last_row["Close"], last_row["SMA_TREND"], last_row["ATR"], last_row["AvgVolume"], last_row["RSI"],
    )
    llm_signal = bool(last_row["LLM_Strategy_Signal"])
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
    # the LLM's own rule, not RSI specifically) -- same treatment every
    # other non-RSI strategy gives it.
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

    # Buy_Price = today's own Close -- same "already happening, buy near
    # market" convention squeeze_breakout/pairs/insider_buying use.
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
        "LLM_Strategy_Signal": llm_signal,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Stop_Loss": stop_loss,
        "RRR": rrr,
        "Distance_to_Buy_Pct": distance_to_buy_pct,
        "Next_Earnings_Date": next_earnings_date_out,
        "Catalyst_Warning": catalyst_warning,
        "Top_Headline": top_headline,
    }


def compute_llm_strategy_levels(
    ticker: str,
    df: pd.DataFrame,
    rule: dict,
    config: TradingConfig = DEFAULT_CONFIG,
    next_earnings_date=None,
    top_headline: str = "",
    market_df: pd.DataFrame | None = None,
) -> dict:
    """Compute one LLM-proposed rule's levels for one ticker's OHLCV
    history -- see precompute_llm_strategy_frame()'s own docstring for the
    full mechanism. Thin wrapper, same pattern as every other
    compute_X_levels() (see compute_squeeze_breakout_levels())."""
    effective_config = rule_exit_to_config(rule, config)
    frame = precompute_llm_strategy_frame(df, rule, effective_config, market_df=market_df)
    as_of = frame.index[-1]
    return llm_strategy_levels_from_frame(ticker, frame, as_of, effective_config, next_earnings_date, top_headline)
