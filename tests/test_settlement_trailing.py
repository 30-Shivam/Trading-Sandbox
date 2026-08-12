"""Promoted from this session's own ad-hoc verification script into
permanent coverage -- swingtrade.settlement.settle_trade_with_trailing().
Caught a real look-ahead sequencing bug during original verification (was
checking today's gap against a trailing level computed from today's OWN
not-yet-known high); these synthetic price-path cases are what caught it.
"""
import pandas as pd
import pytest

import swingtrade
from swingtrade.settlement import settle_trade, settle_trade_with_trailing

BUY_PRICE = 100.0
ATR = 5.0
CONFIG = swingtrade.DEFAULT_CONFIG
STOP_LOSS = round(BUY_PRICE - CONFIG.stop_loss_atr_multiplier * ATR, 2)     # 95.0
SELL_PRICE = round(BUY_PRICE + CONFIG.atr_take_profit_multiplier * ATR, 2)  # 110.0
TRAIL_CONFIG = swingtrade.TradingConfig(**{
    **CONFIG.to_dict(), "trailing_stop_enabled": True, "trailing_stop_atr_multiplier": 1.5,
})


def _bars(ohlc_tuples):
    dates = pd.bdate_range("2024-01-02", periods=len(ohlc_tuples))
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c} for o, h, l, c in ohlc_tuples], index=dates,
    )


def test_never_reaches_target_is_byte_identical_to_fixed_exit():
    bars = _bars([
        (99, 101, 97, 98),
        (98, 100, 96, 97),
        (97, 98, 94, 95),  # low=94 breaches stop_loss=95
    ])
    fixed = settle_trade(BUY_PRICE, STOP_LOSS, SELL_PRICE, bars, CONFIG)
    trailing = settle_trade_with_trailing(BUY_PRICE, STOP_LOSS, SELL_PRICE, ATR, bars, TRAIL_CONFIG)
    assert fixed == trailing
    assert fixed["status"] == "LOSS"


def test_climbs_after_target_captures_more_than_fixed_exit():
    bars = _bars([
        (101, 105, 100, 104),
        (104, 111, 103, 109),   # touches target=110 intraday -> trailing starts
        (109, 120, 108, 118),
        (118, 130, 116, 128),
        (128, 129, 108, 110),   # sharp pullback, hits trailing stop
    ])
    fixed = settle_trade(BUY_PRICE, STOP_LOSS, SELL_PRICE, bars, CONFIG)
    trailing = settle_trade_with_trailing(BUY_PRICE, STOP_LOSS, SELL_PRICE, ATR, bars, TRAIL_CONFIG)
    assert fixed["status"] == "WIN" and fixed["exit_price"] == 110.0
    assert trailing["status"] == "WIN"
    assert trailing["pnl_pct"] > fixed["pnl_pct"]
    assert "trailing" in trailing["exit_reason"]


def test_reverses_hard_still_resolves_sanely():
    bars = _bars([
        (104, 112, 103, 111),   # touches target intraday -> trailing starts, trailing_stop=112-7.5=104.5
        (111, 111, 90, 91),     # crashes well below trailing_stop
    ])
    trailing = settle_trade_with_trailing(BUY_PRICE, STOP_LOSS, SELL_PRICE, ATR, bars, TRAIL_CONFIG)
    assert trailing["status"] in ("WIN", "LOSS")
    assert "trailing" in trailing["exit_reason"]
    assert trailing["status"] == "WIN"  # exit near 104.5, still above buy_price=100


def test_gap_up_through_target_starts_trailing_not_immediate_exit():
    bars = _bars([
        (115, 118, 114, 117),   # opens at 115 >= sell_price=110 -- gap up through target
        (117, 125, 116, 123),
        (123, 124, 100, 102),   # sharp reversal, hits trailing stop
    ])
    fixed = settle_trade(BUY_PRICE, STOP_LOSS, SELL_PRICE, bars, CONFIG)
    trailing = settle_trade_with_trailing(BUY_PRICE, STOP_LOSS, SELL_PRICE, ATR, bars, TRAIL_CONFIG)
    assert fixed["status"] == "WIN" and fixed["exit_price"] == 115.0
    assert trailing["pnl_pct"] > fixed["pnl_pct"]


def test_grinds_to_max_holding_days_while_trailing():
    short_max_holding = swingtrade.TradingConfig(**{**TRAIL_CONFIG.to_dict(), "max_holding_days": 3})
    bars = _bars([
        (104, 112, 103, 111),   # touches target intraday, trailing starts (day 1)
        (111, 113, 110, 112),   # day 2
        (112, 114, 111, 113),   # day 3 -- hits max_holding_days, never touched trailing stop
    ])
    result = settle_trade_with_trailing(BUY_PRICE, STOP_LOSS, SELL_PRICE, ATR, bars, short_max_holding)
    assert result["status"] in ("WIN", "LOSS")
    assert result["exit_reason"] == "expired_after_target"
    assert result["status"] == "WIN"


def test_trailing_stop_enabled_defaults_to_false():
    assert swingtrade.DEFAULT_CONFIG.trailing_stop_enabled is False


@pytest.mark.parametrize("strategy", ["breakout", "squeeze_breakout", "ma_crossover"])
def test_settle_helper_is_noop_when_disabled(strategy, uptrend_ohlcv, market_ohlcv):
    """Real-data no-op regression: trailing_stop_enabled=False must route
    through the exact same settle_trade() call as before this feature
    existed, for each of the 3 currently-active strategies."""
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy})
    explicit_off = swingtrade.TradingConfig(**{**config.to_dict(), "trailing_stop_enabled": False})

    simulate_fn = {
        "breakout": swingtrade.simulate_breakout_signals,
        "squeeze_breakout": swingtrade.simulate_squeeze_breakout_signals,
        "ma_crossover": swingtrade.simulate_ma_crossover_signals,
    }[strategy]

    start, end = uptrend_ohlcv.index[0], uptrend_ohlcv.index[-1]
    trades_base = simulate_fn("TEST", uptrend_ohlcv, market_ohlcv, start, end, config)
    trades_off = simulate_fn("TEST", uptrend_ohlcv, market_ohlcv, start, end, explicit_off)
    assert trades_base == trades_off
