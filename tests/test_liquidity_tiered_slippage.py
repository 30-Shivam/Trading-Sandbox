"""Liquidity-tiered slippage (config.slippage_liquidity_scaling) -- built
per explicit user request ("lets build it all", 2026-09-13) alongside the
third-universe generalization test, as part of the open-ended "make the
backtest as iron-proof as NautilusTrader" directive. A flat slippage_pct
haircut on every stop-hit fill regardless of a ticker's own liquidity is
unrealistic: a name barely clearing min_dollar_volume realistically slips
more on a forced stop-out than a mega-cap does. Deliberately opt-in
(disabled by default) so no existing backtest/Optuna result changes unless
a caller turns this on.

Covers swingtrade.settlement._effective_slippage_pct() (the pure tier
lookup), settle_trade()/settle_trade_with_trailing() actually USING it
(not just accepting the param), and one real strategy's simulate function
(ma_crossover, chosen since it already threads Dollar_Volume through
_settle()) proving the wiring reaches all the way from
*_levels_from_frame()'s own AvgVolume x Last_Close computation down to the
settlement call -- not just that the plumbing COULD carry a value.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import swingtrade
from swingtrade.settlement import _effective_slippage_pct, settle_trade, settle_trade_with_trailing

CONFIG = swingtrade.DEFAULT_CONFIG  # slippage_liquidity_scaling=False, min_dollar_volume=5_000_000
SCALING_CONFIG = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "slippage_liquidity_scaling": True})
# Tier floors under SCALING_CONFIG's defaults: low=5M*3=15M, mid=5M*10=50M.
LOW_TIER_VOLUME = 10_000_000.0    # < 15M -> LOW tier (2.5x)
MID_TIER_VOLUME = 30_000_000.0    # 15M <= x < 50M -> MID tier (1.5x)
FULL_LIQUIDITY_VOLUME = 100_000_000.0  # >= 50M -> unchanged (1.0x)


# --- _effective_slippage_pct(): pure tier-lookup unit tests ---

def test_scaling_disabled_returns_flat_slippage_regardless_of_volume():
    assert _effective_slippage_pct(LOW_TIER_VOLUME, CONFIG) == CONFIG.slippage_pct
    assert _effective_slippage_pct(FULL_LIQUIDITY_VOLUME, CONFIG) == CONFIG.slippage_pct


def test_missing_dollar_volume_returns_flat_slippage_even_when_scaling_enabled():
    assert _effective_slippage_pct(None, SCALING_CONFIG) == SCALING_CONFIG.slippage_pct


def test_low_tier_applies_low_tier_factor():
    expected = SCALING_CONFIG.slippage_pct * SCALING_CONFIG.slippage_liquidity_low_tier_factor
    assert _effective_slippage_pct(LOW_TIER_VOLUME, SCALING_CONFIG) == pytest.approx(expected)


def test_mid_tier_applies_mid_tier_factor():
    expected = SCALING_CONFIG.slippage_pct * SCALING_CONFIG.slippage_liquidity_mid_tier_factor
    assert _effective_slippage_pct(MID_TIER_VOLUME, SCALING_CONFIG) == pytest.approx(expected)


def test_full_liquidity_returns_flat_slippage():
    assert _effective_slippage_pct(FULL_LIQUIDITY_VOLUME, SCALING_CONFIG) == SCALING_CONFIG.slippage_pct


def test_tier_boundaries_are_half_open():
    low_floor = SCALING_CONFIG.min_dollar_volume * SCALING_CONFIG.slippage_liquidity_low_tier_multiple
    mid_floor = SCALING_CONFIG.min_dollar_volume * SCALING_CONFIG.slippage_liquidity_mid_tier_multiple
    # Exactly AT the low floor no longer qualifies for the low tier (mid tier instead).
    assert _effective_slippage_pct(low_floor, SCALING_CONFIG) == pytest.approx(
        SCALING_CONFIG.slippage_pct * SCALING_CONFIG.slippage_liquidity_mid_tier_factor
    )
    # Exactly AT the mid floor is full liquidity (unchanged).
    assert _effective_slippage_pct(mid_floor, SCALING_CONFIG) == SCALING_CONFIG.slippage_pct


# --- settle_trade() / settle_trade_with_trailing(): the scaled slippage
# actually changes the realized fill price on a stop-hit, not just that the
# helper above returns a different number in isolation. ---

def _bars(ohlc_tuples):
    dates = pd.bdate_range("2024-01-02", periods=len(ohlc_tuples))
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c} for o, h, l, c in ohlc_tuples], index=dates,
    )


def test_settle_trade_low_tier_slips_more_than_full_liquidity_on_stop_hit_intraday():
    buy_price, stop_loss, sell_price = 100.0, 95.0, 110.0
    # Intraday low breaches stop_loss without gapping the Open through it,
    # so this resolves via "stop_hit_intraday" (the only path slippage
    # applies to) rather than a gap fill.
    bars = _bars([(99, 100, 93, 94)])

    illiquid = settle_trade(buy_price, stop_loss, sell_price, bars, SCALING_CONFIG, dollar_volume=LOW_TIER_VOLUME)
    liquid = settle_trade(buy_price, stop_loss, sell_price, bars, SCALING_CONFIG, dollar_volume=FULL_LIQUIDITY_VOLUME)
    flat = settle_trade(buy_price, stop_loss, sell_price, bars, CONFIG, dollar_volume=LOW_TIER_VOLUME)

    assert illiquid["exit_reason"] == liquid["exit_reason"] == "stop_hit_intraday"
    assert illiquid["exit_price"] < liquid["exit_price"], (
        "a thin, barely-liquid name should slip WORSE (lower fill) on a forced stop-out "
        "than a fully-liquid one, once liquidity-tiered slippage is enabled"
    )
    # Scaling disabled must reproduce the ORIGINAL flat-slippage behavior exactly,
    # regardless of dollar_volume -- this is the backward-compatibility guarantee.
    assert flat["exit_price"] == round(stop_loss * (1 - CONFIG.slippage_pct), 2)


def test_settle_trade_with_trailing_low_tier_slips_more_on_trailing_stop_hit():
    buy_price, stop_loss, sell_price, atr = 100.0, 95.0, 110.0, 5.0
    trail_config = swingtrade.TradingConfig(**{
        **SCALING_CONFIG.to_dict(), "trailing_stop_enabled": True, "trailing_stop_atr_multiplier": 1.5,
    })
    bars = _bars([
        (104, 112, 103, 111),  # touches target intraday -> trailing starts, trailing_stop=112-7.5=104.5
        (109, 110, 100, 101),  # low=100 breaches trailing_stop=104.5 intraday
    ])

    illiquid = settle_trade_with_trailing(
        buy_price, stop_loss, sell_price, atr, bars, trail_config, dollar_volume=LOW_TIER_VOLUME
    )
    liquid = settle_trade_with_trailing(
        buy_price, stop_loss, sell_price, atr, bars, trail_config, dollar_volume=FULL_LIQUIDITY_VOLUME
    )

    assert illiquid["exit_reason"] == liquid["exit_reason"] == "trailing_stop_hit"
    assert illiquid["exit_price"] < liquid["exit_price"]


# --- End-to-end: a real strategy's simulate function threads a real
# Dollar_Volume (AvgVolume x Last_Close, from *_levels_from_frame()) all
# the way down to the settlement call, not just that the plumbing COULD. ---

def test_ma_crossover_simulate_threads_real_dollar_volume_to_settlement(uptrend_ohlcv, market_ohlcv):
    config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "ma_crossover", "slippage_liquidity_scaling": True,
    })
    captured_dollar_volumes = []
    real_settle = swingtrade.backtest._settle

    def _spy_settle(*args, **kwargs):
        captured_dollar_volumes.append(kwargs.get("dollar_volume"))
        return real_settle(*args, **kwargs)

    with patch("swingtrade.backtest._settle", side_effect=_spy_settle):
        trades = swingtrade.simulate_ma_crossover_signals(
            "TEST", uptrend_ohlcv, market_ohlcv, uptrend_ohlcv.index[0], uptrend_ohlcv.index[-1], config,
            sector="Tech",
        )

    if not trades:
        pytest.skip("synthetic fixture produced no MA crossover this run -- not what's under test here")
    assert captured_dollar_volumes, "simulate_ma_crossover_signals should call _settle() at least once here"
    assert all(dv is not None for dv in captured_dollar_volumes), (
        "every real ma_crossover backtest trade must pass a real Dollar_Volume through to settlement "
        "once ma_crossover_levels_from_frame() computes one -- see levels.py's own Dollar_Volume field"
    )
    # Sanity: it's a real dollar figure (AvgVolume x Last_Close on a
    # multi-million-share, ~$100 synthetic uptrend), not some placeholder.
    assert all(dv > 0 for dv in captured_dollar_volumes)
