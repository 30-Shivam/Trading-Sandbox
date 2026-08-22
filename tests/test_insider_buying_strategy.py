"""Insider-buying strategy (12th signal family, per an external document's
idea, scoped and adapted for this project's own validation discipline --
see improvements.txt) -- buys a ticker when real, open-market insider Form-4
purchases cluster within a recent window, in a confirmed macro uptrend.
"""
import numpy as np
import pandas as pd

import swingtrade
from swingtrade.levels import classify_insider_transaction, precompute_insider_buying_frame

CONFIG = swingtrade.DEFAULT_CONFIG


def _uptrend_ohlcv(n: int, start_close: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_close * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    high = close + rng.uniform(0.1, 0.3, n)
    low = close - rng.uniform(0.1, 0.3, n)
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def _purchases(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date_str, value, insider), ...] -- builds the same
    tz-aware-UTC "effective_date"/"value"/"insider" shape
    run_backtest.fetch_insider_purchases() returns."""
    return pd.DataFrame({
        "effective_date": pd.to_datetime([r[0] for r in rows], utc=True),
        "value": [r[1] for r in rows],
        "insider": [r[2] for r in rows],
    })


# --- classify_insider_transaction(): hand-checked against the real text
# strings observed in the feasibility probe behind this strategy's scoping.

def test_classify_insider_transaction_real_purchase_strings():
    assert classify_insider_transaction("Purchase at price 95.00 per share.") == "purchase"
    assert classify_insider_transaction("Purchase at price 26.34 per share.") == "purchase"


def test_classify_insider_transaction_real_non_purchase_strings():
    assert classify_insider_transaction("Sale at price 307.75 per share.") == "other"
    assert classify_insider_transaction("Stock Gift at price 0.00 per share.") == "other"
    assert classify_insider_transaction("Sale at price 284.57 - 285.04 per share.") == "other"


def test_classify_insider_transaction_blank_or_missing_text():
    assert classify_insider_transaction("") == "other"
    assert classify_insider_transaction(None) == "other"
    assert classify_insider_transaction(float("nan")) == "other"


def test_classify_insider_transaction_case_insensitive():
    assert classify_insider_transaction("PURCHASE AT PRICE 10.00 PER SHARE.") == "purchase"


# --- precompute_insider_buying_frame(): synthetic OHLCV + synthetic
# purchase events, hand-verifying the lookback-window/distinct-buyer/value
# gating produces the right values on the right days.

def test_precompute_insider_buying_frame_absent_without_purchases():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    frame = precompute_insider_buying_frame(df, None, CONFIG)
    assert (frame["Insider_Purchase_Value"] == 0.0).all()
    assert (frame["Insider_Distinct_Buyers"] == 0).all()


def test_precompute_insider_buying_frame_value_window_hand_verified():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    event_date = df.index[10]
    purchases = _purchases([(event_date.strftime("%Y-%m-%d"), 100_000.0, "ALICE")])
    config = swingtrade.TradingConfig(**{**CONFIG.to_dict(), "insider_lookback_days": 14})

    frame = precompute_insider_buying_frame(df, purchases, config)

    # Before the event: no value yet.
    assert frame["Insider_Purchase_Value"].iloc[9] == 0.0
    # On the event day itself, and through day 10+14=24 (inclusive): counted.
    assert frame["Insider_Purchase_Value"].iloc[10] == 100_000.0
    assert frame["Insider_Purchase_Value"].iloc[24] == 100_000.0
    # The day after the window closes: back to zero.
    assert frame["Insider_Purchase_Value"].iloc[25] == 0.0


def test_precompute_insider_buying_frame_distinct_buyers_not_double_counted():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    event_date = df.index[10]
    # Same insider buying twice within the window -- still 1 distinct buyer,
    # but value sums both.
    purchases = _purchases([
        (event_date.strftime("%Y-%m-%d"), 50_000.0, "ALICE"),
        (df.index[12].strftime("%Y-%m-%d"), 30_000.0, "ALICE"),
    ])
    frame = precompute_insider_buying_frame(df, purchases, CONFIG)

    assert frame["Insider_Distinct_Buyers"].iloc[13] == 1
    assert frame["Insider_Purchase_Value"].iloc[13] == 80_000.0


def test_precompute_insider_buying_frame_distinct_buyers_counts_different_insiders():
    n = 40
    df = _uptrend_ohlcv(n, 100.0, seed=1)
    purchases = _purchases([
        (df.index[10].strftime("%Y-%m-%d"), 50_000.0, "ALICE"),
        (df.index[11].strftime("%Y-%m-%d"), 50_000.0, "BOB"),
        (df.index[12].strftime("%Y-%m-%d"), 50_000.0, "CAROL"),
    ])
    frame = precompute_insider_buying_frame(df, purchases, CONFIG)

    assert frame["Insider_Distinct_Buyers"].iloc[13] == 3
    assert frame["Insider_Purchase_Value"].iloc[13] == 150_000.0


# --- add_insider_buying_trade_score(): mirrors test_pairs_strategy.py's
# own add_pairs_trade_score tests exactly.

def _insider_row(signal, signal_strength_pct=1.0):
    return {"Insider_Buy_Signal": signal, "RRR": 2.0, "Signal_Strength_Pct": signal_strength_pct}


def test_add_insider_buying_trade_score_ineligible_when_no_signal():
    df = pd.DataFrame([_insider_row(False)])
    result = swingtrade.add_insider_buying_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == 0.0
    assert result.loc[0, "Signal"] == "Ignore"


def test_add_insider_buying_trade_score_eligible_scores_above_zero():
    df = pd.DataFrame([_insider_row(True, signal_strength_pct=1.0)])
    result = swingtrade.add_insider_buying_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] > 0.0


def test_add_insider_buying_trade_score_strength_saturates_at_cap():
    df = pd.DataFrame([
        _insider_row(True, signal_strength_pct=CONFIG.insider_strength_cap_buyers),
        _insider_row(True, signal_strength_pct=CONFIG.insider_strength_cap_buyers * 10),
    ])
    result = swingtrade.add_insider_buying_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] == result.loc[1, "Trade_Score"]


def test_add_insider_buying_trade_score_best_case_clears_buy_threshold():
    # Same RRR-vs-scoring-ceiling check validation-pipeline point 9
    # mandates for every strategy before it's trusted -- a config could
    # otherwise be structurally unable to ever log a real Buy signal.
    df = pd.DataFrame([_insider_row(True, signal_strength_pct=CONFIG.insider_strength_cap_buyers)])
    result = swingtrade.add_insider_buying_trade_score(df, CONFIG)
    assert result.loc[0, "Trade_Score"] >= CONFIG.signal_buy_threshold
    assert result.loc[0, "Signal"] in ("Buy", "Strong Buy")
