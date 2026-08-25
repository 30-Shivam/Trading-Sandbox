"""settle_trades.settle_one()'s entry-fill gate for UNCONFIRMED signals
(2026-08-24) -- a real bug found via a live user report: RSI mean-reversion
(and any discount-entry strategy) buys at a resting limit below the market
(MA_Discount_Price), but settle_one() used to hand that buy_price straight
to swingtrade.settle_trade() with no check that the market ever actually
touched it, unlike the backtester (which has always gated through
swingtrade.find_entry_fill() first). A Watch-tier signal whose buy_price
sat far below the real market was graded as filled on day one regardless,
manufacturing a fictional win the moment the real (untouched, much higher)
price was compared against it -- confirmed live: research-tier outcomes
were sitting at a literal 100% win rate (142/142) the day this was found.

No Mongo, no yfinance -- fetch_bars_since()/storage.log_trade_outcome()/
storage.mark_settled() are all monkeypatched, same dependency-injection-free
convention as test_settle_trades_notifications.py.
"""
import pandas as pd

import settle_trades
import storage
import swingtrade


def _bars(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {"Open":, "High":, "Low":, "Close":} dicts, oldest
    first -- index values don't matter beyond being strictly increasing."""
    return pd.DataFrame(rows, index=pd.date_range("2026-08-22", periods=len(rows), freq="D"))


def _signal(buy_price=50.0, stop_loss=40.0, sell_price=60.0, confirmed=False, fill_price=None):
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "max_entry_wait_days": 5})
    sig = {
        "ticker": "TEST",
        "signal_date": "2026-08-21",
        "buy_price": buy_price,
        "stop_loss": stop_loss,
        "sell_price": sell_price,
        "config_snapshot": config.to_dict(),
        "confirmed_filled": confirmed,
        "tier": "research",
    }
    if fill_price is not None:
        sig["fill_price"] = fill_price
    return sig


def _patch_storage(monkeypatch):
    logged, settled = [], []
    monkeypatch.setattr(storage, "log_trade_outcome", lambda *a, **kw: logged.append((a, kw)))
    monkeypatch.setattr(storage, "mark_settled", lambda *a, **kw: settled.append((a, kw)))
    return logged, settled


def test_unconfirmed_never_touched_within_window_settles_with_no_outcome(monkeypatch):
    # buy_price=50, but the market never dips anywhere near it across 5
    # bars (>= max_entry_wait_days) -- the resting limit was never
    # realistically touched, so this must NOT manufacture a fictional win.
    bars = _bars([
        {"Open": 90, "High": 92, "Low": 88, "Close": 91},
        {"Open": 91, "High": 93, "Low": 89, "Close": 92},
        {"Open": 92, "High": 94, "Low": 90, "Close": 93},
        {"Open": 93, "High": 95, "Low": 91, "Close": 94},
        {"Open": 94, "High": 96, "Low": 92, "Close": 95},
    ])
    monkeypatch.setattr(settle_trades, "fetch_bars_since", lambda ticker, signal_date: bars)
    logged, settled = _patch_storage(monkeypatch)

    outcome, notify = settle_trades.settle_one(_signal(buy_price=50.0))

    assert outcome.startswith("NEVER_FILLED")
    assert notify is None
    assert len(settled) == 1        # marked settled -- stop re-checking forever
    assert len(logged) == 0         # but NO Trade_Outcomes document -- no fictional win


def test_unconfirmed_still_within_wait_window_leaves_it_unsettled(monkeypatch):
    # Only 3 bars available so far, less than max_entry_wait_days=5, and
    # none have touched buy_price yet -- too early to conclude "never will",
    # so this must be left alone (re-checked on a later run), not resolved.
    bars = _bars([
        {"Open": 90, "High": 92, "Low": 88, "Close": 91},
        {"Open": 91, "High": 93, "Low": 89, "Close": 92},
        {"Open": 92, "High": 94, "Low": 90, "Close": 93},
    ])
    monkeypatch.setattr(settle_trades, "fetch_bars_since", lambda ticker, signal_date: bars)
    logged, settled = _patch_storage(monkeypatch)

    outcome, notify = settle_trades.settle_one(_signal(buy_price=50.0))

    assert outcome == "waiting for entry fill"
    assert notify is None
    assert len(settled) == 0
    assert len(logged) == 0


def test_unconfirmed_filled_via_gap_down_open_grades_from_real_fill_price(monkeypatch):
    # Day 1-2 never touch buy_price=50. Day 3 gaps down through it
    # (Open=48 <= 50) -- find_entry_fill() fills at that better, real Open,
    # not at the original fictional buy_price. Day 4 (the only bar strictly
    # after the real fill) gaps up through sell_price=60 -- a WIN, but it
    # must be graded from the REAL $48 entry, one holding day after the
    # actual fill, not from signal_date with a fictional $50 entry.
    bars = _bars([
        {"Open": 90, "High": 92, "Low": 88, "Close": 91},   # day 1 -- no touch
        {"Open": 89, "High": 91, "Low": 87, "Close": 90},   # day 2 -- no touch
        {"Open": 48, "High": 49, "Low": 46, "Close": 47},   # day 3 -- gap-down fill at 48
        {"Open": 62, "High": 64, "Low": 61, "Close": 63},   # day 4 -- gap-up target hit
    ])
    monkeypatch.setattr(settle_trades, "fetch_bars_since", lambda ticker, signal_date: bars)
    logged, settled = _patch_storage(monkeypatch)

    outcome, notify = settle_trades.settle_one(_signal(buy_price=50.0, stop_loss=40.0, sell_price=60.0))

    assert outcome.startswith("WIN")
    assert len(logged) == 1
    args, kwargs = logged[0]
    ticker, signal_date, strategy, entry_price, result = args
    assert entry_price == 48.0                      # the REAL fill price, not the fictional buy_price=50
    assert result["holding_days"] == 1               # 1 day after the REAL fill (day 4), not from signal_date
    assert result["status"] == "WIN"
    assert kwargs["confirmed_filled"] is False


def test_confirmed_fill_skips_the_gate_entirely_unchanged_behavior(monkeypatch):
    # A CONFIRMED real trade (the user actually filled it) must be graded
    # exactly as before -- unconditionally trusted from day one, no
    # find_entry_fill() check at all, even though the market never touches
    # the plain buy_price here (only the explicit fill_price matters).
    bars = _bars([
        {"Open": 62, "High": 64, "Low": 61, "Close": 63},   # day 1 -- gap-up target hit immediately
    ])
    monkeypatch.setattr(settle_trades, "fetch_bars_since", lambda ticker, signal_date: bars)
    logged, settled = _patch_storage(monkeypatch)

    outcome, notify = settle_trades.settle_one(
        _signal(buy_price=50.0, stop_loss=40.0, sell_price=60.0, confirmed=True, fill_price=52.0)
    )

    assert outcome.startswith("WIN")
    assert len(logged) == 1
    args, kwargs = logged[0]
    entry_price = args[3]
    assert entry_price == 52.0            # the real confirmed fill_price, used unconditionally
    assert args[4]["holding_days"] == 1   # graded from day one, exactly as before
    assert kwargs["confirmed_filled"] is True
