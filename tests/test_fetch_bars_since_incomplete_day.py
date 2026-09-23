"""settle_trades.fetch_bars_since() dropping an in-progress trading day
(2026-09-22, real bug found auditing why 39/83 -- 47% -- of EXPIRED
Trade_Outcomes carried a NaN pnl_pct): yfinance returns TODAY's row with
real Open/High/Low/Volume but a NaN Close while the market is still
mid-session. settle_trade()'s EXPIRED branch (reached when a trade
survives to config.max_holding_days without hitting its stop/target) used
the LAST walked bar's Close as the exit price unconditionally, so an
in-progress day silently produced a NaN pnl_pct instead of being excluded
as not-yet-resolved.

yf.download is monkeypatched -- no real network call, same
dependency-injection-free convention as test_settle_one_entry_fill_gate.py.
"""
import pandas as pd

import settle_trades


def test_trailing_nan_close_row_is_dropped(monkeypatch):
    idx = pd.date_range("2026-09-18", periods=4, freq="D")
    df = pd.DataFrame({
        "Open": [78.86, 79.26, 80.19, 80.19],
        "High": [79.08, 81.21, 83.37, 83.37],
        "Low": [77.08, 78.88, 79.51, 79.51],
        "Close": [78.37, 80.72, float("nan"), float("nan")],
        "Adj Close": [78.25, 80.60, float("nan"), float("nan")],
        "Volume": [8539200, 11652200, 13027344, 13027344],
    }, index=idx)
    monkeypatch.setattr(settle_trades.yf, "download", lambda *a, **kw: df)

    bars = settle_trades.fetch_bars_since("APH", "2026-09-17")

    assert len(bars) == 2                     # both NaN-Close rows dropped
    assert bars["Close"].notna().all()
    assert bars.index[-1] == idx[1]           # last real (closed) bar kept


def test_no_nan_rows_is_unaffected(monkeypatch):
    idx = pd.date_range("2026-09-18", periods=2, freq="D")
    df = pd.DataFrame({
        "Open": [78.86, 79.26], "High": [79.08, 81.21], "Low": [77.08, 78.88],
        "Close": [78.37, 80.72], "Volume": [8539200, 11652200],
    }, index=idx)
    monkeypatch.setattr(settle_trades.yf, "download", lambda *a, **kw: df)

    bars = settle_trades.fetch_bars_since("APH", "2026-09-17")

    assert len(bars) == 2
