"""ratio_tracker pure functions: score_rows (frozen model), compute_label (entry/exit alignment), forward_report."""
import numpy as np, pandas as pd
import ratio_tracker as rt

MODEL = {"features": ["a", "b"], "mu": [0.0, 0.0], "sd": [1.0, 1.0], "weights": [1.0, -1.0], "threshold": 0.5, "hold_days": 5}


def test_score_rows_uses_frozen_params_and_clips_and_flags_picks():
    rows = [{"ticker": "X", "filed": "2026-02-01", "fy_end": "2025-12-31", "a": 10.0, "b": 0.0},   # z clipped to 3
            {"ticker": "Y", "filed": "2026-02-02", "fy_end": "2025-12-31", "a": 0.1, "b": float("nan")}]
    out = rt.score_rows(rows, MODEL)
    assert out[0]["score"] == 3.0 and out[0]["pick"] is True
    assert out[1]["score"] == 0.1 and out[1]["pick"] is False      # NaN feature -> 0 contribution
    assert out[0]["settled"] is False


def _px(n, start="2026-01-01", step=1.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 + np.arange(n) * step, index=idx)


def test_compute_label_entry_is_second_trading_day_after_filing_and_exit_hold_days_later():
    stock, spy = _px(40, step=2.0), _px(40, step=1.0)
    lab = rt.compute_label(stock, spy, "2026-01-05", hold_days=5)         # Monday filing
    idx = stock.index
    i0 = idx.searchsorted(pd.Timestamp("2026-01-06")) + 1                    # 1 day after filing, +1 trading day
    assert lab["entry"] == idx[i0].strftime("%Y-%m-%d") and lab["exit"] == idx[i0 + 5].strftime("%Y-%m-%d")
    assert abs(lab["excess"] - (stock.iloc[i0 + 5] / stock.iloc[i0] - spy.iloc[i0 + 5] / spy.iloc[i0])) < 1e-12


def test_compute_label_none_until_window_complete():
    assert rt.compute_label(_px(8), _px(8), "2026-01-05", hold_days=126) is None


def test_forward_report_small_sample_has_no_statistics():
    docs = [{"ticker": "A", "score": 1.0, "pick": True, "excess": 0.1}]
    assert "too few settled" in rt.forward_report(docs)
