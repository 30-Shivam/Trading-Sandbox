"""audit_data_quality() -- a real, previously entirely unexamined
data-integrity check (2026-09-13). This project has never once validated
that fetched OHLCV data is internally consistent before feeding it into a
backtest. Structural violations (Low > High, negative prices, etc.) should
be ZERO for genuinely real data regardless of volatility; large single-day
moves are flagged for review but never auto-excluded, since a real crash/
spike is a legitimate event this codebase's own no-look-ahead discipline
must still respect.
"""
import numpy as np
import pandas as pd

from swingtrade.levels import audit_data_quality, sanitize_ohlcv


def _ohlcv(rows: list[dict]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=dates)


def test_empty_or_too_short_df_is_clean_by_default():
    assert audit_data_quality(None)["is_clean"] is True
    assert audit_data_quality(pd.DataFrame())["is_clean"] is True


def test_clean_data_reports_zero_violations_and_zero_large_moves():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 101, "High": 103, "Low": 100, "Close": 102, "Volume": 1_100_000},
        {"Open": 102, "High": 104, "Low": 101, "Close": 103, "Volume": 1_050_000},
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 0
    assert result["n_large_moves"] == 0
    assert result["is_clean"] is True


def test_low_greater_than_high_is_a_structural_violation():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 100, "High": 95, "Low": 98, "Close": 96, "Volume": 1_000_000},  # Low > High
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 1
    assert result["is_clean"] is False


def test_close_outside_high_low_range_is_a_structural_violation():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 100, "High": 102, "Low": 99, "Close": 110, "Volume": 1_000_000},  # Close > High
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 1


def test_zero_or_negative_price_is_a_structural_violation():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0},  # a real vendor stub row
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 1
    assert result["is_clean"] is False


def test_negative_volume_is_a_structural_violation():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": -5},
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 1


def test_large_move_is_flagged_but_does_not_fail_is_clean():
    # A real, legitimate 60% single-day crash (e.g. a real earnings-driven
    # collapse) -- flagged for review, but NOT a structural violation, and
    # does NOT make is_clean False (never auto-excluded).
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 100, "Volume": 1_000_000},
        {"Open": 100, "High": 100, "Low": 38, "Close": 40, "Volume": 5_000_000},  # -60%
    ]
    result = audit_data_quality(_ohlcv(rows))
    assert result["n_structural_violations"] == 0
    assert result["is_clean"] is True
    assert result["n_large_moves"] == 1
    assert result["large_move_dates"][0][1] == -60.0


def test_large_move_threshold_is_configurable():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 100, "Volume": 1_000_000},
        {"Open": 100, "High": 112, "Low": 99, "Close": 110, "Volume": 1_000_000},  # +10%
    ]
    result_default = audit_data_quality(_ohlcv(rows))  # default 50% threshold
    assert result_default["n_large_moves"] == 0
    result_tight = audit_data_quality(_ohlcv(rows), max_daily_move_pct=5.0)
    assert result_tight["n_large_moves"] == 1


def test_large_moves_sorted_by_magnitude_descending():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 100, "Volume": 1_000_000},
        {"Open": 100, "High": 165, "Low": 99, "Close": 160, "Volume": 1_000_000},  # +60%
        {"Open": 160, "High": 161, "Low": 60, "Close": 60, "Volume": 1_000_000},   # -62.5%
    ]
    result = audit_data_quality(_ohlcv(rows), max_daily_move_pct=50.0)
    assert result["n_large_moves"] == 2
    # Second move (-62.5%) has a bigger absolute magnitude -> sorted first.
    assert abs(result["large_move_dates"][0][1]) > abs(result["large_move_dates"][1][1])


def test_nan_close_does_not_crash_pct_change():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 100, "Volume": 1_000_000},
        {"Open": np.nan, "High": np.nan, "Low": np.nan, "Close": np.nan, "Volume": 0},
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
    ]
    result = audit_data_quality(_ohlcv(rows))
    # Should not crash; NaN comparisons are simply False, not flagged as violations.
    assert isinstance(result["n_structural_violations"], int)


# --- sanitize_ohlcv() -- 2026-09-17, the repair counterpart to the audit
# above. Built after the SAME 2021-05-05 structural violation recurred
# across 4 unrelated tickers in both this project's ticker universes
# (HUBB -- the original item-144 finding -- plus HLX/UA/WLY), strong
# evidence of a systemic one-day vendor feed anomaly worth a general
# safeguard rather than four hand-patched tickers.

def test_sanitize_ohlcv_clean_data_passes_through_unchanged():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": 101, "High": 103, "Low": 100, "Close": 102, "Volume": 1_100_000},
    ]
    df = _ohlcv(rows)
    result = sanitize_ohlcv(df)
    pd.testing.assert_frame_equal(result, df)


def test_sanitize_ohlcv_none_and_empty_pass_through():
    assert sanitize_ohlcv(None) is None
    empty = pd.DataFrame()
    assert sanitize_ohlcv(empty).empty


def test_sanitize_ohlcv_clips_open_below_low_matching_real_hubb_violation():
    # Real values from HUBB 2021-05-05, the original item-144 finding.
    rows = [{"Open": 195.98, "High": 198.64, "Low": 196.21, "Close": 198.13, "Volume": 500_000}]
    result = sanitize_ohlcv(_ohlcv(rows))
    assert result.iloc[0]["Open"] == 196.21  # clipped up to Low
    assert result.iloc[0]["Close"] == 198.13  # already within bounds, untouched
    assert audit_data_quality(result)["is_clean"] is True


def test_sanitize_ohlcv_clips_open_above_high_matching_real_hlx_violation():
    # Real values from HLX 2023-06-05.
    rows = [{"Open": 7.11, "High": 7.07, "Low": 6.83, "Close": 7.01, "Volume": 300_000}]
    result = sanitize_ohlcv(_ohlcv(rows))
    assert result.iloc[0]["Open"] == 7.07  # clipped down to High
    assert audit_data_quality(result)["is_clean"] is True


def test_sanitize_ohlcv_clips_close_outside_bounds():
    rows = [
        {"Open": 100, "High": 105, "Low": 99, "Close": 110, "Volume": 1_000_000},  # Close > High
        {"Open": 100, "High": 105, "Low": 99, "Close": 90, "Volume": 1_000_000},   # Close < Low
    ]
    result = sanitize_ohlcv(_ohlcv(rows))
    assert result.iloc[0]["Close"] == 105
    assert result.iloc[1]["Close"] == 99
    assert audit_data_quality(result)["is_clean"] is True


def test_sanitize_ohlcv_swaps_low_and_high_when_inverted():
    # Low > High -- deeper corruption than Open/Close alone. Real values
    # preserved, just their Low/High roles corrected (not discarded).
    rows = [{"Open": 100, "High": 95, "Low": 105, "Close": 99, "Volume": 1_000_000}]
    result = sanitize_ohlcv(_ohlcv(rows))
    row = result.iloc[0]
    assert row["Low"] == 95
    assert row["High"] == 105
    # Open(100)/Close(99) both now legitimately fall within the corrected [95, 105] bounds.
    assert row["Open"] == 100
    assert row["Close"] == 99
    assert audit_data_quality(result)["is_clean"] is True


def test_sanitize_ohlcv_drops_non_positive_rows_rather_than_guessing():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
        {"Open": -5, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},  # impossible negative price
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1_000_000},
    ]
    df = _ohlcv(rows)
    result = sanitize_ohlcv(df)
    assert len(result) == 2  # the bad row is dropped, not repaired
    assert -5 not in result["Open"].values


def test_sanitize_ohlcv_drops_negative_volume_row():
    rows = [
        {"Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": -100},
    ]
    result = sanitize_ohlcv(_ohlcv(rows))
    assert result.empty


def test_sanitize_ohlcv_all_rows_dropped_returns_empty_not_error():
    rows = [{"Open": -1, "High": -1, "Low": -1, "Close": -1, "Volume": 0}]
    result = sanitize_ohlcv(_ohlcv(rows))
    assert result.empty


def test_sanitize_ohlcv_does_not_mutate_input():
    rows = [{"Open": 195.98, "High": 198.64, "Low": 196.21, "Close": 198.13, "Volume": 500_000}]
    df = _ohlcv(rows)
    original_open = df.iloc[0]["Open"]
    sanitize_ohlcv(df)
    assert df.iloc[0]["Open"] == original_open  # caller's own frame untouched
