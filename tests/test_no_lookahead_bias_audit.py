"""audit_no_lookahead() -- a direct, empirical look-ahead-bias check
(2026-09-13), built per the user's "operate in a continuous loop...
fully comprehensive of all variables" directive. This project's no-
look-ahead discipline has, until now, relied entirely on manual code
review (every precompute_*_frame()'s own docstring states its intent,
never tested directly). This proves the audit tool itself has real
detection power (deliberately-broken synthetic precompute functions that
DO leak future data are confirmed caught, not just "everything happens to
pass"), then applies it to two real production functions
(precompute_rsi_frame, precompute_ma_crossover_frame) as first real
regression coverage.
"""
import numpy as np
import pandas as pd

import swingtrade
from swingtrade.levels import audit_no_lookahead, precompute_ma_crossover_frame, precompute_rsi_frame

CONFIG = swingtrade.DEFAULT_CONFIG


def _sample_dates(df: pd.DataFrame, n: int = 8) -> list:
    # Skip the first ~250 rows (SMA_TREND/ADX/etc. warmup) so "insufficient
    # history" doesn't dominate the sample -- this audit is about
    # LOOK-AHEAD, not about re-litigating warmup-length behavior.
    eligible = df.index[250:]
    step = max(1, len(eligible) // n)
    return list(eligible[::step][:n])


def test_audit_passes_clean_precompute_rsi_frame(uptrend_ohlcv):
    dates = _sample_dates(uptrend_ohlcv)
    result = audit_no_lookahead(precompute_rsi_frame, uptrend_ohlcv, dates, CONFIG)
    assert result["dates_checked"] == len(dates)
    assert result["mismatches"] == []


def test_audit_passes_clean_precompute_ma_crossover_frame(uptrend_ohlcv, market_ohlcv, sector_ohlcv):
    dates = _sample_dates(uptrend_ohlcv)
    result = audit_no_lookahead(
        precompute_ma_crossover_frame, uptrend_ohlcv, dates, CONFIG,
        extra_series={"market_df": market_ohlcv, "sector_df": sector_ohlcv},
    )
    assert result["dates_checked"] == len(dates)
    assert result["mismatches"] == []


def test_audit_a_real_pandas_ta_quirk_is_not_a_lookahead_leak(uptrend_ohlcv):
    # A REAL finding from building this audit: pandas_ta.rsi() needs the
    # TOTAL input array length to reach config.rsi_window before it
    # computes anything at all (an absolute-length gate, distinct from a
    # per-row trailing-window check) -- a date within the first
    # rsi_window rows genuinely DOES differ between the full (420-row) and
    # truncated (very short) frame: not because of a look-ahead leak, but
    # because the truncated array is simply too short for pandas_ta to
    # attempt RSI at all yet, while the full array (which has plenty of
    # rows by construction) already cleared that gate. Once the truncated
    # array reaches config.rsi_window + 1 total rows, RSI matches the full
    # frame EXACTLY -- proving the computation itself is causal, this is a
    # warmup-length artifact, not a leak. See audit_no_lookahead()'s own
    # docstring caveat about this.
    window = CONFIG.rsi_window
    too_early = uptrend_ohlcv.index[window - 3]
    just_enough = uptrend_ohlcv.index[window]

    early_result = audit_no_lookahead(precompute_rsi_frame, uptrend_ohlcv, [too_early], CONFIG)
    assert any(m[1] == "RSI" for m in early_result["mismatches"])

    late_result = audit_no_lookahead(precompute_rsi_frame, uptrend_ohlcv, [just_enough], CONFIG)
    assert late_result["mismatches"] == []


def _leaky_precompute(df: pd.DataFrame, config) -> pd.DataFrame:
    """Deliberately BROKEN precompute function -- 'Leaky_Close' at day D is
    tomorrow's Close (df["Close"].shift(-1)), a textbook look-ahead leak.
    Exists purely to prove audit_no_lookahead() actually catches a REAL
    leak, not just that clean functions happen to pass."""
    frame = df.copy()
    frame["Leaky_Close"] = frame["Close"].shift(-1)
    frame["Honest_SMA"] = frame["Close"].rolling(20).mean()  # a real, non-leaky column for contrast
    return frame


def test_audit_catches_a_deliberate_lookahead_leak(uptrend_ohlcv):
    dates = _sample_dates(uptrend_ohlcv)
    result = audit_no_lookahead(_leaky_precompute, uptrend_ohlcv, dates, CONFIG)
    leaked_cols = {m[1] for m in result["mismatches"]}
    assert "Leaky_Close" in leaked_cols, "audit failed to catch a deliberate shift(-1) look-ahead leak"
    assert "Honest_SMA" not in leaked_cols, "audit flagged a genuinely non-leaky column as a false positive"


def _leaky_via_extra_series(df: pd.DataFrame, config, market_df: pd.DataFrame) -> pd.DataFrame:
    """Deliberately BROKEN precompute function that leaks through an EXTRA
    series (market_df), not `df` itself -- proves extra_series is actually
    truncated independently, not just passed through unchanged."""
    frame = df.copy()
    # Today's own row paired with tomorrow's market close -- a leak that
    # would be invisible if extra_series weren't independently truncated.
    frame["Leaky_Market_Close"] = market_df["Close"].shift(-1).reindex(frame.index)
    return frame


def test_audit_catches_a_leak_through_an_extra_series(uptrend_ohlcv, market_ohlcv):
    dates = _sample_dates(uptrend_ohlcv)
    result = audit_no_lookahead(
        _leaky_via_extra_series, uptrend_ohlcv, dates, CONFIG, extra_series={"market_df": market_ohlcv},
    )
    leaked_cols = {m[1] for m in result["mismatches"]}
    assert "Leaky_Market_Close" in leaked_cols


def test_audit_ignores_columns_only_present_in_one_frame(uptrend_ohlcv):
    # A truncated frame legitimately missing a trailing-window-derived
    # column near its own shorter history's tail end is NOT a look-ahead
    # bug -- only columns present in BOTH frames are compared.
    def _sometimes_extra_column(df: pd.DataFrame, config) -> pd.DataFrame:
        frame = df.copy()
        if len(df) > 300:
            frame["Only_When_Long_Enough"] = 1.0
        return frame

    dates = _sample_dates(uptrend_ohlcv)
    result = audit_no_lookahead(_sometimes_extra_column, uptrend_ohlcv, dates, CONFIG)
    assert result["mismatches"] == []
