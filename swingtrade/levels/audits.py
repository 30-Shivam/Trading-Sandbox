"""Look-ahead and data-quality audits, OHLCV sanitizing."""

import numpy as np
import pandas as pd
from ..config import DEFAULT_CONFIG, TradingConfig


def audit_no_lookahead(
    precompute_fn,
    df: pd.DataFrame,
    as_of_dates,
    config: TradingConfig = DEFAULT_CONFIG,
    extra_series: dict | None = None,
) -> dict:
    """Direct, empirical look-ahead-bias check for any `precompute_*_frame()`
    function (2026-09-13, per the user's "make the backtest fully
    comprehensive of all variables" directive) -- this project's no-look-
    ahead discipline has, until now, relied entirely on manual code review
    (every precompute function's own docstring states its intent, but
    nothing ever TESTED it directly). The test: a row's own indicator
    values at `as_of` must be IDENTICAL whether computed from the full
    history or from history truncated right after `as_of` -- a genuine
    look-ahead leak (some rolling/shift computation reaching past `as_of`)
    would make truncating the future change that row's own values, since a
    correct point-in-time computation for day `as_of` never needs to know
    what happens after it.

    `precompute_fn` is called as `precompute_fn(truncated_df, config=config,
    **truncated_extras)` -- every `precompute_*_frame()` function in this
    module accepts `config` as a keyword and any additional date-indexed
    inputs (market_df/sector_df/peer_prices/rank_column/etc.) as further
    keywords, so a single generic harness covers all of them without
    needing to know each one's internal logic.

    `extra_series` (optional) maps kwarg name -> date-indexed pd.Series/
    pd.DataFrame that `precompute_fn` also takes (e.g. `market_df=...`) --
    these are independently truncated at each `as_of` too, since a market/
    sector/peer series can leak future information just as easily as `df`
    itself. A non-date-indexed value (e.g. `rule` for
    precompute_llm_strategy_frame, a plain dict) is passed through
    unchanged, never truncated -- caller's responsibility to know which of
    its own extra kwargs are actually time series.

    Only compares columns present in BOTH the full and truncated frames
    (a truncated frame can legitimately be missing trailing-window-derived
    columns near the very end of its own shorter history -- that's
    insufficient-history, not look-ahead, and is excluded from the
    comparison rather than flagged). `as_of` values not present in either
    frame's index are skipped (can't compare what wasn't computed).

    Returns `{"dates_checked": int, "mismatches": [(as_of, column,
    full_value, truncated_value), ...]}` -- an EMPTY `mismatches` list is
    the passing case. A non-empty one is a genuine, previously undetected
    look-ahead bug: the exact column and date pinpoint where to look.

    KNOWN CAVEAT (found while building this): some indicator libraries
    (pandas_ta's `ta.rsi()` confirmed) gate on the TOTAL length of the
    array passed in, not just how much trailing history precedes a given
    row -- a `truncated_df` shorter than roughly that indicator's own
    window will get NaN across the board even at rows where a real
    backtest would never actually be this early (every real caller already
    keeps a long leading buffer -- see backtest.py's LOOKBACK_BUFFER_BARS).
    This shows up as an apparent "mismatch" for `as_of` dates very close to
    the start of `df`, but is a warmup-length artifact, NOT a look-ahead
    leak -- confirmed by checking that values match exactly again as soon
    as the truncated array clears that same absolute-length threshold (see
    tests/test_no_lookahead_bias_audit.py's own dedicated test for this).
    Sample `as_of` dates from well past every relevant window's own length
    for a meaningful audit, same as any real backtest already does."""
    extra_series = extra_series or {}
    full_frame = precompute_fn(df, config=config, **extra_series)

    mismatches = []
    dates_checked = 0
    for as_of in as_of_dates:
        if as_of not in full_frame.index:
            continue
        truncated_df = df.loc[:as_of]
        truncated_extras = {}
        for name, series in extra_series.items():
            if isinstance(series, (pd.Series, pd.DataFrame)):
                truncated_extras[name] = series.loc[:as_of]
            else:
                truncated_extras[name] = series
        truncated_frame = precompute_fn(truncated_df, config=config, **truncated_extras)
        if as_of not in truncated_frame.index:
            continue
        dates_checked += 1

        full_row = full_frame.loc[as_of]
        truncated_row = truncated_frame.loc[as_of]
        shared_cols = [c for c in full_frame.columns if c in truncated_frame.columns]
        for col in shared_cols:
            full_value, truncated_value = full_row[col], truncated_row[col]
            if pd.isna(full_value) and pd.isna(truncated_value):
                continue
            if isinstance(full_value, (int, float)) and isinstance(truncated_value, (int, float)):
                if not np.isclose(float(full_value), float(truncated_value), rtol=1e-9, atol=1e-9):
                    mismatches.append((as_of, col, full_value, truncated_value))
            elif full_value != truncated_value:
                mismatches.append((as_of, col, full_value, truncated_value))

    return {"dates_checked": dates_checked, "mismatches": mismatches}


def audit_data_quality(df: pd.DataFrame, max_daily_move_pct: float = 50.0) -> dict:
    """Real, previously entirely unexamined data-integrity check (2026-09-13,
    per the user's "fully comprehensive of all variables" directive) --
    this project has never once validated that fetched OHLCV data is
    internally consistent before feeding it into a backtest. A free vendor
    (yfinance) occasionally returns a bad tick (a stale/erroneous row, a
    vendor-side data glitch) that could silently corrupt a signal or
    settlement decision with nothing to ever catch it -- this is that
    catch.

    Two DIFFERENT categories, reported separately because they mean very
    different things:

    STRUCTURAL VIOLATIONS -- should be ZERO for genuinely real data,
    regardless of how volatile the ticker is; any nonzero count here is a
    real, actionable data bug, not a market event:
      - Low > High (physically impossible for a real trading session)
      - Open or Close outside the [Low, High] range for that same row
      - Any of Open/High/Low/Close <= 0 (a real security never trades at
        zero or negative)
      - Volume < 0

    LARGE-MOVE FLAGS -- informational only, NOT necessarily bugs: a
    single-day |Close-to-Close % change| exceeding `max_daily_move_pct`
    (default 50%) could be a genuine data glitch (this project has no way
    to be sure from price data alone), but could equally be a REAL
    extreme event this codebase's own no-look-ahead discipline must still
    respect (an earnings-driven crash/spike, an M&A announcement, a halt-
    and-reopen) -- flagged for human review, never auto-excluded, since
    silently dropping a real day would itself be a form of survivorship-
    bias-adjacent look-ahead (deciding after the fact that a real outcome
    "must have been fake").

    Returns `{"n_rows":, "n_structural_violations":, "structural_violation_dates":
    (up to 10, for review), "n_large_moves":, "large_move_dates": [(date,
    pct_change), ...] (up to 10, sorted by |move| descending), "is_clean":
    bool (True iff n_structural_violations == 0 -- large moves alone don't
    fail this, they're just flagged)}`. Returns a degenerate all-zero
    result for an empty/too-short `df` (nothing to check)."""
    if df is None or len(df) < 2:
        return {
            "n_rows": 0 if df is None else len(df), "n_structural_violations": 0,
            "structural_violation_dates": [], "n_large_moves": 0, "large_move_dates": [], "is_clean": True,
        }

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    structural_mask = (
        (l > h) | (o > h) | (o < l) | (c > h) | (c < l)
        | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    if "Volume" in df.columns:
        structural_mask = structural_mask | (df["Volume"] < 0)
    structural_dates = list(df.index[structural_mask])

    daily_move_pct = c.pct_change() * 100
    large_move_mask = daily_move_pct.abs() > max_daily_move_pct
    large_moves = sorted(
        ((d, round(float(daily_move_pct.loc[d]), 2)) for d in df.index[large_move_mask.fillna(False)]),
        key=lambda item: abs(item[1]), reverse=True,
    )

    return {
        "n_rows": len(df),
        "n_structural_violations": len(structural_dates),
        "structural_violation_dates": structural_dates[:10],
        "n_large_moves": len(large_moves),
        "large_move_dates": large_moves[:10],
        "is_clean": len(structural_dates) == 0,
    }


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Repairs the STRUCTURAL VIOLATIONS audit_data_quality() detects
    (Open/Close outside [Low, High], Low > High, non-positive prices)
    instead of merely flagging them -- built 2026-09-17 after the SAME
    2021-05-05 structural violation recurred across FOUR unrelated
    tickers spanning BOTH this project's ticker universes (HUBB in the
    primary watchlist -- the original item-144 finding -- and HLX/UA/WLY
    in the small/mid-cap one), strong evidence of a systemic one-day
    yfinance/Yahoo feed anomaly rather than four independent ticker-
    specific glitches. A vendor data quirk this narrow doesn't justify
    hunting down and hand-patching individual tickers -- it justifies a
    general safeguard, since ANY strategy's ATR/stop-loss/entry-fill math
    silently trusts whatever OHLC values it's handed, live or backtested.

    Minimal-assumption repair, never fabricates a new price level:
      - Open (or Close) outside [Low, High]: clipped to the nearest of
        the two bounds -- the reported Low/High are trusted as-is, only
        the internally-inconsistent Open/Close gets corrected to the
        closest boundary it violates. Confirmed against all 5 known real
        violations (HUBB/HLX x2/UA/WLY) -- every one is exactly this
        shape (Open below Low), and clipping resolves each cleanly.
      - Low > High (the bounds themselves contradict each other, a
        deeper corruption than Open/Close alone): the two are SWAPPED,
        not discarded -- the working theory is a vendor-side column
        mislabeling, not fabricated numbers, so preserving both real
        values while fixing their assigned roles is the smallest
        intervention that resolves the inconsistency. Applied BEFORE the
        Open/Close clipping above so a subsequent Open/Close check
        measures against the corrected bounds.
      - Any of Open/High/Low/Close <= 0: no bound exists to safely clip
        to (a real security never legitimately trades at zero or
        negative) -- these rows are DROPPED entirely rather than
        guessed at, same "don't fabricate data" discipline as everywhere
        else in this codebase's no-look-ahead work.

    Returns a REPAIRED COPY -- never mutates `df` in place. A `df` with
    zero structural violations (the overwhelming majority of real data)
    passes through with values unchanged (only float rounding-safe
    comparisons, no unconditional rewrite). Call audit_data_quality() on
    the RESULT if you want to confirm zero violations remain (repairing
    Low/High can, in principle, still leave an already-adjusted Open/Close
    row that needs a second look -- covered by this function's own test
    suite, not just asserted here)."""
    if df is None or df.empty:
        return df

    df = df.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    non_positive = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    if "Volume" in df.columns:
        non_positive = non_positive | (df["Volume"] < 0)
    df = df.loc[~non_positive]
    if df.empty:
        return df
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    swapped = l > h
    if swapped.any():
        new_low = h.where(swapped, l)
        new_high = l.where(swapped, h)
        df["Low"], df["High"] = new_low, new_high
        h, l = df["High"], df["Low"]

    df["Open"] = o.clip(lower=l, upper=h)
    df["Close"] = c.clip(lower=l, upper=h)
    return df
