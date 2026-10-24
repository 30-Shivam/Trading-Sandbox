"""Buy-and-hold index comparison and the index-gate verdict."""

import pandas as pd


def audit_vs_index(trades: list[dict], index_close: "pd.Series", cluster_days: int = 21) -> dict | None:
    """Does the strategy beat simply HOLDING THE INDEX over the same days?

    Every other check in this module compares a strategy to RANDOM ENTRIES of
    the same strategy ("does the timing add information"). None asked the
    question that decides whether a system is worth running at all: would the
    same money have done better sitting in the index during those exact
    holding windows? For each trade, the index return from entry_date to
    exit_date is the opportunity cost. Excess = trade pnl_pct - index return.

    The t-stat clusters trades into `cluster_days`-calendar-day entry blocks
    (default 21, about the max holding period): trades entered the same day
    share one market move, and trades on neighboring days have overlapping
    holding windows, so per-day clustering alone overstated confidence
    (accruals_rank showed t=4.95 per-day on 2026-10-22). Index returns carry no slippage/commission while pnl_pct
    does, which slightly FAVORS the index -- the realistic direction for a
    buy-and-hold alternative with ~zero turnover cost.

    Returns None if no trade has usable dates inside the index series."""
    import math
    import pandas as pd
    close = index_close.dropna().sort_index()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index).normalize()

    def at(day):
        day = pd.Timestamp(day).normalize()
        pos = close.index.searchsorted(day, side="right") - 1
        return float(close.iloc[pos]) if pos >= 0 else None

    rows = []
    for t in trades:
        if "entry_date" not in t or "exit_date" not in t or t.get("pnl_pct") is None:
            continue
        p0, p1 = at(t["entry_date"]), at(t["exit_date"])
        if not p0 or not p1:
            continue
        rows.append((pd.Timestamp(t["entry_date"]), float(t["pnl_pct"]), (p1 / p0 - 1) * 100))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["entry", "pnl", "index"])
    df["excess"] = df["pnl"] - df["index"]
    start = df["entry"].min()
    block = ((df["entry"] - start).dt.days // max(int(cluster_days), 1))
    daily = df.groupby(block)["excess"].mean()  # one observation per entry block
    n_days = int(df["entry"].nunique())
    n_blocks = len(daily)
    t_stat = None
    if n_blocks > 1 and daily.std(ddof=1) > 0:
        t_stat = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(n_blocks)))
    return {
        "n_trades": int(len(df)),
        "n_entry_days": n_days,
        "n_entry_blocks": int(n_blocks),
        "mean_trade_pnl_pct": round(float(df["pnl"].mean()), 3),
        "mean_index_pnl_pct_same_windows": round(float(df["index"].mean()), 3),
        "mean_excess_pct": round(float(df["excess"].mean()), 3),
        "pct_trades_beating_index": round(float((df["excess"] > 0).mean() * 100), 2),
        "excess_t_stat_clustered_by_entry_day": round(t_stat, 2) if t_stat is not None else None,
    }


INDEX_GATE_MIN_EXCESS_PCT = 0.0


INDEX_GATE_MIN_T = 1.65   # one-sided ~5%; trades are clustered by entry day (see audit_vs_index)


def index_gate_verdict(vs_index: dict | None, min_excess_pct: float = INDEX_GATE_MIN_EXCESS_PCT,
                       min_t: float = INDEX_GATE_MIN_T) -> dict:
    """The capital gate: a strategy must beat simply holding the index over its
    own holding windows, with statistical support, before it earns capital or a
    paper-trading slot. Beating RANDOM entries is necessary but not sufficient.
    Takes audit_vs_index()'s output; returns {"passed": bool, "reason": str}."""
    if not vs_index:
        return {"passed": False, "reason": "no trades with usable dates"}
    excess, t = vs_index.get("mean_excess_pct"), vs_index.get("excess_t_stat_clustered_by_entry_day")
    if excess is None or t is None:
        return {"passed": False, "reason": "excess or t-stat unavailable"}
    if excess <= min_excess_pct:
        return {"passed": False, "reason": f"mean excess {excess:+.3f}% <= {min_excess_pct}% (index did as well or better)"}
    if t < min_t:
        return {"passed": False, "reason": f"excess {excess:+.3f}% but t={t:.2f} < {min_t} (not distinguishable from noise)"}
    return {"passed": True, "reason": f"excess {excess:+.3f}%, t={t:.2f}"}
