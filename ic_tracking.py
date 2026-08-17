"""Information Coefficient / Information Ratio tracking for every "Best
Ideas" methodology (see best_ideas.py) -- rank correlation between a
methodology's OWN score at signal time and its realized forward pnl_pct at
settlement, pooled over a rolling window rather than computed strictly
per-day. This project's own daily signal breadth is too thin for a
meaningful per-day rank correlation (often 0-2 signals/day per
methodology) -- pooling every settled trade whose signal fell within a
window (default 30 calendar days) into one rank correlation is the
honest way to get a usable sample size without waiting months.

Why IC/IR instead of sharpe_like/win_rate for comparing methodologies
against each other (per explicit user request): sharpe_like/win_rate are
confounded by payoff STRUCTURE -- a tight-target/wide-stop bracket
mechanically inflates win_rate independent of real entry-timing skill.
This project's own memory (see [[strategy-validation-pipeline]] point 9)
already caught this for real: breakout_retest/week52_high's original
"validated" win rates turned out to be substantially a payoff-geometry
artifact, not proof of good entries. IC only asks whether a methodology's
own score RANKS tickers in the same order their real forward returns
eventually rank -- true regardless of what stop/target bracket realized
that pnl_pct. Every Best Ideas methodology deliberately shares ONE ATR
bracket (see best_ideas.py) specifically so this isn't even a live
confound here, but IC/IR is the right lens for comparing methodologies
against each other either way, exactly as requested.

Reuses Trade_Signals/Trade_Outcomes exactly as already populated by the
existing settle_trades.py pipeline (swingtrade.settle_trade() is fully
generic -- no strategy-specific branching, confirmed via
swingtrade/settlement.py) -- no new settlement machinery needed, just a
new READ path over the two existing collections.

No new dependency: rank correlation is computed via pandas' own
Series.rank() + Pearson correlation on the ranks, not scipy.stats.spearmanr
(scipy is not currently a requirement of this project).
"""

import pandas as pd

from storage.mongo import get_db

MIN_TRADES_FOR_ANY_IC = 3   # fewer than this and a rank correlation is not
                             # meaningfully defined, regardless of pooling
MIN_TRADES_PER_WINDOW = 5   # a window's own IC is skipped below this, even
                             # pooled -- still too thin to trust
IC_WINDOW_DAYS = 30         # pooling window width, see module docstring
TRUST_FLOOR_TRADES = 20     # matches llm_agent.py's own established
                             # prospective trust floor (20-30 settled
                             # trades, see dip_buy_analyzer.py's "Validation
                             # progress" sections) -- below this, a
                             # methodology's IC/IR is still reported but
                             # does NOT drive its blend weight, see
                             # ensemble_weight()


def rank_ic(scores, pnls) -> float | None:
    """Spearman rank correlation between `scores` (a methodology's own
    score at signal time) and `pnls` (realized forward pnl_pct at
    settlement). Returns None (never 0.0 -- "no measurable skill" and "not
    enough data to tell" are different claims that callers should not
    conflate) if fewer than MIN_TRADES_FOR_ANY_IC pairs, the two sequences
    have mismatched lengths, or either series is constant (a rank
    correlation against a constant series is undefined, not zero)."""
    if len(scores) != len(pnls) or len(scores) < MIN_TRADES_FOR_ANY_IC:
        return None
    s = pd.Series(list(scores), dtype=float)
    p = pd.Series(list(pnls), dtype=float)
    if s.nunique() < 2 or p.nunique() < 2:
        return None
    ic = s.rank().corr(p.rank())
    return float(ic) if pd.notna(ic) else None


def windowed_ic_series(records: list[dict], window_days: int = IC_WINDOW_DAYS) -> list[dict]:
    """`records`: [{"signal_date": "YYYY-MM-DD", "score": float,
    "pnl_pct": float}, ...] (order doesn't matter). Buckets into
    successive `window_days`-wide calendar windows spanning the earliest to
    latest signal_date -- NOT a strict per-day IC, see module docstring for
    why -- computes one pooled rank_ic() per window, and SKIPS any window
    with fewer than MIN_TRADES_PER_WINDOW records (still too thin to trust
    even pooled). Returns
    [{"window_start": date, "window_end": date, "ic": float, "n": int}, ...]
    in chronological order -- empty list if there's nothing to bucket or
    every window is too thin."""
    if not records:
        return []
    df = pd.DataFrame(records)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    start = df["signal_date"].min()
    end = df["signal_date"].max()
    windows = []
    window_start = start
    while window_start <= end:
        window_end = window_start + pd.Timedelta(days=window_days)
        chunk = df[(df["signal_date"] >= window_start) & (df["signal_date"] < window_end)]
        if len(chunk) >= MIN_TRADES_PER_WINDOW:
            ic = rank_ic(chunk["score"].tolist(), chunk["pnl_pct"].tolist())
            if ic is not None:
                windows.append({
                    "window_start": window_start.date(), "window_end": window_end.date(),
                    "ic": ic, "n": len(chunk),
                })
        window_start = window_end
    return windows


def information_ratio(ic_values: list[float]) -> float | None:
    """mean(ic) / std(ic) across windows -- IR, not IC itself: measures how
    STABLE a methodology's ranking skill is across time, not just whether
    it showed up once. Returns None if fewer than 2 windows (sample std is
    undefined with 1 point) or std is exactly 0 (would divide by zero --
    a literally-constant IC across every window is a degenerate case worth
    flagging as None, not an infinite or fabricated IR)."""
    if len(ic_values) < 2:
        return None
    s = pd.Series(ic_values, dtype=float)
    std = s.std(ddof=1)
    # A near-zero (not necessarily bit-exact-zero) std -- floating-point
    # representation of a literally-constant input series (e.g. three
    # copies of 0.2) can leave a tiny nonzero residual rather than exactly
    # 0.0, which would otherwise produce a wildly fabricated IR instead of
    # the intended "degenerate, report None" outcome.
    if pd.isna(std) or std < 1e-9:
        return None
    return float(s.mean() / std)


def fetch_score_pnl_pairs(strategy: str, score_field: str = "trade_score") -> list[dict]:
    """Join Trade_Signals (has `score_field` at signal time) with
    Trade_Outcomes (has pnl_pct at settlement) for one `strategy` label, on
    (ticker, signal_date) -- both collections already carry `strategy` as
    part of their own unique index (see storage/signals.py,
    storage/outcomes.py), so this is a simple two-query-and-merge in
    Python, not a fragile cross-collection guess. Returns
    [{"signal_date":, "score":, "pnl_pct":}, ...] for every settled trade
    under this strategy where BOTH the originating signal's score and its
    pnl_pct are present -- silently skips a settled outcome whose source
    signal is missing the score field (shouldn't happen for any current
    methodology, all of which always write trade_score, but this is a read
    path and should degrade rather than crash if it ever does).

    `score_field` lets a future methodology's "score" be read from a
    different Trade_Signals field if trade_score isn't the right one for
    it -- every CURRENT methodology's own score IS trade_score, including
    llm_agent's own confidence (written directly into that field, see
    dip_buy_analyzer.py/ingest.py's llm_rows construction)."""
    db = get_db()
    outcomes = list(
        db["Trade_Outcomes"].find({"strategy": strategy}, {"ticker": 1, "signal_date": 1, "pnl_pct": 1})
    )
    if not outcomes:
        return []
    signals_by_key = {
        (doc["ticker"], doc["signal_date"]): doc.get(score_field)
        for doc in db["Trade_Signals"].find({"strategy": strategy}, {"ticker": 1, "signal_date": 1, score_field: 1})
    }
    pairs = []
    for outcome in outcomes:
        score = signals_by_key.get((outcome["ticker"], outcome["signal_date"]))
        if score is None:
            continue
        pairs.append({
            "signal_date": outcome["signal_date"], "score": float(score), "pnl_pct": float(outcome["pnl_pct"]),
        })
    return pairs


def methodology_report(strategy: str, score_field: str = "trade_score", window_days: int = IC_WINDOW_DAYS) -> dict:
    """The full IC/IR picture for one methodology (`strategy` label) --
    everything ensemble_weight() and the dashboard need in one call.
    `overall_ic` pools EVERY settled trade into one rank correlation (the
    headline "does this methodology's score rank real outcomes correctly,
    at all" figure); `ic_series`/`ir` measure whether that skill is STABLE
    over time (see windowed_ic_series()/information_ratio()) -- a
    methodology can show a decent overall_ic built from one lucky window
    with an ir near zero (or None), which is exactly the distinction IR
    exists to catch, per the user's own explicit reasoning for wanting it.

    Returns {"strategy":, "n_settled":, "overall_ic": float|None,
    "ic_series": [...], "ir": float|None, "trust_floor_met": bool} --
    trust_floor_met is n_settled >= TRUST_FLOOR_TRADES, the same bar
    llm_agent.py's own tab already established; see ensemble_weight() for
    how this gates whether ir actually drives a blend weight."""
    pairs = fetch_score_pnl_pairs(strategy, score_field=score_field)
    n_settled = len(pairs)
    overall_ic = rank_ic([p["score"] for p in pairs], [p["pnl_pct"] for p in pairs]) if pairs else None
    ic_series = windowed_ic_series(pairs, window_days=window_days)
    ir = information_ratio([w["ic"] for w in ic_series])
    return {
        "strategy": strategy, "n_settled": n_settled, "overall_ic": overall_ic,
        "ic_series": ic_series, "ir": ir, "trust_floor_met": n_settled >= TRUST_FLOOR_TRADES,
    }


def ensemble_weight(report: dict, neutral_prior: float = 1.0) -> float:
    """The blend weight one methodology contributes to the Best Ideas
    composite score (see best_ideas.blend_composite()) -- `neutral_prior`
    (equal weight, same for every methodology) until BOTH
    `report["trust_floor_met"]` is True AND `report["ir"]` is a real
    number; after that, the methodology's OWN demonstrated ranking
    stability -- `max(report["ir"], 0.0)` -- takes over. A negative IR
    methodology is EXCLUDED from the blend (weight 0), not inverted into a
    contrarian signal: this early, a negative IR is much more likely to be
    noise than a genuine anti-signal worth trusting in reverse. This
    function is the literal mechanism behind using IC/IR, not
    sharpe_like/win_rate, to recursively compare/blend methodologies
    against each other."""
    if report.get("trust_floor_met") and report.get("ir") is not None:
        return max(report["ir"], 0.0)
    return neutral_prior
