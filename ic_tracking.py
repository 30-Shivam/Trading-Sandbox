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

Tier-weighted (2026-08-21, per explicit user request): every signal also
carries a "tier" -- "actionable" (real Strong Buy/Buy, real
capital-eligible) vs "research"/"research_loosened" (Watch-only, logged
purely to accumulate outcome data faster, never real capital -- see
storage/signals.py's own tier docstring). A real actionable entry should
count more toward a methodology's demonstrated track record than a
Watch-tier one. Concrete motivation: RSI mean-reversion's first live day
fired 38 highly correlated signals in one broad tech-sector
pullback+recovery, 31 of them research-tier -- unweighted, that single
event could push a methodology past the trust floor on the strength of
its least capital-relevant signals. See TIER_WEIGHTS/_effective_count().

Cluster-weighted (2026-08-23): tier-weighting alone didn't fully close the
gap above -- rsi_mean_reversion's 39 settled signals STILL all fall on
that SAME single calendar day, so tier-weighting alone still read ~35
effective trades (past the 20-trade floor) from what's really one
correlated event, not 20+ independent trials. Now also multiplies in a
same-day/same-sector cluster weight (reusing
swingtrade.compute_cluster_weights(), already used the same way on
backtest trades by optimize.py/reoptimize_best_ideas.py) -- see
_cluster_weights()/methodology_report()'s own docstring for the combined
math. `effective_n_settled` now means tier x cluster combined;
`tier_only_effective_n_settled` keeps the old tier-only number visible so
the size of this correction isn't hidden.
"""

from pathlib import Path

import pandas as pd

import swingtrade
import watchlist
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

# Per-tier weight for pooling Trade_Signals/Trade_Outcomes into IC/trust-floor
# math (2026-08-21, per explicit user request) -- "actionable" (real Strong
# Buy/Buy, real capital-eligible) signals should count more toward a
# methodology's demonstrated track record than "research" (Watch-only,
# logged purely to accumulate data faster, see storage/signals.py's own
# tier docstring) or "research_loosened" (a DIFFERENT, deliberately loosened
# config entirely) signals. Missing tier -> 1.0 (actionable), matching
# storage/signals.py's own "missing tier field means actionable" convention
# for pre-existing documents. Concrete motivation: RSI mean-reversion's
# first live day (2026-08-20) fired 38 correlated signals in one broad
# tech-sector pullback+recovery, 31 of them research-tier -- without
# weighting, that single lucky/unlucky day could push a methodology's raw
# n_settled past TRUST_FLOOR_TRADES on the strength of its least
# capital-relevant signals.
TIER_WEIGHTS = {
    "actionable": 1.0,
    "research": 0.5,
    "research_loosened": 0.25,
}
DEFAULT_TIER_WEIGHT = 1.0


def _effective_count(weights) -> float:
    """Kish's effective sample size -- same formula
    swingtrade.backtest.summarize_trades_weighted() already uses for the
    identical reason: a large pile of low-weight (research-tier) trades
    shouldn't be able to fake a trustworthy sample size the way a raw count
    would. effective_n = (sum(w))^2 / sum(w^2) -- equals len(weights) when
    every weight is equal, and shrinks toward the count of only the
    highest-weight trades as the pool skews toward low-weight ones. Returns
    0.0 for an empty or non-positive-total weight list."""
    if not weights:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 0.0
    sq_sum = sum(w * w for w in weights)
    if sq_sum <= 0:
        return 0.0
    return total * total / sq_sum


def _weighted_corr(x: pd.Series, y: pd.Series, w: pd.Series) -> float | None:
    """Weighted Pearson correlation -- used here on RANK sequences (see
    rank_ic()'s `weights` param), so this is a weighted Spearman
    correlation in effect. Ranks themselves stay plain/positional
    (unweighted); only each pair's CONTRIBUTION to the correlation
    strength is weighted -- the same "weight the aggregate, not each
    trade's own resolved outcome" pattern
    swingtrade.backtest.summarize_trades_weighted() already follows.
    Returns None if either weighted variance is non-positive (degenerate,
    same convention as the unweighted case's nunique() < 2 check)."""
    total = float(w.sum())
    if total <= 0:
        return None
    mx = float((w * x).sum() / total)
    my = float((w * y).sum() / total)
    var_x = float((w * (x - mx) ** 2).sum() / total)
    var_y = float((w * (y - my) ** 2).sum() / total)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = float((w * (x - mx) * (y - my)).sum() / total)
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def rank_ic(scores, pnls, weights=None) -> float | None:
    """Spearman rank correlation between `scores` (a methodology's own
    score at signal time) and `pnls` (realized forward pnl_pct at
    settlement). Returns None (never 0.0 -- "no measurable skill" and "not
    enough data to tell" are different claims that callers should not
    conflate) if fewer than MIN_TRADES_FOR_ANY_IC pairs, the two sequences
    have mismatched lengths, or either series is constant (a rank
    correlation against a constant series is undefined, not zero).

    `weights` (optional, e.g. TIER_WEIGHTS per pair): when given, computes
    a WEIGHTED correlation between the (unweighted) rank sequences instead
    of the plain unweighted one -- see _weighted_corr(). None/omitted
    reproduces the original unweighted behavior exactly, so every existing
    caller (e.g. reoptimize_sector_rs.py's own backtest-driven IC, which
    has no tier concept at all) is unaffected."""
    if len(scores) != len(pnls) or len(scores) < MIN_TRADES_FOR_ANY_IC:
        return None
    if weights is not None and len(weights) != len(scores):
        return None
    s = pd.Series(list(scores), dtype=float)
    p = pd.Series(list(pnls), dtype=float)
    if s.nunique() < 2 or p.nunique() < 2:
        return None
    rs, rp = s.rank(), p.rank()
    if weights is None:
        ic = rs.corr(rp)
        return float(ic) if pd.notna(ic) else None
    w = pd.Series(list(weights), dtype=float)
    return _weighted_corr(rs, rp, w)


def windowed_ic_series(records: list[dict], window_days: int = IC_WINDOW_DAYS) -> list[dict]:
    """`records`: [{"signal_date": "YYYY-MM-DD", "score": float,
    "pnl_pct": float}, ...] (order doesn't matter) -- an optional "weight"
    key (see TIER_WEIGHTS) makes this a tier-weighted window: the window's
    own IC is computed via rank_ic(..., weights=...), and the
    MIN_TRADES_PER_WINDOW floor is checked against the window's EFFECTIVE
    count (_effective_count()), not its raw record count, so a window
    padded mostly with low-weight research-tier signals can't clear the
    floor on volume alone. Records without a "weight" key reproduce the
    original unweighted behavior exactly. Buckets into successive
    `window_days`-wide calendar windows spanning the earliest to latest
    signal_date -- NOT a strict per-day IC, see module docstring for why.
    Returns [{"window_start": date, "window_end": date, "ic": float,
    "n": int, "effective_n": float}, ...] in chronological order -- empty
    list if there's nothing to bucket or every window is too thin."""
    if not records:
        return []
    df = pd.DataFrame(records)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    has_weights = "weight" in df.columns
    start = df["signal_date"].min()
    end = df["signal_date"].max()
    windows = []
    window_start = start
    while window_start <= end:
        window_end = window_start + pd.Timedelta(days=window_days)
        chunk = df[(df["signal_date"] >= window_start) & (df["signal_date"] < window_end)]
        weights = chunk["weight"].tolist() if has_weights else None
        effective_n = _effective_count(weights) if weights is not None else float(len(chunk))
        if effective_n >= MIN_TRADES_PER_WINDOW:
            ic = rank_ic(chunk["score"].tolist(), chunk["pnl_pct"].tolist(), weights=weights)
            if ic is not None:
                windows.append({
                    "window_start": window_start.date(), "window_end": window_end.date(),
                    "ic": ic, "n": len(chunk), "effective_n": round(effective_n, 1),
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
    [{"signal_date":, "score":, "pnl_pct":, "tier":, "weight":, "ticker":},
    ...] for every settled trade under this strategy where BOTH the
    originating signal's score and its pnl_pct are present -- silently
    skips a settled outcome whose source signal is missing the score field
    (shouldn't happen for any current methodology, all of which always
    write trade_score, but this is a read path and should degrade rather
    than crash if it ever does).

    `tier`/`weight` (2026-08-21): the originating signal's own "actionable"
    | "research" | "research_loosened" tier (see storage/signals.py) and
    its TIER_WEIGHTS lookup -- missing tier defaults to "actionable"/1.0,
    matching storage/signals.py's own convention for pre-existing
    documents written before the tier field existed.

    `ticker` (2026-08-23): needed by methodology_report()'s cluster-weight
    correction (see _cluster_weights()) to look up each pair's sector --
    purely additive; `weight` here still means TIER_WEIGHTS alone (the
    combined tier x cluster weight is computed one layer up, in
    methodology_report(), not here).

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
        (doc["ticker"], doc["signal_date"]): (doc.get(score_field), doc.get("tier"))
        for doc in db["Trade_Signals"].find(
            {"strategy": strategy}, {"ticker": 1, "signal_date": 1, score_field: 1, "tier": 1}
        )
    }
    pairs = []
    for outcome in outcomes:
        signal = signals_by_key.get((outcome["ticker"], outcome["signal_date"]))
        if signal is None or signal[0] is None:
            continue
        score, tier = signal
        tier = tier or "actionable"
        pairs.append({
            "signal_date": outcome["signal_date"], "score": float(score), "pnl_pct": float(outcome["pnl_pct"]),
            "tier": tier, "weight": TIER_WEIGHTS.get(tier, DEFAULT_TIER_WEIGHT), "ticker": outcome["ticker"],
        })
    return pairs


_SECTOR_LOOKUP_CACHE: dict[str, str] | None = None


def _get_sector_lookup() -> dict[str, str]:
    """Lazily loads and process-lifetime-caches watchlist.txt's
    {ticker: sector} mapping (see watchlist.read_ticker_sectors(), already
    used the same way by best_ideas.py/optimize.py/reoptimize_best_ideas.py)
    -- avoids re-parsing a 400+-entry file on every methodology_report()
    call (ingest.py calls this once per methodology per run, ~7-8 times)."""
    global _SECTOR_LOOKUP_CACHE
    if _SECTOR_LOOKUP_CACHE is None:
        _SECTOR_LOOKUP_CACHE = watchlist.read_ticker_sectors(Path("watchlist.txt"))
    return _SECTOR_LOOKUP_CACHE


def _cluster_weights(pairs: list[dict], sector_lookup: dict[str, str]) -> list[float]:
    """Per-pair weight correcting for same-day, same-sector correlation --
    thin wrapper around swingtrade.compute_cluster_weights() (already used
    by optimize.py/reoptimize_best_ideas.py for the identical correction on
    backtest trades; see that function's own docstring), adapted to this
    module's record shape (signal_date + ticker, rather than an
    already-resolved entry_date/sector pair). `sector_lookup` is passed
    in explicitly (not read internally) so this stays a pure,
    directly-testable function -- see _get_sector_lookup() for the real,
    cached lookup methodology_report() actually uses. A pair whose ticker
    has no sector entry gets "Unknown", which compute_cluster_weights()
    treats as weight 1.0 (no correlation assumed) -- same convention its
    own docstring already establishes."""
    trades = [{"entry_date": p["signal_date"], "sector": sector_lookup.get(p["ticker"], "Unknown")} for p in pairs]
    return swingtrade.compute_cluster_weights(trades)


def methodology_report(strategy: str, score_field: str = "trade_score", window_days: int = IC_WINDOW_DAYS) -> dict:
    """The full IC/IR picture for one methodology (`strategy` label) --
    everything ensemble_weight() and the dashboard need in one call.
    `overall_ic` pools EVERY settled trade into one TIER-WEIGHTED rank
    correlation (the headline "does this methodology's score rank real
    outcomes correctly, at all" figure -- see rank_ic()'s `weights` param
    and TIER_WEIGHTS); `ic_series`/`ir` measure whether that skill is
    STABLE over time (see windowed_ic_series()/information_ratio()) -- a
    methodology can show a decent overall_ic built from one lucky window
    with an ir near zero (or None), which is exactly the distinction IR
    exists to catch, per the user's own explicit reasoning for wanting it.

    Returns {"strategy":, "n_settled":, "effective_n_settled": float,
    "tier_only_effective_n_settled": float, "overall_ic": float|None,
    "ic_series": [...], "ir": float|None, "trust_floor_met": bool} --
    `n_settled` is the raw count (unchanged meaning: "how many signals
    have resolved"); `trust_floor_met` is gated on `effective_n_settled`
    >= TRUST_FLOOR_TRADES.

    `effective_n_settled` (2026-08-23: now TIER x CLUSTER weighted, not
    tier alone) -- pure tier-weighting (per explicit user request, real
    Strong Buy/Buy "actionable" signals should count more than Watch-only
    "research" ones) left one real gap: a pile of same-day, same-sector
    CORRELATED signals still counted as that many independent data points.
    Concrete case that motivated this: rsi_mean_reversion's first 39
    settled signals all fired the SAME calendar day (one broad tech
    pullback) -- tier-weighting alone still read this as ~35 effective
    trades, comfortably past the floor, when it's really closer to one
    event. Now combines TIER_WEIGHTS with _cluster_weights() (same-day/
    same-sector correction, reusing swingtrade.compute_cluster_weights()
    -- see that function's own docstring) by multiplying the two per pair,
    the same "combine independent weight sources by multiplying them"
    pattern optimize.py already uses for cluster x recency weights.
    `tier_only_effective_n_settled` keeps the OLD (tier-only) number
    visible for comparison, so the size of the correlation correction
    isn't hidden. See ensemble_weight() for how trust_floor_met/ir
    together gate a methodology's blend weight."""
    pairs = fetch_score_pnl_pairs(strategy, score_field=score_field)
    n_settled = len(pairs)
    tier_weights = [p["weight"] for p in pairs]
    tier_only_effective_n_settled = round(_effective_count(tier_weights), 1) if pairs else 0.0
    cluster_weights = _cluster_weights(pairs, _get_sector_lookup()) if pairs else []
    combined_weights = [t * c for t, c in zip(tier_weights, cluster_weights)]
    combined_pairs = [{**p, "weight": w} for p, w in zip(pairs, combined_weights)]
    effective_n_settled = round(_effective_count(combined_weights), 1) if pairs else 0.0
    overall_ic = (
        rank_ic([p["score"] for p in pairs], [p["pnl_pct"] for p in pairs], weights=combined_weights)
        if pairs else None
    )
    ic_series = windowed_ic_series(combined_pairs, window_days=window_days)
    ir = information_ratio([w["ic"] for w in ic_series])
    return {
        "strategy": strategy, "n_settled": n_settled, "effective_n_settled": effective_n_settled,
        "tier_only_effective_n_settled": tier_only_effective_n_settled,
        "overall_ic": overall_ic, "ic_series": ic_series, "ir": ir,
        "trust_floor_met": effective_n_settled >= TRUST_FLOOR_TRADES,
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
