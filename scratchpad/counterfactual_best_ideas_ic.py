"""One-off SCOPING analysis, NOT a permanent feature.

Per the "fix the ensemble" direction (best_ideas' own composite cleared its
trust floor with overall_ic=-0.197): before assuming the negative-IC
regime_switcher/best_ideas_sector_rs removal (already applied to
best_ideas.METHODOLOGIES, not yet committed) actually fixes the composite,
recompute what the composite score WOULD have been, for every real settled
best_ideas signal, using ONLY the surviving methodology set -- reusing each
signal's own stored `methodology_breakdown` (per-methodology score + blend
weight, exactly what best_ideas.blend_composite() used at signal time), then
feed that counterfactual score series through the SAME tier+cluster-weighted
rank-IC math ic_tracking.methodology_report() uses, for a genuinely
apples-to-apples comparison against the real logged -0.197.

Renormalization trick: blend_composite() stores each contributor's weight
ALREADY NORMALIZED to sum to 1 across that signal's own contributors. Since
normalized_i / normalized_j == raw_i / raw_j regardless of who else
contributed, the correct renormalized weight over a SUBSET of survivors is
just (stored survivor weight) / (sum of stored survivor weights) -- no need
to reconstruct the original raw (pre-normalization) ensemble_weight() values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ic_tracking
import storage

REMOVED = {"regime_switcher", "best_ideas_sector_rs"}

db = storage.get_db()
outcomes = list(db["Trade_Outcomes"].find({"strategy": "best_ideas"}, {"ticker": 1, "signal_date": 1, "pnl_pct": 1}))
signals_by_key = {
    (doc["ticker"], doc["signal_date"]): doc
    for doc in db["Trade_Signals"].find(
        {"strategy": "best_ideas"}, {"ticker": 1, "signal_date": 1, "trade_score": 1, "tier": 1, "methodology_breakdown": 1}
    )
}

real_pairs, cf_pairs = [], []
skipped_no_breakdown = 0
skipped_no_survivors = 0
for o in outcomes:
    sig = signals_by_key.get((o["ticker"], o["signal_date"]))
    if sig is None or not sig.get("methodology_breakdown"):
        skipped_no_breakdown += 1
        continue
    breakdown = sig["methodology_breakdown"]
    tier = sig.get("tier") or "actionable"
    weight = ic_tracking.TIER_WEIGHTS.get(tier, ic_tracking.DEFAULT_TIER_WEIGHT)

    survivors = {name: v for name, v in breakdown.items() if name not in REMOVED}
    survivor_weight_sum = sum(v["weight"] for v in survivors.values())
    if survivor_weight_sum <= 0:
        skipped_no_survivors += 1
        continue
    cf_score = sum(v["score"] * v["weight"] for v in survivors.values()) / survivor_weight_sum

    real_pairs.append({
        "signal_date": o["signal_date"], "score": float(sig["trade_score"]), "pnl_pct": float(o["pnl_pct"]),
        "tier": tier, "weight": weight, "ticker": o["ticker"],
    })
    cf_pairs.append({
        "signal_date": o["signal_date"], "score": cf_score, "pnl_pct": float(o["pnl_pct"]),
        "tier": tier, "weight": weight, "ticker": o["ticker"],
    })

print(f"Settled best_ideas outcomes: {len(outcomes)}, usable (had methodology_breakdown): {len(real_pairs)}, "
      f"skipped (no breakdown stored): {skipped_no_breakdown}, skipped (no surviving methodology fired): {skipped_no_survivors}")


def _summarize(pairs, label):
    sector_lookup = ic_tracking._get_sector_lookup()
    tier_weights = [p["weight"] for p in pairs]
    cluster_weights = ic_tracking._cluster_weights(pairs, sector_lookup)
    combined = [t * c for t, c in zip(tier_weights, cluster_weights)]
    eff_n = round(ic_tracking._effective_count(combined), 1)
    ic = ic_tracking.rank_ic([p["score"] for p in pairs], [p["pnl_pct"] for p in pairs], weights=combined)
    print(f"{label}: n={len(pairs)} effective_n={eff_n} overall_ic={ic}")
    return ic


real_ic = _summarize(real_pairs, "REAL composite (as actually logged, incl. regime_switcher/sector_rs)")
cf_ic = _summarize(cf_pairs, "COUNTERFACTUAL composite (regime_switcher/sector_rs excluded, renormalized)")

print(f"\nSanity check -- real recomputed IC should be close to the live-reported -0.197: {real_ic}")
print(f"Would removing the two negative-IC methodologies have flipped/improved the composite's own IC? "
      f"{real_ic} -> {cf_ic}")
