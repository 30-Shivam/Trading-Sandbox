"""Follow-up to counterfactual_best_ideas_ic.py: that script only tested
removing regime_switcher/best_ideas_sector_rs (renormalizing the OLD,
still-buggy stored weights among survivors). This tests BOTH 2026-09-20
fixes together -- exclude the two removed methodologies AND recompute
every survivor's weight using TODAY's corrected ensemble_weight()
(neutral_prior=0.0) instead of the OLD stored (neutral_prior=1.0) weight
-- i.e. "what would the composite's real historical IC have been if it had
always used today's fully-corrected logic." A single current weight per
methodology name is applied uniformly across all historical signals (not
a true point-in-time reconstruction, which isn't possible retroactively --
this answers "how good is the NEW signal as it stands today," not "was it
always this good historically").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import best_ideas
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

# TODAY's real, current, fully-corrected weight per surviving methodology --
# the SAME computation the live pipeline will now use going forward.
today_ic_reports = {name: ic_tracking.methodology_report(name) for name in best_ideas.METHODOLOGIES}
today_weights = {name: ic_tracking.ensemble_weight(today_ic_reports.get(name, {"trust_floor_met": False, "ir": None})) for name in best_ideas.METHODOLOGIES}
print("Today's real, current per-methodology weights (both fixes applied):")
for name, w in sorted(today_weights.items(), key=lambda kv: -kv[1]):
    print(f"  {name:25s} {w:.4f}")
print()

real_pairs, cf_pairs = [], []
skipped_no_breakdown = skipped_no_survivors = 0
for o in outcomes:
    sig = signals_by_key.get((o["ticker"], o["signal_date"]))
    if sig is None or not sig.get("methodology_breakdown"):
        skipped_no_breakdown += 1
        continue
    breakdown = sig["methodology_breakdown"]
    tier = sig.get("tier") or "actionable"
    weight = ic_tracking.TIER_WEIGHTS.get(tier, ic_tracking.DEFAULT_TIER_WEIGHT)

    # Score comes from the REAL historical breakdown (what each methodology
    # actually said at the time) -- only the WEIGHT is replaced with today's
    # corrected, evidence-based value.
    survivors = {name: v["score"] for name, v in breakdown.items() if name not in REMOVED}
    survivor_weight_sum = sum(today_weights.get(name, 0.0) for name in survivors)
    if survivor_weight_sum <= 0:
        skipped_no_survivors += 1
        continue
    cf_score = sum(score * today_weights.get(name, 0.0) for name, score in survivors.items()) / survivor_weight_sum

    real_pairs.append({"score": float(sig["trade_score"]), "pnl_pct": float(o["pnl_pct"]), "tier": tier, "weight": weight, "ticker": o["ticker"], "signal_date": o["signal_date"]})
    cf_pairs.append({"score": cf_score, "pnl_pct": float(o["pnl_pct"]), "tier": tier, "weight": weight, "ticker": o["ticker"], "signal_date": o["signal_date"]})

print(f"Settled best_ideas outcomes: {len(outcomes)}, usable: {len(real_pairs)}, "
      f"skipped (no breakdown): {skipped_no_breakdown}, skipped (no surviving methodology with real weight): {skipped_no_survivors}")


def _summarize(pairs, label):
    sector_lookup = ic_tracking._get_sector_lookup()
    tier_weights = [p["weight"] for p in pairs]
    cluster_weights = ic_tracking._cluster_weights(pairs, sector_lookup)
    combined = [t * c for t, c in zip(tier_weights, cluster_weights)]
    eff_n = round(ic_tracking._effective_count(combined), 1)
    ic = ic_tracking.rank_ic([p["score"] for p in pairs], [p["pnl_pct"] for p in pairs], weights=combined)
    print(f"{label}: n={len(pairs)} effective_n={eff_n} overall_ic={ic}")
    return ic


real_ic = _summarize(real_pairs, "REAL composite (as actually logged, old buggy weights)")
cf_ic = _summarize(cf_pairs, "FULLY-CORRECTED composite (removed methodologies excluded AND today's weights)")

print(f"\nFull effect of both 2026-09-20 fixes combined, on real settled history: {real_ic} -> {cf_ic}")
