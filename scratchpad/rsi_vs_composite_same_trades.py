"""Fair, same-trade-set comparison: on every best_ideas settled outcome where
rsi_mean_reversion actually contributed a score, compare three candidate
signals against the SAME real pnl_pct outcomes:
  1. REAL composite (as actually logged, old buggy weights)
  2. FULLY-CORRECTED composite (2026-09-20 fixes: removed methodologies
     excluded, today's corrected ensemble_weight() used)
  3. rsi_mean_reversion ALONE (its own score from methodology_breakdown,
     ignoring every other methodology entirely)

This answers: does blending help, hurt, or do nothing relative to just
trusting rsi_mean_reversion by itself, on identical trades/outcomes.
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

today_ic_reports = {name: ic_tracking.methodology_report(name) for name in best_ideas.METHODOLOGIES}
today_weights = {name: ic_tracking.ensemble_weight(today_ic_reports.get(name, {"trust_floor_met": False, "ir": None})) for name in best_ideas.METHODOLOGIES}

real_pairs, cf_pairs, rsi_pairs = [], [], []
skipped_no_breakdown = skipped_no_rsi = skipped_no_survivors = 0
for o in outcomes:
    sig = signals_by_key.get((o["ticker"], o["signal_date"]))
    if sig is None or not sig.get("methodology_breakdown"):
        skipped_no_breakdown += 1
        continue
    breakdown = sig["methodology_breakdown"]
    if "rsi_mean_reversion" not in breakdown:
        skipped_no_rsi += 1
        continue

    tier = sig.get("tier") or "actionable"
    weight = ic_tracking.TIER_WEIGHTS.get(tier, ic_tracking.DEFAULT_TIER_WEIGHT)

    survivors = {name: v["score"] for name, v in breakdown.items() if name not in REMOVED}
    survivor_weight_sum = sum(today_weights.get(name, 0.0) for name in survivors)
    if survivor_weight_sum <= 0:
        skipped_no_survivors += 1
        continue
    cf_score = sum(score * today_weights.get(name, 0.0) for name, score in survivors.items()) / survivor_weight_sum
    rsi_score = breakdown["rsi_mean_reversion"]["score"]

    common = {"pnl_pct": float(o["pnl_pct"]), "tier": tier, "weight": weight, "ticker": o["ticker"], "signal_date": o["signal_date"]}
    real_pairs.append({**common, "score": float(sig["trade_score"])})
    cf_pairs.append({**common, "score": cf_score})
    rsi_pairs.append({**common, "score": float(rsi_score)})

print(f"Settled best_ideas outcomes: {len(outcomes)}, usable (rsi_mean_reversion contributed): {len(real_pairs)}")
print(f"skipped (no breakdown): {skipped_no_breakdown}, skipped (no rsi in breakdown): {skipped_no_rsi}, skipped (no surviving weight): {skipped_no_survivors}\n")


def _summarize(pairs, label):
    sector_lookup = ic_tracking._get_sector_lookup()
    tier_weights = [p["weight"] for p in pairs]
    cluster_weights = ic_tracking._cluster_weights(pairs, sector_lookup)
    combined = [t * c for t, c in zip(tier_weights, cluster_weights)]
    eff_n = round(ic_tracking._effective_count(combined), 1)
    ic = ic_tracking.rank_ic([p["score"] for p in pairs], [p["pnl_pct"] for p in pairs], weights=combined)
    print(f"{label}: n={len(pairs)} effective_n={eff_n} overall_ic={ic}")
    return ic


real_ic = _summarize(real_pairs, "REAL composite (old buggy weights)      ")
cf_ic = _summarize(cf_pairs, "FULLY-CORRECTED composite (both fixes)   ")
rsi_ic = _summarize(rsi_pairs, "rsi_mean_reversion ALONE (same trades)   ")

print(f"\nSame-trade-set comparison, identical {len(real_pairs)} outcomes:")
print(f"  REAL composite:      {real_ic}")
print(f"  CORRECTED composite: {cf_ic}")
print(f"  RSI alone:           {rsi_ic}")
