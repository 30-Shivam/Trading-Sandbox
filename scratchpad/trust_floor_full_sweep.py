"""One-off analysis, per user request: how did every real strategy approach
its own trust floor (ic_tracking.TRUST_FLOOR_TRADES=20 effective trades),
and what's the current state of each -- not just a snapshot number, but the
windowed IC trajectory (ic_tracking.windowed_ic_series()) showing whether
trust-floor-clearing coincided with a stable signal or a lucky/unlucky
window."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage
import ic_tracking

db = storage.get_db()
strategies = sorted(db["Trade_Outcomes"].distinct("strategy"))

for strategy in strategies:
    r = ic_tracking.methodology_report(strategy)
    windows = r["ic_series"] or []
    print(f"\n=== {strategy} ===")
    print(f"  n_settled={r['n_settled']} effective_n={r['effective_n_settled']} "
          f"trust_floor_met={r['trust_floor_met']} overall_ic={r['overall_ic']} "
          f"ir={r['ir']} windows={len(windows)} (need >=4 for IR trust)")
    running_eff = 0.0
    crossed = False
    for w in windows:
        running_eff += w["effective_n"]
        flag = ""
        if not crossed and running_eff >= ic_tracking.TRUST_FLOOR_TRADES:
            flag = "  <-- cumulative effective_n crosses TRUST_FLOOR_TRADES here"
            crossed = True
        print(f"    {w['window_start']} .. {w['window_end']}: ic={w['ic']:.3f} "
              f"n={w['n']} effective_n={w['effective_n']:.1f} (cumulative~{running_eff:.1f}){flag}")
