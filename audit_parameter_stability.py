"""
Parameter-stability audit -- how much does Optuna's own "best" hyperparameter
set drift from one re-optimization round to the next?

Every validation check built so far in this project's "iron proof
backtesting" effort operates on the RESULT of a completed search (DSR
corrects for selection bias across trials within ONE search; ticker-holdout
checks generalization across tickers; the real-vs-random gap checks timing
skill). NONE of them ask a different, real question this project's own
`monthly_reoptimize.yml` cron makes directly relevant: if you re-run the
IDENTICAL search process again (a new month of data, a fresh Optuna run),
does it converge on roughly the SAME "optimal" parameters, or does the
"best" trial swing wildly? A parameter that swings a lot between
re-optimization rounds is a parameter Optuna isn't actually learning a
stable signal for -- each round is essentially re-discovering noise, not
refining an estimate.

Real, previously-unexamined finding this project's own MongoDB history
already contains (2026-09-13): ma_crossover's own `System_Config` history
shows `ma_crossover_short_window` ranging from 11 to 29 (~2.6x) and
`atr_take_profit_multiplier` ranging from ~1.0 to ~4.95 (~5x) across search
rounds within the SAME six-week span -- this script is the reusable,
quantitative version of eyeballing that MongoDB query by hand.

Method: pulls every real `System_Config` document for a given strategy
(any status -- candidate, active, retired -- every one represents a REAL
completed search's own winning trial), computes each numeric param's
range/mean (a simple, interpretable instability measure: 0 = no variation,
1.0 = the range is as big as the mean itself) across every version, and
separately computes the DERIVED risk-reward-ratio's own stability (since
atr_take_profit_multiplier/stop_loss_atr_multiplier can each drift a lot
while their RATIO stays put, or vice versa -- a materially different
question from either raw multiplier's own stability).

PURE READ-ONLY measurement -- never writes to MongoDB, never promotes/
retires anything, never re-runs a search.

Usage:
    python audit_parameter_stability.py --strategy ma_crossover
    python audit_parameter_stability.py --strategy pairs
"""
import argparse
import re
import sys
from datetime import date

import storage

# Matches this project's own optimize.py-written notes format, e.g.:
# "Optuna search (strategy=ma_crossover  multi-objective): 45 trials,
#  2025-08-14..2026-08-14, 306 tune / 102 holdout ticker(s), 6 fold(s), ..."
# -- see optimize.py's own note-string construction for the source of truth.
_OPTUNA_NOTES_RE = re.compile(
    r"Optuna search \([^)]*\):\s*(\d+)\s*trials,\s*(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2}),"
    r".*?(\d+)\s*fold"
)


def _parse_search_provenance(notes: str) -> dict:
    """Extracts (n_trials, window_days, n_folds) from a real System_Config
    document's own `notes` field (2026-09-13, improvements.txt item 148) --
    the honest follow-up to item 147's own stated caveat: pooling EVERY
    historical version regardless of why it was created (a manual v1
    baseline, a single-field scoring recalibration byte-identical to its
    parent otherwise, a genuine Optuna search) inflates apparent parameter
    instability, since different search METHODOLOGIES (a 1-year vs a
    5-year rolling window, 6 vs 54 folds, 20 vs 100 trials -- all
    confirmed to have genuinely varied across this project's own real
    ma_crossover history) aren't a fair "does re-optimization converge"
    comparison to begin with.

    Returns `{"is_optuna_search": bool, "n_trials":, "window_days":,
    "n_folds":}` -- the latter three are None when `is_optuna_search` is
    False (a non-search document: an untuned baseline, a manual
    recalibration, or any other notes format this project's `optimize.py`
    didn't write)."""
    if not notes:
        return {"is_optuna_search": False, "n_trials": None, "window_days": None, "n_folds": None}
    match = _OPTUNA_NOTES_RE.search(notes)
    if not match:
        return {"is_optuna_search": False, "n_trials": None, "window_days": None, "n_folds": None}
    n_trials, start_str, end_str, n_folds = match.groups()
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    return {
        "is_optuna_search": True,
        "n_trials": int(n_trials),
        "window_days": (end - start).days,
        "n_folds": int(n_folds),
    }


def _numeric_param_series(docs: list[dict]) -> dict:
    """{param_name: [values across every doc that has it]} for every
    numeric (int/float, not bool) field under `params`."""
    series: dict = {}
    for doc in docs:
        for key, value in doc.get("params", {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            series.setdefault(key, []).append(value)
    return series


def _stability(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    lo, hi = min(values), max(values)
    range_over_mean = round((hi - lo) / abs(mean), 4) if mean != 0 else None
    return {"n": n, "mean": round(mean, 4), "min": lo, "max": hi, "range_over_mean": range_over_mean}


def _report_stability(docs: list[dict], label: str, unstable_threshold: float) -> None:
    """Prints the PARAMETER STABILITY + DERIVED RRR sections for one group
    of docs (either "all versions pooled" or one same-methodology group) --
    shared between both report modes so they stay in exact sync."""
    if len(docs) < 2:
        print(f"\n=== {label} ({len(docs)} version(s) -- need at least 2 to measure any variation, skipped) ===")
        return

    series = _numeric_param_series(docs)
    varying = {k: v for k, v in series.items() if len(set(v)) > 1 and len(v) == len(docs)}

    print(f"\n=== {label} ({len(docs)} version(s), {len(varying)} field(s) that actually varied) ===")
    unstable = []
    for name in sorted(varying):
        stats = _stability(varying[name])
        flag = ""
        if stats["range_over_mean"] is not None and stats["range_over_mean"] > unstable_threshold:
            flag = "  [UNSTABLE]"
            unstable.append(name)
        print(f"  {name}: {stats}{flag}")

    if "atr_take_profit_multiplier" in series and "stop_loss_atr_multiplier" in series and len(series["atr_take_profit_multiplier"]) == len(series["stop_loss_atr_multiplier"]) == len(docs):
        rrr_values = [
            tp / sl for tp, sl in zip(series["atr_take_profit_multiplier"], series["stop_loss_atr_multiplier"]) if sl
        ]
        if rrr_values:
            print(f"  DERIVED RRR (atr_take_profit_multiplier / stop_loss_atr_multiplier): {_stability(rrr_values)}")

    if varying:
        print(f"  -> {len(unstable)}/{len(varying)} flagged UNSTABLE (range/mean > {unstable_threshold}): "
              f"{unstable if unstable else 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, help="e.g. ma_crossover, pairs, rsi")
    parser.add_argument(
        "--unstable-threshold", type=float, default=0.5,
        help="Flag a param as UNSTABLE if range/mean exceeds this. Default 0.5 "
             "(the range is at least half the mean's own size).",
    )
    args = parser.parse_args()

    db = storage.get_db()
    docs = list(
        db[storage.system_config.COLLECTION_NAME]
        .find({"params.strategy": args.strategy}, {"version": 1, "status": 1, "created_at": 1, "params": 1, "notes": 1})
        .sort("version", 1)
    )
    if not docs:
        print(f"[ERROR] No System_Config documents found for strategy={args.strategy!r}.", file=sys.stderr)
        sys.exit(1)

    print(f"Strategy: {args.strategy} -- {len(docs)} real System_Config document(s) "
          f"(v{docs[0]['version']}..v{docs[-1]['version']}, {docs[0]['created_at'].date()}..{docs[-1]['created_at'].date()})\n")

    _report_stability(docs, "ALL VERSIONS POOLED (includes non-search/heterogeneous methodology)", args.unstable_threshold)

    # 2026-09-13 (improvements.txt item 148) -- the honest follow-up to item
    # 147's own stated caveat: pooling every version regardless of WHY it
    # was created isn't a fair "does re-optimization converge" test.
    # Real, previously-hand-noticed-but-never-quantified provenance
    # confirmed directly in this project's own notes text: a manual v1
    # baseline (no search at all), a single-field scoring recalibration
    # (byte-identical to its parent otherwise, not a real re-tune), AND
    # multiple genuinely different Optuna search METHODOLOGIES (a 1-year
    # rolling window with 6 folds vs. a 5-year window with 54 folds) all
    # got pooled together in that first pass. Grouping by (window_days,
    # n_folds) -- the two structural knobs that actually change what
    # question a search is even asking -- isolates genuine apples-to-
    # apples re-optimization rounds from deliberate methodology changes.
    for doc in docs:
        doc["_provenance"] = _parse_search_provenance(doc.get("notes", ""))

    search_docs = [d for d in docs if d["_provenance"]["is_optuna_search"]]
    non_search_docs = [d for d in docs if not d["_provenance"]["is_optuna_search"]]
    print(f"\nProvenance: {len(search_docs)}/{len(docs)} version(s) are genuine Optuna-search results "
          f"(the rest -- {len(non_search_docs)} -- are manual baselines/recalibrations/other, version(s) "
          f"{[d['version'] for d in non_search_docs]}, excluded from the grouped view below).")

    groups: dict = {}
    for doc in search_docs:
        prov = doc["_provenance"]
        key = (prov["window_days"], prov["n_folds"])
        groups.setdefault(key, []).append(doc)

    print(f"\n{len(groups)} distinct search-methodology group(s) found among genuine Optuna searches:")
    for (window_days, n_folds), group_docs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        versions = [d["version"] for d in group_docs]
        print(f"  window={window_days}d, folds={n_folds}: {len(group_docs)} version(s) {versions}")
        _report_stability(
            group_docs,
            f"SAME-METHODOLOGY GROUP (window={window_days}d, folds={n_folds})",
            args.unstable_threshold,
        )

    print("\nA parameter that keeps swinging by more than half its own mean WITHIN a same-methodology group")
    print("is a parameter Optuna isn't actually converging on even when the search question itself is held")
    print("constant -- each round may be re-discovering noise in that specific search's own data window/")
    print("trial-sampling, not refining a genuine estimate. The ALL-VERSIONS-POOLED view above mixes in")
    print("genuine methodology changes too, so treat its own instability numbers as an upper bound, not a")
    print("clean read on re-optimization convergence specifically -- the grouped view below it is the fairer")
    print("comparison. A single-version group (see above) has nothing to compare against yet.")


if __name__ == "__main__":
    main()
