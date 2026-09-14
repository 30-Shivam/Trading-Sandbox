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
import sys

import storage


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
        .find({"params.strategy": args.strategy}, {"version": 1, "status": 1, "created_at": 1, "params": 1})
        .sort("version", 1)
    )
    if not docs:
        print(f"[ERROR] No System_Config documents found for strategy={args.strategy!r}.", file=sys.stderr)
        sys.exit(1)

    print(f"Strategy: {args.strategy} -- {len(docs)} real System_Config document(s) "
          f"(v{docs[0]['version']}..v{docs[-1]['version']}, {docs[0]['created_at'].date()}..{docs[-1]['created_at'].date()})\n")

    series = _numeric_param_series(docs)
    # Only params that actually VARY across versions are interesting --
    # a field held fixed at the DEFAULT_CONFIG value for every version
    # was never tuned, and reporting "range_over_mean=0" for hundreds of
    # such fields would bury the real signal.
    varying = {k: v for k, v in series.items() if len(set(v)) > 1 and len(v) == len(docs)}

    print(f"=== PARAMETER STABILITY ({len(varying)} field(s) that actually varied across versions) ===")
    unstable = []
    for name in sorted(varying):
        stats = _stability(varying[name])
        flag = ""
        if stats["range_over_mean"] is not None and stats["range_over_mean"] > args.unstable_threshold:
            flag = "  [UNSTABLE]"
            unstable.append(name)
        print(f"  {name}: {stats}{flag}")

    if "atr_take_profit_multiplier" in series and "stop_loss_atr_multiplier" in series and len(series["atr_take_profit_multiplier"]) == len(series["stop_loss_atr_multiplier"]) == len(docs):
        rrr_values = [
            tp / sl for tp, sl in zip(series["atr_take_profit_multiplier"], series["stop_loss_atr_multiplier"]) if sl
        ]
        if rrr_values:
            rrr_stats = _stability(rrr_values)
            print(f"\n=== DERIVED: risk-reward ratio (atr_take_profit_multiplier / stop_loss_atr_multiplier) ===")
            print(f"  RRR: {rrr_stats}")
            print("  Compare this against the individual multipliers' own stability above -- a ratio that's "
                  "MORE stable than its own components means Optuna keeps finding different absolute risk "
                  "levels but the same relative payoff shape; a ratio that's LESS stable means even the "
                  "shape itself isn't converging.")

    print(f"\n{len(unstable)}/{len(varying)} varying parameter(s) flagged UNSTABLE (range/mean > "
          f"{args.unstable_threshold}): {unstable if unstable else 'none'}")
    print("\nA parameter that keeps swinging by more than half its own mean across re-optimization rounds")
    print("is a parameter Optuna isn't actually converging on -- each round may be re-discovering noise in")
    print("that specific search's own data window/trial-sampling, not refining a genuine estimate. Worth")
    print("extra scrutiny (a wider Optuna search-space bound, more trials, or accepting this param is")
    print("inherently unstable and shouldn't be over-trusted) before leaning heavily on any ONE version's")
    print("own value for it.")


if __name__ == "__main__":
    main()
