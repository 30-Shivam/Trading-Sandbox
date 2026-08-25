"""One-off reconciliation tool (2026-08-24) for the entry-fill-gate bug
fixed in settle_trades.py the same day: every UNCONFIRMED Trade_Outcomes
document was previously graded as if it filled at its literal buy_price on
day one, with no check that the market ever actually touched a resting
limit that, for a discount-entry strategy like RSI mean-reversion, can sit
far below the real market. Confirmed live: research-tier rsi_mean_reversion
outcomes were sitting at a literal 100% win rate (142/142, avg +19.0%, max
+170.98%) purely from this -- see settle_trades.py's settle_one() docstring
for the full incident writeup.

This tool re-derives every EXISTING outcome for a given --strategy through
the now-fixed settle_trades.settle_one() logic, using each outcome's own
historical Trade_Signals document (matched by ticker/signal_date/strategy)
and REAL price history fetched fresh:

  - If the corrected logic still resolves to WIN/LOSS/EXPIRED (real,
    unconfirmed fill actually happened, or a CONFIRMED trade -- those are
    unaffected either way), the existing document is upserted in place with
    corrected values (settle_one() -> storage.log_trade_outcome() does this
    automatically, keyed on ticker/signal_date/strategy).
  - If it now resolves to NEVER_FILLED/still-waiting/still-OPEN (the
    resting limit was never realistically touched -- this is the bug's
    signature), the existing document is DELETED outright, since it no
    longer represents a real or even realistically-hypothetical outcome.
  - If the original Trade_Signals document is missing (very old data
    predating this schema), the outcome is left untouched and flagged for
    manual review rather than guessed at.

Read-only dry-run by default. settle_trades.settle_one() is the live
production function -- it always writes for real (log_trade_outcome/
mark_settled) whenever it resolves, so a dry run can't just skip an
--apply check around it; storage.log_trade_outcome/storage.mark_settled/
this module's own delete_one call are monkeypatched to no-ops for the
duration of the dry run instead, exactly the same dependency-injection
pattern tests/test_settle_one_entry_fill_gate.py already uses. Pass
--apply to let those writes actually happen.

Usage:
    python reconcile_settled_outcomes.py --strategy rsi_mean_reversion
    python reconcile_settled_outcomes.py --strategy rsi_mean_reversion --apply
"""
import argparse
import sys
import time
from unittest.mock import patch

import settle_trades
import storage

REQUEST_DELAY_SEC = 0.5   # same yfinance rate-limit courtesy as settle_trades.py


def _run(args, outcomes_collection, signals_collection):
    outcomes = list(outcomes_collection.find({"strategy": args.strategy}))
    print(f"Found {len(outcomes)} existing Trade_Outcomes document(s) for strategy={args.strategy}.")
    print(f"Mode: {'APPLY (will delete/upsert)' if args.apply else 'DRY RUN (no changes will be made)'}")
    print()

    counts = {
        "corrected_same_verdict": 0, "corrected_new_verdict": 0,
        "deleted_no_longer_valid": 0, "skipped_no_signal_doc": 0, "error": 0,
    }

    for i, outcome in enumerate(outcomes):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        ticker, signal_date = outcome["ticker"], outcome["signal_date"]
        signal = signals_collection.find_one({"ticker": ticker, "signal_date": signal_date, "strategy": args.strategy})
        if signal is None:
            print(f"  {ticker} ({signal_date}): [SKIP] no matching Trade_Signals document -- leaving untouched.")
            counts["skipped_no_signal_doc"] += 1
            continue

        old_status, old_pnl = outcome["status"], outcome["pnl_pct"]
        try:
            new_outcome_str, _ = settle_trades.settle_one(signal)
        except Exception as exc:
            print(f"  {ticker} ({signal_date}): [ERROR] {exc}")
            counts["error"] += 1
            continue

        new_status = new_outcome_str.split()[0]
        if new_status in ("WIN", "LOSS", "EXPIRED"):
            new_pnl = float(new_outcome_str.split("(")[1].split(",")[1].strip().rstrip("%"))
            if new_status == old_status and abs(new_pnl - old_pnl) < 0.01:
                print(f"  {ticker} ({signal_date}): unchanged -- {new_outcome_str}")
                counts["corrected_same_verdict"] += 1
            else:
                print(f"  {ticker} ({signal_date}): CORRECTED {old_status} ({old_pnl:+.2f}%) -> {new_outcome_str}")
                counts["corrected_new_verdict"] += 1
        else:
            # NEVER_FILLED / waiting for entry fill / still OPEN -- the
            # resting limit was never realistically touched. The old
            # document no longer represents a real or even realistically-
            # hypothetical outcome, so it must go.
            print(f"  {ticker} ({signal_date}): WAS {old_status} ({old_pnl:+.2f}%) -> now {new_outcome_str} "
                  f"-- {'deleting' if args.apply else 'WOULD DELETE'} the old fictional outcome.")
            counts["deleted_no_longer_valid"] += 1
            outcomes_collection.delete_one({"ticker": ticker, "signal_date": signal_date, "strategy": args.strategy})

    print()
    print(f"Summary: {counts}")
    if not args.apply:
        print("This was a DRY RUN -- nothing was changed. Re-run with --apply to actually delete/upsert.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, help="e.g. rsi_mean_reversion")
    parser.add_argument("--apply", action="store_true", help="Actually delete/upsert. Default is a read-only dry run.")
    args = parser.parse_args()

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    db = storage.get_db()
    # Obtained ONCE and reused for every find()/delete_one() call in _run()
    # -- pymongo's Database.__getitem__ constructs a new Collection wrapper
    # object on every call, so patching db["Trade_Outcomes"].delete_one
    # freshly indexed inside the loop would silently miss every iteration
    # after the first; patching this single, reused instance's method
    # actually takes effect throughout the whole run.
    outcomes_collection = db["Trade_Outcomes"]
    signals_collection = db["Trade_Signals"]

    if args.apply:
        _run(args, outcomes_collection, signals_collection)
    else:
        with patch.object(storage, "log_trade_outcome"), \
             patch.object(storage, "mark_settled"), \
             patch.object(outcomes_collection, "delete_one"):
            _run(args, outcomes_collection, signals_collection)


if __name__ == "__main__":
    main()
