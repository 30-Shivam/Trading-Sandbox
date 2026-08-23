"""
Mark a logged signal as an actual, confirmed fill, or record that you
passed on it and why (improvements.txt point 1 / item 92).

Trade_Signals logs every mechanical Strong Buy/Buy the scanner finds -- most
of those were never actually traded. settle_trades.py settles every one of
them regardless (that's what makes "is this signal any good" measurable at
all), but that also means the live win rate you see by default measures
"every signal's hypothetical outcome," not "what actually happened to
trades I made." This script lets you tell the system which signals you
actually acted on (confirming a fill also records a "acted_on" decision,
see storage.record_user_decision()) or explicitly passed on and why, so
reporting can separate hypothetical from real outcomes, AND so
user_preferences.summarize_decisions() has real data to build a revealed-
preference digest from over time.

--price (optional) records your real fill price if it differed from the
system's computed buy_price. Stop_Loss/Sell_Price are never touched --
they're absolute price levels computed at signal time, not relative to
whatever price you actually got filled at -- only the entry price used for
pnl_pct calculation changes.

--pass records that you explicitly declined this signal, with your reason
as free text -- a preference/journal concept, deliberately separate from
confirmed_filled's own financial-truth concept (see
storage.record_user_decision()'s own docstring). Does NOT call
confirm_fill() or touch settlement -- a pass never happened, there's
nothing to re-settle.

Usage:
    python confirm_fill.py                                   # list pending (unconfirmed) signals
    python confirm_fill.py --ticker INTC --date 2026-07-24
    python confirm_fill.py --ticker INTC --date 2026-07-24 --price 89.75
    python confirm_fill.py --undo --ticker INTC --date 2026-07-24
    python confirm_fill.py --ticker INTC --date 2026-07-24 --pass "too extended, RSI already overbought"
"""

import argparse
import sys

import storage
from settle_trades import settle_one


def _resettle_if_already_settled(ticker: str, signal_date: str, strategy: str) -> None:
    """confirm_fill()/unconfirm_fill() only touch the Trade_Signals document.
    If that signal already settled BEFORE you confirmed it, its
    Trade_Outcomes doc is now stale (wrong confirmed_filled, wrong
    entry_price) and won't naturally get re-walked -- get_unsettled_signals
    only returns signals with settled != True. Re-settle it right now so
    the outcome reflects the confirmation immediately instead of silently
    staying wrong forever."""
    db = storage.get_db()
    signal = db["Trade_Signals"].find_one({"ticker": ticker, "signal_date": signal_date, "strategy": strategy})
    if signal and signal.get("settled"):
        outcome, _ = settle_one(signal)  # a manual re-settle, not the scheduled batch job -- no Discord digest
        print(f"  (was already settled -- re-settled: {outcome})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--date", default=None, help="signal_date, YYYY-MM-DD")
    parser.add_argument(
        "--strategy", default="rsi",
        help="Which strategy's signal to confirm (rsi/breakout/pullback/breakout_retest/week52_high) -- "
             "required to disambiguate now that the same ticker/date can have more than one candidate. "
             "Default: rsi (matches every pre-existing signal logged before multi-strategy support).",
    )
    parser.add_argument("--price", type=float, default=None, help="Your actual fill price, if different from the logged buy_price.")
    parser.add_argument("--undo", action="store_true", help="Unmark a previously confirmed fill.")
    parser.add_argument(
        "--pass", dest="pass_reason", default=None, metavar="REASON",
        help="Record that you explicitly passed on this signal, with your reason as free text. "
             "Mutually exclusive with confirming a fill -- does not touch confirmed_filled/settlement.",
    )
    args = parser.parse_args()

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.ticker and args.date:
        if args.undo:
            storage.unconfirm_fill(args.ticker, args.date, args.strategy)
            print(f"Unconfirmed {args.ticker} ({args.date}, strategy={args.strategy}).")
            _resettle_if_already_settled(args.ticker, args.date, args.strategy)
        elif args.pass_reason:
            storage.record_user_decision(args.ticker, args.date, args.strategy, "passed", reason=args.pass_reason)
            print(f"Recorded PASS on {args.ticker} ({args.date}, strategy={args.strategy}): {args.pass_reason}")
        else:
            storage.confirm_fill(args.ticker, args.date, args.strategy, fill_price=args.price)
            storage.record_user_decision(args.ticker, args.date, args.strategy, "acted_on")
            price_note = f" at ${args.price:.2f}" if args.price else " (using the logged buy_price)"
            print(f"Confirmed {args.ticker} ({args.date}, strategy={args.strategy}) as filled{price_note}.")
            _resettle_if_already_settled(args.ticker, args.date, args.strategy)
        return

    pending = storage.get_signals_pending_confirmation()
    if not pending:
        print("No unconfirmed signals.")
        return
    print(f"{len(pending)} unconfirmed signal(s), most recent first:")
    for s in pending:
        strategy = s.get("strategy", "rsi")
        print(
            f"  {s['ticker']:6s} {s['signal_date']}  {strategy:15s}  {s['signal']:11s}  "
            f"buy={s['buy_price']:.2f}  stop={s['stop_loss']:.2f}  sell={s['sell_price']:.2f}"
        )
    print()
    print("Confirm one with: python confirm_fill.py --ticker TICKER --date SIGNAL_DATE --strategy STRATEGY [--price YOUR_FILL_PRICE]")


if __name__ == "__main__":
    main()
