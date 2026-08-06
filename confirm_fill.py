"""
Mark a logged signal as an actual, confirmed fill (improvements.txt point 1).

Trade_Signals logs every mechanical Strong Buy/Buy the scanner finds -- most
of those were never actually traded. settle_trades.py settles every one of
them regardless (that's what makes "is this signal any good" measurable at
all), but that also means the live win rate you see by default measures
"every signal's hypothetical outcome," not "what actually happened to
trades I made." This script lets you tell the system which signals you
actually acted on, so reporting can separate the two.

--price (optional) records your real fill price if it differed from the
system's computed buy_price. Stop_Loss/Sell_Price are never touched --
they're absolute price levels computed at signal time, not relative to
whatever price you actually got filled at -- only the entry price used for
pnl_pct calculation changes.

Usage:
    python confirm_fill.py                                   # list pending (unconfirmed) signals
    python confirm_fill.py --ticker INTC --date 2026-07-24
    python confirm_fill.py --ticker INTC --date 2026-07-24 --price 89.75
    python confirm_fill.py --undo --ticker INTC --date 2026-07-24
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
        outcome = settle_one(signal)
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
        else:
            storage.confirm_fill(args.ticker, args.date, args.strategy, fill_price=args.price)
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
