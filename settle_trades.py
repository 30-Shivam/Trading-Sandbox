"""
Nightly settlement job (Phase 3).

Walks every unsettled Trade_Signals document forward through actual price
history and resolves it to WIN / LOSS / EXPIRED via swingtrade.settle_trade()
(gap-aware: a stop-loss is a trigger price, not a guaranteed fill -- see
ARCHITECTURE_PLAN.md section 3). Terminal outcomes are written to
Trade_Outcomes and the source Trade_Signals document is marked settled.
Trades that haven't resolved yet are left untouched and simply get re-walked
from scratch on the next run -- there's no incremental state to corrupt, so
this is safe to run as often as you like (once a day, via cron, is enough
since it's all daily-bar data).

Each signal is settled using ITS OWN config_snapshot (stored on the
Trade_Signals document at signal time), not whichever System_Config happens
to be "active" when this job runs. That matters: max_holding_days and the
execution-realism fields (slippage_pct, commission_pct_per_trade) are exit
policy, and a trade should be graded against the policy in effect when it
was entered, not retroactively reinterpreted under whatever got promoted
afterward. TradingConfig.from_dict() handles older snapshots that predate a
field (e.g. slippage_pct) fine -- missing keys just fall back to that
field's dataclass default.

Most logged signals were never actually traded -- see confirm_fill.py. Every
signal still gets settled and logged to Trade_Outcomes regardless (that's
what makes "is this signal any good" measurable at all), but a signal
confirmed via confirm_fill.py is settled using your real fill_price as the
entry price if you gave one (Stop_Loss/Sell_Price never change -- they're
absolute levels computed at signal time), and confirmed_filled is carried
onto the Trade_Outcomes document so reporting can separate "every mechanical
signal's hypothetical outcome" from "what actually happened to trades you
made." The summary below breaks both out.

Usage:
    python settle_trades.py
"""

import sys
import time

import pandas as pd
import yfinance as yf

import notifications
import storage
import swingtrade

REQUEST_DELAY_SEC = 0.5   # pause between yfinance calls to avoid rate-limiting


def fetch_bars_since(ticker: str, signal_date: str) -> pd.DataFrame:
    """Daily OHLCV strictly AFTER signal_date, chronological order."""
    df = yf.download(ticker, start=signal_date, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[df.index > pd.Timestamp(signal_date)]


def settle_one(signal: dict) -> tuple[str, dict | None]:
    """Returns (outcome_str, notify_record) -- notify_record is None for the
    two "nothing actually settled" paths (no new bars yet, still OPEN), and
    otherwise a dict with every field _build_settlement_notification() needs
    (ticker, strategy, status, exit_reason, pnl_pct, holding_days, confirmed,
    tier). Exposed as a second return value rather than a new function
    (settle_one() already computes every one of these fields locally) --
    settle_trades.py's main() uses it to batch a per-strategy Discord
    digest; confirm_fill.py's own call site (a manual re-settle, not the
    scheduled batch job) just discards it."""
    ticker = signal["ticker"]
    signal_date = signal["signal_date"]
    bars = fetch_bars_since(ticker, signal_date)
    if bars.empty:
        return "no new bars yet", None

    config = swingtrade.TradingConfig.from_dict(signal["config_snapshot"])
    strategy = config.strategy  # correct for old docs too -- from_dict() defaults missing "strategy" to "rsi"
    confirmed = bool(signal.get("confirmed_filled", False))
    tier = signal.get("tier", "actionable")  # pre-existing docs predate this field -- they were all actionable
    entry_price = signal.get("fill_price") or signal["buy_price"]
    result = swingtrade.settle_trade(
        buy_price=entry_price,
        stop_loss=signal["stop_loss"],
        sell_price=signal["sell_price"],
        bars_since_entry=bars,
        config=config,
    )

    if result["status"] == "OPEN":
        return "OPEN (still open)", None

    storage.log_trade_outcome(
        ticker, signal_date, strategy, entry_price, result, confirmed_filled=confirmed, tier=tier
    )
    storage.mark_settled(ticker, signal_date, strategy)
    tag = "CONFIRMED" if confirmed else "unconfirmed"
    outcome_str = (
        f"{result['status']} ({result['exit_reason']}, "
        f"{result['pnl_pct']:+.2f}%, held {result['holding_days']}d, {tag})"
    )
    notify_record = {
        "ticker": ticker, "strategy": strategy, "status": result["status"],
        "exit_reason": result["exit_reason"], "pnl_pct": result["pnl_pct"],
        "holding_days": result["holding_days"], "confirmed": confirmed, "tier": tier,
    }
    return outcome_str, notify_record


def _build_settlement_notification(strategy: str, records: list[dict]) -> str | None:
    """Short Discord message summarizing this run's newly-settled
    actionable-tier trades for one strategy, sorted by pnl_pct descending
    (biggest mover first) -- mirrors ingest.py's _build_signal_notification()
    in shape/tone (same "closes the loop" framing, just for outcomes
    instead of new signals). Returns None if `records` is empty -- callers
    skip notifying entirely for a strategy with nothing actionable settled
    this run."""
    if not records:
        return None
    records = sorted(records, key=lambda r: r["pnl_pct"], reverse=True)
    lines = [f"**{strategy}: {len(records)} trade(s) settled**"]
    for r in records:
        tag = "CONFIRMED" if r["confirmed"] else "unconfirmed"
        lines.append(
            f"{r['ticker']}: {r['status']} ({r['exit_reason']}, {r['pnl_pct']:+.2f}%, "
            f"held {r['holding_days']}d, {tag})"
        )
    return "\n".join(lines)


def main():
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    unsettled = storage.get_unsettled_signals()
    print(f"Found {len(unsettled)} unsettled signal(s).")

    counts: dict[str, int] = {}
    by_strategy: dict[str, list[dict]] = {}
    for i, signal in enumerate(unsettled):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        ticker = signal["ticker"]
        try:
            outcome, notify_record = settle_one(signal)
        except Exception as exc:
            outcome, notify_record = f"ERROR: {exc}", None
        print(f"  {ticker} ({signal['signal_date']}): {outcome}")
        if outcome.startswith("ERROR"):
            key = "ERROR"
        elif outcome == "no new bars yet":
            key = "PENDING_NO_DATA"
        else:
            key = outcome.split()[0]  # WIN / LOSS / EXPIRED / OPEN
        counts[key] = counts.get(key, 0) + 1
        if notify_record is not None and notify_record["tier"] == "actionable":
            by_strategy.setdefault(notify_record["strategy"], []).append(notify_record)

    print()
    print(f"Summary: {counts}")

    db = storage.get_db()
    all_outcomes = list(db["Trade_Outcomes"].find({}))
    actionable_outcomes = [o for o in all_outcomes if o.get("tier", "actionable") == "actionable"]
    research_outcomes = [o for o in all_outcomes if o.get("tier") == "research"]
    loosened_outcomes = [o for o in all_outcomes if o.get("tier") == "research_loosened"]
    confirmed_outcomes = [o for o in all_outcomes if o.get("confirmed_filled")]
    as_trades = lambda outs: [{"status": o["status"], "pnl_pct": o["pnl_pct"]} for o in outs]  # noqa: E731
    print()
    print(f"All-time, every mechanical signal ({len(all_outcomes)} settled): "
          f"{swingtrade.summarize_trades(as_trades(all_outcomes))}")
    print(f"  Actionable tier (Strong Buy/Buy) only ({len(actionable_outcomes)} settled): "
          f"{swingtrade.summarize_trades(as_trades(actionable_outcomes))}")
    print(f"    ...of which CONFIRMED real fills ({len(confirmed_outcomes)} settled): "
          f"{swingtrade.summarize_trades(as_trades(confirmed_outcomes))}")
    print(f"  Research tier (Watch, never traded/tradeable) ({len(research_outcomes)} settled): "
          f"{swingtrade.summarize_trades(as_trades(research_outcomes))}")
    if loosened_outcomes:
        print(f"  Research_loosened tier (active config scored Ignore, would-be signal under "
              f"loosened filters, never traded/tradeable) ({len(loosened_outcomes)} settled): "
              f"{swingtrade.summarize_trades(as_trades(loosened_outcomes))}")
    if not confirmed_outcomes:
        print("No confirmed fills yet -- see confirm_fill.py to mark real trades as you make them.")

    # One digest message per strategy that had an actionable-tier settlement
    # this run (not one message per trade -- avoids Discord rate-limit risk
    # on a run that settles several trades at once), routed to that
    # strategy's own channel exactly like ingest.py's Strong Buy
    # notifications -- same webhook, now reporting what happened to the
    # signals it already told you about.
    for strategy, records in by_strategy.items():
        message = _build_settlement_notification(strategy, records)
        if message:
            notifications.notify(message, webhook_url=notifications.get_strategy_webhook_url(strategy))


if __name__ == "__main__":
    main()
