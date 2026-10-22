"""Alpaca PAPER-trading executor: turns today's actionable signals into real
(simulated-money) bracket orders, then writes the real fills back to Mongo.

Why: signals are otherwise graded as if bought at exactly Buy_Price, and only
2 fills have ever been confirmed. Paper orders give real fill/slippage/gap
evidence. Paper fills are optimistic (no queue position or market impact), so
treat paper P&L as an upper bound, not proof of edge.

Safety rails
  * Off unless PAPER_TRADING_ENABLED=1. `--dry-run` prints orders, sends none.
  * broker.py refuses any host except paper-api.alpaca.markets.
  * Only strategies in PAPER_STRATEGIES (default value_rank,accruals_rank),
    only tier "actionable", only plain USD tickers (no '.'/'-' share classes), whole shares.
  * Caps: PAPER_MAX_POSITIONS, per-trade risk and notional, total exposure % of equity.
  * client_order_id = ts.<strategy>.<ticker>.<signal_date> is unique, so a
    re-run can never double-order (Alpaca rejects duplicates; we also pre-check).
  * Stale unfilled entries are cancelled after PAPER_ENTRY_TTL_DAYS; positions
    open longer than PAPER_MAX_HOLD_DAYS are closed (matching settlement's expiry).

Usage:
    python paper_trade.py                # place + reconcile + housekeeping
    python paper_trade.py --dry-run
    python paper_trade.py --report       # paper P&L vs hypothetical settlement
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import broker as broker_mod
import storage

ID_PREFIX = "ts"
DEFAULTS = {
    "PAPER_STRATEGIES": "value_rank,accruals_rank",
    "PAPER_MAX_POSITIONS": "10",
    "PAPER_RISK_PER_TRADE_USD": "100",
    "PAPER_MAX_NOTIONAL_USD": "2500",
    "PAPER_MAX_EXPOSURE_PCT": "0.5",
    "PAPER_ENTRY_TTL_DAYS": "3",
    "PAPER_MAX_HOLD_DAYS": "21",
}


def _setting(name: str) -> str:
    return os.getenv(name) or DEFAULTS[name]


def load_settings() -> dict:
    return {
        "strategies": {s.strip() for s in _setting("PAPER_STRATEGIES").split(",") if s.strip()},
        "max_positions": int(_setting("PAPER_MAX_POSITIONS")),
        "risk_usd": float(_setting("PAPER_RISK_PER_TRADE_USD")),
        "max_notional": float(_setting("PAPER_MAX_NOTIONAL_USD")),
        "max_exposure_pct": float(_setting("PAPER_MAX_EXPOSURE_PCT")),
        "entry_ttl_days": int(_setting("PAPER_ENTRY_TTL_DAYS")),
        "max_hold_days": int(_setting("PAPER_MAX_HOLD_DAYS")),
    }


def make_client_order_id(strategy: str, ticker: str, signal_date: str) -> str:
    return f"{ID_PREFIX}.{strategy}.{ticker}.{signal_date}"


def parse_client_order_id(client_order_id: str | None) -> tuple[str, str, str] | None:
    """(strategy, ticker, signal_date) or None if this isn't one of our orders."""
    if not client_order_id or not client_order_id.startswith(f"{ID_PREFIX}."):
        return None
    parts = client_order_id.split(".", 3)
    return (parts[1], parts[2], parts[3]) if len(parts) == 4 else None


def size_position(buy: float, stop: float, risk_usd: float, max_notional: float) -> int:
    """Whole shares (bracket orders can't be fractional): risk-based, capped by notional."""
    per_share_risk = buy - stop
    if buy <= 0 or per_share_risk <= 0:
        return 0
    return max(0, min(math.floor(risk_usd / per_share_risk), math.floor(max_notional / buy)))


def plan_orders(signals: list[dict], held: set[str], open_symbols: set[str], existing_ids: set[str],
                equity: float, exposure_used: float, n_positions: int, cfg: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """Pure decision logic. Returns (orders, skipped[(label, reason)])."""
    orders, skipped = [], []
    exposure_cap = equity * cfg["max_exposure_pct"]
    slots = cfg["max_positions"] - n_positions - len(open_symbols - held)
    for s in sorted(signals, key=lambda d: d.get("trade_score") or 0, reverse=True):
        label = f"{s['strategy']}:{s['ticker']}"
        coid = make_client_order_id(s["strategy"], s["ticker"], s["signal_date"])
        if s["strategy"] not in cfg["strategies"]:
            skipped.append((label, "strategy not enabled for paper trading")); continue
        if "." in s["ticker"] or "-" in s["ticker"] or (s.get("currency") or "USD") != "USD":
            skipped.append((label, "non-USD or share-class ticker (Alpaca symbol mapping differs)")); continue
        if coid in existing_ids:
            skipped.append((label, "already ordered")); continue
        if s["ticker"] in held or s["ticker"] in open_symbols:
            skipped.append((label, "position or order already open for ticker")); continue
        if slots <= 0:
            skipped.append((label, "max positions reached")); continue
        buy, stop, target = s.get("buy_price"), s.get("stop_loss"), s.get("sell_price")
        if not all(isinstance(v, (int, float)) and v > 0 for v in (buy, stop, target)) or not (stop < buy < target):
            skipped.append((label, "invalid price levels")); continue
        qty = size_position(buy, stop, cfg["risk_usd"], cfg["max_notional"])
        if qty < 1:
            skipped.append((label, "position size rounds to zero shares")); continue
        notional = qty * buy
        if exposure_used + notional > exposure_cap:
            skipped.append((label, "total exposure cap reached")); continue
        exposure_used += notional
        slots -= 1
        orders.append({"symbol": s["ticker"], "qty": qty, "limit_price": round(buy, 2),
                       "stop_price": round(stop, 2), "take_profit_price": round(target, 2),
                       "client_order_id": coid, "strategy": s["strategy"], "signal_date": s["signal_date"]})
    return orders, skipped


def fetch_signals(allowed: set[str]) -> list[dict]:
    """Actionable signals from the most recent scan date, for allowed strategies."""
    col = storage.get_db()["Trade_Signals"]
    latest = col.find_one({"tier": "actionable", "strategy": {"$in": sorted(allowed)}},
                          sort=[("signal_date", -1)], projection={"signal_date": 1})
    if latest is None:
        return []
    return list(col.find({"tier": "actionable", "strategy": {"$in": sorted(allowed)},
                          "signal_date": latest["signal_date"]}, {"_id": 0}))


def reconcile_fills(orders: list[dict]) -> int:
    """Write real entry fills back to Mongo (confirmed_filled + fill price) and
    re-settle if needed. Idempotent: signals already confirmed are skipped."""
    from confirm_fill import resettle_if_already_settled
    col = storage.get_db()["Trade_Signals"]
    confirmed = 0
    for o in orders:
        parsed = parse_client_order_id(o.get("client_order_id"))
        if not parsed or o.get("status") != "filled" or not o.get("filled_avg_price"):
            continue
        strategy, ticker, signal_date = parsed
        doc = col.find_one({"ticker": ticker, "signal_date": signal_date, "strategy": strategy})
        if doc is None or doc.get("confirmed_filled"):
            continue
        storage.confirm_fill(ticker, signal_date, strategy, fill_price=float(o["filled_avg_price"]))
        storage.record_user_decision(ticker, signal_date, strategy, "acted_on")
        resettle_if_already_settled(ticker, signal_date, strategy)
        confirmed += 1
    return confirmed


def housekeeping(broker, orders: list[dict], positions: list[dict], cfg: dict, now: datetime) -> tuple[int, int]:
    """Cancel stale unfilled entries; close over-age positions. Returns (cancelled, closed)."""
    cancelled = closed = 0
    for o in orders:
        if not parse_client_order_id(o.get("client_order_id")):
            continue
        if o.get("status") in ("new", "accepted", "pending_new", "partially_filled") and not o.get("filled_at"):
            created = datetime.fromisoformat(o["created_at"].replace("Z", "+00:00"))
            if now - created > timedelta(days=cfg["entry_ttl_days"]):
                broker.cancel_order(o["id"]); cancelled += 1
    held = {p["symbol"] for p in positions}
    for o in orders:
        if parse_client_order_id(o.get("client_order_id")) and o.get("filled_at") and o["symbol"] in held:
            filled = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
            if now - filled > timedelta(days=cfg["max_hold_days"]):
                for leg in o.get("legs") or []:  # legs reserve the shares; cancel before closing
                    if leg.get("status") in ("new", "accepted", "held", "pending_new"):
                        broker.cancel_order(leg["id"])
                broker.close_position(o["symbol"]); closed += 1; held.discard(o["symbol"])
    return cancelled, closed


def equity_vs_index(history: dict, index_close) -> dict | None:
    """Paper account return vs SPY over the same span -- the real scoreboard.
    `history` is Alpaca's portfolio-history payload (timestamp, equity); returns
    None until there are at least two equity points after the first trade."""
    import pandas as pd
    stamps, equity = history.get("timestamp") or [], history.get("equity") or []
    pts = [(pd.Timestamp(t, unit="s").normalize(), e) for t, e in zip(stamps, equity) if e]
    if len(pts) < 2:
        return None
    (d0, e0), (d1, e1) = pts[0], pts[-1]
    close = index_close.dropna()
    close.index = pd.to_datetime(close.index).normalize()
    i0, i1 = close.asof(d0), close.asof(d1)
    if not i0 or not i1 or pd.isna(i0) or pd.isna(i1):
        return None
    acct, idx = (e1 / e0 - 1) * 100, (float(i1) / float(i0) - 1) * 100
    return {"from": str(d0.date()), "to": str(d1.date()), "account_pct": round(acct, 2),
            "spy_pct": round(idx, 2), "excess_pct": round(acct - idx, 2)}


def report(broker) -> None:
    """Realized paper P&L per closed bracket vs the hypothetical settlement for the same signal."""
    col = storage.get_db()["Trade_Outcomes"]
    rows = []
    for o in broker.orders(status="closed"):
        parsed = parse_client_order_id(o.get("client_order_id"))
        if not parsed or not o.get("filled_avg_price"):
            continue
        exit_leg = next((l for l in o.get("legs") or [] if l.get("status") == "filled" and l.get("filled_avg_price")), None)
        if exit_leg is None:
            continue
        strategy, ticker, signal_date = parsed
        entry, exit_px = float(o["filled_avg_price"]), float(exit_leg["filled_avg_price"])
        paper = (exit_px / entry - 1) * 100
        sim = col.find_one({"ticker": ticker, "signal_date": signal_date, "strategy": strategy})
        rows.append((strategy, ticker, signal_date, paper, sim.get("pnl_pct") if sim else None))
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)["Close"].squeeze()
        summary = equity_vs_index(broker.portfolio_history(), spy)
        print("PAPER vs SPY:", summary if summary else "not enough equity history yet.")
    except Exception as exc:  # report is informational; never fail the run
        print(f"PAPER vs SPY: unavailable ({exc})")
    if not rows:
        print("PAPER REPORT: no closed paper brackets yet.")
        return
    print(f"PAPER REPORT: {len(rows)} closed trade(s)")
    for strategy, ticker, d, paper, sim in rows:
        sim_text = f"{sim:+.2f}%" if sim is not None else "n/a"
        print(f"  {strategy:15s} {ticker:6s} {d}  paper {paper:+.2f}%  hypothetical {sim_text}")
    paired = [(p, s) for *_, p, s in rows if s is not None]
    if paired:
        drag = sum(s - p for p, s in paired) / len(paired)
        print(f"  mean execution drag (hypothetical - paper): {drag:+.2f}pp over {len(paired)} paired trade(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print planned orders; send nothing.")
    parser.add_argument("--report", action="store_true", help="Print paper vs hypothetical P&L and exit.")
    args = parser.parse_args()

    if os.getenv("PAPER_TRADING_ENABLED") != "1" and not (args.dry_run or args.report):
        print("PAPER disabled (set PAPER_TRADING_ENABLED=1 to enable).")
        return 0

    try:
        storage.ensure_indexes()
        client = broker_mod.AlpacaPaperBroker()
        account = client.account()
    except (storage.MongoNotConfigured, broker_mod.BrokerError) as exc:
        print(f"[ERROR] paper trading unavailable: {exc}", file=sys.stderr)
        return 1
    if account.get("trading_blocked") or account.get("account_blocked"):
        print("[ERROR] Alpaca paper account is blocked.", file=sys.stderr)
        return 1
    if args.report:
        report(client)
        return 0

    cfg = load_settings()
    equity = float(account["equity"])
    now = datetime.now(timezone.utc)
    positions = client.positions()
    cutoff = (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_orders = client.orders(status="all", after=cutoff)

    confirmed = 0 if args.dry_run else reconcile_fills(all_orders)
    cancelled, closed = (0, 0) if args.dry_run else housekeeping(client, all_orders, positions, cfg, now)

    existing_ids = {o["client_order_id"] for o in all_orders if o.get("client_order_id")}
    held = {p["symbol"] for p in positions}
    open_symbols = {o["symbol"] for o in all_orders
                    if o.get("status") in ("new", "accepted", "pending_new", "partially_filled")}
    exposure = sum(abs(float(p.get("market_value", 0))) for p in positions)
    planned, skipped = plan_orders(fetch_signals(cfg["strategies"]), held, open_symbols, existing_ids,
                                   equity, exposure, len(positions), cfg)

    placed = 0
    for o in planned:
        desc = (f"{o['symbol']} x{o['qty']} limit {o['limit_price']} stop {o['stop_price']} "
                f"target {o['take_profit_price']} [{o['strategy']}]")
        if args.dry_run:
            print(f"  DRY-RUN would place: {desc}"); continue
        try:
            client.submit_bracket_limit(o["symbol"], o["qty"], o["limit_price"], o["stop_price"],
                                        o["take_profit_price"], o["client_order_id"])
            placed += 1; print(f"  placed: {desc}")
        except broker_mod.BrokerError as exc:
            if exc.status == 422 and "client_order_id" in str(exc):
                continue  # duplicate: already placed on an earlier run
            print(f"  [WARN] order failed for {desc}: {exc}", file=sys.stderr)
    for label, reason in skipped:
        if reason != "strategy not enabled for paper trading":
            print(f"  skipped {label}: {reason}")
    print(f"PAPER placed={placed} planned={len(planned)} fills_confirmed={confirmed} cancelled_stale={cancelled} "
          f"closed_expired={closed} positions={len(positions)} equity=${equity:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
