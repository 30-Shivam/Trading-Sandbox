"""Core + tilt + satellite policy for the Alpaca PAPER account.

Objective: beat SPY. Honest structure for doing that:
  * CORE   - SPY. The benchmark itself; the default home for money.
  * TILT   - a deliberate, sized beta bet (default QQQ). This is NOT alpha: it beat
             SPY by ~5%/yr over 2016-2026 with a deeper drawdown (-33% vs -24%) in a
             mega-cap-growth regime that may not repeat. Size it so being wrong is tolerable.
  * SATELLITE - stock signals (paper_trade.py), capped at SATELLITE_RESERVE of equity,
             and only strategies that pass the index gate belong there.

Targets are fractions of account equity; cash for the satellite reserve is held back.
Rebalance only when a sleeve drifts more than DRIFT_BAND from target (or is missing),
sells before buys, never spends more than available cash, and uses idempotent
per-day client_order_ids so a re-run cannot double-trade.

Off unless PAPER_TRADING_ENABLED=1 (or --dry-run, which sends nothing).

Usage:
    python rebalance.py --dry-run
    PAPER_TRADING_ENABLED=1 python rebalance.py
Env overrides: CORE_TARGETS="SPY:0.70,QQQ:0.15"  SATELLITE_RESERVE=0.15  REBALANCE_DRIFT_BAND=0.03
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import broker as broker_mod

DEFAULT_TARGETS = {"SPY": 0.70, "QQQ": 0.15}
DEFAULT_SATELLITE_RESERVE = 0.15
DEFAULT_DRIFT_BAND = 0.03
MIN_TRADE_USD = 50.0


def core_symbols() -> set[str]:
    return set(parse_targets(os.getenv("CORE_TARGETS")))


def parse_targets(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_TARGETS)
    out = {}
    for part in text.split(","):
        sym, _, w = part.partition(":")
        out[sym.strip().upper()] = float(w)
    return out


def plan_rebalance(equity: float, cash: float, holdings: dict[str, float], targets: dict[str, float],
                   satellite_reserve: float, drift_band: float, min_trade: float = MIN_TRADE_USD) -> list[dict]:
    """Pure: returns [{symbol, side, notional, reason}] with sells first.
    `holdings` = market value per core symbol (0 if not held)."""
    if sum(targets.values()) + satellite_reserve > 1.0 + 1e-9:
        raise ValueError("core targets + satellite reserve exceed 100% of equity")
    trades = []
    for sym, w in targets.items():
        target_val, cur = equity * w, holdings.get(sym, 0.0)
        drift = (cur - target_val) / equity
        if cur == 0 and target_val >= min_trade:
            reason = "initial build"
        elif abs(drift) > drift_band:
            reason = f"drift {drift:+.1%} beyond {drift_band:.0%} band"
        else:
            continue
        diff = target_val - cur
        if abs(diff) >= min_trade:
            trades.append({"symbol": sym, "side": "buy" if diff > 0 else "sell", "notional": round(abs(diff), 2),
                           "reason": reason})
    trades.sort(key=lambda t: t["side"] != "sell")           # sells first
    available = cash + sum(t["notional"] for t in trades if t["side"] == "sell")
    kept = []
    for t in trades:
        if t["side"] == "buy":
            spend = min(t["notional"], max(available - equity * satellite_reserve, 0.0))
            if spend < min_trade:
                continue
            available -= spend
            t = {**t, "notional": round(spend, 2)}
        kept.append(t)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if os.getenv("PAPER_TRADING_ENABLED") != "1" and not args.dry_run:
        print("REBALANCE disabled (set PAPER_TRADING_ENABLED=1 to enable).")
        return 0
    try:
        client = broker_mod.AlpacaPaperBroker()
        account = client.account()
        positions = client.positions()
    except broker_mod.BrokerError as exc:
        print(f"[ERROR] rebalance unavailable: {exc}", file=sys.stderr)
        return 1
    if account.get("trading_blocked") or account.get("account_blocked"):
        print("[ERROR] Alpaca paper account is blocked.", file=sys.stderr)
        return 1
    targets = parse_targets(os.getenv("CORE_TARGETS"))
    reserve = float(os.getenv("SATELLITE_RESERVE") or DEFAULT_SATELLITE_RESERVE)
    band = float(os.getenv("REBALANCE_DRIFT_BAND") or DEFAULT_DRIFT_BAND)
    holdings = {p["symbol"]: float(p["market_value"]) for p in positions if p["symbol"] in targets}
    equity, cash = float(account["equity"]), float(account["cash"])
    plan = plan_rebalance(equity, cash, holdings, targets, reserve, band)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    for t in plan:
        desc = f"{t['side'].upper()} ${t['notional']:,.2f} {t['symbol']} ({t['reason']})"
        if args.dry_run:
            print(f"  DRY-RUN would: {desc}"); continue
        try:
            client.submit_market_notional(t["symbol"], t["notional"], t["side"], f"rb.{t['symbol']}.{t['side']}.{today}")
            print(f"  placed: {desc}")
        except broker_mod.BrokerError as exc:
            if exc.status == 422 and "client_order_id" in str(exc):
                print(f"  already placed today: {desc}"); continue
            print(f"  [WARN] failed: {desc}: {exc}", file=sys.stderr)
    weights = {s: f"{holdings.get(s, 0) / equity:.1%}" for s in targets}
    print(f"REBALANCE trades={len(plan)} equity=${equity:,.0f} cash=${cash:,.0f} core_weights={weights} "
          f"targets={targets} satellite_reserve={reserve:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
