"""Human-in-the-loop rebalancing advisor for a real Canadian TFSA (Scotia iTRADE). READ-ONLY toward the broker: it prints
instructions you execute by hand, then `--confirm-fill` records what you actually did. It never talks to a broker.

Policy: 85% VFV.TO (S&P 500) / 15% XQQ.TO (Nasdaq-100, CAD-hedged; QQC.TO is the unhedged alternative: --targets
"VFV.TO:0.85,QQC.TO:0.15"). Both fund legs are CAD-listed, so there is no currency conversion. Weights are of the invested
sleeve (the two funds); other holdings and idle cash are ignored, with a warning.

Cash-first: new cash (--cash) goes to the underweight fund first, up to its target, and only the remainder is split by target
weight. The advisor never recommends a sell, so a rebalance costs no extra commission and, in a TFSA, no tax event. Shares are
whole; commission is charged per trade. If drift is beyond the band and there is no cash to fix it, it tells you how much cash
would fix it instead of selling.

Usage (run from the repo root):
    python -m swingtrade.advisor                              # check drift only
    python -m swingtrade.advisor --cash 2500                  # what to buy with C$2,500 of new cash
    python -m swingtrade.advisor --cash 2500 --price VFV.TO=135.20 --price XQQ.TO=52.10   # use live iTRADE quotes
    python -m swingtrade.advisor --confirm-fill VFV.TO:15@135.20,XQQ.TO:7@52.10 --cash 2500   # record what you executed
    python -m swingtrade.advisor --confirm-fill ... --dry-run # preview the recorded result without saving
    python -m swingtrade.advisor --holdings-csv private/scotia_tfsa.csv   # advise from a CSV instead of Mongo (no saving)

Holdings come from Account_Holdings (account 'scotia_tfsa', the same store portfolio_view.py --save writes), cash as the
'CASH.CAD' row. `--cash` is NEW money to deploy: with --confirm-fill it is first added to the account's cash, then each fill's
cost and commission are deducted. Prices default to the last Yahoo close (not live): check the live quote before ordering.
Assumed, NOT verified against Scotia: --commission 4.99 CAD per trade.
"""
import argparse
import copy
import itertools
import re
import sys
from datetime import date
from pathlib import Path

DEFAULT_ACCOUNT = "scotia_tfsa"
DEFAULT_TARGETS = {"VFV.TO": 0.85, "XQQ.TO": 0.15}
DEFAULT_DRIFT_BAND = 0.03
DEFAULT_COMMISSION = 4.99
DEFAULT_MAX_COMMISSION_SHARE = 0.005
CASH_SYMBOL = "CASH.CAD"
ERROR_BUCKET = 0.002      # weight-error differences below this are a tie, and a tie goes to fewer trades
EPS = 1e-9


def parse_targets(text: str | None) -> dict[str, float]:
    """'VFV.TO:0.85,XQQ.TO:0.15' -> {...}. Default policy when empty. Weights must be positive and sum to 1."""
    if not text:
        return dict(DEFAULT_TARGETS)
    out = {}
    for part in text.split(","):
        sym, sep, w = part.partition(":")
        if not sep or not sym.strip():
            raise ValueError(f"bad target {part!r}; expected SYMBOL:WEIGHT")
        out[sym.strip().upper()] = float(w)
    validate_targets(out)
    return out


def validate_targets(targets: dict[str, float]) -> None:
    if not targets or any(w <= 0 for w in targets.values()):
        raise ValueError("targets need at least one symbol and every weight must be positive")
    if abs(sum(targets.values()) - 1.0) > 1e-6:
        raise ValueError(f"target weights must sum to 1.0 (got {sum(targets.values()):.4f})")


def parse_fills(text: str) -> list[dict]:
    """'VFV.TO:15@135.20,XQQ.TO:7@52.10' -> [{symbol, quantity (whole shares), price}]. Raises ValueError on anything else."""
    fills = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"([A-Za-z0-9.\-]+):(\d+)@(\d+(?:\.\d+)?)", part)
        if not m or int(m.group(2)) <= 0 or float(m.group(3)) <= 0:
            raise ValueError(f"bad fill {part!r}; expected SYMBOL:SHARES@PRICE, e.g. VFV.TO:15@135.20")
        fills.append({"symbol": m.group(1).upper(), "quantity": int(m.group(2)), "price": float(m.group(3))})
    if not fills:
        raise ValueError("no fills given")
    return fills


def sleeve_values(positions: list[dict], prices: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    """CAD market value of each target fund held (0 if not held). A held target with no price raises rather than counting as 0."""
    values = {s: 0.0 for s in targets}
    for p in positions:
        if p.get("kind") != "security" or p["symbol"] not in targets or not p["quantity"]:
            continue
        if p.get("currency", "CAD") != "CAD":
            raise ValueError(f"{p['symbol']} is held in {p['currency']}; the advisor expects CAD-listed funds")
        price = prices.get(p["symbol"])
        if price is None or price <= 0:
            raise ValueError(f"no price for held fund {p['symbol']}")
        values[p["symbol"]] += p["quantity"] * price
    return values


def other_holdings(positions: list[dict], targets: dict[str, float]) -> list[str]:
    return sorted(p["symbol"] for p in positions if p.get("kind") == "security" and p["symbol"] not in targets and p["quantity"])


def compute_drift(values: dict[str, float], targets: dict[str, float], band: float = DEFAULT_DRIFT_BAND) -> dict:
    """Weights of the invested sleeve vs target. A fund is breached when |weight - target| is strictly greater than `band`
    (exactly on the band is not a breach). An empty sleeve is an initial build: nothing has drifted."""
    total = sum(values.get(s, 0.0) for s in targets)
    if total <= 0:
        return {"total": 0.0, "empty": True, "weights": {s: 0.0 for s in targets},
                "drift": {s: -w for s, w in targets.items()}, "breached": []}
    weights = {s: values.get(s, 0.0) / total for s in targets}
    drift = {s: weights[s] - w for s, w in targets.items()}
    return {"total": total, "empty": False, "weights": weights, "drift": drift,
            "breached": [s for s, d in drift.items() if abs(d) > band + EPS]}


def allocate_cash_first(values: dict[str, float], targets: dict[str, float], cash: float) -> dict[str, float]:
    """Ideal CAD to put in each fund (no sells). Water-filling: raise the underweight funds toward target first; a fund that
    is already over its share of (sleeve + cash) gets nothing, and the cash is spread over the rest so that, relative to each
    other, they land on their target weights. The result sums to `cash` and is never negative."""
    out = {s: 0.0 for s in targets}
    if cash <= 0:
        return out
    active = set(targets)
    while True:
        wsum = sum(targets[s] for s in active)
        total = sum(values.get(s, 0.0) for s in active) + cash
        ideal = {s: targets[s] / wsum * total - values.get(s, 0.0) for s in active}
        over = [s for s in active if ideal[s] < -EPS]
        if not over:
            break
        active.discard(max(over, key=lambda s: values.get(s, 0.0) / targets[s]))   # most overweight leaves first
    for s in active:
        out[s] = max(ideal[s], 0.0)
    return out


def _weight_error(values: dict[str, float], targets: dict[str, float], cash_left: float) -> float:
    """L1 distance of the post-trade weights from target, with leftover cash counted in the total (idle cash is not free)."""
    total = sum(values.get(s, 0.0) for s in targets) + cash_left
    if total <= 0:
        return 2.0
    return sum(abs(values.get(s, 0.0) / total - w) for s, w in targets.items())


def plan_purchase(prices: dict[str, float], values: dict[str, float], targets: dict[str, float], cash: float,
                  commission: float = DEFAULT_COMMISSION, max_commission_share: float = DEFAULT_MAX_COMMISSION_SHARE,
                  band: float = DEFAULT_DRIFT_BAND) -> dict:
    """Best whole-share BUY for `cash`, never selling. The search is bounded by the cash-first dollar targets (at most one
    share past each fund's ideal amount), minimizes post-trade weight error, and treats near-ties as ties so an extra trade
    (an extra commission) is only taken when it buys a real improvement.

    recommendation: 'BUY'; 'WAIT_TOO_SMALL' (not even one whole share plus commission); 'WAIT_COMBINE' (commission is more than
    `max_commission_share` of the amount invested AND the sleeve is inside `band`: saving up and buying once costs less; a
    sleeve already beyond the band, or empty, buys regardless, since the purchase is what fixes it); 'NO_CASH'."""
    if any(prices.get(s, 0) <= 0 for s in targets):
        raise ValueError("need a positive price for every target symbol")
    ideal = allocate_cash_first(values, targets, cash)
    out = {"ideal_dollars": ideal, "shares": {}, "trades": 0, "invested": 0.0, "commission": 0.0, "cost": 0.0,
           "cash_left": max(cash, 0.0), "commission_share": None, "error_before": _weight_error(values, targets, max(cash, 0.0))}
    out["weights_after"] = compute_drift(values, targets)["weights"]
    out["error_after"] = out["error_before"]
    if cash <= 0:
        out["recommendation"] = "NO_CASH"
        return out
    syms = list(targets)
    limits = {}
    for s in syms:
        affordable = int((cash - commission) // prices[s]) if cash > commission else 0
        limits[s] = min(affordable, int(ideal[s] // prices[s]) + 1) if ideal[s] > EPS else 0
    best, best_key = None, None
    for counts in itertools.product(*(range(limits[s] + 1) for s in syms)):
        shares = dict(zip(syms, counts))
        trades = sum(1 for n in counts if n)
        invested = sum(n * prices[s] for s, n in shares.items())
        cost = invested + trades * commission
        if cost > cash + EPS:
            continue
        after = {s: values.get(s, 0.0) + shares[s] * prices[s] for s in syms}
        err = _weight_error(after, targets, cash - cost)
        key = (round(err / ERROR_BUCKET), trades, -invested)
        if best_key is None or key < best_key:
            best, best_key = (shares, trades, invested, cost, err, after), key
    assert best is not None      # the all-zero purchase always costs 0, so at least one candidate exists
    shares, trades, invested, cost, err, after = best
    out.update(shares={s: n for s, n in shares.items() if n}, trades=trades, invested=invested, commission=trades * commission,
               cost=cost, cash_left=cash - cost, error_after=err)
    out["weights_after"] = compute_drift(after, targets)["weights"]
    if not out["shares"]:
        wanted = [prices[s] for s in syms if ideal[s] > EPS]
        out["min_cash_for_one_share"] = min(wanted) + commission
        out["recommendation"] = "WAIT_TOO_SMALL"
    else:
        out["commission_share"] = out["commission"] / invested
        needs_fix = (drift_now := compute_drift(values, targets, band))["empty"] or bool(drift_now["breached"])
        if out["commission_share"] > max_commission_share and not needs_fix:
            out["cash_for_efficient_trade"] = commission / max_commission_share
            out["recommendation"] = "WAIT_COMBINE"
        else:
            out["recommendation"] = "BUY"
    return out


def cash_to_fix_without_selling(values: dict[str, float], targets: dict[str, float]) -> float:
    """New cash that would bring every fund exactly to target with no sells: the most overweight fund (value / weight) sets the
    size of the sleeve, so the answer is max(value_i / w_i) - total."""
    return max(max(values.get(s, 0.0) / w for s, w in targets.items()) - sum(values.get(s, 0.0) for s in targets), 0.0)


def apply_fills(positions: list[dict], fills: list[dict], commission: float = DEFAULT_COMMISSION, deposit: float = 0.0,
                targets: dict[str, float] | None = None) -> tuple[list[dict], float]:
    """Record executed buys. Returns (new positions, new CAD cash). `deposit` is added to cash first; each fill (one trade)
    then costs shares * price + commission. Cost basis includes the commission, like the broker's book cost. Raises ValueError
    (and records nothing) for a symbol outside the targets or if cash would go negative; the inputs are never mutated."""
    new = copy.deepcopy(positions)
    cash_row = next((p for p in new if p.get("kind") == "cash" and p["currency"] == "CAD"), None)
    if cash_row is None:
        cash_row = {"symbol": CASH_SYMBOL, "raw_symbol": "CASH", "quantity": 0.0, "currency": "CAD", "avg_cost": None, "kind": "cash"}
        new.append(cash_row)
    cash = cash_row["quantity"] + deposit
    for f in fills:
        if targets is not None and f["symbol"] not in targets:
            raise ValueError(f"{f['symbol']} is not one of the target funds {sorted(targets)}")
        cost = f["quantity"] * f["price"] + commission
        cash -= cost
        row = next((p for p in new if p.get("kind") == "security" and p["symbol"] == f["symbol"]), None)
        if row is None:
            new.append({"symbol": f["symbol"], "raw_symbol": f["symbol"].split(".")[0], "quantity": float(f["quantity"]),
                        "currency": "CAD", "avg_cost": cost / f["quantity"], "kind": "security"})
        else:
            if row.get("avg_cost") is not None or not row["quantity"]:
                row["avg_cost"] = ((row.get("avg_cost") or 0.0) * row["quantity"] + cost) / (row["quantity"] + f["quantity"])
            row["quantity"] += f["quantity"]
    if cash < -0.005:
        raise ValueError(f"cash would go to C${cash:,.2f}: the account only has C${cash_row['quantity'] + deposit:,.2f} "
                         f"to spend. Add the new money with --cash, or correct the fill.")
    cash_row["quantity"] = round(max(cash, 0.0), 2)
    return new, cash_row["quantity"]


def _order_lines(plan: dict, prices: dict[str, float]) -> list[str]:
    lines = []
    for s, n in plan["shares"].items():
        raw = s.split(".")[0]
        lines.append(f"  Scotia iTRADE (TFSA):  BUY {n} x {raw}   exchange TSX   limit near C${prices[s]:,.2f}   "
                     f"est. C${n * prices[s]:,.2f}")
    return lines


def format_report(account: str, prices: dict[str, float], positions: list[dict], values: dict[str, float],
                  targets: dict[str, float], band: float, cash: float, drift: dict, plan: dict | None, commission: float) -> str:
    L = [f"TFSA advisor | {account} | {date.today().isoformat()} | drift band +/-{band:.0%} | prices: last close, NOT live "
         f"({', '.join(f'{s} C${p:,.2f}' for s, p in prices.items())})"]
    held = {p["symbol"]: p["quantity"] for p in positions if p.get("kind") == "security"}
    L.append(f"{'fund':<9}{'shares':>8}{'value C$':>12}{'weight':>9}{'target':>8}{'drift':>8}")
    for s, w in targets.items():
        flag = "  <- BEYOND BAND" if s in drift["breached"] else ""
        L.append(f"{s:<9}{held.get(s, 0):>8,.4g}{values[s]:>12,.2f}{drift['weights'][s]:>9.1%}{w:>8.0%}{drift['drift'][s]:>+8.1%}{flag}")
    L.append(f"sleeve total C${drift['total']:,.2f}" + ("   (nothing held yet: initial build)" if drift["empty"] else ""))
    extra = other_holdings(positions, targets)
    if extra:
        L.append(f"NOTE: other holdings ignored by this advisor: {', '.join(extra)}")
    L.append("")
    if plan is None:
        if drift["empty"]:
            L.append("ACTION: NONE. No target fund is held yet. Run again with --cash to plan the initial build.")
        elif drift["breached"]:
            fix = cash_to_fix_without_selling(values, targets)
            L.append(f"ACTION: HOLD. Drift is beyond the band but there is no new cash. Adding about C${fix:,.0f} (bought into the "
                     f"underweight fund) would restore targets with no sell. Run again with --cash.")
        else:
            L.append("ACTION: NONE. Every fund is within the band; do not trade.")
        return "\n".join(L)
    rec = plan["recommendation"]
    L.append(f"New cash C${cash:,.2f}  |  commission assumed C${commission:.2f}/trade (UNVERIFIED for Scotia)  |  "
             f"cash-first target amounts: " + ", ".join(f"{s} C${v:,.0f}" for s, v in plan["ideal_dollars"].items()))
    if rec == "WAIT_TOO_SMALL":
        L.append(f"ACTION: WAIT. C${cash:,.2f} cannot buy one whole share plus commission; need about "
                 f"C${plan['min_cash_for_one_share']:,.0f}. Keep it as cash and add to it.")
        return "\n".join(L)
    L.append("ACTION: BUY" if rec == "BUY" else "ACTION: CONSIDER WAITING TO COMBINE (order below if you buy anyway)")
    L.extend(_order_lines(plan, prices))
    L.append(f"  invested C${plan['invested']:,.2f} + commission C${plan['commission']:,.2f} ({plan['trades']} trade(s)); "
             f"C${plan['cash_left']:,.2f} stays as cash (include it in next time's --cash)")
    after = compute_drift({s: values[s] + plan["shares"].get(s, 0) * prices[s] for s in targets}, targets, band)
    L.append("  after: " + ", ".join(f"{s} {after['weights'][s]:.1%} (target {targets[s]:.0%})" for s in targets)
             + ("" if not after["breached"] else f"   STILL BEYOND BAND: {', '.join(after['breached'])}"))
    if rec == "WAIT_COMBINE":
        L.append(f"  NOTE: commission is {plan['commission_share']:.2%} of the amount invested. About "
                 f"C${plan['cash_for_efficient_trade']:,.0f} per trade keeps it under the limit; savings from batching are real, "
                 f"drifting a little meanwhile is cheap.")
    fills = ",".join(f"{s}:{n}@<fill price>" for s, n in plan["shares"].items())
    L.append("")
    L.append(f"After you place the order, record it:  python -m swingtrade.advisor --confirm-fill {fills} --cash {cash:g}")
    return "\n".join(L)


def _load_holdings(args) -> list[dict]:
    if args.holdings_csv:
        from swingtrade import account_portfolio as ap
        return ap.parse_holdings_csv(Path(args.holdings_csv).read_text(encoding="utf-8-sig"), default_currency="CAD")
    import storage
    return storage.get_account_holdings(args.account)


def _save_holdings(account: str, positions: list[dict]) -> int:
    import storage
    storage.ensure_indexes()
    return storage.set_account_holdings(account, positions)


def _fetch_prices(symbols: list[str], overrides: dict[str, float]) -> dict[str, float]:
    prices = dict(overrides)
    missing = [s for s in symbols if s not in prices]
    if missing:
        import portfolio_view as pv
        prices.update(pv.fetch_prices(missing))
    return prices


def _parse_price_overrides(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items or []:
        sym, sep, px = it.partition("=")
        if not sep or float(px) <= 0:
            raise ValueError(f"bad --price {it!r}; expected SYMBOL=PRICE")
        out[sym.strip().upper()] = float(px)
    return out


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cash", type=float, default=0.0, help="NEW CAD cash to invest (default 0: drift check only)")
    ap_.add_argument("--confirm-fill", metavar="FILLS", help='record executed buys, e.g. "VFV.TO:15@135.20,XQQ.TO:7@52.10"')
    ap_.add_argument("--dry-run", action="store_true", help="with --confirm-fill: show the result without saving")
    ap_.add_argument("--commission", type=float, default=DEFAULT_COMMISSION, help="CAD per trade (default 4.99, unverified)")
    ap_.add_argument("--max-commission-pct", type=float, default=DEFAULT_MAX_COMMISSION_SHARE * 100)
    ap_.add_argument("--band-pct", type=float, default=DEFAULT_DRIFT_BAND * 100, help="drift tolerance in percentage points")
    ap_.add_argument("--targets", help='default "VFV.TO:0.85,XQQ.TO:0.15"')
    ap_.add_argument("--account", default=DEFAULT_ACCOUNT)
    ap_.add_argument("--holdings-csv", help="read holdings from a CSV instead of Mongo (advice only; cannot be combined with --confirm-fill)")
    ap_.add_argument("--price", action="append", metavar="SYMBOL=PRICE", help="use this CAD price instead of Yahoo (repeatable)")
    args = ap_.parse_args(argv)

    try:
        targets = parse_targets(args.targets)
        overrides = _parse_price_overrides(args.price)
        fills = parse_fills(args.confirm_fill) if args.confirm_fill else None
        if args.cash < 0 or args.commission < 0 or args.band_pct <= 0:
            raise ValueError("--cash and --commission must be >= 0 and --band-pct must be positive")
        if fills and args.holdings_csv:
            raise ValueError("--confirm-fill records into the saved account; do not combine it with --holdings-csv")
        positions = _load_holdings(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:    # Mongo not configured / unreachable, unreadable CSV
        print(f"[ERROR] could not load holdings: {exc}. Use --holdings-csv for an offline run.", file=sys.stderr)
        return 1

    if fills is not None:
        try:
            new_positions, cash_after = apply_fills(positions, fills, args.commission, args.cash, targets)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        spent = sum(f["quantity"] * f["price"] for f in fills) + args.commission * len(fills)
        print(f"Recorded {len(fills)} fill(s): spent C${spent:,.2f} incl. commission; cash now C${cash_after:,.2f}.")
        for p in new_positions:
            if p["kind"] == "security" and p["symbol"] in targets:
                print(f"  {p['symbol']}: {p['quantity']:g} shares, avg cost C${p['avg_cost']:,.4f}" if p.get("avg_cost")
                      else f"  {p['symbol']}: {p['quantity']:g} shares")
        if args.dry_run:
            print("DRY RUN: nothing saved.")
            return 0
        try:
            n = _save_holdings(args.account, new_positions)
        except Exception as exc:
            print(f"[ERROR] could not save: {exc}", file=sys.stderr)
            return 1
        print(f"Saved {n} position(s) to Account_Holdings (account {args.account!r}).")
        return 0

    held_targets = [p["symbol"] for p in positions if p.get("kind") == "security" and p["symbol"] in targets and p["quantity"]]
    wanted = sorted(set(held_targets) | (set(targets) if args.cash > 0 else set()))
    try:
        prices = _fetch_prices(wanted, overrides)
        missing = [s for s in wanted if s not in prices]
        if missing:
            raise ValueError(f"could not get a price for {', '.join(missing)}; pass --price SYMBOL=PRICE")
        values = sleeve_values(positions, prices, targets)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    band = args.band_pct / 100.0
    drift = compute_drift(values, targets, band)
    plan = None
    if args.cash > 0:
        plan = plan_purchase(prices, values, targets, args.cash, args.commission, args.max_commission_pct / 100.0, band)
    shown_prices = {s: prices[s] for s in targets if s in prices}
    print(format_report(args.account, shown_prices, positions, values, targets, band, args.cash, drift, plan, args.commission))
    return 0


if __name__ == "__main__":
    sys.exit(main())
