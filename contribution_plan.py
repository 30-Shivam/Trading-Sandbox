"""What to buy with a new CAD contribution (option A: keep existing holdings, steer ALL new money into the index sleeve).
READ-ONLY: it prints a plan; you place the orders yourself in your broker. Never talks to a broker.

DEFAULT (CAD funds): delegates to swingtrade.advisor, i.e. whole shares of VFV.TO / XQQ.TO toward 85/15, cash-first, no
currency conversion, so no FX cost. Same as `python -m swingtrade.advisor --cash N`.

--us-listed: the original US-ticker path. Reads your saved account, live prices and USD/CAD from Yahoo, and picks whole
shares of SPY/QQQ so the sleeve lands closest to 85/15, after commission and the CAD->USD conversion spread.

Usage:
    python contribution_plan.py --cad 2500
    python contribution_plan.py --cad 2500 --us-listed --usd-cash 120 --fx-spread-pct 1.0 --commission 4.99
Assumptions you should replace with your broker's real numbers: --commission (default 4.99 per trade) and, for --us-listed,
--fx-spread-pct (default 1.5, UNVERIFIED for Scotia). Shares are whole; leftover USD stays in the USD account.
"""
import argparse
import sys

from swingtrade.contribution import plan_contribution

DEFAULT_ACCOUNT = "scotia_tfsa"
DEFAULT_TARGETS = {"SPY": 0.85, "QQQ": 0.15}


def format_plan(plan: dict, cash_cad: float, usdcad: float, prices: dict[str, float], fx_spread: float, sleeve: dict) -> str:
    L = [f"Contribution C${cash_cad:,.2f}  |  USD/CAD {usdcad:.4f}  |  assumed FX spread {fx_spread:.2%}  |  "
         f"prices " + ", ".join(f"{s} US${p:,.2f}" for s, p in prices.items())]
    rec = plan["recommendation"]
    if rec == "WAIT_TOO_SMALL":
        L.append(f"RECOMMENDATION: WAIT. This cannot buy one whole share after commission. You need about "
                 f"C${plan['min_cash_cad_for_one_share']:,.0f} for the cheapest fund. Keep it as cash and add to it.")
        return "\n".join(L)
    verb = {"BUY": "BUY", "WAIT_COMBINE": "CONSIDER WAITING / COMBINING"}[rec]
    L.append(f"RECOMMENDATION: {verb}")
    for s, n in plan["shares"].items():
        L.append(f"  buy {n} x {s} @ ~US${prices[s]:,.2f} = US${n * prices[s]:,.2f}")
    L.append(f"  convert C${plan['cad_converted']:,.2f} -> US${plan['usd_from_cad']:,.2f} "
             f"(spread cost ~C${plan['fx_cost_cad']:,.2f}); {plan['trades']} trade(s), commissions US${plan['commission_usd']:.2f}")
    L.append(f"  total friction C${plan['friction_cad']:,.2f} = {plan['friction_share_of_invested']:.2%} of the amount invested "
             f"(commissions alone {plan['commission_share']:.2%})")
    L.append(f"  left over: C${plan['cad_left']:,.2f} cash, US${plan['usd_left']:,.2f} in the USD account")
    L.append("  index sleeve after: " + ", ".join(f"{s} {w:.1%} (target {DEFAULT_TARGETS.get(s, 0):.0%})"
                                                for s, w in plan["weights_after"].items()))
    if rec == "WAIT_COMBINE":
        L.append(f"  NOTE: one commission is large relative to this order. Combining into a larger purchase "
                 f"(roughly C${plan['cash_for_efficient_trade_cad']:,.0f}+ per trade) cuts the commission share below the limit. "
                 f"The currency spread is a percentage and does not shrink by batching.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cad", type=float, required=True, help="new CAD cash to invest")
    ap.add_argument("--us-listed", action="store_true",
                    help="buy US-listed SPY/QQQ (CAD->USD conversion) instead of the CAD-listed VFV.TO/XQQ.TO default")
    ap.add_argument("--usd-cash", type=float, default=None, help="USD cash already in the account (--us-listed only)")
    ap.add_argument("--fx-spread-pct", type=float, default=None, help="--us-listed only; default 1.5")
    ap.add_argument("--commission", type=float, default=4.99)
    ap.add_argument("--max-commission-pct", type=float, default=0.5)
    ap.add_argument("--account", default=DEFAULT_ACCOUNT)
    ap.add_argument("--targets", help='default "VFV.TO:0.85,XQQ.TO:0.15" (or SPY/QQQ with --us-listed); weights WITHIN the sleeve')
    args = ap.parse_args(argv)

    if not args.us_listed:
        if args.usd_cash is not None or args.fx_spread_pct is not None:
            print("[ERROR] --usd-cash and --fx-spread-pct only apply with --us-listed (CAD funds have no FX step).", file=sys.stderr)
            return 1
        from swingtrade import advisor
        fwd = ["--cash", str(args.cad), "--commission", str(args.commission), "--max-commission-pct",
               str(args.max_commission_pct), "--account", args.account]
        return advisor.main(fwd + (["--targets", args.targets] if args.targets else []))

    args.usd_cash = args.usd_cash or 0.0
    args.fx_spread_pct = 1.5 if args.fx_spread_pct is None else args.fx_spread_pct
    targets = DEFAULT_TARGETS if not args.targets else {
        p.partition(":")[0].strip().upper(): float(p.partition(":")[2]) for p in args.targets.split(",")}

    import portfolio_view as pv
    import storage
    try:
        held = storage.get_account_holdings(args.account)
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr); return 1
    usdcad = pv.last_price(pv.FX_SYMBOL)
    prices = pv.fetch_prices(list(targets))
    if usdcad is None or set(prices) != set(targets):
        print("[ERROR] could not fetch USD/CAD and prices for " + ", ".join(targets) + "; try again shortly.", file=sys.stderr)
        return 1
    sleeve = {s: sum(h["quantity"] for h in held if h["symbol"] == s) * prices[s] for s in targets}
    plan = plan_contribution(args.cad, prices, usdcad, sleeve, targets, cash_usd=args.usd_cash,
                             commission_usd=args.commission, fx_spread=args.fx_spread_pct / 100.0,
                             max_commission_share=args.max_commission_pct / 100.0)
    print(format_plan(plan, args.cad, usdcad, prices, args.fx_spread_pct / 100.0, sleeve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
