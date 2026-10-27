"""See a real brokerage account (e.g. the Scotia iTRADE TFSA): holdings valued in CAD and USD, P&L vs your cost, currency
split, and weights vs the frozen 85% SPY / 15% QQQ target. READ-ONLY: this never places an order and never talks to a broker.
You supply the holdings (a CSV you export or type); prices and USD/CAD come from Yahoo.

Usage:
    python portfolio_view.py --csv private/scotia_tfsa.csv                  # view only, no database needed
    python portfolio_view.py --csv private/scotia_tfsa.csv --save           # also store it (Account_Holdings, account 'scotia_tfsa')
    python portfolio_view.py --account scotia_tfsa                          # view what was saved
    python portfolio_view.py --csv ... --default-currency CAD               # rows with no Currency column value

CSV: a header row with Symbol and Quantity (aliases accepted: Ticker, Shares, Units, ...) plus optional Currency, Average Cost
(or Book Cost), Exchange. A row whose symbol is CASH is treated as a cash balance. See docs/holdings_template.csv.
Keep the real file under private/ (gitignored). CAD securities with no exchange suffix are assumed TSX (.TO).
"""
import argparse
import sys
from pathlib import Path

from swingtrade import account_portfolio as ap

DEFAULT_ACCOUNT = "scotia_tfsa"
DEFAULT_TARGETS = {"SPY": 0.85, "QQQ": 0.15}
FX_SYMBOL = "USDCAD=X"


def parse_targets(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_TARGETS)
    return {p.partition(":")[0].strip().upper(): float(p.partition(":")[2]) for p in text.split(",")}


def last_price(symbol: str) -> float | None:
    """Most recent close from Yahoo (retrying), or None if unavailable. Network."""
    import yfinance as yf
    import market_data
    try:
        hist = market_data._with_retry(lambda: yf.Ticker(symbol).history(period="7d", auto_adjust=False))
        close = hist["Close"].dropna()
        return float(close.iloc[-1]) if len(close) else None
    except Exception:
        return None


def fetch_prices(symbols: list[str], fetch=None) -> dict[str, float]:
    fetch = fetch or last_price   # resolved at call time so tests (and callers) can patch module-level last_price
    return {s: p for s in symbols if (p := fetch(s)) is not None}


def format_report(v: dict, account: str) -> str:
    L = [f"{account}: total C${v['total_cad']:,.2f}  =  US${v['total_usd']:,.2f}   (USD/CAD {v['usdcad']:.4f}; "
         f"{v['cad_denominated_share']:.0%} of value is CAD-denominated)"]
    L.append(f"{'symbol':<12}{'qty':>10}{'cur':>5}{'price':>11}{'value C$':>13}{'value US$':>12}{'weight':>8}{'P&L %':>9}")
    for r in v["rows"]:
        pnl = f"{r['pnl_pct']:+.1f}" if "pnl_pct" in r else ""
        L.append(f"{r['symbol']:<12}{r['quantity']:>10,.4g}{r['currency']:>5}{r['price']:>11,.2f}{r['value_cad']:>13,.2f}"
                 f"{r['value_usd']:>12,.2f}{r['weight']:>8.1%}{pnl:>9}")
    if v["unpriced"]:
        L.append(f"NOT VALUED (no price found, excluded from totals): {', '.join(v['unpriced'])}")
    c = v.get("core")
    if c:
        L.append("")
        L.append("vs frozen policy (weights are % of the WHOLE account, cash included):")
        for sym, tgt in c["targets"].items():
            L.append(f"  {sym:<6} target {tgt:.0%}  actual {c['actual_weight'][sym]:.1%}  drift {c['actual_weight'][sym] - tgt:+.1%}")
        L.append(f"  other holdings {c['other_holdings_weight']:.1%} | cash {c['cash_weight']:.1%}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--csv", type=Path); ap_.add_argument("--account", default=DEFAULT_ACCOUNT)
    ap_.add_argument("--save", action="store_true", help="store the parsed CSV in Mongo (replaces ONLY this account)")
    ap_.add_argument("--default-currency", default="USD", choices=["USD", "CAD"])
    ap_.add_argument("--targets", help='default "SPY:0.85,QQQ:0.15"')
    args = ap_.parse_args(argv)

    if args.csv:
        try:
            positions = ap.parse_holdings_csv(args.csv.read_text(encoding="utf-8-sig"), args.default_currency)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
    else:
        import storage
        try:
            positions = storage.get_account_holdings(args.account)
        except storage.MongoNotConfigured as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        if not positions:
            print(f"No holdings saved for account {args.account!r}. Use --csv ... --save first.")
            return 1
    if not positions:
        print("[ERROR] no holdings rows parsed from the CSV.", file=sys.stderr)
        return 1

    fx = last_price(FX_SYMBOL)
    if fx is None:
        print("[ERROR] could not fetch USD/CAD from Yahoo; try again shortly.", file=sys.stderr)
        return 1
    prices = fetch_prices([p["symbol"] for p in positions if p["kind"] == "security"])
    print(format_report(ap.value_portfolio(positions, prices, fx, parse_targets(args.targets)), args.account))

    if args.csv and args.save:
        import storage
        try:
            storage.ensure_indexes()
            n = storage.set_account_holdings(args.account, positions)
        except storage.MongoNotConfigured as exc:
            print(f"[ERROR] could not save: {exc}", file=sys.stderr)
            return 1
        print(f"\nSaved {n} position(s) to Account_Holdings (account {args.account!r}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
