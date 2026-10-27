"""Pure helpers for a real brokerage account's holdings (e.g. a Scotia iTRADE TFSA): CSV parsing, Yahoo symbol
normalization, and valuation in both CAD and USD. No network, no files, no Mongo -- prices and the FX rate are passed in.

Position dict: {"symbol": yahoo symbol, "raw_symbol": as exported, "quantity": float, "currency": "CAD"|"USD",
"avg_cost": per-share native-currency cost or None, "kind": "security"|"cash"}.
"""
import csv
import io
import re

CANADIAN_SUFFIXES = (".TO", ".V", ".NE", ".CN")
_ALIASES = {
    "symbol": ("symbol", "ticker", "security", "security symbol", "stock symbol"),
    "quantity": ("quantity", "qty", "shares", "units", "position", "qty held"),
    "currency": ("currency", "curr", "ccy"),
    "avg_cost": ("average cost", "avg cost", "avg price", "average price", "cost per share", "avg_cost", "unit cost"),
    "book_cost": ("book cost", "book value", "total cost", "cost basis", "book_cost"),
    "exchange": ("exchange", "market", "listing"),
}
_EXCHANGE_SUFFIX = {"TSX": ".TO", "TORONTO": ".TO", "TSXV": ".V", "TSX VENTURE": ".V", "NEO": ".NE", "CBOE CANADA": ".NE",
                    "CSE": ".CN"}
_CASH_WORDS = ("cash", "cad cash", "usd cash", "cash balance", "sweep", "money market")


def parse_number(text) -> float | None:
    """'$1,234.56' -> 1234.56; '(12.5)' -> -12.5; '' / '--' / None -> None."""
    if text is None:
        return None
    s = str(text).strip().replace("$", "").replace(",", "").replace(" ", "")
    if s in ("", "-", "--", "n/a", "N/A"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def normalize_symbol(symbol: str, currency: str = "USD", exchange: str | None = None) -> str:
    """Yahoo symbol. Keeps an explicit Canadian suffix; maps an exchange column to a suffix; otherwise a CAD-denominated
    security with no suffix is assumed to be TSX-listed (.TO). Class-share dots become dashes (BRK.B -> BRK-B)."""
    s = symbol.strip().upper().replace(" ", "")
    for suf in CANADIAN_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)].replace(".", "-") + suf
    suffix = _EXCHANGE_SUFFIX.get((exchange or "").strip().upper())
    if suffix is None and currency.upper() == "CAD":
        suffix = ".TO"
    return s.replace(".", "-") + (suffix or "")


def _find_columns(header: list[str]) -> dict[str, int]:
    norm = [h.strip().lower() for h in header]
    cols = {}
    for key, names in _ALIASES.items():
        for i, h in enumerate(norm):
            if h in names:
                cols[key] = i
                break
    return cols


def parse_holdings_csv(text: str, default_currency: str = "USD") -> list[dict]:
    """Flexible parser for a broker export / hand-typed CSV. Needs a header row naming at least a symbol and a quantity
    column (aliases in _ALIASES). Cash rows (symbol like 'CASH'/'CAD CASH') become kind='cash' with quantity = balance.
    Rows without a symbol, or with total/subtotal wording, are skipped. Raises ValueError if the header can't be found."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    header_idx = next((i for i, r in enumerate(rows) if {"symbol", "quantity"} <= set(_find_columns(r))), None)
    if header_idx is None:
        raise ValueError("no header row with a symbol column and a quantity column was found")
    cols = _find_columns(rows[header_idx])
    out = []
    for r in rows[header_idx + 1:]:
        if len(r) <= max(cols.values()):
            continue
        raw = r[cols["symbol"]].strip()
        if not raw or re.match(r"^(total|subtotal|grand total)", raw, re.I):
            continue
        qty = parse_number(r[cols["quantity"]])
        if qty is None:
            continue
        currency = (r[cols["currency"]].strip().upper() if "currency" in cols and r[cols["currency"]].strip() else default_currency.upper())
        if currency not in ("CAD", "USD"):
            raise ValueError(f"unsupported currency {currency!r} for {raw!r} (only CAD and USD)")
        if raw.lower() in _CASH_WORDS:
            out.append({"symbol": f"CASH.{currency}", "raw_symbol": raw, "quantity": qty, "currency": currency,
                        "avg_cost": None, "kind": "cash"})
            continue
        exchange = r[cols["exchange"]] if "exchange" in cols else None
        avg = parse_number(r[cols["avg_cost"]]) if "avg_cost" in cols else None
        if avg is None and "book_cost" in cols:
            book = parse_number(r[cols["book_cost"]])
            avg = book / qty if book is not None and qty else None
        out.append({"symbol": normalize_symbol(raw, currency, exchange), "raw_symbol": raw, "quantity": qty,
                    "currency": currency, "avg_cost": avg, "kind": "security"})
    return out


def value_portfolio(positions: list[dict], prices: dict[str, float], usdcad: float,
                    core_targets: dict[str, float] | None = None) -> dict:
    """Value positions in CAD and USD. `usdcad` = CAD per 1 USD. `prices` = native-currency price per symbol.
    Securities with no price are listed under 'unpriced' and excluded from totals (never valued at 0 silently).
    If core_targets (e.g. {'SPY': .85, 'QQQ': .15}) is given, reports each target's actual weight of the TOTAL account and
    how much sits in non-target holdings and cash."""
    if usdcad <= 0:
        raise ValueError("usdcad must be positive")
    rows, unpriced = [], []
    for p in positions:
        if p["kind"] == "cash":
            price = 1.0
        else:
            price = prices.get(p["symbol"])
        if price is None or price != price:
            unpriced.append(p["symbol"])
            continue
        mv = p["quantity"] * price
        mv_cad = mv if p["currency"] == "CAD" else mv * usdcad
        row = {"symbol": p["symbol"], "kind": p["kind"], "quantity": p["quantity"], "currency": p["currency"],
               "price": price, "market_value": mv, "value_cad": mv_cad, "value_usd": mv_cad / usdcad,
               "avg_cost": p.get("avg_cost")}
        if p.get("avg_cost") and p["kind"] == "security":
            row["pnl_pct"] = (price / p["avg_cost"] - 1.0) * 100.0
            row["pnl_native"] = (price - p["avg_cost"]) * p["quantity"]
        rows.append(row)
    total_cad = sum(r["value_cad"] for r in rows)
    for r in rows:
        r["weight"] = r["value_cad"] / total_cad if total_cad else 0.0
    cad_exposure = sum(r["value_cad"] for r in rows if r["currency"] == "CAD")
    out = {"rows": sorted(rows, key=lambda r: -r["value_cad"]), "unpriced": unpriced, "total_cad": total_cad,
           "total_usd": total_cad / usdcad, "usdcad": usdcad,
           "cash_cad": sum(r["value_cad"] for r in rows if r["kind"] == "cash"),
           "cad_denominated_share": cad_exposure / total_cad if total_cad else 0.0}
    if core_targets:
        # target symbols may appear under US (SPY) or any suffix; match on the exact symbol only (no guessing proxies)
        by_symbol = {r["symbol"]: r["value_cad"] for r in rows}
        actual = {s: (by_symbol.get(s, 0.0) / total_cad if total_cad else 0.0) for s in core_targets}
        in_targets = sum(actual.values())
        cash_w = out["cash_cad"] / total_cad if total_cad else 0.0
        out["core"] = {"targets": core_targets, "actual_weight": actual, "other_holdings_weight": max(1.0 - in_targets - cash_w, 0.0),
                       "cash_weight": cash_w}
    return out
