"""Pure parsing helpers for the odd-lot tender screen (spec: docs/ODDLOT_SPEC.md). No network, no files."""
import re
from collections import Counter

import numpy as np

TENDER_FEE_USD = 25.0
ODD_LOT_SHARES = 99

_NUM = r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
_ODD = re.compile(r"odd[\s\-]*lots?", re.I)
_PRIORITY = re.compile(
    r"(?:fewer|less)\s+than\s+100\s+shares|(?:99|ninety[\s\-]nine)\s+(?:or\s+fewer\s+)?shares|"
    r"not\s+more\s+than\s+99|(?:before|prior\s+to)\s+(?:any\s+)?proration|without\s+(?:any\s+)?proration|"
    r"not\s+(?:be\s+)?subject\s+to\s+proration|priority", re.I)
_NAV = re.compile(
    r"\d+(?:\.\d+)?\s*%\s+of\s+(?:the\s+)?(?:fund[’']?s\s+|its\s+)?(?:net\s+asset\s+value|nav)|"
    r"equal\s+to\s+(?:\d+(?:\.\d+)?\s*%\s+of\s+)?(?:the\s+)?(?:net\s+asset\s+value|nav)\s+per\s+share", re.I)
_RANGES = [
    re.compile(r"not\s+(?:greater|more|higher)\s+than\s+" + _NUM + r"\s*(?:per\s+share\s*)?(?:nor|and)\s+(?:not\s+)?(?:less|lower)\s+than\s+" + _NUM, re.I),
    re.compile(r"not\s+(?:less|lower)\s+than\s+" + _NUM + r"\s*(?:per\s+share\s*)?(?:nor|and)\s+(?:not\s+)?(?:greater|more|higher)\s+than\s+" + _NUM, re.I),
    re.compile(r"(?:between|from)\s+" + _NUM + r"\s*(?:per\s+share\s*)?(?:and|to)\s+" + _NUM + r"\s+per\s+share", re.I),
]
_FIXED = [
    re.compile(r"(?:purchase\s+price|price)\s+of\s+" + _NUM + r"\s+per\s+share", re.I),
    re.compile(_NUM + r"\s+per\s+share\s+(?:in\s+)?(?:net\s+to\s+the\s+seller\s+)?in\s+cash", re.I),
    re.compile(r"at\s+a\s+(?:cash\s+)?price\s+of\s+" + _NUM, re.I),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("’", "'").replace("“", '"').replace("”", '"'))


_IDX_ROW = re.compile(r"^(\S+(?: \S+)?)\s+(.*?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(edgar/\S+)\s*$")


def parse_form_idx_line(line: str) -> dict | None:
    """EDGAR form.idx row -> dict, or None for non-data lines. Regex, not fixed offsets: data rows are not
    column-aligned with the header (verified on 2023 Q1)."""
    m = _IDX_ROW.match(line)
    if not m:
        return None
    form, name, cik, date, fname = m.groups()
    return {"form": form, "company": name, "cik": int(cik), "filed": date, "path": fname,
            "accession": fname.rsplit("/", 1)[-1].removesuffix(".txt")}


def detect_odd_lot(text: str) -> bool:
    """Odd-lot mention within 400 chars of a priority / non-proration phrase."""
    for m in _ODD.finditer(text):
        window = text[max(0, m.start() - 400): m.end() + 400]
        if _PRIORITY.search(window):
            return True
    return False


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def extract_price_terms(text: str) -> dict:
    """{"type": nav|range|fixed|unknown, "price": float|None (range -> LOW end), "high": float|None}."""
    if _NAV.search(text):
        return {"type": "nav", "price": None, "high": None}
    for rx in _RANGES:
        m = rx.search(text)
        if m:
            a, b = _f(m.group(1)), _f(m.group(2))
            lo, hi = min(a, b), max(a, b)
            if lo > 0 and lo != hi:
                return {"type": "range", "price": lo, "high": hi}
    prices = [_f(m.group(1)) for rx in _FIXED for m in rx.finditer(text)]
    prices = [p for p in prices if p > 0]
    if prices:
        return {"type": "fixed", "price": Counter(prices).most_common(1)[0][0], "high": None}
    return {"type": "unknown", "price": None, "high": None}


_STATED = re.compile(
    r"(?:last\s+(?:reported\s+)?(?:sale|sales|trade)\s+price|closing\s+(?:sale\s+)?price|closing\s+price)[^$]{0,160}?" + _NUM, re.I)
_CAD = re.compile(r"(?<![A-Za-z0-9])(?:C|Cdn\.?|CAD|CDN)\s?\$\s?\d|Canadian\s+dollars", re.I)


def stated_market_price(text: str) -> float | None:
    """First 'last reported sale / closing price ... $X' amount the filing itself states (independent of Yahoo)."""
    m = _STATED.search(text)
    return _f(m.group(1)) if m else None


def is_cad(text: str) -> bool:
    return len(_CAD.findall(text)) >= 3


def gross_spread_pct(offer: float, entry: float) -> float:
    return (offer - entry) / entry * 100.0


def net_payoff_usd(offer: float, entry: float, shares: int = ODD_LOT_SHARES, fee: float = TENDER_FEE_USD) -> float:
    return shares * (offer - entry) - fee


def summarize(events: list[dict], full_years=range(2021, 2026), min_per_year=12, min_median_net=75.0) -> dict:
    """events: qualifying events with keys year, spread_pct, net_usd. Applies the frozen kill rules."""
    per_year = {y: sum(1 for e in events if e["year"] == y) for y in sorted({e["year"] for e in events} | set(full_years))}
    yearly = [per_year.get(y, 0) for y in full_years]
    median_count = float(np.median(yearly)) if yearly else 0.0
    nets = [e["net_usd"] for e in events]
    median_net = float(np.median(nets)) if nets else float("nan")
    rule1_fail = median_count < min_per_year
    rule2_fail = not nets or median_net < min_median_net
    return {"per_year": per_year, "median_yearly_count": median_count, "median_net_usd": median_net,
            "median_spread_pct": float(np.median([e["spread_pct"] for e in events])) if events else float("nan"),
            "rule1_fail": rule1_fail, "rule2_fail": rule2_fail, "passed": not (rule1_fail or rule2_fail)}
