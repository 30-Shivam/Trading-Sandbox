"""Single source of truth for the tradable ticker universe.

Replaces hand-curated lists (watchlist.txt was picked from today's winners --
NVDA is tagged "Growth/AI" -- and smallmid_watchlist.txt/adr_watchlist.txt
were separate files with separate loaders) with a rules-based universe:

  1. Constituents of the S&P 500 / 400 / 600 (large / mid / small cap).
  2. Liquidity filter: last close >= MIN_PRICE and median 60-day dollar
     volume >= MIN_DOLLAR_VOLUME, so every name is realistically tradable.
  3. Every dropped name is recorded with a reason, never silently lost.

Output is `universe.json`, written in the SAME shape watchlist.py already
parses ({"watchlist": [{"ticker", "sector", "cap", ...}]}), so
`ingest.py --watchlist universe.json`, `watchlist.read_tickers()` and
`watchlist.read_ticker_sectors()` consume it with no changes.

Switching the live scan to this universe is deliberately opt-in: changing
the universe mid-stream changes every strategy's prospective track record
(ic_tracking trust floors count calendar time and trades on the old list).

Usage:
    python universe.py build [--out universe.json]
    python universe.py show  [--caps large,mid]
"""

import argparse
import io
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_UNIVERSE_FILE = SCRIPT_DIR / "universe.json"

CAP_SOURCES = {
    "large": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "mid": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "small": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 3_000_000.0  # median daily $ volume over the lookback
LIQUIDITY_LOOKBACK_DAYS = 60
DOWNLOAD_CHUNK = 100
# Below this share of requested tickers actually analyzed, a run is "degraded".
MIN_COVERAGE = 0.90

# Wikipedia's GICS names -> this project's sector vocabulary (watchlist.SECTOR_ETF).
SECTOR_NORMALIZE = {"Health Care": "Healthcare", "Information Technology": "Technology"}


def normalize_ticker(raw: str) -> str:
    """Wikipedia writes share classes as BRK.B; yfinance wants BRK-B."""
    return str(raw).strip().upper().replace(".", "-")


def parse_constituents(table: pd.DataFrame, cap: str) -> list[dict]:
    """Rows from a Wikipedia constituents table. Column names differ slightly
    between the three index pages (Symbol/Ticker symbol, GICS Sector)."""
    symbol_col = next((c for c in table.columns if str(c).lower() in ("symbol", "ticker symbol", "ticker")), None)
    sector_col = next((c for c in table.columns if "gics sector" in str(c).lower()), None)
    if symbol_col is None:
        raise ValueError(f"no symbol column in {list(table.columns)}")
    rows = []
    for _, r in table.iterrows():
        ticker = normalize_ticker(r[symbol_col])
        if not re.fullmatch(r"[A-Z0-9\-]{1,7}", ticker):
            continue
        sector = str(r[sector_col]).strip() if sector_col is not None else ""
        rows.append({"ticker": ticker, "sector": SECTOR_NORMALIZE.get(sector, sector), "cap": cap})
    return rows


def fetch_constituents() -> list[dict]:
    """All three indices, de-duplicated (a ticker keeps its largest cap tag)."""
    seen: dict[str, dict] = {}
    for cap, url in CAP_SOURCES.items():  # large first, so large wins on overlap
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        table = pd.read_html(io.StringIO(html))[0]
        for row in parse_constituents(table, cap):
            seen.setdefault(row["ticker"], row)
    return list(seen.values())


def liquidity_verdict(close: pd.Series, volume: pd.Series) -> tuple[bool, str]:
    """(keep, reason) for one ticker's recent daily Close/Volume."""
    close, volume = close.dropna(), volume.dropna()
    if close.empty or volume.empty:
        return False, "no_data"
    if float(close.iloc[-1]) < MIN_PRICE:
        return False, "low_price"
    dollar_volume = (close * volume).dropna().tail(LIQUIDITY_LOOKBACK_DAYS)
    if dollar_volume.empty or float(dollar_volume.median()) < MIN_DOLLAR_VOLUME:
        return False, "low_dollar_volume"
    return True, "ok"


def _download_chunk(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    return yf.download(
        tickers, period="6mo", interval="1d", auto_adjust=True, group_by="ticker",
        progress=False, threads=True, timeout=30,
    )


def apply_liquidity_filter(members: list[dict], downloader=_download_chunk) -> tuple[list[dict], list[dict]]:
    """Returns (kept, dropped); each dropped row carries a `reason`."""
    kept, dropped = [], []
    for i in range(0, len(members), DOWNLOAD_CHUNK):
        chunk = members[i:i + DOWNLOAD_CHUNK]
        tickers = [m["ticker"] for m in chunk]
        try:
            data = downloader(tickers)
        except Exception as exc:
            dropped.extend({**m, "reason": f"download_error: {exc}"} for m in chunk)
            continue
        for m in chunk:
            try:
                frame = data[m["ticker"]] if len(tickers) > 1 else data
                ok, reason = liquidity_verdict(frame["Close"], frame["Volume"])
            except (KeyError, TypeError):
                ok, reason = False, "no_data"
            (kept if ok else dropped).append(m if ok else {**m, "reason": reason})
    return kept, dropped


def build_universe(members: list[dict] | None = None, downloader=_download_chunk) -> dict:
    members = members if members is not None else fetch_constituents()
    kept, dropped = apply_liquidity_filter(members, downloader)
    kept.sort(key=lambda m: m["ticker"])
    return {
        "generated": date.today().isoformat(),
        "rules": {
            "indices": list(CAP_SOURCES), "min_price": MIN_PRICE,
            "min_median_dollar_volume": MIN_DOLLAR_VOLUME, "lookback_days": LIQUIDITY_LOOKBACK_DAYS,
        },
        "watchlist": kept,
        "dropped": sorted(dropped, key=lambda m: m["ticker"]),
    }


def write_universe(universe: dict, path: Path = DEFAULT_UNIVERSE_FILE) -> None:
    path.write_text(json.dumps(universe, indent=1) + "\n", encoding="utf-8")


def load_universe(path: Path = DEFAULT_UNIVERSE_FILE, caps: set[str] | None = None) -> list[dict]:
    """Members (ticker/sector/cap), optionally restricted to cap buckets."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("watchlist", [])
    return [e for e in entries if caps is None or e.get("cap") in caps]


EXCLUDED_BY_FILTER_PREFIX = "excluded:"   # reason prefix used by swingtrade.levels for deliberate filter rejections


def coverage_summary(requested: int, skipped: list[tuple[str, str]]) -> dict:
    """How much of the requested universe was actually analyzed, and why the
    rest wasn't. Fetch failures were previously a long, unsummarized list at
    the bottom of the log (a 38% loss on the small/mid scan went unnoticed)."""
    reasons: dict[str, int] = {}
    excluded = 0
    for _, reason in skipped:
        text = str(reason).lower()
        if text.startswith(EXCLUDED_BY_FILTER_PREFIX):
            # A strategy filter (macro downtrend, illiquid) rejected a ticker whose data
            # fetched and scored fine -- 2026-10-24: 172/407 such exclusions on a weak-market
            # day made every run read DEGRADED. Tracked separately, not as lost coverage.
            excluded += 1
            continue
        if "delist" in text or "no price data" in text or "no data" in text:
            key = "no_data_or_delisted"
        elif "timeout" in text or "timed out" in text:
            key = "timeout"
        elif "rate" in text or "429" in text:
            key = "rate_limited"
        else:
            key = "other"
        reasons[key] = reasons.get(key, 0) + 1
    failed = len(skipped) - excluded
    analyzed = max(requested - failed, 0)
    ratio = analyzed / requested if requested else 0.0
    return {
        "requested": requested, "analyzed": analyzed, "skipped": failed, "excluded": excluded,
        "coverage": ratio, "degraded": ratio < MIN_COVERAGE, "reasons": reasons,
    }


def format_coverage_line(summary: dict) -> str:
    """Stable one-liner; daily_run.py greps for the leading `COVERAGE` token."""
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(summary["reasons"].items())) or "none"
    flag = " DEGRADED" if summary["degraded"] else ""
    excluded = f" excluded_by_filter={summary['excluded']}" if summary.get("excluded") else ""
    return (f"COVERAGE {summary['analyzed']}/{summary['requested']} "
            f"({summary['coverage']:.1%}){flag} skipped: {reasons}{excluded}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the rules-based ticker universe.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Fetch S&P 500/400/600, liquidity-filter, write universe.json")
    build.add_argument("--out", type=Path, default=DEFAULT_UNIVERSE_FILE)
    show = sub.add_parser("show", help="Summarize an existing universe.json")
    show.add_argument("--path", type=Path, default=DEFAULT_UNIVERSE_FILE)
    show.add_argument("--caps", default="", help="comma list of large,mid,small")
    args = parser.parse_args()

    if args.cmd == "build":
        universe = build_universe()
        write_universe(universe, args.out)
        reasons: dict[str, int] = {}
        for m in universe["dropped"]:
            reasons[m["reason"]] = reasons.get(m["reason"], 0) + 1
        print(f"Wrote {len(universe['watchlist'])} tickers to {args.out}; dropped {len(universe['dropped'])} {reasons}")
        return 0

    caps = {c for c in args.caps.split(",") if c} or None
    members = load_universe(args.path, caps)
    frame = pd.DataFrame(members)
    print(f"{len(members)} tickers")
    if not frame.empty:
        print(frame.groupby(["cap", "sector"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
