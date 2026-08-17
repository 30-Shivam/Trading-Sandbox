"""Watchlist ticker parsing. Shared by dip_buy_analyzer.py (Streamlit
sidebar default) and run_backtest.py (CLI) -- no yfinance/streamlit/pymongo
dependency, just text parsing.
"""

import json
import re
from pathlib import Path

# Real SPDR sector ETFs, one per GICS sector -- watchlist.txt's own sector
# labels (see read_ticker_sectors below) are the exact standard GICS names,
# so this mapping is complete for every ticker in the watchlist. Single
# source of truth for every consumer of sector relative-strength (see
# improvements.txt items 68/70) -- optimize.py, market_data.py, and
# benchmark_sector_relative_strength.py all import this instead of keeping
# their own copies.
SECTOR_ETF = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def read_tickers(path: Path) -> list[str]:
    """Read tickers from watchlist.txt. Supports either:
      - a JSON object/array with {"ticker": "..."} entries (or a "watchlist" key), or
      - a plain text file with one ticker per line.
    Whitespace is stripped and empty lines/entries are ignored.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            entries = data.get("watchlist", []) if isinstance(data, dict) else data
            tickers = []
            for entry in entries:
                if isinstance(entry, dict) and entry.get("ticker"):
                    tickers.append(str(entry["ticker"]).strip().upper())
                elif isinstance(entry, str) and entry.strip():
                    tickers.append(entry.strip().upper())
            if tickers:
                return tickers

    # Fallback: plain text, one ticker per line
    return [line.strip().upper() for line in raw.splitlines() if line.strip()]


def read_ticker_sectors(path: Path) -> dict[str, str]:
    """Read {ticker: sector} from the JSON watchlist form (the "sector" field
    on each entry, e.g. {"ticker": "NVDA", "sector": "Technology", ...}).
    Returns {} for a plain-text watchlist or any entry missing a sector --
    callers should treat a ticker with no sector as un-capped rather than
    erroring, since sector-based risk limits (see swingtrade.allocate_capital)
    are a best-effort feature, not a hard requirement of the watchlist format.
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    entries = data.get("watchlist", []) if isinstance(data, dict) else data
    sectors = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("ticker") and entry.get("sector"):
            sectors[str(entry["ticker"]).strip().upper()] = str(entry["sector"]).strip()
    return sectors


def parse_ticker_text(raw: str) -> list[str]:
    """Parse a free-form, user-pasted ticker list (newline and/or comma separated)."""
    if not raw or not raw.strip():
        return []
    seen = set()
    tickers = []
    for token in re.split(r"[\s,]+", raw.strip()):
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers
