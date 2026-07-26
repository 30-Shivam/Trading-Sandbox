"""Watchlist ticker parsing. Shared by dip_buy_analyzer.py (Streamlit
sidebar default) and run_backtest.py (CLI) -- no yfinance/streamlit/pymongo
dependency, just text parsing.
"""

import json
import re
from pathlib import Path


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
