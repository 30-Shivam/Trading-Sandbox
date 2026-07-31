"""Point-in-time S&P 500 constituent lookup, for measuring how much of this
project's backtested performance is watchlist curation bias rather than
genuine edge -- see check_survivorship_bias.py and backtest_diagnostic.txt.

Data: sp500_historical_components.csv, a daily point-in-time snapshot of
S&P 500 membership from 1996-01-02 through 2025-08-23, sourced from
https://github.com/hanshof/sp500_constituents (MIT licensed). Covers
essentially this whole project's usual backtest window (2021-06-01 onward);
a date past 2025-08-23 falls back to that last available snapshot, which is
a reasonable approximation since index membership doesn't turn over
dramatically within a year.
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "sp500_historical_components.csv"


def load_membership_table(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_membership_asof(date, table: pd.DataFrame | None = None) -> list[str]:
    """Return the S&P 500 ticker list as of `date` -- the most recent
    snapshot on or before that date. Falls back to the earliest snapshot if
    `date` predates the dataset, or the latest (2025-08-23) if `date` is
    past it."""
    if table is None:
        table = load_membership_table()
    date = pd.Timestamp(date)
    eligible = table[table["date"] <= date]
    row = eligible.iloc[-1] if not eligible.empty else table.iloc[0]
    return [t.strip().upper() for t in row["tickers"].split(",") if t.strip()]
