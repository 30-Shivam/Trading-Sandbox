"""Pure helpers for the point-in-time dataset (tested in tests/test_pit_utils.py)."""
import numpy as np, pandas as pd


def make_membership(table: pd.DataFrame):
    """table: columns date, tickers (comma string) sorted by date -> (np datetime64 array, list of sets)."""
    return table["date"].values, [set(x.split(",")) for x in table["tickers"]]


def member_asof(ticker: str, date, dates, sets) -> bool:
    """True iff `ticker` was in the snapshot on or before `date` (earliest snapshot if date predates the data)."""
    i = np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="right") - 1
    return ticker in sets[max(i, 0)]


def label_with_truncation(stock: pd.Series, spy: pd.Series, filed, hold: int = 126, today=None):
    """Entry = close of the 2nd trading day after `filed`. Exit = `hold` trading days later.
    If the stock's series ENDS before the exit (delisted/acquired) the exit is its last price (flag truncated=True;
    the cash-out is treated as ~ last close). If the series is still live but too short -> None (label pending).
    Returns dict(entry, exit, fwd_ret, spy_ret, excess, truncated) or None."""
    s = stock.dropna()
    if s.empty: return None
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    idx = s.index.searchsorted(pd.Timestamp(filed) + pd.Timedelta(days=1)) + 1
    if idx >= len(s): return None
    truncated = False
    if idx + hold >= len(s):
        if s.index[-1] >= today - pd.Timedelta(days=10): return None     # still trading: window not complete yet
        j, truncated = len(s) - 1, True
        if j <= idx: return None
    else:
        j = idx + hold
    d0, d1 = s.index[idx], s.index[j]
    if d0 not in spy.index: return None
    sp1 = spy.asof(d1)
    if pd.isna(sp1): return None
    ret, bench = s.iloc[j] / s.iloc[idx] - 1, sp1 / spy[d0] - 1
    return dict(entry=d0, exit=d1, fwd_ret=float(ret), spy_ret=float(bench), excess=float(ret - bench), truncated=truncated)
