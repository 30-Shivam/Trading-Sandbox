"""Point-in-time 10-K dataset: (1) a filing counts only if the ticker was an S&P 500 member AS OF its filing date,
(2) delisted names come from Tiingo prices (cache/tiingo/*.pkl) + the validated ticker->CIK bridge, (3) labels
truncate at delisting (cash-out ~ last close) instead of being dropped. Financials without a revenue tag fall out.
Writes cache/dataset_pit.pkl and cache/prices_pit.pkl. Requires: build_dataset.py (Yahoo names), pull_tiingo.py, bridge_cik.py.
Run: python llm_research/build_dataset_pit.py"""
import sys, time, pathlib
import numpy as np, pandas as pd, yfinance as yf
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import sec_fundamentals as sf
from edgar_features import facts_for, rows_for
from pit_utils import make_membership, member_asof, label_with_truncation
from sp500_membership import load_membership_table
HOLD = 126

def main():
    table = load_membership_table(); dates, sets = make_membership(table)
    names = sorted(set().union(*[set(table[table["date"] >= f"{y}-01-01"].iloc[0]["tickers"].split(",")) for y in range(2010, 2026)]))
    yahoo_ok = set(pd.read_csv(CACHE / "yahoo_ok.csv").iloc[:, 0])
    cur = sf.fetch_cik_map(); br = pd.read_csv(CACHE / "ticker_cik_bridge.csv", dtype=str).fillna("")
    bridge = {r.ticker: r.cik for r in br.itertuples() if r.status == "validated"}
    cik = {t: (cur[t] if t in yahoo_ok and t in cur else bridge.get(t, cur.get(t))) for t in names}
    cik = {t: c for t, c in cik.items() if c}
    print(f"names {len(names)} | with CIK {len(cik)} (bridge supplied {sum(1 for t in cik if t in bridge and t not in yahoo_ok)})")
    rows, dropped_nonmember = [], 0
    for i, (t, c) in enumerate(cik.items()):
        try: rr = rows_for(t, facts_for(t, c, CACHE))
        except Exception: continue
        for r in rr:
            if not (pd.Timestamp("2010-01-01") <= r["filed"] <= pd.Timestamp("2025-12-31")): continue
            if not member_asof(t, r["filed"], dates, sets): dropped_nonmember += 1; continue
            rows.append(r)
        if i % 100 == 0: print(f"  facts {i}/{len(cik)} rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows); print(f"filings kept (member as-of filing): {len(df)} | dropped non-member-period filings: {dropped_nonmember} | tickers {df.ticker.nunique()}")
    # prices: Yahoo for served names (+SPY), Tiingo cache for the rest
    need = sorted(df.ticker.unique()); px = {}
    ytick = [t for t in need if t in yahoo_ok] + ["SPY"]
    for i in range(0, len(ytick), 80):
        d = yf.download(ytick[i:i+80], start="2009-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(d, pd.Series): d = d.to_frame(ytick[i])
        for c in d.columns:
            if d[c].notna().sum() > 300: px[c] = d[c]
        time.sleep(2)
    tg = 0
    for t in need:
        f = CACHE / "tiingo" / f"{t}.pkl"
        if t not in px and f.exists():
            d = pd.read_pickle(f); px[t] = d["adjClose"]; tg += 1
    print(f"price series: yahoo {len(px) - tg}, tiingo {tg}; tickers with CIK rows but no prices: {len(set(need) - set(px))}")
    P = pd.DataFrame(px).sort_index(); spy = P["SPY"]; out = []
    for r in df.itertuples():
        if r.ticker not in P: continue
        lab = label_with_truncation(P[r.ticker], spy, r.filed, HOLD)
        if lab: out.append({**r._asdict(), **lab})
    res = pd.DataFrame(out).drop(columns="Index")
    print(f"labelled {len(res)} filings, {res.ticker.nunique()} tickers, truncated(delisted) labels: {int(res.truncated.sum())}")
    P.to_pickle(CACHE / "prices_pit.pkl"); res.to_pickle(CACHE / "dataset_pit.pkl")
    print(res.groupby(res.filed.dt.year).size().to_string())

if __name__ == "__main__":
    main()
