"""Build the (ticker, 10-K filing) dataset for the masked-fundamentals LLM test.

Universe: tickers that were S&P 500 members at ANY year start 2010-2025 (point-in-time list,
sp500_historical_components.csv) AND resolvable to a CIK via SEC's current ticker map. Delisted
names missing from the map / Yahoo are dropped -- survivorship is MEASURED and printed, not hidden.
Features are as-filed (filed date = first public date). Label: forward 126-trading-day return
vs SPY, entered at the close of the 2nd trading day after filing.
Run: python llm_research/build_dataset.py
"""
import json, sys, time, pathlib
import numpy as np, pandas as pd, requests, yfinance as yf
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT))
import sec_fundamentals as sf
from edgar_features import facts_for, rows_for
from sp500_membership import load_membership_table

HOLD = 126

def universe():
    t = load_membership_table(); names = set()
    for y in range(2010, 2026):
        row = t[t["date"] >= f"{y}-01-01"].iloc[0]; names |= set(row["tickers"].split(","))
    return sorted(names)

def main():
    cik = sf.fetch_cik_map(); names = universe()
    have = [t for t in names if t in cik]
    print(f"point-in-time S&P members 2010-2025: {len(names)}; resolvable to CIK: {len(have)} ({len(have)/len(names):.0%})")
    allrows = []
    for i, t in enumerate(have):
        try: allrows += rows_for(t, facts_for(t, cik[t], CACHE))
        except Exception as e: print("skip", t, e)
        if i % 100 == 0: print(f"  facts {i}/{len(have)} rows={len(allrows)}", flush=True)
    df = pd.DataFrame(allrows); df = df[(df.filed >= "2010-01-01") & (df.filed <= "2025-12-31")]
    print("filings with features:", len(df), "tickers:", df.ticker.nunique())
    # prices
    tick = sorted(df.ticker.unique()) + ["SPY"]; px = {}
    for i in range(0, len(tick), 80):
        chunk = tick[i:i+80]
        d = yf.download(chunk, start="2009-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(d, pd.Series): d = d.to_frame(chunk[0])
        for c in d.columns:
            if d[c].notna().sum() > 300: px[c] = d[c]
        time.sleep(2); print(f"  prices {min(i+80,len(tick))}/{len(tick)} have={len(px)}", flush=True)
    P = pd.DataFrame(px); spy = P["SPY"]
    out = []
    for r in df.itertuples():
        if r.ticker not in P: continue
        s = P[r.ticker].dropna(); idx = s.index.searchsorted(r.filed + pd.Timedelta(days=1)) + 1
        if idx + HOLD >= len(s): continue
        d0, d1 = s.index[idx], s.index[idx + HOLD]
        if d0 not in spy.index or d1 not in spy.index: continue
        ret = s.iloc[idx + HOLD] / s.iloc[idx] - 1; b = spy[d1] / spy[d0] - 1
        out.append({**r._asdict(), "entry": d0, "exit": d1, "fwd_ret": ret, "spy_ret": b, "excess": ret - b})
    res = pd.DataFrame(out).drop(columns="Index")
    print(f"labelled: {len(res)} filings, {res.ticker.nunique()} tickers (price-survivors only; {len(df)-len(res)} unlabelled)")
    P.to_pickle(CACHE / "prices.pkl"); res.to_pickle(CACHE / "dataset.pkl"); print(res.groupby(res.filed.dt.year).size().to_string())
if __name__ == "__main__":
    main()
