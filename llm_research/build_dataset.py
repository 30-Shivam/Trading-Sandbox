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
sys.path.insert(0, str(ROOT.parent))
import sec_fundamentals as sf
from sp500_membership import load_membership_table

REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]
HOLD = 126

def universe():
    t = load_membership_table(); names = set()
    for y in range(2010, 2026):
        row = t[t["date"] >= f"{y}-01-01"].iloc[0]; names |= set(row["tickers"].split(","))
    return sorted(names)

def facts_for(ticker, cik):
    f = CACHE / f"{cik}.json"
    if f.exists(): return json.loads(f.read_text())
    r = requests.get(sf.COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": sf.USER_AGENT}, timeout=30)
    time.sleep(sf.REQUEST_DELAY_SEC)
    d = r.json().get("facts", {}).get("us-gaap", {}) if r.status_code == 200 else {}
    f.write_text(json.dumps(d)); return d

def series(d, tags):
    out = []
    for tag in tags:
        out += d.get(tag, {}).get("units", {}).get("USD", [])
    return out

def annual_by_accn(entries):
    """{(accn, end): (val, filed)} for 10-K entries with ~1y duration (or instant if no start)."""
    o = {}
    for u in entries:
        if u.get("form") != "10-K" or not u.get("end"): continue
        if u.get("start"):
            days = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
            if not 350 <= days <= 380: continue
        o[(u["accn"], u["end"])] = (u["val"], u["filed"])
    return o

def rows_for(ticker, d):
    rev = annual_by_accn(series(d, REV)); ni = annual_by_accn(series(d, ["NetIncomeLoss"]))
    gp = annual_by_accn(series(d, ["GrossProfit"])); op = annual_by_accn(series(d, ["OperatingIncomeLoss"]))
    cf = annual_by_accn(series(d, ["NetCashProvidedByUsedInOperatingActivities"]))
    at = annual_by_accn(series(d, ["Assets"])); li = annual_by_accn(series(d, ["Liabilities"]))
    rows = []
    for (accn, end), (r1, filed) in rev.items():
        prev_end = (pd.Timestamp(end) - pd.DateOffset(years=1))
        pe = next((e for (a, e) in rev if a == accn and abs((pd.Timestamp(e) - prev_end).days) <= 10), None)
        if pe is None: continue
        g = lambda m, e: m.get((accn, e), (None,))[0]
        v = dict(r1=r1, r0=g(rev, pe), n1=g(ni, end), n0=g(ni, pe), g1=g(gp, end), g0=g(gp, pe), o1=g(op, end), o0=g(op, pe),
                 c1=g(cf, end), a1=g(at, end), a0=g(at, pe), l1=g(li, end), l0=g(li, pe))
        if any(v[k] in (None, 0) for k in ("r1", "r0", "a1", "a0")) or v["n1"] is None or v["c1"] is None or v["l1"] is None: continue
        rows.append(dict(ticker=ticker, filed=pd.Timestamp(filed), fy_end=pd.Timestamp(end),
            rev_growth=v["r1"]/v["r0"]-1, gross_margin=(v["g1"]/v["r1"]) if v["g1"] else np.nan,
            gm_change=(v["g1"]/v["r1"]-v["g0"]/v["r0"]) if v["g1"] and v["g0"] else np.nan,
            op_margin=(v["o1"]/v["r1"]) if v["o1"] is not None else np.nan,
            opm_change=(v["o1"]/v["r1"]-v["o0"]/v["r0"]) if v["o1"] is not None and v["o0"] is not None else np.nan,
            roa=v["n1"]/v["a1"], roa_change=(v["n1"]/v["a1"]-v["n0"]/v["a0"]) if v["n0"] is not None else np.nan,
            accruals=(v["n1"]-v["c1"])/v["a1"], cfo_to_assets=v["c1"]/v["a1"],
            leverage=v["l1"]/v["a1"], lev_change=(v["l1"]/v["a1"]-v["l0"]/v["a0"]) if v["l0"] is not None else np.nan,
            asset_growth=v["a1"]/v["a0"]-1))
    return rows

def main():
    cik = sf.fetch_cik_map(); names = universe()
    have = [t for t in names if t in cik]
    print(f"point-in-time S&P members 2010-2025: {len(names)}; resolvable to CIK: {len(have)} ({len(have)/len(names):.0%})")
    allrows = []
    for i, t in enumerate(have):
        try: allrows += rows_for(t, facts_for(t, cik[t]))
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
main()
