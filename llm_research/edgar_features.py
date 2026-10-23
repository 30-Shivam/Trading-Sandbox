"""EDGAR companyfacts -> as-filed annual ratio rows (shared by build_dataset.py and ratio_tracker.py).
Caching is optional: pass cache_dir=None (CI) to always fetch."""
import json, sys, time, pathlib
import numpy as np, pandas as pd, requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import sec_fundamentals as sf

FEATURES = ["rev_growth","gross_margin","gm_change","op_margin","opm_change","roa","roa_change","accruals","cfo_to_assets","leverage","lev_change","asset_growth"]
REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]

def facts_for(ticker, cik, cache_dir=None):
    f = pathlib.Path(cache_dir) / f"{cik}.json" if cache_dir else None
    if f is not None and f.exists(): return json.loads(f.read_text())
    r = requests.get(sf.COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": sf.USER_AGENT}, timeout=30)
    time.sleep(sf.REQUEST_DELAY_SEC)
    d = r.json().get("facts", {}).get("us-gaap", {}) if r.status_code == 200 else {}
    if f is not None: f.write_text(json.dumps(d))
    return d

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

