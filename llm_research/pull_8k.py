"""Pull Item 2.02 (earnings) 8-K acceptance timestamps from EDGAR submissions JSON (recent + older pages) for every ticker in
prices_pit. Cache: cache/sub8k/<cik>.json = [{accn, filed, acc}] . Resumable. Rate-limited ~6 req/s. Run as a script."""
import sys, json, time, pathlib, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"; OUT = CACHE / "sub8k"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent)); import sec_fundamentals as sf
S = requests.Session(); S.headers["User-Agent"] = sf.USER_AGENT
def get(url):
    for k in range(4):
        r = S.get(url, timeout=30)
        if r.status_code == 200: return r.json()
        time.sleep(2 * (k + 1))
    return None
def rows_from(block):
    out = []
    for f, it, a, acc in zip(block["form"], block.get("items", [""] * len(block["form"])), block["acceptanceDateTime"], block["accessionNumber"]):
        if f in ("8-K", "8-K/A") and "2.02" in (it or "").split(","): out.append(dict(accn=acc, form=f, acc=a))
    return out
if __name__ == "__main__":
    yahoo_ok = set(pd.read_csv(CACHE / "yahoo_ok.csv").iloc[:, 0]); cur = sf.fetch_cik_map()
    br = pd.read_csv(CACHE / "ticker_cik_bridge.csv", dtype=str).fillna(""); bridge = {r.ticker: r.cik for r in br.itertuples() if r.status == "validated"}
    px = pd.read_pickle(CACHE / "prices_pit.pkl"); ciks = {}
    for t in px.columns:
        c = cur[t] if (t in yahoo_ok and t in cur) else bridge.get(t, cur.get(t))
        if c and t != "SPY": ciks[t] = str(c).zfill(10)
    print(len(ciks), "tickers", flush=True)
    for i, (t, c) in enumerate(ciks.items()):
        f = OUT / f"{c}.json"
        if f.exists(): continue
        j = get(f"https://data.sec.gov/submissions/CIK{c}.json"); time.sleep(0.17)
        if j is None: print("FAIL", t, c, flush=True); continue
        rows = rows_from(j["filings"]["recent"])
        for p in j["filings"].get("files", []):
            jj = get("https://data.sec.gov/submissions/" + p["name"]); time.sleep(0.17)
            if jj: rows += rows_from(jj)
        json.dump(dict(ticker=t, rows=rows), open(f, "w"))
        if i % 50 == 0: print(i, t, len(rows), flush=True)
    print("DONE")
