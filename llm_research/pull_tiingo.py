"""Resumable Tiingo EOD pull for point-in-time S&P 500 names that Yahoo no longer serves.
Free tier limits: 50 requests/hour, 1,000/day, 500 unique symbols/month -> one price request per symbol, paced.
Token comes from env TIINGO_API_KEY (never written to disk). Output: cache/tiingo/<TICKER>.pkl (+ status log)."""
import os, sys, time, pathlib, requests, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; OUT = ROOT / "cache" / "tiingo"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT.parent))
from sp500_membership import load_membership_table
KEY = os.environ["TIINGO_API_KEY"]; H = {"Content-Type": "application/json", "Authorization": f"Token {KEY}"}
SPACING = 74  # seconds -> ~48 requests/hour
t = load_membership_table(); names = set()
for y in range(2010, 2026): names |= set(t[t["date"] >= f"{y}-01-01"].iloc[0]["tickers"].split(","))
ok = set(pd.read_csv(ROOT / "cache" / "yahoo_ok.csv").iloc[:, 0])
todo = sorted(names - ok)
log = OUT / "_status.csv"; done = {l.split(",")[0] for l in log.read_text().splitlines()} if log.exists() else set()
print(f"{len(todo)} missing names, {len(done)} already attempted", flush=True)
for tk in todo:
    if tk in done: continue
    sym = tk.replace(".", "-")
    while True:
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{sym}/prices", params={"startDate": "2000-01-01"}, headers=H, timeout=60)
        if r.status_code == 429: print("rate limited, sleeping 15 min", flush=True); time.sleep(900); continue
        break
    rows, first, last = 0, "", ""
    if r.status_code == 200 and r.json():
        d = pd.DataFrame(r.json()); d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None); d = d.set_index("date")
        d.to_pickle(OUT / f"{tk}.pkl"); rows, first, last = len(d), str(d.index[0].date()), str(d.index[-1].date())
    with open(log, "a") as f: f.write(f"{tk},{r.status_code},{rows},{first},{last}\n")
    print(f"{tk}: {r.status_code} rows={rows} {first}..{last}", flush=True)
    time.sleep(SPACING)
print("DONE", flush=True)
