"""Historical ticker -> CIK bridge for names Yahoo/SEC's current ticker map no longer serve.
Inputs: cache/ticker_names.json (ticker -> company name + alt names, from an LLM pass), cache/missing_tickers.tsv,
SEC cik-lookup-data.txt (all registrant names). A candidate CIK is ACCEPTED only if its XBRL annual filings
(edgar_features.rows_for) include one filed within [first index year - 1, last index year + 1]; otherwise 'unvalidated'.
Output: cache/ticker_cik_bridge.csv (ticker, cik, matched_name, n_rows, status)."""
import json, re, sys, difflib, pathlib, collections
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from edgar_features import facts_for, rows_for
SUFFIX = r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LTD|LIMITED|PLC|LLC|LP|HOLDINGS|HOLDING|GROUP|THE|NV|SA|AG|DE|NEW|OLD)\b"
def norm(s): s = re.sub(r"[^A-Z0-9 ]", " ", s.upper().replace("&", " AND ")); s = re.sub(SUFFIX, " ", s); return re.sub(r"\s+", " ", s).strip()
idx = collections.defaultdict(set); raw = {}
for line in open(ROOT/"cache"/"cik-lookup-data.txt", encoding="latin1"):
    name, _, rest = line.rstrip("\n").rpartition(":")[0].rpartition(":")[0], None, line.rstrip("\n").rpartition(":")[0].rpartition(":")[2]
    if not name or not rest.isdigit(): continue
    idx[norm(name)].add(rest.zfill(10)); raw[(norm(name), rest.zfill(10))] = name
keys = list(idx)
names = json.loads((ROOT/"cache"/"ticker_names.json").read_text())
miss = pd.read_csv(ROOT/"cache"/"missing_tickers.tsv", sep="\t")
out = []
for r in miss.itertuples():
    y0, y1 = (int(x) for x in r.years_in_index.split("-")); info = names.get(r.ticker) or {}
    cands = [n for n in [info.get("name")] + (info.get("alt_names") or []) if n]
    ciks = []
    for n in cands:
        k = norm(n)
        if k in idx: ciks += [(c, raw[(k, c)]) for c in idx[k]]
        else: ciks += [(c, raw[(m, c)]) for m in difflib.get_close_matches(k, keys, n=2, cutoff=0.93) for c in idx[m]]
    best = None
    for cik, nm in dict.fromkeys(ciks):
        try: rows = rows_for(r.ticker, facts_for(r.ticker, cik, ROOT/"cache"))
        except Exception: continue
        ok = [x for x in rows if y0 - 1 <= pd.Timestamp(x["filed"]).year <= y1 + 1]
        if ok and (best is None or len(ok) > best[2]): best = (cik, nm, len(ok))
    out.append(dict(ticker=r.ticker, years=r.years_in_index, cik=best[0] if best else "", matched_name=best[1] if best else "",
                    n_rows=best[2] if best else 0, status="validated" if best else ("no_candidate" if not ciks else "unvalidated"),
                    llm_name=info.get("name") or "", confidence=info.get("confidence", "")))
df = pd.DataFrame(out); df.to_csv(ROOT/"cache"/"ticker_cik_bridge.csv", index=False)
print(df.status.value_counts().to_string()); print(df[df.status != "validated"][["ticker", "years", "llm_name", "status"]].to_string(index=False))
