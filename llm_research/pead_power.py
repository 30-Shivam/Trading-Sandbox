"""PEAD power screen. Zero trials: the basket is a RANDOM 5% of that quarter's filings (no signal used), so this measures the
sampling noise of an earnings-season basket, not any edge. Event proxy = 10-Q/10-K 'filed' dates of PIT S&P 500 members
(10-Qs lag the earnings release, so this is a count/clustering proxy only)."""
import sys, json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import sec_fundamentals as sf
from pit_utils import make_membership, member_asof
from sp500_membership import load_membership_table
rng = np.random.default_rng(0)
table = load_membership_table(); dates, sets = make_membership(table)
yahoo_ok = set(pd.read_csv(CACHE / "yahoo_ok.csv").iloc[:, 0]); cur = sf.fetch_cik_map()
br = pd.read_csv(CACHE / "ticker_cik_bridge.csv", dtype=str).fillna(""); bridge = {r.ticker: r.cik for r in br.itertuples() if r.status == "validated"}
px = pd.read_pickle(CACHE / "prices_pit.pkl"); spy = px["SPY"].dropna()
ev = []
for t in px.columns:
    if t == "SPY": continue
    c = cur[t] if (t in yahoo_ok and t in cur) else bridge.get(t, cur.get(t))
    if not c: continue
    f = CACHE / f"{str(c).zfill(10)}.json"
    if not f.exists(): continue
    j = json.load(open(f)); filed = set()
    for concept in ("NetIncomeLoss", "EarningsPerShareDiluted", "Assets"):
        for u in j.get(concept, {}).get("units", {}).values():
            for r in u:
                if r.get("form") in ("10-Q", "10-K"): filed.add(r["filed"])
    for d in filed:
        d = pd.Timestamp(d)
        if d >= pd.Timestamp("2010-01-01") and member_asof(t, d, dates, sets): ev.append((t, d))
ev = pd.DataFrame(ev, columns=["ticker", "filed"]).sort_values("filed").reset_index(drop=True)
ev["year"] = ev.filed.dt.year; print("PIT-member filing events per year:\n", ev.groupby("year").size().to_dict())
# clustering: share of a year's events in its 4 busiest 4-week windows
ev["wk"] = ev.filed.dt.isocalendar().week.values
sh = ev.groupby("year").apply(lambda g: g.groupby(g.wk // 4).size().nlargest(4).sum() / len(g)); print("share of events in 4 busiest 4-week windows:", sh.round(2).mean())
# 20d excess from entry (2 td after filing); cluster = (year, quarter of filing date)
out = []
for t, d in zip(ev.ticker, ev.filed):
    s = px[t].dropna(); i = s.index.searchsorted(d + pd.Timedelta(days=1)) + 1
    if i + 20 >= len(s): out.append(np.nan); continue
    a, b = s.index[i], s.index[i + 20]
    out.append(s.iloc[i + 20] / s.iloc[i] - spy.asof(b) / spy.asof(a)) if a in spy.index else out.append(np.nan)
ev["x"] = out; ev = ev.dropna(subset=["x"]); lo, hi = ev.x.quantile([.01, .99]); ev["x"] = ev.x.clip(lo, hi)
ev["cl"] = ev.year.astype(str) + "Q" + ((ev.filed.dt.month - 1) // 3 + 1).astype(str)
def mbb_se(series, block=4, B=3000):
    n = len(series); m = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, int(np.ceil(n / block)))])[:n]; m.append(series[idx].mean())
    return float(np.std(m))
for lab, mask in (("2013+", ev.year >= 2013), ("2010-2012", ev.year < 2013), ("2010+ all", ev.year >= 2010)):
    g = ev[mask]; cls = sorted(g.cl.unique()); n = len(g)
    if len(cls) < 8: print(lab, "too few clusters", len(cls)); continue
    sd_trade = g.x.std(); ses = {}
    for pct in (0.05, 1.0):
        bm = []
        for c in cls:
            xs = g.x[g.cl == c].values; k = max(int(round(len(xs) * pct)), 5)
            bm.append(np.mean([rng.choice(xs, min(k, len(xs)), replace=False).mean() for _ in range(100)]))
        ses[pct] = mbb_se(np.array(bm), 4 if len(cls) > 12 else 2)
        # note: mean over random draws removes basket sampling noise; add it back per quarter:
        k_sd = np.mean([g.x[g.cl == c].std() / np.sqrt(max(int(round((g.cl == c).sum() * pct)), 5)) for c in cls])
        ses[pct] = float(np.sqrt(ses[pct] ** 2 + (k_sd ** 2) / len(cls)))
    top5 = int(0.05 * n)
    print(f"\n{lab}: events {n}, quarters {len(cls)}, per-event 20d excess sd {sd_trade:.2%}")
    print(f"  naive iid N={top5} (top 5%): SE {sd_trade/np.sqrt(top5):.2%} -> t=3.09 needs {3.09*sd_trade/np.sqrt(top5):.2%}")
    print(f"  block-bootstrap (random 5% basket per quarter, 4-qtr blocks): SE {ses[.05]:.2%} -> t=3.09 needs {3.09*ses[.05]:.2%}, t=2.58 {2.58*ses[.05]:.2%}, t=1.65 {1.65*ses[.05]:.2%}")
