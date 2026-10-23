"""Evaluate blind LLM scores (cache/scores/b_*.json) against realized 126d excess returns.
Arm 'pick' = filings the causal ratio-ridge selected (top quintile); arm 'rand' = random non-picks, 2014-2025.
Questions: (1) within picks, does the LLM rank outcomes (filter value)? (2) on the random arm, does LLM IC beat/add to ridge?
(3) permutation p-values for every IC (shuffle LLM scores within arm)."""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from portfolio_test import D, F, ridge_scorer, ROOT
imap = pd.read_pickle(ROOT/"cache"/"item_map.pkl")
sc = {}
for f in sorted(glob.glob(str(ROOT/"cache"/"scores"/"b_*.json"))): sc.update(json.load(open(f)))
imap["llm"] = imap["id"].map(sc); print(f"scored {imap.llm.notna().sum()}/{len(imap)} items")
X = D.loc[imap.row.values].copy(); X["arm"] = imap.arm.values; X["llm"] = imap.llm.values; X["id"] = imap.id.values
# ridge score for every item (causal: trained on labels that ended before the item's filing year)
ridge = pd.Series(np.nan, index=X.index); fn = ridge_scorer()
for Y in sorted(X.year.unique()):
    tr = D[D.exit < pd.Timestamp(f"{Y}-01-01")]
    if len(tr) < 400: continue
    te = X[X.year == Y]; ridge[te.index] = fn(tr, te)[0]
X["ridge"] = ridge; X = X.dropna(subset=["llm"])
rk = lambda s: s.rank(pct=True)
def ic(a, b): return a.rank().corr(b.rank())
def perm_p(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed); obs = ic(a, b); av = a.values
    null = np.array([ic(pd.Series(rng.permutation(av)), b.reset_index(drop=True)) for _ in range(n)]); return obs, (null >= obs).mean()
for arm in ("pick", "rand"):
    d = X[X.arm == arm]; o, p = perm_p(d.llm.reset_index(drop=True), d.excess.reset_index(drop=True))
    print(f"[{arm}] n={len(d)}  LLM IC={o:+.3f} (perm p={p:.4f})  ridge IC={ic(d.ridge, d.excess):+.3f}  LLM-vs-ridge corr={ic(d.llm, d.ridge):+.2f}")
d = X[X.arm == "rand"].copy(); d["combo"] = rk(d.llm) + rk(d.ridge)
print(f"[rand] combo(LLM+ridge) IC={ic(d.combo, d.excess):+.3f}   (compare ridge-only {ic(d.ridge, d.excess):+.3f}, LLM-only {ic(d.llm, d.excess):+.3f})")
p = X[X.arm == "pick"].copy(); hi = p[p.llm >= p.llm.median()]; lo = p[p.llm < p.llm.median()]
print(f"[pick] LLM-keep mean excess {hi.excess.mean():+.4f} (n={len(hi)}) vs LLM-veto {lo.excess.mean():+.4f} (n={len(lo)}) ; diff {hi.excess.mean()-lo.excess.mean():+.4f}")
print("LLM score distribution:", X.llm.value_counts().sort_index().to_dict())
print("by year (pick arm) LLM IC:", {int(y): round(ic(g.llm, g.excess), 3) for y, g in p.groupby("year") if len(g) > 25})
