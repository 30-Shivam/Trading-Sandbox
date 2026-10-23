"""Diagnostics for sec_ratio_ridge. Reports everything; nothing here is used to re-select the model."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from portfolio_test import D, P, F, ridge_scorer
R = P.pct_change()
# walk-forward scores for every filing 2014+ (causal, same as portfolio_test)
D = D.copy(); D["score"] = np.nan; fn = ridge_scorer()
for Y in range(2014, 2026):
    tr = D[D.exit < pd.Timestamp(f"{Y}-01-01")]; te = D[D.year == Y]
    if len(tr) >= 400: D.loc[te.index, "score"] = fn(tr, te)[0]
T = D.dropna(subset=["score"]).copy()
spy = P["SPY"]
def fwd_excess(r, h):
    s = P[r.ticker].dropna(); i = s.index.searchsorted(r.entry)
    if i + h >= len(s) or s.index[i] != r.entry or r.entry not in spy.index: return np.nan
    d1 = s.index[i + h]
    return (s.iloc[i + h] / s.iloc[i] - 1) - (spy[d1] / spy[r.entry] - 1) if d1 in spy.index else np.nan
def ic_years(col, df):
    v = np.array([g[col].rank().corr(g.ex.rank()) for _, g in df.groupby("year") if len(g) > 30]); return v.mean(), v.mean()/(v.std(ddof=1)/np.sqrt(len(v))), len(v)
print("== A. horizon sweep (rank-IC of walk-forward score vs excess return, mean over years, t across years)")
for h in (21, 63, 126, 252):
    T["ex"] = [fwd_excess(r, h) for r in T.itertuples()]
    m, t, n = ic_years("score", T.dropna(subset=["ex"])); q = T.dropna(subset=["ex"]); hi = q[q.score > q.score.quantile(.8)]
    print(f"  h={h:3d}d: IC {m:+.3f} t={t:+.2f} (years={n})  top-quintile mean excess {hi.ex.mean():+.2%}  all {q.ex.mean():+.2%}")
T["ex"] = T.excess
print("== B. simple vs complex (126d): single feature cfo_to_assets (no fitting) vs ridge")
T["cfo"] = T.groupby("year").cfo_to_assets.rank(pct=True)
for name, col in (("ridge", "score"), ("cfo_to_assets only", "cfo")):
    m, t, n = ic_years(col, T); hi = T[T[col] > T[col].quantile(.8)]
    print(f"  {name:20s}: IC {m:+.3f} t={t:+.2f}  top-quintile mean excess {hi.ex.mean():+.2%}")
print("== C. placebo: shuffled scores within year (expect IC~0)")
rng = np.random.default_rng(1); T["rand"] = T.groupby("year").score.transform(lambda s: rng.permutation(s.values))
m, t, n = ic_years("rand", T); print(f"  IC {m:+.3f} t={t:+.2f}")
print("== D. concentration: top-quintile mean excess excluding best 5% and by ticker")
hi = T[T.score > T.score.quantile(.8)].sort_values("ex")
trim = hi.iloc[: int(len(hi) * .95)]; print(f"  picks n={len(hi)} mean {hi.ex.mean():+.2%} | median {hi.ex.median():+.2%} | mean ex best 5% {trim.ex.mean():+.2%} | hit-rate (excess>0) {(hi.ex>0).mean():.1%}")
print("  top 8 tickers by pick count:", hi.ticker.value_counts().head(8).to_dict())
print("== E. regime split of top-quintile mean excess")
for lab, a, b in (("2014-2018", 2014, 2018), ("2019-2021", 2019, 2021), ("2022-2025", 2022, 2025)):
    s = hi[(hi.year >= a) & (hi.year <= b)]; print(f"  {lab}: n={len(s)} mean excess {s.ex.mean():+.2%} hit {(s.ex>0).mean():.1%}")
