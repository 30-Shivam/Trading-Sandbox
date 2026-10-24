"""Non-LLM control: per-feature rank IC and an expanding-window ridge on rank-transformed ratios.
IC is computed within filing-year cross-sections (rank vs excess return), reported as mean IC and t across years."""
import numpy as np, pandas as pd, pathlib
def spearmanr(a, b):
    return (pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()), None)
D = pd.read_pickle(pathlib.Path(__file__).parent / "cache" / "dataset.pkl")
F = ["rev_growth","gross_margin","gm_change","op_margin","opm_change","roa","roa_change","accruals","cfo_to_assets","leverage","lev_change","asset_growth"]
D["year"] = D.filed.dt.year
R = D.copy()
for f in F: R[f] = R.groupby("year")[f].rank(pct=True) - 0.5
R[F] = R[F].fillna(0)
def ic_by_year(score, df):
    s = [spearmanr(score[g.index], g.excess)[0] for _, g in df.groupby("year") if len(g) > 30]
    s = np.array(s); return s.mean(), s.mean()/(s.std(ddof=1)/np.sqrt(len(s))), len(s)
print("per-feature IC (all years):")
for f in F:
    m,t,n = ic_by_year(R[f], R); print(f"  {f:14s} IC {m:+.3f} t={t:+.2f} (n_years={n})")
# expanding ridge, train on years < Y, test year Y
from numpy.linalg import solve
pred = pd.Series(np.nan, index=R.index)
for y in range(2014, 2026):
    tr, te = R[R.year < y], R[R.year == y]
    y_tr = tr.groupby("year").excess.rank(pct=True) - 0.5
    X = tr[F].values; w = solve(X.T@X + 50*np.eye(len(F)), X.T@y_tr.values)
    pred[te.index] = te[F].values @ w
T = R[R.year >= 2014].copy(); T["pred"] = pred
m,t,n = ic_by_year(T.pred, T); print(f"\nRIDGE walk-forward 2014-2025: IC {m:+.3f} t={t:+.2f} n_years={n}")
H = T[T.year >= 2019]; m,t,n = ic_by_year(H.pred, H); print(f"RIDGE holdout 2019-2025: IC {m:+.3f} t={t:+.2f} n_years={n}")
# top-quintile minus universe within year
q = T.groupby("year").pred.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
T["q"] = q; print("\nmean excess by quintile (6m, vs SPY, all 2014-2025):\n", T.groupby("q").excess.mean().round(4).to_string())
print("top-quintile mean excess by year:\n", T[T.q==4].groupby("year").excess.mean().round(3).to_string())
