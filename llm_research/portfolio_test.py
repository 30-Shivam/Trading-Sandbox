"""Causal walk-forward test of a per-filing score as a daily portfolio vs SPY.
score_fn(train_df, test_df) -> test scores; selection threshold = top (1-q) of the TRAIN fitted scores.
Training uses only filings whose 126d label had ALREADY ended before the test year started."""
import sys, numpy as np, pandas as pd, pathlib
sys.path.insert(0, "/private/tmp/claude-501/-Users-30-shivam-Downloads-Trading-Sandbox/a118247b-3bee-4075-8406-247029065205/scratchpad/research")
ROOT = pathlib.Path(__file__).parent; D = pd.read_pickle(ROOT/"cache"/"dataset.pkl"); P = pd.read_pickle(ROOT/"cache"/"prices.pkl")
F = ["rev_growth","gross_margin","gm_change","op_margin","opm_change","roa","roa_change","accruals","cfo_to_assets","leverage","lev_change","asset_growth"]
D["year"] = D.filed.dt.year

def ridge_scorer(lam=50.0):
    def fn(tr, te):
        mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
        z = lambda d: ((d[F]-mu)/sd).clip(-3, 3).fillna(0)
        y = tr.excess - tr.excess.mean(); X = z(tr).values
        w = np.linalg.solve(X.T@X + lam*np.eye(len(F)), X.T@y.values)
        return z(te).values@w, z(tr).values@w
    return fn

def walk_forward(score_fn, q=0.8, first=2014, last=2025):
    picks = []
    for Y in range(first, last+1):
        tr = D[D.exit < pd.Timestamp(f"{Y}-01-01")]; te = D[D.year == Y]
        if len(tr) < 400 or te.empty: continue
        s_te, s_tr = score_fn(tr, te); thr = np.quantile(s_tr, q)
        sel = te[s_te > thr].copy(); sel["score"] = s_te[s_te > thr]; picks.append(sel)
    return pd.concat(picks)

def daily_portfolio(picks, cost_bps=10):
    R = P.pct_change(); spy = R["SPY"]; days = P.index
    book = pd.DataFrame(0.0, index=days, columns=["n"]); acc = pd.Series(0.0, index=days); cnt = pd.Series(0, index=days)
    for r in picks.itertuples():
        i0, i1 = days.get_loc(r.entry), days.get_loc(r.exit)
        rr = R[r.ticker].iloc[i0+1:i1+1].fillna(0)
        acc.iloc[i0+1:i1+1] += rr.values; cnt.iloc[i0+1:i1+1] += 1
        acc.iloc[i0+1] -= cost_bps/1e4; acc.iloc[i1] -= cost_bps/1e4
    pr = (acc/cnt.replace(0, np.nan))
    port = pr.fillna(spy)           # SPY when nothing is held
    start = picks.entry.min(); return port[port.index >= start], spy[spy.index >= start], cnt[cnt.index >= start]

def evaluate(port, spy, label):
    from metrics import stats, excess_t
    out = [f"== {label}"]
    for lab, a, b in (("ALL", port.index[0], port.index[-1]), ("HOLDOUT 2019+", pd.Timestamp("2019-01-01"), port.index[-1])):
        r = port[(port.index >= a)&(port.index <= b)]; bm = spy.reindex(r.index)
        s, bs = stats(r), stats(bm); ex, t, p = excess_t(r, bm)
        out.append(f"  {lab}: CAGR {s['cagr']:.2%} vs SPY {bs['cagr']:.2%} | maxDD {s['maxdd']:.1%} vs {bs['maxdd']:.1%} | Sharpe {s['sharpe']:.2f} vs {bs['sharpe']:.2f} | excess {ex:+.2%}/yr t={t:.2f} p={p:.4f}")
    print("\n".join(out))

if __name__ == "__main__":
    picks = walk_forward(ridge_scorer())
    print("picks:", len(picks), "per year:", picks.groupby("year").size().to_dict())
    port, spy, cnt = daily_portfolio(picks)
    print("avg positions held:", round(cnt[cnt>0].mean(),1), " days with no positions:", int((cnt==0).sum()))
    evaluate(port, spy, "sec_ratio_ridge (top quintile, EW, 10bps/side, SPY when idle)")
