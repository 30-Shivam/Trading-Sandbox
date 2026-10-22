import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit('/',1)[0])
from data import px; from metrics import stats, excess_t
spy = px("SPY"); irx = px("^IRX")/100/252
irx = irx.reindex(spy.index).ffill().fillna(0)
ret = spy.pct_change().dropna()
def run(lmax, target, win=21, fin_spread=0.005, cost_bps=2):
    rv = ret.rolling(win).std()*np.sqrt(252)
    w = (target/rv).clip(upper=lmax).shift(1)   # decided on prior close
    w = w.clip(lower=0).reindex(ret.index)
    fin = (w-1).clip(lower=0)*(irx.reindex(ret.index)+fin_spread/252)
    cash = (1-w).clip(lower=0)*irx.reindex(ret.index)
    turn = w.diff().abs().fillna(0)*cost_bps/1e4
    return (w*ret - fin + cash - turn).dropna()
split = pd.Timestamp("2013-01-01")
design = ret[ret.index < split]
target = design.std()*np.sqrt(252)   # untuned: SPY's own design-period vol
print("design-period SPY vol (target):", round(target,4))
for lmax in (1.0, 1.5):
    r = run(lmax, target)
    for name, sl in (("DESIGN", r.index<split), ("HOLDOUT", r.index>=split)):
        rr, bb = r[sl], ret.reindex(r.index)[sl]
        s, b = stats(rr), stats(bb); a,t,p = excess_t(rr, bb)
        print(f"lmax={lmax} {name}: CAGR {s['cagr']:.3%} vs SPY {b['cagr']:.3%} | maxDD {s['maxdd']:.1%} vs {b['maxdd']:.1%} | Sharpe {s['sharpe']:.2f} vs {b['sharpe']:.2f} | excess {a:.2%}/yr t={t:.2f} p={p:.3f}")
