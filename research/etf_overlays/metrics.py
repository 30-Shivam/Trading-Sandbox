import numpy as np, pandas as pd
def stats(r, rf=0.0):
    r = r.dropna(); eq = (1+r).cumprod(); yrs = len(r)/252
    cagr = eq.iloc[-1]**(1/yrs)-1; dd = (eq/eq.cummax()-1).min()
    vol = r.std()*np.sqrt(252); sh = (r.mean()*252)/vol if vol else np.nan
    return dict(cagr=cagr, maxdd=dd, vol=vol, sharpe=sh, final=eq.iloc[-1])
def excess_t(r, b, block=21, n=4000, seed=0):
    """Block-bootstrap t of mean daily excess return (annualized), block=21d."""
    d = (r-b).dropna().values; m = len(d); nb = m//block
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m-block, size=(n, nb))
    means = np.array([np.concatenate([d[i:i+block] for i in row]).mean() for row in idx])
    ann = d.mean()*252
    return ann, ann/(means.std()*252), (means<=0).mean()  # one-sided p that mean<=0
