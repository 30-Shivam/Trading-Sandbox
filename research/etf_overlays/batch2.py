import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit('/',1)[0])
from data import px; from metrics import stats, excess_t
split = pd.Timestamp("2013-01-01")
spy = px("SPY"); ret = spy.pct_change().dropna()
rf = (px("^IRX")/100/252).reindex(ret.index).ffill().fillna(0)

def report(name, r, bench, start):
    r = r[r.index>=start]; b = bench.reindex(r.index)
    print(f"== {name}")
    for lab, sl in (("DESIGN", r.index<split), ("HOLDOUT", r.index>=split)):
        s, bs = stats(r[sl]), stats(b[sl]); a,t,p = excess_t(r[sl], b[sl])
        print(f"  {lab}: CAGR {s['cagr']:.2%} vs {bs['cagr']:.2%} | maxDD {s['maxdd']:.1%} vs {bs['maxdd']:.1%} | Sharpe {s['sharpe']:.2f} vs {bs['sharpe']:.2f} | excess {a:+.2%}/yr t={t:.2f} p={p:.3f}")

# 2. 2x SPY above 200d SMA (signal lagged 1d), else T-bills. 1x financing at rf+0.5%, 2bp cost per unit turnover.
sma = spy.rolling(200).mean(); on = (spy>sma).shift(1).reindex(ret.index).fillna(False)
w = on.astype(float)*2.0
r2 = w*ret - (w-1).clip(lower=0)*(rf+0.005/252) + (1-w).clip(lower=0)*rf - w.diff().abs().fillna(0)*2e-4
report("lev2x_trend200", r2, ret, pd.Timestamp("1994-06-01"))

def monthly_rotation(prices, pick, cost_bps=5):
    """prices: daily df. pick(ts_row_mom...) -> weights dict at month ends; applied next day."""
    rets = prices.pct_change()
    me = prices.resample("ME").last().index
    me = [d for d in me if d in prices.index or True]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last = None
    for d in prices.resample("ME").apply(lambda x: x.index[-1] if len(x) else None).iloc[:,0].dropna().values:
        d = pd.Timestamp(d); i = prices.index.get_loc(d)
        wt = pick(prices.iloc[:i+1])
        if wt is not None: last = wt
        if last is not None and i+1 < len(prices): w.iloc[i+1:, :] = pd.Series(last).reindex(prices.columns).fillna(0).values
    turn = w.diff().abs().sum(axis=1).fillna(0)*cost_bps/1e4
    return (w.shift(0)*rets).sum(axis=1) - turn, w

# 3. Sector ETF momentum, top 3 by 12-1 month return, equal weight
sec = pd.concat({t:px(t) for t in "XLK XLF XLE XLV XLY XLP XLI XLB XLU".split()}, axis=1).dropna()
def sector_pick(p):
    if len(p)<253: return None
    m = p.iloc[-22]/p.iloc[-253]-1
    top = m.nlargest(3).index; return {t:1/3 for t in top}
r3, _ = monthly_rotation(sec, sector_pick)
report("sector_etf_mom_top3", r3.dropna(), ret, sec.index[260])

# 4. GEM dual momentum: SPY vs EFA vs IEF(bonds); absolute momentum vs T-bills (12m)
g = pd.concat({t:px(t) for t in ["SPY","EFA","IEF"]}, axis=1).dropna()
tb = (1+rf).cumprod().reindex(g.index).ffill()
def gem_pick(p):
    if len(p)<253: return None
    m = p.iloc[-1]/p.iloc[-253]-1; tbm = tb.reindex(p.index).iloc[-1]/tb.reindex(p.index).iloc[-253]-1
    best = "SPY" if m["SPY"]>=m["EFA"] else "EFA"
    return {best:1.0} if m[best]>tbm else {"IEF":1.0}
r4, _ = monthly_rotation(g, gem_pick)
report("gem_dual_momentum", r4.dropna(), ret, g.index[260])
