import zipfile, io, re, sys, numpy as np, pandas as pd, pathlib
D = pathlib.Path(__file__).parent / "french"
def read_block(zname, which=0):
    """Return the `which`-th monthly YYYYMM table (0 = value-weighted) as DataFrame in decimal returns."""
    z = zipfile.ZipFile(D/zname); txt = z.read(z.namelist()[0]).decode("latin1").splitlines()
    blocks, cur, hdr = [], [], None
    for i, ln in enumerate(txt):
        parts = [p.strip() for p in ln.split(",")]
        if re.fullmatch(r"\d{6}", parts[0]) and len(parts) > 2:
            if not cur: hdr = [p for p in re.split(",", txt[i-1])] if not blocks or True else hdr
            cur.append((parts, hdr))
        elif cur:
            blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    rows, hdr = zip(*blocks[which]); hdr = [h.strip() for h in hdr[0]]
    # header line may lack the date column label -> align from the right
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m"))
    df = df.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan)/100
    cols = hdr[-df.shape[1]:]; df.columns = cols
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df
ff = read_block("F-F_Research_Data_Factors_CSV.zip", 0)
mkt = ff["Mkt-RF"] + ff["RF"]; rf = ff["RF"]

def stats(r):
    r = r.dropna(); eq=(1+r).cumprod(); yrs=len(r)/12
    return dict(cagr=eq.iloc[-1]**(1/yrs)-1, maxdd=(eq/eq.cummax()-1).min(), sharpe=(r.mean()*12)/(r.std()*np.sqrt(12)))
def excess_t(r, b, block=12, n=4000, seed=0):
    d=(r-b).dropna().values; m=len(d); nb=max(m//block,1); rng=np.random.default_rng(seed)
    idx=rng.integers(0,m-block,size=(n,nb)); means=np.array([np.concatenate([d[i:i+block] for i in row]).mean() for row in idx])
    a=d.mean()*12; return a, a/(means.std()*12), (means<=0).mean()
