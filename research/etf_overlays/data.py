import pandas as pd, yfinance as yf, pathlib
C = pathlib.Path(__file__).parent / "cache"; C.mkdir(exist_ok=True)
def px(t, start="1993-01-01"):
    f = C / f"{t.replace('^','_')}.pkl"
    if f.exists(): return pd.read_pickle(f)
    d = yf.download(t, start=start, auto_adjust=True, progress=False)
    d = d["Close"].squeeze() if "Close" in d else d.squeeze()
    d.to_pickle(f); return d
