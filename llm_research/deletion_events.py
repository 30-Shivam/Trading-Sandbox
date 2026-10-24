"""DESCRIPTIVE event count for the S&P 500 deletion-rebound idea. Looks at NO post-removal returns.
Removal = ticker present in membership snapshot i, absent in i+1 (event date = snapshot i+1 date).
Classification uses only whether/when the price series ENDS. Volatility for the power calc is the unconditional
20-day excess-vs-SPY std of the same tickers, with all windows within +-40 trading days of their removal excluded."""
import sys, pathlib, numpy as np, pandas as pd, yfinance as yf
ROOT = pathlib.Path(__file__).resolve().parents[1]; C = ROOT / "llm_research" / "cache"

def removals(table):
    sets = [set(x.split(",")) for x in table["tickers"]]; out = []
    for i in range(len(sets) - 1):
        for t in sets[i] - sets[i + 1]: out.append((t, table["date"].iloc[i + 1]))
    return pd.DataFrame(out, columns=["ticker", "date"])

def load_prices(tickers):
    px = {}
    for t in tickers:
        f = C / "tiingo" / f"{t}.pkl"
        if f.exists():
            d = pd.read_pickle(f)
            if len(d): px[t] = d["adjClose"].dropna() if "adjClose" in d else d["close"].dropna()
    rest = [t for t in tickers if t not in px]
    for i in range(0, len(rest), 80):
        d = yf.download(rest[i:i + 80], start="1999-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(d, pd.Series): d = d.to_frame(rest[i])
        for t in d.columns:
            s = d[t].dropna()
            if len(s): px[t] = s
    return px

def classify(s, date):
    """exclude_ended (<=5 td after), ambiguous (6-19), surviving (>=20), no_data, pre_prices (series starts after date)."""
    if s is None: return "no_data"
    if s.index[0] > date: return "pre_prices"
    after = s.index[s.index > date]
    if len(after) <= 5: return "exclude_ended"
    if len(after) < 20: return "ambiguous"
    return "surviving"

if __name__ == "__main__":
    t = pd.read_csv(ROOT / "sp500_historical_components.csv", parse_dates=["date"]).sort_values("date")
    r = removals(t); r = r[r.date >= "2000-01-01"].reset_index(drop=True)
    px = load_prices(sorted(r.ticker.unique()))
    spy = yf.download("SPY", start="1999-06-01", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    r["cls"] = [classify(px.get(a), d) for a, d in zip(r.ticker, r.date)]
    r["period"] = np.where(r.date < "2013-01-01", "design<2013", "holdout2013+")
    print(r.groupby(["period", "cls"]).size().unstack(fill_value=0))
    r.to_csv(C / "deletion_events_classified.csv", index=False)
    surv = r[r.cls == "surviving"]
    # unconditional 20d excess vol, windows non-overlapping, excluding +-40 td of that ticker's removal
    vols = {}
    for per in ("design<2013", "holdout2013+", "all"):
        ex = []
        for a, d in zip(surv.ticker[surv.period == per] if per != "all" else surv.ticker,
                        surv.date[surv.period == per] if per != "all" else surv.date):
            s = px[a]; idx = s.index; k = idx.searchsorted(d)
            for st in range(0, len(idx) - 20, 20):
                if abs(st - k) < 60: continue
                d0, d1 = idx[st], idx[st + 20]
                if d0 in spy.index and d1 in spy.index: ex.append(s.iloc[st + 20] / s.iloc[st] - spy[d1] / spy[d0])
        vols[per] = float(np.std(ex)) if ex else np.nan
    for per in ("design<2013", "holdout2013+"):
        n = int((surv.period == per).sum()); sd = vols[per]
        print(f"{per}: surviving N={n}, unconditional 20d excess sd={sd:.3%}")
        for z in (3.09, 2.58, 1.65): print(f"   min mean 20d excess for t={z}: {z*sd/np.sqrt(max(n,1)):.2%}")
    n = len(surv); sd = vols["all"]
    print(f"pooled N={n} sd={sd:.3%}: t=3.09 needs {3.09*sd/np.sqrt(n):.2%}; t=2.58 needs {2.58*sd/np.sqrt(n):.2%}")
