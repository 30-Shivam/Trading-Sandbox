"""Descriptive risk comparison of candidate allocations vs the frozen 85% SPY / 15% QQQ policy. NOT a hypothesis test: it claims no
alpha, registers no trial, and changes no policy. Mixes are fixed here BEFORE any result was seen. Monthly rebalance, no costs/taxes,
USD total-return ETF prices (a CAD investor also carries USD/CAD moves). QLD = 2x daily QQQ. BTC-USD is spot (a Canadian TFSA would hold a
spot-BTC ETF, which tracks it less fees). Windows exist because BTC's full history is short and flatters it: we also show buying at the
2017 and 2021 peaks. Run: python research/allocation_mixes/compare.py > scratchpad/allocation_mixes.log"""
import sys, pathlib
import numpy as np, pandas as pd, yfinance as yf
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parent / "etf_overlays"))
from metrics import stats  # noqa: E402

TICKS = ["SPY", "QQQ", "GLD", "QLD", "BTC-USD"]
MIXES = {
    "A frozen 85SPY/15QQQ":        {"SPY": .85, "QQQ": .15},
    "B 100 QQQ":                   {"QQQ": 1.0},
    "C 70QQQ/20SPY/10GLD":         {"QQQ": .70, "SPY": .20, "GLD": .10},
    "D 75SPY/15QQQ/10BTC":         {"SPY": .75, "QQQ": .15, "BTC-USD": .10},
    "E 70QQQ/20SPY/10BTC":         {"QQQ": .70, "SPY": .20, "BTC-USD": .10},
    "F 60QQQ/20SPY/10BTC/10GLD":   {"QQQ": .60, "SPY": .20, "BTC-USD": .10, "GLD": .10},
    "G 50QQQ/30QLD/20SPY":         {"QQQ": .50, "QLD": .30, "SPY": .20},
    "H 100 QLD (2x, the cliff)":   {"QLD": 1.0},
}
WINDOWS = [("W1 BTC era: 2014-09-17 -> today (all mixes)", "2014-09-17", None),
           ("W2 incl. 2008 (no-BTC mixes only): 2007-01-02 -> today", "2007-01-02", "noBTC"),
           ("W3 BTC bought at 2017 peak: 2017-12-17 -> today", "2017-12-17", "BTC+A"),
           ("W4 BTC bought at 2021 peak: 2021-11-10 -> today", "2021-11-10", "BTC+A")]


def run(rets: pd.DataFrame, w: dict, start: pd.Timestamp) -> pd.Series:
    r = rets.loc[start:, list(w)]; wt = pd.Series(w); cur, prev, out = wt.copy(), None, np.zeros(len(r)); mon = r.index.to_period("M")
    for i, row in enumerate(r.to_numpy()):
        if mon[i] != prev: cur, prev = wt.copy(), mon[i]
        pr = float((cur.to_numpy() * row).sum()); out[i] = pr; cur = cur * (1 + row) / (1 + pr)
    return pd.Series(out, index=r.index)


def metrics(p: pd.Series) -> dict:
    s = stats(p); eq = (1 + p).cumprod(); yr = p.groupby(p.index.year); full = yr.size() >= 200
    worst_year = (yr.apply(lambda x: (1 + x).prod() - 1))[full].min() if full.any() else np.nan
    roll = eq.pct_change(252).min(); under = eq < eq.cummax(); runs = (under != under.shift()).cumsum()[under]
    longest = (runs.groupby(runs).apply(lambda g: (g.index[-1] - g.index[0]).days).max() / 365.25) if len(runs) else 0.0
    return dict(cagr=s["cagr"], maxdd=s["maxdd"], vol=s["vol"], worst_year=worst_year, worst_12m=roll, under_yrs=longest, x10k=10000 * s["final"])


def main():
    d = yf.download(TICKS, start="1999-03-10", auto_adjust=True, progress=False, threads=False)["Close"]
    d = d.reindex(d["SPY"].dropna().index).ffill(); rets = d.pct_change(fill_method=None)
    first = {t: d[t].first_valid_index() for t in TICKS}
    print("data start:", {t: str(v.date()) for t, v in first.items()}, "| last", d.index[-1].date())
    for title, start, kind in WINDOWS:
        rows = []
        for name, w in MIXES.items():
            has_btc = "BTC-USD" in w
            if kind == "noBTC" and has_btc: continue
            if kind == "BTC+A" and not (has_btc or name.startswith("A ") or name.startswith("E ")): continue
            st = max(pd.Timestamp(start), max(first[t] for t in w) + pd.Timedelta(days=1))
            p = run(rets, w, st).dropna(); m = metrics(p); rows.append((name, str(p.index[0].date()), m))
        print(f"\n=== {title}")
        print(f"{'mix':<30}{'from':>12}{'CAGR':>8}{'vol':>7}{'maxDD':>8}{'worstYr':>9}{'worst12m':>9}{'underwater':>11}{'$10k ->':>11}")
        for name, s0, m in rows:
            print(f"{name:<30}{s0:>12}{m['cagr']:>8.1%}{m['vol']:>7.1%}{m['maxdd']:>8.1%}{m['worst_year']:>9.1%}{m['worst_12m']:>9.1%}"
                  f"{m['under_yrs']:>9.1f}yr{m['x10k']:>11,.0f}")


if __name__ == "__main__":
    main()
