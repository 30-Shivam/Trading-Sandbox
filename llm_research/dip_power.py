"""Phase 0 power screen for docs/DIP_SPEC.md. COUNTS ONLY: it never computes a forward return and spends no trial.
Builds an augmented PIT price panel (cache/prices_dip.pkl: prices_pit + cached Tiingo + one Yahoo batch for PIT names
the 10-K-based build dropped), finds trigger events (>=15% below the 126-session high, above a rising SMA200, 20-session
cooldown from the previous entry), and reports counts, clustering and how many a 5%-NAV / 20-slot book could actually take.
Liquidity filter (needs volume) is NOT applied: the price cache is adjusted closes only (disclosed in the spec outcome).
Run: python llm_research/dip_power.py > scratchpad/dip_phase0.log"""
import sys, time, pathlib
import numpy as np, pandas as pd, yfinance as yf
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from sp500_membership import load_membership_table
from pit_utils import make_membership

DD, HI_WIN, SMA_WIN, SMA_LAG, COOLDOWN, HOLD, SLOTS = 0.85, 126, 200, 20, 20, 20, 20
START = pd.Timestamp("2010-01-01")


def build_panel(names: set[str]) -> tuple[pd.DataFrame, dict]:
    px = pd.read_pickle(CACHE / "prices_pit.pkl"); info = {"base": px.shape[1]}
    missing = sorted(names - set(px.columns)); added = {}
    for t in missing:
        f = CACHE / "tiingo" / f"{t}.pkl"
        if f.exists():
            s = pd.read_pickle(f)["adjClose"].dropna()
            if len(s) > 300: added[t] = s
    info["tiingo_added"] = len(added)
    yok = set(pd.read_csv(CACHE / "yahoo_ok.csv").iloc[:, 0]); ytick = [t for t in missing if t in yok and t not in added]
    ny = 0
    for i in range(0, len(ytick), 60):
        batch = ytick[i:i + 60]
        for attempt in range(3):
            try:
                d = yf.download(batch, start="2009-06-01", auto_adjust=True, progress=False, threads=False)["Close"]; break
            except Exception as exc:
                print(f"  yahoo retry {attempt}: {exc}", flush=True); time.sleep(20); d = None
        if d is None: continue
        if isinstance(d, pd.Series): d = d.to_frame(batch[0])
        for c in d.columns:
            if d[c].notna().sum() > 300: added[c] = d[c]; ny += 1
        time.sleep(3)
    info["yahoo_added"] = ny
    P = pd.concat([px, pd.DataFrame(added)], axis=1).sort_index()
    info["still_missing"] = sorted(names - set(P.columns))
    return P, info


def triggers(s: pd.Series) -> pd.DatetimeIndex:
    s = s.dropna()
    if len(s) < SMA_WIN + SMA_LAG: return pd.DatetimeIndex([])
    hi = s.rolling(HI_WIN).max(); sma = s.rolling(SMA_WIN).mean()
    ok = (s <= DD * hi) & (s > sma) & (sma > sma.shift(SMA_LAG)) & (s.index >= START)
    out, last = [], -10**9
    pos = {d: i for i, d in enumerate(s.index)}
    for d in s.index[ok.values]:
        if pos[d] - last > COOLDOWN: out.append(d); last = pos[d]
    return pd.DatetimeIndex(out)


def main():
    table = load_membership_table(); dates, sets = make_membership(table)
    names = set().union(*[set(table[table["date"] >= f"{y}-01-01"].iloc[0]["tickers"].split(",")) for y in range(2010, 2026)])
    P, info = build_panel(names)
    P.to_pickle(CACHE / "prices_dip.pkl")
    print(f"panel: base {info['base']} + tiingo {info['tiingo_added']} + yahoo {info['yahoo_added']} = {P.shape[1]} series; "
          f"PIT names 2010-2025 {len(names)}, still missing {len(info['still_missing'])}: {info['still_missing']}")
    idx = P.index; snap_i = np.searchsorted(dates, idx.values, side="right") - 1; snap_i = np.maximum(snap_i, 0)
    last_ok = idx[-1 - (HOLD + 1)]                       # events later than this lack a full 20-session window
    ev = []
    for t in P.columns:
        if t == "SPY": continue
        ts = triggers(P[t])
        if not len(ts): continue
        member = np.fromiter((t in sets[i] for i in snap_i), bool, len(idx)); m = pd.Series(member, idx)
        for d in ts:
            if m.get(d, False) and d <= last_ok: ev.append((t, d))
    ev = pd.DataFrame(ev, columns=["ticker", "date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    ev["year"] = ev.date.dt.year; ev["ym"] = ev.date.dt.to_period("M")
    n, n19 = len(ev), int((ev.year >= 2019).sum())
    print(f"\nRAW EVENTS (PIT member, 15%/126/SMA200, cooldown): total {n} | 2019+ {n19}")
    print("per year:", ev.groupby("year").size().to_dict())
    bym = ev.groupby("ym").size().sort_values(ascending=False)
    print(f"distinct months with >=1 event: {ev.ym.nunique()} | top-3 months hold {bym.head(3).sum() / n:.0%} of events: "
          f"{[(str(k), int(v)) for k, v in bym.head(3).items()]}")
    bt = ev.ticker.value_counts()
    print(f"distinct tickers: {len(bt)} | top ticker {bt.index[0]} {bt.iloc[0]} ({bt.iloc[0] / n:.1%}) | top-10 tickers hold {bt.head(10).sum() / n:.0%}")
    per_day = ev.groupby("date").size()
    print(f"triggers/day: median {per_day.median():.0f}, max {per_day.max()} on {per_day.idxmax().date()} | days with >{SLOTS}: {int((per_day > SLOTS).sum())}")
    # takeable under 20 slots: entry t+1, hold 20 sessions, deeper drawdown first (drawdown computed from price only)
    pos = {d: i for i, d in enumerate(idx)}
    ev["dd"] = [float(P[t].loc[d] / P[t].loc[:d].tail(HI_WIN).max() - 1) for t, d in zip(ev.ticker, ev.date)]
    open_exits, taken = [], 0
    for d, g in ev.groupby("date"):
        i = pos[d]; open_exits = [e for e in open_exits if e > i + 1]
        for _, r in g.sort_values("dd").iterrows():
            if len(open_exits) >= SLOTS: break
            open_exits.append(i + 1 + HOLD); taken += 1
    print(f"TAKEABLE with {SLOTS} slots, 20-session hold: {taken} of {n} raw events")
    verdict = "PASS (run the full test, register trial #46)" if n >= 300 and n19 >= 100 else "FAIL: underpowered, DROP without spending a trial"
    print(f"\nPHASE 0 VERDICT (spec: >=300 total AND >=100 in 2019+): {verdict}")


if __name__ == "__main__":
    main()
