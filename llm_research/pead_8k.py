"""PEAD: Item 2.02 8-K + SUE(net income) + reaction-day return, FROZEN spec in docs/PEAD_SPEC.md. Run as a script.
Pieces (quarter_series, assign_reaction_day, etc.) are pure so tests/test_pead_8k.py can exercise them."""
import sys, json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT.parent / "research" / "etf_overlays"))
HOLD, SLOT, MAXPOS, COST, WIN_DAYS, MIN_HIST_EVENTS, Q = 20, 0.05, 20, 0.0010, 365, 200, 0.80

def quarter_series(units):
    """units: companyfacts NetIncomeLoss USD rows -> DataFrame[end, val, rep_filed] of standalone quarters, FIRST-REPORTED values.
    Q1-Q3 = 3-month rows; Q4 = 12-month (10-K) minus the 9-month YTD (same start) . Never uses later restatements."""
    first = {}
    for r in sorted(units, key=lambda r: r["filed"]):
        if r.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A") or "start" not in r: continue
        k = (r["start"], r["end"])
        if k not in first: first[k] = r
    out = []
    for (s, e), r in first.items():
        d = (pd.Timestamp(e) - pd.Timestamp(s)).days
        if 80 <= d <= 100: out.append((pd.Timestamp(e), r["val"], pd.Timestamp(r["filed"])))
        elif 350 <= d <= 380:
            ytd = [x for (s2, e2), x in first.items() if s2 == s and 260 <= (pd.Timestamp(e2) - pd.Timestamp(s2)).days <= 290]
            if ytd: out.append((pd.Timestamp(e), r["val"] - ytd[0]["val"], pd.Timestamp(r["filed"])))
    q = pd.DataFrame(out, columns=["end", "val", "rep_filed"]).drop_duplicates("end").sort_values("end").reset_index(drop=True)
    return q

def sue_table(q):
    """SUE_q = (X_q - X_{q-4}) / std(prior 8 seasonal changes). Needs contiguous quarters (~91d apart)."""
    if len(q) < 13: return pd.DataFrame(columns=["end", "sue", "rep_filed"])
    gap_ok = q.end.diff().dt.days.between(80, 105).values
    d = np.full(len(q), np.nan)
    for i in range(4, len(q)):
        if gap_ok[i - 3:i + 1].all() and 350 <= (q.end[i] - q.end[i - 4]).days <= 380: d[i] = q.val[i] - q.val[i - 4]
    rows = []
    for i in range(12, len(q)):
        prior = d[i - 8:i]
        if np.isnan(d[i]) or np.isnan(prior).any(): continue
        sd = prior.std(ddof=1)
        if sd > 0: rows.append((q.end[i], d[i] / sd, q.rep_filed[i]))
    return pd.DataFrame(rows, columns=["end", "sue", "rep_filed"])

def assign_reaction_day(acc_utc, trading_days):
    """EDGAR acceptanceDateTime is UTC. Convert to ET: accepted before 16:00 ET -> reaction day = that calendar day (or next trading
    day if not a session); at/after 16:00 ET -> next trading day. Entry/measurement are always AFTER acceptance, so this is look-ahead-safe."""
    t = pd.Timestamp(acc_utc).tz_convert("America/New_York"); day = pd.Timestamp(t.date())
    if t.hour >= 16: day = day + pd.Timedelta(days=1)
    i = trading_days.searchsorted(day)
    return trading_days[i] if i < len(trading_days) else None

def match_8k(end, rep_filed, k_rows):
    """First 2.02 8-K accepted (ET date) after quarter end and on/before the periodic report, within 45d of it."""
    best = None
    for a in k_rows:
        d = pd.Timestamp(a).tz_convert("America/New_York").tz_localize(None).normalize()
        if end < d <= rep_filed and (rep_filed - d).days <= 45 and (best is None or a < best): best = a
    return best

def trailing_cut(react_dates, vals, i, q=Q):
    """80th pct of vals over events with react date in [react_i - 365d, react_i) ; None if < MIN_HIST_EVENTS."""
    lo = react_dates[i] - pd.Timedelta(days=WIN_DAYS)
    m = (react_dates >= lo) & (react_dates < react_dates[i]) & ~np.isnan(vals)
    return None if m.sum() < MIN_HIST_EVENTS else float(np.quantile(vals[m], q))

def simulate(trig, px, spy, entry_shift=1):
    """trig: DataFrame[ticker, react, sue]. Entry = close of react+entry_shift, exit HOLD sessions later (or last close if delisted).
    SLOT of NAV per position, MAXPOS concurrent, remainder in SPY, excess accounting 0.05*(r_i - r_spy), COST per side on the stock leg.
    Returns daily excess series vs SPY, count taken, count skipped."""
    days = spy.index; r_spy = spy.pct_change(fill_method=None); ex = pd.Series(0.0, index=days); active, taken, skipped = [], 0, 0
    rets = px.pct_change(fill_method=None)
    for row in trig.sort_values(["react", "sue"], ascending=[True, False]).itertuples():
        s = px[row.ticker].dropna(); j = days.searchsorted(row.react) + entry_shift
        if j + 1 >= len(days): continue
        d0 = days[j]
        active = [e for e in active if e > d0]
        if len(active) >= MAXPOS: skipped += 1; continue
        if d0 not in s.index: continue
        k = min(j + HOLD, len(days) - 1); d1 = days[k]
        end_i = s.index.searchsorted(d1, side="right") - 1; d1 = min(d1, s.index[end_i])
        active.append(d1); taken += 1
        seg = days[(days > d0) & (days <= d1)]
        ex.loc[seg] += SLOT * (rets[row.ticker].reindex(seg).fillna(0.0) - r_spy.reindex(seg).fillna(0.0))
        ex.loc[d0] -= SLOT * COST; ex.loc[d1] -= SLOT * COST
    return ex, taken, skipped

if __name__ == "__main__":
    import sec_fundamentals as sf
    from pit_utils import make_membership, member_asof
    from sp500_membership import load_membership_table
    from metrics import stats, excess_t
    table = load_membership_table(); mdates, msets = make_membership(table)
    px = pd.read_pickle(CACHE / "prices_pit.pkl"); spy = px["SPY"].dropna(); px = px.reindex(spy.index)
    days = spy.index; ret = px.pct_change(fill_method=None); exr = ret.sub(ret["SPY"], axis=0)
    ev, diag = [], dict(no_facts=0, quarters=0, no_sue=0, no_8k=0, member_no=0, no_react=0, lag10q=[])
    for f in sorted((CACHE / "sub8k").glob("*.json")):
        sub = json.load(open(f)); t = sub["ticker"]; cf = CACHE / f.name
        if t not in px.columns or not cf.exists(): diag["no_facts"] += 1; continue
        j = json.load(open(cf)); u = next((j[c]["units"]["USD"] for c in ("NetIncomeLoss", "ProfitLoss") if c in j and "USD" in j[c]["units"]), None)
        if not u: diag["no_facts"] += 1; continue
        st = sue_table(quarter_series(u)); k_rows = [r["acc"] for r in sub["rows"] if r["form"] == "8-K"]; diag["quarters"] += len(st)
        for r in st.itertuples():
            a = match_8k(r.end, r.rep_filed, k_rows)
            if a is None: diag["no_8k"] += 1; continue
            rd = assign_reaction_day(a, days)
            if rd is None or rd <= days[70]: diag["no_react"] += 1; continue
            if not member_asof(t, rd, mdates, msets): diag["member_no"] += 1; continue
            i = days.get_loc(rd); sd60 = exr[t].iloc[i - 60:i].std()
            R = exr[t].iloc[i]
            diag["lag10q"].append((r.rep_filed - pd.Timestamp(a).tz_convert("America/New_York").tz_localize(None).normalize()).days)
            if np.isnan(R) or np.isnan(sd60) or sd60 == 0: continue
            ev.append(dict(ticker=t, react=rd, sue=r.sue, z=R / sd60))
    ev = pd.DataFrame(ev).sort_values("react").reset_index(drop=True)
    lag = np.array(diag.pop("lag10q")); print("DIAG", diag, "| 8-K->periodic-report lag days: median", np.median(lag), "p90", np.percentile(lag, 90))
    rd_, sue_, z_ = ev.react.values.astype("datetime64[ns]"), ev.sue.values, ev.z.values
    rd_ts = pd.DatetimeIndex(rd_); trig = []
    for i in range(len(ev)):
        cs, cz = trailing_cut(rd_ts, sue_, i), trailing_cut(rd_ts, z_, i)
        if cs is not None and cz is not None and sue_[i] >= cs and z_[i] >= cz: trig.append(i)
    trig = ev.iloc[trig].copy(); print(f"events {len(ev)} | triggers {len(trig)} | per year", trig.groupby(trig.react.dt.year).size().to_dict())
    for shift, label in ((1, "PRIMARY entry close(react+1)"), (0, "sensitivity entry close(react)")):
        exs, taken, skipped = simulate(trig, px, spy, shift)
        start = trig.react.min() if not len(trig) == 0 else days[0]; exs = exs[exs.index >= start]
        r_spy = spy.pct_change(fill_method=None).reindex(exs.index); port = r_spy + exs
        sp, pp = stats(r_spy), stats(port); ann, t_, p_ = excess_t(port, r_spy)
        print(f"\n== {label}: taken {taken}, skipped(full) {skipped}, from {exs.index[0].date()}")
        print(f"   CAGR {pp['cagr']:.2%} vs SPY {sp['cagr']:.2%} | maxDD {pp['maxdd']:.1%} vs {sp['maxdd']:.1%} | excess {ann:.2%}/yr t={t_:.2f} p={p_:.4f}")
        h = exs.index >= "2019-01-01"; a2, t2, p2 = excess_t(port[h], r_spy[h]); print(f"   2019+: excess {a2:.2%}/yr t={t2:.2f} p={p2:.4f}")
