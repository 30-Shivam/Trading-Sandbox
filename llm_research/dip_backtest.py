"""Full test for docs/DIP_SPEC.md (trial `trend_dip_buy`). FROZEN spec: no parameter here may be changed after seeing results.
Needs cache/prices_dip.pkl from `python llm_research/dip_power.py` (a local cache this repo built itself; pickle is safe).
Primary: entry = close of signal day + 1, hold 20 sessions, 5% NAV/slot, 20 slots, deeper drawdown first, remainder in SPY,
10 bps/side. Decision: block-bootstrap t >= 3.09, maxDD no worse than SPY, 2019+ same sign, and beats matched random entries
(same dates, ticker replaced by a random PIT member). Registers the trial in validation_trials_registry.txt unless --no-register.
Run: python llm_research/dip_backtest.py > scratchpad/dip_backtest.log"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; CACHE = ROOT / "cache"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, str(ROOT.parent / "research" / "etf_overlays"))

HOLD, SLOTS, SLOT, COST, HI_WIN = 20, 20, 0.05, 0.0010, 126
T_BAR, N_RANDOM, MIN_HISTORY, STRATEGY = 3.09, 2000, 252, "trend_dip_buy"


def simulate(ev_k, ev_j, order, shift, R, rs, valid, last_valid, hold=HOLD, slots=SLOTS, slot=SLOT, cost=COST):
    """Pure. ev_k/ev_j: ticker column and signal-day row per event; `order`: event indices in processing order (day, then
    priority); entry row = signal row + shift. R: daily returns (days x tickers, NaN->0); rs: SPY daily returns; valid: price
    exists; last_valid: last priced row per ticker (delisted names exit at their last close). Returns (daily excess vs SPY
    array, taken, skipped_full). Excess accounting: slot * (r_stock - r_spy) while held, `cost` per side on the stock leg."""
    n = R.shape[0]; ex = np.zeros(n); active, taken, skipped = [], 0, 0
    for e in order:
        k, j = ev_k[e], ev_j[e] + shift
        if j + 1 >= n: continue
        active = [a for a in active if a > j]
        if len(active) >= slots: skipped += 1; continue
        if not valid[j, k]: continue
        end = min(j + hold, n - 1, last_valid[k])
        if end <= j: continue
        active.append(end); taken += 1
        ex[j + 1:end + 1] += slot * (R[j + 1:end + 1, k] - rs[j + 1:end + 1])
        ex[j] -= slot * cost; ex[end] -= slot * cost
    return ex, taken, skipped


def load_panel():
    P = pd.read_pickle(CACHE / "prices_dip.pkl"); spy = P["SPY"].dropna(); px = P.reindex(spy.index)
    tick = [c for c in px.columns if c != "SPY"]; V = px[tick].to_numpy(float)
    R = np.nan_to_num(px[tick].pct_change(fill_method=None).to_numpy(float), nan=0.0)
    rs = spy.pct_change(fill_method=None).fillna(0.0).to_numpy(float); valid = ~np.isnan(V); n = len(px)
    return px, tick, V, R, rs, valid, valid.argmax(axis=0), n - 1 - valid[::-1].argmax(axis=0)


def membership_matrix(days, tick):
    from pit_utils import make_membership
    from sp500_membership import load_membership_table
    mdates, msets = make_membership(load_membership_table())
    snap = np.maximum(np.searchsorted(mdates, days.values, side="right") - 1, 0); M = np.zeros((len(days), len(tick)), bool)
    for k, t in enumerate(tick): M[:, k] = np.fromiter((t in msets[i] for i in snap), bool, len(days))
    return M


def find_events(px, tick, M):
    from dip_power import triggers
    n = len(px); last_ok = n - 1 - (HOLD + 1); ek, ej, edd = [], [], []
    for k, t in enumerate(tick):
        s = px[t].dropna(); ts = triggers(s)
        if not len(ts): continue
        dd = s / s.rolling(HI_WIN).max() - 1
        for d in ts:
            j = px.index.get_loc(d)
            if M[j, k] and j <= last_ok: ek.append(k); ej.append(j); edd.append(float(dd[d]))
    return np.array(ek), np.array(ej), np.array(edd)


def summarize(ex, start, r_spy_full, label, days):
    from metrics import stats, excess_t
    exs = pd.Series(ex, index=days).iloc[start:]; r_spy = pd.Series(r_spy_full, index=days).iloc[start:]; port = r_spy + exs
    sp, pp = stats(r_spy), stats(port); ann, t, p = excess_t(port, r_spy)
    h = exs.index >= "2019-01-01"; a2, t2, p2 = excess_t(port[h], r_spy[h])
    print(f"   {label}: CAGR {pp['cagr']:.2%} vs SPY {sp['cagr']:.2%} | maxDD {pp['maxdd']:.1%} vs {sp['maxdd']:.1%} | "
          f"excess {ann:.2%}/yr t={t:.2f} p={p:.4f} | 2019+ excess {a2:.2%}/yr t={t2:.2f}")
    return dict(ann=ann, t=t, p=p, dd_ok=pp["maxdd"] >= sp["maxdd"], ann19=a2, t19=t2)


def random_baseline(ev_k, ev_j, shift, start, R, rs, valid, last_valid, M, first_valid, rng):
    n = R.shape[0]; cache = {}

    def pool(j):
        if j not in cache:
            jn = min(j + shift, n - 1)
            cache[j] = np.flatnonzero(M[j] & valid[j] & valid[jn] & (j - first_valid >= MIN_HISTORY))
        return cache[j]
    pools = [pool(j) for j in ev_j]; keep = np.array([len(p) > 0 for p in pools]); out = np.empty(N_RANDOM)
    for s in range(N_RANDOM):
        ks = np.array([p[rng.integers(len(p))] if len(p) else 0 for p in pools])
        order = [i for i in np.lexsort((rng.random(len(ev_j)), ev_j)) if keep[i]]
        ex, _, _ = simulate(ks, ev_j, order, shift, R, rs, valid, last_valid)
        out[s] = ex[start:].mean() * 252
    return out


def named_ticker_look():
    """Descriptive only (spec): LITE and CSU.TO dip events under the same trigger. Hindsight-selected names, never evidence."""
    import yfinance as yf
    from dip_power import triggers
    print("\n== DESCRIPTIVE LOOK (hindsight-selected, NOT evidence, never a promotion input)")
    d = yf.download(["LITE", "CSU.TO", "SPY"], start="2009-06-01", auto_adjust=True, progress=False, threads=False)["Close"]
    for t in ("LITE", "CSU.TO"):
        if t not in d or d[t].dropna().empty: print(f"   {t}: no data returned"); continue
        s = d[t].dropna(); f20 = s.shift(-HOLD) / s - 1; ts = [x for x in triggers(s) if not np.isnan(f20.get(x, np.nan))]
        if not ts: print(f"   {t}: no qualifying dip events with a full 20-session window"); continue
        fw = f20.loc[ts]
        print(f"   {t}: {len(ts)} dip events ({s.index[0].date()}..{s.index[-1].date()}) | mean fwd20 after dip {fw.mean():+.2%} (median {fw.median():+.2%}, "
              f"{(fw > 0).mean():.0%} positive) vs ALL-days mean fwd20 {f20.mean():+.2%}" + ("  [CAD price; SPY is USD]" if t.endswith(".TO") else ""))


def main(register=True):
    px, tick, V, R, rs, valid, first_valid, last_valid = load_panel(); days = px.index
    M = membership_matrix(days, tick); ek, ej, edd = find_events(px, tick, M)
    print(f"panel {len(tick)} tickers, {days[0].date()}..{days[-1].date()} | raw events {len(ek)} (2019+: {int((days[ej] >= '2019-01-01').sum())})")
    res = {}
    for shift, label in ((1, "PRIMARY entry close(t+1)"), (0, "sensitivity entry close(t)")):
        order = np.lexsort((edd, ej)); ex, taken, skipped = simulate(ek, ej, order, shift, R, rs, valid, last_valid)
        start = int(ej.min()) + shift; print(f"\n== {label}: taken {taken}, skipped(full) {skipped}, from {days[start].date()}")
        res[shift] = summarize(ex, start, rs, "REAL", days)
        if shift == 1:
            rng = np.random.default_rng(0); rnd = random_baseline(ek, ej, shift, start, R, rs, valid, last_valid, M, first_valid, rng)
            real = res[1]["ann"]; p_rand = (1 + int((rnd >= real).sum())) / (N_RANDOM + 1)
            res[1]["p_rand"] = p_rand
            print(f"   RANDOM same-date entries x{N_RANDOM}: excess mean {rnd.mean():.2%}/yr sd {rnd.std():.2%} | REAL {real:.2%}/yr -> p(random >= real) = {p_rand:.4f}")
    r = res[1]; n_trials = None
    if register:
        try:
            from benchmark_random_entry import bonferroni_check
            n_trials = bonferroni_check(STRATEGY, r["p"]); print(f"\nREGISTERED: {n_trials}")
        except Exception as exc:
            print(f"\n[WARN] could not register trial in validation_trials_registry.txt ({exc}); add `{STRATEGY}` by hand")
    crit = {"t >= 3.09": r["t"] >= T_BAR, "maxDD no worse than SPY": r["dd_ok"], "2019+ same sign (excess > 0)": r["ann19"] > 0,
            f"beats random entries (p<=0.0011)": r["p_rand"] <= 0.0011}
    print("\nDECISION (all required):"); [print(f"   [{'PASS' if v else 'FAIL'}] {k}") for k, v in crit.items()]
    print(f"VERDICT: {'PASS (PROVISIONAL: 41 PIT names with no price data, see spec)' if all(crit.values()) else 'FAIL (final for this specification)'}")
    print("Secondary 'expensive stocks' arm: DROPPED per spec (no unadjusted closes in the price panel).")
    try: named_ticker_look()
    except Exception as exc: print(f"   [WARN] descriptive look skipped: {exc}")


if __name__ == "__main__":
    main(register="--no-register" not in sys.argv)
