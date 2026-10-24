"""Forward (out-of-sample) tracker for the frozen SEC-ratio ridge (llm_research/frozen_model.json).

Each run: (1) scores every NEW 10-K filed since the model was frozen by an S&P 500 name and stores the score in
Mongo `Forward_Ratio_Scores` (all scores, not just picks, so forward rank-IC is measurable); (2) settles entries
whose 126-trading-day window has completed (excess return vs SPY from the close 2 trading days after filing).
Never trades or allocates capital -- evidence only. `--report` prints forward IC / pick-vs-rest and the gate status.

Usage: python ratio_tracker.py [--dry-run] [--report] [--weekly] [--since 2026-01-01]
  --weekly   full scan only on Mondays (the scan takes ~5 min: ~490 EDGAR fetches); other days just report.
  --dry-run  compute and print everything in memory; no Mongo writes (and no Mongo needed).
"""
import argparse, json, pathlib, sys, time
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm_research"))
COLLECTION = "Forward_Ratio_Scores"
MODEL_PATH = ROOT / "llm_research" / "frozen_model.json"


def load_model(path=MODEL_PATH) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def score_rows(rows: list[dict], model: dict) -> list[dict]:
    """Pure: attach ridge score and pick flag using the FROZEN mu/sd/weights/threshold."""
    feats = model["features"]; mu, sd, w = (np.array(model[k]) for k in ("mu", "sd", "weights"))
    out = []
    for r in rows:
        x = np.array([r.get(f, np.nan) for f in feats], dtype=float)
        z = np.nan_to_num(np.clip((x - mu) / sd, -3, 3))
        score = float(z @ w)
        out.append({"ticker": r["ticker"], "filed": pd.Timestamp(r["filed"]).strftime("%Y-%m-%d"),
                    "fy_end": pd.Timestamp(r["fy_end"]).strftime("%Y-%m-%d"), "score": score,
                    "pick": bool(score > model["threshold"]), "settled": False})
    return out


def compute_label(stock: pd.Series, spy: pd.Series, filed: str, hold_days: int = 126) -> dict | None:
    """Pure: entry = close of 2nd trading day after `filed`, exit = hold_days later. None until the window completes."""
    s = stock.dropna()
    idx = s.index.searchsorted(pd.Timestamp(filed) + pd.Timedelta(days=1)) + 1
    if idx + hold_days >= len(s):
        return None
    d0, d1 = s.index[idx], s.index[idx + hold_days]
    if d0 not in spy.index or d1 not in spy.index:
        return None
    ret, bench = s.iloc[idx + hold_days] / s.iloc[idx] - 1, spy[d1] / spy[d0] - 1
    return {"entry": d0.strftime("%Y-%m-%d"), "exit": d1.strftime("%Y-%m-%d"), "fwd_ret": float(ret),
            "spy_ret": float(bench), "excess": float(ret - bench)}


def forward_report(docs: list[dict]) -> str:
    """Pure: summarize settled docs (needs 'excess', 'score', 'pick')."""
    done = pd.DataFrame([d for d in docs if d.get("excess") is not None])
    lines = [f"RATIO TRACKER: {len(docs)} scored, {len(done)} settled, "
             f"{int(sum(d.get('pick', False) for d in docs))} picks flagged"]
    if len(done) < 30:
        return "\n".join(lines + ["  too few settled filings for statistics (need >= 30; >= 150 for any real verdict)"])
    ic = done.score.rank().corr(done.excess.rank())
    rng = np.random.default_rng(0); null = np.array([done.score.sample(frac=1, random_state=int(i)).reset_index(drop=True).rank().corr(done.excess.rank()) for i in rng.integers(0, 10**6, 2000)])
    p = (null >= ic).mean(); pk, rest = done[done.pick], done[~done.pick]
    lines.append(f"  forward rank-IC {ic:+.3f} (perm p={p:.3f}, n={len(done)})")
    lines.append(f"  picks: n={len(pk)} mean 6m excess {pk.excess.mean() if len(pk) else float('nan'):+.2%} | rest: n={len(rest)} {rest.excess.mean():+.2%}")
    lines.append("  gate status: UNPROVEN until >= 150 settled, IC p < 0.0012 (Bonferroni, 42 trials) and a daily-portfolio drawdown check")
    return "\n".join(lines)


def _universe(cik_map):
    from sp500_membership import get_membership_asof
    names = get_membership_asof(pd.Timestamp.today().normalize())
    return [t for t in names if t in cik_map]


def run(dry_run: bool, since: str, report_only: bool) -> int:
    import yfinance as yf
    import sec_fundamentals as sf
    from edgar_features import facts_for, rows_for
    model = load_model(); db = None
    if not dry_run:
        import storage
        try:
            db = storage.get_db()[COLLECTION]
            db.create_index([("ticker", 1), ("fy_end", 1)], unique=True)
        except storage.MongoNotConfigured as exc:
            print(f"[ERROR] {exc}", file=sys.stderr); return 1
    docs = {(d["ticker"], d["fy_end"]): d for d in db.find({}, {"_id": 0})} if db is not None else {}
    if not report_only:
        cik = sf.fetch_cik_map(); tickers = _universe(cik); new = 0; fail = 0
        print(f"scanning {len(tickers)} S&P 500 names for 10-Ks filed since {since}")
        for t in tickers:
            try:
                rows = [r for r in rows_for(t, facts_for(t, cik[t])) if pd.Timestamp(r["filed"]) >= pd.Timestamp(since)]
            except Exception:
                fail += 1; continue
            for d in score_rows(rows, model):
                key = (d["ticker"], d["fy_end"])
                if key not in docs:
                    docs[key] = d; new += 1
                    if db is not None: db.update_one({"ticker": d["ticker"], "fy_end": d["fy_end"]}, {"$setOnInsert": d}, upsert=True)
        print(f"new scored filings: {new} (fetch/parse failures: {fail})")
        # settle completed windows
        pend = [d for d in docs.values() if not d.get("settled") and pd.Timestamp(d["filed"]) < pd.Timestamp.today() - pd.Timedelta(days=185)]
        if pend:
            tk = sorted({d["ticker"] for d in pend}) + ["SPY"]; px = {}
            for i in range(0, len(tk), 80):
                data = yf.download(tk[i:i+80], start=since, auto_adjust=True, progress=False, threads=True)["Close"]
                if isinstance(data, pd.Series): data = data.to_frame(tk[i])
                px.update({c: data[c] for c in data.columns}); time.sleep(1)
            for d in pend:
                if d["ticker"] not in px: continue
                lab = compute_label(px[d["ticker"]], px["SPY"], d["filed"], model["hold_days"])
                if lab:
                    d.update(lab, settled=True)
                    if db is not None: db.update_one({"ticker": d["ticker"], "fy_end": d["fy_end"]}, {"$set": {**lab, "settled": True}})
            print(f"settled this run: {sum(1 for d in pend if d.get('settled'))}")
    print(forward_report(list(docs.values())))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--since", default="2026-01-01"); ap.add_argument("--weekly", action="store_true")
    a = ap.parse_args()
    report_only = a.report or (a.weekly and pd.Timestamp.today().weekday() != 0)
    sys.exit(run(a.dry_run, a.since, report_only))


if __name__ == "__main__":
    main()
