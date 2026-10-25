"""Odd-lot tender offer feasibility screen (spec: docs/ODDLOT_SPEC.md). Resumable; caches under ./cache (gitignored).

    python research/oddlot_tender/classify.py index      # stage 1: listed SC TO-I originals -> cache/listed.json
    python research/oddlot_tender/classify.py fetch      # stage 2: download filing text (rate-limited, resumable)
    python research/oddlot_tender/classify.py classify   # stage 3: flags + prices -> cache/events.csv
    python research/oddlot_tender/classify.py report     # stage 4: summary vs frozen kill rules
"""
import gzip, json, pathlib, sys, time

import pandas as pd
import requests

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; (CACHE / "filings").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import parse as P            # noqa: E402
import sec_fundamentals as sf  # noqa: E402  (USER_AGENT)

S = requests.Session(); S.headers["User-Agent"] = sf.USER_AGENT
LISTED_EXCHANGES = {"Nasdaq", "NYSE", "CBOE"}
QUARTERS = [(y, q) for y in range(2021, 2027) for q in range(1, 5) if not (y == 2026 and q > 3)]


def get(url, **kw):
    for k in range(4):
        try:
            r = S.get(url, timeout=(10, 100), **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    return None


def stage_index():
    ex = get("https://www.sec.gov/files/company_tickers_exchange.json").json()
    by_cik = {}
    for cik, name, ticker, exch in ex["data"]:
        if exch in LISTED_EXCHANGES:
            by_cik.setdefault(int(cik), (ticker, exch))
    rows, seen, unmapped = [], set(), 0
    for y, q in QUARTERS:
        r = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.gz")
        if r is None:
            print("FAILED index", y, q); continue
        for line in gzip.decompress(r.content).decode("latin-1").splitlines():
            if not line.startswith("SC TO-I ") or line.startswith("SC TO-I/A"):
                continue
            rec = P.parse_form_idx_line(line)
            if not rec or rec["accession"] in seen:
                continue
            seen.add(rec["accession"])
            if rec["cik"] in by_cik:
                rec["ticker"], rec["exchange"] = by_cik[rec["cik"]]
                rows.append(rec)
            else:
                unmapped += 1
        time.sleep(0.2)
    json.dump({"rows": rows, "unmapped_or_unlisted": unmapped, "total_originals": len(seen)}, open(CACHE / "listed.json", "w"))
    print(f"originals={len(seen)} listed_by_exchange_map={len(rows)} unmapped_or_unlisted={unmapped}")


def stage_fetch():
    rows = json.load(open(CACHE / "listed.json"))["rows"]
    for i, r in enumerate(rows):
        f = CACHE / "filings" / f"{r['accession']}.txt"
        if f.exists():
            continue
        resp = get("https://www.sec.gov/Archives/" + r["path"])
        f.write_text(resp.content.decode("latin-1") if resp is not None else "")
        time.sleep(0.15)
        if i % 50 == 0:
            print(i, len(rows), flush=True)
    print("fetched")


def filing_text(path: pathlib.Path) -> str:
    import re
    from lxml import html
    raw = path.read_text(errors="replace")
    parts = []
    for doc in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, re.S):
        dtype = re.search(r"<TYPE>(\S+)", doc)
        if dtype and dtype.group(1).upper().startswith(("GRAPHIC", "PDF", "ZIP", "EXCEL", "XML", "JSON", "COVER")):
            continue  # uuencoded binaries otherwise pollute regex matching
        body = doc.split("<TEXT>", 1)[-1]
        try:
            parts.append(html.fromstring(body).text_content() if "<" in body[:2000] else body)
        except Exception:
            parts.append(re.sub(r"<[^>]+>", " ", body))
    return P.normalize(" ".join(parts))


def closes(ticker: str, lo: str, hi: str) -> pd.Series | None:
    """Daily Close for [lo, hi) as it was QUOTED then: Yahoo's Close is adjusted for later splits, so multiply by the
    product of split ratios that happened after each date (verified necessary: COKE/MNST/ELAB showed pre-split offers
    against post-split prices)."""
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(start=lo, end=pd.Timestamp.today().strftime("%Y-%m-%d"), interval="1d",
                                       auto_adjust=False, actions=True, timeout=20)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.index = df.index.tz_localize(None).normalize()
    splits = df["Stock Splits"][df["Stock Splits"] > 0]
    close = df["Close"].dropna()
    factor = pd.Series([float(splits[splits.index > d].prod()) if (splits.index > d).any() else 1.0 for d in close.index], index=close.index)
    adj = close * factor
    return adj[adj.index < pd.Timestamp(hi)]


def stage_classify():
    rows = json.load(open(CACHE / "listed.json"))["rows"]
    out = []
    for i, r in enumerate(rows):
        f = CACHE / "filings" / f"{r['accession']}.txt"
        text = filing_text(f) if f.exists() and f.stat().st_size else ""
        terms = P.extract_price_terms(text) if text else {"type": "nofile", "price": None, "high": None}
        rec = {**{k: r[k] for k in ("accession", "company", "ticker", "exchange", "filed")},
               "year": int(r["filed"][:4]), "odd_lot": P.detect_odd_lot(text) if text else False,
               "price_type": terms["type"], "offer": terms["price"], "high": terms["high"],
               "stated_price": P.stated_market_price(text) if text else None, "cad": P.is_cad(text) if text else False,
               "prev_close": None, "next_close": None, "spread_pct": None, "net_usd": None, "net_usd_t1": None}
        if rec["odd_lot"] and terms["price"]:
            filed = pd.Timestamp(r["filed"])
            c = closes(r["ticker"], (filed - pd.Timedelta(days=12)).strftime("%Y-%m-%d"), (filed + pd.Timedelta(days=8)).strftime("%Y-%m-%d"))
            if c is not None:
                before, after = c[c.index < filed], c[c.index > filed]
                if len(before):
                    rec["prev_close"] = float(before.iloc[-1])
                    rec["spread_pct"] = P.gross_spread_pct(terms["price"], rec["prev_close"])
                    rec["net_usd"] = P.net_payoff_usd(terms["price"], rec["prev_close"])
                if len(after):
                    rec["next_close"] = float(after.iloc[0])
                    rec["net_usd_t1"] = P.net_payoff_usd(terms["price"], rec["next_close"])
            time.sleep(0.3)
        out.append(rec)
        if i % 25 == 0:
            print(i, len(rows), flush=True)
    pd.DataFrame(out).to_csv(CACHE / "events.csv", index=False)
    print("classified", len(out))


# Hand-read 2026-10-24 (price language in each filing): the extracted price/security is NOT the common-stock tender.
MANUAL_FALSE_POSITIVES = {
    ("WHLR", "2021-04-19"): "tender is for Series D preferred, not the common",
    ("NHP", "2026-05-18"): "tender is for Series A/B preferred, not the common",
    ("PLTK", "2022-08-29"): "$23-27 is from historical bidder indications, not the tender price",
    ("ELAB", "2024-10-04"): "$0.38 is a warrant exercise price",
}


def _tier_summary(name, q):
    events = [dict(year=int(r.year), spread_pct=r.spread_pct, net_usd=r.net_usd) for r in q.itertuples()]
    s = P.summarize(events)
    yrs = {y: int((q.year == y).sum()) for y in range(2021, 2027)}
    print(f"\n[{name}] events={len(q)} per-year={yrs}")
    print(f"   median spread {s['median_spread_pct']:.2f}% | median net ${s['median_net_usd']:.0f} | median yearly count (2021-25) "
          f"{s['median_yearly_count']:.0f} | Rule1 {'FAIL' if s['rule1_fail'] else 'ok'} | Rule2 {'FAIL' if s['rule2_fail'] else 'ok'} "
          f"| {'PASS' if s['passed'] else 'FAIL'}")
    return s


def stage_report():
    meta = json.load(open(CACHE / "listed.json"))
    df = pd.read_csv(CACHE / "events.csv")
    print(f"SC TO-I originals {meta['total_originals']}; exchange-listed {len(df)}; unmapped/unlisted {meta['unmapped_or_unlisted']}")
    print("\nprice_type x odd_lot (listed filings):\n", pd.crosstab(df["price_type"], df["odd_lot"]))
    df["price_mismatch"] = (df.stated_price.notna() & df.prev_close.notna() & ((df.stated_price / df.prev_close - 1).abs() > 0.15))
    base = df[df.odd_lot & df.price_type.isin(["fixed", "range"])]
    pos = base[base.spread_pct > 0]
    print(f"\nUPPER BOUND: odd-lot fixed/range, positive spread, NO data-quality filters: {len(pos)} events "
          f"= {len(pos) / 5.75:.1f}/yr over 5.75 yrs (Rule 1 needs >= 12/yr)")
    A = pos[~pos.cad & ~pos.price_mismatch]
    manual = [(r.ticker, r.filed) in MANUAL_FALSE_POSITIVES for r in A.itertuples()]
    B = A[[not m for m in manual]]
    C_ = B[~(B.stated_price.isna() & (B.spread_pct > 20))]
    sA = _tier_summary("A mechanical (FROZEN verdict basis)", A)
    _tier_summary("B hand-cleaned (4 false positives removed)", B)
    _tier_summary("C strict (B minus unverifiable rows)", C_)
    print("\nPer-year table, tier B:")
    print("year | qualifying | median spread % | median net $ | nav-linked odd-lot filings | neg/zero-spread odd-lot priced")
    for y in range(2021, 2027):
        d = df[df.year == y]; b = B[B.year == y]
        neg = int(((d.odd_lot) & d.price_type.isin(["fixed", "range"]) & (d.spread_pct <= 0)).sum())
        nav = int(((d.odd_lot) & (d.price_type == "nav")).sum())
        print(f"{y} | {len(b)} | {b.spread_pct.median():.2f} | {b.net_usd.median():.0f} | {nav} | {neg}")
    t1 = B.dropna(subset=["net_usd_t1"])
    print(f"\nSENSITIVITY tier B, entry = first close AFTER filing: positive-payoff events {int((t1.net_usd_t1 > 0).sum())} of {len(t1)}, "
          f"median net ${t1.net_usd_t1.median():.0f}")
    print("\nTier B events:"); print(B[["ticker", "filed", "price_type", "offer", "prev_close", "spread_pct", "net_usd", "net_usd_t1"]].round(2).to_string(index=False))
    print("\nFROZEN VERDICT (tier A):", "PASS" if sA["passed"] else "FAIL")


if __name__ == "__main__":
    {"index": stage_index, "fetch": stage_fetch, "classify": stage_classify, "report": stage_report}[sys.argv[1]]()
