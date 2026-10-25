"""Funding-rate pull + report for the crypto-basis screen (frozen spec: docs/CRYPTO_BASIS_SPEC.md). Zero trials.

    python research/crypto_basis/pull.py pull      # Binance fapi funding history -> cache/*.csv, ^IRX -> cache/irx.csv
    python research/crypto_basis/pull.py report    # yearly table per asset + kill-rule verdict
"""
import pathlib, sys, time

import pandas as pd
import requests

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import screen as S  # noqa: E402

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding(symbol: str, start: str = "2021-01-01") -> pd.Series:
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    out = []
    while True:
        for attempt in range(4):
            r = requests.get(URL, params={"symbol": symbol, "startTime": cursor, "limit": 1000}, timeout=30)
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"{symbol}: HTTP {r.status_code} {r.text[:120]}")
        rows = r.json()
        if not rows:
            break
        out += rows
        cursor = rows[-1]["fundingTime"] + 1
        if len(rows) < 1000:
            break
        time.sleep(0.3)
    idx = pd.to_datetime([x["fundingTime"] for x in out], unit="ms", utc=True).round("s")
    s = pd.Series([float(x["fundingRate"]) for x in out], index=idx, name=symbol)
    return s[~s.index.duplicated()].sort_index()


def stage_pull():
    import yfinance as yf
    for asset, sym in SYMBOLS.items():
        s = fetch_funding(sym)
        s.to_csv(CACHE / f"{asset}_funding.csv")
        gaps = (s.index.to_series().diff().dropna() > pd.Timedelta(hours=9)).sum()
        print(f"{asset}: {len(s)} intervals {s.index[0]} -> {s.index[-1]}; gaps>9h: {int(gaps)}")
    irx = yf.download("^IRX", start="2021-01-01", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    irx.to_csv(CACHE / "irx.csv", header=["irx_pct"])
    print("IRX", len(irx), irx.index[0].date(), irx.index[-1].date())


def stage_report():
    irx = pd.read_csv(CACHE / "irx.csv", index_col=0, parse_dates=True)["irx_pct"]
    tables, uw = {}, {}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for asset in SYMBOLS:
        f = pd.read_csv(CACHE / f"{asset}_funding.csv", index_col=0, parse_dates=True).iloc[:, 0]
        f.index = pd.DatetimeIndex(f.index).tz_convert("UTC") if f.index.tz else pd.DatetimeIndex(f.index).tz_localize("UTC")
        tables[asset], uw[asset] = S.yearly_table(f, irx), S.underwater(f)
        t = tables[asset]
        print(f"\n=== {asset} (Binance {SYMBOLS[asset]}) ===")
        print(t[["days", "gross_pct", "negative_interval_share_pct", "cash_pct", f"net[{S.PRIMARY}]", f"spread[{S.PRIMARY}]",
                 "spread_on_capital_primary"]].round(2).to_string())
        print("fee sensitivities, net spread over cash:")
        print(t[[c for c in t.columns if c.startswith("spread[") and S.PRIMARY not in c]].round(2).to_string())
        print("cumulative-funding curve since 2022:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in uw[asset].items()})
    v = S.evaluate_kill_rules(tables, uw)
    print("\nKILL RULES (primary fees, notional basis):")
    print(f"  Rule 1: avg net spread over cash 2022+ = {v['avg_net_spread_pct']:.2f} pp (BTC {v['per_asset_pct']['BTC']:.2f}, "
          f"ETH {v['per_asset_pct']['ETH']:.2f}) vs 4.0 -> {'FAIL' if v['rule1_fail'] else 'ok'}")
    print(f"  Rule 2: worst longest-underwater spell = {v['worst_underwater_days']:.0f} days vs 365 -> {'FAIL' if v['rule2_fail'] else 'ok'}")
    print("VERDICT:", "DROP (no trial registered)" if v["dropped"] else "survives the screen: worth a deeper look only")
    flips = []
    for name in S.FEE_MODELS:
        avg = float(sum(t.loc[t.index >= 2022, f"spread[{name}]"].mean() for t in tables.values()) / len(tables))
        flips.append((name, round(avg, 2)))
    print("avg spread by fee model:", flips)
    cap = float(sum(t.loc[t.index >= 2022, "spread_on_capital_primary"].mean() for t in tables.values()) / len(tables))
    print(f"capital basis (20% short margin): avg spread {cap:.2f} pp")


if __name__ == "__main__":
    {"pull": stage_pull, "report": stage_report}[sys.argv[1]]()
