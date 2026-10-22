"""Monthly-rebalanced, long-only factor portfolios vs SPY.

Why: swing signals with stop/target exits dilute factor premia, which are
documented for hold-and-rebalance portfolios. This tests the factors the way the
academic evidence defines them, against the objective that matters: beat SPY.

For each month-end, rank the universe on a PRICE-ONLY factor, hold the top
quintile equal-weight for the next month, charge costs on turnover, and compare
against (a) SPY and (b) the equal-weight universe (the control: any long-only
equal-weight basket gets a size/survivorship tilt, so a factor must beat THAT too).

Caveats stated up front: the universe is today's constituents (survivorship bias
flatters every row, including the control); prices are adjusted closes.

Usage:
    python factor_portfolios.py [--universe universe.json] [--years 8] [--cost-bps 10] [--top 0.2]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import market_data
import universe as universe_mod

TRADING_DAYS = 252


def fetch_prices(tickers: list[str], years: int) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        data = market_data._with_retry(
            market_data.yf.download, chunk, period=f"{years}y", interval="1d", auto_adjust=True,
            progress=False, group_by="ticker", threads=True, timeout=60,
        )
        for t in chunk:
            try:
                s = (data[t] if isinstance(data.columns, pd.MultiIndex) else data)["Close"].dropna()
                if len(s) > 300:
                    frames.append(s.rename(t))
            except KeyError:
                continue
    return pd.concat(frames, axis=1).sort_index()


def factor_scores(prices: pd.DataFrame, volumes_unused=None) -> dict[str, pd.DataFrame]:
    """Month-end factor panels (higher = more attractive). Price-only, no lookahead:
    each value uses data up to and including that month-end."""
    me = prices.resample("ME").last()
    daily_ret = prices.pct_change()
    scores = {
        "momentum_12_1": me.shift(1) / me.shift(12) - 1,                     # skip most recent month
        "momentum_6_1": me.shift(1) / me.shift(6) - 1,
        "low_vol_12m": -daily_ret.rolling(252).std().resample("ME").last(),
        "reversal_1m": -(me / me.shift(1) - 1),                              # short-term reversal
        "near_52w_high": me / prices.rolling(252).max().resample("ME").last(),
        "trend_above_200d": me / prices.rolling(200).mean().resample("ME").last() - 1,
    }
    return scores


def run_portfolio(me_prices: pd.DataFrame, score: pd.DataFrame, top_frac: float, cost_bps: float) -> pd.Series:
    """Monthly returns of the top-`top_frac` equal-weight basket, net of turnover costs."""
    rets = me_prices.pct_change()
    out, prev = {}, pd.Series(dtype=float)
    dates = score.index
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        s = score.loc[d].dropna()
        if len(s) < 50 or nxt not in rets.index:
            continue
        picks = s.nlargest(max(int(len(s) * top_frac), 5)).index
        w = pd.Series(1.0 / len(picks), index=picks)
        r = rets.loc[nxt, picks].fillna(0.0)
        turnover = (w.sub(prev, fill_value=0.0).abs().sum()) / 1.0
        out[nxt] = float((w * r).sum() - turnover * cost_bps / 1e4)
        prev = w
    return pd.Series(out)


def stats(m: pd.Series, bench: pd.Series | None = None) -> dict:
    m = m.dropna()
    eq = (1 + m).cumprod()
    yrs = len(m) / 12
    res = {
        "CAGR%": 100 * (eq.iloc[-1] ** (1 / yrs) - 1), "vol%": 100 * m.std() * math.sqrt(12),
        "Sharpe": m.mean() / m.std() * math.sqrt(12) if m.std() > 0 else float("nan"),
        "maxDD%": 100 * (eq / eq.cummax() - 1).min(),
    }
    if bench is not None:
        b = bench.reindex(m.index).dropna(); x = m.reindex(b.index)
        ex = x - b
        res["excess%/yr"] = 100 * ex.mean() * 12
        res["t(excess)"] = ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex))) if len(ex) > 2 and ex.std() > 0 else float("nan")
        beta = np.cov(x, b)[0, 1] / b.var()
        res["beta"] = beta
        res["alpha%/yr"] = 100 * (x.mean() - beta * b.mean()) * 12
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=universe_mod.DEFAULT_UNIVERSE_FILE)
    ap.add_argument("--caps", default="", help="comma list of large,mid,small (default: all)")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--cost-bps", type=float, default=10.0, help="one-way cost per unit turnover")
    ap.add_argument("--top", type=float, default=0.2)
    ap.add_argument("--eval-years", type=int, default=5)
    ap.add_argument("--pit-sp500", action="store_true",
                    help="Point-in-time S&P 500: at each month-end only rank names that were index members THEN "
                         "(sp500_historical_components.csv). Removes the lookahead of using today's constituents; "
                         "delisted names yfinance can't serve remain a (upward) bias, reported as coverage.")
    args = ap.parse_args()

    caps = {c for c in args.caps.split(",") if c} or None
    member_mask = None
    if args.pit_sp500:
        import sp500_membership
        table = sp500_membership.load_membership_table()
        month_ends = pd.date_range(pd.Timestamp.today() - pd.DateOffset(years=args.years), pd.Timestamp.today(), freq="ME")
        members = {d: {universe_mod.normalize_ticker(t) for t in sp500_membership.get_membership_asof(d, table)} for d in month_ends}
        tickers = sorted(set().union(*members.values()))
    else:
        tickers = [m["ticker"] for m in universe_mod.load_universe(args.universe, caps)]
    print(f"Fetching {len(tickers)} tickers x {args.years}y ...")
    prices = fetch_prices(tickers, args.years)
    if args.pit_sp500:
        have = set(prices.columns)
        cov = pd.Series({d: len(m & have) / max(len(m), 1) for d, m in members.items()})
        print(f"PIT coverage (share of then-members with price data): mean {cov.mean():.1%}, min {cov.min():.1%} "
              f"({cov.idxmin().date()}), latest {cov.iloc[-1]:.1%}")
        member_mask = pd.DataFrame({d: pd.Series(True, index=sorted(m & have)) for d, m in members.items()}).T.reindex(
            columns=prices.columns).fillna(False).astype(bool)
    spy = fetch_prices(["SPY"], args.years)["SPY"]
    print(f"Got {prices.shape[1]} tickers with usable history, {prices.index[0].date()}..{prices.index[-1].date()}")

    me = prices.resample("ME").last()
    spy_m = spy.resample("ME").last().pct_change()
    ew = me.pct_change().mean(axis=1)                          # equal-weight universe control (monthly rebalanced)
    start = me.index[-1] - pd.DateOffset(years=args.eval_years)
    rows = {"SPY": stats(spy_m[start:]), "EW universe (control)": stats(ew[start:], spy_m)}
    if member_mask is not None:  # EW control also restricted to then-members
        mm = member_mask.reindex(me.index, method="nearest", tolerance=pd.Timedelta(days=5)).fillna(False).astype(bool)
        ew = me.pct_change().where(mm.shift(1, fill_value=False)).mean(axis=1)
        rows["EW universe (control)"] = stats(ew[start:], spy_m)
    for name, sc in factor_scores(prices).items():
        if member_mask is not None:
            sc = sc.where(mm)
        pm = run_portfolio(me, sc, args.top, args.cost_bps)
        row = stats(pm[start:], spy_m)
        row["excess_vs_EW%/yr"] = 100 * (pm[start:] - ew).dropna().mean() * 12
        rows[f"top{int(args.top*100)}% {name}"] = row
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(f"\nEvaluation window: last {args.eval_years}y, monthly rebalance, {args.cost_bps:.0f}bps/turnover, long-only equal-weight")
    print(out.round(2).to_string())
    print("\nA factor is interesting only if excess%/yr > 0 with |t| >= ~2 AND it also beats the EW control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
