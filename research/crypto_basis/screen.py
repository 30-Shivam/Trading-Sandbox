"""Pure metrics for the crypto-basis funding screen (frozen spec: docs/CRYPTO_BASIS_SPEC.md). No network, no files.

Input funding series: pd.Series of per-interval funding rates (fraction, e.g. 0.0001 = 0.01%) indexed by UTC timestamp.
The simulated carry position is long spot / short perp, so it EARNS +rate (and pays when rate < 0).
"""
import numpy as np
import pandas as pd

INTERVAL = pd.Timedelta(hours=8)
FEE_MODELS = {   # name -> (cost per round trip as fraction of notional, round trips per year)
    "primary_taker_0.05%_x4legs_1RT": (4 * 0.0005, 1),
    "maker_0.02%_1RT": (4 * 0.0002, 1),
    "taker_12RT": (4 * 0.0005, 12),
    "maker_12RT": (4 * 0.0002, 12),
    "spot0.10%+perp0.05%_1RT": (2 * 0.0010 + 2 * 0.0005, 1),
}
PRIMARY = "primary_taker_0.05%_x4legs_1RT"
KILL_MIN_NET_SPREAD_PCT = 4.0
KILL_MAX_UNDERWATER_DAYS = 365
CAPITAL_MARGIN = 0.20


def yearly_table(funding: pd.Series, cash_pct: pd.Series, years=range(2021, 2027)) -> pd.DataFrame:
    """One row per calendar year: gross/net annualized yield on notional (%), cash (%), net spread over cash (%)."""
    f = funding.sort_index()
    last_end = f.index[-1] + INTERVAL
    rows = []
    for y in years:
        start, end = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        end = min(end, last_end)
        chunk = f[(f.index >= start) & (f.index < end)]
        if chunk.empty or end <= start:
            continue
        days = (end - start) / pd.Timedelta(days=1)
        gross = float(chunk.sum() * 365.0 / days * 100.0)
        cash_chunk = cash_pct[(cash_pct.index >= start.tz_localize(None)) & (cash_pct.index < end.tz_localize(None))]
        cash = float(cash_chunk.mean()) if len(cash_chunk) else float("nan")
        row = {"year": y, "days": round(days, 1), "intervals": len(chunk), "gross_pct": gross,
               "negative_interval_share_pct": float((chunk < 0).mean() * 100.0), "cash_pct": cash}
        for name, (per_rt, n_rt) in FEE_MODELS.items():
            row[f"net[{name}]"] = gross - per_rt * n_rt * 100.0
            row[f"spread[{name}]"] = row[f"net[{name}]"] - cash
        row["spread_on_capital_primary"] = row[f"spread[{PRIMARY}]"] / (1.0 + CAPITAL_MARGIN)
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def underwater(funding: pd.Series, since: str = "2022-01-01") -> dict:
    """Cumulative funding income (% of notional, fees excluded) from `since`; max drawdown (pct-points) and the longest
    spell (days) spent below a previous peak. A spell ends when the curve makes a new high."""
    f = funding[funding.index >= pd.Timestamp(since, tz="UTC")].sort_index()
    if f.empty:
        return {"max_drawdown_pp": None, "longest_underwater_days": None, "underwater_at_end": None, "final_cum_pct": None}
    cum = f.cumsum() * 100.0
    peak = cum.cummax().clip(lower=0.0)            # the starting balance (0) counts as the first peak
    dd = float((peak - cum).max())
    longest, spell_start, last_peak_t = 0.0, None, f.index[0]
    for t, c, p in zip(cum.index, cum.values, peak.values):
        if c >= p:                                  # at/above prior peak: spell (if any) ends
            if spell_start is not None:
                longest = max(longest, (t - spell_start) / pd.Timedelta(days=1))
                spell_start = None
            last_peak_t = t
        elif spell_start is None:
            spell_start = last_peak_t
    at_end = spell_start is not None
    if at_end:
        longest = max(longest, (cum.index[-1] - spell_start) / pd.Timedelta(days=1))
    return {"max_drawdown_pp": dd, "longest_underwater_days": float(longest), "underwater_at_end": at_end,
            "final_cum_pct": float(cum.iloc[-1])}


def evaluate_kill_rules(tables: dict[str, pd.DataFrame], uw: dict[str, dict], first_year: int = 2022) -> dict:
    """tables: asset -> yearly_table; uw: asset -> underwater(). Applies the frozen rules on the PRIMARY fee model."""
    per_asset = {}
    for asset, t in tables.items():
        sel = t.loc[t.index >= first_year, f"spread[{PRIMARY}]"].dropna()
        per_asset[asset] = float(sel.mean())
    avg = float(np.mean(list(per_asset.values())))
    worst_uw = max(u["longest_underwater_days"] for u in uw.values())
    rule1_fail = avg < KILL_MIN_NET_SPREAD_PCT
    rule2_fail = worst_uw > KILL_MAX_UNDERWATER_DAYS
    return {"avg_net_spread_pct": avg, "per_asset_pct": per_asset, "worst_underwater_days": worst_uw,
            "rule1_fail": rule1_fail, "rule2_fail": rule2_fail, "dropped": rule1_fail or rule2_fail}
