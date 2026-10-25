"""Absolute-return / market-neutral evaluation: excess return over CASH, not over SPY.

Sits beside index_gate.py (which asks "did it beat holding SPY?"). This module asks the question for a sleeve that is
not meant to track the equity market: "does it earn a statistically real return above T-bills, smoothly enough to be
worth holding next to the 85/15 core?" Inputs are DAILY return series (a NAV curve converted with nav_to_returns), so it
suits continuous carry/overlay strategies that the trade-by-trade index gate does not fit.

Everything here is pure (no network, no files). Cash comes from `irx_to_daily_returns` (13-week T-bill yield, ^IRX) or a
BIL total-return series via `nav_to_returns`.
"""
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_ZERO_VOL_REL = 1e-10            # std below this fraction of |mean| is treated as zero volatility

# The user-stated sleeve targets (2026-10-24): uncorrelated +8% absolute, Sharpe >= 1.5, max drawdown <= 5%.
CASH_GATE_MIN_SHARPE = 1.5
CASH_GATE_MIN_CAGR_PCT = 8.0
CASH_GATE_MAX_DRAWDOWN_PCT = 5.0
CASH_GATE_ALPHA = 0.05            # one-sided, before the Bonferroni division by the number of trials ever tested
CASH_GATE_MIN_YEARS = 3.0         # fewer years than this cannot support the Sharpe claim (see sharpe_se / min_years)


def nav_to_returns(nav: pd.Series) -> pd.Series:
    """Daily simple returns from a NAV / price / total-return-index series."""
    nav = pd.Series(nav).dropna().sort_index()
    return nav.pct_change().dropna()


def irx_to_daily_returns(irx_pct: pd.Series, days_per_year: int = TRADING_DAYS) -> pd.Series:
    """^IRX quotes the 13-week T-bill yield in percent per year. Daily cash return = yield / 100 / days_per_year,
    forward-filled across gaps (the yield is a rate, not a price). Simple (not compounded) daily accrual."""
    s = pd.Series(irx_pct).dropna().sort_index().astype(float)
    return (s / 100.0 / days_per_year).rename("cash")


def excess_returns(strategy: pd.Series, cash: pd.Series) -> pd.Series:
    """strategy - cash on the strategy's own dates; cash is forward-filled onto them. Dates before the first cash
    observation are dropped rather than assumed zero (a silent zero would flatter the strategy)."""
    strategy = pd.Series(strategy).dropna().sort_index()
    cash_aligned = pd.Series(cash).sort_index().reindex(strategy.index, method="ffill")
    both = pd.concat([strategy, cash_aligned], axis=1, keys=["s", "c"]).dropna()
    return (both["s"] - both["c"]).rename("excess")


def annualized_sharpe(daily: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float | None:
    """mean/std (ddof=1) * sqrt(periods). None if fewer than 2 observations or zero variance."""
    x = np.asarray(daily, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return None
    sd, mean = x.std(ddof=1), x.mean()
    if not math.isfinite(sd) or sd <= _ZERO_VOL_REL * max(abs(mean), 1e-12):
        return None   # a constant series has float noise ~1e-18, not exactly 0 -- would give a Sharpe of ~1e16
    return float(mean / sd * math.sqrt(periods))


def sharpe_se(sharpe: float, years: float) -> float:
    """Large-sample standard error of an ANNUAL Sharpe under iid returns: sqrt((1 + 0.5 SR^2) / years).
    Optimistic when returns are autocorrelated or fat-tailed -- hence the bootstrap below."""
    return math.sqrt((1.0 + 0.5 * sharpe ** 2) / years)


def min_years_for_z(sharpe: float, z: float) -> float:
    """Years of history needed for `sharpe` to reach t-stat `z` under the iid approximation."""
    return (z ** 2) * (1.0 + 0.5 * sharpe ** 2) / (sharpe ** 2)


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: index paths of length n. At each step, with probability 1/mean_block start a
    new block at a uniformly random position, otherwise continue the previous block (wrapping around)."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def default_block_length(n: int) -> float:
    """Rule-of-thumb mean block length ~ n^(1/3) (floor 5 days): long enough to keep short-range autocorrelation."""
    return max(5.0, n ** (1.0 / 3.0))


def bootstrap_sharpe(excess: pd.Series, n_boot: int = 2000, mean_block: float | None = None, seed: int = 42,
                     ci: float = 0.95) -> dict | None:
    """Stationary-block-bootstrap distribution of the annualized Sharpe of `excess`.
    Returns point estimate, percentile CI, and p_le_zero (share of resamples with Sharpe <= 0) -- the bootstrap analogue
    of a one-sided p-value for H0: Sharpe <= 0. Resolution of p is 1/n_boot."""
    x = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = len(x)
    if n < 30:
        return None
    mean_block = mean_block or default_block_length(n)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
    samples = x[idx]
    sd = samples.std(axis=1, ddof=1)
    mean = samples.mean(axis=1)
    ok = sd > _ZERO_VOL_REL * np.maximum(np.abs(mean), 1e-12)
    sharpes = np.full(n_boot, np.nan)
    sharpes[ok] = mean[ok] / sd[ok] * math.sqrt(TRADING_DAYS)
    sharpes = sharpes[~np.isnan(sharpes)]
    if len(sharpes) == 0:
        return None
    lo, hi = np.percentile(sharpes, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"sharpe": annualized_sharpe(x), "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci,
            "p_le_zero": float((sharpes <= 0).mean()), "n_boot": n_boot, "mean_block": float(mean_block)}


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + pd.Series(returns).dropna()).cumprod()


def max_drawdown_pct(returns: pd.Series) -> float | None:
    """Worst peak-to-trough decline of the compounded curve, as a positive percent (5.0 = a 5% drawdown)."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return None
    curve = equity_curve(r)
    peak = curve.cummax().clip(lower=1.0)   # start-of-series value (1.0) counts as a peak
    return float(((peak - curve) / peak).max() * 100.0)


def cagr_pct(returns: pd.Series) -> float | None:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return None
    years = len(r) / TRADING_DAYS
    return float((float(equity_curve(r).iloc[-1]) ** (1.0 / years) - 1.0) * 100.0)


def calmar(returns: pd.Series) -> float | None:
    """CAGR% / max drawdown%. None if there is no drawdown to divide by."""
    dd, g = max_drawdown_pct(returns), cagr_pct(returns)
    if dd is None or g is None or dd == 0:
        return None
    return g / dd


def tail_metrics(returns: pd.Series) -> dict:
    """Skew-sensitive numbers a Sharpe hides: worst day, 5% CVaR of daily returns, skewness."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"worst_day_pct": None, "cvar_95_daily_pct": None, "skew": None}
    cutoff = np.percentile(r, 5)
    return {"worst_day_pct": float(r.min() * 100), "cvar_95_daily_pct": float(r[r <= cutoff].mean() * 100),
            "skew": float(r.skew()) if len(r) > 2 else None}


def audit_vs_cash(strategy_returns: pd.Series, cash_returns: pd.Series, n_boot: int = 2000, seed: int = 42,
                  mean_block: float | None = None) -> dict | None:
    """Full absolute-return audit of a DAILY return series against cash. Returns None if there is too little overlap.

    Reports both views: absolute (CAGR, max drawdown, Calmar of the strategy itself) and excess-over-cash (annualized
    excess, Sharpe with a stationary-bootstrap CI, iid standard error, years of history vs the years needed)."""
    s = pd.Series(strategy_returns).dropna().sort_index()
    ex = excess_returns(s, cash_returns)
    if len(ex) < 30:
        return None
    s = s.loc[ex.index]
    years = len(ex) / TRADING_DAYS
    boot = bootstrap_sharpe(ex, n_boot=n_boot, mean_block=mean_block, seed=seed)
    sharpe = annualized_sharpe(ex)
    out = {
        "n_days": int(len(ex)), "years": years,
        "start": str(ex.index[0])[:10], "end": str(ex.index[-1])[:10],
        "cagr_pct": cagr_pct(s), "max_drawdown_pct": max_drawdown_pct(s), "calmar": calmar(s),
        "annualized_excess_pct": float(ex.mean() * TRADING_DAYS * 100), "sharpe_excess": sharpe,
        "sharpe_se_iid": sharpe_se(sharpe, years) if sharpe is not None else None,
        "bootstrap": boot, "tail": tail_metrics(s),
    }
    if sharpe is not None and sharpe > 0:
        out["years_needed_z1_65"] = min_years_for_z(sharpe, 1.65)
    return out


def cash_gate_verdict(audit: dict | None, n_trials: int = 1, min_sharpe: float = CASH_GATE_MIN_SHARPE,
                      min_cagr_pct: float = CASH_GATE_MIN_CAGR_PCT, max_drawdown_pct: float = CASH_GATE_MAX_DRAWDOWN_PCT,
                      alpha: float = CASH_GATE_ALPHA, min_years: float = CASH_GATE_MIN_YEARS) -> dict:
    """The capital gate for an absolute-return sleeve. ALL of these must hold; the first failure is reported:
      1. enough history (>= min_years);
      2. Sharpe over cash is statistically > 0 at the Bonferroni-corrected level alpha / n_trials (bootstrap p);
      3. point-estimate Sharpe >= min_sharpe;
      4. CAGR >= min_cagr_pct and max drawdown <= max_drawdown_pct.
    `n_trials` should be the registry count INCLUDING this strategy (see benchmark_random_entry.bonferroni_check)."""
    if not audit:
        return {"passed": False, "reason": "insufficient overlapping data"}
    boot = audit.get("bootstrap")
    if audit["years"] < min_years:
        return {"passed": False, "reason": f"only {audit['years']:.1f} years of data (< {min_years})"}
    if not boot or audit.get("sharpe_excess") is None:
        return {"passed": False, "reason": "Sharpe or bootstrap unavailable"}
    bar = alpha / max(n_trials, 1)
    if boot["p_le_zero"] >= bar:
        resolvable = " (below bootstrap resolution; raise n_boot)" if bar < 1.0 / boot["n_boot"] else ""
        return {"passed": False, "reason": f"bootstrap p={boot['p_le_zero']:.4f} >= Bonferroni bar {bar:.5f} (n_trials={n_trials}){resolvable}"}
    if audit["sharpe_excess"] < min_sharpe:
        return {"passed": False, "reason": f"Sharpe {audit['sharpe_excess']:.2f} < {min_sharpe}"}
    if audit["cagr_pct"] is None or audit["cagr_pct"] < min_cagr_pct:
        return {"passed": False, "reason": f"CAGR {audit['cagr_pct']:.2f}% < {min_cagr_pct}%"}
    if audit["max_drawdown_pct"] > max_drawdown_pct:
        return {"passed": False, "reason": f"max drawdown {audit['max_drawdown_pct']:.2f}% > {max_drawdown_pct}%"}
    return {"passed": True, "reason": f"Sharpe {audit['sharpe_excess']:.2f} (p={boot['p_le_zero']:.4f}), "
                                       f"CAGR {audit['cagr_pct']:.2f}%, maxDD {audit['max_drawdown_pct']:.2f}%"}
