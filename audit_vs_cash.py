"""Absolute-return audit: does a DAILY return / NAV series earn a real, smooth return OVER CASH?

The counterpart of the SPY index gate (benchmark_random_entry.py's `INDEX GATE`) for sleeves that are not supposed to track
equities (carry, overlays, market-neutral). Cash = 13-week T-bill yield (^IRX, default, long history) or BIL total return.
Reports CAGR, max drawdown, Calmar, excess Sharpe with a stationary-block-bootstrap CI, tail metrics, and a gate verdict whose
significance bar is Bonferroni-corrected by validation_trials_registry.txt (read-only unless --register is given).

Usage:
    python audit_vs_cash.py --returns-csv sleeve.csv                  # columns: date,return  (daily simple returns)
    python audit_vs_cash.py --nav-csv sleeve_nav.csv --cash bil       # columns: date,nav
    python audit_vs_cash.py --returns-csv sleeve.csv --strategy my_sleeve --register   # counts as a registered trial

Descriptive screens (e.g. docs/CRYPTO_BASIS_SPEC.md) are NOT trials: do not pass --register for them.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import swingtrade
from swingtrade.backtest import cash_gate

REGISTRY_PATH = Path(__file__).parent / "validation_trials_registry.txt"


def registry_count(path: Path = REGISTRY_PATH) -> int:
    return len({l.strip() for l in path.read_text().splitlines() if l.strip()}) if path.exists() else 0


def load_series(csv_path: Path, column: str) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    col = column if column in df.columns else df.columns[0]
    return df[col].astype(float).sort_index()


def fetch_cash_returns(kind: str, start: str, end: str) -> pd.Series:
    """Daily cash returns. ^IRX -> yield/252; BIL -> adjusted-close pct change. Network (yfinance)."""
    import yfinance as yf
    if kind == "bil":
        px = yf.download("BIL", start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze()
        return cash_gate.nav_to_returns(px)
    irx = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)["Close"].squeeze()
    return cash_gate.irx_to_daily_returns(irx)


def format_report(label: str, audit: dict | None, verdict: dict, n_trials: int) -> str:
    if not audit:
        return f"{label}: insufficient overlapping data.\nCASH GATE: FAIL ({verdict['reason']})"
    b, t = audit["bootstrap"], audit["tail"]
    lines = [
        f"{label}: {audit['start']} -> {audit['end']} ({audit['years']:.1f} yrs, {audit['n_days']} days)",
        f"  CAGR {audit['cagr_pct']:.2f}% | max drawdown {audit['max_drawdown_pct']:.2f}% | Calmar "
        f"{'n/a' if audit['calmar'] is None else format(audit['calmar'], '.2f')}",
        f"  Excess over cash: {audit['annualized_excess_pct']:.2f}%/yr | Sharpe {audit['sharpe_excess']:.2f} "
        f"(iid SE {audit['sharpe_se_iid']:.2f}; bootstrap {int(b['ci_level'] * 100)}% CI [{b['ci_low']:.2f}, {b['ci_high']:.2f}], "
        f"p(Sharpe<=0)={b['p_le_zero']:.4f}, mean block {b['mean_block']:.1f}d)",
        f"  Tail: worst day {t['worst_day_pct']:.2f}% | CVaR95 daily {t['cvar_95_daily_pct']:.2f}% | skew {t['skew']:.2f}",
    ]
    if "years_needed_z1_65" in audit:
        lines.append(f"  History needed for t=1.65 at this Sharpe (iid): {audit['years_needed_z1_65']:.1f} yrs")
    lines.append(f"  Bonferroni bar: alpha 0.05 / {n_trials} trials = {0.05 / n_trials:.5f}")
    lines.append(f"CASH GATE: {'PASS' if verdict['passed'] else 'FAIL'} -- {verdict['reason']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--returns-csv", type=Path); src.add_argument("--nav-csv", type=Path)
    ap.add_argument("--column", default="return", help="column name in the CSV (default: first column)")
    ap.add_argument("--cash", choices=["irx", "bil"], default="irx")
    ap.add_argument("--strategy", default="(unnamed)"); ap.add_argument("--register", action="store_true",
                    help="record --strategy in the trials registry (counts toward the Bonferroni denominator)")
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    series = load_series(args.returns_csv or args.nav_csv, args.column)
    strat = series if args.returns_csv else cash_gate.nav_to_returns(series)
    cash = fetch_cash_returns(args.cash, str(strat.index[0].date()), str((strat.index[-1] + pd.Timedelta(days=1)).date()))

    if args.register:
        from benchmark_random_entry import bonferroni_check   # lazy: heavy import, only when writing the registry
        n_trials = bonferroni_check(args.strategy, None)["n_trials_ever_tested"]
    else:
        n_trials = registry_count() + 1   # the strategy under test counts toward its own denominator
    audit = swingtrade.audit_vs_cash(strat, cash, n_boot=args.n_boot, seed=args.seed)
    verdict = swingtrade.cash_gate_verdict(audit, n_trials=n_trials)
    print(format_report(args.strategy, audit, verdict, n_trials))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
