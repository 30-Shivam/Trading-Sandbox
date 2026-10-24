"""Train the ratio ridge ONCE on every filing whose 126d label ended before 2026-01-01 and freeze it to frozen_model.json.
Forward filings (filed >= 2026-01-01, entered and exited after the cutoff) are therefore genuinely out-of-sample. Do not re-run casually: re-freezing
resets the forward test (record the change in docs/BACKLOG.md)."""
import json, sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from edgar_features import FEATURES
LAM, Q = 50.0, 0.8
CUTOFF = pd.Timestamp("2026-01-01")   # model sees only labels that ENDED before this date
D = pd.read_pickle(ROOT / "cache" / "dataset.pkl"); D = D[D.exit < CUTOFF]
mu, sd = D[FEATURES].mean(), D[FEATURES].std().replace(0, 1)
z = lambda d: ((d[FEATURES] - mu) / sd).clip(-3, 3).fillna(0).values
y = (D.excess - D.excess.mean()).values; X = z(D)
w = np.linalg.solve(X.T @ X + LAM * np.eye(len(FEATURES)), X.T @ y)
thr = float(np.quantile(X @ w, Q))
out = dict(trained_on=len(D), filed_max=str(D.filed.max().date()), exit_max=str(D.exit.max().date()), lam=LAM, quantile=Q,
           features=FEATURES, mu=mu.tolist(), sd=sd.tolist(), weights=w.tolist(), threshold=thr, hold_days=126)
(ROOT / "frozen_model.json").write_text(json.dumps(out, indent=1)); print({k: out[k] for k in ("trained_on", "filed_max", "exit_max", "threshold")})
print(dict(zip(FEATURES, np.round(w, 4))))
