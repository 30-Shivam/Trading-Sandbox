# Trend-filtered dip buying: FROZEN SPEC (2026-10-27, committed BEFORE any returns computed)
Trial name in `validation_trials_registry.txt`: `trend_dip_buy` (would be the 46th trial). Reopened by explicit user decision 2026-10-27 (the 2026-10-24 research closure otherwise stands). Nothing here authorises capital: a pass only means "bring the result back to the user".

## Hypothesis
Large, uptrending stocks that fall >= 15% from their recent high mean-revert upward over the next 20 sessions by more than SPY does, after 10 bps/side costs. Motivated by the user's interest in dip-buying high-priced names (LITE, CSU.TO).

## Disclosures (prior art and known biases)
- This is a retry of a failed family. `rsi` (v-config RSI-oversold at structural support, the repo's original strategy), `rsi_mean_reversion` and `factor_reversal_1m` are already registered and failed. The rule below differs (drawdown from a 126-day high + uptrend filter, no RSI), so it is registered as a new hypothesis, but the prior for success is low.
- Share price is NOT a mechanism: a 10% move is 10% on a $9 or a $900 stock. The "expensive stock" idea is tested only as a pre-declared secondary arm (below), expected null.
- LITE and CSU.TO were named because they have done well (hindsight selection). CSU.TO is not an S&P 500 member and has no PIT data in this repo. Neither name is part of the decision test.
- Dip entries on a stock in a bull run flatter any dip rule when tested on survivors. The universe is therefore PIT membership (hanshof table `sp500_historical_components.csv`) with delisted names exiting at their last close.

## Phase 0: power screen (counts only, NO returns, spends NO trial)
Count qualifying trigger events over the PIT S&P 500 sample using the trigger below. If total events < 300, or fewer than 100 fall in 2019+, the spec is DROPPED as underpowered and is not registered. Only if the screen passes is the full test run and the trial registered.

## Universe and data
PIT S&P 500 members as of the signal date; adjusted close from the PIT price source (Tiingo/Yahoo, the same source as the PEAD trial; `prices_pit.pkl` is not in the repo and must be rebuilt, which is data work, not a return computation). Min liquidity: trailing 60-day median dollar volume >= US$20M.

## Trigger (evaluated on the close of day t, no later data)
1. close[t] <= 0.85 * max(close[t-125 .. t])  (>= 15% below the trailing 126-session high), AND
2. close[t] > SMA200[t]  AND  SMA200[t] > SMA200[t-20]  (still in a long-term uptrend), AND
3. no trigger for this ticker in the previous 20 sessions (cooldown; counted from the previous entry).
No parameter is tuned: 15%, 126, 200 and 20 are fixed here.

## Entry / exit / sizing / costs
- Entry: close of t+1 (the signal uses close[t]; entering at the same close would be circular). Sensitivity (reported, NOT decisive): close of t.
- Exit: 20 sessions after entry. No stop, no profit target (a path-dependent stop would be a tuned parameter).
- Sizing: 5% NAV per position, max 20 concurrent. If more triggers than slots on a day, take the deeper drawdown first. Remainder in SPY, SPY rebalancing free.
- Cost: 10 bps per side on the stock leg.

## Decision metrics (all required to PASS)
1. Daily portfolio excess vs SPY, 21-day block-bootstrap t via `research/etf_overlays/metrics.excess_t`: t >= 3.09 (the trial-45 bar, deliberately not relaxed for the extra trial).
2. maxDD no worse than SPY's over the same window.
3. 2019+ half keeps the same sign of excess return.
4. Beats the matched random-entry benchmark (`benchmark_random_entry.py` convention: same entry count, same holding period, same universe, same dates) at the same significance, i.e. the dip condition adds something beyond being long equities.
Whole sample from the first trigger date is the test. No period is used for tuning.

## Secondary arm (reported, NOT decisive, same rule)
High-share-price names: restrict the universe to the top quintile of UNADJUSTED close on the signal date. Purpose: test the "expensive stocks" idea directly. If unadjusted closes are unavailable in the PIT source, drop this arm and say so in the outcome.

## Descriptive look at named tickers (after the primary result only)
Per-name dip events for LITE and CSU.TO under the same trigger, shown for context. Hindsight-selected, never evidence, never a promotion input.

## Kill rules
- Phase 0 underpowered -> DROP, no trial spent.
- Any decision metric fails -> FAIL, final for this specification. Allowed afterwards: nothing (no re-parameterising 15%/126/200/20, no new filters on the same data).
- Post-hoc data fixes must be disclosed in the outcome section below.

## Outcome
### Phase 0 (run 2026-10-27, `llm_research/dip_power.py`, counts only, 0 trials spent): PASS
- Raw events (PIT member, trigger above, 20-session cooldown): **4,106 total, 2,581 in 2019+** (bars: 300 / 100). 537 distinct tickers, 201 distinct months with an event, top-3 months hold 6%, top ticker AMD 1.3%, median 1 trigger/day, max 36 (2010-05-20), only 1 day above the 20-slot cap. A 20-slot book could take 2,624 of the 4,106.
- Events per year: 2010 377, 2011 237, 2012 185, 2013 127, 2014 114, 2015 76, 2016 86, 2017 87, 2018 236, 2019 114, 2020 300, 2021 520, 2022 300, 2023 228, 2024 303, 2025 352, 2026 464.
- Counts measure sampling power only. Events inside one selloff are correlated, so the effective number of independent observations is far below 4,106.

### Post-hoc data disclosures (made before any return was computed)
1. Correction to the Universe section: `prices_pit.pkl` DOES exist (`llm_research/cache/`, gitignored, built 2026-10-23). It holds only names with 10-K filing rows, so 117 of the 752 PIT names (2010-2025) were absent. Phase 0 filled 76 of them (25 from cached Tiingo, 51 from one Yahoo batch) into `cache/prices_dip.pkl`.
2. **Survivorship bias remains, and it favors a dip-buyer.** 41 PIT names have no price data from any free source: ABC, ACS, ADS, ANTM, ATGE, BBBY, BDK, BIG, BLL, BNI, CBE, CBS, CEPH, COG, CTL, CVC, EKDKQ, ENDP, EQR, ESV, FBHS, FRC, GPS, GR, HFC, IGT, JCP, MDP, MEE, MMC, MOLX, NFX, NOVL, NYX, PKI, QLGC, RE, RX, TLAB, TMK, WLTW. About half are ticker renames (ANTM, BLL, FBHS, MMC, PKI, RE, WLTW); the rest include bankruptcies and collapses (BBBY, FRC, JCP, ENDP, BIG, CTL). Those are exactly the names a dip rule would have bought on the way down. Any PASS in the full test is therefore PROVISIONAL until they are covered or their bias is bounded; a FAIL is still a FAIL.
3. The liquidity filter (median dollar volume >= US$20M) is NOT applied: the Yahoo-sourced panel is adjusted closes only. PIT S&P 500 membership is the liquidity proxy.
4. Cooldown is measured from the previous trigger date (entry is t+1), a one-session difference from "previous entry"; events in the last 21 sessions of data are excluded for lack of a full window.

### FULL TEST (run 2026-10-27, `llm_research/dip_backtest.py`, simulator unit-tested first in `tests/test_dip_backtest.py`): **FAIL (final)**
Registered as trial #46 `trend_dip_buy` (Bonferroni alpha 0.05/46 = 0.00109). 4,069 raw events here vs 4,106 in Phase 0 because the backtest reindexes to SPY trading days. 2,587 taken, 1,482 skipped (slots full). From 2010-04-15.
- PRIMARY (entry close t+1): CAGR 16.28% vs SPY 13.80%, maxDD **-35.2% vs -33.7%**, excess **+3.26%/yr, t=1.27, p=0.090**. 2019+: +5.58%/yr, t=1.18.
- Sensitivity (entry close t): +2.95%/yr, t=1.14, maxDD -33.8% vs -33.7%.
- Matched random entries (same dates, ticker replaced by a random PIT member, 2000 draws): mean -2.04%/yr (sd 1.14); REAL +3.26%/yr, p(random >= real) = 0.0005.
- Criteria: t >= 3.09 FAIL (1.27); maxDD no worse than SPY FAIL; 2019+ same sign PASS; beats random PASS. Needs all four.
- Reading it: the dip selection beats a random stock bought on the same days by ~5.3 pp/yr, but that does not reach SPY-level significance. Costs alone are ~1.6%/yr of NAV by construction (2,587 round trips x 5% x 20 bps over 16.5 yrs), and an equal-weighted sample of S&P 500 names lagged the cap-weighted SPY (random baseline -2.04%/yr). The survivorship gap (41 missing names, above) flatters a dip-buyer, so the true result is no better than this. The drawdown is slightly WORSE than SPY because the book is up to 100% equities.
- Secondary 'expensive stocks' arm: DROPPED per spec (the price panel has no unadjusted closes).
- Descriptive look (hindsight-selected, not evidence): LITE 43 dip events, mean fwd-20d +5.24% (median +2.25%, 58% positive) vs +3.92% across ALL days; CSU.TO only 4 events (it rarely dips 15% while above a rising 200-day), mean +0.47% vs +2.45% all days (CAD price). LITE simply rose on average, and the dip added little to that.
- Per the spec, nothing may be re-parameterised or re-filtered on this data.
