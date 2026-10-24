# Trial outcomes (companion to validation_trials_registry.txt)
The registry is a plain one-name-per-line file that `benchmark_random_entry.bonferroni_check` rewrites, so it cannot hold notes. Outcomes live here. 45 trials registered; ZERO passed. Active alpha research CONCLUDED 2026-10-23 (user decision).
Bonferroni bar at 45 trials: alpha = 0.05/45 = 0.0011 (t ~ 3.09).

| # | Trial | Outcome |
|---|---|---|
| 42 | sec_ratio_ridge | Original +9.9%/yr (t=2.7) was index-inclusion look-ahead. Clean PIT data (Tiingo delisted prices): +2.64%/yr t=1.23 p=0.097, 2019+ +3.34%/yr t=1.07, maxDD -36.8% vs SPY -33.7%. FAILED |
| 43 | llm_masked_fundamentals | Blind LLM on anonymised ratios: random-row IC +0.049 p=0.083; vetoed ridge picks beat kept ones (+8.0% vs -0.2%). FAILED |
| 44 | btc_trend200 / spy_btc_trend_blend | Holdout +8%/yr t=0.42, maxDD -63%; 80/20 blend +1.6%/yr p=0.39. FAILED |
| 45 | pead_8k_sue_react | 8-K Item 2.02 + net-income SUE + reaction-day return, spec frozen before returns (docs/PEAD_SPEC.md). Primary: excess +0.70%/yr t=0.89 p=0.23; 2019+ -0.61%/yr; maxDD -36.1% vs SPY -33.7%. FAILED (final) |

Screens dropped WITHOUT spending a trial: S&P 500 deletion rebound (134 events, needed >2.5% 20d drift), HY-OAS regime gate (FRED truncated the series to 3 years in Apr 2026), GP/A x low-beta, filing-lag.

## Closing decision (2026-10-23)
- Portfolio policy FROZEN: CORE SPY 85% + TILT QQQ 15% (3% drift rebalance), satellite reserve 0%, no active alpha trades. 85/15 is a beta bet, not alpha: 2013+ CAGR 15.63% vs SPY 14.85%, maxDD -32.8% vs -33.7% (passes narrowly thanks to 2020; 2022 was 1.5 pts worse).
- Small-cap expansion (MarketParquet, $79) DECLINED: small caps carry a beta/size headwind against a cap-weighted SPY benchmark (2013-2026 punished size), and higher microstructure costs (spreads, impact, borrow) erode the thin edges that survive; survivorship bias is also worst there. Estimated chance of a true pass <=10%.
- Still running as evidence only (no capital): ratio_tracker.py (frozen sec_ratio_ridge, first real scan Monday 2026-10-26), pairs/value_rank/accruals_rank trackers.
- Reopen only with a genuinely new data source or mechanism, a pre-registered spec, and a power screen first (see llm_research/deletion_events.py, pead_power.py for the pattern).
