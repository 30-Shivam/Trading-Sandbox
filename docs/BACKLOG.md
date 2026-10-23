# Backlog (2026-10-22)  -  status: [x] done, [~] running/partial, [ ] open

## Decide what the system is for
- [x] Objective set by user 2026-10-22: BEAT SPY (net of costs). Judge everything against holding SPY.

## Evidence (does anything beat holding the index?)
- [x] Index gate built (`audit_vs_index`, `index_gate_verdict`, promote lock). ma_crossover FAIL (t=-0.09), value_rank FAIL (t=-0.57), accruals_rank PASS on curated list (+0.213%/trade, per-day t=4.95; random baseline also +0.061%).
- [~] accruals_rank rerun with 21-day-block t on default list and on `universe.json` (logs in session scratchpad `accruals_rank_blocks_*.log`).
- [ ] If accruals survives: test per cap bucket (large/mid/small) and on a point-in-time universe (`sp500_historical_components.csv` + `sp500_membership.py`); delisted names are the known blind spot.
- [ ] pairs v58 vs-index (running, old per-day clustering; rerun with blocks).
- [ ] Update `validation_trials_registry.txt` Bonferroni denominator when a strategy family is added.
- [x] SPY 10/12-month trend timing: cuts drawdown only pre-2010, lags since (not pursued).

## Beat-SPY findings (2026-10-22, `factor_portfolios.py`)
- [x] Monthly long-only top-quintile factor portfolios vs SPY. Using TODAY's constituents (survivorship-biased): large-cap momentum_12_1 +6.0%/yr (t 1.76), trend_above_200d +6.5%/yr (t 1.80) over 7y. Point-in-time S&P 500 (10y, 88% price coverage): NOTHING beats SPY (momentum_12_1 -1.5%/yr, trend -0.9%/yr, EW S&P -1.9%/yr). The earlier gain was survivorship/lookahead. Small caps: nothing.
- [ ] Remaining routes to beat SPY: (a) accruals_rank if it survives the block-clustered, universe-wide rerun; (b) deliberate beta/tilt (e.g. QQQ/mega-cap growth, leverage) - a risk bet, not alpha; (c) fundamentals factors (quality/value/accruals) on a point-in-time universe, long history (needs delisted data, e.g. a paid dataset); (d) accept SPY as the core and only run satellites that pass the gate.

## Core + tilt + satellite (built 2026-10-22)
- [x] `rebalance.py`: SPY 70% core, QQQ 15% tilt, 15% satellite reserve; drift band 3%; sells before buys; idempotent daily order IDs. Wired into daily_run behind PAPER_TRADING_ENABLED. Historical: QQQ beat SPY ~5%/yr 2016-26 with -33% vs -24% maxDD (hindsight regime bet); 80/20 blend +1.0%/yr excess (t 1.9).
- [ ] Judge after 3+ months: account vs SPY via `python paper_trade.py --report`; tilt should be cut if QQQ underperforms SPY for 12 months or drawdown exceeds SPY's by >10pp.
- [ ] Satellite sleeve only for strategies that pass the index gate; currently value_rank (fails) + accruals_rank (pending).

## Execution / forward test
- [x] Alpaca paper trading (`broker.py`, `paper_trade.py`), 9 orders queued 2026-10-22, report vs SPY via `python paper_trade.py --report`.
- [ ] After ~20 trading days: compare paper vs hypothetical settlement (execution drag) and paper equity vs SPY.
- [ ] Alerts: Discord message when a paper position hits stop/target or an order is cancelled stale.
- [ ] Alpaca market-data API as a fallback when Yahoo rate-limits (check feed/volume consistency first: IEX volume is a small slice of consolidated volume).

## Cost / simplification
- [x] llm_agent + best_ideas + llm_strategy_research removed 2026-10-22 (net negative live). Restore from git history if ever needed.
- [x] Retired small/mid variants, momentum_rank, regime_switcher logging, sector_rs logging.
- [x] `llm_strategy_research.py` dropped (0 promoted). `simulate_llm_strategy_*` / `levels/llm_rules.py` / the `llm_strategy` config path are now orphaned but still dispatch-tested.
- [x] ma_crossover v71 retired as primary 2026-10-22 (explicit "no primary"; ingest skips logging the rsi fallback). accruals_rank NOT promoted: universe run excess +0.001%, t=0.52 (FAIL); curated-list pass was bias.
- [x] Dead code removed: llm_agent, best_ideas, llm_strategy_research, regime_switcher, reoptimize_*, benchmark_consensus_signal.
- [ ] Still in place: core ma_crossover code (config fields, dispatch, ~20 tests), `llm_strategy` plumbing, market_data macro/qualitative snapshot helpers (now unused), smallmid references.

## Reliability
- [x] Batched fetch + retry (1,487 tickers in 2.2 min), COVERAGE line, DEGRADED status.
- [x] Dead-man's-switch ping (`HEALTHCHECK_URL` secret; no-op without it). TODO: create a healthchecks.io check and add the secret.
- [x] `requirements.lock` (pip freeze). CI still installs `requirements.txt`; consider pinning.
- [x] pandas_ta replaced by `swingtrade/indicators.py` (exact-match tests vs pandas_ta; dev-only oracle, skipped if absent).
- [x] Split swingtrade/backtest.py (6.8k lines) and levels.py (4.5k) into packages. Not split: optimize.py, benchmark_random_entry.py, dip_buy_analyzer.py, config.py, scoring.py.
- [ ] Switch live scan to `universe.json` (opt-in) once accruals/value verdicts are in; rank strategies change meaning with the universe.

## Code health
- [ ] Split `optimize.py` (2.2k lines), `dip_buy_analyzer.py` (2k), `swingtrade/backtest.py` (6.8k) when touching them; no tests cover the Streamlit UI.
- [ ] Move ~25 one-off `benchmark_*`/`audit_*`/`reoptimize_*` scripts into `research/` (needs import fixes + workflow paths).

- [x] 2026-10-22: llm_strategy_research soft-disabled; Bonferroni registry now 28 trials (factor portfolios, SPY timing, QQQ tilt added).
- [ ] Consider dropping value_rank/accruals_rank paper satellite too if they stay at excess~0 (they fail the gate); decide after ~20 trading days of paper data.

## Strategy search loop (2026-10-22, protocol: one fixed spec per idea, design <2013 / holdout 2013+, block-bootstrap t, every try registered; budget 5, 4 used)
Scripts in `research/etf_overlays/` (cache dir is regenerated). Bonferroni bar is now 0.05/32 = 0.0016. ALL FOUR FAILED:
- vol_managed_spy (target = SPY design vol, cap 1.5x): design excess -0.1%/yr; holdout +2.4%/yr p=0.065. Reject (design fail).
- lev2x_trend200 (2x SPY above 200d SMA else T-bills): design +2.3%/yr t=0.5; holdout +6.5%/yr p=0.04 BUT maxDD -36.3% vs SPY -33.7% (fails the drawdown gate) and not Bonferroni-significant. Reject.
- sector_etf_mom_top3: design +3.7%/yr p=0.07; holdout -0.1%/yr. Reject.
- gem_dual_momentum: design +4.4%/yr (DD -20% vs -55%); holdout -6.3%/yr t=-3.0. Reject.
- Not testable: LLM/news sentiment (no point-in-time news history; live LLM variants lost money).
- [ ] 1 slot left in this session's budget; remaining paths are beta bets (2x trend is the closest) or paid point-in-time fundamentals data.

## Strategy search, round 2 (2026-10-23) - option (c) adopted: SPY core, satellite only for gate-passers
- [x] `paper_trade.py` default PAPER_STRATEGIES is now empty; `rebalance.py` default is SPY 85% + QQQ 15%, satellite reserve 0%. Existing satellite positions (7) are not force-sold; they age out via max-hold.
- Free survivorship-free data: no free survivorship-free STOCK price set exists (searched; Quandl WIKI has no public mirror, others paid). Workaround: Ken French library decile portfolios (CRSP incl. delisted, 1926+), files in `research/french_deciles/README.md`. EDGAR Financial Statement Data Sets (free, as-filed, 2009+) are the point-in-time fundamentals source.
- French long-only value-weighted deciles vs market, design 1963-2002 / holdout 2003-2026-08 (gross, no costs): mom +8.9% -> +3.1%/yr (holdout t=1.3); B/M value +5.8% -> +2.1% (t=0.6); E/P +4.5% -> +1.7%; low-accruals +1.8% -> +2.1% (t=1.2); profitability +0.8% -> +1.7%; low-investment +3.4% -> +0.5%. Equal-weight blend (mom+B/M+low-AC+profitability): +4.3% -> +2.3%/yr (design t=5.1, holdout t=2.3) but maxDD -54.7% vs market -50.3% (fails the DD gate). Edge visibly decays after publication.
- Tradable check: real factor ETFs since 2013-07 (MTUM+VLUE+QUAL equal weight, net of fees): CAGR 13.91% vs SPY 13.98%, excess +0.05%/yr t=0.05, maxDD -35.6% vs -33.7%. FAIL. The paper edge is not capturable with investable ETFs.
- Registry: 8 more trials, now 40 (Bonferroni 0.05/40 = 0.00125). Nothing passes.
- [ ] Open: stock-level versions on a point-in-time universe still need delisted prices (paid) - the French test says the factor premia exist historically but shrink to ~0 after costs/crowding.

## LLM / free-data fundamentals loop (2026-10-23, `llm_research/`, registry now 42 trials, Bonferroni 0.05/42 = 0.0012)
- Dataset: 4,781 as-filed 10-K ratio rows (2010-2025) from EDGAR + Yahoo; S&P point-in-time members only; survivor-tilted (394 of 568 resolvable names).
- sec_ratio_ridge (NON-LLM; 12 as-filed ratios, expanding-window ridge, top quintile, EW, hold 126d, 10bps/side, SPY when idle): 2014-2025 CAGR 24.2% vs SPY 14.2%, alpha +7.8%/yr (beta 1.14), excess +9.9%/yr t=2.7 p=0.0018; holdout 2019+ +8.8%/yr t=1.7 p=0.038. maxDD -40.1% vs -33.7% (50/50 with SPY: -36.9%) -> FAILS drawdown gate; p NOT below Bonferroni bar. Top features: operating cash flow/assets (IC +0.10, t=2.8), low accruals (IC -0.056 on accruals, t=-2.1). Edge weakening 2023-25. Status: promising but unproven. Best next step is a forward paper tracker (no capital), not tuning.
- llm_masked_fundamentals (26 blind Sonnet scorers, 1,548 anonymized rows: 748 ridge picks + 800 random): FAIL. Within picks LLM IC -0.050; LLM-vetoed picks returned +8.0% vs LLM-kept -0.2% (inverted). On random rows LLM IC +0.049, permutation p=0.083. LLM adds no value over ratios; live LLM variants also lost money earlier.
- Leakage note: LLMs know 2010-2025 outcomes, so even blind historical LLM tests are suspect; a clean LLM test needs forward-only data.

## Forward tracker for sec_ratio_ridge (built 2026-10-23)
- `llm_research/freeze_model.py` -> `llm_research/frozen_model.json` (ridge trained ONLY on filings whose label ended before 2026-01-01; do not re-freeze without noting it here, it resets the test). `ratio_tracker.py` (daily_run step, full EDGAR scan Mondays only, report other days, NOT part of all_ok) scores new S&P 500 10-Ks into Mongo `Forward_Ratio_Scores` and settles 126d excess vs SPY. `python ratio_tracker.py --report`.
- First out-of-sample read (dry run 2026-10-23, filings 2026-01-01+, 298 settled, one market window so heavily correlated): forward rank-IC +0.018 (perm p=0.39); picks +1.7% vs rest -4.4% mean 6m excess (n=49 vs 249). Neither confirms nor refutes. Verdict needs >= 150 independent-ish settled filings across more than one market window, IC p < 0.0012, and a daily-portfolio drawdown check.
- Stale input: universe = S&P 500 membership snapshot of 2025-08-23 (sp500_membership.py), refresh the CSV from hanshof/sp500_constituents when convenient.

## Robustness of sec_ratio_ridge + out-of-project probes (2026-10-23, registry now 44 trials, Bonferroni 0.05/44 = 0.0011)
- `llm_research/robustness.py` (diagnostics only, nothing re-selected). The "promising" ridge is FRAGILE: IC only at 126d (+0.046, t=1.7; 21d -0.001, 63d +0.002, 252d +0.026); picks' median excess -0.8%, hit-rate 48.8%, mean excluding best 5% = -2.35% (headline driven by a few big growth/software winners: NOW, WDAY, FTNT, DXCM); top-quintile mean excess by regime +5.4% (2014-18), +4.0% (2019-21), +0.2% (2022-25). Single feature operating-cash-flow/assets ranks better than the ridge (IC +0.10, t=2.6) but with a lower top-quintile mean. Verdict: weak signs, probably growth/quality-tech beta plus lottery-style tails. Forward tracker stays on as the arbiter.
- Congressional trades: no usable free bulk source (houses' PDFs; HouseStockWatcher S3 dead, other feeds paid/partial). Not tested.
- btc_trend200 (BTC > 200d SMA else T-bills): 2015-20 +76%/yr excess (bubble years) but 2021+ +8%/yr t=0.42, maxDD -63% vs SPY -24%. spy_btc_trend_blend (80% SPY + 20% sleeve): holdout +1.6%/yr p=0.39, maxDD -25.2% vs -24.5%. Both FAIL.
