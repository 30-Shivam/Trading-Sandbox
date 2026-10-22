# Backlog (2026-10-22)  -  status: [x] done, [~] running/partial, [ ] open

## Decide what the system is for
- [ ] Define the objective: beat SPY net of costs? learn? low-drawdown income? Everything below depends on it. Default if undecided: index core + small satellite tested on paper.

## Evidence (does anything beat holding the index?)
- [x] Index gate built (`audit_vs_index`, `index_gate_verdict`, promote lock). ma_crossover FAIL (t=-0.09), value_rank FAIL (t=-0.57), accruals_rank PASS on curated list (+0.213%/trade, per-day t=4.95; random baseline also +0.061%).
- [~] accruals_rank rerun with 21-day-block t on default list and on `universe.json` (logs in session scratchpad `accruals_rank_blocks_*.log`).
- [ ] If accruals survives: test per cap bucket (large/mid/small) and on a point-in-time universe (`sp500_historical_components.csv` + `sp500_membership.py`); delisted names are the known blind spot.
- [ ] pairs v58 vs-index (running, old per-day clustering; rerun with blocks).
- [ ] Update `validation_trials_registry.txt` Bonferroni denominator when a strategy family is added.
- [x] SPY 10/12-month trend timing: cuts drawdown only pre-2010, lags since (not pursued).

## Execution / forward test
- [x] Alpaca paper trading (`broker.py`, `paper_trade.py`), 9 orders queued 2026-10-22, report vs SPY via `python paper_trade.py --report`.
- [ ] After ~20 trading days: compare paper vs hypothetical settlement (execution drag) and paper equity vs SPY.
- [ ] Alerts: Discord message when a paper position hits stop/target or an order is cancelled stale.
- [ ] Alpaca market-data API as a fallback when Yahoo rate-limits (check feed/volume consistency first: IEX volume is a small slice of consolidated volume).

## Cost / simplification
- [x] Soft-disabled llm_agent + best_ideas steps (`config_loader.DISABLED_RESEARCH_STEPS`): net negative live, burn LLM quota. Re-enable by editing the set.
- [x] Retired small/mid variants, momentum_rank, regime_switcher logging, sector_rs logging.
- [ ] Decide on `llm_strategy_research.py` (15 cycles in 2 months, 1 escalated, 0 promoted): throttle to weekly or drop.
- [ ] Decide on `ma_crossover` v71 as active primary (fails gate; default capital $0). Needs a replacement or an explicit "no primary".
- [ ] Remove dead code for retired strategies (regime_switcher.py, best_ideas.py, llm_agent.py, smallmid references) once the above decisions are final.

## Reliability
- [x] Batched fetch + retry (1,487 tickers in 2.2 min), COVERAGE line, DEGRADED status.
- [x] Dead-man's-switch ping (`HEALTHCHECK_URL` secret; no-op without it). TODO: create a healthchecks.io check and add the secret.
- [x] `requirements.lock` (pip freeze). CI still installs `requirements.txt`; consider pinning.
- [ ] pandas_ta emits a pandas-4 deprecation warning; replace or pin before upgrading pandas.
- [ ] Switch live scan to `universe.json` (opt-in) once accruals/value verdicts are in; rank strategies change meaning with the universe.

## Code health
- [ ] Split `optimize.py` (2.2k lines), `dip_buy_analyzer.py` (2k), `swingtrade/backtest.py` (6.8k) when touching them; no tests cover the Streamlit UI.
- [ ] Move ~25 one-off `benchmark_*`/`audit_*`/`reoptimize_*` scripts into `research/` (needs import fixes + workflow paths).
