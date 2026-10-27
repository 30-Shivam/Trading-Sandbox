# Agent Directives
- Write clean, modular code and prioritize error handling.
- Ask for clarification before deleting large blocks of code.

# Objective
BEAT SPY (total return, net of costs, with drawdown no worse than SPY's). Every strategy and every change is judged against that. Declared by the user 2026-10-22.

# Project Map (read this instead of exploring; updated 2026-10-22)
Swing-trading research + signal system. Python 3.13, venv at `.venv` (`source .venv/bin/activate`), Mongo Atlas via `.env`, 630 tests (`python -m pytest -q`, ~25s).
- **Daily pipeline**: `daily_run.py` (GH Actions cron 21:35 UTC) -> `ingest.py` (scan + log signals; prints a `COVERAGE` line) -> optional `paper_trade.py` -> `settle_trades.py` -> `review_positions.py` -> `daily_briefing.py`.
- **Layers**: `swingtrade/` pure calc (config, `indicators.py` RSI/ATR/ADX [no pandas_ta], `levels/`, `scoring`, `backtest/`, allocation, settlement). `levels/` and `backtest/` are packages that re-export every old name, so `swingtrade.levels.X` / `swingtrade.backtest.X` imports still work; to edit code, open the matching submodule (`backtest/`: core, signals_{technical,cross_sectional,fundamental,other}, engine, stats, portfolio_audits, walkforward, index_gate; `levels/`: base, technical, cross_sectional, fundamental, llm_rules, trend, audits). Patch names in the submodule that USES them (e.g. `swingtrade.backtest.signals_other._settle`), not the package; `market_data.py` fetch (batched yfinance + retry); `storage/` Mongo (Trade_Signals, Trade_Outcomes, System_Config, ...); `config_loader.py` which configs are live; `universe.py` rules-based S&P 500/400/600 universe (`universe.json`, opt-in; live scan still reads `watchlist.txt`).
- **Validation**: `benchmark_random_entry.py --strategy X --config-version N` prints REAL vs RANDOM *and* `VS INDEX` + `INDEX GATE: PASS/FAIL`. `promote_config.py --promote N` needs `--index-gate-passed`. Beating random entries is not enough; the strategy must beat holding SPY over the same windows (t>=1.65).
- **Cash gate (absolute-return sleeves, added 2026-10-24; the SPY objective above is unchanged)**: for carry/overlay/market-neutral sleeves that are not meant to track equities, `python audit_vs_cash.py --returns-csv|--nav-csv ...` (`swingtrade/backtest/cash_gate.py`, exported as `swingtrade.audit_vs_cash` / `cash_gate_verdict`) benchmarks DAILY returns against cash (^IRX default, or BIL): CAGR, max drawdown, Calmar, tail metrics, excess Sharpe with a stationary-block-bootstrap CI. Gate (the user's sleeve targets): >=3 yrs of data AND bootstrap p(Sharpe<=0) < 0.05/N AND Sharpe>=1.5 AND CAGR>=8% AND maxDD<=5%. It is the counterpart of the SPY gate, not a replacement: a sleeve needs this gate, the core still lives or dies by the SPY gate.
- **Registry rules**: `validation_trials_registry.txt` is the single Bonferroni denominator for BOTH gates (N = entries + the strategy under test). A hypothesis test of a strategy registers (`benchmark_random_entry.py` or `audit_vs_cash.py --register`). Descriptive feasibility screens do NOT register: they get a frozen spec in `docs/*_SPEC.md` (kill rules written BEFORE data, post-hoc data fixes disclosed) and an outcome appended to it. Done so far (all 0 trials): odd-lot tender screen (FAIL, `docs/ODDLOT_SPEC.md`); crypto basis funding screen (DROP, avg net spread over cash 1.96 pp since 2022 vs 4.0 bar, `docs/CRYPTO_BASIS_SPEC.md`).
- **State**: NO strategy passes the index gate (46 registered trials, 0 passed; #46 trend_dip_buy FAILED 2026-10-27, docs/DIP_SPEC.md. ACTIVE ALPHA RESEARCH CONCLUDED 2026-10-23 by user decision. Final trial #45 pead_8k_sue_react failed: excess +0.70%/yr t=0.89, 2019+ -0.61%/yr, maxDD -36.1% vs SPY -33.7%. Small-cap expansion declined: small-cap beta headwind vs the SPY benchmark + higher microstructure costs. Outcomes: docs/TRIAL_OUTCOMES.md; the registry file is names-only because code rewrites it). ma_crossover v71 was retired as primary 2026-10-22 (no active System_Config; ingest scans on the rsi fallback but logs nothing for it). accruals_rank passed only on the curated 407 list (t=1.84) and FAILED on the 1,487 universe (excess +0.001%, t=0.52) = curation bias. value_rank v76 / accruals_rank v77 (experimental) and pairs v58 (secondary) still run as unfunded trackers; paper satellite is OFF (both failed the gate). `llm_agent`, `best_ideas`, `llm_strategy_research`, `regime_switcher`, `reoptimize_best_ideas/_sector_rs` and the monthly workflow were DELETED 2026-10-22 (net negative live; recover from git history or `.claude_archive/retired_modules/`). The dashboard now has only Daily Briefing, Mechanical Strategies and Daily Signals tabs. `python paper_trade.py --report` is the scoreboard vs SPY.
- **RESEARCH PROGRAM CLOSED (user decision 2026-10-24)**: ALL active alpha research is formally closed: S&P 500 equity signals (45 trials, 0 passed), odd-lot tender offers (screen FAIL, `docs/ODDLOT_SPEC.md`), crypto basis/funding carry (screen DROP, `docs/CRYPTO_BASIS_SPEC.md`) and options selling (declined WITHOUT a screen: its tail correlation with SPY and negative skew conflict with the 5% max-drawdown cash gate, and the stack has no clean historical options surface). Do not reopen any of these, or start new alpha research, without an explicit user decision. (2026-10-27: the user explicitly approved ONE pre-registered test, trend-filtered dip buying, frozen spec `docs/DIP_SPEC.md`: it was run and FAILED (trial #46, excess +3.26%/yr t=1.27, maxDD worse than SPY); the closure stands again.) The two validation gates (Index Gate vs SPY, Cash Gate vs ^IRX/BIL) stay as standard repo assets for any future idea.
- **Gotchas**: Yahoo rate-limits bursts (don't run several full-universe fetches at once); `.env` is read-deny for Claude (use scripts that load it); never print API secrets; `improvements.txt` is 688KB of history -- `grep -n "item N"` it, never read it whole; `scratchpad/` and `research_logs/` are old run logs, ignore them; `.claude_archive/` holds unused ruflo-generated commands/skills (move back to `.claude/` to restore).
- **Open work**: see `docs/BACKLOG.md` (prioritized, with status).
- **Portfolio policy (paper, policy "F" = 60% QQQ / 20% SPY / 10% IBIT (spot-BTC ETF) / 10% GLD, user decision 2026-10-27, replacing the 85/15 freeze of 2026-10-24; chosen for a -35% max-DD tolerance, NOT validated alpha; changing it needs an explicit user decision. The REAL Scotia TFSA tools (`advisor.py`, `portfolio_view.py`, `contribution_plan.py`) still default to 85/15 CAD funds until the user picks CAD gold/BTC funds)**: the daily pipeline runs in MONITORING MODE: `daily_run.py` tracks the 85/15 baseline plus unfunded forward trackers (value_rank v76, accruals_rank v77, pairs v58, `ratio_tracker.py`) and places NO strategy/alpha trades; the only orders ever sent are `rebalance.py`'s 3%-drift rebalances back to the policy weights on the Alpaca PAPER account (when `PAPER_TRADING_ENABLED=1`). Details: `rebalance.py` `DEFAULT_TARGETS` holds the policy-F weights (a beta/risk bet, not alpha; mixes compared in `research/allocation_mixes/compare.py`, 2008-style crash would still cost ~50%), satellite reserve 0% (`paper_trade.py` runs no strategies until one passes the index gate); rebalances on 3% drift. `factor_portfolios.py` showed no factor beats SPY point-in-time (10y).
- **Retire/extend rule**: new strategy ideas are budgeted (a few per quarter); anything new must pass the applicable gate (Index Gate vs SPY for the core, Cash Gate for an absolute-return sleeve) before capital or promotion, and per the closure above needs an explicit user decision to start.

# Repo Hygiene
- Never commit run output: `*.log`, `*.pkl`, caches, `research_logs/`, and non-script files in `scratchpad/` are gitignored. Only reusable scripts in `scratchpad/` stay tracked.
- Redirect experiment output to `scratchpad/` (e.g. `python x.py > scratchpad/x.log`), never the repo root.
- Before committing, run `git status --short` and `git ls-files -ci --exclude-standard`; the second must print nothing. If it does, `git rm --cached <path>` (keeps the local file).
- Data that tracking depends on (`validation_trials_registry.txt`, `universe.json`, `sp500_historical_components.csv`, watchlists) stays tracked; don't add ignore rules for it.
- New generated artifact type? Add its ignore rule to `.gitignore` in the same change.

# Memory Storage Protocol
Ruflo is the only project memory (Obsidian was dropped 2026-10-22).

- **Before starting a task**: `mcp__ruflo__memory_search` (namespace `trading-sandbox`) for past context.
- **During/after a session**: `mcp__ruflo__memory_store` (namespace `trading-sandbox`) for decisions, findings and
  state changes. Update an existing key in place (same key) when a topic's current state changes, instead of only
  adding new dated keys, so the key stays the single source of truth for that topic.
- Durable detail that belongs with the code goes in `improvements.txt` / `TUTORIAL.md`, not in memory.
- Claude Code's native memory (`~/.claude/projects/<project>/memory/`) is governed by Claude's own instructions;
  keep `MEMORY.md` entries to one-line pointers (~150 chars) since it loads every session.

## Autonomous Skill Creation & Evolution
- After completing any multi-step task, non-trivial debugging session, or repeated workflow (e.g., running specific test suites, executing backtest validations, checking MongoDB states, or config promotions), evaluate whether this represents a reusable repo pattern.
- If it does, automatically create or update a dedicated skill in `.claude/skills/<skill-name>/SKILL.md`.
- Every skill must follow this standard structure:
  ```markdown
  ---
  name: <kebab-case-name>
  description: "<Concise Claude auto-activate description of should skill this trigger when>"
  ---
  # <Skill Name>

  ## When to Use
  - Triggers and conditions.

  ## Operational Workflow
  1. Exact commands, scripts, or Python entry points to execute.
  2. Expected inputs, arguments, and required flags.

  ## Failure Recovery & Edge Cases
  - Common errors observed in this repo and how to bypass them.
