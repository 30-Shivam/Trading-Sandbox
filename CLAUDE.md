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
- **State**: NO strategy passes the index gate. ma_crossover v71 was retired as primary 2026-10-22 (no active System_Config; ingest scans on the rsi fallback but logs nothing for it). accruals_rank passed only on the curated 407 list (t=1.84) and FAILED on the 1,487 universe (excess +0.001%, t=0.52) = curation bias. value_rank v76 / accruals_rank v77 (experimental) and pairs v58 (secondary) still run as unfunded trackers; paper trading runs value_rank/accruals_rank on a 15% satellite. `llm_agent`, `best_ideas`, `llm_strategy_research`, `regime_switcher`, `reoptimize_best_ideas/_sector_rs` and the monthly workflow were DELETED 2026-10-22 (net negative live; recover from git history or `.claude_archive/retired_modules/`). The dashboard now has only Daily Briefing, Mechanical Strategies and Daily Signals tabs. `python paper_trade.py --report` is the scoreboard vs SPY.
- **Gotchas**: Yahoo rate-limits bursts (don't run several full-universe fetches at once); `.env` is read-deny for Claude (use scripts that load it); never print API secrets; `improvements.txt` is 688KB of history -- `grep -n "item N"` it, never read it whole; `scratchpad/` and `research_logs/` are old run logs, ignore them; `.claude_archive/` holds unused ruflo-generated commands/skills (move back to `.claude/` to restore).
- **Open work**: see `docs/BACKLOG.md` (prioritized, with status).
- **Portfolio policy (paper)**: `rebalance.py` holds CORE SPY 70% + TILT QQQ 15% (a beta bet, not alpha) with a 15% satellite reserve for `paper_trade.py` stock signals; rebalances on 3% drift. `factor_portfolios.py` showed no factor beats SPY point-in-time (10y).
- **Retire/extend rule**: new strategy ideas are budgeted (a few per quarter); anything new must pass the index gate before capital or promotion.

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
