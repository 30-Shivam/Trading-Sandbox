# Agent Directives
- Write clean, modular code and prioritize error handling.
- Ask for clarification before deleting large blocks of code.

# Project Map (read this instead of exploring; updated 2026-10-22)
Swing-trading research + signal system. Python 3.13, venv at `.venv` (`source .venv/bin/activate`), Mongo Atlas via `.env`, 742 tests (`python -m pytest -q`, ~25s).
- **Daily pipeline**: `daily_run.py` (GH Actions cron 21:35 UTC) -> `ingest.py` (scan + log signals; prints a `COVERAGE` line) -> optional `paper_trade.py` -> `settle_trades.py` -> `review_positions.py` -> `llm_strategy_research.py` -> `daily_briefing.py`.
- **Layers**: `swingtrade/` pure calc (config, levels, scoring, backtest, allocation, settlement); `market_data.py` fetch (batched yfinance + retry); `storage/` Mongo (Trade_Signals, Trade_Outcomes, System_Config, ...); `config_loader.py` which configs are live; `universe.py` rules-based S&P 500/400/600 universe (`universe.json`, opt-in; live scan still reads `watchlist.txt`).
- **Validation**: `benchmark_random_entry.py --strategy X --config-version N` prints REAL vs RANDOM *and* `VS INDEX` + `INDEX GATE: PASS/FAIL`. `promote_config.py --promote N` needs `--index-gate-passed`. Beating random entries is not enough; the strategy must beat holding SPY over the same windows (t>=1.65).
- **State**: only `value_rank` v76 / `accruals_rank` v77 (experimental) and `pairs` v58 (secondary) are positive-ish live; active primary `ma_crossover` v71 FAILS the index gate (t=-0.09). Small/mid variants, momentum_rank, regime_switcher logging were retired. Paper trading (Alpaca, paper-only) runs value_rank/accruals_rank; `python paper_trade.py --report` is the scoreboard vs SPY.
- **Gotchas**: Yahoo rate-limits bursts (don't run several full-universe fetches at once); `.env` is read-deny for Claude (use scripts that load it); never print API secrets; `improvements.txt` is 688KB of history -- `grep -n "item N"` it, never read it whole; `scratchpad/` and `research_logs/` are old run logs, ignore them; `.claude_archive/` holds unused ruflo-generated commands/skills (move back to `.claude/` to restore).
- **Open work**: see `docs/BACKLOG.md` (prioritized, with status).
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
