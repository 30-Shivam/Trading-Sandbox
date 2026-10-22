# Agent Directives
- Write clean, modular code and prioritize error handling.
- Ask for clarification before deleting large blocks of code.

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
