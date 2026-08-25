# Agent Directives
- Write clean, modular code and prioritize error handling.
- Ask for clarification before deleting large blocks of code.

# Memory Storage Protocol
Always use BOTH of these — every session, not just "complex" or "major" ones:

- **Obsidian vault** (`mcp__obsidian__*` tools): the durable project record — architecture, daily/session logs, decisions.
  - All architectural notes, daily logs, and context for this project must be saved in the vault under `/Trading-Sandbox/` (already exists — do not recreate it).
  - Before starting ANY task, use `mcp__obsidian__search_notes` (not `query_wiki` — that tool name doesn't exist in this MCP server) to check `/Trading-Sandbox/` for past context.
  - At the end of the session, write a `session-summary-YYYY-MM-DD.md` note (see existing ones in the folder for format) summarizing what changed and why.
- **ruflo memory** (`mcp__ruflo__memory_store` / `memory_search`): fast semantic key-value recall across sessions, for smaller facts/decisions that don't warrant a full vault note.
  - Requires `CLAUDE_FLOW_ENABLE_NATIVE_BRIDGE_ON_WINDOWS=1` in `~/.claude/settings.json` (set 2026-08-25) — without it, writes silently fail on Windows.

Do not skip either one because a task "seems small" — check both at the start, write to both (vault note for anything architecturally relevant, ruflo for quick facts) before ending the session.