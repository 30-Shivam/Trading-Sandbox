# Agent Directives
- Write clean, modular code and prioritize error handling.
- Ask for clarification before deleting large blocks of code.

# Memory Storage Protocol
Always use BOTH of these — every session, not just "complex" or "major" ones:

- **Obsidian vault** (`mcp__obsidian__*` tools): the durable project record — architecture, daily/session logs, decisions.
  - All architectural notes, daily logs, and context for this project must be saved in the vault under `/Trading-Sandbox/` (already exists — do not recreate it).
  - Before starting ANY task, use `mcp__obsidian__search_notes` (not `query_wiki` — that tool name doesn't exist in this MCP server) to check `/Trading-Sandbox/` for past context.
  - At the end of the session, write a `session-summary-YYYY-MM-DD.md` note (see existing ones in the folder for format) summarizing what changed and why.
  - **Also refresh the relevant topic note's own "CURRENT STATE" section in place** whenever a session's finding changes something an existing topic note already covers (`trading-strategy-status.md`, `best-ideas-status.md`, `llm-agent-status.md`, `strategy-validation-pipeline.md`, etc.). A dated session-summary note alone is NOT enough — every topic note in this vault was found frozen at its original 2026-08-23 port date and silently stale for a month (found+fixed 2026-09-21) because updates only ever went into new dated notes, never back into the topic note itself. Keep each topic note's "CURRENT STATE" section short (what's true now, and why) and let full chronological detail stay below a divider — don't duplicate the current-state facts down into a new paragraph at the bottom instead of updating the top.
- **ruflo memory** (`mcp__ruflo__memory_store` / `memory_search`): fast semantic key-value recall across sessions, for smaller facts/decisions that don't warrant a full vault note.
  - Requires `CLAUDE_FLOW_ENABLE_NATIVE_BRIDGE_ON_WINDOWS=1` in `~/.claude/settings.json` (set 2026-08-25) — without it, writes silently fail on Windows.

Do not skip either one because a task "seems small" — check both at the start, write to both (vault note for anything architecturally relevant, ruflo for quick facts) before ending the session.

The Claude Code native memory system (`C:\Users\ks303\.claude\projects\d--Trading-Sandbox\memory\`, governed by Claude's own global instructions, not this file) mirrors these same topic notes and needs the same discipline: update each memory file's own "CURRENT STATE" section in place rather than only prepending a new dated paragraph, and keep `MEMORY.md`'s index entries to real one-line pointers (~150 chars) — it's loaded into every session automatically, so bloat there costs tokens on every single session, not just when a topic is relevant (found+fixed 2026-09-21, index had grown to 2,000+ character entries).