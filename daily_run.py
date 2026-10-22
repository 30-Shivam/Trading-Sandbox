"""
Scheduled daily driver -- closes the automation gap (item 1 in
improvements.txt's NOT DONE list) without needing full containerization.
Runs ingest.py, settle_trades.py, review_positions.py,
llm_strategy_research.py --run-once, then daily_briefing.py back-to-back
as SUBPROCESSES (isolated from this process, so a crash or sys.exit() in
any of them can't take daily_run.py down with it -- their real exit code
is all that matters here), and
reports EVERY run (success or failure) via notifications.py --
a short status line plus the full combined stdout/stderr of both steps,
sent through TWO independent channels: a Discord webhook (if
DISCORD_WEBHOOK_URL is configured) and GitHub Actions' own step summary
(always available in CI, zero configuration -- see
notifications.notify_github_summary()). This two-channel design exists
because relying on Discord alone had a real, confirmed failure mode: if
the webhook secret was never set, every run -- including failures -- was
silently invisible, which is exactly how a real run of consecutive
scheduled-run failures went unnoticed until a human happened to check
manually (see improvements.txt). A failed run also prints a GitHub Actions
`::error::` annotation, which is what feeds GitHub's own optional
"email me when a scheduled workflow fails" account setting.

Meant to be triggered once a day (weekdays, after US market close) by
GitHub Actions or Windows Task Scheduler -- see TUTORIAL.md section 9 for
both setup paths. Safe to run more often, or on non-trading days too: both
ingest.py and settle_trades.py are upsert-based and idempotent, so an extra
run just harmlessly confirms there's nothing new to do.

Known limitation, stated plainly: this can only report from WITHIN a run
that actually started. If the scheduler itself doesn't fire (Task
Scheduler task disabled/PC off, or GitHub Actions has an outage), nothing
here can detect that -- checking Task Scheduler's own History tab / GitHub's
Actions tab, or a separate external dead-man's-switch service (e.g.
healthchecks.io's free tier), would be needed to close that specific gap.
Not built here; flagged rather than silently assumed away.

Usage:
    py -3 daily_run.py
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import notifications

SCRIPT_DIR = Path(__file__).resolve().parent


# Per-step caps sized to fit inside the workflow's 45-minute limit with room to report.
STEP_TIMEOUT_SEC = {
    "ingest.py": 22 * 60, "settle_trades.py": 8 * 60, "review_positions.py": 3 * 60,
    "llm_strategy_research.py": 6 * 60, "daily_briefing.py": 3 * 60, "paper_trade.py": 3 * 60, "rebalance.py": 2 * 60,
}


def run_step(
    name: str, script: str, timeout_sec: int, args: list[str] | None = None,
) -> tuple[bool, str]:
    """Runs `script` as a subprocess, returns (ok, full combined stdout+stderr
    prefixed with a one-line status header) -- untruncated, since the
    caller attaches this as a file rather than cramming it into a Discord
    message body.

    Explicit encoding="utf-8" (with errors="replace") -- ingest.py's own
    stdout/stderr are forced to UTF-8 (see its own reconfigure() call,
    2026-08-26) specifically because real LLM-generated rationale text can
    contain characters Windows' default codepage can't encode; without a
    matching explicit encoding here, subprocess.run's text=True would
    decode those UTF-8 bytes using the OS default codepage instead,
    reintroducing the exact same class of crash one level up. Same reason
    llm_strategy_research.py itself needed the identical stdout/stderr
    reconfigure() fix applied directly to it (2026-08-29) -- this
    subprocess-level encoding only protects daily_run.py's own capture,
    not a crash inside the child script before it ever writes output."""
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(SCRIPT_DIR / script), *(args or [])],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=SCRIPT_DIR,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        # A hang used to burn the whole workflow timeout and get the job
        # cancelled with NO report (daily_run.py itself was killed first).
        # Killing just this step and attaching its partial output means a
        # hang shows up as a normal FAILED run saying where it stalled.
        partial = "".join(
            p.decode("utf-8", "replace") if isinstance(p, bytes) else (p or "") for p in (exc.stdout, exc.stderr)
        )
        return False, f"=== {name}: TIMED OUT after {timeout_sec}s ===\n{partial[-20000:]}"
    ok = result.returncode == 0
    header = f"=== {name}: {'OK' if ok else 'FAILED'} (exit {result.returncode}) ==="
    body = result.stdout + result.stderr
    return ok, f"{header}\n{body}"


def ping_healthcheck(ok: bool, url: str | None = None) -> bool:
    """Dead-man's switch (healthchecks.io-style): GET HEALTHCHECK_URL on success,
    HEALTHCHECK_URL/fail on failure. If the scheduler never fires, no ping arrives
    and the service alerts -- the one failure this script cannot report from
    inside. No-op without HEALTHCHECK_URL; never raises."""
    url = url or os.getenv("HEALTHCHECK_URL")
    if not url:
        return False
    try:
        import requests
        requests.get(url.rstrip("/") + ("" if ok else "/fail"), timeout=10)
        return True
    except Exception as exc:
        print(f"[WARN] healthcheck ping failed: {exc}", file=sys.stderr)
        return False


def extract_coverage(ingest_output: str) -> str | None:
    """The `COVERAGE n/m (pct) [DEGRADED] ...` line ingest.py prints (see
    universe.format_coverage_line), or None if the scan never got that far."""
    match = re.search(r"^COVERAGE .*$", ingest_output, re.MULTILINE)
    return match.group(0) if match else None


def main() -> int:
    start = time.time()
    started_at = datetime.now(timezone.utc)

    ok_ingest, output_ingest = run_step("ingest.py", "ingest.py", STEP_TIMEOUT_SEC["ingest.py"])
    print(output_ingest)

    # Optional: only when PAPER_TRADING_ENABLED=1. Runs after ingest (needs
    # today's signals) and BEFORE settle (so settlement sees real fills).
    paper_enabled = os.getenv("PAPER_TRADING_ENABLED") == "1"
    ok_paper, output_paper = (True, "")
    if paper_enabled:
        ok_paper, output_paper = run_step("paper_trade.py", "paper_trade.py", STEP_TIMEOUT_SEC["paper_trade.py"])
        print(output_paper)
        ok_core, output_core = run_step("rebalance.py", "rebalance.py", STEP_TIMEOUT_SEC["rebalance.py"])
        print(output_core)
        ok_paper, output_paper = ok_paper and ok_core, f"{output_paper}\n{output_core}"

    ok_settle, output_settle = run_step("settle_trades.py", "settle_trades.py", STEP_TIMEOUT_SEC["settle_trades.py"])
    print(output_settle)

    ok_review, output_review = run_step("review_positions.py", "review_positions.py", STEP_TIMEOUT_SEC["review_positions.py"])
    print(output_review)

    # Self-contained (own fetch_history()/watchlist read, no dependency on
    # ingest.py's output) and writes only to Strategy_Research_Journal --
    # never Trade_Signals/System_Config, so its outcome can't affect
    # anything else in this pipeline. Failure here (e.g. every LLM provider
    # rate-limited that day) is real but low-stakes -- degrades to a
    # graceful "no usable proposal" cycle rather than crashing (see
    # llm_strategy_research.py's own docstring), same non-blocking
    # treatment run_llm_agent() gets inside ingest.py itself. Placed before
    # daily_briefing.py (which runs last) on the same "nothing downstream
    # depends on daily_briefing's own output" logic every other step here
    # already follows.
    ok_research, output_research = run_step(
        "llm_strategy_research.py", "llm_strategy_research.py", STEP_TIMEOUT_SEC["llm_strategy_research.py"],
        args=["--run-once"],
    )
    print(output_research)

    # Runs LAST and unconditionally -- it only reads/synthesizes what the
    # steps above already wrote, so it should report the day's FINAL
    # state even if an earlier step failed partway through (same as
    # settle_trades.py already running regardless of ingest.py's outcome).
    ok_briefing, output_briefing = run_step("daily_briefing.py", "daily_briefing.py", STEP_TIMEOUT_SEC["daily_briefing.py"])
    print(output_briefing)

    elapsed = time.time() - start
    all_ok = ok_ingest and ok_paper and ok_settle and ok_review and ok_research and ok_briefing

    status = "OK" if all_ok else "FAILED"
    # One line naming exactly which step(s) failed -- the per-step headers
    # are buried in a long log, and a bare "FAILED" gave no hint where.
    step_results = {
        "ingest": ok_ingest, "settle": ok_settle, "review": ok_review,
        "research": ok_research, "briefing": ok_briefing,
    }
    if paper_enabled:
        step_results["paper"] = ok_paper
    step_line = " ".join(f"{n}={'ok' if ok else 'FAILED'}" for n, ok in step_results.items())
    print(f"daily_run.py: {status} ({elapsed:.0f}s elapsed). Steps: {step_line}")

    coverage_line = extract_coverage(output_ingest)
    if coverage_line is None:
        coverage_line = "COVERAGE unknown (ingest never reached scoring)"
    degraded = all_ok and "DEGRADED" in coverage_line
    if degraded:
        status = "DEGRADED"
    message = (
        f"[Trading-Sandbox] daily_run.py {status} -- {elapsed:.0f}s elapsed, "
        f"{started_at:%Y-%m-%d %H:%M} UTC\n{step_line}\n{coverage_line}"
    )
    full_report = f"{output_ingest}\n\n{output_paper}\n\n{output_settle}\n\n{output_review}\n\n{output_research}\n\n{output_briefing}"
    filename = f"daily_run_{started_at:%Y%m%d_%H%M%S}.txt"
    notifications.notify_with_file(message, filename, full_report)
    # Second, independent channel -- always live in GitHub Actions (no
    # DISCORD_WEBHOOK_URL required), specifically so a run of real
    # failures can never go silently unnoticed again the way it did
    # before this was added (see improvements.txt).
    notifications.notify_github_summary(message, full_report)
    if not all_ok:
        # A recognized GitHub Actions annotation -- renders as a visible
        # red error on the run page and is what feeds GitHub's own
        # "notify me when a scheduled workflow fails" account setting
        # (Settings -> Notifications -> Actions), the free/zero-code path
        # to an actual proactive alert on top of the two channels above.
        print(f"::error title=daily_run.py FAILED::{message}")

    ping_healthcheck(all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
