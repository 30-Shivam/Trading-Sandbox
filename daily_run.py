"""
Scheduled daily driver -- closes the automation gap (item 1 in
improvements.txt's NOT DONE list) without needing full containerization.
Runs ingest.py, settle_trades.py, then review_positions.py back-to-back as
SUBPROCESSES (isolated from this process, so a crash or sys.exit() in any
of them can't take daily_run.py down with it -- their real exit code is
all that matters here), and reports EVERY run (success or failure) via
notifications.py --
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

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import notifications

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, script: str) -> tuple[bool, str]:
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
    reintroducing the exact same class of crash one level up."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=SCRIPT_DIR,
    )
    ok = result.returncode == 0
    header = f"=== {name}: {'OK' if ok else 'FAILED'} (exit {result.returncode}) ==="
    body = result.stdout + result.stderr
    return ok, f"{header}\n{body}"


def main() -> int:
    start = time.time()
    started_at = datetime.now(timezone.utc)

    ok_ingest, output_ingest = run_step("ingest.py", "ingest.py")
    print(output_ingest)

    ok_settle, output_settle = run_step("settle_trades.py", "settle_trades.py")
    print(output_settle)

    ok_review, output_review = run_step("review_positions.py", "review_positions.py")
    print(output_review)

    elapsed = time.time() - start
    all_ok = ok_ingest and ok_settle and ok_review

    status = "OK" if all_ok else "FAILED"
    print(f"daily_run.py: {status} ({elapsed:.0f}s elapsed).")

    message = f"[Trading-Sandbox] daily_run.py {status} -- {elapsed:.0f}s elapsed, {started_at:%Y-%m-%d %H:%M} UTC"
    full_report = f"{output_ingest}\n\n{output_settle}\n\n{output_review}"
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

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
