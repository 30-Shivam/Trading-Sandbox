"""
Scheduled daily driver -- closes the automation gap (item 1 in
improvements.txt's NOT DONE list) without needing full containerization.
Runs ingest.py then settle_trades.py back-to-back as SUBPROCESSES (isolated
from this process, so a crash or sys.exit() in either can't take
daily_run.py down with it -- their real exit code is all that matters
here), and reports EVERY run (success or failure) via notifications.py
(Discord webhook) -- a short status line plus the full combined stdout/
stderr of both steps as an attached text file, so Discord doubles as your
observability layer even when this runs somewhere (Task Scheduler) that
doesn't otherwise capture output anywhere visible. Failures still get the
same treatment, just with a FAILED status line instead of OK.

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
    message body."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
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

    elapsed = time.time() - start
    all_ok = ok_ingest and ok_settle

    status = "OK" if all_ok else "FAILED"
    print(f"daily_run.py: {status} ({elapsed:.0f}s elapsed).")

    message = f"[Trading-Sandbox] daily_run.py {status} -- {elapsed:.0f}s elapsed, {started_at:%Y-%m-%d %H:%M} UTC"
    full_report = f"{output_ingest}\n\n{output_settle}"
    filename = f"daily_run_{started_at:%Y%m%d_%H%M%S}.txt"
    notifications.notify_with_file(message, filename, full_report)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
