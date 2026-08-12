"""Notifications for scheduled/unattended runs (daily_run.py, ingest.py).

Discord webhook, chosen for being free and requiring no SMTP credentials
-- just a webhook URL from a Discord server you control (Server Settings
-> Integrations -> Webhooks -> New Webhook -> Copy Webhook URL). Put it in
your local `.env` file as DISCORD_WEBHOOK_URL=... (same file MONGODB_URI
already lives in) -- never paste a webhook URL into a chat or commit it,
it's a secret capability token, not just an identifier.

Degrades to a no-op (prints a warning, never raises) if no webhook is
configured or the request fails -- same fallback philosophy this repo
already uses for MongoDB/Gemini connectivity. A notification failure must
never be allowed to mask or replace the original failure it's reporting.
**This degrade-quietly design has a real, confirmed failure mode of its
own, though**: if DISCORD_WEBHOOK_URL is simply never configured, every
run -- including failures -- was silently invisible outside whatever the
raw job log happened to show, which is exactly how a run of real,
consecutive scheduled-run failures went unnoticed for a while (see
improvements.txt). notify_github_summary() below exists specifically to
give failure reporting a channel that works with zero setup, as a second,
independent line of defense against exactly that gap.

Three notification paths exist, deliberately separate:
  1. daily_run.py's own run-status report (every run, OK/FAILED + full
     combined output as an attachment) -- uses the shared
     DISCORD_WEBHOOK_URL (if configured) AND, on failure specifically,
     also writes to GitHub's own step summary (see notify_github_summary()
     -- always available in GitHub Actions, no configuration needed) --
     for overall job health/crash monitoring regardless of results.
  2. Per-strategy signal notifications (see ingest.py) -- fire ONLY on days
     a strategy finds a real Strong Buy/Buy, optionally routed to that
     strategy's OWN Discord channel via get_strategy_webhook_url() below,
     falling back to the shared DISCORD_WEBHOOK_URL if that specific
     channel isn't configured -- so every strategy posts somewhere from
     day one, and channels can be split apart incrementally.
  3. notify_github_summary() -- GitHub Actions step-summary writes, used
     by (1) above. Requires no secret at all (reads the
     $GITHUB_STEP_SUMMARY path GitHub itself sets during a run), so it's
     the one channel that's ALWAYS live in CI regardless of what the user
     has or hasn't configured.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL_ENV = "DISCORD_WEBHOOK_URL"
MAX_DISCORD_MESSAGE_CHARS = 2000  # Discord's own hard limit on a message's "content" field
REQUEST_TIMEOUT_SEC = 15


def is_available() -> bool:
    return bool(os.environ.get(DISCORD_WEBHOOK_URL_ENV))


def get_strategy_webhook_url(strategy: str) -> str | None:
    """Look up a per-strategy Discord webhook, e.g. strategy="breakout_retest"
    -> DISCORD_WEBHOOK_URL_BREAKOUT_RETEST. Derived directly from
    config.strategy (not from any human-readable display label), so it
    can't silently drift from the actual strategy tag. Returns None if
    that specific env var isn't set -- deliberately no fallback logic
    here; pass the result straight to notify(webhook_url=...), whose own
    default-parameter fallback to the shared DISCORD_WEBHOOK_URL handles
    "not configured yet" in one place."""
    return os.environ.get(f"{DISCORD_WEBHOOK_URL_ENV}_{strategy.upper()}")


def notify(message: str, webhook_url: str | None = None) -> bool:
    """Best-effort Discord webhook post. `webhook_url`, if given, is used
    directly (e.g. a per-strategy channel via get_strategy_webhook_url());
    if not given (the default -- every pre-existing call site keeps this
    behavior unchanged), falls back to the shared DISCORD_WEBHOOK_URL env
    var, exactly as before this parameter was added. Returns True on
    apparent success, False otherwise (including when unconfigured) --
    never raises."""
    webhook_url = webhook_url or os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        print(f"[notify] no webhook configured, message not sent:\n{message}")
        return False
    try:
        response = requests.post(
            webhook_url, json={"content": message[:MAX_DISCORD_MESSAGE_CHARS]}, timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[notify] Discord webhook post failed: {exc}")
        return False


def notify_with_file(message: str, filename: str, content: str, webhook_url: str | None = None) -> bool:
    """Like notify(), but attaches `content` as a downloadable/viewable text
    file instead of truncating it into the 2000-char message body -- lets a
    full run's output (which routinely runs to several thousand characters
    once you include a 150+-ticker skip list and a settlement summary) show
    up in Discord without either getting cut off or spamming multiple
    split messages. `message` is still capped at MAX_DISCORD_MESSAGE_CHARS
    (it's meant to be a short status line, not the full report) -- `content`
    (the attachment) has no such practical limit, well under Discord's
    standard 8MB-per-file webhook cap for any realistic run's output.
    `webhook_url` behaves exactly as in notify() -- explicit override, or
    falls back to the shared DISCORD_WEBHOOK_URL when not given. Returns
    True on apparent success, False otherwise (including when
    unconfigured) -- never raises."""
    webhook_url = webhook_url or os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        print(f"[notify] no webhook configured, message not sent:\n{message}")
        return False
    try:
        payload_json = json.dumps({"content": message[:MAX_DISCORD_MESSAGE_CHARS]})
        files = {"files[0]": (filename, content.encode("utf-8"), "text/plain")}
        response = requests.post(
            webhook_url, data={"payload_json": payload_json}, files=files, timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[notify] Discord webhook post (with attachment) failed: {exc}")
        return False


GITHUB_STEP_SUMMARY_ENV = "GITHUB_STEP_SUMMARY"
MAX_GITHUB_SUMMARY_CONTENT_CHARS = 60_000  # GitHub's own step-summary cap is 1MB; this is a
                                            # generous but sane ceiling for a single run's report


def notify_github_summary(message: str, content: str) -> bool:
    """Writes `message` + `content` to GitHub Actions' own step summary
    (the file path in $GITHUB_STEP_SUMMARY -- a native Actions feature:
    anything appended there renders as markdown directly on the workflow
    run's summary page, visible in the Actions tab with zero webhooks or
    secrets needed). Exists because Discord-based alerting depends on
    DISCORD_WEBHOOK_URL actually being configured -- if it isn't, every
    run (including failures) was silently invisible outside whatever the
    raw job log happened to show, which is exactly what let a run of real
    failures go unnoticed. This is a second, independent channel that
    works out of the box in GitHub Actions specifically, no setup required.

    No-op (returns False, never raises) outside GitHub Actions ($GITHUB_STEP_SUMMARY
    unset -- e.g. a local run) or on any write failure -- same
    degrade-quietly philosophy as notify()/notify_with_file(). `content`
    is capped at MAX_GITHUB_SUMMARY_CONTENT_CHARS (GitHub's own hard limit
    on step summaries is 1MB total across a whole job; this leaves
    generous headroom rather than risking silently exceeding it)."""
    summary_path = os.environ.get(GITHUB_STEP_SUMMARY_ENV)
    if not summary_path:
        return False
    try:
        trimmed = content[:MAX_GITHUB_SUMMARY_CONTENT_CHARS]
        truncated_note = "\n\n_(truncated)_" if len(content) > MAX_GITHUB_SUMMARY_CONTENT_CHARS else ""
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"## {message}\n\n```\n{trimmed}\n```{truncated_note}\n")
        return True
    except Exception as exc:
        print(f"[notify] GitHub step summary write failed: {exc}")
        return False
