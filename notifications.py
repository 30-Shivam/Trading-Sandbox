"""Notifications for scheduled/unattended runs (daily_run.py, ingest.py).

Discord-webhook-only for now, chosen for being free and requiring no SMTP
credentials -- just a webhook URL from a Discord server you control (Server
Settings -> Integrations -> Webhooks -> New Webhook -> Copy Webhook URL).
Put it in your local `.env` file as DISCORD_WEBHOOK_URL=... (same file
MONGODB_URI already lives in) -- never paste a webhook URL into a chat or
commit it, it's a secret capability token, not just an identifier.

Degrades to a no-op (prints a warning, never raises) if no webhook is
configured or the request fails -- same fallback philosophy this repo
already uses for MongoDB/Gemini connectivity. A notification failure must
never be allowed to mask or replace the original failure it's reporting.

Two layers of notification exist, deliberately separate:
  1. daily_run.py's own run-status report (every run, OK/FAILED + full
     combined output as an attachment) -- unchanged by the below, always
     uses the shared DISCORD_WEBHOOK_URL, for overall job health/crash
     monitoring regardless of results.
  2. Per-strategy signal notifications (see ingest.py) -- fire ONLY on days
     a strategy finds a real Strong Buy/Buy, optionally routed to that
     strategy's OWN Discord channel via get_strategy_webhook_url() below,
     falling back to the shared DISCORD_WEBHOOK_URL if that specific
     channel isn't configured -- so every strategy posts somewhere from
     day one, and channels can be split apart incrementally.
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
