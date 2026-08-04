"""Failure notifications for scheduled/unattended runs (daily_run.py).

Discord-webhook-only for now, chosen for being free and requiring no SMTP
credentials -- just a webhook URL from a Discord server you control (Server
Settings -> Integrations -> Webhooks -> New Webhook -> Copy Webhook URL).
Put it in your local `.env` file as DISCORD_WEBHOOK_URL=... (same file
MONGODB_URI already lives in) -- never paste a webhook URL into a chat or
commit it, it's a secret capability token, not just an identifier.

Degrades to a no-op (prints a warning, never raises) if DISCORD_WEBHOOK_URL
isn't configured or the request fails -- same fallback philosophy this repo
already uses for MongoDB/Gemini connectivity. A notification failure must
never be allowed to mask or replace the original failure it's reporting.
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


def notify(message: str) -> bool:
    """Best-effort Discord webhook post. Returns True on apparent success,
    False otherwise (including when unconfigured) -- never raises."""
    webhook_url = os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        print(f"[notify] {DISCORD_WEBHOOK_URL_ENV} not set, message not sent:\n{message}")
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


def notify_with_file(message: str, filename: str, content: str) -> bool:
    """Like notify(), but attaches `content` as a downloadable/viewable text
    file instead of truncating it into the 2000-char message body -- lets a
    full run's output (which routinely runs to several thousand characters
    once you include a 150+-ticker skip list and a settlement summary) show
    up in Discord without either getting cut off or spamming multiple
    split messages. `message` is still capped at MAX_DISCORD_MESSAGE_CHARS
    (it's meant to be a short status line, not the full report) -- `content`
    (the attachment) has no such practical limit, well under Discord's
    standard 8MB-per-file webhook cap for any realistic run's output.
    Returns True on apparent success, False otherwise (including when
    unconfigured) -- never raises."""
    webhook_url = os.environ.get(DISCORD_WEBHOOK_URL_ENV)
    if not webhook_url:
        print(f"[notify] {DISCORD_WEBHOOK_URL_ENV} not set, message not sent:\n{message}")
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
