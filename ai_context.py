"""Live-only, human-facing context aid for Strong Buy/Buy signals -- the
scoped first step recommended in improvements.txt's agentic/multi-LLM item.

Deliberately narrow: an LLM-generated summary of a ticker's recent
headlines is surfaced next to a signal for a human to read, and NOTHING
else. It never touches Trade_Score, Signal, position sizing, or the
backtester. That scoping sidesteps the two hardest problems flagged before
any of this was built:

  1. Look-ahead bias. An LLM asked to evaluate a ticker "as of" a historical
     date can still draw on training knowledge of what happened after that
     date -- backtesting an LLM-based signal honestly needs genuinely
     point-in-time text data, which doesn't exist here. Staying live-only
     (never called from swingtrade/backtest.py) means this never needs to
     answer that question.
  2. A fluent, confident-sounding summary is not automatically a MORE valid
     signal than a bare RSI number -- this project's own random-entry
     benchmark proved a plausible, well-tuned mechanical signal (RSI
     timing) can carry zero real predictive information. An LLM opinion
     would need the same kind of validation before being trusted as a
     score input, and that bar hasn't been cleared (or attempted) here.

Uses Google Gemini's free tier (not Anthropic -- deliberate choice so this
feature costs nothing to run) via the `google-genai` package. Degrades to
unavailable (never raises) if GEMINI_API_KEY isn't set or the package isn't
installed -- same fallback philosophy this repo already uses for MongoDB
connectivity.
"""

import os

MODEL = "gemini-2.5-flash"
MAX_HEADLINES = 5
MAX_OUTPUT_TOKENS = 300


def is_available() -> bool:
    """True if a live call to summarize_ticker_context() stands a chance of
    working -- an API key is configured and the google-genai package is
    installed. Callers should gate the feature (e.g. a sidebar checkbox) on
    this rather than discovering unavailability via a failed call."""
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def summarize_ticker_context(ticker: str, signal: str, headlines: list[str]) -> str | None:
    """A short (2-3 sentence) plain-language summary of what `headlines`
    suggest is going on with `ticker`, for a human reviewing a mechanically
    generated `signal` (e.g. "Strong Buy"). Explicitly informational --
    the prompt itself instructs the model not to rate, score, or recommend
    a trade action, keeping this a research aid rather than a second
    scoring system. Returns None (never raises) on missing headlines, a
    missing/invalid API key, or any API failure -- a flaky or unconfigured
    LLM call must never break a scan that would otherwise succeed."""
    if not headlines:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        trimmed = headlines[:MAX_HEADLINES]
        headline_block = "\n".join(f"- {h}" for h in trimmed)
        prompt = (
            f"A mechanical price/volume trading system just flagged {ticker} as a "
            f"\"{signal}\" signal. No news or text data was used to generate that "
            f"signal -- it's purely technical. Here are its {len(trimmed)} most recent "
            f"news headlines:\n{headline_block}\n\n"
            "In 2-3 plain-language sentences, summarize what these headlines suggest "
            "is going on with this company right now. Do not say whether to buy or "
            "sell, do not assign a rating or score, and do not speculate beyond what "
            "the headlines actually say -- just give the context a human would want "
            "before looking at the chart themselves."
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
        )
        text = (response.text or "").strip()
        return text or None
    except Exception:
        return None
