"""Experimental LLM-derived trading judgment -- the "LLM Agent" dashboard
tab, built after reviewing github.com/TauricResearch/TradingAgents for
inspiration and deliberately NOT mirroring its full multi-agent (separate
fundamentals/sentiment/news/technical analysts + bull/bear debate + risk
team) architecture. One synthesis call per candidate ticker instead --
matches the "lean v1" discipline every mechanical strategy this session
started with, and keeps cost/rate-limit exposure and surface-area-to-get-
wrong low before this even reaches a real validation window.

Unlike every strategy in `swingtrade/` (breakout, pullback, breakout_retest,
week52_high), this can NEVER be validated via `benchmark_random_entry.py`/
`optimize.py`'s walk-forward machinery. Re-prompting a model today with
"historical" context to fake a backtest risks the model already knowing
what happened next from training data -- lookahead bias baked into the
test itself, not filtered out by it. `ai_context.py` sidesteps this same
problem by staying a pure informational summary, never touching
Trade_Score, never called from the backtester; this module extends that
same discipline to a genuine Buy/Hold/Avoid judgment instead, but the
consequence is real: this can ONLY be validated PROSPECTIVELY (real time
passing, real settled trades via the normal settle_trades.py pipeline),
never backtested. See dip_buy_analyzer.py's "LLM Agent" tab for the
trust-floor counter that makes this visible in the UI, not just here.

Deliberately never wired into swingtrade/backtest.py, optimize.py, or
benchmark_random_entry.py -- calling historical walk-forward replay on an
LLM strategy would silently produce lookahead-contaminated nonsense.

Two free-tier providers, PRIMARY + FALLBACK, picked by comparing actual
published limits (see github.com/open-free-llm-api/awesome-freellm-apis)
rather than defaulting to whichever was already wired up:
  - PRIMARY: Google Gemini (gemini-2.5-flash, same model ai_context.py
    already uses, via `google-genai`) -- 1,500 requests/day, 1M-token
    context, no credit card, and the integration this codebase already
    has proven working (ai_context.py). Genuinely the better option on
    every axis that matters here, not just the incumbent by default.
  - FALLBACK: Groq (llama-3.3-70b-versatile, via the `openai` SDK pointed
    at Groq's OpenAI-compatible endpoint) -- 250 requests/day, 262K-token
    context, no credit card. Only tried if Gemini is unavailable or its
    call fails/returns an unusable response -- at this tab's real volume
    (<= MAX_LLM_CANDIDATES calls/day, see dip_buy_analyzer.py), either
    provider's daily limit is far more than enough; the fallback exists
    for resilience (one provider having a bad day), not because Gemini's
    limits are a real constraint.
  - Explicitly did NOT add OpenRouter: its free tier requires a paid
    top-up for sustained use, unlike every other option here -- not a
    genuinely free fallback, so not worth the added complexity.

Both providers share the exact same JSON-shape validation (_parse_response)
and the exact same prompt -- only the API call differs. Degrades to
unavailable (never raises) if NEITHER provider is configured, and
evaluate_ticker() returns None (never raises) if BOTH fail -- same
fallback philosophy as ai_context.py and this repo's MongoDB connectivity
checks: a flaky or unconfigured LLM call must never break a scan that
would otherwise succeed.
"""

import json
import os

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_HEADLINES = 5
MAX_OUTPUT_TOKENS = 400
VALID_DECISIONS = ("Buy", "Hold", "Avoid")


def _gemini_available() -> bool:
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _groq_available() -> bool:
    if not os.environ.get("GROQ_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def is_available() -> bool:
    """True if a live call to evaluate_ticker() stands a chance of working
    -- at least ONE of Gemini (primary) or Groq (fallback) is configured
    and its package installed. Callers should gate the LLM Agent tab on
    this rather than discovering unavailability via a failed call (mirrors
    ai_context.is_available())."""
    return _gemini_available() or _groq_available()


def _parse_response(text: str) -> dict | None:
    """Parse and validate a model's JSON response -- shared by both
    providers, since the expected shape is identical regardless of which
    one produced it. Returns None (never raises) on any malformed shape --
    a confidently-wrong JSON blob is not more trustworthy than no response
    at all."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    decision = data.get("decision")
    if decision not in VALID_DECISIONS:
        return None

    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if not (0 <= confidence <= 100):
        return None

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None

    return {"decision": decision, "confidence": confidence, "rationale": rationale.strip()}


def _build_prompt(ticker: str, context: dict) -> tuple[str, int]:
    """Build the (identical, provider-agnostic) prompt from `context` --
    see evaluate_ticker()'s docstring for expected keys. Returns
    (prompt, trimmed_headline_count) since the prompt text itself embeds
    the count."""
    headlines = context.get("headlines") or []
    trimmed_headlines = headlines[:MAX_HEADLINES]
    headline_block = "\n".join(f"- {h}" for h in trimmed_headlines) if trimmed_headlines else "(none available)"

    mechanical_scores = context.get("mechanical_scores") or {}
    mechanical_block = (
        ", ".join(f"{strategy}={score:.1f}" for strategy, score in mechanical_scores.items())
        or "(none)"
    )

    fundamentals = context.get("fundamentals") or {}
    fundamentals_block = ", ".join(f"{k}={v}" for k, v in fundamentals.items()) or "(unavailable)"

    prompt = (
        f"A mechanical price/volume trading system flagged {ticker} today with the "
        f"following technical readings: Last_Close={context.get('last_close')}, "
        f"RSI={context.get('rsi')}, ATR={context.get('atr')}, "
        f"mechanical Trade_Score(s) by strategy: {mechanical_block}. "
        f"Catalyst_Warning={context.get('catalyst_warning')}, "
        f"next earnings date={context.get('next_earnings_date')}. "
        f"Basic fundamentals: {fundamentals_block}. "
        f"Its {len(trimmed_headlines)} most recent news headlines:\n{headline_block}\n\n"
        "Given all of this, is this ticker worth a human's attention as a candidate "
        "swing trade right now? Respond ONLY with a JSON object matching exactly this "
        'shape: {"decision": "Buy" | "Hold" | "Avoid", "confidence": <integer 0-100>, '
        '"rationale": "<2-3 plain-language sentences explaining the call>"}. '
        '"Buy" means genuinely worth considering now, "Hold" means mixed/uncertain but '
        'worth tracking, "Avoid" means a real red flag outweighs the technical setup. '
        "Base confidence on how strong the combined evidence is, not just the mechanical "
        "score alone."
    )
    return prompt, len(trimmed_headlines)


def _call_gemini(prompt: str) -> dict | None:
    """PRIMARY provider -- see module docstring for why. Returns None
    (never raises) on any failure so evaluate_ticker() can fall through to
    Groq."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return None
        return _parse_response(text)
    except Exception:
        return None


def _call_groq(prompt: str) -> dict | None:
    """FALLBACK provider -- only tried if Gemini is unavailable or its call
    fails/returns an unusable response. Uses the standard `openai` SDK
    pointed at Groq's OpenAI-compatible endpoint (see GROQ_BASE_URL) --
    Groq's own API is a drop-in match for the Chat Completions shape, no
    Groq-specific SDK needed. Returns None (never raises) on any failure."""
    try:
        import openai
    except ImportError:
        return None
    try:
        client = openai.OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        return _parse_response(text)
    except Exception:
        return None


def evaluate_ticker(ticker: str, context: dict) -> dict | None:
    """Ask an LLM for a genuine Buy/Hold/Avoid judgment on `ticker`, given
    `context` -- a dict of whatever's already been computed this page
    load, expected keys: `last_close`, `rsi`, `atr`, `mechanical_scores`
    (dict of strategy label -> Trade_Score for whichever mechanical
    strategies flagged this ticker today), `catalyst_warning`,
    `next_earnings_date`, `headlines` (list[str]), and `fundamentals`
    (dict, e.g. {"pe_ratio":..., "market_cap":..., "sector":...} from
    yfinance's Ticker.info -- any subset, missing keys are fine).

    Tries Gemini (primary) first; falls back to Groq only if Gemini is
    unavailable or its call fails/returns an unusable response -- see the
    module docstring for why Gemini is primary (better published limits
    AND the already-proven integration, not just picked by default).

    Returns {"decision": "Buy"|"Hold"|"Avoid", "confidence": 0-100,
    "rationale": str} or None (never raises) if BOTH providers are
    unavailable/fail -- a flaky or unconfigured LLM call must never break
    a scan that would otherwise succeed, same philosophy as
    ai_context.summarize_ticker_context().

    This is explicitly a SECOND OPINION, not a blind scan: the prompt
    frames the ticker as one a mechanical system already flagged, and asks
    the model to weigh in given that context plus recent headlines and
    basic fundamentals -- not to discover candidates on its own."""
    prompt, _ = _build_prompt(ticker, context)

    if _gemini_available():
        result = _call_gemini(prompt)
        if result is not None:
            return result

    if _groq_available():
        return _call_groq(prompt)

    return None
