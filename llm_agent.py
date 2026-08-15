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

Input richness (added after reviewing TauricResearch/TradingAgents for
inspiration a second time): the prompt now optionally folds in a shared
market-wide "macro" snapshot (see market_data.get_macro_snapshot() --
VIX level/change + broad-index headlines, fetched ONCE per dashboard page
load, not per ticker, so this adds zero extra LLM calls) alongside the
ticker-specific inputs, and the requested JSON output gained its own
"news_sentiment" field (Bullish/Bearish/Neutral) so sentiment is a
trackable signal in its own right instead of buried in rationale prose.
Deliberately still ONE synthesis call per candidate, not a multi-agent
debate pipeline -- each input fetcher (headlines, fundamentals, macro) is
its own small function precisely so a future multi-agent version could
reuse them behind separate per-role prompts, but that orchestration layer
is intentionally not built yet: this tab hasn't passed its own
prospective trust floor (20-30 settled trades, 4-6 weeks, see
dip_buy_analyzer.py's LLM Agent tab) and a debate architecture would add
real cost/complexity on top of a still-unproven signal.

Prompt-variant A/B testing (PROMPT_VARIANTS, see evaluate_ticker()'s
`variant` param): since real backtesting is impossible here, the only way
to compare prompt-framing choices is to run several concurrently against
the SAME real candidates and let real settled trades decide. ingest.py's
headless run evaluates every candidate under all variants; the dashboard's
live tab always uses the default ("balanced") only, to avoid showing a
user multiple conflicting verdicts for one ticker. Each variant logs under
its own Trade_Signals `strategy` field (e.g. "llm_agent_evidence_strict")
so they settle and get graded completely independently -- see
storage/signals.py's unique-index note on why that field must be distinct
per strategy.
"""

import json
import os

GEMINI_MODEL = "gemini-3.5-flash"  # gemini-2.5-flash was retired for this
                                    # account ("no longer available to new
                                    # users", a permanent 404, not
                                    # transient) -- see improvements.txt.
                                    # 3.5-flash is a "thinking" model by
                                    # default: without thinking_budget=0
                                    # below, it can spend its entire
                                    # max_output_tokens budget on internal
                                    # reasoning and return truncated JSON.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_HEADLINES = 5
MAX_MACRO_HEADLINES = 5
MAX_OUTPUT_TOKENS = 400
VALID_DECISIONS = ("Buy", "Hold", "Avoid")
VALID_SENTIMENTS = ("Bullish", "Bearish", "Neutral")
VALID_HOLD_ACTIONS = ("Hold For More", "Take Profit")

# Lower value = more conservative -- the call _resolve_dual() picks when
# the two providers disagree. Extends evaluate_holding()'s own existing
# "default to caution unless the evidence is real and specific" philosophy
# to also cover cross-provider disagreement, not just a single model's
# uncertainty.
CONSERVATIVE_ORDER_DECISION = {"Avoid": 0, "Hold": 1, "Buy": 2}
CONSERVATIVE_ORDER_HOLD_ACTION = {"Take Profit": 0, "Hold For More": 1}


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

    news_sentiment = data.get("news_sentiment")
    if news_sentiment not in VALID_SENTIMENTS:
        return None

    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale.strip(),
        "news_sentiment": news_sentiment,
    }


def _parse_holding_response(text: str) -> dict | None:
    """Parse and validate a model's JSON response for evaluate_holding() --
    same validation shape as _parse_response(), just a different `action`
    vocabulary in place of `decision` (VALID_HOLD_ACTIONS instead of
    VALID_DECISIONS). Kept as a separate function rather than a parameter
    on _parse_response() since the two schemas are semantically distinct
    (a fresh-candidate judgment vs. a hold-past-target judgment), not
    interchangeable. Returns None (never raises) on any malformed shape."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    action = data.get("action")
    if action not in VALID_HOLD_ACTIONS:
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

    news_sentiment = data.get("news_sentiment")
    if news_sentiment not in VALID_SENTIMENTS:
        return None

    return {
        "action": action,
        "confidence": confidence,
        "rationale": rationale.strip(),
        "news_sentiment": news_sentiment,
    }


def _build_qualitative_block(ticker: str, qualitative: dict | None) -> str:
    """Shared by _build_prompt()/_build_holding_prompt() -- renders
    market_data.get_qualitative_snapshot()'s dict as one labeled paragraph
    per sub-section that's actually present, cleanly omitting any
    sub-section that failed to fetch (each degrades independently -- see
    get_qualitative_snapshot()'s own docstring) rather than showing a
    misleading "(unavailable)" for the whole block over one partial
    failure. Returns "" (not an error) if `qualitative` is falsy entirely,
    same "optional, cleanly omitted" convention as the macro block."""
    if not qualitative:
        return ""
    parts = []

    analyst = qualitative.get("analyst")
    if analyst:
        targets = analyst.get("targets") or {}
        targets_str = ", ".join(f"{k}={v}" for k, v in targets.items()) or "(unavailable)"
        trend = analyst.get("trend") or []
        trend_str = " -> ".join(str(t) for t in trend) or "(unavailable)"
        actions = analyst.get("recent_actions") or []
        actions_str = "; ".join(actions) if actions else "(none recent)"
        parts.append(
            f"Analyst consensus: price targets [{targets_str}], recommendation trend over the "
            f"last 4 months (oldest to newest) [{trend_str}], recent upgrades/downgrades: {actions_str}."
        )

    insider = qualitative.get("insider")
    if insider:
        parts.append(
            f"Insider activity: net recent direction={insider.get('net_direction')}, "
            f"institutional ownership={insider.get('pct_institutions')}, "
            f"insider ownership={insider.get('pct_insiders')}."
        )

    short_interest = qualitative.get("short_interest")
    if short_interest:
        parts.append(
            f"Short interest: {short_interest.get('pct_of_float')}% of float, "
            f"trend vs. prior month={short_interest.get('trend')}."
        )

    filings = qualitative.get("filings")
    if filings:
        filings_str = "; ".join(
            f"{f.get('type')} ({f.get('date')}){' [material]' if f.get('type') == '8-K' else ''}"
            for f in filings
        )
        parts.append(f"Recent SEC filings: {filings_str}.")

    options = qualitative.get("options")
    if options and options.get("put_call_volume_ratio") is not None:
        parts.append(f"Options positioning: nearest-expiry put/call volume ratio={options.get('put_call_volume_ratio')}.")

    if not parts:
        return ""
    return f"\n\nAdditional qualitative context for {ticker} (beyond news/fundamentals above):\n" + "\n".join(parts)


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

    # Optional -- see market_data.get_macro_snapshot(). One shared snapshot
    # covers every candidate ticker evaluated this run, not fetched here
    # per-ticker, so this paragraph is cleanly omitted (not an error) for
    # any caller that doesn't pass "macro".
    macro = context.get("macro")
    macro_block = ""
    if macro:
        macro_headlines = macro.get("headlines") or []
        trimmed_macro_headlines = macro_headlines[:MAX_MACRO_HEADLINES]
        macro_headline_block = (
            "\n".join(f"- {h}" for h in trimmed_macro_headlines) if trimmed_macro_headlines else "(none available)"
        )
        macro_block = (
            f"\n\nBroader market backdrop today (shared across every candidate, distinct from "
            f"{ticker}'s own news above): VIX={macro.get('vix')} "
            f"(change {macro.get('vix_change_pct')}%), S&P-level macro headlines:\n{macro_headline_block}"
        )

    qualitative_block = _build_qualitative_block(ticker, context.get("qualitative"))

    prompt = (
        f"A mechanical price/volume trading system flagged {ticker} today with the "
        f"following technical readings: Last_Close={context.get('last_close')}, "
        f"RSI={context.get('rsi')}, ATR={context.get('atr')}, "
        f"mechanical Trade_Score(s) by strategy: {mechanical_block}. "
        f"Catalyst_Warning={context.get('catalyst_warning')}, "
        f"next earnings date={context.get('next_earnings_date')}. "
        f"Basic fundamentals: {fundamentals_block}. "
        f"Its {len(trimmed_headlines)} most recent news headlines:\n{headline_block}"
        f"{macro_block}{qualitative_block}\n\n"
        "Given all of this, is this ticker worth a human's attention as a candidate "
        "swing trade right now? Respond ONLY with a JSON object matching exactly this "
        'shape: {"decision": "Buy" | "Hold" | "Avoid", "confidence": <integer 0-100>, '
        '"news_sentiment": "Bullish" | "Bearish" | "Neutral", "rationale": "<2-3 '
        'plain-language sentences explaining the call>"}. '
        '"Buy" means genuinely worth considering now, "Hold" means mixed/uncertain but '
        'worth tracking, "Avoid" means a real red flag outweighs the technical setup. '
        '"news_sentiment" is your read of the combined ticker-specific AND broader-market '
        "news tone, tracked separately from the decision itself. Base confidence on how "
        "strong the combined evidence is, not just the mechanical score alone."
    )
    return prompt, len(trimmed_headlines)


def _build_prompt_qualitative_weighted(ticker: str, context: dict) -> tuple[str, int]:
    """Challenger variant -- same context-rendering as _build_prompt(), only
    the instructional framing differs: explicitly tells the model the
    mechanical Trade_Score is a starting FILTER, not the primary evidence,
    and to weigh where the qualitative/analyst/insider/macro picture agrees
    or disagrees with it. Tests whether leaning harder into "more than just
    numbers" (the axis the user specifically said they value about this
    feature) produces better real outcomes than the balanced control. See
    PROMPT_VARIANTS."""
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

    macro = context.get("macro")
    macro_block = ""
    if macro:
        macro_headlines = (macro.get("headlines") or [])[:MAX_MACRO_HEADLINES]
        macro_headline_block = (
            "\n".join(f"- {h}" for h in macro_headlines) if macro_headlines else "(none available)"
        )
        macro_block = (
            f"\n\nBroader market backdrop today (shared across every candidate, distinct from "
            f"{ticker}'s own news above): VIX={macro.get('vix')} "
            f"(change {macro.get('vix_change_pct')}%), S&P-level macro headlines:\n{macro_headline_block}"
        )

    qualitative_block = _build_qualitative_block(ticker, context.get("qualitative"))

    prompt = (
        f"A mechanical price/volume trading system flagged {ticker} today: Last_Close="
        f"{context.get('last_close')}, RSI={context.get('rsi')}, ATR={context.get('atr')}, "
        f"mechanical Trade_Score(s) by strategy: {mechanical_block}. Catalyst_Warning="
        f"{context.get('catalyst_warning')}, next earnings date={context.get('next_earnings_date')}. "
        f"Basic fundamentals: {fundamentals_block}. Treat the mechanical Trade_Score as a starting "
        "FILTER only -- it already confirms a price/volume setup exists, so it should NOT drive your "
        "decision on its own. Your real job is to weigh whether the qualitative and macro evidence "
        "below CONFIRMS or CONTRADICTS that mechanical setup, and let that agreement or disagreement "
        f"be the deciding factor. Its {len(trimmed_headlines)} most recent news headlines:\n{headline_block}"
        f"{macro_block}{qualitative_block}\n\n"
        "Given all of this -- weighing the qualitative/macro picture as the primary evidence, the "
        "mechanical score as secondary confirmation only -- is this ticker worth a human's attention "
        "as a candidate swing trade right now? Respond ONLY with a JSON object matching exactly this "
        'shape: {"decision": "Buy" | "Hold" | "Avoid", "confidence": <integer 0-100>, '
        '"news_sentiment": "Bullish" | "Bearish" | "Neutral", "rationale": "<2-3 '
        'plain-language sentences explaining the call, naming which qualitative/macro factor(s) '
        'drove it>"}. "Buy" means the qualitative/macro picture genuinely supports the setup, "Hold" '
        'means mixed or the qualitative picture is silent/neutral, "Avoid" means the qualitative '
        'picture contradicts the mechanical setup.'
    )
    return prompt, len(trimmed_headlines)


def _build_prompt_evidence_strict(ticker: str, context: dict) -> tuple[str, int]:
    """Challenger variant -- same context-rendering as _build_prompt(), only
    the instructional framing differs: requires at least one specific,
    named data point (a headline, filing, analyst action, or insider trend)
    to justify "Buy", defaulting harder to "Hold"/"Avoid" on generic,
    vaguely-positive setups with no concrete evidence. Tests whether a
    stricter evidence bar than the balanced control's produces a better
    real win rate. See PROMPT_VARIANTS."""
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

    macro = context.get("macro")
    macro_block = ""
    if macro:
        macro_headlines = (macro.get("headlines") or [])[:MAX_MACRO_HEADLINES]
        macro_headline_block = (
            "\n".join(f"- {h}" for h in macro_headlines) if macro_headlines else "(none available)"
        )
        macro_block = (
            f"\n\nBroader market backdrop today (shared across every candidate, distinct from "
            f"{ticker}'s own news above): VIX={macro.get('vix')} "
            f"(change {macro.get('vix_change_pct')}%), S&P-level macro headlines:\n{macro_headline_block}"
        )

    qualitative_block = _build_qualitative_block(ticker, context.get("qualitative"))

    prompt = (
        f"A mechanical price/volume trading system flagged {ticker} today: Last_Close="
        f"{context.get('last_close')}, RSI={context.get('rsi')}, ATR={context.get('atr')}, "
        f"mechanical Trade_Score(s) by strategy: {mechanical_block}. Catalyst_Warning="
        f"{context.get('catalyst_warning')}, next earnings date={context.get('next_earnings_date')}. "
        f"Basic fundamentals: {fundamentals_block}. Its {len(trimmed_headlines)} most recent news "
        f"headlines:\n{headline_block}{macro_block}{qualitative_block}\n\n"
        "Given all of this, is this ticker worth a human's attention as a candidate swing trade right "
        'now? Apply a STRICT evidence bar: only respond "Buy" if you can cite at least one SPECIFIC, '
        "named data point from the material above (a specific headline, SEC filing, analyst upgrade/"
        "downgrade, or insider-buying trend) that concretely supports it -- generic optimism, a merely "
        'neutral news tone, or "no bad news" is NOT sufficient evidence for "Buy". Default to "Hold" '
        'when the setup is real but the evidence is vague or absent, and to "Avoid" when a real red '
        "flag is present. Respond ONLY with a JSON object matching exactly this shape: "
        '{"decision": "Buy" | "Hold" | "Avoid", "confidence": <integer 0-100>, "news_sentiment": '
        '"Bullish" | "Bearish" | "Neutral", "rationale": "<2-3 plain-language sentences, naming the '
        'specific evidence cited if \'Buy\'>"}.'
    )
    return prompt, len(trimmed_headlines)


# Maps a variant name -> its prompt-builder function. "balanced" is the
# original, unchanged control (keeps strategy="llm_agent" with no suffix so
# its already-accumulating trust-floor data isn't fragmented). The other two
# are challengers -- see each builder's own docstring for what they test.
# evaluate_ticker()'s `variant` param looks up into this dict; adding a new
# variant later means adding one function + one entry here, nothing else.
PROMPT_VARIANTS = {
    "balanced": _build_prompt,
    "qualitative_weighted": _build_prompt_qualitative_weighted,
    "evidence_strict": _build_prompt_evidence_strict,
}


def variant_strategy_name(variant: str) -> str:
    """Maps a PROMPT_VARIANTS key -> its Trade_Signals `strategy` field.
    "balanced" keeps the original strategy="llm_agent" (no suffix) so its
    already-accumulating Trade_Outcomes data isn't fragmented; challenger
    variants get their own suffixed strategy string so they settle and get
    graded completely independently (distinct (ticker, signal_date, strategy)
    unique key, see storage/signals.py). Single source of truth -- both
    ingest.py and dip_buy_analyzer.py call this rather than each keeping
    their own copy of the mapping."""
    return "llm_agent" if variant == "balanced" else f"llm_agent_{variant}"


def _build_holding_prompt(ticker: str, context: dict) -> str:
    """Build the prompt for evaluate_holding() -- a DIFFERENT question from
    _build_prompt()'s "is this worth a human's attention" framing. This
    position already hit its mechanical ATR target (see review_holding());
    the question here is specifically whether the evidence supports
    holding past that target for more profit, or taking the win now.
    Expected `context` keys: `avg_cost`, `last_close`, `sell_price` (the
    target that was just hit), `unrealized_pnl_pct`, `headlines`
    (list[str]), and optionally `macro`/`qualitative` -- same shapes as
    _build_prompt()'s context, reused directly."""
    headlines = context.get("headlines") or []
    trimmed_headlines = headlines[:MAX_HEADLINES]
    headline_block = "\n".join(f"- {h}" for h in trimmed_headlines) if trimmed_headlines else "(none available)"

    macro = context.get("macro")
    macro_block = ""
    if macro:
        macro_headlines = (macro.get("headlines") or [])[:MAX_MACRO_HEADLINES]
        macro_headline_block = (
            "\n".join(f"- {h}" for h in macro_headlines) if macro_headlines else "(none available)"
        )
        macro_block = (
            f"\n\nBroader market backdrop today (shared across every position reviewed, distinct "
            f"from {ticker}'s own news above): VIX={macro.get('vix')} "
            f"(change {macro.get('vix_change_pct')}%), S&P-level macro headlines:\n{macro_headline_block}"
        )

    qualitative_block = _build_qualitative_block(ticker, context.get("qualitative"))

    prompt = (
        f"A trading position in {ticker} was opened at avg_cost={context.get('avg_cost')} and has "
        f"just reached its predetermined mechanical profit target: Last_Close={context.get('last_close')} "
        f"has hit or exceeded Sell_Price={context.get('sell_price')} "
        f"(unrealized P&L: {context.get('unrealized_pnl_pct')}%). The mechanical system's default "
        f"action is to sell now and lock in the gain. Its {len(trimmed_headlines)} most recent news "
        f"headlines:\n{headline_block}"
        f"{macro_block}{qualitative_block}\n\n"
        "Given all of this, does the evidence support holding this position PAST its target for "
        "potentially more profit (genuine continuing strength/momentum, positive catalysts ahead), "
        "or does it make more sense to take the win now (nothing suggests the move continues, or a "
        "real risk factor argues for locking in the gain)? Respond ONLY with a JSON object matching "
        'exactly this shape: {"action": "Hold For More" | "Take Profit", "confidence": <integer '
        '0-100>, "news_sentiment": "Bullish" | "Bearish" | "Neutral", "rationale": "<2-3 '
        'plain-language sentences explaining the call>"}. This is a genuine second opinion on an '
        "ALREADY-PROFITABLE position, not a fresh buy/sell screen -- default to \"Take Profit\" "
        "unless the evidence for continuation is real and specific, not just generically positive."
    )
    return prompt


def _call_gemini(prompt: str, parse_fn=_parse_response) -> dict | None:
    """PRIMARY provider -- see module docstring for why. Returns None
    (never raises) on any failure so callers can fall through to Groq.
    `parse_fn` is pluggable (defaults to _parse_response, the
    evaluate_ticker() schema) so evaluate_holding() can reuse this exact
    call machinery with _parse_holding_response instead -- only the
    expected JSON shape differs, not how either provider is called."""
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
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return None
        return parse_fn(text)
    except Exception:
        return None


def _call_groq(prompt: str, parse_fn=_parse_response) -> dict | None:
    """FALLBACK provider -- only tried if Gemini is unavailable or its call
    fails/returns an unusable response. Uses the standard `openai` SDK
    pointed at Groq's OpenAI-compatible endpoint (see GROQ_BASE_URL) --
    Groq's own API is a drop-in match for the Chat Completions shape, no
    Groq-specific SDK needed. Returns None (never raises) on any failure.
    `parse_fn` pluggable, see _call_gemini()'s docstring."""
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
        return parse_fn(text)
    except Exception:
        return None


def _resolve_dual(
    gemini_result: dict | None, groq_result: dict | None, decision_key: str, conservative_order: dict,
) -> dict | None:
    """Combine both providers' results into one final result, extended with
    `provider_agreement`/`secondary_provider`/`secondary_decision`/
    `secondary_confidence` -- shared by evaluate_ticker()/evaluate_holding()
    since the combination logic is identical, only `decision_key`
    ("decision" vs. "action") and `conservative_order` (which value counts
    as "more cautious" for that schema) differ.

    - Only one provider has a result (the other unconfigured or failed):
      that result unchanged, `provider_agreement=None`,
      `secondary_provider=None` -- today's original single-provider
      behavior, preserved exactly for anyone still running with only one
      key configured.
    - Both agree: use it, `confidence` = average of the two (a genuine
      second confirmation earns a confidence bump/blend, not just a copy
      of one model's own number), `provider_agreement=True`.
    - Both disagree: use whichever call is MORE CONSERVATIVE per
      `conservative_order` (confidence = that provider's own, NOT
      averaged -- averaging a confident "Buy" with a confident "Avoid"
      would produce a meaningless middle number), `provider_agreement=False`,
      rationale explicitly notes both providers' reasoning so a human
      reviewing later can see exactly what was disputed.

    Every existing key (`decision`/`action`, `confidence`, `rationale`,
    `news_sentiment`) is preserved on the returned dict -- these four new
    keys are purely additive, so nothing that only reads the original
    schema breaks."""
    if gemini_result is None and groq_result is None:
        return None
    if gemini_result is None:
        return {**groq_result, "provider_agreement": None, "secondary_provider": None,
                "secondary_decision": None, "secondary_confidence": None}
    if groq_result is None:
        return {**gemini_result, "provider_agreement": None, "secondary_provider": None,
                "secondary_decision": None, "secondary_confidence": None}

    gemini_decision = gemini_result[decision_key]
    groq_decision = groq_result[decision_key]

    if gemini_decision == groq_decision:
        combined_confidence = round((gemini_result["confidence"] + groq_result["confidence"]) / 2, 1)
        return {
            **gemini_result, "confidence": combined_confidence, "provider_agreement": True,
            "secondary_provider": "groq", "secondary_decision": groq_decision,
            "secondary_confidence": groq_result["confidence"],
        }

    if conservative_order[gemini_decision] <= conservative_order[groq_decision]:
        primary, secondary, secondary_label = gemini_result, groq_result, "groq"
    else:
        primary, secondary, secondary_label = groq_result, gemini_result, "gemini"

    combined_rationale = (
        f"Providers disagreed -- defaulted to the more conservative call. "
        f"Gemini: {gemini_result['rationale']} | Groq: {groq_result['rationale']}"
    )
    return {
        **primary, "rationale": combined_rationale, "provider_agreement": False,
        "secondary_provider": secondary_label, "secondary_decision": secondary[decision_key],
        "secondary_confidence": secondary["confidence"],
    }


def evaluate_ticker(ticker: str, context: dict, variant: str = "balanced") -> dict | None:
    """Ask an LLM for a genuine Buy/Hold/Avoid judgment on `ticker`, given
    `context` -- a dict of whatever's already been computed this page
    load, expected keys: `last_close`, `rsi`, `atr`, `mechanical_scores`
    (dict of strategy label -> Trade_Score for whichever mechanical
    strategies flagged this ticker today), `catalyst_warning`,
    `next_earnings_date`, `headlines` (list[str]), `fundamentals`
    (dict, e.g. {"pe_ratio":..., "market_cap":..., "sector":...} from
    yfinance's Ticker.info -- any subset, missing keys are fine), and
    optionally `macro` (dict from market_data.get_macro_snapshot() --
    {"vix":..., "vix_change_pct":..., "headlines":[...]} -- the SAME
    snapshot shared across every candidate ticker evaluated this run, not
    fetched per-ticker; safely omitted entirely by any caller that doesn't
    have one), and optionally `qualitative` (dict from
    market_data.get_qualitative_snapshot(ticker) -- analyst/insider/
    short-interest/filings/options context, see that function's docstring
    for the expected shape; each sub-section is independently optional,
    safely omitted entirely by any caller that doesn't have one).

    Calls BOTH Gemini and Groq whenever both are configured/available (not
    fallback-only) and combines them via _resolve_dual() -- two independent
    models agreeing is a stronger decision than either alone, and their
    DISAGREEING is itself useful information, not noise to discard. Falls
    back to whichever single provider IS available if only one is
    configured, unchanged from the original single-provider behavior.

    Returns {"decision": "Buy"|"Hold"|"Avoid", "confidence": 0-100,
    "news_sentiment": "Bullish"|"Bearish"|"Neutral", "rationale": str,
    "provider_agreement": bool | None, "secondary_provider": str | None,
    "secondary_decision": str | None, "secondary_confidence": float | None}
    or None (never raises) if BOTH providers are unavailable/fail -- a
    flaky or unconfigured LLM call must never break a scan that would
    otherwise succeed, same philosophy as ai_context.summarize_ticker_context().
    See _resolve_dual()'s own docstring for exactly how the four new keys
    are derived.

    This is explicitly a SECOND OPINION, not a blind scan: the prompt
    frames the ticker as one a mechanical system already flagged, and asks
    the model to weigh in given that context plus recent headlines and
    basic fundamentals -- not to discover candidates on its own.

    `variant` selects which prompt-builder to use from PROMPT_VARIANTS
    (default "balanced", today's original prompt, unchanged). See that
    dict's own comment for what the other variants test -- this parameter
    exists so ingest.py's headless automation can run the same candidate
    through multiple prompt framings for prospective A/B comparison; the
    dashboard's live tab always uses the default and never passes this."""
    build_fn = PROMPT_VARIANTS.get(variant, _build_prompt)
    prompt, _ = build_fn(ticker, context)
    gemini_result = _call_gemini(prompt) if _gemini_available() else None
    groq_result = _call_groq(prompt) if _groq_available() else None
    return _resolve_dual(gemini_result, groq_result, "decision", CONSERVATIVE_ORDER_DECISION)


def evaluate_holding(ticker: str, context: dict) -> dict | None:
    """Ask an LLM whether an ALREADY-PROFITABLE position (one that just hit
    its mechanical ATR target -- see swingtrade.review_holding()) is worth
    holding PAST that target for more profit, or should be sold now as the
    mechanical system's default recommendation says. `context` expected
    keys: `avg_cost`, `last_close`, `sell_price`, `unrealized_pnl_pct`,
    `headlines` (list[str]), and optionally `macro`/`qualitative` (same
    shapes as evaluate_ticker()'s context -- see that function's
    docstring).

    Deliberately a SEPARATE function/schema from evaluate_ticker(), not a
    parameter on it -- "is this worth buying" and "should I keep holding
    a winner" are different questions with different default framings
    (evaluate_ticker has no inherent bias either way; this one explicitly
    defaults to "Take Profit" unless the evidence for continuation is
    real, per _build_holding_prompt()'s own prompt text) -- collapsing
    them into one schema would blur that intentional asymmetry.

    Same dual-provider behavior as evaluate_ticker() (both called whenever
    both are available, combined via _resolve_dual() using
    CONSERVATIVE_ORDER_HOLD_ACTION -- "Take Profit" wins a disagreement,
    consistent with this function's own single-model default-to-caution
    framing), same "never raises, None on total failure" contract. NEVER
    affects position sizing or any capital-allocation path -- purely
    informational, shown alongside (never replacing) the mechanical
    SELL (target hit) recommendation it was called for.

    Returns {"action": "Hold For More"|"Take Profit", "confidence": 0-100,
    "news_sentiment": "Bullish"|"Bearish"|"Neutral", "rationale": str,
    "provider_agreement": bool | None, "secondary_provider": str | None,
    "secondary_decision": str | None, "secondary_confidence": float | None}
    or None. See _resolve_dual()'s own docstring for the four new keys."""
    prompt = _build_holding_prompt(ticker, context)
    gemini_result = _call_gemini(prompt, parse_fn=_parse_holding_response) if _gemini_available() else None
    groq_result = _call_groq(prompt, parse_fn=_parse_holding_response) if _groq_available() else None
    return _resolve_dual(gemini_result, groq_result, "action", CONSERVATIVE_ORDER_HOLD_ACTION)
