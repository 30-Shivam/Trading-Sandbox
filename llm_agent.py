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

Three free-tier providers -- PRIMARY + a two-deep SECONDARY fallback pool
-- picked by comparing actual published limits (see
github.com/open-free-llm-api/awesome-freellm-apis) rather than defaulting
to whichever was already wired up:
  - PRIMARY: Google Gemini (gemini-3.5-flash, via `google-genai`) -- a
    real, confirmed 20 requests/day/model (found 2026-08-22 via a live
    429 RESOURCE_EXHAUSTED response -- an earlier 1,500/day assumption in
    this docstring was WRONG). A subagent audit the same day found this
    project's EXISTING daily automation alone (ingest.py's LLM Agent
    candidates x prompt variants, plus audits, plus Best Ideas'
    evaluate_ticker()/evaluate_meta_synthesis()) already reaches 50-80
    real Gemini calls/day on a normal trading day -- comfortably past
    this real cap, independent of anything this docstring assumed.
  - SECONDARY: a two-provider fallback POOL, tried via call_secondary()
    only when Gemini itself is unavailable or fails -- Groq first
    (openai/gpt-oss-120b, via the `openai` SDK pointed at Groq's
    OpenAI-compatible endpoint), then OpenRouter only if Groq itself is
    unavailable/fails (via the same `openai` SDK pattern pointed at
    OpenRouter's own OpenAI-compatible endpoint, using one of its
    genuinely free ":free"-suffixed models -- see OPENROUTER_MODEL).
    Added 2026-08-22 specifically to address the real capacity finding
    above -- deliberately a FALLBACK POOL for the secondary slot, not a
    third simultaneous call on every request, since tripling baseline
    call volume would make the actual (capacity) problem worse, not
    better.
  - CORRECTION, same day: this docstring originally claimed OpenRouter's
    free tier "requires a paid top-up for sustained use" and rejected it
    in favor of a dedicated Mistral key -- the user found and shared
    OpenRouter's real published terms, which contradict that: free
    ":free"-suffixed models need NO credit card or top-up at all. Verified
    directly via OpenRouter's own public /api/v1/models endpoint (no auth
    needed) rather than trusting either claim on faith. The real
    constraint is narrower than "needs payment": a shared account-wide cap
    of 20 requests/minute and 50 requests/day across ALL free models
    combined if you've never added credit (rising to 1000/day if you ever
    add as little as $10, one-time, credits don't expire) -- and the free
    ":free" model roster itself rotates/gets retired often (confirmed:
    Meta's Llama 3.3 70B free variant was delisted since this was first
    written), the same "verify against the live catalog, don't trust a
    hardcoded model forever" lesson Groq's own retirement already taught
    this module once. OpenRouter replaced the dedicated-Mistral-key plan
    as the second-deep fallback for exactly this reason: same or better
    terms, one fewer API key for the user to go create.

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

from dotenv import load_dotenv

# Reads GEMINI_API_KEY/GROQ_API_KEY from the environment -- this module used
# to rely entirely on SOME OTHER already-imported module (storage/mongo.py,
# notifications.py) having called load_dotenv() first as a side effect,
# which only worked by import-order luck in every real caller (ingest.py
# imports notifications; dip_buy_analyzer.py/llm_strategy_research.py both
# import storage). Found 2026-08-22 debugging why a bare `import llm_agent`
# saw _gemini_available()/_groq_available() as False even with real,
# non-empty keys configured -- this module should never depend on a caller's
# own unrelated imports for its own credentials to load. Matches
# storage/mongo.py's own identical load_dotenv() call for the same reason.
load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash"  # gemini-2.5-flash was retired for this
                                    # account ("no longer available to new
                                    # users", a permanent 404, not
                                    # transient) -- see improvements.txt.
                                    # 3.5-flash is a "thinking" model by
                                    # default: without thinking_budget=0
                                    # below, it can spend its entire
                                    # max_output_tokens budget on internal
                                    # reasoning and return truncated JSON.
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired for
                                    # this account (404 "does not exist or you
                                    # do not have access to it") -- found
                                    # 2026-08-22 while verifying real API
                                    # credentials for llm_strategy_research.py
                                    # (improvements.txt item 93), the exact
                                    # same failure mode as GEMINI_MODEL's own
                                    # earlier retirement above. Confirmed
                                    # against this account's real available
                                    # model list (client.models.list()) and
                                    # live-verified with a real JSON-mode call
                                    # before switching -- the largest
                                    # general-purpose chat model currently
                                    # available, matching Groq's own role here
                                    # as the "genuine judgment call" fallback.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_MODEL = "dots-studio/dots-3-note-preview:free"  # added 2026-08-22
                                    # as a second-deep fallback behind Groq
                                    # (see call_secondary()) -- a real
                                    # capacity audit found existing daily
                                    # automation alone already exceeds
                                    # Gemini's real 20/day limit, so the
                                    # secondary slot needed its own
                                    # resilience, not a third simultaneous
                                    # call on every request. Chosen by
                                    # querying OpenRouter's own public
                                    # /api/v1/models endpoint directly
                                    # (no auth needed) for a genuinely free
                                    # (pricing.prompt/completion == "0"),
                                    # ":free"-suffixed, response_format-
                                    # capable, large-context model --
                                    # OpenRouter's free-model roster
                                    # rotates/retires often (confirmed:
                                    # Llama 3.3 70B's free variant was
                                    # delisted since this was first
                                    # researched), so re-verify against
                                    # that same endpoint if this one ever
                                    # starts silently 404ing, same lesson
                                    # GROQ_MODEL's own retirement taught.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
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


def _openrouter_available() -> bool:
    if not os.environ.get("OPENROUTER_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def is_available() -> bool:
    """True if a live call to evaluate_ticker() stands a chance of working
    -- at least ONE of Gemini (primary), Groq, or OpenRouter (the two-deep
    secondary fallback pool, see call_secondary()) is configured and its
    package installed. Callers should gate the LLM Agent tab on this
    rather than discovering unavailability via a failed call (mirrors
    ai_context.is_available())."""
    return _gemini_available() or _groq_available() or _openrouter_available()


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


def _parse_audit_response(text: str) -> dict | None:
    """Parse and validate audit_verdict()'s JSON response -- same
    conservative "malformed shape means None, never fabricate" philosophy
    as _parse_response()."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    audit_result = data.get("audit_result")
    if audit_result not in ("PASS", "FAIL"):
        return None

    audit_notes = data.get("audit_notes")
    if not isinstance(audit_notes, str) or not audit_notes.strip():
        return None

    return {"audit_result": audit_result, "audit_notes": audit_notes.strip()}


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


def _build_audit_prompt(ticker: str, context: dict, verdict: dict) -> str:
    """Build the adversarial-review prompt for audit_verdict() -- presents
    the SAME context evaluate_ticker() saw, plus the verdict it produced,
    and asks a FRESH pass to check whether the verdict's own rationale
    actually holds up against that data. Reuses the same context-rendering
    blocks _build_prompt() uses (headlines/mechanical scores/fundamentals/
    macro/qualitative) so the auditor sees exactly what the original call
    saw, not a summarized/lossy version of it -- deliberately duplicated
    rather than factored out, since _build_prompt()'s own closing
    instructions (fresh judgment) and this prompt's closing instructions
    (adversarial review of an existing judgment) are genuinely different
    tasks sharing only the context-rendering, same reasoning
    _parse_holding_response() gives for staying separate from
    _parse_response()."""
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
        macro_headlines = macro.get("headlines") or []
        trimmed_macro_headlines = macro_headlines[:MAX_MACRO_HEADLINES]
        macro_headline_block = (
            "\n".join(f"- {h}" for h in trimmed_macro_headlines) if trimmed_macro_headlines else "(none available)"
        )
        macro_block = (
            f"\n\nBroader market backdrop (shared across every candidate): VIX={macro.get('vix')} "
            f"(change {macro.get('vix_change_pct')}%), S&P-level macro headlines:\n{macro_headline_block}"
        )

    qualitative_block = _build_qualitative_block(ticker, context.get("qualitative"))

    return (
        f"A trading analyst was given the following data on {ticker} and asked for a "
        f"Buy/Hold/Avoid judgment: Last_Close={context.get('last_close')}, "
        f"RSI={context.get('rsi')}, ATR={context.get('atr')}, "
        f"mechanical Trade_Score(s) by strategy: {mechanical_block}. "
        f"Catalyst_Warning={context.get('catalyst_warning')}, "
        f"next earnings date={context.get('next_earnings_date')}. "
        f"Basic fundamentals: {fundamentals_block}. "
        f"Its {len(trimmed_headlines)} most recent news headlines:\n{headline_block}"
        f"{macro_block}{qualitative_block}\n\n"
        f'The analyst responded: decision="{verdict.get("decision")}", '
        f'confidence={verdict.get("confidence")}, rationale="{verdict.get("rationale")}".\n\n'
        "You are a SEPARATE reviewer. You did not make this call and have no loyalty to "
        "it -- assume it may be wrong and find out. Check, in order: does every specific "
        "claim in the rationale (a number, a named catalyst, a sentiment characterization) "
        "actually appear supported by the data given above, or is anything asserted that "
        "the data doesn't actually show? Is the decision consistent with the rationale's "
        'own tone (e.g. a bearish-sounding rationale paired with "Buy")? Is the stated '
        "confidence level plausible given how hedged or certain the rationale's own "
        "language is? You may NOT change the decision or rewrite the rationale -- report "
        "only. Respond ONLY with a JSON object matching exactly this shape: "
        '{"audit_result": "PASS" | "FAIL", "audit_notes": "<1-2 plain-language sentences -- '
        'if FAIL, cite the SPECIFIC unsupported claim or inconsistency>"}.'
    )


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


def _build_meta_synthesis_prompt(ticker: str, context: dict) -> str:
    """Prompt for evaluate_meta_synthesis() -- a DIFFERENT question from
    every other prompt-builder in this file: given MULTIPLE methodologies'
    own independent calls on the SAME ticker today (mechanical strategies,
    the regime switcher, a sector-relative-strength momentum score, a
    rule-based qualitative/fundamentals composite -- see best_ideas.py)
    AND each one's own real, measured Information Coefficient/Information
    Ratio track record (see ic_tracking.py), synthesize ONE final call and
    explain which methodology(ies) most drove it. The model is handed the
    actual IC/IR numbers and explicitly instructed to weigh a methodology
    with demonstrated positive, trust-floor-cleared ranking skill more
    than an unproven or historically poor one -- not asked to guess at
    reliability from vibes."""
    methodologies = context.get("methodologies") or []
    lines = []
    for m in methodologies:
        ic = m.get("overall_ic")
        ir = m.get("ir")
        if ic is not None:
            ir_str = f"{ir:.2f}" if ir is not None else "not enough windows yet"
            track_record = (
                f"overall IC={ic:.2f}, IR={ir_str} ({m.get('n_settled', 0)} settled trades, "
                f"{'trust floor met' if m.get('trust_floor_met') else 'still building trust floor'})"
            )
        else:
            track_record = f"not enough settled trades yet ({m.get('n_settled', 0)}) to measure a track record"
        score = m.get("score")
        score_str = f"{score:.1f}/100" if score is not None else "n/a"
        lines.append(f"- {m.get('name')}: call={m.get('signal')}, score={score_str}, {track_record}")
    methodology_block = "\n".join(lines) if lines else "(no methodology opinions available)"

    return (
        f"Multiple independent trading methodologies have each produced their OWN call on "
        f"{ticker} today (Last_Close={context.get('last_close')}):\n{methodology_block}\n\n"
        "Each methodology's 'IC' (Information Coefficient) is the REAL, measured rank "
        "correlation between that methodology's own past scores and what actually happened "
        "to price afterward -- a positive IC means it has genuinely ranked winners above "
        "losers historically; near-zero or negative means it hasn't (yet, or ever). 'IR' "
        "measures how STABLE that skill has been over time, not just whether it showed up "
        "once. Weigh methodologies with a real, trust-floor-cleared, positive IC/IR more "
        "heavily than ones that are unproven or have shown poor ranking skill -- do not just "
        "average every opinion equally. Synthesize ONE final judgment: given everything above, "
        "is this ticker worth a human's attention as a candidate swing trade right now? "
        'Respond ONLY with a JSON object matching exactly this shape: {"decision": "Buy" | '
        '"Hold" | "Avoid", "confidence": <integer 0-100>, "news_sentiment": "Bullish" | '
        '"Bearish" | "Neutral", "rationale": "<2-3 plain-language sentences explaining the '
        "call, explicitly naming which methodology(ies) most drove it and why -- cite their "
        'track record if it influenced the weighting>"}.'
    )


def call_gemini(prompt: str, parse_fn=_parse_response, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict | None:
    """PRIMARY provider -- see module docstring for why. Returns None
    (never raises) on any failure so callers can fall through to Groq.
    `parse_fn` is pluggable (defaults to _parse_response, the
    evaluate_ticker() schema) so evaluate_holding() can reuse this exact
    call machinery with _parse_holding_response instead -- only the
    expected JSON shape differs, not how either provider is called.
    `max_output_tokens` defaults to MAX_OUTPUT_TOKENS (right-sized for the
    short decision/confidence/rationale schema every existing caller uses)
    -- override it for a caller whose own expected JSON shape is
    genuinely larger (e.g. llm_strategy_research.propose_rule()'s nested
    rule object), found 2026-08-22 when the default budget was silently
    truncating that schema's response mid-JSON."""
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
                max_output_tokens=max_output_tokens,
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


def call_groq(prompt: str, parse_fn=_parse_response, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict | None:
    """FALLBACK provider -- only tried if Gemini is unavailable or its call
    fails/returns an unusable response. Uses the standard `openai` SDK
    pointed at Groq's OpenAI-compatible endpoint (see GROQ_BASE_URL) --
    Groq's own API is a drop-in match for the Chat Completions shape, no
    Groq-specific SDK needed. Returns None (never raises) on any failure.
    `parse_fn`/`max_output_tokens` pluggable, see call_gemini()'s
    docstring."""
    try:
        import openai
    except ImportError:
        return None
    try:
        client = openai.OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        return parse_fn(text)
    except Exception:
        return None


def call_openrouter(prompt: str, parse_fn=_parse_response, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict | None:
    """Second-deep SECONDARY fallback -- only ever tried by call_secondary()
    when Groq itself is unavailable or fails. Uses the standard `openai`
    SDK pointed at OpenRouter's own OpenAI-compatible endpoint (see
    OPENROUTER_BASE_URL), same drop-in pattern as call_groq(). Returns None
    (never raises) on any failure -- including OPENROUTER_MODEL having been
    retired/delisted from the free roster, which OpenRouter's free tier
    does periodically (see OPENROUTER_MODEL's own comment). `parse_fn`/
    `max_output_tokens` pluggable, see call_gemini()'s docstring.

    `extra_body={"reasoning": {"effort": "none"}}` -- OPENROUTER_MODEL is a
    "thinking" model by default (same failure mode GEMINI_MODEL hit
    earlier, see its own comment): without this, a real live-verified call
    returned finish_reason="length" with content=None, having spent its
    entire max_output_tokens budget on the `reasoning` field instead of
    the actual JSON answer (confirmed via the raw response object -- the
    correct answer was fully present in `reasoning`, just never emitted to
    `content` before the token budget ran out). Verified live: with this
    parameter, the same prompt returns finish_reason="stop", valid JSON
    content, reasoning_tokens=0."""
    try:
        import openai
    except ImportError:
        return None
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"effort": "none"}},
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        return parse_fn(text)
    except Exception:
        return None


def call_secondary(
    prompt: str, parse_fn=_parse_response, max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> tuple[dict | None, str | None]:
    """The SECONDARY slot's fallback pool -- tries Groq first, OpenRouter
    only if Groq is unavailable or its call fails/returns an unusable
    response. Added 2026-08-22 after a real capacity audit found this
    project's existing daily automation alone already exceeds Gemini's
    real 20/day limit -- this makes the secondary slot resilient to ONE
    provider having a bad day without tripling baseline call volume
    (which would make the actual capacity problem worse, not better;
    Gemini's own primary role and fallback-through-here behavior are
    unchanged).

    Returns (result, provider_name) -- provider_name is "groq" or
    "openrouter", whichever actually answered, so callers/_resolve_dual()
    can honestly report which one it was instead of assuming "groq".
    Returns (None, None) if both are unavailable/fail, same "never
    raises" contract as every provider call in this module."""
    if _groq_available():
        result = call_groq(prompt, parse_fn=parse_fn, max_output_tokens=max_output_tokens)
        if result is not None:
            return result, "groq"
    if _openrouter_available():
        result = call_openrouter(prompt, parse_fn=parse_fn, max_output_tokens=max_output_tokens)
        if result is not None:
            return result, "openrouter"
    return None, None


def _resolve_dual(
    gemini_result: dict | None,
    secondary_result: dict | None,
    secondary_provider_name: str | None,
    decision_key: str,
    conservative_order: dict,
) -> dict | None:
    """Combine both providers' results into one final result, extended with
    `provider_agreement`/`secondary_provider`/`secondary_decision`/
    `secondary_confidence` -- shared by evaluate_ticker()/evaluate_holding()
    since the combination logic is identical, only `decision_key`
    ("decision" vs. "action") and `conservative_order` (which value counts
    as "more cautious" for that schema) differ.

    `secondary_provider_name` is the REAL name ("groq" or "openrouter") of
    whichever provider actually produced `secondary_result` -- see
    call_secondary(), added 2026-08-22 when the secondary slot became a
    two-provider fallback pool instead of always being Groq. Ignored if
    `secondary_result` is None (nothing to attribute a name to).

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
    if gemini_result is None and secondary_result is None:
        return None
    if gemini_result is None:
        return {**secondary_result, "provider_agreement": None, "secondary_provider": None,
                "secondary_decision": None, "secondary_confidence": None}
    if secondary_result is None:
        return {**gemini_result, "provider_agreement": None, "secondary_provider": None,
                "secondary_decision": None, "secondary_confidence": None}

    gemini_decision = gemini_result[decision_key]
    secondary_decision = secondary_result[decision_key]

    if gemini_decision == secondary_decision:
        combined_confidence = round((gemini_result["confidence"] + secondary_result["confidence"]) / 2, 1)
        return {
            **gemini_result, "confidence": combined_confidence, "provider_agreement": True,
            "secondary_provider": secondary_provider_name, "secondary_decision": secondary_decision,
            "secondary_confidence": secondary_result["confidence"],
        }

    if conservative_order[gemini_decision] <= conservative_order[secondary_decision]:
        primary, secondary, secondary_label = gemini_result, secondary_result, secondary_provider_name
    else:
        primary, secondary, secondary_label = secondary_result, gemini_result, "gemini"

    secondary_display_name = {"groq": "Groq", "openrouter": "OpenRouter"}.get(secondary_provider_name, "Secondary")
    combined_rationale = (
        f"Providers disagreed -- defaulted to the more conservative call. "
        f"Gemini: {gemini_result['rationale']} | {secondary_display_name}: {secondary_result['rationale']}"
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
    gemini_result = call_gemini(prompt) if _gemini_available() else None
    secondary_result, secondary_provider_name = call_secondary(prompt)
    return _resolve_dual(
        gemini_result, secondary_result, secondary_provider_name, "decision", CONSERVATIVE_ORDER_DECISION,
    )


def audit_verdict(ticker: str, context: dict, verdict: dict) -> dict | None:
    """Adversarial SECOND PASS over an already-produced evaluate_ticker()
    verdict -- a genuinely different check from _resolve_dual()'s own
    dual-provider consensus (which asks two models the SAME fresh question
    in PARALLEL and reconciles disagreement). This instead takes the
    WINNING verdict as given and asks a fresh, SEQUENTIAL call to review
    whether its own stated rationale actually holds up against the source
    data -- "assume it's wrong, find out how." Never rewrites the verdict,
    only reports -- same "checker never fixes" discipline as this
    project's own champion/challenger promotion flow (promote_config.py
    never lets Optuna's own search results self-promote; settle_trades.py
    grades every signal regardless of whether it was ever acted on).

    Only worth calling for verdict["decision"] == "Buy" (the only ones a
    human might actually act on) -- callers gate this themselves (see
    ingest.py/dip_buy_analyzer.py), this function doesn't check
    verdict["decision"] internally so it stays a pure, reusable "audit
    whatever verdict you give me" primitive.

    Single-provider (Gemini if available, else the call_secondary()
    fallback pool -- Groq, then OpenRouter) -- NOT dual, unlike
    evaluate_ticker(). This is a bounded-cost sanity pass on top of an
    already-dual-provider-reconciled verdict, not the primary judgment
    itself; doubling every call site's volume isn't worth it for a
    checking pass.

    Returns {"audit_result": "PASS"|"FAIL", "audit_notes": str} or None
    (never raises) if every provider is unavailable/fails -- same "a
    flaky LLM call must never break a scan" philosophy as every other
    function here."""
    prompt = _build_audit_prompt(ticker, context, verdict)
    if _gemini_available():
        result = call_gemini(prompt, parse_fn=_parse_audit_response)
        if result is not None:
            return result
    result, _ = call_secondary(prompt, parse_fn=_parse_audit_response)
    return result


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
    gemini_result = call_gemini(prompt, parse_fn=_parse_holding_response) if _gemini_available() else None
    secondary_result, secondary_provider_name = call_secondary(prompt, parse_fn=_parse_holding_response)
    return _resolve_dual(
        gemini_result, secondary_result, secondary_provider_name, "action", CONSERVATIVE_ORDER_HOLD_ACTION,
    )


def evaluate_meta_synthesis(ticker: str, context: dict) -> dict | None:
    """Ask an LLM to synthesize ONE final call across MULTIPLE already-
    scored methodologies (see best_ideas.py), weighted by each
    methodology's own REAL, measured IC/IR track record (see
    ic_tracking.py) rather than by generic prose reasoning about which
    signal "sounds" more trustworthy. This is the genuine meta-reasoning
    layer of the "Best Ideas" dashboard tab -- distinct from
    evaluate_ticker() (a single second opinion on ONE mechanical setup) and
    from a plain numeric IC/IR-weighted blend (best_ideas.blend_composite()
    computes that independently and does NOT depend on this function --
    the composite score stays fully auditable/deterministic even if this
    call fails or is unavailable).

    `context` expected keys: `last_close`, `methodologies` (list of
    {"name":, "signal":, "score":, "overall_ic":, "ir":, "n_settled":,
    "trust_floor_met":} -- see ic_tracking.methodology_report() for where
    the IC/IR fields come from).

    Same dual-provider behavior as every other evaluate_*() function here
    (both Gemini/Groq called whenever available, combined via
    _resolve_dual() using CONSERVATIVE_ORDER_DECISION), same "never
    raises, None on total failure" contract, and reuses _parse_response
    unchanged -- this prompt asks for the exact same JSON shape
    evaluate_ticker() does, just synthesized across methodologies instead
    of from one setup's raw technicals/news.

    Its own decision/confidence get logged to MongoDB under
    strategy="best_ideas_meta" (see best_ideas.py) so THIS layer's own
    ranking skill is independently IC/IR-tracked over time, same as every
    other methodology feeding the ensemble -- literally answering, with
    real settled-trade data rather than assumption, whether this
    meta-reasoning step is worth its own added LLM cost/complexity.

    Returns the same shape as evaluate_ticker() -- {"decision":,
    "confidence":, "news_sentiment":, "rationale":, "provider_agreement":,
    "secondary_provider":, "secondary_decision":, "secondary_confidence":}
    -- or None."""
    prompt = _build_meta_synthesis_prompt(ticker, context)
    gemini_result = call_gemini(prompt) if _gemini_available() else None
    secondary_result, secondary_provider_name = call_secondary(prompt)
    return _resolve_dual(
        gemini_result, secondary_result, secondary_provider_name, "decision", CONSERVATIVE_ORDER_DECISION,
    )
