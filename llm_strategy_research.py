"""
LLM-invented strategy research loop -- Phase 1 (improvements.txt item 93).

An LLM composes a structured JSON rule (which of this project's
already-computed technical indicators to condition on, which already-built
exit mechanic to use), NEVER generates or executes code. Every proposed
rule runs through the same generic, human-written backtest engine every
other strategy in this project uses (swingtrade.simulate_llm_strategy_signals()/
simulate_random_llm_strategy_entries(), swingtrade/levels.py's
evaluate_llm_rule_conditions()/rule_exit_to_config()).

This is a pure RESEARCH loop: it never writes to Trade_Signals, never
touches System_Config, never promotes anything. Every cycle's rule,
rationale, backtest result, and the LLM's own reflection notes get written
to Strategy_Research_Journal (storage/research_journal.py) -- a knowledge
base a human can review, and the ONLY path from here to a real capital
decision is a separate, deliberate, later human step, same as every other
strategy's own promotion discipline in this project.

Phase 1 scope (see improvements.txt item 93 for the full write-up):
propose -> validate -> small-sample backtest -> full-watchlist backtest
only if the sample looked genuinely promising -> write the cycle. NOT
built this pass: a scheduled daily cron, a dashboard tab, or any
promotion path. Callable manually:

    python llm_strategy_research.py --run-once
    python llm_strategy_research.py --run-once --sample-size 20
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr regardless of the invoking environment's default
# codepage -- Windows console/redirected-file default is cp1252, which
# cannot encode characters real LLM output routinely contains (e.g. U+2011
# NON-BREAKING HYPHEN). Same fix already applied to ingest.py (2026-08-26)
# for the identical crash on LLM rationale text; this script has its own
# separate entry point and never inherited it -- found 2026-08-29 running
# the first real cycle, which crashed printing a genuine LLM proposal's
# rationale before it ever reached Strategy_Research_Journal. errors="replace"
# rather than a stricter mode -- a mis-rendered character in a log line is
# cosmetic; crashing the whole cycle over one, discarding a real proposal,
# is not. Must happen before any print() call.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import llm_agent
import storage
import swingtrade
from run_backtest import MARKET_INDEX_TICKER, fetch_history
from watchlist import read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
BACKTEST_YEARS = 5

_ALLOWED_CONDITION_OPS = ("<", "<=", ">", ">=", "==")
_ALLOWED_EXIT_TYPES = ("atr_bracket", "trailing_stop", "time_based")
_ATR_MULTIPLIER_BOUNDS = (0.1, 8.0)
_MAX_HOLDING_DAYS_BOUNDS = (1, 90)
_CONDITION_VALUE_BOUNDS = (-1_000_000.0, 1_000_000.0)  # generic sanity net, not per-field domain bounds


def validate_rule(rule: dict) -> tuple[bool, str]:
    """Pure validation of one LLM-proposed rule, BEFORE it's ever trusted
    with a real backtest -- same "malformed input is never more
    trustworthy than no input" discipline llm_agent._parse_response()
    already applies elsewhere. Returns (True, "") if the rule is safe to
    run, (False, "<specific reason>") otherwise. Deliberately stricter
    than evaluate_llm_rule_conditions()'s own field/op checks (which only
    guard the interpreter itself) -- this ALSO clamps every numeric
    parameter to a sane bound, so a malformed or adversarial proposal
    can't produce a nonsensical backtest (e.g. a 500x ATR stop, a
    9000-day holding period) even if the field/op names were technically
    valid."""
    if not isinstance(rule, dict):
        return False, "rule is not a dict"

    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return False, "rule has no conditions"

    logic = rule.get("logic", "AND")
    if logic not in ("AND", "OR"):
        return False, f"unknown logic: {logic!r}"

    for cond in conditions:
        if not isinstance(cond, dict):
            return False, f"condition is not a dict: {cond!r}"
        field = cond.get("field")
        if field not in swingtrade.KNOWN_INDICATOR_FIELDS:
            return False, f"unknown field: {field!r}"
        op = cond.get("op")
        if op not in _ALLOWED_CONDITION_OPS:
            return False, f"unknown op: {op!r}"
        value = cond.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"condition value must be numeric, got {value!r}"
        if not (_CONDITION_VALUE_BOUNDS[0] <= value <= _CONDITION_VALUE_BOUNDS[1]):
            return False, f"condition value out of bounds: {value!r}"

    exit_spec = rule.get("exit")
    if not isinstance(exit_spec, dict):
        return False, "rule has no exit spec"
    exit_type = exit_spec.get("type")
    if exit_type not in _ALLOWED_EXIT_TYPES:
        return False, f"unknown exit type: {exit_type!r}"

    for param_name in ("take_profit_atr_multiplier", "stop_loss_atr_multiplier"):
        ok, reason = _check_bounded_param(exit_spec, param_name, _ATR_MULTIPLIER_BOUNDS)
        if not ok:
            return False, reason

    if exit_type == "trailing_stop":
        ok, reason = _check_bounded_param(exit_spec, "trailing_stop_atr_multiplier", _ATR_MULTIPLIER_BOUNDS)
        if not ok:
            return False, reason
    if exit_type == "time_based":
        ok, reason = _check_bounded_param(exit_spec, "max_holding_days", _MAX_HOLDING_DAYS_BOUNDS, is_int=True)
        if not ok:
            return False, reason

    return True, ""


def _check_bounded_param(exit_spec: dict, name: str, bounds: tuple, is_int: bool = False) -> tuple[bool, str]:
    value = exit_spec.get(name)
    expected_type = int if is_int else (int, float)
    if not isinstance(value, expected_type) or isinstance(value, bool):
        return False, f"exit.{name} must be {'an int' if is_int else 'numeric'}, got {value!r}"
    if not (bounds[0] <= value <= bounds[1]):
        return False, f"exit.{name} out of bounds ({bounds[0]}-{bounds[1]}): {value!r}"
    return True, ""


def _build_proposal_prompt(recent_cycles: list[dict]) -> str:
    """Builds the "here's what you've tried, propose today's rule" prompt
    from Strategy_Research_Journal history -- see storage/research_journal.py
    for the cycle document shape this reads."""
    if not recent_cycles:
        history_block = "(no prior cycles yet -- this is the very first research cycle.)"
    else:
        lines = []
        for c in recent_cycles:
            result = c.get("full_validation_result") or c.get("sample_result") or {}
            lines.append(
                f"- Cycle {c.get('cycle_id')} ({c.get('date')}): rule={c.get('rule')}, "
                f"escalated_to_full_validation={c.get('escalated_to_full_validation')}, "
                f"result={result}, your own notes at the time: {c.get('notes')!r}"
            )
        history_block = "\n".join(lines)

    fields_block = ", ".join(swingtrade.KNOWN_INDICATOR_FIELDS)

    return (
        "You are a systematic trading strategy researcher. You compose RULES, not code -- "
        "a JSON object choosing which already-computed technical indicators to condition on "
        "and which exit mechanic to use, interpreted by an existing, already-tested backtest "
        "engine. You never write or execute code, and you never see or influence real capital "
        "-- this is a pure research loop.\n\n"
        f"Available indicator fields (choose ONLY from this list): {fields_block}.\n"
        f"Available condition operators: {', '.join(_ALLOWED_CONDITION_OPS)}.\n"
        'Available exit types: "atr_bracket" (fixed take-profit/stop-loss, both required as '
        f"ATR multipliers between {_ATR_MULTIPLIER_BOUNDS[0]} and {_ATR_MULTIPLIER_BOUNDS[1]}), "
        '"trailing_stop" (same two params PLUS trailing_stop_atr_multiplier, same bounds), '
        '"time_based" (same two params PLUS max_holding_days, an integer between '
        f"{_MAX_HOLDING_DAYS_BOUNDS[0]} and {_MAX_HOLDING_DAYS_BOUNDS[1]}).\n\n"
        f"Your prior research cycles, most recent first:\n{history_block}\n\n"
        "Propose today's rule: either a genuine refinement of a promising recent idea "
        "(set parent_cycle_id to that cycle's own id), or something new if recent ideas "
        "plateaued or failed outright. Respond ONLY with a JSON object matching exactly this "
        'shape: {"rule": {"conditions": [{"field": ..., "op": ..., "value": ...}, ...], '
        '"logic": "AND" | "OR", "exit": {"type": ..., ...}}, '
        '"parent_cycle_id": <int or null>, '
        '"rationale": "<why you think this rule might find a real edge>", '
        '"notes_for_next_time": "<anything worth remembering when you next read this>"}.'
    )


def _parse_proposal_response(text: str) -> dict | None:
    """Parse and validate propose_rule()'s JSON response -- same
    conservative "malformed shape means None, never fabricate" philosophy
    as llm_agent._parse_response(). Runs the proposed rule through
    validate_rule() itself, so a syntactically-valid-JSON-but-semantically-
    unsafe rule (unknown field, out-of-bounds param) is rejected here too,
    not just deferred to the backtest layer."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    rule = data.get("rule")
    if not isinstance(rule, dict):
        return None
    is_valid, _ = validate_rule(rule)
    if not is_valid:
        return None

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None

    notes = data.get("notes_for_next_time")
    notes = notes.strip() if isinstance(notes, str) else ""

    parent_cycle_id = data.get("parent_cycle_id")
    if parent_cycle_id is not None and not isinstance(parent_cycle_id, int):
        parent_cycle_id = None

    return {"rule": rule, "rationale": rationale.strip(), "notes_for_next_time": notes, "parent_cycle_id": parent_cycle_id}


PROPOSAL_MAX_OUTPUT_TOKENS = 1200  # this schema (nested rule object + rationale
                                     # + notes_for_next_time) is genuinely larger
                                     # than every other llm_agent.py caller's own
                                     # short decision/confidence/rationale shape
                                     # -- llm_agent.MAX_OUTPUT_TOKENS (400) was
                                     # found silently truncating real responses
                                     # mid-JSON, 2026-08-22.


def propose_rule(recent_cycles: list[dict]) -> dict | None:
    """Asks an LLM to propose today's rule, informed by
    Strategy_Research_Journal's own recent history. Gemini first, then
    llm_agent.call_secondary()'s fallback pool (Groq, then OpenRouter -- both
    already degrade to None internally on any failure/unavailability, no
    need to pre-check availability here). Returns None if every provider
    is unavailable/fails, or the proposal was malformed -- callers must
    handle a None proposal, never assume one exists."""
    prompt = _build_proposal_prompt(recent_cycles)
    result = llm_agent.call_gemini(prompt, parse_fn=_parse_proposal_response, max_output_tokens=PROPOSAL_MAX_OUTPUT_TOKENS)
    if result is not None:
        return result
    result, _ = llm_agent.call_secondary(
        prompt, parse_fn=_parse_proposal_response, max_output_tokens=PROPOSAL_MAX_OUTPUT_TOKENS,
    )
    return result


def _fetch_and_backtest(tickers: list[str], rule: dict, config, log=print) -> dict:
    """Shared real-vs-random backtest loop for both the small-sample and
    full-watchlist passes -- fetches real OHLCV (no mocking), simulates
    the rule's own real signals plus a matched-count random baseline per
    ticker, pools and summarizes via the same cluster-weighted
    machinery benchmark_random_entry.py's own summarize() uses."""
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=365 * BACKTEST_YEARS)
    market_data = fetch_history(MARKET_INDEX_TICKER, start, end)

    real_trades, random_trades = [], []
    rng = random.Random(42)
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            ohlcv = fetch_history(ticker, start, end)
        except Exception as exc:
            log(f"  [WARN] {ticker}: {exc}")
            continue
        if ohlcv.empty:
            continue
        real = swingtrade.simulate_llm_strategy_signals(ticker, ohlcv, market_data, start, end, rule, config)
        real_trades.extend(real)
        random_trades.extend(
            swingtrade.simulate_random_llm_strategy_entries(
                ticker, ohlcv, market_data, start, end, len(real), rng, rule, config,
            )
        )

    def summarize(trades):
        resolved = [t for t in trades if t["status"] != "OPEN"]
        weights = swingtrade.compute_cluster_weights(resolved)
        return swingtrade.summarize_trades_weighted(resolved, weights)

    return {"real": summarize(real_trades), "random": summarize(random_trades), "n_tickers": len(tickers)}


def _looks_promising(result: dict) -> bool:
    """Same real-beats-random bar every other strategy's own validation
    uses (sharpe_like AND win_rate both better than the matched-count
    random baseline) -- the gate for whether a sample result escalates to
    a full-watchlist backtest. Deliberately conservative: `None` sharpe
    (too few resolved trades) never counts as promising."""
    real, rand = result["real"], result["random"]
    if real.get("sharpe_like") is None or rand.get("sharpe_like") is None:
        return False
    if real.get("win_rate") is None or rand.get("win_rate") is None:
        return False
    return real["sharpe_like"] > rand["sharpe_like"] and real["win_rate"] > rand["win_rate"]


def run_daily_cycle(
    tickers: list[str], sample_size: int = 40, config=None, propose_fn=propose_rule, log=print,
) -> dict:
    """Orchestrates one research cycle: read history -> propose -> validate
    -> small-sample backtest -> full-watchlist backtest only if promising
    -> write the cycle to Strategy_Research_Journal. Never writes to
    Trade_Signals, never touches System_Config, never promotes anything --
    a human reviewing the journal decides whether any full-validation-passed
    rule is worth building into a real strategy later.

    `propose_fn` defaults to the real propose_rule() (a real LLM call) but
    is overridable -- lets this whole orchestration be smoke-tested with a
    hand-built proposal when no LLM credentials are configured, without
    faking the rest of the pipeline (fetch/backtest/journal-write all stay
    real)."""
    config = config or swingtrade.DEFAULT_CONFIG
    recent_cycles = storage.get_recent_cycles(limit=10)

    proposal = propose_fn(recent_cycles)
    if proposal is None:
        cycle_id = storage.write_cycle({
            "date": str(pd.Timestamp.now().date()),
            "rule": None, "rationale": None, "parent_cycle_id": None,
            "sample_tickers": [], "sample_result": None,
            "escalated_to_full_validation": False, "full_validation_result": None,
            "notes": "No usable proposal this cycle -- LLM unavailable or returned a malformed rule.",
        })
        log(f"Cycle {cycle_id}: no usable proposal.")
        return {"cycle_id": cycle_id, "escalated": False}

    rule = proposal["rule"]
    sample = tickers[:sample_size]
    log(f"Proposal: {rule}")
    log(f"Rationale: {proposal['rationale']}")

    sample_result = _fetch_and_backtest(sample, rule, config, log=log)
    log(f"Sample result ({len(sample)} tickers): REAL={sample_result['real']}  RANDOM={sample_result['random']}")

    escalate = _looks_promising(sample_result)
    full_validation_result = None
    if escalate:
        log("Sample looked promising -- escalating to full-watchlist validation...")
        full_validation_result = _fetch_and_backtest(tickers, rule, config, log=log)
        log(f"Full validation ({len(tickers)} tickers): "
            f"REAL={full_validation_result['real']}  RANDOM={full_validation_result['random']}")

    cycle_id = storage.write_cycle({
        "date": str(pd.Timestamp.now().date()),
        "rule": rule,
        "rationale": proposal["rationale"],
        "parent_cycle_id": proposal["parent_cycle_id"],
        "sample_tickers": sample,
        "sample_result": sample_result,
        "escalated_to_full_validation": escalate,
        "full_validation_result": full_validation_result,
        "notes": proposal["notes_for_next_time"],
    })
    log(f"Cycle {cycle_id} written to Strategy_Research_Journal.")
    return {"cycle_id": cycle_id, "escalated": escalate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-once", action="store_true", help="Run exactly one research cycle.")
    parser.add_argument("--sample-size", type=int, default=40, help="Small-sample ticker count. Default: 40.")
    args = parser.parse_args()

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.run_once:
        parser.print_help()
        return

    if not WATCHLIST_FILE.exists():
        print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
        sys.exit(1)
    tickers = read_tickers(WATCHLIST_FILE)
    run_daily_cycle(tickers, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
