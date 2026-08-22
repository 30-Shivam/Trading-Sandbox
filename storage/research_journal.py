"""Strategy_Research_Journal persistence -- the LLM-invented-strategy
research loop's own knowledge base (improvements.txt item 93). One
document per daily research cycle: the proposed rule, why it was
proposed, what happened when it was backtested, and the LLM's own notes
for next time. See llm_strategy_research.py for how this gets
written/read.

NEVER read by allocate_capital()/anything capital-adjacent -- this is a
pure research log, same "isolated, non-capital-eligible" status
regime_switcher.py/llm_agent.py/every Best Ideas methodology already has.
Confirmed with the user: whether to ever act on anything found here stays
an explicit, separate human decision.

Document shape:
    {
        "cycle_id": int,          # sequential, assigned by write_cycle()
        "date": str,               # ISO date this cycle ran
        "rule": dict,              # the JSON DSL, see
                                    # swingtrade.evaluate_llm_rule_conditions()
        "rationale": str,          # the LLM's own reasoning for proposing this rule
        "parent_cycle_id": int | None,  # links a refinement back to the
                                    # idea it refined -- makes "recursive"
                                    # a real traceable chain, not just a
                                    # fresh idea every day
        "sample_tickers": [str],   # which tickers the small-sample backtest used
        "sample_result": dict,     # {"real": {...}, "random": {...}} --
                                    # summarize_trades()-style dicts
        "escalated_to_full_validation": bool,
        "full_validation_result": dict | None,  # the full 407-ticker
                                    # ALL/TUNE/HOLDOUT/BY-YEAR/Monte-Carlo
                                    # report, only present if escalated
        "notes": str,              # the LLM's own reflection for tomorrow
        "created_at": datetime,
    }
"""
from datetime import datetime, timezone

from .mongo import get_db

COLLECTION_NAME = "Strategy_Research_Journal"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("cycle_id", 1)], unique=True)
    db[COLLECTION_NAME].create_index([("date", -1)])


def write_cycle(doc: dict) -> int:
    """Assigns the next sequential cycle_id and inserts `doc` (already
    carrying every other field the module docstring describes, except
    cycle_id/created_at, which this function sets). Returns the assigned
    cycle_id. Append-only by design -- a research cycle's own record is
    never edited after the fact, same "never delete history" discipline
    the source idea's own state.md concept called for."""
    db = get_db()
    last = db[COLLECTION_NAME].find_one(sort=[("cycle_id", -1)])
    next_id = (last["cycle_id"] + 1) if last else 1
    full_doc = {**doc, "cycle_id": next_id, "created_at": datetime.now(timezone.utc)}
    db[COLLECTION_NAME].insert_one(full_doc)
    return next_id


def get_recent_cycles(limit: int = 10) -> list[dict]:
    """Most recent `limit` cycles, most-recent-first -- what
    llm_strategy_research.propose_rule() reads to build its own "here's
    what you've tried, here's what happened" context for the next
    proposal."""
    db = get_db()
    return list(db[COLLECTION_NAME].find().sort("cycle_id", -1).limit(limit))
