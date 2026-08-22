"""Trade_Signals persistence.

Document shape (one per ticker per trading day):
    {
        "ticker": str,
        "signal_date": str,          # ISO date, matches compute_levels' As_Of
        "strategy": str,             # "rsi" | "breakout" | "pullback" |
                                      # "breakout_retest" | "week52_high" --
                                      # from config_snapshot["strategy"].
                                      # Part of the unique key alongside
                                      # ticker/signal_date: once more than
                                      # one strategy can be live at once
                                      # (see dip_buy_analyzer.py's secondary
                                      # scan sections), the SAME ticker can
                                      # legitimately fire under two
                                      # different strategies on the same
                                      # day, and those must be two separate
                                      # documents, not one overwriting the
                                      # other. A missing strategy field
                                      # (pre-existing documents written
                                      # before this field existed) means
                                      # "rsi" -- every signal logged before
                                      # this was one, and was backfilled
                                      # explicitly (see improvements.txt).
        "signal": "Strong Buy" | "Buy" | "Watch",
        "tier": "actionable" | "research" | "research_loosened",
                                      # "actionable" = Strong Buy/Buy (real
                                      # capital-allocation-eligible);
                                      # "research" = Watch, logged/settled
                                      # purely to accumulate live outcome
                                      # data faster than actionable-only
                                      # signals allow, WITHOUT touching real
                                      # capital -- see improvements.txt.
                                      # "research_loosened" = would only
                                      # score Strong Buy/Buy/Watch under
                                      # swingtrade.loosened_breakout_config()
                                      # (the six extra breakout filters
                                      # disabled) -- the active config
                                      # scored it Ignore. Same purpose as
                                      # "research" (accumulate real outcome
                                      # data when the active config is too
                                      # selective to fire), never traded,
                                      # never capital-allocation-eligible;
                                      # kept as its own tier (not folded into
                                      # "research") so its outcomes are never
                                      # pooled with real active-config
                                      # research-tier performance -- they
                                      # reflect a DIFFERENT, deliberately
                                      # loosened config, not the real one.
                                      # A missing tier field (pre-existing
                                      # documents written before this field
                                      # existed) means "actionable" -- every
                                      # signal logged before this was one.
        "trade_score": float,
        "buy_price": float, "sell_price": float, "stop_loss": float,
        "rrr": float, "rsi": float, "atr": float,
        "distance_to_buy_pct": float,
        "shares_to_buy": float, "est_cost": float,
        "next_earnings_date": str | None,
        "catalyst_warning": bool,
        "currency": str,        # "USD" | "CAD" -- see market_data.get_ticker_currency().
                                 # This system does NOT do FX conversion: buy_price/sell_price/
                                 # stop_loss/est_cost above are always in THIS currency, not
                                 # necessarily USD. Defaults to "USD" for documents logged
                                 # before this field existed.
        "provider_agreement": bool | None,  # llm_agent.py rows only -- True if
                                      # Gemini/Groq agreed, False if they
                                      # disagreed (the more conservative call
                                      # won), None if only one provider was
                                      # available. Always None for every
                                      # mechanical strategy's rows.
        "secondary_provider": str | None,  # "gemini" | "groq" | None -- whichever
                                      # provider's call did NOT become the
                                      # final decision, llm_agent.py rows only.
        "secondary_decision": str | None,  # that provider's own decision/action
        "secondary_confidence": float | None,  # that provider's own confidence
        "audit_result": str | None,  # "PASS" | "FAIL" | None -- llm_agent.py's
                                      # audit_verdict(), rows whose decision was
                                      # "Buy" only (see that function's own docstring
                                      # for why). A SEPARATE, sequential adversarial
                                      # review of this row's own rationale, distinct
                                      # from provider_agreement's parallel dual-model
                                      # consensus above.
        "audit_notes": str | None,   # the auditor's own 1-2 sentence explanation,
                                      # citing the specific unsupported claim if FAIL.
        "regime": str | None,        # "trending" | "choppy" | None -- regime_switcher.py
                                      # rows only (strategy="regime_switcher"), the
                                      # ADX-based classification that picked this row's
                                      # source strategy. See regime_switcher.py --
                                      # deliberately NOT backtested (structurally can't
                                      # be, prospective-only by design), permanently
                                      # non-capital-allocated until it separately clears
                                      # its own real settled-trade trust floor.
        "source_strategy": str | None,  # "breakout" | "squeeze_breakout" | "ma_crossover" |
                                      # None -- regime_switcher.py rows only, which
                                      # underlying mechanical strategy's signal this
                                      # regime pick actually came from.
        "config_snapshot": dict,     # swingtrade.TradingConfig.to_dict() at trigger time
        "settled": bool,             # flipped by the Phase 3 settlement job
        "confirmed_filled": bool,    # absent/False until confirm_fill() is called --
                                      # NOT written by log_trade_signal, see below
        "fill_price": float,         # present only if confirm_fill() was given one
        "confirmed_at": datetime,    # present only once confirmed
        "user_decision": str,        # "acted_on" | "passed" | absent -- record_user_decision(),
                                      # a preference/journal concept, deliberately SEPARATE
                                      # from confirmed_filled's own financial-truth concept.
                                      # NOT written by log_trade_signal, see below.
        "user_decision_reason": str, # optional free text, mainly meaningful for "passed" --
                                      # see user_preferences.summarize_decisions()
        "user_decision_at": datetime,  # present only once a decision is recorded
        "created_at": datetime, "updated_at": datetime,
    }

Signals are logged for the *pre-allocation* Signal (Strong Buy / Buy / Watch
as scored by swingtrade.add_trade_score/add_breakout_trade_score), not the
capital-allocator's overlay -- "Insufficient Funds" reflects a personal cash
constraint on a given day, not a change in the underlying technical signal,
and the learning loop (Phase 5) needs to judge the signal itself independent
of that. allocate_capital() only ever acts on Strong Buy/Buy rows (skips
everything else including Watch unconditionally), so logging the research
tier here can never leak into a real capital-allocation decision.

Research-tier (Watch) logging exists specifically because live outcome-data
volume was bottlenecked -- not by trading capacity (settle_trades.py already
grades every logged signal against real subsequent price action whether or
not you traded it, confirmed_filled is purely a reporting split, see
settle_trades.py), but by how selective the active config is: only Strong
Buy/Buy ever got logged, and a selective config like v19 fires rarely. Watch
(one tier below actionable) is a natural next slice: still meaningfully
scored, far more frequent, and directly useful for checking whether the
Trade_Score/Signal thresholds themselves are well-calibrated (do outcomes
actually improve as score rises through Watch -> Buy -> Strong Buy, the way
the score implies they should).

confirmed_filled deliberately does NOT appear in _build_document's $set:
log_trade_signal() re-runs every time the scanner re-scans (same ticker,
same day), and if confirmed_filled were part of that $set, a confirmation
you made this morning would get silently wiped out the next time ingest.py
or the dashboard re-scans today. It's only ever touched by confirm_fill()/
unconfirm_fill() below, so a scan can never undo a confirmation.

user_decision/user_decision_reason/user_decision_at follow the identical
never-in-$set discipline, for the identical reason -- see
record_user_decision() below.
"""

import math
from datetime import datetime, timezone

import pandas as pd

from .mongo import get_db

COLLECTION_NAME = "Trade_Signals"
ACTIONABLE_SIGNALS = ("Strong Buy", "Buy")
RESEARCH_SIGNALS = ("Watch",)
LOGGABLE_SIGNALS = ACTIONABLE_SIGNALS + RESEARCH_SIGNALS
LOOSENED_RESEARCH_TIER = "research_loosened"
# Matches pre-existing documents (written before the tier field existed) as
# "actionable" too -- see the module docstring.
ACTIONABLE_TIER_FILTER = {"$or": [{"tier": "actionable"}, {"tier": {"$exists": False}}]}


def _tier_for_signal(signal: str) -> str:
    return "actionable" if signal in ACTIONABLE_SIGNALS else "research"


def ensure_indexes() -> None:
    db = get_db()
    db[COLLECTION_NAME].create_index([("ticker", 1), ("signal_date", 1), ("strategy", 1)], unique=True)


def _native(value):
    """Coerce a pandas/numpy scalar to a plain, BSON-encodable Python value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (float64, bool_, int64, ...)
        return value.item()
    return value


def _build_document(row: dict, config_snapshot: dict, now: datetime, tier: str | None = None) -> dict:
    next_earnings = _native(row.get("Next_Earnings_Date"))
    return {
        "ticker": _native(row["Ticker"]),
        "signal_date": str(row["As_Of"]),
        "strategy": config_snapshot.get("strategy", "rsi"),
        "signal": _native(row["Signal"]),
        "tier": tier if tier is not None else _tier_for_signal(row["Signal"]),
        "trade_score": _native(row["Trade_Score"]),
        "buy_price": _native(row["Buy_Price"]),
        "sell_price": _native(row["Sell_Price"]),
        "stop_loss": _native(row["Stop_Loss"]),
        "rrr": _native(row["RRR"]),
        # RSI is present in every OTHER strategy's row dict (informational
        # even where not used for gating), but "pairs" (improvements.txt
        # item 82) has no RSI concept at all and never carries this key --
        # first strategy in this project's history to omit it, so this must
        # degrade gracefully like every other strategy-specific optional
        # field below rather than assume the key always exists.
        "rsi": _native(row.get("RSI")),
        "atr": _native(row["ATR"]),
        "distance_to_buy_pct": _native(row["Distance_to_Buy_Pct"]),
        "shares_to_buy": _native(row["Shares_To_Buy"]),
        "est_cost": _native(row["Est_Cost"]),
        "next_earnings_date": str(next_earnings) if next_earnings is not None else None,
        "catalyst_warning": bool(_native(row["Catalyst_Warning"])),
        "currency": row.get("Currency", "USD"),  # "USD"/"CAD", see market_data.get_ticker_currency() --
                                                  # defaults to "USD" for rows logged before this field
                                                  # existed (this project's entire pre-Canadian-ticker
                                                  # history), not just missing/None
        "oversold_streak_days": _native(row.get("Oversold_Streak_Days")),
        "extended_decline_warning": bool(_native(row.get("Extended_Decline_Warning", False))),
        # Dual-provider LLM cross-validation (llm_agent.py's _resolve_dual())
        # -- present only for strategy="llm_agent" rows, None for every
        # other strategy's rows (these keys simply aren't in `row` for
        # mechanical signals, and .get() returns None cleanly). Kept here
        # so a future analysis of settled llm_agent trades can check
        # whether provider agreement correlated with better outcomes.
        "provider_agreement": _native(row.get("Provider_Agreement")),
        "secondary_provider": _native(row.get("Secondary_Provider")),
        "secondary_decision": _native(row.get("Secondary_Decision")),
        "secondary_confidence": _native(row.get("Secondary_Confidence")),
        # Adversarial second-pass review (llm_agent.py's audit_verdict())
        # -- present only for strategy="llm_agent" rows whose decision was
        # "Buy" (the only ones a human might act on; see audit_verdict()'s
        # own docstring for why it isn't run on every verdict). None for
        # every other row, same "these keys simply aren't in `row`"
        # degrade-cleanly convention as provider_agreement above.
        "audit_result": _native(row.get("Audit_Result")),
        "audit_notes": row.get("Audit_Notes"),
        # regime_switcher.py rows only (strategy="regime_switcher") -- which
        # ADX-based regime was classified and which underlying mechanical
        # strategy's signal got surfaced as a result, None for every other
        # strategy's rows. Persisted (not just printed) specifically so a
        # future analysis of settled regime_switcher trades can check
        # whether trending-regime picks and choppy-regime picks perform
        # differently -- the actual hypothesis this feature exists to test.
        "regime": _native(row.get("Regime")),
        "source_strategy": _native(row.get("Source_Strategy")),
        # "pairs" rows only (strategy="pairs", improvements.txt item 82) --
        # which same-sector peer triggered the signal and the real
        # correlation/z-score behind it, None for every other strategy's
        # rows. Persisted for later audit/display, same "don't discard the
        # strategy's own distinguishing detail" treatment regime/
        # source_strategy above already get.
        "pair_partner": _native(row.get("Pair_Partner")),
        "pair_correlation": _native(row.get("Pair_Correlation")),
        "pair_spread_zscore": _native(row.get("Pair_Spread_Zscore")),
        # best_ideas.py rows only (strategy="best_ideas") -- the
        # meta-synthesis LLM's own written rationale (str|None) and a
        # snapshot of {methodology: {"score":, "weight":}} used to build
        # that row's composite Trade_Score, for later audit/display. Not
        # run through _native() -- already plain Python str/float/dict by
        # the time best_ideas.py builds these rows (no numpy/pandas
        # scalars to unwrap), and _native()'s pd.isna() check is only
        # meant for scalars, not nested dicts.
        "rationale": row.get("Rationale"),
        "methodology_breakdown": row.get("Methodology_Breakdown"),
        "config_snapshot": config_snapshot,
        "settled": False,
        "updated_at": now,
    }


def log_trade_signal(row: dict, config_snapshot: dict, tier: str | None = None) -> None:
    """Upsert one Trade_Signals document, keyed on (ticker, signal_date,
    strategy). Re-running the same scan the same day updates the existing
    document in place instead of creating a duplicate -- and, critically,
    a DIFFERENT strategy firing on the same ticker/day gets its OWN
    document instead of overwriting this one (see the module docstring's
    `strategy` field note). `tier`, if given, overrides the normal
    actionable/research split derived from `row["Signal"]` -- used for
    LOOSENED_RESEARCH_TIER, where the Signal value itself came from a
    loosened config and shouldn't be mistaken for a real actionable one."""
    db = get_db()
    now = datetime.now(timezone.utc)
    doc = _build_document(row, config_snapshot, now, tier=tier)

    db[COLLECTION_NAME].update_one(
        {"ticker": doc["ticker"], "signal_date": doc["signal_date"], "strategy": doc["strategy"]},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


def log_trade_signals(df: pd.DataFrame, config_snapshot: dict, tier: str | None = None) -> dict[str, int]:
    """Log every Strong Buy / Buy / Watch row in df (expects pre-allocation
    Signal values) -- Strong Buy/Buy tagged tier="actionable", Watch tagged
    tier="research" (see module docstring for why). Returns
    {"actionable": n, "research": m}.

    Pass `tier` to force every logged row to that tier instead (e.g.
    LOOSENED_RESEARCH_TIER) -- returns {tier: n} instead. Used for
    loosened-config rows, whose Signal values (Strong Buy/Buy/Watch) would
    otherwise be mistaken for real actionable/research-tier ones."""
    eligible = df[df["Signal"].isin(LOGGABLE_SIGNALS)]
    for _, row in eligible.iterrows():
        log_trade_signal(row.to_dict(), config_snapshot, tier=tier)
    if tier is not None:
        return {tier: len(eligible)}
    actionable_count = int(eligible["Signal"].isin(ACTIONABLE_SIGNALS).sum())
    return {"actionable": actionable_count, "research": len(eligible) - actionable_count}


def get_unsettled_signals() -> list[dict]:
    """Return every Trade_Signals document not yet resolved to a terminal
    outcome. The settlement job re-walks each of these from scratch every
    run -- cheap, and avoids needing to track incremental per-trade state."""
    db = get_db()
    return list(db[COLLECTION_NAME].find({"settled": {"$ne": True}}))


def mark_settled(ticker: str, signal_date: str, strategy: str) -> None:
    db = get_db()
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date, "strategy": strategy},
        {"$set": {"settled": True}},
    )


def confirm_fill(ticker: str, signal_date: str, strategy: str, fill_price: float | None = None) -> None:
    """Mark a logged signal as an actual, confirmed fill -- distinct from
    merely being logged, since most mechanical signals are never actually
    traded. `fill_price` (optional) records what you actually paid if it
    differed from the system's computed buy_price; settle_trades.py uses it
    as the real entry price for pnl_pct when set. Stop_Loss/Sell_Price are
    never touched -- they're absolute levels computed at signal time, not
    relative to whatever price you actually got filled at. `strategy` is
    required (not optional) because a ticker/date pair can now have more
    than one candidate document -- see the module docstring."""
    db = get_db()
    update = {"confirmed_filled": True, "confirmed_at": datetime.now(timezone.utc)}
    if fill_price is not None:
        update["fill_price"] = float(fill_price)
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date, "strategy": strategy},
        {"$set": update},
    )


def unconfirm_fill(ticker: str, signal_date: str, strategy: str) -> None:
    """Undo a mistaken confirm_fill call."""
    db = get_db()
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date, "strategy": strategy},
        {"$set": {"confirmed_filled": False}, "$unset": {"fill_price": "", "confirmed_at": ""}},
    )


def record_user_decision(
    ticker: str, signal_date: str, strategy: str, decision: str, reason: str | None = None,
) -> None:
    """Record what the user actually DID with a logged signal -- a
    preference/journal concept, deliberately SEPARATE from confirm_fill()'s
    own financial-truth concept (whether a real fill happened, at what
    price). A user can choose to act on a signal whose limit order never
    actually touched (confirmed_filled stays False), and "passed" needs to
    exist independent of any fill concept at all -- so this is its own
    field, not folded into confirmed_filled.

    `decision` must be "acted_on" or "passed". `reason` (optional, mainly
    meaningful for "passed") is free text -- see user_preferences.py's
    summarize_decisions() for how this gets read back. Same "not part of
    _build_document()'s $set" discipline confirmed_filled already
    established (see that field's own comment above) -- a routine re-scan
    of the same ticker/day must never silently wipe a recorded decision,
    so this is only ever touched by this function (and its CLI, see
    confirm_fill.py's --pass mode)."""
    if decision not in ("acted_on", "passed"):
        raise ValueError(f"decision must be 'acted_on' or 'passed', got {decision!r}")
    db = get_db()
    update = {"user_decision": decision, "user_decision_at": datetime.now(timezone.utc)}
    if reason is not None:
        update["user_decision_reason"] = reason
    db[COLLECTION_NAME].update_one(
        {"ticker": ticker, "signal_date": signal_date, "strategy": strategy},
        {"$set": update},
    )


def get_signals_pending_confirmation() -> list[dict]:
    """Actionable (Strong Buy/Buy) signals not yet marked confirmed_filled,
    most recent first -- for a human to review and confirm or ignore.
    Deliberately excludes the research and research_loosened tiers -- those
    were never meant to be traded/confirmed, only tracked for outcome data.
    Filters on tier (not just `signal`) because a research_loosened row can
    also carry signal="Strong Buy"/"Buy" (that's the loosened config's own
    label for it) without being a real, capital-eligible signal."""
    db = get_db()
    return list(
        db[COLLECTION_NAME]
        .find({**ACTIONABLE_TIER_FILTER, "confirmed_filled": {"$ne": True}})
        .sort("signal_date", -1)
    )
