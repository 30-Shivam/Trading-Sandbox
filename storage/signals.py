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
        "config_snapshot": dict,     # swingtrade.TradingConfig.to_dict() at trigger time
        "settled": bool,             # flipped by the Phase 3 settlement job
        "confirmed_filled": bool,    # absent/False until confirm_fill() is called --
                                      # NOT written by log_trade_signal, see below
        "fill_price": float,         # present only if confirm_fill() was given one
        "confirmed_at": datetime,    # present only once confirmed
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
        "rsi": _native(row["RSI"]),
        "atr": _native(row["ATR"]),
        "distance_to_buy_pct": _native(row["Distance_to_Buy_Pct"]),
        "shares_to_buy": _native(row["Shares_To_Buy"]),
        "est_cost": _native(row["Est_Cost"]),
        "next_earnings_date": str(next_earnings) if next_earnings is not None else None,
        "catalyst_warning": bool(_native(row["Catalyst_Warning"])),
        "oversold_streak_days": _native(row.get("Oversold_Streak_Days")),
        "extended_decline_warning": bool(_native(row.get("Extended_Decline_Warning", False))),
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
