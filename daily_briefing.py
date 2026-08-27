"""
Daily Briefing -- one consolidated synthesis of the day's state, closing
the gap where getting "what actually matters today, and why" required
manually stitching together Position Review, LLM Agent candidates/audit
results, and portfolio-health numbers through conversation each time (see
the 2026-08-26 HWM/TDY/HST walkthrough this was built in response to).

Runs as the 4th and final daily_run.py step, after ingest.py (fresh
signals), settle_trades.py (settlement), and review_positions.py (flip
detection/state) have all completed -- so it reports the day's FINAL
state, not a partial one.

Deliberately does NOT re-invent anything: reuses market_data.review_holdings()
for holdings status (same as review_positions.py/dip_buy_analyzer.py),
queries today's already-logged llm_agent-family Trade_Signals for
candidates + audit results (the same fields ingest.py's run_llm_agent()
already writes -- audit_result, and confidence stored under the shared
trade_score field, see that function's own docstring), and
ic_tracking.methodology_report() for portfolio health (same call every
other trust-floor check in this project makes). Purely a SYNTHESIS/
presentation layer on top of already-existing data -- no new scoring,
scanning, or LLM calls of its own, and never capital-allocated.

Unlike review_positions.py's alert (which deliberately only fires on a
FLIP, to avoid repeat-spamming the same SELL state), this digest shows
full current status every time it runs -- it's a once-daily consolidated
view, not a repeated push alert, so restating "still SELL" alongside
whatever's new is the right behavior here, not spam.

Usage:
    python daily_briefing.py
"""

import sys
from datetime import date

import best_ideas
import config_loader
import ic_tracking
import market_data
import notifications
import storage

LLM_AGENT_STRATEGIES = ("llm_agent", "llm_agent_qualitative_weighted", "llm_agent_evidence_strict")


def build_position_review_section(rows: list[dict]) -> str:
    """`rows`: swingtrade.review_holding()-shaped dicts (Ticker,
    Recommendation, Avg_Cost, Last_Close, Unrealized_PnL_Pct, ...) --
    same shape market_data.review_holdings() returns. Sorted worst PnL
    first (same convention as review_positions.py's own notification).
    Returns "" if there's nothing to review -- caller omits the section."""
    if not rows:
        return ""
    lines = ["**Position Review**"]
    for r in sorted(rows, key=lambda r: r["Unrealized_PnL_Pct"]):
        flag = " (!)" if r["Recommendation"].startswith("SELL") else ""
        lines.append(
            f"  {r['Ticker']}: {r['Recommendation']}{flag} "
            f"({r['Unrealized_PnL_Pct']:+.2f}%, avg_cost {r['Avg_Cost']:.2f}, last {r['Last_Close']:.2f})"
        )
    return "\n".join(lines)


def build_candidates_section(candidates_by_ticker: dict[str, list[dict]]) -> str:
    """`candidates_by_ticker`: {ticker: [{"strategy":, "audit_result":,
    "confidence":}, ...]} -- one entry per llm_agent-family prompt variant
    that fired Buy for that ticker today (see LLM_AGENT_STRATEGIES).
    Tickers with at least one audit PASS lead (the one thing worth seeing
    first -- see the 2026-08-26 walkthrough where confidence alone didn't
    distinguish a clean call from 5 that failed audit at the same
    confidence level). Returns "" if nothing fired Buy today.

    audit_result is None (not "FAIL") whenever the audit call itself never
    returned a verdict (llm_agent.audit_verdict()'s own "never raises,
    None on failure" contract -- a real API/parse failure, not a
    content critique) -- distinguished from a real FAIL rather than
    silently counted as one, confirmed live 2026-08-26 when URI's one
    variant had 0 pass/0 fail purely because its audit call itself
    produced no verdict, not because it failed on the merits."""
    if not candidates_by_ticker:
        return ""

    def pass_count(ticker):
        return sum(1 for c in candidates_by_ticker[ticker] if c["audit_result"] == "PASS")

    def fail_count(ticker):
        return sum(1 for c in candidates_by_ticker[ticker] if c["audit_result"] == "FAIL")

    tickers = sorted(candidates_by_ticker.keys(), key=lambda t: (-pass_count(t), t))
    lines = ["**Today's LLM Agent Buy candidates**"]
    for ticker in tickers:
        variants = candidates_by_ticker[ticker]
        p, f = pass_count(ticker), fail_count(ticker)
        if p and not f:
            tag = "CLEAN"
        elif p:
            tag = "MIXED"
        elif f:
            tag = "ALL FAILED AUDIT"
        else:
            tag = "AUDIT UNAVAILABLE"
        lines.append(f"  {ticker}: {tag} ({p} pass / {f} fail of {len(variants)} variant(s))")
    return "\n".join(lines)


def build_portfolio_health_section(reports: dict[str, dict]) -> str:
    """`reports`: {strategy: ic_tracking.methodology_report() result}.
    Only reports methodologies that have cleared their own trust floor --
    below that, the IC number isn't trustworthy enough to be worth a
    daily line (see TRUST_FLOOR_TRADES). Returns "" if none have."""
    cleared = {name: r for name, r in reports.items() if r["trust_floor_met"] and r["overall_ic"] is not None}
    if not cleared:
        return ""
    lines = ["**Portfolio health** (trust-floor-cleared methodologies)"]
    for name, r in sorted(cleared.items(), key=lambda kv: -kv[1]["overall_ic"]):
        direction = "positive" if r["overall_ic"] > 0 else "negative"
        lines.append(f"  {name}: IC {r['overall_ic']:+.3f} ({direction}), n={r['effective_n_settled']:.0f}")
    return "\n".join(lines)


def build_daily_briefing(
    position_rows: list[dict], candidates_by_ticker: dict[str, list[dict]], health_reports: dict[str, dict],
    today: str | None = None,
) -> str:
    """Compose the three sections into one message. `today` (optional,
    defaults to date.today()) is injectable for deterministic testing."""
    sections = [
        build_position_review_section(position_rows),
        build_candidates_section(candidates_by_ticker),
        build_portfolio_health_section(health_reports),
    ]
    sections = [s for s in sections if s]
    header = f"**Daily Briefing -- {today or date.today().isoformat()}**"
    if not sections:
        return f"{header}\nNothing to report today -- no holdings, no Buy candidates, no trust-floor-cleared methodologies."
    return header + "\n\n" + "\n\n".join(sections)


def gather() -> str:
    """Real I/O: pulls today's actual state and builds the briefing.
    Shared by both main() below and the dashboard's own thin "Daily
    Briefing" tab (dip_buy_analyzer.py) -- one source of truth for what
    the synthesis actually says, not two independently-drifting copies."""
    holdings = storage.get_holdings()
    reviewable = {t: info["avg_cost"] for t, info in holdings.items() if info.get("avg_cost")}
    position_rows: list[dict] = []
    if reviewable:
        config, _ = config_loader.load_active_config()
        position_rows, _ = market_data.review_holdings(reviewable, config)

    today = date.today().isoformat()
    docs = list(storage.get_db()["Trade_Signals"].find(
        {"signal_date": today, "strategy": {"$in": list(LLM_AGENT_STRATEGIES)}, "signal": "Buy"}
    ))
    candidates_by_ticker: dict[str, list[dict]] = {}
    for d in docs:
        candidates_by_ticker.setdefault(d["ticker"], []).append({
            "strategy": d["strategy"], "audit_result": d.get("audit_result"), "confidence": d.get("trade_score"),
        })

    health_reports = {
        name: ic_tracking.methodology_report(name) for name in best_ideas.METHODOLOGIES + ["best_ideas"]
    }

    return build_daily_briefing(position_rows, candidates_by_ticker, health_reports, today=today)


def main() -> int:
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    message = gather()
    print(message)
    notifications.notify(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
