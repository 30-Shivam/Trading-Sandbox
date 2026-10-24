"""
Daily Briefing -- one consolidated synthesis of the day's state, closing
the gap where getting "what actually matters today, and why" required
manually stitching together Position Review and portfolio-health numbers through conversation each time (see
the 2026-08-26 HWM/TDY/HST walkthrough this was built in response to).

Runs as the 4th and final daily_run.py step, after ingest.py (fresh
signals), settle_trades.py (settlement), and review_positions.py (flip
detection/state) have all completed -- so it reports the day's FINAL
state, not a partial one.

Deliberately does NOT re-invent anything: reuses market_data.review_holdings()
for holdings status (same as review_positions.py/dip_buy_analyzer.py),
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

import config_loader
import ic_tracking
import market_data
import notifications
import storage

# Logged strategy names whose live IC feeds the portfolio-health line.
HEALTH_METHODOLOGIES = ("pairs", "value_rank", "accruals_rank")


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




_LOGGED_NAME_TO_LABEL = {logged: label for label, logged in config_loader.SECONDARY_LOG_STRATEGY_OVERRIDES.items()}


def build_portfolio_health_section(reports: dict[str, dict]) -> str:
    """`reports`: {strategy: ic_tracking.methodology_report() result}.
    Only reports methodologies that have cleared their own trust floor --
    below that, the IC number isn't trustworthy enough to be worth a
    daily line (see TRUST_FLOOR_TRADES). Returns "" if none have.

    2026-08-31: a methodology with an open entry in
    config_loader.SECONDARY_VALIDATION_CAVEATS (e.g. RSI Mean-Reversion,
    which failed a fresh random-entry benchmark the same real IC here
    doesn't yet reflect) gets a short inline flag pointing at the dashboard
    for the full caveat text -- this digest stays a quick daily read, not a
    second copy of the full warning banner render_secondary_section()
    already shows."""
    cleared = {name: r for name, r in reports.items() if r["trust_floor_met"] and r["overall_ic"] is not None}
    if not cleared:
        return ""
    lines = ["**Portfolio health** (trust-floor-cleared methodologies)"]
    for name, r in sorted(cleared.items(), key=lambda kv: -kv[1]["overall_ic"]):
        direction = "positive" if r["overall_ic"] > 0 else "negative"
        line = f"  {name}: IC {r['overall_ic']:+.3f} ({direction}), n={r['effective_n_settled']:.0f}"
        label = _LOGGED_NAME_TO_LABEL.get(name, name)
        if label in config_loader.SECONDARY_VALIDATION_CAVEATS:
            line += "  [!] backtest validation caveat -- see dashboard"
        lines.append(line)
    return "\n".join(lines)


def build_daily_briefing(
    position_rows: list[dict], health_reports: dict[str, dict], today: str | None = None,
) -> str:
    """Compose the sections into one message. `today` (optional,
    defaults to date.today()) is injectable for deterministic testing."""
    sections = [
        build_position_review_section(position_rows),
        build_portfolio_health_section(health_reports),
    ]
    sections = [s for s in sections if s]
    header = f"**Daily Briefing -- {today or date.today().isoformat()}**"
    if not sections:
        return f"{header}\nNothing to report today -- no holdings, no trust-floor-cleared methodologies."
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
    health_reports = {name: ic_tracking.methodology_report(name) for name in HEALTH_METHODOLOGIES}

    return build_daily_briefing(position_rows, health_reports, today=today)


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
