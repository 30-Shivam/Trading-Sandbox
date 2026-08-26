"""
Position Review flip-to-SELL alerting -- closes the last piece of
improvements.txt item 2's "Alerting" ask (the other two, a new Strong Buy
signal and settlement outcomes, were already built -- see ingest.py and
settle_trades.py respectively).

The dashboard's interactive-only "Position Review" section
(market_data.review_holdings() -> swingtrade.review_holding() per held
ticker) has never had a scheduled equivalent: a position quietly crossing
its stop or target is invisible unless a human happens to have the
dashboard open that day. This is the headless counterpart, meant to run
daily alongside ingest.py/settle_trades.py (see daily_run.py), against
storage.get_holdings() -- the same real, persisted Current_Holdings a
human maintains via the dashboard sidebar (deliberately NOT inferred from
Trade_Signals; a logged signal doesn't mean you actually got filled -- see
storage/holdings.py's own docstring).

Only holdings with a known avg_cost are reviewable -- matches the
dashboard's own gating (see dip_buy_analyzer.py's parse_holdings_text
docstring: an amount-only holding still counts toward the sector cap but
has no cost basis to check a stop/target against).

Alerts ONLY on a FLIP into a SELL recommendation (HOLD -> SELL, or a
changed SELL reason e.g. "target hit" -> "stop breached"), not every day a
position remains in the same SELL state -- Stop_Loss/Sell_Price are
anchored to avg_cost but sized off TODAY's rolling ATR (see
swingtrade.review_holding), so they move daily and a genuine SELL -> HOLD
flip-back is possible, not just one-directional. Prior state persisted via
storage/position_review_state.py, deliberately a separate collection from
Current_Holdings itself (which is wholesale-replaced on every user edit).

Each flip's alert is enriched with an LLM second opinion when
llm_agent.is_available() -- llm_agent.evaluate_holding() for a target hit
("worth holding past target for more?") or evaluate_stop_breach() for a
stop breach ("worth holding through for a genuine recovery, or cut the
loss?"), the same two functions/schemas dip_buy_analyzer.py's own sidebar
calls for the identical mechanical flags (2026-08-25 -- previously this
opinion only existed inside the interactive dashboard, so the actual
proactive alert carried just the bare mechanical flag with none of the
"is holding longer worth it" nuance a human would want at the moment the
alert fires, not only when they happen to open the dashboard). Still
purely informational -- never overrides the mechanical recommendation or
this alert firing; an LLM failure/unavailability just omits that one
line, it never blocks the flip alert itself.

Usage:
    python review_positions.py
"""

import sys

import config_loader
import llm_agent
import market_data
import notifications
import storage

DEFAULT_RECOMMENDATION = "HOLD"  # a ticker never previously reviewed is treated as if
                                  # it were HOLD, so a newly-added position that's
                                  # already broken still triggers a flip alert


def _detect_flips(results: list[dict], last_recommendations: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    """Pure comparison step, split out from main() for direct testing (no
    Mongo/network involved) -- mirrors settle_trades.settle_one() vs.
    _build_settlement_notification()'s own split between "do the work" and
    "decide what to report." Returns (flips, new_state): `flips` is the
    subset of `results` whose Recommendation is a SELL that differs from
    what was last observed for that ticker (or from DEFAULT_RECOMMENDATION
    if never observed before); `new_state` is {ticker: recommendation} for
    every row in `results`, meant to be persisted via
    storage.set_position_review_state() regardless of whether it flipped."""
    flips = []
    new_state = {}
    for row in results:
        ticker = row["Ticker"]
        recommendation = row["Recommendation"]
        new_state[ticker] = recommendation
        previous = last_recommendations.get(ticker, DEFAULT_RECOMMENDATION)
        if recommendation.startswith("SELL") and recommendation != previous:
            flips.append(row)
    return flips, new_state


def _llm_hold_opinion(row: dict, macro_snapshot: dict) -> dict | None:
    """Get the appropriate LLM second opinion for one newly-flipped-to-SELL
    row -- llm_agent.evaluate_holding() for a target hit or
    evaluate_stop_breach() for a stop breach, matching exactly which
    function dip_buy_analyzer.py's own dashboard calls for the same
    Recommendation. Both have a documented "never raises, None on total
    failure" contract, so no try/except needed here -- this is purely
    additive enrichment and must never be able to block the flip alert
    itself from firing."""
    ticker = row["Ticker"]
    context = {
        "avg_cost": row["Avg_Cost"],
        "last_close": row["Last_Close"],
        "unrealized_pnl_pct": row["Unrealized_PnL_Pct"],
        "headlines": market_data.get_multi_headlines(ticker),
        "macro": macro_snapshot,
        "qualitative": market_data.get_qualitative_snapshot(ticker),
    }
    if row["Recommendation"] == "SELL (target hit)":
        return llm_agent.evaluate_holding(ticker, {**context, "sell_price": row["Sell_Price"]})
    if row["Recommendation"] == "SELL (stop breached)":
        return llm_agent.evaluate_stop_breach(ticker, {**context, "stop_loss": row["Stop_Loss"]})
    return None


def _build_flip_notification(flips: list[dict], llm_opinions: dict[str, dict] | None = None) -> str | None:
    """Short Discord message for this run's newly-flipped-to-SELL
    positions, sorted by Unrealized_PnL_Pct ascending (worst first -- the
    most urgent one leads). Mirrors settle_trades.py's own
    _build_settlement_notification() in shape/tone. Returns None if
    `flips` is empty -- caller skips notifying entirely. `llm_opinions`
    (optional, {ticker: evaluate_holding()/evaluate_stop_breach() result})
    adds one indented line per ticker that has one -- see
    _llm_hold_opinion()."""
    if not flips:
        return None
    llm_opinions = llm_opinions or {}
    flips = sorted(flips, key=lambda f: f["Unrealized_PnL_Pct"])
    lines = [f"**Position Review: {len(flips)} holding(s) flipped to SELL**"]
    for f in flips:
        lines.append(
            f"{f['Ticker']}: {f['Recommendation']} (avg_cost {f['Avg_Cost']:.2f}, "
            f"last {f['Last_Close']:.2f}, {f['Unrealized_PnL_Pct']:+.2f}%)"
        )
        opinion = llm_opinions.get(f["Ticker"])
        if opinion:
            lines.append(
                f"    LLM second opinion: **{opinion['action']}** "
                f"(confidence {opinion['confidence']:.0f}/100) -- {opinion['rationale']}"
            )
    return "\n".join(lines)


def main() -> int:
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    holdings = storage.get_holdings()
    reviewable = {t: info["avg_cost"] for t, info in holdings.items() if info.get("avg_cost")}
    if not reviewable:
        print("No holdings with a known avg_cost to review.")
        storage.prune_position_review_state(set())
        return 0

    config, config_source = config_loader.load_active_config()
    print(f"Reviewing {len(reviewable)} holding(s) against {config_source} (strategy={config.strategy}).")

    results, skipped = market_data.review_holdings(reviewable, config)
    for ticker, reason in skipped:
        print(f"  {ticker}: SKIPPED ({reason})")

    last_recommendations = storage.get_position_review_state()
    flips, new_state = _detect_flips(results, last_recommendations)
    flipped_tickers = {f["Ticker"] for f in flips}
    for row in results:
        ticker = row["Ticker"]
        previous = last_recommendations.get(ticker, DEFAULT_RECOMMENDATION)
        tag = " <- FLIP" if ticker in flipped_tickers else ""
        print(f"  {ticker}: {row['Recommendation']} (was: {previous}){tag}")

    storage.set_position_review_state(new_state)
    storage.prune_position_review_state(set(reviewable))

    print()
    print(f"{len(flips)} flip(s) to SELL this run.")

    llm_opinions = {}
    if flips and llm_agent.is_available():
        macro_snapshot = market_data.get_macro_snapshot()
        for row in flips:
            opinion = _llm_hold_opinion(row, macro_snapshot)
            if opinion:
                llm_opinions[row["Ticker"]] = opinion
                print(f"    {row['Ticker']} LLM: {opinion['action']} (confidence {opinion['confidence']:.0f})")

    message = _build_flip_notification(flips, llm_opinions)
    if message:
        notifications.notify(message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
