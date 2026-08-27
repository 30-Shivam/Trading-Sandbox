"""daily_briefing.build_position_review_section()/build_candidates_section()/
build_portfolio_health_section()/build_daily_briefing() -- pure functions,
dict fixtures, no Mongo/network. gather() itself is a thin real-I/O
wrapper (same "not directly tested" convention as every other gather-style
function in this codebase, e.g. tests/test_ic_tracking.py's own docstring).
"""
import daily_briefing as db


def _row(ticker, recommendation="HOLD", avg_cost=100.0, last_close=100.0, pnl_pct=0.0):
    return {
        "Ticker": ticker, "Recommendation": recommendation,
        "Avg_Cost": avg_cost, "Last_Close": last_close, "Unrealized_PnL_Pct": pnl_pct,
    }


def _candidate(strategy, audit_result, confidence=78.0):
    return {"strategy": strategy, "audit_result": audit_result, "confidence": confidence}


# ---- build_position_review_section ----

def test_position_review_empty_returns_blank():
    assert db.build_position_review_section([]) == ""


def test_position_review_flags_sell_recommendations():
    rows = [_row("DDOG", "SELL (stop breached)", pnl_pct=-12.0)]
    section = db.build_position_review_section(rows)
    assert "**Position Review**" in section
    assert "DDOG: SELL (stop breached) (!)" in section


def test_position_review_no_flag_for_hold():
    rows = [_row("MSFT", "HOLD", pnl_pct=2.0)]
    section = db.build_position_review_section(rows)
    assert "MSFT: HOLD (" in section
    assert "(!)" not in section


def test_position_review_sorted_worst_pnl_first():
    rows = [_row("AAA", pnl_pct=8.0), _row("BBB", pnl_pct=-15.0), _row("CCC", pnl_pct=-3.0)]
    section = db.build_position_review_section(rows)
    lines = section.splitlines()[1:]
    tickers_in_order = [line.split(":")[0].strip() for line in lines]
    assert tickers_in_order == ["BBB", "CCC", "AAA"]


# ---- build_candidates_section ----

def test_candidates_empty_returns_blank():
    assert db.build_candidates_section({}) == ""


def test_candidates_clean_when_all_pass():
    result = db.build_candidates_section({"HST": [_candidate("llm_agent_qualitative_weighted", "PASS")]})
    assert "HST: CLEAN (1 pass / 0 fail of 1 variant(s))" in result


def test_candidates_all_failed_audit():
    result = db.build_candidates_section({"TDY": [_candidate("llm_agent", "FAIL")]})
    assert "TDY: ALL FAILED AUDIT (0 pass / 1 fail of 1 variant(s))" in result


def test_candidates_audit_unavailable_distinct_from_failed():
    # Confirmed live 2026-08-26 (URI): audit_result=None means the audit
    # call itself never returned a verdict -- a technical gap, not a
    # content critique -- and must not be conflated with a real FAIL.
    result = db.build_candidates_section({"URI": [_candidate("llm_agent", None)]})
    assert "URI: AUDIT UNAVAILABLE (0 pass / 0 fail of 1 variant(s))" in result
    assert "FAILED" not in result


def test_candidates_mixed_when_some_pass_some_fail():
    candidates = [_candidate("llm_agent", "FAIL"), _candidate("llm_agent_qualitative_weighted", "PASS")]
    result = db.build_candidates_section({"HST": candidates})
    assert "HST: MIXED (1 pass / 1 fail of 2 variant(s))" in result


def test_candidates_with_pass_sorted_before_without():
    candidates_by_ticker = {
        "ATD.TO": [_candidate("llm_agent", "FAIL")],
        "HST": [_candidate("llm_agent_qualitative_weighted", "PASS")],
    }
    result = db.build_candidates_section(candidates_by_ticker)
    lines = result.splitlines()[1:]
    tickers_in_order = [line.split(":")[0].strip() for line in lines]
    assert tickers_in_order == ["HST", "ATD.TO"]


# ---- build_portfolio_health_section ----

def _report(ic, trust_floor_met=True, eff_n=30.0):
    return {"overall_ic": ic, "trust_floor_met": trust_floor_met, "effective_n_settled": eff_n}


def test_portfolio_health_empty_when_none_cleared_floor():
    reports = {"rsi_mean_reversion": _report(-0.4, trust_floor_met=False)}
    assert db.build_portfolio_health_section(reports) == ""


def test_portfolio_health_excludes_none_ic_even_if_floor_met():
    reports = {"ma_crossover": _report(None, trust_floor_met=True)}
    assert db.build_portfolio_health_section(reports) == ""


def test_portfolio_health_shows_direction_and_ic():
    reports = {"llm_agent": _report(0.281)}
    section = db.build_portfolio_health_section(reports)
    assert "llm_agent: IC +0.281 (positive), n=30" in section


def test_portfolio_health_sorted_best_ic_first():
    reports = {
        "squeeze_breakout": _report(-0.32),
        "llm_agent": _report(0.28),
        "best_ideas_qualitative": _report(0.16),
    }
    section = db.build_portfolio_health_section(reports)
    lines = section.splitlines()[1:]
    names_in_order = [line.split(":")[0].strip() for line in lines]
    assert names_in_order == ["llm_agent", "best_ideas_qualitative", "squeeze_breakout"]


# ---- build_daily_briefing ----

def test_daily_briefing_all_empty_sections_gives_placeholder():
    result = db.build_daily_briefing([], {}, {}, today="2026-08-27")
    assert "Daily Briefing -- 2026-08-27" in result
    assert "Nothing to report today" in result


def test_daily_briefing_combines_all_present_sections():
    rows = [_row("DDOG", "SELL (stop breached)", pnl_pct=-12.0)]
    candidates = {"HST": [_candidate("llm_agent_qualitative_weighted", "PASS")]}
    health = {"llm_agent": _report(0.28)}
    result = db.build_daily_briefing(rows, candidates, health, today="2026-08-27")
    assert "Daily Briefing -- 2026-08-27" in result
    assert "**Position Review**" in result
    assert "**Today's LLM Agent Buy candidates**" in result
    assert "**Portfolio health**" in result


def test_daily_briefing_omits_empty_sections_individually():
    # Only candidates present -- Position Review and Portfolio health
    # sections should not appear at all, not appear empty.
    candidates = {"HST": [_candidate("llm_agent_qualitative_weighted", "PASS")]}
    result = db.build_daily_briefing([], candidates, {}, today="2026-08-27")
    assert "**Position Review**" not in result
    assert "**Portfolio health**" not in result
    assert "**Today's LLM Agent Buy candidates**" in result
