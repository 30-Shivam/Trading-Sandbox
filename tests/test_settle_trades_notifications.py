"""settle_trades._build_settlement_notification() -- mirrors ingest.py's own
_build_signal_notification() in test shape: pure function, dict fixtures,
no live Discord calls, no Mongo. See tests/test_llm_resolve_dual.py for the
same dependency-injection-free convention.
"""
import settle_trades


def _mk(ticker, status, exit_reason, pnl_pct, holding_days, confirmed, tier="actionable", strategy="breakout"):
    return {
        "ticker": ticker, "strategy": strategy, "status": status, "exit_reason": exit_reason,
        "pnl_pct": pnl_pct, "holding_days": holding_days, "confirmed": confirmed, "tier": tier,
    }


def test_empty_records_returns_none():
    assert settle_trades._build_settlement_notification("breakout", []) is None


def test_single_record_renders_header_and_line():
    records = [_mk("NVDA", "WIN", "target_hit", 3.45, 5, True)]
    message = settle_trades._build_settlement_notification("breakout", records)
    assert "**breakout: 1 trade(s) settled**" in message
    assert "NVDA: WIN (target_hit, +3.45%, held 5d, CONFIRMED)" in message


def test_unconfirmed_tag_rendered_correctly():
    records = [_mk("TSLA", "LOSS", "stop_hit", -1.20, 3, False)]
    message = settle_trades._build_settlement_notification("breakout", records)
    assert "TSLA: LOSS (stop_hit, -1.20%, held 3d, unconfirmed)" in message


def test_multiple_records_sorted_by_pnl_descending():
    records = [
        _mk("AAA", "LOSS", "stop_hit", -2.0, 4, False),
        _mk("BBB", "WIN", "target_hit", 5.0, 6, True),
        _mk("CCC", "EXPIRED", "max_holding_days", 0.5, 10, True),
    ]
    message = settle_trades._build_settlement_notification("rsi", records)
    lines = message.splitlines()
    assert lines[0] == "**rsi: 3 trade(s) settled**"
    tickers_in_order = [line.split(":")[0] for line in lines[1:]]
    assert tickers_in_order == ["BBB", "CCC", "AAA"]


def test_expired_status_included():
    records = [_mk("MSFT", "EXPIRED", "max_holding_days", 0.12, 21, True)]
    message = settle_trades._build_settlement_notification("ma_crossover", records)
    assert "MSFT: EXPIRED (max_holding_days, +0.12%, held 21d, CONFIRMED)" in message
