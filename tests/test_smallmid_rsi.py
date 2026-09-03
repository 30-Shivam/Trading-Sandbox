"""RSI Mean-Reversion (Small/Mid-Cap) -- the new experimental variant added
2026-09-01 (improvements.txt items 114/115) that reuses RSI Mean-Reversion's
own System_Config v66 UNMODIFIED against a completely separate ticker
universe (smallmid_watchlist.txt, ~1000 S&P 600/400 tickers) instead of the
primary watchlist. Deliberately kept OUT of
config_loader.EXPERIMENTAL_STRATEGY_VERSIONS (see that dict's own docstring
for why), so it gets its own dedicated tests here rather than automatic
coverage from the dict-parametrized tests in test_config_candidates_load.py
/ test_strategy_dispatch_parity.py.
"""
from pathlib import Path

import pandas as pd
import pytest

import config_loader
import ingest
import swingtrade
import watchlist

SMALLMID_WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "smallmid_watchlist.txt"


def _mongo_available() -> bool:
    try:
        import storage
        storage.get_db()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_smallmid_rsi_config_loads():
    config, source = config_loader.load_config_by_version(config_loader.SMALLMID_RSI_CONFIG_VERSION)
    assert config is not None, f"{config_loader.SMALLMID_RSI_LABEL}: failed to load -- {source}"
    assert config.strategy == "rsi", (
        f"{config_loader.SMALLMID_RSI_LABEL}: expected strategy='rsi' (reused unmodified from "
        f"RSI Mean-Reversion), got {config.strategy!r}"
    )


def test_smallmid_watchlist_parses_real_tickers():
    assert SMALLMID_WATCHLIST_FILE.exists(), f"{SMALLMID_WATCHLIST_FILE} not found"
    tickers = watchlist.read_tickers(SMALLMID_WATCHLIST_FILE)
    sectors = watchlist.read_ticker_sectors(SMALLMID_WATCHLIST_FILE)

    assert len(tickers) > 900, f"expected ~1000 tickers, got {len(tickers)}"
    assert len(set(tickers)) == len(tickers), "duplicate tickers in smallmid_watchlist.txt"

    # Spot-check real, known constituents from each source index.
    assert "CROX" in tickers  # S&P 600 SmallCap
    assert "AAL" in tickers  # S&P 400 MidCap
    assert sectors.get("CROX") == "Consumer Discretionary"
    assert sectors.get("AAL") == "Industrials"

    # Every sector present must be one watchlist.SECTOR_ETF actually knows --
    # a real mismatch (e.g. Wikipedia's "Health Care" vs this project's own
    # "Healthcare") would silently drop sector-ETF data for every affected
    # ticker (see market_data.fetch_ticker_bundle()'s set-intersection logic).
    unknown_sectors = set(sectors.values()) - set(watchlist.SECTOR_ETF.keys())
    assert not unknown_sectors, f"sector name(s) not in watchlist.SECTOR_ETF: {unknown_sectors}"

    # Confirmed non-fetchable via yfinance this session -- must stay excluded.
    assert "CWEN.A" not in tickers
    assert "MOG.A" not in tickers

    # No overlap with the primary large-cap watchlist -- the whole point of
    # this variant is a DIFFERENT universe.
    main_watchlist = Path(__file__).resolve().parent.parent / "watchlist.txt"
    main_tickers = set(watchlist.read_tickers(main_watchlist))
    assert not (set(tickers) & main_tickers), "smallmid_watchlist.txt overlaps with the primary watchlist"


def test_run_smallmid_rsi_experimental_never_capital_allocates(monkeypatch, uptrend_ohlcv):
    """Same real bug class run_experimental_strategies() was fixed for
    2026-08-27 (storage/signals.py unconditionally reads Shares_To_Buy/
    Est_Cost) plus the label-collision risk found while building this
    variant (render_experimental_section() logging under bare config.strategy
    would collide with RSI's own old contaminated "rsi" history) -- both
    guarded here directly, without touching real Mongo/network."""
    config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "rsi", "rsi_oversold_threshold": 100.0,
    })
    levels = swingtrade.compute_levels("TEST", uptrend_ohlcv, config)

    monkeypatch.setattr(config_loader, "load_config_by_version", lambda version: (config, "test"))
    monkeypatch.setattr(
        ingest.market_data, "fetch_ticker_bundle",
        lambda tickers, sector_lookup=None: ({}, None, [], {}, None),
    )
    monkeypatch.setattr(
        ingest.market_data, "score_bundle_for_strategy",
        lambda bundle, market_df, config, **kwargs: ([levels], []),
    )

    captured = {}

    def fake_log_trade_signals(df, config_snapshot):
        captured["df"] = df.copy()
        captured["strategy"] = config_snapshot["strategy"]
        return {"actionable": 0, "research": len(df)}

    monkeypatch.setattr(ingest.storage, "log_trade_signals", fake_log_trade_signals)
    monkeypatch.setattr(ingest, "read_tickers", lambda path: ["TEST"])
    monkeypatch.setattr(ingest, "read_ticker_sectors", lambda path: {})

    # Pass a real, existing file (this test module itself) explicitly --
    # watchlist_path's default is bound at function-definition time, so
    # monkeypatching the module-level SMALLMID_WATCHLIST_FILE constant would
    # NOT affect an already-defined default argument. Its content is never
    # read here since read_tickers/read_ticker_sectors are mocked above.
    ingest.run_smallmid_rsi_experimental(watchlist_path=Path(__file__))

    assert "df" in captured, "log_trade_signals was never called"
    assert (captured["df"]["Shares_To_Buy"] == 0.0).all(), "Shares_To_Buy must always be 0.0 (never capital-allocated)"
    assert (captured["df"]["Est_Cost"] == 0.0).all(), "Est_Cost must always be 0.0 (never capital-allocated)"
    assert captured["strategy"] == config_loader.SMALLMID_RSI_LOG_STRATEGY, (
        f"logged under {captured['strategy']!r}, expected the distinct label "
        f"{config_loader.SMALLMID_RSI_LOG_STRATEGY!r} -- logging under the bare 'rsi' label would "
        "collide with RSI Mean-Reversion's own old contaminated pre-v66 history"
    )


def test_render_experimental_section_log_strategy_override(monkeypatch, uptrend_ohlcv):
    """dip_buy_analyzer.render_experimental_section() gained a new optional
    `log_strategy_override` param (2026-09-01, for this same variant --
    without it, calling this shared function with v66 unmodified would log
    under the bare 'rsi' label, colliding with RSI Mean-Reversion's own old
    contaminated pre-v66 history). Default None must stay a no-op (existing
    callers like Momentum Rank), and passing an override must actually
    change what gets logged."""
    import dip_buy_analyzer

    config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "rsi", "rsi_oversold_threshold": 100.0,
    })
    levels = swingtrade.compute_levels("TEST", uptrend_ohlcv, config)

    monkeypatch.setattr(
        dip_buy_analyzer.market_data, "score_bundle_for_strategy",
        lambda bundle, market_df, config, **kwargs: ([levels], []),
    )
    captured = {}

    def fake_log_trade_signals(df, config_snapshot):
        captured["strategy"] = config_snapshot["strategy"]
        return {"actionable": 0, "research": len(df)}

    monkeypatch.setattr(dip_buy_analyzer.storage, "log_trade_signals", fake_log_trade_signals)

    # No override -- must log under config.strategy unchanged (Momentum Rank's own behavior).
    dip_buy_analyzer.render_experimental_section(
        "label", config, {}, None, [], storage_ok=True,
    )
    assert captured["strategy"] == "rsi"

    # With override -- must log under the distinct label instead.
    dip_buy_analyzer.render_experimental_section(
        "label", config, {}, None, [], storage_ok=True,
        log_strategy_override=config_loader.SMALLMID_RSI_LOG_STRATEGY,
    )
    assert captured["strategy"] == config_loader.SMALLMID_RSI_LOG_STRATEGY


def test_run_smallmid_rsi_experimental_skips_gracefully_when_watchlist_missing(capsys):
    missing_path = Path(__file__).resolve().parent / "nonexistent_smallmid_watchlist_for_test.txt"
    assert not missing_path.exists()
    ingest.run_smallmid_rsi_experimental(watchlist_path=missing_path)  # must not raise
    assert "skipped" in capsys.readouterr().out.lower()
