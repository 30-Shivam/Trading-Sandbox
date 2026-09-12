"""Mean-Reversion Pairs (Small/Mid-Cap) -- the new experimental variant
added 2026-09-11 (improvements.txt item 125) that reuses Mean-Reversion
Pairs' own live System_Config v58 UNMODIFIED against the same separate
ticker universe (smallmid_watchlist.txt, ~1000 S&P 600/400 tickers) the
other small/mid-cap variants use. Deliberately kept OUT of
config_loader.EXPERIMENTAL_STRATEGY_VERSIONS (see that dict's own docstring
for why), so it gets its own dedicated tests here rather than automatic
coverage from the dict-parametrized tests in test_config_candidates_load.py
/test_strategy_dispatch_parity.py -- same convention as
tests/test_smallmid_rsi.py and tests/test_smallmid_ma_crossover.py, which
this file mirrors closely.

smallmid_watchlist.txt's own parsing/schema and render_experimental_section()'s
log_strategy_override mechanism are already covered generically elsewhere
(test_smallmid_rsi.py), not duplicated here."""
from pathlib import Path

import pytest

import config_loader
import ingest
import swingtrade

SMALLMID_WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "smallmid_watchlist.txt"


def _mongo_available() -> bool:
    try:
        import storage
        storage.get_db()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mongo_available(), reason="MONGODB_URI not configured/reachable")
def test_smallmid_pairs_config_loads():
    config, source = config_loader.load_config_by_version(config_loader.SMALLMID_PAIRS_CONFIG_VERSION)
    assert config is not None, f"{config_loader.SMALLMID_PAIRS_LABEL}: failed to load -- {source}"
    assert config.strategy == "pairs", (
        f"{config_loader.SMALLMID_PAIRS_LABEL}: expected strategy='pairs' (reused unmodified from "
        f"the live primary-watchlist config), got {config.strategy!r}"
    )


def test_run_smallmid_pairs_experimental_never_capital_allocates(monkeypatch, uptrend_ohlcv):
    """Same real bug class guarded for the other smallmid variants
    (storage/signals.py unconditionally reads Shares_To_Buy/Est_Cost; a bare
    config.strategy log label would collide with Mean-Reversion Pairs' own
    existing primary-watchlist history) -- both checked here directly,
    without touching real Mongo/network. Also confirms
    market_data.build_pair_price_panels() is actually called (a real gap
    caught while building this: without it, simulate_pairs_signals() would
    silently score zero signals)."""
    config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "pairs",
    })
    levels = swingtrade.compute_pairs_levels("TEST", uptrend_ohlcv, config, peer_prices=None)

    monkeypatch.setattr(config_loader, "load_config_by_version", lambda version: (config, "test"))
    monkeypatch.setattr(
        ingest.market_data, "fetch_ticker_bundle",
        lambda tickers, sector_lookup=None: ({}, None, [], {}, None),
    )
    built_panels = {}

    def fake_build_pair_price_panels(bundle, sector_lookup):
        built_panels["called"] = True
        return {}

    monkeypatch.setattr(ingest.market_data, "build_pair_price_panels", fake_build_pair_price_panels)
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

    # Pass a real, existing file (this test module itself) explicitly -- the
    # default arg is bound at function-definition time, so monkeypatching the
    # module-level SMALLMID_WATCHLIST_FILE constant would NOT affect it. Its
    # content is never read here since read_tickers/read_ticker_sectors are mocked.
    ingest.run_smallmid_pairs_experimental(watchlist_path=Path(__file__))

    assert built_panels.get("called"), "build_pair_price_panels() was never called -- pairs would silently score 0 signals"
    assert "df" in captured, "log_trade_signals was never called"
    assert (captured["df"]["Shares_To_Buy"] == 0.0).all(), "Shares_To_Buy must always be 0.0 (never capital-allocated)"
    assert (captured["df"]["Est_Cost"] == 0.0).all(), "Est_Cost must always be 0.0 (never capital-allocated)"
    assert captured["strategy"] == config_loader.SMALLMID_PAIRS_LOG_STRATEGY, (
        f"logged under {captured['strategy']!r}, expected the distinct label "
        f"{config_loader.SMALLMID_PAIRS_LOG_STRATEGY!r} -- logging under the bare "
        "'pairs' label would collide with Mean-Reversion Pairs' own existing primary-watchlist history"
    )


def test_run_smallmid_pairs_experimental_skips_gracefully_when_watchlist_missing(capsys):
    missing_path = Path(__file__).resolve().parent / "nonexistent_smallmid_watchlist_for_test.txt"
    assert not missing_path.exists()
    ingest.run_smallmid_pairs_experimental(watchlist_path=missing_path)  # must not raise
    assert "skipped" in capsys.readouterr().out.lower()
