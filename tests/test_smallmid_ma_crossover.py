"""MA Crossover (Small/Mid-Cap) -- the new experimental variant added
2026-09-09 (improvements.txt item 121) that reuses ma_crossover's own live
System_Config v71 UNMODIFIED against the same separate ticker universe
(smallmid_watchlist.txt, ~1000 S&P 600/400 tickers) RSI's own small/mid-cap
variant uses. Deliberately kept OUT of config_loader.EXPERIMENTAL_STRATEGY_VERSIONS
(see that dict's own docstring for why), so it gets its own dedicated tests
here rather than automatic coverage from the dict-parametrized tests in
test_config_candidates_load.py/test_strategy_dispatch_parity.py -- same
convention as tests/test_smallmid_rsi.py, which this file mirrors closely.

smallmid_watchlist.txt's own parsing/schema is already covered by
test_smallmid_rsi.py::test_smallmid_watchlist_parses_real_tickers (not
strategy-specific, not duplicated here), as is
render_experimental_section()'s log_strategy_override mechanism itself
(test_smallmid_rsi.py::test_render_experimental_section_log_strategy_override,
already generic over any override value)."""
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
def test_smallmid_ma_crossover_config_loads():
    config, source = config_loader.load_config_by_version(config_loader.SMALLMID_MA_CROSSOVER_CONFIG_VERSION)
    assert config is not None, f"{config_loader.SMALLMID_MA_CROSSOVER_LABEL}: failed to load -- {source}"
    assert config.strategy == "ma_crossover", (
        f"{config_loader.SMALLMID_MA_CROSSOVER_LABEL}: expected strategy='ma_crossover' (reused "
        f"unmodified from the live primary-watchlist config), got {config.strategy!r}"
    )


def test_run_smallmid_ma_crossover_experimental_never_capital_allocates(monkeypatch, uptrend_ohlcv):
    """Same real bug class guarded for RSI's own variant (storage/signals.py
    unconditionally reads Shares_To_Buy/Est_Cost; a bare config.strategy log
    label would collide with ma_crossover's own existing primary-watchlist
    history) -- both checked here directly, without touching real
    Mongo/network."""
    config = swingtrade.TradingConfig(**{
        **swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": "ma_crossover",
    })
    levels = swingtrade.compute_ma_crossover_levels("TEST", uptrend_ohlcv, config)

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

    # Pass a real, existing file (this test module itself) explicitly -- the
    # default arg is bound at function-definition time, so monkeypatching the
    # module-level SMALLMID_WATCHLIST_FILE constant would NOT affect it. Its
    # content is never read here since read_tickers/read_ticker_sectors are mocked.
    ingest.run_smallmid_ma_crossover_experimental(watchlist_path=Path(__file__))

    assert "df" in captured, "log_trade_signals was never called"
    assert (captured["df"]["Shares_To_Buy"] == 0.0).all(), "Shares_To_Buy must always be 0.0 (never capital-allocated)"
    assert (captured["df"]["Est_Cost"] == 0.0).all(), "Est_Cost must always be 0.0 (never capital-allocated)"
    assert captured["strategy"] == config_loader.SMALLMID_MA_CROSSOVER_LOG_STRATEGY, (
        f"logged under {captured['strategy']!r}, expected the distinct label "
        f"{config_loader.SMALLMID_MA_CROSSOVER_LOG_STRATEGY!r} -- logging under the bare "
        "'ma_crossover' label would collide with ma_crossover's own existing primary-watchlist history"
    )


def test_run_smallmid_ma_crossover_experimental_skips_gracefully_when_watchlist_missing(capsys):
    missing_path = Path(__file__).resolve().parent / "nonexistent_smallmid_watchlist_for_test.txt"
    assert not missing_path.exists()
    ingest.run_smallmid_ma_crossover_experimental(watchlist_path=missing_path)  # must not raise
    assert "skipped" in capsys.readouterr().out.lower()
