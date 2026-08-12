"""Guards against the exact bug class that silently broke ingest.py's
"Squeeze Breakout"/"MA Crossover" secondary scans: dip_buy_analyzer.py and
ingest.py each maintain their OWN copy of the strategy -> add_*_trade_score
dispatch function, and ingest.py's copy was never updated when
squeeze_breakout/ma_crossover were promoted into
config_loader.SECONDARY_STRATEGY_VERSIONS -- any day either strategy had a
real signal, it crashed with KeyError: 'Oversold_Streak_Days' (falling
through to the RSI-only scorer) before ever reaching MongoDB. See
improvements.txt for the incident.

For every strategy this codebase supports, builds one REAL, schema-complete
levels row (via the actual swingtrade.compute_*_levels() functions against
a synthetic-but-realistic OHLCV fixture -- see conftest.py -- not a
hand-typed dict, so the row's columns can't silently drift from what
production actually produces) and asserts BOTH dispatch copies (a) don't
raise and (b) return the identical Trade_Score for it.
"""
import pandas as pd
import pytest

import swingtrade
import dip_buy_analyzer
import ingest

STRATEGIES = [
    "rsi", "breakout", "pullback", "breakout_retest", "week52_high",
    "momentum_burst", "squeeze_breakout", "adx_trend_entry", "ma_crossover",
]

COMPUTE_LEVELS_FN = {
    "rsi": swingtrade.compute_levels,
    "breakout": swingtrade.compute_breakout_levels,
    "pullback": swingtrade.compute_pullback_levels,
    "breakout_retest": swingtrade.compute_breakout_retest_levels,
    "week52_high": swingtrade.compute_week52_levels,
    "momentum_burst": swingtrade.compute_momentum_burst_levels,
    "squeeze_breakout": swingtrade.compute_squeeze_breakout_levels,
    "adx_trend_entry": swingtrade.compute_adx_trend_entry_levels,
    "ma_crossover": swingtrade.compute_ma_crossover_levels,
}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_dispatch_parity(strategy, uptrend_ohlcv):
    config = swingtrade.TradingConfig(**{**swingtrade.DEFAULT_CONFIG.to_dict(), "strategy": strategy})
    levels = COMPUTE_LEVELS_FN[strategy]("TEST", uptrend_ohlcv, config)
    df = pd.DataFrame([levels])

    dashboard_result = dip_buy_analyzer._score_for_strategy(df.copy(), config)
    ingest_result = ingest._score_for_strategy(df.copy(), config)

    assert "Trade_Score" in dashboard_result.columns, f"{strategy}: dashboard dispatch produced no Trade_Score column"
    assert "Trade_Score" in ingest_result.columns, f"{strategy}: ingest dispatch produced no Trade_Score column"
    assert dashboard_result["Trade_Score"].iloc[0] == ingest_result["Trade_Score"].iloc[0], (
        f"{strategy}: dip_buy_analyzer._score_for_strategy and ingest._score_for_strategy disagree -- "
        f"dashboard={dashboard_result['Trade_Score'].iloc[0]}, ingest={ingest_result['Trade_Score'].iloc[0]} "
        "(the two dispatch copies have drifted out of sync -- see this module's own docstring)"
    )
