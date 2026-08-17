"""Regime-conditional strategy switcher -- EXPLICITLY prospective-only,
skips this project's own established backtesting-validation pipeline
(benchmark_random_entry.py's 5-year real-vs-random gate) by deliberate user
choice. Every other strategy in this codebase went through that gate
before going live, even at research/experimental tier -- this one doesn't,
because the hypothesis it tests (that no single mechanical strategy is
best in every market condition) is about WHEN to trust which existing,
already-individually-validated strategy, not a new signal of its own that
backtesting could evaluate the same way. It can only be judged by real
settled trades accumulating over real calendar time -- see the "Validation
progress" section wherever this is displayed for the same trust-floor
convention llm_agent.py already established (20-30 settled trades, 4-6+
weeks minimum before drawing any conclusion).

Motivated by a real, session-specific finding: a per-ticker diagnostic
comparing two ma_crossover configs (v51 vs. a faster-reacting candidate,
v54) found the faster one won specifically in defensive/low-volatility
sectors (Consumer Staples, Energy) and lost broadly in higher-volatility
growth/cyclical sectors (Materials, Technology, Consumer Discretionary,
Healthcare, Industrials) -- real evidence that "which methodology wins"
genuinely shifts with market conditions, not just an assumption.

Deliberately NOT a trained/tuned/optimized model -- that would reintroduce
the exact overfitting-surface risk multiple Optuna searches this session
have already been burned by. Instead, a small, hand-specified, explainable
hypothesis grounded in each mechanical strategy's OWN already-documented
design intent, for prospective data to actually test:

- Regime signal: ADX (Average Directional Index, already computed
  unconditionally for breakout/squeeze_breakout/ma_crossover via
  swingtrade.levels.precompute_breakout_frame()). Reuses the exact 25.0
  "trending" threshold convention adx_trend_entry_threshold already
  established (classical TA: 25+ is "trending", 40+ is "strong trend") --
  not a new number invented for this feature.
- ADX >= 25.0 -> "trending" regime -> prefer breakout, then ma_crossover --
  both are trend-following BY CONSTRUCTION (breakout requires a fresh
  N-day high in a confirmed uptrend; ma_crossover requires a short/long
  SMA crossover, a definitional trend-confirmation event).
- ADX < 25.0 -> "choppy" regime -> prefer squeeze_breakout -- its own
  documented design is explicitly built to NOT require a pre-existing
  trend, targeting the volatility-contraction-then-expansion ("coiled
  spring") pattern that characteristically occurs during range-bound/
  choppy conditions.

Only draws from the 3 CURRENTLY LIVE, capital-eligible strategy versions
(breakout, squeeze_breakout, ma_crossover -- whichever versions
config_loader currently points at), not undecided candidates like v53/v54,
so there's no ambiguity about which parameter variant is being tested.

NEVER capital-allocated -- Shares_To_Buy/Est_Cost are always 0, no cash
pool, no allocate_capital() call, same as llm_agent.py.
"""

import pandas as pd

ADX_TREND_THRESHOLD = 25.0  # matches config.adx_trend_entry_threshold's own default

REGIME_STRATEGY_PREFERENCE = {
    "trending": ["breakout", "ma_crossover"],
    "choppy": ["squeeze_breakout"],
}


def classify_regime(adx: float | None, adx_trend_threshold: float = ADX_TREND_THRESHOLD) -> str | None:
    """"trending" (adx >= threshold), "choppy" (adx < threshold), or None if
    adx itself is unavailable (can't classify without it -- see
    select_regime_pick(), which treats an unclassifiable ticker as no pick,
    not a fallback guess)."""
    if adx is None or pd.isna(adx):
        return None
    return "trending" if adx >= adx_trend_threshold else "choppy"


def select_regime_pick(ticker: str, strategy_rows: dict[str, dict]) -> dict | None:
    """`strategy_rows` is {"breakout": row_dict, "squeeze_breakout": row_dict,
    "ma_crossover": row_dict} for whichever of the 3 strategies actually
    fired (Signal != "Ignore") for this ticker today -- strategies that
    didn't fire simply aren't keys in this dict. Reads ADX from any
    available row (identical value regardless of which strategy computed
    it -- same underlying OHLCV/config.adx_window), classifies the regime,
    then walks that regime's preference list IN ORDER and returns the
    first strategy's row that's actually among the ones that fired.

    Returns None (no pick) if: no strategy fired for this ticker, ADX
    isn't available on any fired row, or none of the regime's preferred
    strategies are among the ones that fired (a different, non-preferred
    strategy fired instead) -- deliberately no secondary fallback/tie-break
    rule invented on top of this; a clean "no pick" is more honest than a
    guessed one."""
    if not strategy_rows:
        return None

    adx = None
    for row in strategy_rows.values():
        candidate_adx = row.get("ADX")
        if candidate_adx is not None and not pd.isna(candidate_adx):
            adx = candidate_adx
            break
    regime = classify_regime(adx)
    if regime is None:
        return None

    for preferred_strategy in REGIME_STRATEGY_PREFERENCE[regime]:
        if preferred_strategy in strategy_rows:
            row = dict(strategy_rows[preferred_strategy])
            row["Regime"] = regime
            row["Source_Strategy"] = preferred_strategy
            return row
    return None
