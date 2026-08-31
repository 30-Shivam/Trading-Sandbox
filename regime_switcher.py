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
  unconditionally for every mechanical strategy that runs through
  swingtrade.levels.precompute_breakout_frame()/its own equivalents).
  Reuses the exact 25.0 "trending" threshold convention
  adx_trend_entry_threshold already established (classical TA: 25+ is
  "trending", 40+ is "strong trend") -- not a new number invented for
  this feature.
- ADX >= 25.0 -> "trending" regime -> prefer ma_crossover -- trend-following
  BY CONSTRUCTION (a short/long SMA crossover is a definitional
  trend-confirmation event).
- ADX < 25.0 -> "choppy" regime -> prefer pairs -- statistical-arbitrage
  mean-reversion between two correlated peers, which by design doesn't
  depend on either ticker having its own pre-existing trend, and (unlike
  the strategy this replaces) has real, currently-standing backtest
  validation (beats matched-count random-entry timing on every cut,
  ALL/TUNE/HOLDOUT -- see improvements.txt item 82).

2026-08-31 fix: BOTH original preference lists had silently rotted --
`breakout` (trending) was fully retired from every live-scanning path
weeks ago (item 73, superseded by ma_crossover as primary), and
`squeeze_breakout` (choppy, the ENTIRE original list for that regime) was
separately retired for a real, proven-negative live IC (-0.32 over 40
settled trades, item 102's own trading-strategy-status entry). Neither
retirement touched this file, so for weeks: "trending" silently degraded
to ma_crossover-only (breakout never fired to begin with), and "choppy"
silently became DEAD -- its one preferred strategy could never appear in
`strategy_rows` at all, so select_regime_pick() returned None for every
single choppy-regime ticker. This wasn't caught earlier because a clean
"no pick" and a real "nothing preferred fired" both look identical from
the caller's side -- only found by directly reading this file's own
REGIME_STRATEGY_PREFERENCE against what's actually still live, prompted
by regime_switcher's real settled-trade IC coming back solidly negative
(-0.47 over 22 effective trades) and tracing why.

Chose `pairs` over `rsi_mean_reversion` as squeeze_breakout's choppy-regime
replacement deliberately, not by default: RSI Mean-Reversion mechanically
fits the "choppy" thesis just as well on paper, but a fresh, rigorous
random-entry re-check of its live v66 config (2026-08-29) found it LOSES
to matched-count random timing on the holdout cut (win_rate 15.0% vs
random's 17.9%, sharpe_like 0.059 vs random's 0.095) -- no demonstrated
real edge, despite looking validated by a weaker internal check at
promotion time. Swapping in a strategy with no proven edge wouldn't have
actually fixed anything, just replaced one broken pillar with another.
`pairs` is the only currently-live strategy besides ma_crossover with a
real, standing beat-random validation on record.

Only draws from CURRENTLY LIVE, capital-eligible strategy versions
(whichever versions config_loader currently points at), not undecided
candidates, so there's no ambiguity about which parameter variant is
being tested.

NEVER capital-allocated -- Shares_To_Buy/Est_Cost are always 0, no cash
pool, no allocate_capital() call, same as llm_agent.py.
"""

import pandas as pd

ADX_TREND_THRESHOLD = 25.0  # matches config.adx_trend_entry_threshold's own default

REGIME_STRATEGY_PREFERENCE = {
    "trending": ["ma_crossover"],
    "choppy": ["pairs"],
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
    """`strategy_rows` is {strategy_name: row_dict, ...} for whichever
    currently-live strategies actually fired (Signal != "Ignore") for this
    ticker today (e.g. "ma_crossover", "pairs", "rsi_mean_reversion") --
    strategies that didn't fire simply aren't keys in this dict, and this
    function has no fixed notion of which strategies exist beyond whatever
    REGIME_STRATEGY_PREFERENCE names. Reads ADX from any
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
