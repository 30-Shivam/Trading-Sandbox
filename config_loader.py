"""Load the active TradingConfig from MongoDB's System_Config, falling back
to swingtrade.DEFAULT_CONFIG when Mongo is unreachable, unconfigured, or
nothing has been promoted yet. Shared by the interactive dashboard
(dip_buy_analyzer.py, which wraps this in st.cache_data on a TTL) and the
standalone scheduled scan (ingest.py, which calls it once per run) so both
always agree on which config is "live" -- see ARCHITECTURE_PLAN.md Phase 6/7.
"""

import storage
import swingtrade

# Fixed, immutable candidate System_Config versions for the validated
# secondary strategies shown alongside the active config -- see
# improvements.txt items 27/28. Single source of truth shared by
# dip_buy_analyzer.py (the dashboard's secondary sections) and ingest.py
# (the headless scheduled scan) so the two can never silently drift apart
# on which candidate versions are "the" secondary strategies.
#
# Breakout Retest (v27) and 52-Week High (v28) removed 2026-08-11 per
# explicit user request ("remove all the losing strategies off the
# dashboard") -- both strategies' apparent edges evaporated once forced
# into any RRR shape that could actually clear the live scoring ceiling
# (v44/v45, improvements.txt item 42/43): win rates collapsed to
# statistically indistinguishable from DEFAULT_CONFIG's own unfiltered
# baseline, regardless of which end of the RRR moved. v27/v28 themselves
# (at their original, non-fireable RRR) are UNCHANGED in Mongo -- this is
# purely a dashboard/capital-eligibility removal, not a data deletion; the
# strategies could be re-added here if a genuine refinement is ever found.
#
# MA Crossover (v49) promoted 2026-08-11 per explicit user request
# ("promote this to capital eligible as well") -- the first strategy since
# squeeze_breakout to clear the full validation pipeline: real-vs-random
# benchmark passed on all 3 cuts at untuned defaults (item 49), its own
# core-trigger parameters (short/long window) refined with nothing beating
# baseline (item 50), entry-fill sensitivity checked under both limit/
# next_open (edge holds direction under both, doesn't reverse like
# momentum_burst's did), RRR-vs-scoring-ceiling confirmed compatible
# (clears signal_buy_threshold, same ceiling shape as active v43), and a
# live smoke test against the real watchlist confirmed clean dispatch +
# real per-ticker score differentiation. See improvements.txt item 51.
#
# MA Crossover swapped v49 -> v51 (2026-08-15) per explicit user request
# ("promote v51"), after a full Optuna re-tune under the fully-corrected
# objective (RRR_FLOOR + recency + correlation + ticker-holdout + multi-
# objective drawdown, all together for the first time) found v51 beats v49
# head-to-head on an identical 408-ticker/5-year real-vs-random benchmark
# (ALL 0.029 vs 0.003, TUNE 0.029 vs 0.019, HOLDOUT 0.06 vs -0.038) with
# only a mild ~21% trade-frequency reduction. v49 itself was independently
# found to now LOSE to random-entry timing on ALL/TUNE on today's expanded
# ticker universe -- v51 isn't just an incremental improvement, it's a real
# fix for an edge that had quietly weakened. v51's RRR (1.6675) only just
# clears signal_buy_threshold (best-case Trade_Score=61.1 vs the 60 floor,
# a much thinner margin than v49's 66.7) -- worth remembering it may fire
# live Buy signals less often than its backtested trade count implies. See
# improvements.txt item 59/60. v49 stays in Mongo as status=candidate,
# unchanged, in case this ever needs reverting.
#
# MA Crossover swapped v51 -> v55 (2026-08-16) per explicit user request
# ("yes" to promoting v55), after a fresh Optuna re-tune using the new
# backtest/Optuna-only sector_relative_strength_min filter (improvements.txt
# items 68-71) found v55 (short_window=24, long_window=89,
# sector_relative_strength_min=-3.207, tp/sl pinned at v51's own exact
# values, so RRR=1.6675 is unchanged and the signal_buy_threshold ceiling
# compatibility already established for v51 carries over unchanged).
# v55's 1-year WFO screen showed a HOLDOUT reversal (the classic TUNE-wins/
# HOLDOUT-loses overfitting shape) -- but the mandatory 5-year
# benchmark_random_entry.py check, run with the NEW multi-seed averaged
# holdout (item 69, applied to a real promotion decision for the first
# time), showed that reversal was itself a single-seed artifact: v55 beats
# RANDOM decisively on ALL/TUNE/HOLDOUT (every one of 10 TUNE seeds
# positive; HOLDOUT's worst seed still beat RANDOM's best). Head-to-head
# against v51 on the matched universe: v55 beats v51 on EVERY cut and
# EVERY metric -- ALL (sharpe 0.09 vs 0.031), TUNE (0.091 vs 0.027),
# HOLDOUT (0.078 vs 0.041), win_rate higher on all three too -- no
# tradeoff anywhere, unlike v54's earlier ALL/TUNE-win-HOLDOUT-loss result
# (item 65). Trade frequency stays healthy (~88% of v51's). See
# improvements.txt item 72. v51 stays in Mongo as status=candidate,
# unchanged, in case this ever needs reverting.
#
# MA Crossover PROMOTED FROM SECONDARY TO PRIMARY (2026-08-16, improvements.txt
# item 73) per explicit user request ("promote ma_crossover v55 to primary"),
# replacing breakout (v43) as the PRIMARY/active System_Config. Motivated by
# two real findings the same day: (1) a clean multi-seed-averaged re-check of
# v43 (item 69's methodology) showed only a thin, cut-dependent edge -- beats
# RANDOM narrowly on ALL/TUNE but clearly LOSES on HOLDOUT (sharpe 0.020 vs
# 0.046, win_rate 33.9% vs 37.0%); (2) a strict-vs-loosened-filter comparison
# proved v43's own inherited filters (from v19, esp. breakout_volume_ratio_min
# ~1.55) are NOT the problem -- removing them makes every cut worse, including
# flipping HOLDOUT negative -- so the underlying "fresh 45-day high" TRIGGER
# itself, not the filters, is the weak link, confirming a recommendation this
# project already made and never fully acted on back on 2026-08-11 (item 47:
# "pause new capital to breakout... or leave the slot deliberately empty").
# Rather than leave the slot empty, replaced it with v55 -- already the
# strongest, most cleanly-validated candidate found this entire session (see
# item 72 above). `promote_config.py --promote 55` handles the actual Mongo
# state change (sets v55 status=active, v43 status=retired automatically) --
# see storage/system_config.py::promote_candidate(). MA Crossover REMOVED from
# this dict (was "MA Crossover": 55 immediately above) since it's now covered
# by the primary slot instead -- leaving it here too would double-scan/
# double-display it (once as primary, once as its own secondary section).
# v43 (breakout) is now fully retired from every live-scanning path (neither
# primary nor secondary) -- this IS the "removal" the user asked about,
# achieved as a natural side effect of the promotion itself, no separate code
# needed. v43 stays in Mongo unchanged (status=retired, not deleted) in case
# of a revert. Real capital consequence worth remembering: ma_crossover moving
# from secondary (its own DEFAULT_TOTAL_CASH=$5,000 pool) to primary
# (DEFAULT_PRIMARY_CASH=$0, per item 48's original breakout-specific caution)
# means LESS capital sizes against it by default now, not more -- the user
# must deliberately type in a primary cash amount, same "nothing moves without
# an explicit action" posture item 48 established.
# RSI Mean-Reversion (v17) added 2026-08-20 (improvements.txt item 81) after
# clearing real validation (beats matched-count random-entry timing on every
# cut, ALL/TUNE/HOLDOUT -- see item 77/80). Real capital consequence: gets
# the same default $5,000 cash pool ma_crossover got when IT first cleared
# this bar (item 51's precedent), per explicit user approval -- NOT added to
# dip_buy_analyzer.py's own SECONDARY_DEFAULT_CASH_OVERRIDES. Its Trade_
# Signals/Trade_Outcomes log under strategy="rsi_mean_reversion", NOT bare
# "rsi" -- Mongo already has 175 settled trades tagged strategy="rsi" from
# 2026-07-24/31, predating v17's tuning and this project's own ticker-
# holdout/multi-seed/RRR-floor fixes; a distinct label avoids silently
# pooling those stale, unrelated trades into v17's fresh IC/IR track record
# -- see SECONDARY_LOG_STRATEGY_OVERRIDES immediately below.
# Mean-Reversion Pairs (v58, lean v1 -- all DEFAULT_CONFIG values, no Optuna
# tuning) added 2026-08-21 (improvements.txt items 82/84) after clearing the
# same real validation gate. Deliberately the LEAN candidate, not the
# Optuna-tuned v57 (also validated, real improvement on sharpe_like/
# k_ratio) -- v57's HOLDOUT win_rate (12.32%) exceeded the user's own risk
# tolerance despite the stronger aggregate numbers; v58's much higher
# win_rate (42%) was chosen instead, per explicit user decision. v57 stays
# an unpromoted, unwired candidate for future reconsideration once pairs
# has its own real prospective track record. No existing Trade_Signals/
# Trade_Outcomes contamination risk under strategy="pairs" (confirmed via a
# direct Mongo count before wiring -- 0 documents, unlike RSI's case) -- no
# SECONDARY_LOG_STRATEGY_OVERRIDES entry needed, logs under its own real
# "pairs" label directly. Real capital consequence: gets the same default
# $5,000 cash pool every other newly-cleared secondary strategy has gotten.
SECONDARY_STRATEGY_VERSIONS = {
    # Squeeze Breakout REMOVED 2026-08-26, per explicit user request during
    # a portfolio-simplification pass ("I don't want it influencing anything
    # else"). v53 (promoted 2026-08-21) cleared the backtest/real-vs-random
    # validation that earned it a spot here, but its live IC never recovered
    # -- confirmed -0.28 (v39, 27 trades, 2026-08-20) and then -0.32 (v53, 40
    # trades, 2026-08-25/26), a real, worsening-not-improving pattern over a
    # sizeable, trust-floor-cleared sample. It had already been fully
    # neutralized everywhere that matters (SECONDARY_DEFAULT_CASH_OVERRIDES
    # defaulted its cash pool to $0 since 2026-08-20; ensemble_weight()
    # zeroes it out of the Best Ideas composite as of the 2026-08-25 fix) --
    # this removes it from the daily scan/LLM-candidate-pool entirely,
    # rather than leaving dead weight running forever. Same "remove all the
    # losing strategies" treatment momentum_burst (v38) and adx_trend_entry
    # (v40) got 2026-08-11 -- the underlying strategy code, Mongo candidates,
    # and historical Trade_Signals/Trade_Outcomes are untouched, only taken
    # out of this dict (and best_ideas.METHODOLOGIES, see that list's own
    # comment). Revisit only if a future retune's own live IC turns out
    # genuinely non-negative -- a better backtest result alone isn't enough,
    # per the exact lesson SECONDARY_DEFAULT_CASH_OVERRIDES's history taught.
    #
    # Promoted 2026-08-25 (v17 -> v66): v17's RRR (9.75) saturated
    # rrr_score_cap (4.0), granting the RRR score term's full 40/100
    # points to every ticker regardless of real RSI -- confirmed live,
    # 295/407 watchlist tickers flagged on 2026-08-25 alone (some with
    # RSI as high as 78, badly overbought). Also silently loosened v17's
    # OWN backtest entry gate during its original tuning (ENTRY_SIGNALS
    # is computed from the same saturated Trade_Score), so its validation
    # history was suspect too, not just its live display. v66 was tuned
    # under a corrected search space (optimize.py's new RRR_CEILING) that
    # can no longer produce a saturated ratio (v66's RRR=1.7, fully
    # discriminating) and clearly beats v17 on true ticker-holdout
    # (sharpe_like 0.412 vs 0.057) -- though its real-vs-random gap check
    # did not improve on DEFAULT_CONFIG's own, so treat this as "fixes a
    # real bug and beats what was live," not "found genuine new alpha."
    "RSI Mean-Reversion": 66,
    "Mean-Reversion Pairs": 58,
}

# Per-label override of what strategy name gets LOGGED to Mongo, separate
# from config.strategy (which every scoring/dispatch function still keys
# off unchanged -- see dip_buy_analyzer.py's render_secondary_section()/
# ingest.py's run_secondary_strategy(), both take a log_strategy_override
# param that builds a throwaway TradingConfig with ONLY .strategy swapped,
# used solely for storage.log_trade_signals()'s config_snapshot argument).
# Same "config drives dispatch, a separate label drives what's logged"
# pattern llm_agent.py's variant_strategy_name() already established for
# prompt variants. A label absent from this dict logs under its own real
# config.strategy unchanged (e.g. Mean-Reversion Pairs -> "pairs").
SECONDARY_LOG_STRATEGY_OVERRIDES: dict[str, str] = {
    "RSI Mean-Reversion": "rsi_mean_reversion",
}

# Fixed candidate System_Config versions for EXPERIMENTAL strategies --
# deliberately a SEPARATE dict from SECONDARY_STRATEGY_VERSIONS above, not
# merged into it. This separation is the actual mechanism that keeps an
# experimental strategy out of ingest.py's automation loop (which only
# ever iterates SECONDARY_STRATEGY_VERSIONS) and out of any capital-
# allocation path -- see dip_buy_analyzer.py's "Daily Signals" tab and
# improvements.txt item 34/35. A strategy only moves from here to
# SECONDARY_STRATEGY_VERSIONS once it's been promoted after passing
# validation, the same graduation breakout_retest/week52_high went
# through -- squeeze_breakout (v39) graduated 2026-08-10 per explicit user
# request (improvements.txt item 43/44): the only Daily Signals candidate
# that beat its own random-entry baseline on ALL/TUNE/HOLDOUT, now made
# capital-eligible with its own secondary dashboard section.
#
# Momentum Burst (v38) and ADX Trend Entry (v40) removed 2026-08-11, same
# "remove all the losing strategies" request -- both lost to their own
# random-entry baseline on HOLDOUT in the real 5-year benchmark comparison
# run 2026-08-09/10 (momentum_burst additionally flat/negative on every
# cut). Empty until Momentum Rank (v65) below -- nothing else currently in
# the Daily Signals tab; the tab itself still renders regardless (just
# shows nothing experimental when this dict is empty), and the underlying
# strategy code/Mongo candidates of anything removed are untouched, only
# taken out of this dict.
#
# Momentum Rank (v65) added 2026-08-24, this project's first genuinely new
# strategy ARCHITECTURE (cross-sectional trailing-return ranking against the
# whole watchlist, not one ticker's own price history in isolation --
# improvements.txt). Cleared the real validation gate this same day: beats
# its own matched-random baseline on ticker-universe HOLDOUT (candidate
# sharpe_like 0.34 vs baseline 0.075) AND -- the decisive check added the
# same day specifically to catch payoff-bracket artifacts -- genuinely
# widens the REAL-vs-RANDOM gap (0.045) versus the untuned baseline's own
# gap (-0.005, i.e. DEFAULT_CONFIG alone shows no real timing edge over
# random). RRR-vs-scoring-ceiling check clears signal_buy_threshold=60, but
# only by 1.83 points (best-case Trade_Score=61.83) -- a thin buffer,
# meaning it will fire real Buy signals noticeably less often than its raw
# backtest trade count alone would suggest. Ships EXPERIMENTAL/tracked-only
# per this project's own graduated-promotion discipline -- no cash pool,
# no allocate_capital() call, same as every strategy's first live exposure.
EXPERIMENTAL_STRATEGY_VERSIONS = {
    "Momentum Rank": 65,
}


def load_active_config() -> tuple[swingtrade.TradingConfig, str]:
    try:
        doc = storage.get_active_config_doc()
    except storage.MongoNotConfigured:
        return swingtrade.DEFAULT_CONFIG, "MongoDB not configured -- using built-in defaults."
    except Exception as exc:
        return swingtrade.DEFAULT_CONFIG, f"Could not reach MongoDB ({exc}) -- using built-in defaults."

    if doc is None:
        return swingtrade.DEFAULT_CONFIG, "No active System_Config yet -- using built-in defaults."

    try:
        config = swingtrade.TradingConfig.from_dict(doc["params"])
    except Exception as exc:
        return swingtrade.DEFAULT_CONFIG, f"Active System_Config failed to parse ({exc}) -- using built-in defaults."

    return config, f"Using System_Config v{doc['version']} (active)."


def load_config_by_version(version: int) -> tuple[swingtrade.TradingConfig | None, str]:
    """Load one specific, fixed System_Config document by version number --
    for secondary/candidate strategies shown alongside the active config
    (see dip_buy_analyzer.py's breakout_retest/week52_high sections), NOT
    a substitute for load_active_config(). Unlike load_active_config(),
    which always falls back to a sensible default (no active config is a
    normal, expected state), a missing/unparseable SPECIFIC version is a
    genuine error worth surfacing distinctly -- silently substituting
    DEFAULT_CONFIG (strategy="rsi") for, say, a missing breakout_retest
    candidate would display a nonsensical "Breakout Retest" section
    actually running RSI logic. Returns `(None, reason)` on any failure;
    callers should skip rendering that section rather than fall back."""
    try:
        doc = storage.get_config_by_version(version)
    except storage.MongoNotConfigured:
        return None, "MongoDB not configured."
    except Exception as exc:
        return None, f"Could not reach MongoDB ({exc})."

    if doc is None:
        return None, f"No System_Config document with version={version}."

    try:
        config = swingtrade.TradingConfig.from_dict(doc["params"])
    except Exception as exc:
        return None, f"System_Config v{version} failed to parse ({exc})."

    return config, f"Using System_Config v{version} ({doc.get('status', 'unknown status')})."
