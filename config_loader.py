"""Load the active TradingConfig from MongoDB's System_Config, falling back
to swingtrade.DEFAULT_CONFIG when Mongo is unreachable, unconfigured, or
nothing has been promoted yet. Shared by the interactive dashboard
(dip_buy_analyzer.py, which wraps this in st.cache_data on a TTL) and the
standalone scheduled scan (ingest.py, which calls it once per run) so both
always agree on which config is "live" -- see ARCHITECTURE_PLAN.md Phase 6/7.
"""

import storage
import swingtrade

# Fixed, immutable candidate System_Config versions for the secondary
# strategies shown alongside the active config. Single source of truth shared
# by dip_buy_analyzer.py (dashboard sections) and ingest.py (headless scan) so
# the two can never drift apart on which versions are "the" secondaries.
#
# History (condensed 2026-10-22; full narrative in improvements.txt -- grep
# the item numbers, never read that file whole):
#   - Breakout Retest v27, 52-Week High v28: removed 2026-08-11 (edge vanished
#     once forced into a ceiling-clearing RRR; items 42/43).
#   - MA Crossover: v49 -> v51 -> v55 (items 51/59/72), promoted to PRIMARY
#     2026-08-16 (item 73, replaced breakout v43); the active primary is now
#     v71 and it FAILS the index gate (item: 2026-10-22 vs-index audit).
#   - RSI Mean-Reversion v17/v66: secondary 2026-08-20 (item 81), removed
#     2026-09-23 (decisively negative live, z=-4.58).
#   - Mean-Reversion Pairs v58: secondary 2026-08-21 (items 82/84), lean v1
#     chosen over Optuna-tuned v57 for its higher win rate; flagged failing
#     the fair validation bar 2026-09-27 (see SECONDARY_VALIDATION_CAVEATS).
#   - Squeeze Breakout v53: removed 2026-08-26 (portfolio simplification).
# Retired strategies stay in Mongo (status=candidate/retired), never deleted.
# Research-only steps the daily scan skips (soft retirement: code stays, flip a
# name out of this set to re-enable). Disabled 2026-10-22 on live evidence --
# best_ideas* (4 variants, ~1,000 settled trades: 9-19% win rate, avg pnl
# -0.2..-1.0%) and llm_agent* (3 variants, ~860 trades: 9-19% win rate, avg
# -0.5..-1.5%) were net negative while SPY rose, and the LLM calls consume API
# quota and daily-run time. The dashboard tabs can still run them on demand.
# NOTE: ma_crossover/pairs keep logging independently of this.
DISABLED_RESEARCH_STEPS = frozenset({"llm_agent", "best_ideas"})

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
    #
    # REMOVED 2026-09-23 (same "confirmed-negative live evidence -> full
    # removal" treatment squeeze_breakout got above, not a lesser
    # caveat-only response): a full audit found ic_tracking.ensemble_weight()
    # only ever measured rank-IC, never real expectancy -- this strategy
    # carried this project's dominant Best Ideas composite weight almost
    # entirely from a strong IC (+0.45) while confidently LOSING ~3%/trade
    # on 391+ real v66-era settled trades (both August and September, not a
    # one-off drawdown; z=-4.58 on ic_tracking.weighted_expectancy(), the
    # most decisively negative result of any strategy this project has
    # measured). Separately, its own original promotion permutation
    # significance (p=0.020) does not survive a fair Bonferroni correction
    # across the ~20 distinct strategy architectures this project has tried
    # (see benchmark_random_entry.bonferroni_check(),
    # validation_trials_registry.txt) -- so even the CLAIM this was
    # "validated" in the first place doesn't hold up, independent of the
    # live-loss finding. Code/historical trades untouched, just no longer
    # scanned/traded -- revisit only if a future retune clears BOTH a
    # Bonferroni-corrected significance bar AND a real, non-negative live
    # expectancy over ic_tracking.MINIMUM_PROSPECTIVE_DAYS of genuinely
    # prospective data, the same standard now required of every strategy.
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

# Per-label validation-status caveat, rendered as a warning banner wherever
# a secondary strategy's own dashboard section shows (dip_buy_analyzer.py's
# render_secondary_section()) -- deliberately NOT a capital-eligibility gate
# (per explicit user direction 2026-08-31: capital eligibility doesn't
# matter to how they actually trade, they read signals and decide
# themselves either way, same as the real ATD.TO trade that came from a
# never-capital-eligible Best Ideas signal). The thing that actually
# protects a discretionary decision is knowing a signal's real validation
# status before trusting it, not a backend allocation flag. A label absent
# from this dict has no known open caveat -- remove an entry once a
# strategy either gets re-validated for real or is retired outright (same
# "real negative IC -> full removal, not just a warning" treatment
# squeeze_breakout got, see best_ideas.METHODOLOGIES's own comment).
SECONDARY_VALIDATION_CAVEATS: dict[str, str] = {
    "Mean-Reversion Pairs": (
        "FLAGGED 2026-09-27, resuming the 2026-09-23 audit's re-evaluation loop: "
        "Mean-Reversion Pairs is now the ONLY strategy left in "
        "SECONDARY_STRATEGY_VERSIONS after RSI Mean-Reversion's removal below, but it "
        "has never actually cleared the fair validation bar now required of every "
        "strategy here. Its own permutation-test significance -- the same check "
        "ma_crossover (p=0.009) and RSI Mean-Reversion (p=0.020) at least passed "
        "uncorrected -- was p=0.382 at full 992-ticker smallmid-universe scale "
        "(2026-09-17), which fails even an UNCORRECTED p<0.05 bar, let alone the "
        "Bonferroni-corrected 0.0025 (benchmark_random_entry.bonferroni_check('pairs', "
        "0.382) confirms survives_uncorrected=False, survives_corrected=False against "
        "validation_trials_registry.txt's 20 tracked architectures). Real live IC "
        "(ic_tracking.methodology_report('pairs'), checked 2026-09-27, 38 settled / "
        "32.5 effective trades, trust floor met) has also turned NEGATIVE and is "
        "WORSENING: overall_ic=-0.179, IR=-0.377, first rolling window +0.151 -> most "
        "recent window -0.494. Honest caveat cutting the other way, unlike RSI "
        "Mean-Reversion's confidently-negative removal: weighted_expectancy is "
        "actually mildly POSITIVE (+0.94%/trade, z=0.95) -- not confidently losing "
        "money, just a genuinely inconclusive, statistically-thin sample, and the "
        "2026-09-13 slippage stress test found the real-vs-random GAP holds even at "
        "10x costs (its own ABSOLUTE sharpe_like is what goes negative under cost "
        "stress, a different, already-known caveat). NOT removed outright this pass "
        "(unlike squeeze_breakout/RSI Mean-Reversion's full removal) because "
        "regime_switcher.py's own REGIME_STRATEGY_PREFERENCE hard-codes "
        "'choppy': ['pairs'] as its only choppy-regime candidate -- removing pairs "
        "from SECONDARY_STRATEGY_VERSIONS would silently starve strategy_frames of a "
        "'pairs' key and regress the choppy-regime pick back to the exact dead/stale "
        "state the 2026-09-21 ADX-plumbing fix just resolved (see [[project_best_ideas_status]]). "
        "That cross-dependency needs its own explicit fix (e.g. a genuinely different "
        "choppy-regime candidate, or computing pairs' frame for regime purposes "
        "independent of capital-eligibility) before pairs can be safely retired the "
        "same way RSI Mean-Reversion was -- flagged as the clear next step, not done "
        "unilaterally in this pass given the real regression risk."
    ),
    "RSI Mean-Reversion": (
        "UPDATED 2026-09-20 (was stale and, it turns out, WRONG that the prior finding was "
        "'fixed historical fact that doesn't change with new data' -- it changed): a fresh full "
        "407-ticker benchmark_random_entry.py run (2019-01-01..2026-09-15, after the 2026-09-17 "
        "sanitize_ohlcv() data-quality fix was wired into the backtest path) now shows v66's entry "
        "timing REVERSING the 2026-08-31 failure -- TUNE sharpe_like 0.121 vs random's 0.096, HOLDOUT "
        "sharpe_like 0.103 vs random's 0.068 (TUNE/HOLDOUT agree), permutation test p=0.020 (observed "
        "gap unlikely under 'no real timing skill'), portfolio-constrained $10k replay ends at $22,432 "
        "(+124.3%) vs random's $14,262 (+42.6%). Edge holds in 6/8 calendar years and both volatility "
        "regimes. Two honest caveats that did NOT change: win_rate is still slightly below random's "
        "(15.8% vs 16.9% -- the edge is entirely a bigger-win/smaller-loss payoff-shape effect, not a "
        "higher hit rate), and backtest_ic (does the score rank which of ITS OWN fired trades do "
        "better) is ~0 (0.011) -- consistent with the project-wide finding that no strategy's backtest "
        "score has real trade-by-trade ranking power at scale. Also 93.8% of trades touch a same-ticker "
        "overlap (v66 fires repeatedly on the same names), higher concentration than ma_crossover's "
        "5.3% -- a real factor for position-sizing, not yet a disqualifier. "
        "UPDATED 2026-09-09: real settled-trade IC (does the LIVE trade_score rank realized outcomes "
        "across different signals/tickers -- a different question from backtest_ic above) strengthened "
        "to +0.605 over 255 settled / 134.9 effective trades, clearing ic_tracking.TRUST_FLOOR_TRADES; "
        "re-measured 2026-09-20 at +0.546 (IR=2.08). Net: both the entry-timing backtest AND the live "
        "ranking IC now independently support this as the strongest-evidence single strategy in the "
        "project; the fixed best_ideas composite (2026-09-20) also now weights it ~0.47, the dominant "
        "share, and beats trusting it alone on the same 63 real settled trades (composite IC 0.307 vs "
        "RSI-alone IC 0.242) -- so use the composite, not RSI alone, as the actionable signal."
    ),
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
    # Value Rank (Price-to-Book, cheapest decile) added 2026-09-21 -- the
    # first genuinely new strategy this cycle's continuous algorithm-
    # derivation loop actually PASSED (4 prior candidates -- Low-Volatility
    # Rank, discrete + rebalanced Sector Rotation, Quality Rank -- all hit
    # the TUNE-good/HOLDOUT-reverses disqualifying pattern). Full 407-ticker
    # validation: TUNE/HOLDOUT agree (no reversal), edge holds 6/8 calendar
    # years and both volatility regimes, portfolio-constrained $10k replay
    # +105% vs random's +62%. A real stock-split/book-value bug (SEC's
    # as-filed share count vs yfinance's split-adjusted price falsely
    # deflating Price-to-Book for any pre-split date, e.g. NVDA) was found
    # and fixed BEFORE this result was trusted -- see sec_fundamentals.py's
    # own top-of-file writeup. Same experimental/tracked-only staging as
    # every strategy here -- real settled-trade IC has to clear its own
    # trust floor over real weeks before any capital-eligibility question
    # is even asked. v76.
    #
    # CORRECTION (2026-09-23): the "permutation test p=0.0000, strongest
    # significance found anywhere in this project" claim above was WRONG --
    # a measurement artifact of permutation_test_gap()'s default
    # n_permutations=1000, which floors any p_value below 0.001 at exactly
    # 0.0000 rather than reporting how far below. Rerun at n_permutations=
    # 100000 (benchmark_random_entry.py --strategy value_rank
    # --n-permutations 100000): the real p_value is 0.0169, not ~0 -- and
    # backtest_ic is -0.009 (no real trade-by-trade ranking power, the
    # project-wide pattern, not an exception). Against
    # bonferroni_check()'s corrected bar (0.05/20 strategy architectures
    # ever tested = 0.0025), value_rank does NOT survive. Demoted from
    # this project's strongest result to noise-level evidence -- see
    # [[trading-strategy-status]]'s CURRENT STATE and
    # feedback_continuous_algorithm_derivation_loop for the fuller context.
    # Left in EXPERIMENTAL_STRATEGY_VERSIONS (not deleted) since it's
    # already tracked-only with zero capital exposure -- no real-money
    # consequence to correct, just a false claim about its evidence
    # strength.
    "Value Rank": 76,
    # Accruals Rank (Sloan 1996's classic accruals anomaly, buys the
    # lowest-accruals decile) added 2026-09-21 -- the 2nd new strategy this
    # cycle to actually PASS (after value_rank), and the cleanest result
    # found yet: TUNE/HOLDOUT agree, permutation p=0.0000, REAL beats
    # RANDOM in ALL 8/8 calendar years (the most consistent year-by-year
    # result of any strategy in this project), both volatility regimes,
    # LOWER max drawdown AND higher return than random, portfolio-
    # constrained $10k replay +280% vs random's +140% (the strongest
    # capital-realistic result of any strategy this session). Given the
    # project's own track record of strategy failures (and the real
    # value_rank stock-split bug found the same day), independently
    # re-verified before trusting: permutation math confirmed correct on
    # the SAME code that correctly returned p=0.846 for analyst_revision
    # hours earlier, look-ahead audit clean (0/100 real date-checks), and
    # the accruals ratio is scaled by total assets (never price or shares
    # outstanding) -- structurally immune to the stock-split confound
    # class of bug value_rank needed a real fix for. Same experimental/
    # tracked-only staging as every strategy here. v77.
    "Accruals Rank": 77,
}

# Small/Mid-Cap RSI / MA Crossover / Pairs variants RETIRED 2026-10-21 (live: 12.8%/5%/0-39%
# win rates, rsi_mean_reversion_smallmid 6.9% win / 58% expired over 693 actionable trades).
# Workflows, runners and dashboard sections deleted; history stays in Mongo.


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
