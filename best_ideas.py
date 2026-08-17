""""Best Ideas" -- a research-only meta-ensemble that manufactures ONE
recommendation per candidate ticker by blending every methodology this
codebase can bring to bear: the live mechanical strategies (ma_crossover,
squeeze_breakout), the regime-conditional switcher, an LLM second opinion
(llm_agent.py), a NEW cross-sectional sector-relative-strength momentum
score, a NEW rule-based qualitative/fundamentals composite, and a NEW LLM
meta-synthesis call that reasons explicitly across all of the above.

Full creative discretion, explicit scope decisions:

- Deliberately does NOT resurrect strategies this project's own validation
  pipeline already found weak/negative (rsi, pullback, breakout v43,
  breakout_retest, week52_high, momentum_burst, adx_trend_entry) as
  ensemble inputs -- diluting a "best possible call" system with
  already-disproven signals would be counter to the goal, not in service
  of it. Only currently-live/currently-trusted methodologies (ma_crossover,
  squeeze_breakout, regime_switcher, llm_agent) plus genuinely NEW ones
  built for this tab are blended.
- Every Best Ideas signal (the composite AND every sub-methodology built
  here) shares ONE ATR bracket (config.atr_take_profit_multiplier /
  config.stop_loss_atr_multiplier, the ACTIVE config's own values) rather
  than each methodology inventing its own payoff shape. This is
  deliberate, not laziness: it means every methodology's Information
  Coefficient (see ic_tracking.py) is measured under an IDENTICAL payoff
  structure, so differences in measured IC reflect real differences in
  ranking skill, not differences in bracket geometry -- the exact confound
  IC is being used here specifically to avoid (per the user's own
  explicit reasoning, and this project's own memory of being burned by
  it once already for breakout_retest/week52_high).
- Genuinely new methodologies here (best_ideas_sector_rs,
  best_ideas_qualitative, best_ideas_meta) are shipped WITHOUT first
  clearing benchmark_random_entry.py's usual 5-year backtest gate -- by
  explicit user instruction for this tab ("don't feel bound by... validate
  before build"), and following the one prior precedent for this exact
  posture in this codebase (regime_switcher.py): build it, log it
  honestly under its own strategy label, and let real settled trades and
  real IC/IR decide whether it earns a blend weight. Never capital-
  eligible regardless of how any of this looks -- Shares_To_Buy/Est_Cost
  are always 0, no cash pool, no allocate_capital() call, same hard rule
  as llm_agent.py/regime_switcher.py.
- The blend weight for every methodology is IC/IR-driven (see
  ic_tracking.ensemble_weight()), not sharpe_like/win_rate-driven, per the
  user's own explicit instruction and reasoning (IC isolates genuine
  ranking skill from payoff-structure confounds; see ic_tracking.py's
  module docstring for the fuller argument and the concrete prior incident
  that motivated it). Before a methodology clears its own 20-settled-trade
  trust floor, it contributes at a neutral, equal prior weight -- a brand
  new methodology (sector_rs, qualitative, meta on day one) is not
  silently zeroed out, but it's not blindly trusted either.
- The composite score ITSELF is also logged under its own strategy label
  ("best_ideas") and is therefore independently IC/IR-trackable over time
  -- the honest test of whether ensembling several methodologies actually
  beats trusting the single best one alone, not an assumption.

This module is intentionally storage/Streamlit-free (only pandas,
pandas_ta, yfinance, and this codebase's own swingtrade/market_data/
llm_agent/ic_tracking modules) -- run_best_ideas() returns plain row-dicts;
callers (dip_buy_analyzer.py's dashboard tab, ingest.py's headless
automation) are responsible for turning those into DataFrames and calling
storage.log_trade_signals(), same architectural split as regime_switcher.py
and llm_agent.py."""

import re
import time

import pandas as pd
import pandas_ta as ta
import yfinance as yf

import ic_tracking
import llm_agent
import market_data
import swingtrade
from swingtrade.levels import compute_relative_strength  # not re-exported at the
                                                            # swingtrade package level
                                                            # (swingtrade/__init__.py) --
                                                            # imported directly from its
                                                            # submodule instead of adding
                                                            # a new package-level export
                                                            # for a single new caller

MAX_BEST_IDEAS_CANDIDATES = 10  # same cost/rate-limit posture as
                                  # MAX_LLM_CANDIDATES elsewhere in this
                                  # codebase -- a Best Ideas candidate can
                                  # cost UP TO 2x the LLM calls of the
                                  # plain LLM Agent tab (one evaluate_ticker
                                  # call, plus one evaluate_meta_synthesis
                                  # call once >=2 methodologies have an
                                  # opinion), so this stays capped at the
                                  # same size rather than a larger one
REQUEST_DELAY_SEC = 0.5  # same pacing every other yfinance-calling loop in
                          # this codebase uses (market_data.py/optimize.py/
                          # the benchmark scripts) -- per-candidate here this
                          # loop can fire up to 6 yfinance calls (5 inside
                          # get_qualitative_snapshot + 1 fundamentals lookup)
                          # PLUS 2+ LLM calls, immediately after the main
                          # watchlist scan has already made 400+ yfinance
                          # calls -- unthrottled, this is exactly the
                          # condition that's caused real transient
                          # "possibly delisted" yfinance failures elsewhere
                          # in this project this session.

# Every methodology independently IC/IR-tracked and eligible to feed the
# composite blend -- see ic_tracking.methodology_report(). "best_ideas"
# itself (the composite/final call) is deliberately NOT in this list (it
# doesn't feed itself) but IS its own separately IC-trackable strategy
# label, so the ensembling step itself can be judged against its own best
# individual input over time, not just assumed to help.
METHODOLOGIES = [
    "ma_crossover", "squeeze_breakout", "regime_switcher", "llm_agent",
    "best_ideas_sector_rs", "best_ideas_qualitative", "best_ideas_meta",
]

# Composite/sub-methodology score -> Signal thresholds, all on Best Ideas'
# own 0-100 "ensemble conviction" scale -- distinct from (not directly
# comparable to) any individual TradingConfig's own RRR-weighted
# Trade_Score formula. 60 is reused as the Buy cutoff to loosely anchor to
# this project's existing signal_buy_threshold convention, but this is a
# genuinely different scale underneath.
BEST_IDEAS_STRONG_BUY_THRESHOLD = 75.0
BEST_IDEAS_BUY_THRESHOLD = 60.0
BEST_IDEAS_WATCH_THRESHOLD = 45.0


def best_ideas_signal(score: float | None) -> str:
    """Map a 0-100 Best Ideas conviction score to a Signal string, reusing
    this codebase's existing Strong Buy/Buy/Watch/Ignore vocabulary (so
    existing display/coloring/logging machinery just works). None (no
    score, nothing to say) maps to "Ignore"."""
    if score is None:
        return "Ignore"
    if score >= BEST_IDEAS_STRONG_BUY_THRESHOLD:
        return "Strong Buy"
    if score >= BEST_IDEAS_BUY_THRESHOLD:
        return "Buy"
    if score >= BEST_IDEAS_WATCH_THRESHOLD:
        return "Watch"
    return "Ignore"


def llm_bullishness_score(decision: str, confidence: float) -> float:
    """Map an LLM verdict (Buy/Hold/Avoid + 0-100 confidence, see
    llm_agent.py) onto the same 0-100 "bullishness" scale every other
    methodology's score already lives on, centered at 50 (neutral).
    "Buy" stretches upward toward 100 by confidence, "Avoid" stretches
    downward toward 0 by confidence, "Hold" stays flatly neutral (50)
    REGARDLESS of confidence -- a highly-confident "Hold" is still a
    genuinely uncertain/mixed call, not a disguised lean either way."""
    if decision == "Buy":
        return 50.0 + confidence / 2.0
    if decision == "Avoid":
        return 50.0 - confidence / 2.0
    return 50.0


def compute_sector_rs_scores(
    bundle: dict[str, dict], sector_data: dict[str, pd.DataFrame],
    sector_lookup: dict[str, str], lookback_days: int,
) -> dict[str, dict]:
    """A NEW methodology for this tab: cross-sectional sector-relative-
    strength MOMENTUM. Reuses swingtrade.compute_relative_strength() --
    already a live, dashboard-safe, non-backtest-only primitive (fed
    sector OHLCV in place of market OHLCV; see market_data.fetch_ticker_
    bundle()'s own sector_data fetch) -- but turns its raw additive
    return-difference into a 0-100 CROSS-SECTIONAL PERCENTILE RANK across
    every ticker in the bundle that has one. Unlike every Trade_Score-based
    methodology here, this one has an opinion on every ticker with
    sufficient history (no discrete "fired"/"Ignore" trigger) since
    relative strength is a continuous ranking, not a discrete event.

    Returns {ticker: {"relative_strength": float, "percentile": float}}
    for every ticker with both its own OHLCV and its sector's OHLCV
    available -- silently excluded otherwise (no sector mapping, no sector
    ETF data, or insufficient history), same as
    compute_relative_strength()'s own None-return convention."""
    raw: dict[str, float] = {}
    for ticker, entry in bundle.items():
        sector = sector_lookup.get(ticker)
        sector_df = sector_data.get(sector) if sector else None
        if sector_df is None:
            continue
        rs = compute_relative_strength(entry["df"], sector_df, lookback_days)
        if rs is not None:
            raw[ticker] = rs
    if not raw:
        return {}
    series = pd.Series(raw)
    percentiles = series.rank(pct=True) * 100
    return {
        ticker: {"relative_strength": round(float(rs), 4), "percentile": round(float(percentiles[ticker]), 2)}
        for ticker, rs in raw.items()
    }


_ANALYST_TREND_RE = re.compile(
    r"strongBuy=(?P<strongBuy>\d+|None), buy=(?P<buy>\d+|None), hold=(?P<hold>\d+|None), "
    r"sell=(?P<sell>\d+|None), strongSell=(?P<strongSell>\d+|None)"
)


def _parse_trend_entry(entry: str) -> float | None:
    """Extract a single "bullish net" score from one of market_data.
    _get_analyst_data()'s own formatted trend strings -- (strongBuy*2 + buy)
    minus (strongSell*2 + sell), a simple weighted net-bullish count.
    Returns None if the string doesn't match the expected format at all."""
    match = _ANALYST_TREND_RE.search(entry)
    if not match:
        return None
    values = {k: (0 if v == "None" else int(v)) for k, v in match.groupdict().items()}
    return (values["strongBuy"] * 2 + values["buy"]) - (values["strongSell"] * 2 + values["sell"])


def qualitative_composite_score(qualitative: dict | None) -> dict | None:
    """A NEW methodology for this tab: a rule-based 0-100 conviction score
    built PURELY from market_data.get_qualitative_snapshot()'s
    fundamentals/ownership/positioning data -- genuinely new in kind, not
    just parameters, distinct from every price-action-based signal
    (mechanical strategies, sector relative strength) AND from the LLM's
    own holistic judgment (this is a fixed, auditable rule set evaluated
    the same way every time, not a model call).

    Starts at 50 (neutral) and applies bounded +/- adjustments per
    sub-signal actually present; a sub-signal that's None/unavailable
    contributes nothing (never penalized for being missing, same "cleanly
    omit, don't guess" philosophy as llm_agent._build_qualitative_block()).
    Returns None (not 50.0) if EVERY sub-signal is unavailable or neutral
    -- "neutral because the real evidence is genuinely balanced" and
    "neutral because there's no evidence at all" are different claims, and
    only the first should count as this methodology having an opinion.

    Returns {"score": float, "breakdown": list[str]} -- breakdown is a
    human-readable list of which adjustments actually fired, for
    display/audit (see dip_buy_analyzer.py's Best Ideas tab)."""
    if not qualitative:
        return None
    score = 50.0
    breakdown: list[str] = []
    any_signal = False

    analyst = qualitative.get("analyst")
    if analyst:
        trend = analyst.get("trend") or []
        if len(trend) >= 2:
            first_net = _parse_trend_entry(trend[0])
            last_net = _parse_trend_entry(trend[-1])
            if first_net is not None and last_net is not None and first_net != last_net:
                any_signal = True
                if last_net > first_net:
                    score += 8
                    breakdown.append("Analyst trend improving (+8)")
                else:
                    score -= 8
                    breakdown.append("Analyst trend deteriorating (-8)")
        recent_actions = analyst.get("recent_actions") or []
        upgrades = sum(1 for a in recent_actions if "(up)" in a.lower())
        downgrades = sum(1 for a in recent_actions if "(down)" in a.lower())
        if upgrades or downgrades:
            any_signal = True
            delta = min(upgrades, 2) * 5 - min(downgrades, 2) * 5
            score += delta
            breakdown.append(f"Recent analyst actions: {upgrades} upgrade(s), {downgrades} downgrade(s) ({delta:+.0f})")

    insider = qualitative.get("insider")
    if insider and insider.get("net_direction") in ("Buying", "Selling"):
        any_signal = True
        if insider["net_direction"] == "Buying":
            score += 10
            breakdown.append("Insider net buying (+10)")
        else:
            score -= 10
            breakdown.append("Insider net selling (-10)")

    short_interest = qualitative.get("short_interest")
    if short_interest and short_interest.get("trend") in ("Increasing", "Decreasing"):
        any_signal = True
        if short_interest["trend"] == "Decreasing":
            score += 8
            breakdown.append("Short interest decreasing (+8)")
        else:
            score -= 8
            breakdown.append("Short interest increasing (-8)")

    options = qualitative.get("options")
    if options and options.get("put_call_volume_ratio") is not None:
        ratio = options["put_call_volume_ratio"]
        if ratio <= 0.7:
            any_signal = True
            score += 6
            breakdown.append(f"Call-heavy options positioning (put/call={ratio:.2f}, +6)")
        elif ratio >= 1.3:
            any_signal = True
            score -= 6
            breakdown.append(f"Put-heavy options positioning (put/call={ratio:.2f}, -6)")

    if not any_signal:
        return None
    return {"score": round(max(0.0, min(100.0, score)), 2), "breakdown": breakdown}


def _fallback_atr(df: pd.DataFrame, config: swingtrade.TradingConfig) -> float | None:
    """ATR (config.atr_window) computed directly for a candidate that
    reached the Best Ideas universe PURELY via sector_rs -- no mechanical
    strategy fired for it today, so there's no existing levels-row to read
    an ATR column from. Same pandas_ta call every compute_*_levels()
    function in swingtrade/levels.py already makes internally, just
    standalone here since there's no full frame to piggyback on for this
    one ticker/day. Returns None (never a fabricated number) if there's
    insufficient history."""
    try:
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=config.atr_window)
        atr = atr_series.iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None


def gather_candidate_universe(
    strategy_frames: dict[str, pd.DataFrame], regime_picks: list[dict],
    sector_rs_scores: dict[str, dict], max_candidates: int = MAX_BEST_IDEAS_CANDIDATES,
) -> list[str]:
    """Union of every ticker any live mechanical strategy or the regime
    switcher flagged above Ignore today, PLUS a small allowance of
    sector_rs-only ideas (top decile relative strength, not already
    present) so that pure cross-sectional momentum can occasionally
    surface a candidate the trigger-based methodologies missed entirely --
    genuinely new ideas, not just re-ranking ones already found. Ranked by
    best available raw score, capped at `max_candidates` for cost control
    (LLM calls, extra yfinance fetches), same posture as MAX_LLM_CANDIDATES
    elsewhere in this codebase."""
    best_score: dict[str, float] = {}
    for df in strategy_frames.values():
        if df is None or df.empty:
            continue
        for _, row in df[df["Signal"] != "Ignore"].iterrows():
            ticker = row["Ticker"]
            best_score[ticker] = max(best_score.get(ticker, 0.0), float(row["Trade_Score"]))
    for pick in regime_picks:
        ticker = pick["Ticker"]
        best_score[ticker] = max(best_score.get(ticker, 0.0), float(pick["Trade_Score"]))

    sector_only = sorted(
        ((t, v["percentile"]) for t, v in sector_rs_scores.items() if t not in best_score and v["percentile"] >= 90),
        key=lambda item: item[1], reverse=True,
    )
    remaining_slots = max(0, max_candidates - len(best_score))
    for ticker, percentile in sector_only[:remaining_slots]:
        best_score[ticker] = percentile

    ranked = sorted(best_score.items(), key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in ranked[:max_candidates]]


def blend_composite(scores: dict[str, float], weights: dict[str, float]) -> tuple[float | None, dict]:
    """IC/IR-weighted blend of whichever methodologies have a score for one
    ticker today (`scores`: {methodology_name: 0-100 score, ...} --
    methodologies with no opinion simply aren't keys, not zeros). `weights`
    comes from ic_tracking.ensemble_weight() per methodology.

    If every contributing methodology currently has weight <= 0 (either
    none have cleared their trust floor and something set an explicit 0
    prior, or -- the real case this guards -- every trust-floor-cleared
    methodology has shown IR <= 0 so far), falls back to equal weighting
    among whatever fired rather than dividing by zero: still genuinely
    informative (this system currently has no demonstrated edge from any
    contributing methodology, which is itself worth surfacing, not hiding
    behind a crash or a silently-empty result).

    Returns (composite_score, breakdown) where breakdown is
    {methodology_name: {"score":, "weight":}} (weight normalized to sum to
    1 across contributors) -- (None, {}) if `scores` is empty."""
    contributions = {name: (score, weights.get(name, 1.0)) for name, score in scores.items() if score is not None}
    if not contributions:
        return None, {}
    total_weight = sum(w for _, w in contributions.values())
    if total_weight <= 0:
        contributions = {name: (score, 1.0) for name, (score, _) in contributions.items()}
        total_weight = len(contributions)
    composite = sum(score * w for score, w in contributions.values()) / total_weight
    breakdown = {
        name: {"score": round(score, 2), "weight": round(w / total_weight, 4)}
        for name, (score, w) in contributions.items()
    }
    return round(composite, 2), breakdown


def run_best_ideas(
    strategy_frames: dict[str, pd.DataFrame],
    regime_picks: list[dict],
    bundle: dict[str, dict],
    sector_data: dict[str, pd.DataFrame],
    sector_lookup: dict[str, str],
    config: swingtrade.TradingConfig,
    ic_reports: dict[str, dict],
    macro_snapshot: dict | None = None,
    max_candidates: int = MAX_BEST_IDEAS_CANDIDATES,
    fetch_llm: bool = True,
) -> dict[str, list[dict]]:
    """The full Best Ideas pipeline for one scan: gather a candidate
    universe, score every available methodology for each candidate, blend
    them into one composite recommendation via IC/IR-weighted averaging,
    and (when >=2 methodologies actually have an opinion) run a genuine
    LLM meta-synthesis across them. NEVER capital-eligible -- every
    returned row has Shares_To_Buy=Est_Cost=0.0, and this function never
    imports/calls swingtrade.allocate_capital.

    `strategy_frames`: {"ma_crossover": df, "squeeze_breakout": df, ...},
    keyed by the REAL strategy identifier, same convention as
    ingest.py's own strategy_frames / regime_switcher.py's
    strategy_rows -- already-scored today's results for each live
    mechanical strategy.
    `regime_picks`: the raw list of dicts from
    regime_switcher.select_regime_pick() (NOT the reduced row-dicts
    ingest.py logs -- the full picks, so this function can read
    Last_Close/ATR/etc off them if needed).
    `ic_reports`: {methodology_name: ic_tracking.methodology_report(...)}
    precomputed ONCE by the caller for every name in METHODOLOGIES (a
    handful of Mongo queries -- deliberately not recomputed per ticker).
    `fetch_llm`: set False to skip every LLM/yfinance-fundamentals call
    entirely (still computes the mechanical/regime/sector_rs/qualitative-
    free composite) -- useful for tests or a llm_agent-unavailable
    environment; the sector_rs/qualitative layers still run regardless
    (qualitative needs its own yfinance fetch, gated by this flag too,
    since it's part of the same "extra network cost" budget as the LLM
    calls).

    Returns {"best_ideas": [...], "best_ideas_sector_rs": [...],
    "best_ideas_qualitative": [...], "best_ideas_meta": [...]} -- each a
    list of row-dicts in this codebase's standard Trade_Signals row shape
    (Ticker/As_Of/Signal/Trade_Score/Buy_Price/Sell_Price/Stop_Loss/RRR/
    RSI/ATR/Distance_to_Buy_Pct=0.0/Shares_To_Buy=0.0/Est_Cost=0.0/
    Next_Earnings_Date/Catalyst_Warning/Top_Headline/Currency), ready for
    `pd.DataFrame(rows)` + `storage.log_trade_signals(df, cfg.to_dict())`
    per label. The "best_ideas" rows additionally carry "Rationale" and
    "Methodology_Breakdown" (see storage/signals.py)."""
    weights = {
        name: ic_tracking.ensemble_weight(ic_reports.get(name, {"trust_floor_met": False, "ir": None}))
        for name in METHODOLOGIES
    }

    sector_rs_scores = compute_sector_rs_scores(
        bundle, sector_data, sector_lookup, config.sector_relative_strength_lookback_days
    )
    candidates = gather_candidate_universe(strategy_frames, regime_picks, sector_rs_scores, max_candidates)

    mechanical_rows: dict[str, dict[str, dict]] = {}
    for strategy_name, df in strategy_frames.items():
        if df is None or df.empty:
            continue
        for _, row in df[df["Signal"] != "Ignore"].iterrows():
            mechanical_rows.setdefault(row["Ticker"], {})[strategy_name] = row.to_dict()
    regime_by_ticker = {pick["Ticker"]: pick for pick in regime_picks}

    composite_rows: list[dict] = []
    sector_rs_rows: list[dict] = []
    qualitative_rows: list[dict] = []
    meta_rows: list[dict] = []

    for i, ticker in enumerate(candidates):
        if fetch_llm and i > 0:
            # Paced only when this iteration will actually make network
            # calls (fetch_llm=False means everything below is already-
            # computed mechanical/regime/sector_rs data, no new fetches) --
            # see REQUEST_DELAY_SEC's own comment for why this matters.
            time.sleep(REQUEST_DELAY_SEC)

        ticker_mech = mechanical_rows.get(ticker, {})
        any_mech_row = next(iter(ticker_mech.values()), None)

        if any_mech_row is not None:
            last_close = float(any_mech_row["Last_Close"])
            atr = float(any_mech_row["ATR"]) if pd.notna(any_mech_row.get("ATR")) else None
            rsi = any_mech_row.get("RSI")
            currency = any_mech_row.get("Currency", "USD")
            next_earnings = any_mech_row.get("Next_Earnings_Date")
            catalyst_warning = bool(any_mech_row.get("Catalyst_Warning", False))
            top_headline = any_mech_row.get("Top_Headline", "")
            as_of = any_mech_row.get("As_Of")
        else:
            df = bundle[ticker]["df"]
            last_close = float(df["Close"].iloc[-1])
            atr = _fallback_atr(df, config)
            rsi = None
            currency = bundle[ticker].get("currency", "USD")
            next_earnings = bundle[ticker].get("next_earnings")
            catalyst_warning = False
            top_headline = bundle[ticker].get("top_headline", "")
            as_of = str(df.index[-1].date())

        if atr is None:
            continue  # can't build a Buy/Sell/Stop bracket without it -- skip, don't fabricate

        available_scores: dict[str, float] = {}
        if "ma_crossover" in ticker_mech:
            available_scores["ma_crossover"] = float(ticker_mech["ma_crossover"]["Trade_Score"])
        if "squeeze_breakout" in ticker_mech:
            available_scores["squeeze_breakout"] = float(ticker_mech["squeeze_breakout"]["Trade_Score"])
        regime_pick = regime_by_ticker.get(ticker)
        if regime_pick is not None:
            available_scores["regime_switcher"] = float(regime_pick["Trade_Score"])
        sector_rs_entry = sector_rs_scores.get(ticker)
        if sector_rs_entry is not None:
            available_scores["best_ideas_sector_rs"] = sector_rs_entry["percentile"]

        qualitative = market_data.get_qualitative_snapshot(ticker) if fetch_llm else None
        qual_result = qualitative_composite_score(qualitative)
        if qual_result is not None:
            available_scores["best_ideas_qualitative"] = qual_result["score"]

        llm_verdict = None
        if fetch_llm and llm_agent.is_available():
            fundamentals = {}
            try:
                info = yf.Ticker(ticker).info
                fundamentals = {k: info[k] for k in ("trailingPE", "marketCap", "sector") if info.get(k) is not None}
            except Exception:
                pass
            llm_context = {
                "last_close": last_close, "rsi": rsi, "atr": atr,
                "mechanical_scores": {name: row["Trade_Score"] for name, row in ticker_mech.items()},
                "catalyst_warning": catalyst_warning, "next_earnings_date": next_earnings,
                "headlines": market_data.get_multi_headlines(ticker), "fundamentals": fundamentals,
                "macro": macro_snapshot, "qualitative": qualitative,
            }
            llm_verdict = llm_agent.evaluate_ticker(ticker, llm_context)
            if llm_verdict is not None:
                available_scores["llm_agent"] = llm_bullishness_score(llm_verdict["decision"], llm_verdict["confidence"])

        meta_verdict = None
        if fetch_llm and llm_agent.is_available() and len(available_scores) >= 2:
            methodology_summaries = []
            for name, score in available_scores.items():
                report = ic_reports.get(name, {})
                methodology_summaries.append({
                    "name": name, "signal": best_ideas_signal(score), "score": score,
                    "overall_ic": report.get("overall_ic"), "ir": report.get("ir"),
                    "n_settled": report.get("n_settled", 0), "trust_floor_met": report.get("trust_floor_met", False),
                })
            meta_verdict = llm_agent.evaluate_meta_synthesis(
                ticker, {"last_close": last_close, "methodologies": methodology_summaries}
            )
            if meta_verdict is not None:
                available_scores["best_ideas_meta"] = llm_bullishness_score(
                    meta_verdict["decision"], meta_verdict["confidence"]
                )

        composite_score, breakdown = blend_composite(available_scores, weights)
        if composite_score is None:
            continue

        sell_price = round(last_close + config.atr_take_profit_multiplier * atr, 2)
        stop_loss = round(last_close - config.stop_loss_atr_multiplier * atr, 2)
        risk = last_close - stop_loss
        rrr = round((sell_price - last_close) / risk, 2) if risk > 0 else 0.0
        base_row = {
            "Ticker": ticker, "As_Of": as_of, "Last_Close": round(last_close, 2),
            "Buy_Price": round(last_close, 2), "Sell_Price": sell_price, "Stop_Loss": stop_loss,
            "RRR": rrr, "RSI": rsi, "ATR": atr, "Distance_to_Buy_Pct": 0.0,
            "Shares_To_Buy": 0.0, "Est_Cost": 0.0, "Next_Earnings_Date": next_earnings,
            "Catalyst_Warning": catalyst_warning, "Top_Headline": top_headline, "Currency": currency,
        }

        composite_signal = best_ideas_signal(composite_score)
        if composite_signal != "Ignore":
            if meta_verdict is not None:
                rationale = meta_verdict["rationale"]
            elif llm_verdict is not None:
                rationale = llm_verdict["rationale"]
            else:
                rationale = (
                    f"Ensembled from {len(available_scores)} "
                    f"methodolog{'y' if len(available_scores) == 1 else 'ies'}: "
                    + ", ".join(f"{name} ({info['score']:.0f}, weight {info['weight']:.0%})"
                                for name, info in breakdown.items())
                )
            composite_rows.append({
                **base_row, "Signal": composite_signal, "Trade_Score": composite_score,
                "Rationale": rationale, "Methodology_Breakdown": breakdown,
            })

        if sector_rs_entry is not None:
            sig = best_ideas_signal(sector_rs_entry["percentile"])
            if sig != "Ignore":
                sector_rs_rows.append({**base_row, "Signal": sig, "Trade_Score": sector_rs_entry["percentile"]})

        if qual_result is not None:
            sig = best_ideas_signal(qual_result["score"])
            if sig != "Ignore":
                qualitative_rows.append({**base_row, "Signal": sig, "Trade_Score": qual_result["score"]})

        if meta_verdict is not None:
            meta_score = available_scores.get("best_ideas_meta")
            sig = best_ideas_signal(meta_score)
            if sig != "Ignore":
                meta_rows.append({**base_row, "Signal": sig, "Trade_Score": meta_score})

    return {
        "best_ideas": composite_rows,
        "best_ideas_sector_rs": sector_rs_rows,
        "best_ideas_qualitative": qualitative_rows,
        "best_ideas_meta": meta_rows,
    }
