"""
Standalone scheduled scan + signal-logging service (Phase 7).

Runs the exact same fetch -> compute -> score pipeline as the Streamlit
dashboard (dip_buy_analyzer.py), via the shared `market_data`/`swingtrade`/
`config_loader` modules, but as a one-shot headless process instead of
something that only runs when a human happens to have the dashboard open.
This is the concrete gap it closes: previously, Trade_Signals only
accumulated on days someone opened the dashboard and it happened to compute
a Buy/Strong Buy -- starving the settlement job (Phase 3) and the learning
loop (Phase 5) of data whenever nobody looked. Running this on a schedule
(cron locally, GitHub Actions' daily_run.yml in production) makes signal
generation independent of the UI.

Scans the active System_Config (config_loader.load_active_config()) plus
every strategy in config_loader.SECONDARY_STRATEGY_VERSIONS (currently
squeeze_breakout v39 and ma_crossover v49 -- see improvements.txt; this
dict is the single shared source of truth with dip_buy_analyzer.py, so
whichever strategies are capital-eligible on the dashboard are exactly
the ones scanned here too), all against ONE shared fetch of the
watchlist's OHLCV data (market_data.fetch_ticker_bundle()) so running N
strategies costs one network round-trip, not N. A secondary strategy that
fails to load (e.g. its System_Config document is ever deleted) is
skipped with a warning, not a fatal error -- the primary scan still runs
regardless.

Also runs the LLM Agent (see llm_agent.py) if GEMINI_API_KEY/GROQ_API_KEY
are configured -- previously dashboard-only, meaning it only evaluated
and logged a signal on days someone happened to open the dashboard,
starving its own prospective validation trust floor (20-30 settled
trades, 4-6 weeks) of data. Same candidate-selection logic as the
dashboard's "LLM Agent" tab (tickers any mechanical strategy flagged
above Ignore, capped at MAX_LLM_CANDIDATES, ranked by Trade_Score) --
never capital-allocated, same as the dashboard.

Not rewritten in Go (amendment #1) -- see market_data.py's docstring.

Position sizing here uses --position-budget (default matches the dashboard's
default) purely to populate the shares_to_buy/est_cost fields Trade_Signals
expects, for all three strategies alike. This script does no capital
allocation across signals -- greedily spending a cash pool across today's
top signals is a personal, interactive decision that belongs in the
dashboard, not something to bake into a scheduled job or log as if it were
part of the technical signal.

Usage:
    python ingest.py
    python ingest.py --watchlist my_list.txt --position-budget 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

import ai_context
import best_ideas
import config_loader
import ic_tracking
import llm_agent
import market_data
import notifications
import regime_switcher
import storage
import swingtrade
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
DEFAULT_POSITION_BUDGET = 250.0
MAX_LLM_CANDIDATES = 10  # same cap as dip_buy_analyzer.py's LLM Agent tab -- cost/rate-limit
                          # control, see llm_agent.py's module docstring
CHALLENGER_VARIANTS = ["qualitative_weighted", "evidence_strict"]  # run alongside the
                          # default "balanced" variant for every candidate -- see
                          # llm_agent.PROMPT_VARIANTS and run_llm_agent()'s docstring


def _score_for_strategy(df: pd.DataFrame, config: swingtrade.TradingConfig) -> pd.DataFrame:
    """Dispatch to the right add_*_trade_score() for config.strategy --
    covers all eight strategies, kept byte-for-byte in sync with
    dip_buy_analyzer.py's own _score_for_strategy() (a real, previously
    undetected gap: this copy was last updated when breakout_retest/
    week52_high were the only secondaries, and was never extended when
    squeeze_breakout/ma_crossover were promoted into
    config_loader.SECONDARY_STRATEGY_VERSIONS -- any day either strategy
    had a real signal, run_secondary_strategy() would crash with
    KeyError: 'Oversold_Streak_Days' before ever reaching MongoDB, since
    the old fallback (swingtrade.add_trade_score(), the RSI scorer) expects
    columns those strategies' levels dicts don't carry. See improvements.txt."""
    if config.strategy == "breakout":
        return swingtrade.add_breakout_trade_score(df, config)
    elif config.strategy == "pullback":
        return swingtrade.add_pullback_trade_score(df, config)
    elif config.strategy == "breakout_retest":
        return swingtrade.add_breakout_retest_trade_score(df, config)
    elif config.strategy == "week52_high":
        return swingtrade.add_week52_trade_score(df, config)
    elif config.strategy == "momentum_burst":
        return swingtrade.add_momentum_burst_trade_score(df, config)
    elif config.strategy == "squeeze_breakout":
        return swingtrade.add_squeeze_breakout_trade_score(df, config)
    elif config.strategy == "adx_trend_entry":
        return swingtrade.add_adx_trend_entry_trade_score(df, config)
    elif config.strategy == "ma_crossover":
        return swingtrade.add_ma_crossover_trade_score(df, config)
    elif config.strategy == "pairs":
        return swingtrade.add_pairs_trade_score(df, config)
    else:
        return swingtrade.add_trade_score(df, config)


def _size_positions(results_df: pd.DataFrame, config: swingtrade.TradingConfig,
                     position_budget: float, risk_amount: float | None) -> None:
    """Populate Shares_To_Buy/Est_Cost in place -- shared sizing logic for
    every strategy scanned this run."""
    if risk_amount:
        results_df["Shares_To_Buy"] = swingtrade.size_by_risk(
            results_df["Buy_Price"], results_df["Stop_Loss"], risk_amount, config.fractional_share_decimals
        )
    else:
        results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(config.fractional_share_decimals)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)


def _build_signal_notification(label: str, results_df: pd.DataFrame) -> str | None:
    """Short Discord message listing today's Strong Buy/Buy tickers for one
    strategy, sorted by Trade_Score descending. Returns None if there are
    none -- callers skip notifying entirely on quiet days (confirmed
    cadence: signal channels fire only when there's a real signal, unlike
    the always-fires daily run-status report)."""
    signal_rows = results_df[results_df["Signal"].isin(["Strong Buy", "Buy"])]
    if signal_rows.empty:
        return None
    signal_rows = signal_rows.sort_values("Trade_Score", ascending=False)
    lines = [f"**{label}: {len(signal_rows)} signal(s)**"]
    for _, row in signal_rows.iterrows():
        lines.append(f"{row['Ticker']}: {row['Signal']} (score {row['Trade_Score']:.0f}, buy {row['Buy_Price']:.2f})")
    return "\n".join(lines)


def run_secondary_strategy(
    label: str,
    config: swingtrade.TradingConfig,
    bundle: dict,
    market_df,
    position_budget: float,
    risk_amount: float | None,
    sector_lookup: dict[str, str] | None = None,
    sector_data: dict | None = None,
    log_strategy_override: str | None = None,
    pair_price_panels: dict | None = None,
) -> pd.DataFrame:
    """Score, size, and log one secondary strategy's results against the
    SAME already-fetched bundle the primary scan used -- no extra network
    fetch (see market_data.fetch_ticker_bundle()). Mirrors
    dip_buy_analyzer.py's render_secondary_section(), adapted for
    print-based CLI output. A leaner treatment than the primary scan below
    (no fatal error on empty results, no research_loosened logging) since
    these are the newly-automated secondary strategies -- see
    improvements.txt. Returns the scored results_df (empty if nothing was
    analyzed) so --with-ai-context can use it.

    `log_strategy_override`, if set, logs Trade_Signals/Trade_Outcomes under
    a DIFFERENT strategy label than `config.strategy` -- see
    config_loader.SECONDARY_LOG_STRATEGY_OVERRIDES's own docstring and
    render_secondary_section()'s matching param (same mechanism, kept in
    sync so dashboard-triggered and automated runs log identically)."""
    results, score_skipped = market_data.score_bundle_for_strategy(
        bundle, market_df, config, sector_lookup=sector_lookup, sector_data=sector_data,
        pair_price_panels=pair_price_panels,
    )
    if not results:
        print(f"{label}: no tickers were successfully analyzed.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    _size_positions(results_df, config, position_budget, risk_amount)
    results_df = _score_for_strategy(results_df, config)

    log_config = (
        swingtrade.TradingConfig(**{**config.to_dict(), "strategy": log_strategy_override})
        if log_strategy_override else config
    )
    logged = storage.log_trade_signals(results_df, log_config.to_dict())
    strong_buys = int((results_df["Signal"] == "Strong Buy").sum())
    buys = int((results_df["Signal"] == "Buy").sum())
    watches = int((results_df["Signal"] == "Watch").sum())
    print(f"{label}: analyzed {len(results_df)} ticker(s): {strong_buys} Strong Buy, {buys} Buy, {watches} Watch.")
    print(f"{label}: logged {logged['actionable']} actionable + {logged['research']} research signal(s) to MongoDB.")

    message = _build_signal_notification(label, results_df)
    if message:
        notifications.notify(message, webhook_url=notifications.get_strategy_webhook_url(config.strategy))

    if score_skipped:
        print(f"{label}: {len(score_skipped)} ticker(s) could not be scored for this strategy:")
        for ticker, reason in score_skipped:
            print(f"  {ticker}: {reason}")

    return results_df


def run_llm_agent(all_signal_frames: dict[str, pd.DataFrame], config: swingtrade.TradingConfig) -> None:
    """Headless counterpart to dip_buy_analyzer.py's "LLM Agent" tab --
    same candidate-selection logic (tickers any mechanical strategy
    flagged above Ignore today, capped at MAX_LLM_CANDIDATES, ranked by
    highest Trade_Score), same context-building (fundamentals, headlines,
    macro snapshot, qualitative snapshot -- see market_data.get_qualitative_snapshot()),
    same Trade_Signals logging convention ("Hold" mapped to "Watch" since it
    isn't in the shared Strong Buy/Buy/Watch/Ignore vocabulary). NEVER
    capital-allocated -- Shares_To_Buy/Est_Cost are always 0, no cash pool,
    no allocate_capital() call, same as the dashboard tab.

    Unlike the dashboard tab, evaluates EVERY candidate under all of
    llm_agent.PROMPT_VARIANTS (default "balanced" plus CHALLENGER_VARIANTS)
    for prospective A/B comparison -- context is built once per candidate
    and reused across variants (only the prompt framing differs, not the
    underlying data), so this doesn't add extra fetch cost, only extra LLM
    calls (well within free-tier limits, see llm_agent.py's module
    docstring). Each variant logs to its own strategy string (see
    llm_agent.variant_strategy_name()) and settles completely independently.

    Silently does nothing if llm_agent.is_available() is False (no keys
    configured) -- callers should still call this unconditionally, same
    "degrade quietly" philosophy as everything else optional in this
    pipeline."""
    if not llm_agent.is_available():
        print("LLM Agent: unavailable (set GEMINI_API_KEY and/or GROQ_API_KEY to enable).")
        return

    candidates: dict[str, dict] = {}
    for strategy_label, df in all_signal_frames.items():
        if df is None or df.empty:
            continue
        interesting = df[df["Signal"] != "Ignore"]
        for _, row in interesting.iterrows():
            entry = candidates.setdefault(row["Ticker"], {"scores": {}, "row": row})
            entry["scores"][strategy_label] = float(row["Trade_Score"])

    ranked_candidates = sorted(
        candidates.items(), key=lambda item: max(item[1]["scores"].values()), reverse=True
    )[:MAX_LLM_CANDIDATES]

    if not ranked_candidates:
        print("LLM Agent: no tickers scored above Ignore under any mechanical strategy today -- nothing to evaluate.")
        return

    all_variants = ["balanced"] + CHALLENGER_VARIANTS
    print(f"LLM Agent: evaluating {len(ranked_candidates)} ticker(s) flagged by at least one "
          f"mechanical strategy today, highest mechanical Trade_Score first, under {len(all_variants)} "
          f"prompt variant(s) ({', '.join(all_variants)}).")
    macro_snapshot = market_data.get_macro_snapshot()

    # One row-list per variant -- each logs to its own strategy string and
    # settles independently. Context is built ONCE per candidate (below) and
    # reused across variants; only the prompt framing passed to
    # evaluate_ticker() differs, so this adds no extra fetch cost.
    llm_rows_by_variant: dict[str, list] = {v: [] for v in all_variants}

    for ticker, entry in ranked_candidates:
        row = entry["row"]
        fundamentals = {}
        try:
            info = yf.Ticker(ticker).info
            fundamentals = {k: info[k] for k in ("trailingPE", "marketCap", "sector") if info.get(k) is not None}
        except Exception:
            pass
        context = {
            "last_close": float(row["Last_Close"]),
            "rsi": float(row["RSI"]) if pd.notna(row.get("RSI")) else None,
            "atr": float(row["ATR"]),
            "mechanical_scores": entry["scores"],
            "catalyst_warning": bool(row.get("Catalyst_Warning", False)),
            "next_earnings_date": row.get("Next_Earnings_Date"),
            "headlines": market_data.get_multi_headlines(ticker),
            "fundamentals": fundamentals,
            "macro": macro_snapshot,
            "qualitative": market_data.get_qualitative_snapshot(ticker),
        }

        for variant in all_variants:
            verdict = llm_agent.evaluate_ticker(ticker, context, variant=variant)
            if verdict is None:
                print(f"  {ticker} [{variant}]: LLM evaluation failed or returned an unusable response.")
                continue

            agreement_note = (
                f", providers agreed ({verdict['secondary_provider']})" if verdict["provider_agreement"] is True
                else f", providers DISAGREED ({verdict['secondary_provider']} said {verdict['secondary_decision']}, "
                     "defaulted to the more conservative call)" if verdict["provider_agreement"] is False else ""
            )
            print(f"  {ticker} [{variant}]: {verdict['decision']} (confidence {verdict['confidence']:.0f}/100, "
                  f"news_sentiment={verdict['news_sentiment']}{agreement_note}) -- {verdict['rationale']}")

            if verdict["decision"] in ("Buy", "Hold"):
                variant_config = swingtrade.TradingConfig(
                    **{**config.to_dict(), "strategy": llm_agent.variant_strategy_name(variant)}
                )
                atr = context["atr"]
                buy_price = round(context["last_close"], 2)
                sell_price = round(buy_price + variant_config.atr_take_profit_multiplier * atr, 2)
                stop_loss = round(buy_price - variant_config.stop_loss_atr_multiplier * atr, 2)
                risk = buy_price - stop_loss
                rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0
                llm_rows_by_variant[variant].append({
                    "Ticker": ticker, "As_Of": row["As_Of"], "Signal": verdict["decision"],
                    "Trade_Score": verdict["confidence"], "Last_Close": context["last_close"],
                    "Buy_Price": buy_price, "Sell_Price": sell_price, "Stop_Loss": stop_loss,
                    "RRR": rrr, "RSI": context["rsi"], "ATR": atr,
                    "Distance_to_Buy_Pct": 0.0, "Shares_To_Buy": 0.0, "Est_Cost": 0.0,
                    "Next_Earnings_Date": context["next_earnings_date"],
                    "Catalyst_Warning": context["catalyst_warning"], "Top_Headline": "",
                    "Currency": row.get("Currency", "USD"),
                    "Provider_Agreement": verdict["provider_agreement"],
                    "Secondary_Provider": verdict["secondary_provider"],
                    "Secondary_Decision": verdict["secondary_decision"],
                    "Secondary_Confidence": verdict["secondary_confidence"],
                })

    for variant, llm_rows in llm_rows_by_variant.items():
        if not llm_rows:
            continue
        strategy_name = llm_agent.variant_strategy_name(variant)
        variant_config = swingtrade.TradingConfig(**{**config.to_dict(), "strategy": strategy_name})
        llm_df = pd.DataFrame(llm_rows)
        llm_df["Signal"] = llm_df["Signal"].replace("Hold", "Watch")
        try:
            logged = storage.log_trade_signals(llm_df, variant_config.to_dict())
            print(f"LLM Agent [{variant}]: logged {logged['actionable']} actionable + {logged['research']} "
                  f"research signal(s) to MongoDB (strategy={strategy_name}, never capital-allocated).")
        except Exception as exc:
            print(f"[WARN] LLM Agent [{variant}] signal logging failed: {exc}", file=sys.stderr)


def run_regime_switcher(strategy_frames: dict[str, pd.DataFrame], config: swingtrade.TradingConfig) -> None:
    """Headless automation for regime_switcher.py -- see that module's own
    docstring for the full design/rationale (EXPLICITLY prospective-only,
    skips the backtesting-validation pipeline every other strategy went
    through, by deliberate user choice). `strategy_frames` is keyed by the
    REAL strategy identifier ("breakout"/"squeeze_breakout"/"ma_crossover"),
    not a display label -- see run()'s own construction of this dict,
    built alongside (not from) all_signal_frames since that dict's keys
    are human-readable labels like "Primary (breakout)".

    Automated here (not left dashboard-only like a normal new experimental
    strategy would be) specifically so its own prospective trust floor
    accumulates by calendar time regardless of whether anyone opens the
    dashboard -- the exact gap already identified and fixed for the LLM
    Agent in item 54. NEVER capital-allocated -- Shares_To_Buy/Est_Cost
    are always 0, same as llm_agent.py."""
    candidates: dict[str, dict[str, dict]] = {}
    for strategy_name, df in strategy_frames.items():
        if df is None or df.empty:
            continue
        interesting = df[df["Signal"] != "Ignore"]
        for _, row in interesting.iterrows():
            candidates.setdefault(row["Ticker"], {})[strategy_name] = row.to_dict()

    if not candidates:
        print("Regime Switcher: no tickers scored above Ignore under breakout/squeeze_breakout/ma_crossover today -- nothing to evaluate.")
        return

    switcher_config = swingtrade.TradingConfig(**{**config.to_dict(), "strategy": "regime_switcher"})
    rows = []
    for ticker, strategy_rows in candidates.items():
        pick = regime_switcher.select_regime_pick(ticker, strategy_rows)
        if pick is None:
            continue
        rows.append({
            "Ticker": ticker, "As_Of": pick["As_Of"], "Signal": pick["Signal"],
            "Trade_Score": pick["Trade_Score"], "Last_Close": pick["Last_Close"],
            "Buy_Price": pick["Buy_Price"], "Sell_Price": pick["Sell_Price"], "Stop_Loss": pick["Stop_Loss"],
            "RRR": pick["RRR"], "RSI": pick.get("RSI"), "ATR": pick["ATR"],
            "Distance_to_Buy_Pct": 0.0, "Shares_To_Buy": 0.0, "Est_Cost": 0.0,
            "Next_Earnings_Date": pick.get("Next_Earnings_Date"),
            "Catalyst_Warning": pick.get("Catalyst_Warning", False), "Top_Headline": pick.get("Top_Headline", ""),
            "Currency": pick.get("Currency", "USD"),
            "Regime": pick["Regime"], "Source_Strategy": pick["Source_Strategy"],
        })
        print(f"  {ticker}: {pick['Signal']} (regime={pick['Regime']}, source={pick['Source_Strategy']}, "
              f"Trade_Score={pick['Trade_Score']:.1f})")

    if not rows:
        print("Regime Switcher: candidates existed but none matched their regime's preferred strategy -- no picks today.")
        return

    switcher_df = pd.DataFrame(rows)
    try:
        logged = storage.log_trade_signals(switcher_df, switcher_config.to_dict())
        print(f"Regime Switcher: logged {logged['actionable']} actionable + {logged['research']} research "
              "signal(s) to MongoDB (strategy=regime_switcher, never capital-allocated, prospective-only -- "
              "see regime_switcher.py).")
    except Exception as exc:
        print(f"[WARN] Regime Switcher signal logging failed: {exc}", file=sys.stderr)


def run_best_ideas_strategy(
    strategy_frames: dict[str, pd.DataFrame], regime_picks: list[dict],
    bundle: dict, market_df, sector_lookup: dict[str, str] | None,
    sector_data: dict | None, config: swingtrade.TradingConfig,
) -> None:
    """Headless counterpart to dip_buy_analyzer.py's "Best Ideas" tab -- see
    best_ideas.py for the full methodology/reasoning (an IC/IR-weighted
    ensemble across ma_crossover/squeeze_breakout/regime_switcher/
    llm_agent plus 3 new methodologies built for that tab: sector-relative-
    strength momentum, a rule-based qualitative composite, and an LLM
    meta-synthesis call). NEVER capital-allocated -- Shares_To_Buy/Est_Cost
    are always 0, no cash pool, no allocate_capital() call, same as
    llm_agent.py/regime_switcher.py. Automated here (not left dashboard-
    only) for the same reason regime_switcher.py's own automation exists:
    its real prospective trust floor (and every sub-methodology's own
    IC/IR) should accumulate by calendar time, not just on days someone
    opens the dashboard.

    Degrades gracefully if llm_agent.is_available() is False -- still
    computes/logs the mechanical/regime/sector_rs/qualitative-only
    composite (best_ideas.run_best_ideas() itself gates every LLM/extra-
    fetch call on `fetch_llm`), just without the LLM/meta enrichment
    layers that day."""
    ic_reports = {name: ic_tracking.methodology_report(name) for name in best_ideas.METHODOLOGIES}
    macro_snapshot = market_data.get_macro_snapshot()
    results = best_ideas.run_best_ideas(
        strategy_frames, regime_picks, bundle, sector_data or {}, sector_lookup or {}, config, ic_reports,
        macro_snapshot=macro_snapshot, fetch_llm=llm_agent.is_available(),
    )
    for label, rows in results.items():
        if not rows:
            print(f"Best Ideas [{label}]: no signals today.")
            continue
        row_config = swingtrade.TradingConfig(**{**config.to_dict(), "strategy": label})
        df = pd.DataFrame(rows)
        try:
            logged = storage.log_trade_signals(df, row_config.to_dict())
            print(f"Best Ideas [{label}]: logged {logged['actionable']} actionable + {logged['research']} "
                  "research signal(s) to MongoDB (never capital-allocated).")
        except Exception as exc:
            print(f"[WARN] Best Ideas [{label}] signal logging failed: {exc}", file=sys.stderr)


def run(
    watchlist_path: Path, position_budget: float, with_ai_context: bool = False, risk_amount: float | None = None,
) -> int:
    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    config, config_source = config_loader.load_active_config()
    print(f"{config_source} (strategy={config.strategy})")

    tickers = tuple(read_tickers(watchlist_path))
    if not tickers:
        print(f"[ERROR] no tickers found in {watchlist_path}", file=sys.stderr)
        return 1
    print(f"Scanning {len(tickers)} ticker(s)...")

    try:
        market_uptrend, market_close, market_sma200 = market_data.check_market_uptrend(config)
    except Exception as exc:
        print(f"[ERROR] could not evaluate {market_data.MARKET_INDEX_TICKER} macro trend: {exc}", file=sys.stderr)
        return 1

    if not market_uptrend:
        print(
            f"{market_data.MARKET_INDEX_TICKER} is in a macro downtrend "
            f"(Last_Close {market_close:.2f} < SMA200 {market_sma200:.2f}) -- "
            "skipping scan, no signals logged."
        )
        return 0

    # Fetched ONCE, shared by the primary scan and both secondary
    # strategies below -- see market_data.fetch_ticker_bundle()'s
    # docstring for why this replaces calling market_data.scan_tickers()
    # (or its equivalent) three separate times. sector_lookup/sector_data
    # back the backtest/Optuna-only Sector_Relative_Strength filter
    # (improvements.txt items 68/70/71) -- fast local file read, not a
    # network call.
    sector_lookup = read_ticker_sectors(watchlist_path)
    bundle, market_df, fetch_skipped, sector_data = market_data.fetch_ticker_bundle(
        tickers, sector_lookup=sector_lookup,
    )

    primary_results, primary_score_skipped = market_data.score_bundle_for_strategy(
        bundle, market_df, config, sector_lookup=sector_lookup, sector_data=sector_data,
    )
    skipped = fetch_skipped + primary_score_skipped
    if not primary_results:
        print("[ERROR] no tickers were successfully analyzed.", file=sys.stderr)
        for ticker, reason in skipped:
            print(f"  skipped {ticker}: {reason}", file=sys.stderr)
        return 1

    results_df = pd.DataFrame(primary_results)
    _size_positions(results_df, config, position_budget, risk_amount)
    results_df = _score_for_strategy(results_df, config)

    logged = storage.log_trade_signals(results_df, config.to_dict())
    strong_buys = int((results_df["Signal"] == "Strong Buy").sum())
    buys = int((results_df["Signal"] == "Buy").sum())
    watches = int((results_df["Signal"] == "Watch").sum())
    print(f"Analyzed {len(results_df)}/{len(tickers)} ticker(s): {strong_buys} Strong Buy, {buys} Buy, {watches} Watch.")
    print(f"Logged {logged['actionable']} actionable + {logged['research']} research signal(s) to MongoDB.")

    message = _build_signal_notification(f"Primary ({config.strategy})", results_df)
    if message:
        notifications.notify(message, webhook_url=notifications.get_strategy_webhook_url(config.strategy))

    if config.strategy == "breakout":
        loosened_config = swingtrade.loosened_breakout_config(config)
        loosened_df = swingtrade.add_breakout_trade_score(results_df.copy(), loosened_config)
        strict_signal_by_ticker = dict(zip(results_df["Ticker"], results_df["Signal"]))
        loosened_only = loosened_df[
            loosened_df["Ticker"].map(strict_signal_by_ticker).eq("Ignore")
        ]
        loosened_logged = storage.log_trade_signals(
            loosened_only, loosened_config.to_dict(), tier=storage.LOOSENED_RESEARCH_TIER
        )
        print(
            f"Logged {loosened_logged.get(storage.LOOSENED_RESEARCH_TIER, 0)} research_loosened "
            "signal(s) to MongoDB (would-be signals under loosened filters that the active "
            "config scored Ignore -- see storage/signals.py)."
        )

    all_signal_frames = {f"Primary ({config.strategy})": results_df}
    # Keyed by the REAL strategy identifier (config.strategy), not a
    # display label -- regime_switcher.py needs unambiguous strategy names
    # ("breakout"/"squeeze_breakout"/"ma_crossover"), not "Squeeze Breakout
    # (v39)"-style labels. Built alongside all_signal_frames from the exact
    # same already-computed DataFrames, no extra scan cost.
    strategy_frames = {config.strategy: results_df}
    # No new network fetch -- built once from the already-fetched bundle,
    # shared across every secondary strategy the same way sector_data is
    # (only "pairs" actually consumes it; harmless/unused otherwise).
    pair_price_panels = market_data.build_pair_price_panels(bundle, sector_lookup)
    for label, version in config_loader.SECONDARY_STRATEGY_VERSIONS.items():
        secondary_config, source = config_loader.load_config_by_version(version)
        if secondary_config is None:
            print(f"{label} (v{version}): skipped -- {source}")
            continue
        secondary_df = run_secondary_strategy(
            f"{label} (v{version})", secondary_config, bundle, market_df, position_budget, risk_amount,
            sector_lookup=sector_lookup, sector_data=sector_data,
            log_strategy_override=config_loader.SECONDARY_LOG_STRATEGY_OVERRIDES.get(label),
            pair_price_panels=pair_price_panels,
        )
        if not secondary_df.empty:
            all_signal_frames[f"{label} (v{version})"] = secondary_df
            strategy_frames[secondary_config.strategy] = secondary_df

    if with_ai_context:
        if not ai_context.is_available():
            print("[WARN] --with-ai-context requested but unavailable (set GEMINI_API_KEY).", file=sys.stderr)
        else:
            for df in all_signal_frames.values():
                signal_rows = df[df["Signal"].isin(["Strong Buy", "Buy"])]
                for _, row in signal_rows.iterrows():
                    headlines = market_data.get_multi_headlines(row["Ticker"])
                    summary = ai_context.summarize_ticker_context(row["Ticker"], row["Signal"], headlines)
                    if summary:
                        print(f"\n[{row['Ticker']} ({row['Signal']})] {summary}")

    run_llm_agent(all_signal_frames, config)
    run_regime_switcher(strategy_frames, config)

    # Recomputes the same regime picks run_regime_switcher() just derived
    # internally (cheap, no network -- pure re-application of
    # regime_switcher.select_regime_pick() over strategy_frames, already
    # fetched/scored above) rather than threading a return value through
    # run_regime_switcher()'s existing signature -- keeps that
    # already-working function untouched. See best_ideas.py for why the
    # RAW pick dicts (not run_regime_switcher()'s own reduced row shape)
    # are what run_best_ideas_strategy() needs.
    best_ideas_regime_candidates: dict[str, dict[str, dict]] = {}
    for strategy_name, df in strategy_frames.items():
        if df is None or df.empty:
            continue
        for _, row in df[df["Signal"] != "Ignore"].iterrows():
            best_ideas_regime_candidates.setdefault(row["Ticker"], {})[strategy_name] = row.to_dict()
    regime_picks = [
        pick for ticker, rows in best_ideas_regime_candidates.items()
        if (pick := regime_switcher.select_regime_pick(ticker, rows)) is not None
    ]
    run_best_ideas_strategy(strategy_frames, regime_picks, bundle, market_df, sector_lookup, sector_data, config)

    if skipped:
        print(f"Skipped {len(skipped)} ticker(s) (primary scan -- fetch + scoring failures):")
        for ticker, reason in skipped:
            print(f"  {ticker}: {reason}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Scan the watchlist and log Strong Buy/Buy signals to MongoDB.")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST_FILE)
    parser.add_argument("--position-budget", type=float, default=DEFAULT_POSITION_BUDGET,
                         help="Flat $ position size per trade (ignored if --risk-amount is given).")
    parser.add_argument(
        "--risk-amount", type=float, default=None,
        help="Dollars you're willing to lose if the stop is hit -- sizes Shares_To_Buy as "
             "risk_amount / (Buy_Price - Stop_Loss) instead of the flat --position-budget / "
             "Buy_Price, so a wide-stop (volatile) ticker gets fewer shares than a tight-stop "
             "(calm) one for the same dollar risk. See swingtrade.size_by_risk.",
    )
    parser.add_argument(
        "--with-ai-context", action="store_true",
        help="Print an AI-generated news summary (informational only, not a rating) for each "
             "Strong Buy/Buy signal found across all three strategies. Requires GEMINI_API_KEY "
             "(free tier); degrades to a warning if unavailable rather than failing the scan.",
    )
    args = parser.parse_args()
    sys.exit(run(args.watchlist, args.position_budget, args.with_ai_context, args.risk_amount))


if __name__ == "__main__":
    main()
