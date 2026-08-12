"""
Interactive Streamlit dashboard for a mechanical swing-trading watchlist scan.

This app does NOT predict price direction and is NOT investment advice.
It applies a fixed, deterministic rule set to historical price data. Which
rule set depends on the active System_Config's `strategy` field -- "rsi"
(default, described below) or "breakout" (buys a new
breakout_lookback_days-day CLOSING high in a confirmed uptrend instead of an
oversold dip; same ATR-based stop/target math, own Trade_Score formula --
see swingtrade.compute_breakout_levels/add_breakout_trade_score). Both
compute, for every scanned ticker:

  1. A limit BUY price, defined as the structural support level: the lowest
     daily Low over the last support_lookback_days trading days (recent swing
     low). The SMA discount price is still reported for context but no
     longer used to set the buy price.

     Buy_Signal is True only when both are satisfied:
       a) the last close is at/below the structural support buy price, and
       b) the 14-day RSI is below rsi_oversold_threshold (mathematically
          oversold), computed via pandas_ta.

  2. A limit SELL (take-profit) price, defined as
       buy_price + (atr_take_profit_multiplier * ATR14)
     where ATR14 is the 14-day Average True Range (via pandas_ta), so the
     take-profit target scales with the stock's actual current volatility
     instead of a static percentage.

  3. A Stop_Loss level, defined as buy_price - (1.0 * ATR14), and a
     Risk-to-Reward Ratio (RRR) = (sell_price - buy_price) / (buy_price - stop_loss).

  4. Distance_to_Buy_Pct: how far the last close is from the buy trigger,
       ((last_close - buy_price) / buy_price) * 100
     Positive = still above the buy trigger (needs to fall further).
     Negative/zero = last close is already at or below the buy trigger.

  5. Catalyst awareness: the next upcoming earnings date and the most recent
     news headline (yfinance). Catalyst_Warning is True when the next
     earnings date falls within earnings_warning_days days, flagging a
     volatile binary event -- shown in the table (highlighted red) rather
     than removed, so it can inform rather than hide the decision.

  6. Macro trend and liquidity gates: tickers are excluded from the results
     when the last close is below the 200-day SMA (macro downtrend) or when
     20-day average dollar volume is below min_dollar_volume (too illiquid to
     swing-trade safely). A broad-market gate (MARKET_INDEX_TICKER vs. its
     own 200-day SMA) halts the whole scan when the index itself is in a
     macro downtrend.

  6b. Extended_Decline_Warning / Oversold_Streak_Days: how many consecutive
      trading days RSI has stayed below rsi_oversold_threshold. A loose
      threshold (this system's tuned active value has landed as high as
      ~52, well above a "classic" RSI<30 reading) can stay satisfied for
      weeks during a genuine sustained decline, not just a brief dip --
      combined with support_lookback_days recalculating the structural low
      on a rolling window, that can produce repeated fresh Buy/Strong Buy
      signals on a ticker simply making new lows day after day, not
      reversing. Flagged (highlighted orange) once the streak reaches
      extended_decline_warning_days (default 5) -- purely informational,
      does NOT affect Trade_Score/Signal.

  7. Shares_To_Buy, one of two sidebar-selectable modes:
       - Flat: position_budget / buy_price -- the same $ position size on
         every trade regardless of the ticker's own stop distance.
       - Risk-based: risk_amount / (buy_price - stop_loss) -- sized so the
         $ amount you'd actually lose if the stop is hit is held roughly
         constant across every trade instead, so a violently volatile
         ticker (wide stop) gets fewer shares than a calm one (tight stop)
         for the same dollar risk. See swingtrade.size_by_risk.
     Both rounded to fractional_share_decimals places, recalculated live
     from the sidebar. Capital is
     then allocated greedily down the Trade_Score-ranked list against "Total
     Available Cash": a trade that no longer fits the remaining cash is
     marked Insufficient Funds, and (if watchlist.txt's JSON form supplies a
     "sector" per ticker) a trade that would push its sector's cumulative
     spend past max_sector_allocation_pct of your whole portfolio value is
     marked Sector Limit Reached instead -- five Strong Buys in one sector on
     the same day are one concentrated bet wearing five tickers, not five
     independent ones, and nothing else in this pipeline catches that. A
     positive max_total_deployed_pct (0/disabled by default) additionally
     caps TOTAL spend across ALL sectors combined at that fraction of
     portfolio value -- the sector cap alone can't stop a day with signals
     spread evenly across many sectors from still deploying 100% of cash; a
     trade that would breach this cap is marked Portfolio Limit Reached
     instead.
     "Whole portfolio value" includes the sidebar's "Current Holdings" box
     (persisted to MongoDB's Current_Holdings collection via a Save button,
     manually maintained rather than inferred from unsettled signals, since a
     logged signal doesn't guarantee you actually got filled) -- a sector
     (or, for the portfolio cap, your whole book) you're already overweight
     in from prior holdings gets little or no new room today, even though
     holdings never reduce Total Available Cash itself.

  8. Trade_Score (0-100) blends Risk-to-Reward Ratio, RSI, and how close the
     last close is to the buy trigger into a single priority score, mapped to
     a Signal of Strong Buy / Buy / Watch / Ignore.

  9. Every "Strong Buy"/"Buy" signal is logged to MongoDB's Trade_Signals
     collection (idempotent per ticker/day), using the *pre-allocation*
     Signal -- a personal cash shortfall ("Insufficient Funds") shouldn't be
     recorded as if the underlying technical signal changed. If MONGODB_URI
     isn't configured, the dashboard still works; logging is just skipped
     with a sidebar note. The same scan-and-log pipeline also runs headless,
     independent of anyone having this dashboard open, via `ingest.py`
     (see ARCHITECTURE_PLAN.md Phase 7) -- both share the fetch logic in
     `market_data.py` and the config-loading logic in `config_loader.py` so
     they can never silently drift apart.

  10. All of the above run on whichever TradingConfig is currently "active"
      in MongoDB's System_Config (the output of optimize.py + a deliberate
      promote_config.py decision -- see ARCHITECTURE_PLAN.md Phase 5),
      re-checked periodically at the same cadence as the rest of the scan
      cache. If nothing is active yet, or Mongo is unreachable, this falls
      back to swingtrade.DEFAULT_CONFIG rather than crashing -- the sidebar
      always shows which one is actually in effect.

  11. The sidebar's "Apply capital allocation" checkbox, when unchecked,
      shows raw Strong Buy/Buy/Watch/Ignore signals with Total Available
      Cash and Current Holdings ignored entirely -- useful for just
      screening candidates without your portfolio state influencing which
      ones get labeled fundable. Separately, any Current Holdings entry with
      an AVG_COST populates a "Position Review" table: the same ATR-based
      stop/target math applied to a position you already own (anchored to
      your real entry price, not a freshly computed support level), with a
      HOLD / SELL (stop breached) / SELL (target hit) recommendation --
      informational, not a guarantee.

  12. Below the primary scan, two validated secondary strategies (see
      improvements.txt items 27/28) are shown in their own sections --
      "Breakout Retest" (a genuine breakout's own trigger level retested
      within a following window) and "52-Week High" (price near its own
      trailing 52-week high) -- each with its OWN separate "Total Available
      Cash" pool (never merged with the primary or each other) and its own
      independent capital-allocation pass against your SAME real Current
      Holdings. All three sections share one fetch of the watchlist's OHLCV
      data (see market_data.fetch_ticker_bundle) so running three
      strategies costs one network fetch, not three. Genuinely tradeable,
      capital-eligible signals -- logged to MongoDB the same as the primary
      scan, not a research-only/informational view like the Loosened
      Filters section above.

  13. A second top-level tab, "LLM Agent (experimental)" -- a genuine
      LLM-derived Buy/Hold/Avoid judgment (see llm_agent.py), NOT another
      mechanical strategy. Unlike everything in the "Mechanical Strategies"
      tab, this can never be validated via walk-forward backtesting
      (re-prompting a model with "historical" context risks it already
      knowing what happened next from training data) -- it can only be
      validated PROSPECTIVELY, over real elapsed time, which the tab's own
      validation-progress counter makes visible. Evaluates only a small,
      capped set of tickers the mechanical strategies already flagged
      today (never an independent blind scan), logs Buy/Hold decisions to
      MongoDB the same way research-tier signals are (tier="research" for
      Hold, "actionable" for Buy) purely to accumulate real graded
      outcomes over time, and is **never** passed to
      swingtrade.allocate_capital() -- no cash pool, not capital-eligible,
      regardless of how confident a call looks.

The actual level/score/allocation math lives in the `swingtrade` package
(no yfinance/streamlit dependency there); persistence lives in the
`storage` package (no yfinance/streamlit dependency there either). This
file only handles data fetching, Streamlit UI, and wiring it all together.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

import ai_context
import config_loader
import llm_agent
import market_data
import storage
import swingtrade
from watchlist import parse_ticker_text, read_ticker_sectors, read_tickers

# ----------------------------- Configuration -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"

DEFAULT_POSITION_BUDGET = 250    # default $ sidebar value for flat position sizing
DEFAULT_RISK_AMOUNT = 25         # default $ sidebar value for risk-based position sizing
MAX_LLM_CANDIDATES = 10          # cap on the "LLM Agent" tab's per-page-load evaluation
                                  # count -- cost/rate-limit control, see llm_agent.py
DEFAULT_TOTAL_CASH = 5_000       # default $ sidebar value for total available cash --
                                  # used by the secondary-strategy cash pools (currently
                                  # just Squeeze Breakout, v39)
DEFAULT_PRIMARY_CASH = 0         # default $ sidebar value for the PRIMARY (breakout,
                                  # v43) cash pool specifically -- deliberately 0, not
                                  # DEFAULT_TOTAL_CASH, since 2026-08-11: breakout (v43)
                                  # was found to lose to random-entry timing on every
                                  # variant tested (improvements.txt items 44/47), so
                                  # nothing should size against it by default. Still
                                  # shown/scanned (useful context while a replacement is
                                  # sought) -- just requires deliberately typing in an
                                  # amount to actually allocate capital, rather than
                                  # defaulting to $5,000 every session.

SCAN_CACHE_TTL_SEC = 900         # how long a scan result stays cached (15 min)

SIGNAL_COLORS = {
    "Strong Buy": "background-color: #1b7a3d; color: #ffffff;",
    "Buy": "background-color: #8bc34a; color: #1a1a1a;",
    "Watch": "background-color: #f6c945; color: #1a1a1a;",
    "Ignore": "background-color: #e57373; color: #1a1a1a;",
    "Insufficient Funds": "background-color: #78909c; color: #ffffff;",
    "Sector Limit Reached": "background-color: #5e35b1; color: #ffffff;",
    "Portfolio Limit Reached": "background-color: #ad1457; color: #ffffff;",
}
CATALYST_WARNING_STYLE = "background-color: #c62828; color: #ffffff; font-weight: 600;"
EXTENDED_DECLINE_STYLE = "background-color: #e65100; color: #ffffff; font-weight: 600;"

REVIEW_COLORS = {
    "SELL (stop breached)": "background-color: #c62828; color: #ffffff; font-weight: 600;",
    "SELL (target hit)": "background-color: #1b7a3d; color: #ffffff; font-weight: 600;",
    "HOLD": "",
}

DISPLAY_COLUMNS_RSI = [
    "Ticker", "Signal", "Trade_Score", "Last_Close", "Buy_Price", "Stop_Loss",
    "Sell_Price", "RRR", "RSI", "ATR", "Distance_to_Buy_Pct", "Shares_To_Buy",
    "Est_Cost", "Next_Earnings_Date", "Catalyst_Warning", "Oversold_Streak_Days",
    "Extended_Decline_Warning", "Top_Headline", "As_Of",
]
# Breakout signals have no Oversold_Streak_Days/Extended_Decline_Warning
# equivalent (RSI-mean-reversion-specific concepts -- see
# add_breakout_trade_score's docstring) -- dropped rather than shown as
# always-empty columns. RSI is still shown (compute_breakout_levels
# computes it informationally, e.g. to see how extended a breakout is).
DISPLAY_COLUMNS_BREAKOUT = [
    "Ticker", "Signal", "Trade_Score", "Last_Close", "Buy_Price", "Stop_Loss",
    "Sell_Price", "RRR", "RSI", "ATR", "Distance_to_Buy_Pct", "Shares_To_Buy",
    "Est_Cost", "Next_Earnings_Date", "Catalyst_Warning", "Top_Headline", "As_Of",
]

# ---------------------------------------------------------------------------


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Checking broad-market macro trend...")
def cached_market_uptrend(config: swingtrade.TradingConfig) -> tuple[bool, float, float]:
    return market_data.check_market_uptrend(config)


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def cached_macro_snapshot() -> dict:
    """VIX level/change + broad-market headlines for the LLM Agent tab (see
    market_data.get_macro_snapshot()) -- fetched ONCE per page load and
    shared across every candidate ticker evaluated that run, same caching
    pattern as cached_market_uptrend/cached_fetch_bundle above."""
    return market_data.get_macro_snapshot()


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Fetching watchlist data...")
def cached_fetch_bundle(tickers: tuple[str, ...]):
    """Fetch OHLCV + earnings + headlines for every ticker ONCE, shared
    across every strategy section below (primary v19 + the secondary
    breakout_retest/week52_high sections) -- see
    market_data.fetch_ticker_bundle()'s docstring. Cached on `tickers`
    alone, unlike the old per-config scan_watchlist this replaces --
    correct, since fetching doesn't depend on which strategy will score
    the data. Each section then calls market_data.score_bundle_for_strategy()
    against this same bundle, so running 3 strategies costs 1x the network
    fetch, not 3x."""
    return market_data.fetch_ticker_bundle(tickers)


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner="Reviewing current holdings...")
def cached_review_holdings(
    holdings_key: tuple[tuple[str, float], ...], config: swingtrade.TradingConfig
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Evaluate each held ticker (with a known avg_cost) against the active
    config's stop/target rules. `holdings_key` is a hashable
    ((ticker, avg_cost), ...) tuple so this can be cached like cached_fetch_bundle."""
    results, skipped = market_data.review_holdings(dict(holdings_key), config)
    return pd.DataFrame(results), skipped


def style_review(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    formats = {
        "Avg_Cost": "{:.2f}", "Last_Close": "{:.2f}", "ATR": "{:.2f}",
        "Stop_Loss": "{:.2f}", "Sell_Price": "{:.2f}", "Unrealized_PnL_Pct": "{:+.2f}%",
    }
    return (
        df.style
        .format(formats, na_rep="-")
        .map(lambda v: REVIEW_COLORS.get(v, ""), subset=["Recommendation"])
    )


def parse_holdings_text(raw: str) -> dict[str, dict]:
    """Parse 'TICKER,AMOUNT[,AVG_COST]' (or space-separated) lines into
    {ticker: {"amount": dollars committed, "avg_cost": price/share or None}}.
    AVG_COST is optional -- without it, the holding still counts toward the
    sector cap but can't appear in the Position Review table (needs a cost
    basis to compute a stop/target against). Blank/malformed lines and
    non-positive amounts are skipped rather than erroring -- this is a
    manually-typed box, not a validated form."""
    holdings = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) not in (2, 3):
            continue
        ticker = parts[0]
        try:
            amount = float(parts[1])
        except ValueError:
            continue
        if amount <= 0:
            continue
        avg_cost = None
        if len(parts) == 3:
            try:
                avg_cost = float(parts[2])
            except ValueError:
                avg_cost = None
        holdings[ticker.upper()] = {"amount": amount, "avg_cost": avg_cost}
    return holdings


def style_results(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    formats = {
        "Trade_Score": "{:.1f}",
        "Last_Close": "{:.2f}",
        "Buy_Price": "{:.2f}",
        "Stop_Loss": "{:.2f}",
        "Sell_Price": "{:.2f}",
        "RRR": "{:.2f}",
        "RSI": "{:.1f}",
        "ATR": "{:.2f}",
        "Distance_to_Buy_Pct": "{:.2f}%",
        "Shares_To_Buy": "{:.4f}",
        "Est_Cost": "{:.2f}",
    }
    styler = (
        df.style
        .format(formats, na_rep="-")
        .map(lambda v: SIGNAL_COLORS.get(v, ""), subset=["Signal"])
        .map(lambda v: CATALYST_WARNING_STYLE if v else "", subset=["Catalyst_Warning"])
    )
    # Extended_Decline_Warning is RSI-only (see DISPLAY_COLUMNS_BREAKOUT) --
    # guard rather than assume the column is always present.
    if "Extended_Decline_Warning" in df.columns:
        styler = styler.map(lambda v: EXTENDED_DECLINE_STYLE if v else "", subset=["Extended_Decline_Warning"])
    return styler


@st.cache_resource(show_spinner=False)
def init_storage() -> tuple[bool, str]:
    """One-time-per-process MongoDB connectivity check + index setup.
    Returns (ok, message) rather than raising, so a missing/unreachable
    database degrades the dashboard gracefully instead of crashing it."""
    try:
        storage.ensure_indexes()
        return True, ""
    except storage.MongoNotConfigured as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Could not connect to MongoDB: {exc}"


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def load_active_config() -> tuple[swingtrade.TradingConfig, str]:
    """Load the active TradingConfig from MongoDB's System_Config (re-checked
    on the same TTL cadence as the rest of the scan cache, so a newly
    promoted config takes effect without restarting the app). Logic lives in
    config_loader (shared with ingest.py, unwrapped there) so the dashboard
    and the standalone scan can never disagree on what "active" means."""
    return config_loader.load_active_config()


# Single source of truth now lives in config_loader.py (shared with
# ingest.py) -- aliased here so every existing reference in this file
# keeps working unchanged.
SECONDARY_STRATEGY_VERSIONS = config_loader.SECONDARY_STRATEGY_VERSIONS
# Deliberately a SEPARATE dict, NOT merged into SECONDARY_STRATEGY_VERSIONS
# -- see config_loader.EXPERIMENTAL_STRATEGY_VERSIONS' docstring. ingest.py
# only ever iterates SECONDARY_STRATEGY_VERSIONS, so keeping this separate
# is what actually keeps an experimental strategy out of automation.
EXPERIMENTAL_STRATEGY_VERSIONS = config_loader.EXPERIMENTAL_STRATEGY_VERSIONS


@st.cache_data(ttl=SCAN_CACHE_TTL_SEC, show_spinner=False)
def load_secondary_config(version: int) -> tuple[swingtrade.TradingConfig | None, str]:
    """Load one fixed candidate System_Config version for a secondary scan
    section -- see config_loader.load_config_by_version()'s docstring for
    why this returns (None, reason) on failure rather than a silent
    DEFAULT_CONFIG fallback."""
    return config_loader.load_config_by_version(version)


def _score_for_strategy(df: pd.DataFrame, config: swingtrade.TradingConfig) -> pd.DataFrame:
    """Dispatch to the right add_*_trade_score() for config.strategy --
    covers all eight strategies (previously this dispatch only handled
    "breakout" vs. everything-else, silently mis-scoring pullback/
    breakout_retest/week52_high rows as RSI; fixed alongside the same gap
    in market_data.score_bundle_for_strategy())."""
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
    else:
        return swingtrade.add_trade_score(df, config)


def render_secondary_section(
    label: str,
    config: swingtrade.TradingConfig,
    bundle: dict,
    market_df,
    fetch_skipped: list[tuple[str, str]],
    sector_lookup: dict[str, str],
    existing_holdings: dict[str, float],
    total_cash: float,
    apply_allocation: bool,
    risk_amount: float | None,
    position_budget: float | None,
    storage_ok: bool,
) -> pd.DataFrame:
    """Score, log, allocate, and display one secondary strategy's results
    against the SAME already-fetched bundle the primary scan used -- no
    extra network fetch (see cached_fetch_bundle()). `existing_holdings` is
    your real portfolio, shared and NOT updated between this call and the
    primary/other secondary section's own allocate_capital() call within
    the same page load -- each strategy's allocation is evaluated against
    your actual holdings, not against what another strategy hypothetically
    proposed today. A leaner treatment than the primary section (no
    per-sector breakdown expander, no Loosened Filters View) since these
    are newly-added, alongside-only signals -- see improvements.txt items
    27/28.

    Returns the scored (post-allocation, if applied) results_df -- empty if
    nothing was analyzed -- so callers (see the LLM Agent tab's candidate
    pre-filter) can see what this strategy found today without re-scoring."""
    st.subheader(label)
    results, score_skipped = market_data.score_bundle_for_strategy(bundle, market_df, config)
    if not results:
        st.caption("No tickers were successfully analyzed.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    if risk_amount:
        results_df["Shares_To_Buy"] = swingtrade.size_by_risk(
            results_df["Buy_Price"], results_df["Stop_Loss"], risk_amount, config.fractional_share_decimals
        )
    else:
        results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(config.fractional_share_decimals)
    results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
    results_df = _score_for_strategy(results_df, config)
    results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)

    if storage_ok:
        try:
            logged = storage.log_trade_signals(results_df, config.to_dict())
            st.caption(
                f"Logged {logged['actionable']} actionable + {logged['research']} research signal(s) to MongoDB."
            )
        except Exception as exc:
            st.warning(f"Signal logging failed: {exc}")
    else:
        st.caption("MongoDB not connected -- signals aren't being logged.")

    if apply_allocation:
        results_df, capital_allocated = swingtrade.allocate_capital(
            results_df, total_cash,
            sector_lookup=sector_lookup, max_sector_allocation_pct=config.max_sector_allocation_pct,
            existing_holdings=existing_holdings, max_total_deployed_pct=config.max_total_deployed_pct,
        )
        remaining_idle_cash = round(total_cash - capital_allocated, 2)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Analyzed", len(results_df))
        col2.metric("Starting Cash", f"${total_cash:,.2f}")
        col3.metric("Allocated", f"${capital_allocated:,.2f}")
        col4.metric("Idle Cash", f"${remaining_idle_cash:,.2f}")
        if config.max_total_deployed_pct and config.max_total_deployed_pct > 0:
            portfolio_value = total_cash + sum(existing_holdings.values())
            st.caption(
                f"Portfolio cap: {config.max_total_deployed_pct * 100:.0f}% of "
                f"${portfolio_value:,.2f} = ${config.max_total_deployed_pct * portfolio_value:,.2f} "
                "total across all sectors combined."
            )
    else:
        st.caption(f"{len(results_df)} analyzed. Capital allocation is off.")

    display_columns = DISPLAY_COLUMNS_RSI if config.strategy == "rsi" else DISPLAY_COLUMNS_BREAKOUT
    st.dataframe(style_results(results_df[display_columns]), width="stretch", hide_index=True)

    all_skipped = fetch_skipped + score_skipped
    if all_skipped:
        with st.expander(f"Skipped for {label} ({len(all_skipped)})"):
            st.dataframe(pd.DataFrame(all_skipped, columns=["Ticker", "Reason"]), hide_index=True)

    return results_df


def render_experimental_section(
    label: str,
    config: swingtrade.TradingConfig,
    bundle: dict,
    market_df,
    fetch_skipped: list[tuple[str, str]],
    storage_ok: bool,
) -> pd.DataFrame:
    """Score, log, and display one EXPERIMENTAL strategy's results against
    the SAME already-fetched bundle every other section uses -- no extra
    network fetch. Unlike render_secondary_section(), this NEVER sizes
    positions and NEVER calls swingtrade.allocate_capital() -- no cash pool
    exists for an experimental strategy, full stop (see
    config_loader.EXPERIMENTAL_STRATEGY_VERSIONS' docstring for why this
    stays a separate dict from SECONDARY_STRATEGY_VERSIONS, the actual
    mechanism enforcing this).

    Unlike the LLM Agent tab, an experimental strategy here IS a mechanical
    one and was already run through the SAME offline validation every
    strategy in this codebase is held to (benchmark_random_entry.py against
    5 years of held-out history) BEFORE this tab ever renders it -- see
    improvements.txt for the specific strategy's result. Signals are still
    logged to MongoDB (strategy=config.strategy) so settle_trades.py grades
    real outcomes over time for ongoing review, but that logging is
    record-keeping, not a substitute for the offline validation that
    already happened.

    Shares_To_Buy/Est_Cost are deliberately omitted from the display
    (unlike render_secondary_section()'s DISPLAY_COLUMNS_BREAKOUT) --
    showing zeroed sizing columns for a strategy with no cash pool would
    read as a bug, not a design choice."""
    st.subheader(label)
    results, score_skipped = market_data.score_bundle_for_strategy(bundle, market_df, config)
    if not results:
        st.caption("No tickers were successfully analyzed.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = _score_for_strategy(results_df, config)
    results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)

    if storage_ok:
        try:
            logged = storage.log_trade_signals(results_df, config.to_dict())
            st.caption(
                f"Logged {logged['actionable']} actionable + {logged['research']} research signal(s) "
                "to MongoDB (never capital-allocated -- see warning above)."
            )
        except Exception as exc:
            st.warning(f"Signal logging failed: {exc}")
    else:
        st.caption("MongoDB not connected -- signals aren't being logged.")

    # Extra columns showing WHY a signal fired differ per experimental
    # strategy -- momentum_burst's volume confirmation isn't a column
    # squeeze_breakout's frame has, and vice versa.
    if config.strategy == "momentum_burst":
        trigger_columns = ["Day_Gain_Pct", "Volume_Ratio"]
    elif config.strategy == "squeeze_breakout":
        trigger_columns = ["Day_Gain_Pct", "Recent_Min_Squeeze_Zscore"]
    elif config.strategy == "adx_trend_entry":
        trigger_columns = ["ADX", "Short_MA"]
    else:
        trigger_columns = []
    display_columns = [
        "Ticker", "Signal", "Trade_Score", "Last_Close", *trigger_columns,
        "Buy_Price", "Sell_Price", "Stop_Loss", "RRR", "RSI", "ATR",
        "Next_Earnings_Date", "Catalyst_Warning", "Top_Headline", "As_Of",
    ]
    st.dataframe(style_results(results_df[display_columns]), width="stretch", hide_index=True)

    all_skipped = fetch_skipped + score_skipped
    if all_skipped:
        with st.expander(f"Skipped for {label} ({len(all_skipped)})"):
            st.dataframe(pd.DataFrame(all_skipped, columns=["Ticker", "Reason"]), hide_index=True)

    return results_df


def main():
    st.set_page_config(page_title="Swing-Trading Dashboard", layout="wide")
    st.title("Swing-Trading Dashboard")
    st.caption(
        "Mechanical, rule-based output from historical data -- not a forecast or "
        "recommendation. Verify live price/liquidity before placing orders."
    )

    config, config_source = load_active_config()
    secondary_configs = {
        label: load_secondary_config(version) for label, version in SECONDARY_STRATEGY_VERSIONS.items()
    }
    experimental_configs = {
        label: load_secondary_config(version) for label, version in EXPERIMENTAL_STRATEGY_VERSIONS.items()
    }

    with st.sidebar:
        st.header("Configuration")
        st.caption(f"{config_source} (strategy: {config.strategy})")
        with st.expander("Active trading parameters"):
            if config.strategy == "breakout":
                st.json({
                    "strategy": config.strategy,
                    "breakout_lookback_days": config.breakout_lookback_days,
                    "atr_take_profit_multiplier": config.atr_take_profit_multiplier,
                    "stop_loss_atr_multiplier": config.stop_loss_atr_multiplier,
                    "max_holding_days": config.max_holding_days,
                    "max_sector_allocation_pct": config.max_sector_allocation_pct,
                    "max_total_deployed_pct": config.max_total_deployed_pct,
                })
            else:
                st.json({
                    "strategy": config.strategy,
                    "rsi_oversold_threshold": config.rsi_oversold_threshold,
                    "atr_take_profit_multiplier": config.atr_take_profit_multiplier,
                    "stop_loss_atr_multiplier": config.stop_loss_atr_multiplier,
                    "max_holding_days": config.max_holding_days,
                    "max_sector_allocation_pct": config.max_sector_allocation_pct,
                    "max_total_deployed_pct": config.max_total_deployed_pct,
                })
        sizing_mode = st.radio(
            "Position sizing mode",
            ["Flat $ per trade", "Risk-based ($ risked per trade)"],
            help="Flat: every trade gets the same $ position size, regardless of the "
                 "ticker's own stop distance. Risk-based: position size is scaled so "
                 "shares x (Buy_Price - Stop_Loss) is roughly the same $ amount across "
                 "every trade -- a violently volatile ticker (wide stop) gets fewer shares "
                 "than a calm one (tight stop) for the same dollar risk, instead of the "
                 "same flat position size for both. See swingtrade.size_by_risk.",
        )
        if sizing_mode == "Risk-based ($ risked per trade)":
            risk_amount = st.number_input(
                "Risk per Trade ($)",
                min_value=1.0,
                value=float(DEFAULT_RISK_AMOUNT),
                step=5.0,
                help="Dollars you're willing to lose if the stop is hit; drives the "
                     "fractional Shares_To_Buy column below as risk_amount / "
                     "(Buy_Price - Stop_Loss) instead of a flat position_budget / Buy_Price.",
            )
            position_budget = None
        else:
            position_budget = st.number_input(
                "Position Budget ($)",
                min_value=1.0,
                value=float(DEFAULT_POSITION_BUDGET),
                step=10.0,
                help="Max $ allocated per trade; drives the fractional Shares_To_Buy column below.",
            )
            risk_amount = None
        total_cash = st.number_input(
            "Total Available Cash ($)",
            min_value=0.0,
            value=float(DEFAULT_PRIMARY_CASH),
            step=100.0,
            help="Capital pool spent greedily down the Trade_Score-ranked Buy/Strong Buy list; "
                 "defaults to $0 -- breakout (v43) lost to random-entry timing on every variant "
                 "tested (2026-08-11, improvements.txt items 44/47), so nothing sizes against it "
                 "unless you deliberately enter an amount. Type in a value if you want to size a "
                 "trade anyway. "
                 "trades that no longer fit are marked Insufficient Funds, trades that would "
                 "over-concentrate one sector are marked Sector Limit Reached, and trades that "
                 "would push TOTAL spend past a portfolio-wide cap are marked Portfolio Limit "
                 "Reached (see max_sector_allocation_pct / max_total_deployed_pct in the active "
                 "config above -- the latter defaults to disabled).",
        )

        st.caption(
            "Each secondary strategy below gets its OWN separate cash pool -- never merged "
            "with the primary pool above or with each other."
        )
        secondary_cash: dict[str, float] = {}
        for label in SECONDARY_STRATEGY_VERSIONS:
            secondary_config, secondary_source = secondary_configs[label]
            if secondary_config is not None:
                secondary_cash[label] = st.number_input(
                    f"Total Available Cash -- {label} ($)",
                    min_value=0.0,
                    value=float(DEFAULT_TOTAL_CASH),
                    step=100.0,
                    help=f"Separate cash pool for the {label} secondary section below -- "
                         "not shared with the primary or any other secondary pool.",
                )
            else:
                secondary_cash[label] = 0.0
                st.caption(f"{label} section unavailable: {secondary_source}")

        st.subheader("Current Holdings")
        try:
            saved_holdings = storage.get_holdings()
        except Exception:
            saved_holdings = {}
        default_holdings_text = "\n".join(
            f"{t},{info['amount']:g}" + (f",{info['avg_cost']:g}" if info.get("avg_cost") else "")
            for t, info in saved_holdings.items()
        )
        holdings_text = st.text_area(
            "What you're actually holding right now (TICKER,AMOUNT[,AVG_COST] per line)",
            value=default_holdings_text,
            height=100,
            help="AMOUNT ($ currently committed) counts toward the sector cap below as "
                 "already-deployed capital -- never subtracted from Total Available Cash "
                 "itself. AVG_COST (optional, your entry price/share) additionally enables "
                 "the Position Review section further down, which checks the holding against "
                 "the active config's stop/target. Used live from this box every run; click "
                 "Save to persist it as the default for next time.",
        )
        holdings_detail = parse_holdings_text(holdings_text)
        existing_holdings = {t: info["amount"] for t, info in holdings_detail.items()}
        if st.button("Save holdings"):
            try:
                storage.set_holdings(holdings_detail)
                st.success(f"Saved {len(holdings_detail)} holding(s).")
            except Exception as exc:
                st.warning(f"Could not save holdings: {exc}")

        apply_allocation = st.checkbox(
            "Apply capital allocation (cash + sector + portfolio caps)",
            value=True,
            help="When off, the Scan Results table shows raw Strong Buy/Buy/Watch/Ignore "
                 "signals with no Insufficient Funds / Sector Limit Reached / Portfolio Limit "
                 "Reached overlay -- your Total Available Cash and Current Holdings are ignored "
                 "for screening purposes (Position Review below is unaffected either way).",
        )

        ai_context_available = ai_context.is_available()
        generate_ai_context = st.checkbox(
            "Generate AI context for Strong Buy/Buy signals",
            value=False,
            disabled=not ai_context_available,
            help="Summarizes each Strong Buy/Buy ticker's recent news headlines via Google "
                 "Gemini's free tier -- informational only, purely for you to read. Never "
                 "feeds back into Trade_Score, Signal, or position sizing (see ai_context.py). "
                 "Uses your GEMINI_API_KEY (free tier)." +
                 ("" if ai_context_available else " Unavailable: set GEMINI_API_KEY to enable."),
        )

        default_ticker_text = "\n".join(read_tickers(WATCHLIST_FILE)) if WATCHLIST_FILE.exists() else ""
        ticker_text = st.text_area(
            "Watchlist (one ticker per line, or comma-separated)",
            value=default_ticker_text,
            height=280,
        )
        tickers = tuple(parse_ticker_text(ticker_text))
        st.caption(f"{len(tickers)} ticker(s) loaded.")

    tab1, tab_daily, tab2 = st.tabs(
        ["Mechanical Strategies", "Daily Signals (experimental)", "LLM Agent (experimental)"]
    )

    with tab1:
        sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

        if not tickers:
            st.warning("No tickers to scan. Paste some tickers in the sidebar watchlist box.")
            st.stop()

        try:
            market_uptrend, market_close, market_sma200 = cached_market_uptrend(config)
        except Exception as exc:
            st.error(f"Could not evaluate {market_data.MARKET_INDEX_TICKER} macro trend: {exc}")
            st.stop()

        if not market_uptrend:
            st.error(
                f"**{market_data.MARKET_INDEX_TICKER} is in a macro downtrend** "
                f"(Last_Close {market_close:.2f} < SMA200 {market_sma200:.2f}). "
                "Individual structural-support levels are unreliable when the broad market "
                "itself is breaking down -- watchlist analysis has been skipped."
            )
            st.stop()
        st.success(f"{market_data.MARKET_INDEX_TICKER} is above its 200-day SMA ({market_close:.2f} >= {market_sma200:.2f}).")

        bundle, market_df, fetch_skipped = cached_fetch_bundle(tickers)
        primary_results, primary_score_skipped = market_data.score_bundle_for_strategy(bundle, market_df, config)
        results_df = pd.DataFrame(primary_results)
        skipped = fetch_skipped + primary_score_skipped

        if results_df.empty:
            st.error("No tickers were successfully analyzed. See skipped tickers below.")
            if skipped:
                with st.expander(f"Skipped tickers ({len(skipped)})"):
                    st.dataframe(pd.DataFrame(skipped, columns=["Ticker", "Reason"]), hide_index=True)
            st.stop()

        if risk_amount:
            results_df["Shares_To_Buy"] = swingtrade.size_by_risk(
                results_df["Buy_Price"], results_df["Stop_Loss"], risk_amount, config.fractional_share_decimals
            )
        else:
            results_df["Shares_To_Buy"] = (position_budget / results_df["Buy_Price"]).round(config.fractional_share_decimals)
        results_df["Est_Cost"] = (results_df["Shares_To_Buy"] * results_df["Buy_Price"]).round(2)
        results_df = _score_for_strategy(results_df, config)
        results_df = results_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)

        # Log signals BEFORE the capital-allocation overlay: Trade_Signals should
        # reflect the underlying technical signal, not whether cash happened to
        # be available today.
        storage_ok, storage_message = init_storage()
        if storage_ok:
            try:
                logged = storage.log_trade_signals(results_df, config.to_dict())
                st.sidebar.caption(
                    f"Logged {logged['actionable']} actionable + {logged['research']} research signal(s) to MongoDB."
                )
            except Exception as exc:
                st.sidebar.warning(f"Signal logging failed: {exc}")
        else:
            st.sidebar.caption(f"MongoDB not connected ({storage_message}) -- signals aren't being logged.")

        if apply_allocation:
            results_df, capital_allocated = swingtrade.allocate_capital(
                results_df, total_cash,
                sector_lookup=sector_lookup, max_sector_allocation_pct=config.max_sector_allocation_pct,
                existing_holdings=existing_holdings, max_total_deployed_pct=config.max_total_deployed_pct,
            )
            remaining_idle_cash = round(total_cash - capital_allocated, 2)
        else:
            capital_allocated = None
            remaining_idle_cash = None

        strong_buys = int((results_df["Signal"] == "Strong Buy").sum())
        catalyst_warnings = int(results_df["Catalyst_Warning"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickers Scanned", len(tickers))
        col2.metric("Successfully Analyzed", len(results_df))
        col3.metric("Active Strong Buys", strong_buys)
        col4.metric("Catalyst Warnings", catalyst_warnings)

        if apply_allocation:
            col5, col6, col7 = st.columns(3)
            col5.metric("Starting Cash", f"${total_cash:,.2f}")
            col6.metric("Capital Allocated to Orders", f"${capital_allocated:,.2f}")
            col7.metric("Remaining Idle Cash", f"${remaining_idle_cash:,.2f}")
            if config.max_total_deployed_pct and config.max_total_deployed_pct > 0:
                portfolio_value = total_cash + sum(existing_holdings.values())
                st.caption(
                    f"Portfolio cap: {config.max_total_deployed_pct * 100:.0f}% of "
                    f"${portfolio_value:,.2f} = ${config.max_total_deployed_pct * portfolio_value:,.2f} "
                    "total across all sectors combined."
                )
        else:
            st.info(
                "Capital allocation is off -- Signal column shows raw Strong Buy/Buy/Watch/Ignore, "
                "unaffected by Total Available Cash or Current Holdings."
            )

        if apply_allocation and sector_lookup:
            funded = results_df[results_df["Signal"].isin(["Strong Buy", "Buy"])].copy()
            funded["Sector"] = funded["Ticker"].map(sector_lookup).fillna("Unknown")
            new_by_sector = funded.groupby("Sector")["Est_Cost"].sum()

            holdings_by_sector: dict[str, float] = {}
            for ticker, amount in existing_holdings.items():
                sector = sector_lookup.get(ticker, "Unknown")
                holdings_by_sector[sector] = holdings_by_sector.get(sector, 0.0) + amount
            holdings_by_sector = pd.Series(holdings_by_sector, dtype=float)

            breakdown = pd.DataFrame({
                "Existing Holdings ($)": holdings_by_sector,
                "New Today ($)": new_by_sector,
            }).fillna(0.0)
            breakdown["Total ($)"] = breakdown["Existing Holdings ($)"] + breakdown["New Today ($)"]
            breakdown = breakdown.sort_values("Total ($)", ascending=False)

            portfolio_value = total_cash + sum(existing_holdings.values())
            sector_cap_dollars = (
                config.max_sector_allocation_pct * portfolio_value if config.max_sector_allocation_pct > 0 else None
            )
            with st.expander("Capital allocated by sector"):
                if sector_cap_dollars is not None:
                    st.caption(
                        f"Cap: {config.max_sector_allocation_pct * 100:.0f}% of total portfolio value "
                        f"(${portfolio_value:,.2f} = today's cash + existing holdings) per sector "
                        f"= ${sector_cap_dollars:,.2f}."
                    )
                st.dataframe(breakdown.reset_index(names="Sector"), width="stretch", hide_index=True)

        holdings_with_cost = tuple(sorted(
            (t, info["avg_cost"]) for t, info in holdings_detail.items() if info.get("avg_cost")
        ))
        if holdings_with_cost:
            st.subheader("Position Review")
            st.caption(
                "Checks each holding with a known AVG_COST against the active config's ATR-based "
                "stop/target, anchored to your real entry price -- not a guarantee, same caveats "
                "as every other signal in this system."
            )
            review_df, review_skipped = cached_review_holdings(holdings_with_cost, config)
            if not review_df.empty:
                st.dataframe(style_review(review_df), width="stretch", hide_index=True)
            if review_skipped:
                with st.expander(f"Could not review ({len(review_skipped)})"):
                    st.dataframe(pd.DataFrame(review_skipped, columns=["Ticker", "Reason"]), hide_index=True)

        st.subheader("Scan Results")
        display_columns = DISPLAY_COLUMNS_BREAKOUT if config.strategy == "breakout" else DISPLAY_COLUMNS_RSI
        st.dataframe(style_results(results_df[display_columns]), width="stretch", hide_index=True)

        if config.strategy == "breakout":
            st.subheader("Loosened Filters View (research/informational only)")
            st.caption(
                "Same underlying scan, re-scored with the six 'sharpening' filters (overbought, "
                "relative-strength, volume-ratio, ADX, OBV, squeeze) reset to disabled -- shows what "
                "would score a signal under just the base breakout+ATR strategy, without the extra "
                "selectivity that's WHY the active config is validated to be this selective (see "
                "improvements.txt items 17/18/23). The core trigger (breakout_lookback_days) and "
                "ATR stop/target levels are unchanged -- only the extra gates are loosened. "
                "**Never used for capital allocation** -- rows the active config scored Ignore are "
                "logged to MongoDB tagged tier=\"research_loosened\" purely to accumulate real "
                "outcome data on days the Scan Results table above is empty or thin (see "
                "storage/signals.py); never mixed with actionable/research-tier outcomes."
            )
            loosened_config = swingtrade.loosened_breakout_config(config)
            loosened_df = swingtrade.add_breakout_trade_score(results_df.copy(), loosened_config)

            if storage_ok:
                try:
                    strict_signal_by_ticker = dict(zip(results_df["Ticker"], results_df["Signal"]))
                    loosened_only = loosened_df[
                        loosened_df["Ticker"].map(strict_signal_by_ticker).eq("Ignore")
                    ]
                    loosened_logged = storage.log_trade_signals(
                        loosened_only, loosened_config.to_dict(), tier=storage.LOOSENED_RESEARCH_TIER
                    )
                    st.caption(
                        f"Logged {loosened_logged.get(storage.LOOSENED_RESEARCH_TIER, 0)} "
                        "research_loosened signal(s) to MongoDB."
                    )
                except Exception as exc:
                    st.caption(f"Loosened-signal logging failed: {exc}")

            loosened_df = loosened_df.sort_values("Trade_Score", ascending=False).reset_index(drop=True)
            shown = loosened_df[loosened_df["Signal"] != "Ignore"]
            if shown.empty:
                st.caption("Nothing scores above Ignore even with the extra filters disabled -- "
                           "genuinely no breakout activity anywhere in the watchlist today.")
            else:
                st.dataframe(style_results(shown[display_columns]), width="stretch", hide_index=True)

        secondary_results_by_label: dict[str, pd.DataFrame] = {}
        for label in SECONDARY_STRATEGY_VERSIONS:
            secondary_config, _ = secondary_configs[label]
            if secondary_config is None:
                continue
            st.divider()
            secondary_results_by_label[label] = render_secondary_section(
                f"{label} Signals (v{SECONDARY_STRATEGY_VERSIONS[label]}, secondary)",
                secondary_config,
                bundle, market_df, fetch_skipped,
                sector_lookup, existing_holdings,
                secondary_cash[label], apply_allocation,
                risk_amount, position_budget,
                storage_ok,
            )

        if generate_ai_context:
            signal_tickers = results_df[results_df["Signal"].isin(["Strong Buy", "Buy"])]["Ticker"].tolist()
            if not signal_tickers:
                st.caption("AI context: no Strong Buy/Buy signals today, nothing to summarize.")
            else:
                st.subheader("AI Context (Strong Buy / Buy signals)")
                st.caption(
                    "Informational only -- summarizes each ticker's recent headlines for you to "
                    "read. Not a rating, not a recommendation, and never fed back into Trade_Score "
                    "or Signal. See ai_context.py for the full reasoning behind keeping this scoped "
                    "this narrowly."
                )
                for ticker in signal_tickers:
                    signal = results_df.loc[results_df["Ticker"] == ticker, "Signal"].iloc[0]
                    with st.expander(f"{ticker} ({signal})"):
                        with st.spinner(f"Summarizing recent news for {ticker}..."):
                            headlines = market_data.get_multi_headlines(ticker)
                            summary = ai_context.summarize_ticker_context(ticker, signal, headlines)
                        if summary:
                            st.write(summary)
                        elif headlines:
                            st.caption("AI summary unavailable -- raw headlines:")
                            for h in headlines:
                                st.write(f"- {h}")
                        else:
                            st.caption("No recent headlines found for this ticker.")

        st.download_button(
            "Download full results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="swing_orders.csv",
            mime="text/csv",
        )

        if skipped:
            with st.expander(f"Skipped tickers ({len(skipped)})"):
                st.dataframe(pd.DataFrame(skipped, columns=["Ticker", "Reason"]), hide_index=True)

    with tab_daily:
        st.warning(
            "**Experimental -- built to fire faster, not yet promoted.** Every strategy in "
            "the Mechanical Strategies tab was proven to beat matched-count random-entry "
            "timing on 5 years of held-out historical data before being trusted with real "
            "capital (see improvements.txt items 24-28). Strategies here HAVE already been "
            "run through that same offline validation -- see improvements.txt for each "
            "one's actual result -- but ship experimental regardless of outcome until "
            "explicitly promoted. **Never used for capital allocation** -- no cash pool, "
            "no allocate_capital() call, no matter how good a backtest result looks."
        )
        if not EXPERIMENTAL_STRATEGY_VERSIONS:
            st.caption("No experimental strategies configured.")
        for label, version in EXPERIMENTAL_STRATEGY_VERSIONS.items():
            experimental_config, experimental_source = experimental_configs[label]
            if experimental_config is None:
                st.caption(f"{label} (v{version}): unavailable -- {experimental_source}")
                continue
            render_experimental_section(
                f"{label} (v{version}, experimental)",
                experimental_config,
                bundle, market_df, fetch_skipped,
                storage_ok,
            )

    with tab2:
        st.subheader("LLM Agent (experimental)")
        st.warning(
            "**Experimental -- not mechanically validated.** Every strategy in the "
            "Mechanical Strategies tab was proven to beat matched-count random-entry "
            "timing on 5 years of held-out historical data before being trusted (see "
            "improvements.txt items 24-28). An LLM judgment CANNOT be validated that "
            "way -- re-prompting a model with 'historical' context risks it already "
            "knowing what happened next from training data. This can only be validated "
            "PROSPECTIVELY: real time passing, real settled trades -- see the progress "
            "counter below. **Never used for capital allocation** -- no cash pool, no "
            "allocate_capital() call, regardless of how confident a call looks."
        )

        if not llm_agent.is_available():
            st.caption(
                "Unavailable: set GEMINI_API_KEY (same key used by the AI Context "
                "feature in the other tab; primary provider) and/or GROQ_API_KEY "
                "(fallback provider, used only if Gemini is unavailable or fails) "
                "to enable -- either one alone is enough."
            )
        else:
            # Candidate pre-filter: never independently scans the full watchlist --
            # only tickers at least one mechanical strategy already found interesting
            # today, capped at MAX_LLM_CANDIDATES. See llm_agent.py's module
            # docstring for why (cost/rate-limit control, and framing this as a
            # second opinion rather than a blind scan).
            mechanical_frames = {"Primary (v19)": results_df, **secondary_results_by_label}
            candidates: dict[str, dict] = {}
            for strategy_label, df in mechanical_frames.items():
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
                st.caption(
                    "No tickers scored above Ignore under any mechanical strategy today -- "
                    "nothing for the LLM Agent to weigh in on (see the Mechanical Strategies "
                    "tab). Genuinely empty, not a bug."
                )
            else:
                st.caption(
                    f"Evaluating {len(ranked_candidates)} ticker(s) flagged by at least one "
                    "mechanical strategy today, highest mechanical Trade_Score first."
                )
                llm_config = swingtrade.TradingConfig(**{**config.to_dict(), "strategy": "llm_agent"})
                # Fetched ONCE for the whole tab, not per ticker -- see
                # cached_macro_snapshot()/market_data.get_macro_snapshot().
                # Every candidate this run shares the same macro backdrop.
                macro_snapshot = cached_macro_snapshot()
                llm_rows = []
                for ticker, entry in ranked_candidates:
                    row = entry["row"]
                    fundamentals = {}
                    try:
                        info = yf.Ticker(ticker).info
                        fundamentals = {
                            k: info[k] for k in ("trailingPE", "marketCap", "sector") if info.get(k) is not None
                        }
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
                    }

                    with st.spinner(f"Evaluating {ticker}..."):
                        verdict = llm_agent.evaluate_ticker(ticker, context)

                    with st.expander(f"{ticker} -- {verdict['decision'] if verdict else 'unavailable'}"):
                        if verdict is None:
                            st.caption("LLM evaluation failed or returned an unusable response for this ticker.")
                            continue
                        st.write(
                            f"**{verdict['decision']}** (confidence: {verdict['confidence']:.0f}/100) -- "
                            f"news sentiment: **{verdict['news_sentiment']}**"
                        )
                        st.write(verdict["rationale"])

                        if verdict["decision"] in ("Buy", "Hold"):
                            atr = context["atr"]
                            buy_price = round(context["last_close"], 2)
                            sell_price = round(buy_price + llm_config.atr_take_profit_multiplier * atr, 2)
                            stop_loss = round(buy_price - llm_config.stop_loss_atr_multiplier * atr, 2)
                            risk = buy_price - stop_loss
                            rrr = round((sell_price - buy_price) / risk, 2) if risk > 0 else 0.0
                            llm_rows.append({
                                "Ticker": ticker, "As_Of": row["As_Of"], "Signal": verdict["decision"],
                                "Trade_Score": verdict["confidence"], "Last_Close": context["last_close"],
                                "Buy_Price": buy_price, "Sell_Price": sell_price, "Stop_Loss": stop_loss,
                                "RRR": rrr, "RSI": context["rsi"], "ATR": atr,
                                "Distance_to_Buy_Pct": 0.0, "Shares_To_Buy": 0.0, "Est_Cost": 0.0,
                                "Next_Earnings_Date": context["next_earnings_date"],
                                "Catalyst_Warning": context["catalyst_warning"], "Top_Headline": "",
                            })

                if llm_rows and storage_ok:
                    llm_df = pd.DataFrame(llm_rows)
                    # "Hold" isn't in the shared Strong Buy/Buy/Watch/Ignore vocabulary
                    # storage/signals.py expects -- map it to "Watch" (research tier),
                    # same actionable/research split every mechanical strategy uses.
                    llm_df["Signal"] = llm_df["Signal"].replace("Hold", "Watch")
                    try:
                        logged = storage.log_trade_signals(llm_df, llm_config.to_dict())
                        st.caption(
                            f"Logged {logged['actionable']} actionable + {logged['research']} "
                            "research signal(s) to MongoDB (strategy=llm_agent, never capital-allocated)."
                        )
                    except Exception as exc:
                        st.warning(f"Signal logging failed: {exc}")
                elif llm_rows:
                    st.caption("MongoDB not connected -- signals aren't being logged.")

        st.divider()
        st.subheader("Validation progress")
        st.caption(
            "Tracked exactly like every mechanical strategy's signals (same "
            "settle_trades.py grading), but can ONLY be trusted after real time "
            "passes. Proposed floor before drawing ANY conclusion: at least 20-30 "
            "settled trades AND at least 4-6 weeks since the first one -- roughly "
            "double optimize.py's own 15-trade trust floor, since prospective, "
            "LLM-noise data deserves a higher bar than a large backtested sample, "
            "not a lower one."
        )
        if storage_ok:
            try:
                db = storage.get_db()
                outcomes = list(db["Trade_Outcomes"].find({"strategy": "llm_agent"}))
                if not outcomes:
                    st.caption("0 settled llm_agent trades yet.")
                else:
                    first_exit_date = min(o["exit_date"] for o in outcomes)
                    days_elapsed = (pd.Timestamp.now().normalize() - pd.Timestamp(first_exit_date)).days
                    win_count = sum(1 for o in outcomes if o["status"] == "WIN")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Settled llm_agent trades", len(outcomes))
                    col2.metric("Days since first settled trade", days_elapsed)
                    col3.metric("Win rate so far", f"{win_count / len(outcomes) * 100:.1f}%")
            except Exception as exc:
                st.caption(f"Could not load validation progress: {exc}")
        else:
            st.caption("MongoDB not connected -- no progress to show.")


if __name__ == "__main__":
    main()
