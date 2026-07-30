"""Walk-Forward backtesting harness (Phase 4). Pure -- takes pre-fetched
historical OHLCV DataFrames in, returns simulated trades and fold-level
metrics out. Reuses compute_levels / add_trade_score / settle_trade /
is_market_uptrend exactly as the live dashboard and nightly settlement job
do, so a backtested TradingConfig is scored by identical rules to a live
signal -- that's the whole point of Phase 1 having pulled this math out of
the Streamlit app in the first place.

Walk-Forward Optimization, not a single static window: a backtest over one
fixed historical range and calling the winner "optimal" is a curve-fitting
trap -- it rewards parameters that memorized that period's noise. Instead,
generate_folds() produces a sequence of rolling in-sample/out-of-sample
windows; Optuna (Phase 5) will score a candidate config on the AGGREGATE
out-of-sample performance across all folds, never a single fold's in-sample
fit. See ARCHITECTURE_PLAN.md amendment #3 for the full rationale.

No look-ahead: on each simulated day `as_of`, only bars up to and including
`as_of` are used to decide whether a signal fires (mirrors the live app,
where the last row of fetched data IS "today"). Settling a trade started on
`as_of` is free to use bars strictly after `as_of` -- grading a past
decision against realized future prices is not look-ahead bias, it's just
how you find out whether the trade won.

Catalyst awareness IS simulated, given an optional per-ticker
`earnings_dates` (see run_backtest.fetch_earnings_dates): yfinance's
get_earnings_dates(limit=40) returns roughly a decade of REPORTED earnings
dates, not just upcoming estimates. Only the calendar date is used, never
the EPS/Surprise columns -- once a company schedules a report, the date
itself is a fixed fact that doesn't get revised after the fact the way an
EPS estimate does, so looking up "the next earnings date after `as_of`"
from that static list introduces no look-ahead leakage even for a backtest
day years in the past. Without `earnings_dates` passed in, Catalyst_Warning
is always False (the old behavior) rather than raising -- callers that
don't need catalyst simulation aren't forced to fetch it.
"""

from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_CONFIG, TradingConfig
from .levels import compute_levels, is_market_uptrend
from .scoring import add_trade_score
from .settlement import settle_trade

ENTRY_SIGNALS = ("Strong Buy", "Buy")
LOOKBACK_BUFFER_BARS = 60   # extra trailing bars beyond sma_trend_window, for indicator warmup safety


@dataclass(frozen=True)
class Fold:
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_sample_start: pd.Timestamp
    out_sample_end: pd.Timestamp


def generate_folds(
    start,
    end,
    in_sample_days: int = 182,
    out_sample_days: int = 30,
    step_days: int = 30,
) -> list[Fold]:
    """Rolling walk-forward folds over [start, end): each fold trains on
    `in_sample_days` and validates on the immediately following
    `out_sample_days` (no gap, no overlap between the two), then the whole
    window slides forward by `step_days` and repeats. Stops once a fold
    would extend past `end`."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    folds = []
    cursor = start
    while True:
        in_end = cursor + pd.Timedelta(days=in_sample_days)
        out_end = in_end + pd.Timedelta(days=out_sample_days)
        if out_end > end:
            break
        folds.append(Fold(cursor, in_end, in_end, out_end))
        cursor = cursor + pd.Timedelta(days=step_days)
    return folds


def _next_earnings_date(earnings_dates: pd.DatetimeIndex | None, as_of: pd.Timestamp):
    """Mirrors market_data.get_next_earnings_date's live semantics (the
    earliest date >= as_of), just reading from a static historical list
    instead of a fresh API call. `earnings_dates` must be tz-aware UTC (see
    run_backtest.fetch_earnings_dates) -- `as_of` is localized to UTC here
    to match, the same way compute_levels handles its own as_of internally."""
    if earnings_dates is None or len(earnings_dates) == 0:
        return None
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    future = earnings_dates[earnings_dates >= as_of]
    return future.min() if len(future) > 0 else None


def simulate_signals(
    ticker: str,
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_dates: pd.DatetimeIndex | None = None,
) -> list[dict]:
    """Walk every trading day in [window_start, window_end) for one ticker,
    simulating a trade wherever swingtrade would fire a Strong Buy/Buy
    signal using only data available as of that day. Each simulated trade
    is resolved with settle_trade() against the ticker's actual subsequent
    price history (which may extend past window_end).

    `ohlcv` and `market_ohlcv` must each contain enough leading history
    before window_start to cover config.sma_trend_window -- see
    LOOKBACK_BUFFER_BARS and run_backtest.py's fetch buffering.

    `earnings_dates` (optional, tz-aware UTC, see run_backtest.fetch_earnings_dates)
    lets Catalyst_Warning be computed honestly instead of always False.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    lookback_bars = config.sma_trend_window + LOOKBACK_BUFFER_BARS

    trades = []
    eligible_dates = ohlcv.index[(ohlcv.index >= window_start) & (ohlcv.index < window_end)]

    for as_of in eligible_dates:
        market_window = market_ohlcv.loc[:as_of].tail(lookback_bars)
        try:
            market_uptrend, _, _ = is_market_uptrend(market_window, config)
        except RuntimeError:
            continue  # insufficient market history yet
        if not market_uptrend:
            continue

        price_window = ohlcv.loc[:as_of].tail(lookback_bars)
        next_earnings = _next_earnings_date(earnings_dates, as_of)
        try:
            levels = compute_levels(ticker, price_window, config, next_earnings_date=next_earnings)
        except RuntimeError:
            continue  # insufficient history / macro downtrend / illiquid that day

        scored = add_trade_score(pd.DataFrame([levels]), config).iloc[0]
        if scored["Signal"] not in ENTRY_SIGNALS:
            continue

        bars_since_entry = ohlcv[ohlcv.index > as_of]
        result = settle_trade(
            buy_price=scored["Buy_Price"],
            stop_loss=scored["Stop_Loss"],
            sell_price=scored["Sell_Price"],
            bars_since_entry=bars_since_entry,
            config=config,
        )

        trades.append({
            "ticker": ticker,
            "entry_date": as_of.date(),
            "signal": scored["Signal"],
            "trade_score": float(scored["Trade_Score"]),
            "rsi": float(scored["RSI"]),
            "atr": float(scored["ATR"]),
            "rrr": float(scored["RRR"]),
            "buy_price": float(scored["Buy_Price"]),
            "stop_loss": float(scored["Stop_Loss"]),
            "sell_price": float(scored["Sell_Price"]),
            "catalyst_warning": bool(scored["Catalyst_Warning"]),
            **result,
        })

    return trades


def run_backtest(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    window_start,
    window_end,
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
) -> list[dict]:
    """Simulate signals for every ticker in ticker_data over
    [window_start, window_end), settling each against its own subsequent
    history. Returns the combined trade list. `earnings_data` (optional,
    ticker -> tz-aware UTC DatetimeIndex, see run_backtest.fetch_earnings_dates)
    enables honest Catalyst_Warning simulation; without it every trade has
    Catalyst_Warning=False."""
    earnings_data = earnings_data or {}
    all_trades = []
    for ticker, ohlcv in ticker_data.items():
        all_trades.extend(
            simulate_signals(
                ticker, ohlcv, market_data, window_start, window_end, config,
                earnings_dates=earnings_data.get(ticker),
            )
        )
    return all_trades


def summarize_trades(trades: list[dict]) -> dict:
    """Aggregate simulated trades into summary performance metrics. OPEN
    trades (not yet resolved as of the available data) are excluded from
    PnL stats but counted separately."""
    resolved = [t for t in trades if t["status"] != "OPEN"]
    open_count = len(trades) - len(resolved)

    if not resolved:
        return {
            "trade_count": 0, "open_count": open_count,
            "win_count": 0, "loss_count": 0, "expired_count": 0,
            "win_rate": None, "avg_pnl_pct": None, "total_pnl_pct": 0.0,
            "pnl_std": None, "sharpe_like": None,
        }

    pnls = pd.Series([t["pnl_pct"] for t in resolved])
    win_count = sum(1 for t in resolved if t["status"] == "WIN")
    loss_count = sum(1 for t in resolved if t["status"] == "LOSS")
    expired_count = sum(1 for t in resolved if t["status"] == "EXPIRED")
    pnl_std = float(pnls.std()) if len(pnls) > 1 else 0.0

    return {
        "trade_count": len(resolved),
        "open_count": open_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "expired_count": expired_count,
        "win_rate": round(win_count / len(resolved) * 100, 2),
        "avg_pnl_pct": round(float(pnls.mean()), 2),
        "total_pnl_pct": round(float(pnls.sum()), 2),
        "pnl_std": round(pnl_std, 2),
        "sharpe_like": round(float(pnls.mean() / pnl_std), 3) if pnl_std > 0 else None,
    }


def summarize_by_catalyst(trades: list[dict]) -> dict:
    """Split summarize_trades() output by whether each trade carried a
    Catalyst_Warning at entry -- lets you actually see whether performance
    differs around earnings instead of assuming it doesn't. Requires trades
    simulated with earnings_dates passed through (see run_backtest.py
    --with-catalyst); without that, every trade has catalyst_warning=False
    and this just reports everything under the false bucket."""
    with_catalyst = [t for t in trades if t.get("catalyst_warning")]
    without_catalyst = [t for t in trades if not t.get("catalyst_warning")]
    return {
        "catalyst_warning_true": summarize_trades(with_catalyst),
        "catalyst_warning_false": summarize_trades(without_catalyst),
    }


@dataclass
class FoldResult:
    fold: Fold
    in_sample_trades: list[dict]
    out_sample_trades: list[dict]
    in_sample_metrics: dict
    out_sample_metrics: dict


def run_walk_forward(
    ticker_data: dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    folds: list[Fold],
    config: TradingConfig = DEFAULT_CONFIG,
    earnings_data: dict[str, pd.DatetimeIndex] | None = None,
) -> list[FoldResult]:
    """Run the backtest across every fold, in-sample and out-of-sample
    separately. Optuna (Phase 5) evaluates a candidate config by scoring it
    across the aggregate of every fold's out_sample_metrics -- never a
    single fold's in-sample fit."""
    results = []
    for fold in folds:
        in_trades = run_backtest(
            ticker_data, market_data, fold.in_sample_start, fold.in_sample_end, config, earnings_data
        )
        out_trades = run_backtest(
            ticker_data, market_data, fold.out_sample_start, fold.out_sample_end, config, earnings_data
        )
        results.append(FoldResult(
            fold=fold,
            in_sample_trades=in_trades,
            out_sample_trades=out_trades,
            in_sample_metrics=summarize_trades(in_trades),
            out_sample_metrics=summarize_trades(out_trades),
        ))
    return results
