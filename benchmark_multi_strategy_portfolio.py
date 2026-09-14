"""
Multi-strategy portfolio-constrained backtest -- what happens when
multiple REAL, already-validated strategies share ONE real, finite-capital
account, instead of each being judged in isolation?

Every prior validation in this project (real-vs-random, DSR, backtest IC,
even swingtrade.simulate_portfolio_constrained() itself, item 138) judges
ONE strategy's own signals against ONE dedicated capital pool. That's not
how the user actually trades: ma_crossover (primary), RSI Mean-Reversion,
and Mean-Reversion Pairs (both secondary) already share the SAME real
account today. When more than one fires on the same day, they compete for
the same limited capital -- a question nothing before this checked.

2026-09-14: extended from its original 2-strategy scope (ma_crossover +
pairs) to all THREE currently-live, capital-eligible strategies on the
primary watchlist (RSI Mean-Reversion added) -- the real, complete roster
per config_loader.SECONDARY_STRATEGY_VERSIONS, not just a subset.

Method: run every live strategy's own REAL, currently-live config
(config_loader.load_active_config() for ma_crossover,
config_loader.SECONDARY_STRATEGY_VERSIONS for RSI/Pairs) against the SAME
real ticker universe/window, tag each trade with its own strategy (every
simulate_*_signals() trade dict already carries a "signal" field for
exactly this purpose). Then compare, via swingtrade.simulate_portfolio_constrained()
and swingtrade.compute_strategy_correlation():
  1. Each strategy ALONE, with the full starting_capital to itself
  2. POOLED -- all together, sharing the SAME starting_capital
       (group_key="signal" attributes each taken trade to its own strategy)
  3. Pairwise real-vs-real correlation across every strategy pair

The gap between (1)'s independent capture (summed) and (2)'s pooled
capture is the real cost of capital contention a siloed, one-strategy-at-
a-time validation can never surface. (3) answers a different question:
even with unlimited capital, do these strategies' real edges actually
come from different market moments, or the same ones?

Usage:
    python benchmark_multi_strategy_portfolio.py
    python benchmark_multi_strategy_portfolio.py --tickers AAPL,MSFT,NVDA --starting-capital 5000
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config_loader
import swingtrade
from run_backtest import fetch_history
from watchlist import read_ticker_sectors, read_tickers

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = SCRIPT_DIR / "watchlist.txt"
REQUEST_DELAY_SEC = 0.5
MARKET_INDEX_TICKER = "SPY"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Backtest window start (YYYY-MM-DD). Default: 3y before --end.")
    parser.add_argument("--end", default=None, help="Backtest window end (YYYY-MM-DD). Default: today.")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers to override watchlist.txt.")
    parser.add_argument("--starting-capital", type=float, default=10_000.0, help="Shared account starting capital.")
    parser.add_argument("--position-budget", type=float, default=250.0, help="Flat $ per position (matches ingest.py's DEFAULT_POSITION_BUDGET).")
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=365 * 3)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        if not WATCHLIST_FILE.exists():
            print(f"[ERROR] Watchlist file not found: {WATCHLIST_FILE}", file=sys.stderr)
            sys.exit(1)
        tickers = read_tickers(WATCHLIST_FILE)
    sector_lookup = read_ticker_sectors(WATCHLIST_FILE)

    print(f"Fetching {MARKET_INDEX_TICKER} (market-uptrend proxy)...")
    market_ohlcv = fetch_history(MARKET_INDEX_TICKER, start, end)
    if market_ohlcv.empty:
        print(f"[ERROR] No data returned for {MARKET_INDEX_TICKER}", file=sys.stderr)
        sys.exit(1)

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        try:
            df = fetch_history(ticker, start, end)
            if not df.empty:
                ticker_data[ticker] = df
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}", file=sys.stderr)
    print(f"Fetched {len(ticker_data)}/{len(tickers)} ticker(s), {start.date()}..{end.date()}.\n")
    if not ticker_data:
        print("[ERROR] No ticker data available.", file=sys.stderr)
        sys.exit(1)

    by_sector: dict[str, list[str]] = {}
    for t in ticker_data:
        by_sector.setdefault(sector_lookup.get(t, "Unknown"), []).append(t)
    sector_price_panels = {
        sector: pd.DataFrame({m: ticker_data[m]["Close"] for m in members})
        for sector, members in by_sector.items() if len(members) >= 2
    }

    ma_config, ma_label = config_loader.load_active_config()
    if ma_config.strategy != "ma_crossover":
        print(f"[WARN] Active config's strategy is {ma_config.strategy!r}, not ma_crossover -- "
              "using it anyway, but this script's own framing assumes ma_crossover is primary.", file=sys.stderr)
    pairs_config, pairs_label = config_loader.load_config_by_version(
        config_loader.SECONDARY_STRATEGY_VERSIONS["Mean-Reversion Pairs"]
    )
    rsi_config, rsi_label = config_loader.load_config_by_version(
        config_loader.SECONDARY_STRATEGY_VERSIONS["RSI Mean-Reversion"]
    )
    if pairs_config is None or rsi_config is None:
        print(f"[ERROR] Could not load a secondary config (pairs={pairs_label!r}, rsi={rsi_label!r}).", file=sys.stderr)
        sys.exit(1)

    print(f"Strategy 1: ma_crossover ({ma_label})")
    print(f"Strategy 2: pairs ({pairs_label})")
    print(f"Strategy 3: rsi ({rsi_label})\n")

    print("Simulating ma_crossover...")
    ma_trades = []
    for ticker, ohlcv in ticker_data.items():
        ma_trades.extend(swingtrade.simulate_ma_crossover_signals(
            ticker, ohlcv, market_ohlcv, start, end, ma_config, sector=sector_lookup.get(ticker, "Unknown"),
        ))
    print(f"  {len(ma_trades)} real signal(s)")

    print("Simulating pairs...")
    pairs_trades = []
    for ticker, ohlcv in ticker_data.items():
        sector = sector_lookup.get(ticker, "Unknown")
        panel = sector_price_panels.get(sector)
        peer_prices = panel.drop(columns=[ticker]) if panel is not None and ticker in panel.columns else None
        pairs_trades.extend(swingtrade.simulate_pairs_signals(
            ticker, ohlcv, market_ohlcv, start, end, pairs_config, sector=sector, peer_prices=peer_prices,
        ))
    print(f"  {len(pairs_trades)} real signal(s)")

    print("Simulating rsi...")
    rsi_trades = []
    for ticker, ohlcv in ticker_data.items():
        rsi_trades.extend(swingtrade.simulate_signals(
            ticker, ohlcv, market_ohlcv, start, end, rsi_config, sector=sector_lookup.get(ticker, "Unknown"),
        ))
    print(f"  {len(rsi_trades)} real signal(s)\n")

    trades_by_strategy = {"MA_Crossover": ma_trades, "Pairs": pairs_trades, "RSI": rsi_trades}
    configs_by_strategy = {"MA_Crossover": ma_config, "Pairs": pairs_config, "RSI": rsi_config}

    if not any(trades_by_strategy.values()):
        print("[ERROR] No strategy produced any real signals on this universe/window.", file=sys.stderr)
        sys.exit(1)

    solo_results = {
        label: swingtrade.simulate_portfolio_constrained(
            trades, args.starting_capital, args.position_budget,
            configs_by_strategy[label].max_sector_allocation_pct,
            configs_by_strategy[label].max_total_deployed_pct, sector_lookup,
        )
        for label, trades in trades_by_strategy.items()
    }
    pooled_trades = [t for trades in trades_by_strategy.values() for t in trades]
    pooled = swingtrade.simulate_portfolio_constrained(
        pooled_trades, args.starting_capital, args.position_budget,
        ma_config.max_sector_allocation_pct, ma_config.max_total_deployed_pct, sector_lookup,
        group_key="signal",
    )

    print(f"=== SOLO (each strategy with its OWN ${args.starting_capital:,.0f} account, ${args.position_budget:,.0f}/position) ===")
    for label, result in solo_results.items():
        print(f"  {label} alone: {result}")

    print(f"\n=== POOLED (ALL {len(trades_by_strategy)} strategies sharing ONE ${args.starting_capital:,.0f} account) ===")
    print(f"  {pooled}")

    print("\n=== WHAT CAPITAL CONTENTION ACTUALLY COST ===")
    solo_total_taken = sum(r["n_taken"] for r in solo_results.values())
    pooled_total_taken = pooled["n_taken"]
    taken_by_group = pooled.get("taken_by_group", {})
    solo_breakdown = " + ".join(f"{solo_results[label]['n_taken']} {label}" for label in trades_by_strategy)
    pooled_breakdown = " + ".join(f"{taken_by_group.get(label, 0)} {label}" for label in trades_by_strategy)
    print(f"  If each strategy had its own dedicated ${args.starting_capital:,.0f} account: "
          f"{solo_total_taken} total trades taken ({solo_breakdown}).")
    print(f"  Sharing ONE real ${args.starting_capital:,.0f} account instead: {pooled_total_taken} total trades taken "
          f"({pooled_breakdown}).")
    if solo_total_taken > 0:
        crowd_out_pct = round((1 - pooled_total_taken / solo_total_taken) * 100, 1)
        print(f"  Capital contention crowded out {crowd_out_pct}% of the trades that WOULD have been taken "
              "with unlimited/siloed capital -- a real, previously-unmeasured cost of running multiple "
              "strategies against one real account, exactly how the user actually trades today.")

    # 2026-09-13 (improvements.txt item 149) -- a DIFFERENT question from
    # capital contention above: even with UNLIMITED capital (no fighting
    # over the same dollars at all), do these strategies make/lose money
    # on the SAME days for the SAME reasons? Two individually validated
    # strategies that are highly correlated add much less real
    # diversification than their separate validation reports suggest on
    # their own. See swingtrade.compute_strategy_correlation()'s own
    # docstring for the full method. Extended 2026-09-14 to all 3 live
    # strategies (full pairwise matrix), not just the original 2.
    correlation_result = swingtrade.compute_strategy_correlation(trades_by_strategy)
    print("\n=== STRATEGY CORRELATION (are these real edges actually diversified, or redundant?) ===")
    for pair_label, pair_stats in correlation_result["pairs"].items():
        if pair_stats["correlation"] is not None:
            print(f"  {pair_label}: correlation={pair_stats['correlation']}, n_days={pair_stats['n_days']}, "
                  f"ticker_day_overlap_pct={pair_stats['ticker_day_overlap_pct']}%")
        else:
            print(f"  {pair_label}: {pair_stats} (not enough combined activity for a real correlation)")
    print("  A correlation near 0 means two strategies' P&L genuinely comes from different market moments --")
    print("  real diversification. A correlation pushing toward 1.0 would mean combining them adds much less")
    print("  true risk reduction than running either one alone at double size.")


if __name__ == "__main__":
    main()
