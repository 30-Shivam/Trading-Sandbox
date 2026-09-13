"""
Multi-strategy portfolio-constrained backtest -- what happens when
multiple REAL, already-validated strategies share ONE real, finite-capital
account, instead of each being judged in isolation?

Every prior validation in this project (real-vs-random, DSR, backtest IC,
even swingtrade.simulate_portfolio_constrained() itself, item 138) judges
ONE strategy's own signals against ONE dedicated capital pool. That's not
how the user actually trades: ma_crossover (primary) and Mean-Reversion
Pairs (secondary) already share the SAME real account today. When both
fire on the same day, they compete for the same limited capital -- a
question nothing before this checked.

Method: run both strategies' own REAL, currently-live configs
(config_loader.load_active_config() for ma_crossover, System_Config v58
for Mean-Reversion Pairs) against the SAME real ticker universe/window,
tag each trade with its own strategy (every simulate_*_signals() trade
dict already carries a "signal" field for exactly this purpose). Then
compare THREE views via swingtrade.simulate_portfolio_constrained():
  1. ma_crossover ALONE, with the full starting_capital to itself
  2. pairs ALONE, with the full starting_capital to itself
  3. POOLED -- both together, sharing the SAME starting_capital
       (group_key="signal" attributes each taken trade to its own strategy)

The gap between (1)+(2)'s independent capture and (3)'s pooled capture is
the real cost of capital contention a siloed, one-strategy-at-a-time
validation can never surface.

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
    if pairs_config is None:
        print(f"[ERROR] Could not load Mean-Reversion Pairs config: {pairs_label}", file=sys.stderr)
        sys.exit(1)

    print(f"Strategy 1: ma_crossover ({ma_label})")
    print(f"Strategy 2: pairs ({pairs_label})\n")

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
    print(f"  {len(pairs_trades)} real signal(s)\n")

    if not ma_trades and not pairs_trades:
        print("[ERROR] Neither strategy produced any real signals on this universe/window.", file=sys.stderr)
        sys.exit(1)

    ma_solo = swingtrade.simulate_portfolio_constrained(
        ma_trades, args.starting_capital, args.position_budget,
        ma_config.max_sector_allocation_pct, ma_config.max_total_deployed_pct, sector_lookup,
    )
    pairs_solo = swingtrade.simulate_portfolio_constrained(
        pairs_trades, args.starting_capital, args.position_budget,
        pairs_config.max_sector_allocation_pct, pairs_config.max_total_deployed_pct, sector_lookup,
    )
    pooled_trades = ma_trades + pairs_trades
    pooled = swingtrade.simulate_portfolio_constrained(
        pooled_trades, args.starting_capital, args.position_budget,
        ma_config.max_sector_allocation_pct, ma_config.max_total_deployed_pct, sector_lookup,
        group_key="signal",
    )

    print(f"=== SOLO (each strategy with its OWN ${args.starting_capital:,.0f} account, ${args.position_budget:,.0f}/position) ===")
    print(f"  ma_crossover alone: {ma_solo}")
    print(f"  pairs alone:        {pairs_solo}")

    print(f"\n=== POOLED (BOTH strategies sharing ONE ${args.starting_capital:,.0f} account) ===")
    print(f"  {pooled}")

    print("\n=== WHAT CAPITAL CONTENTION ACTUALLY COST ===")
    solo_total_taken = ma_solo["n_taken"] + pairs_solo["n_taken"]
    pooled_total_taken = pooled["n_taken"]
    ma_taken_pooled = pooled.get("taken_by_group", {}).get("MA_Crossover", 0)
    pairs_taken_pooled = pooled.get("taken_by_group", {}).get("Pairs", 0)
    print(f"  If each strategy had its own dedicated ${args.starting_capital:,.0f} account: "
          f"{solo_total_taken} total trades taken ({ma_solo['n_taken']} ma_crossover + {pairs_solo['n_taken']} pairs).")
    print(f"  Sharing ONE real ${args.starting_capital:,.0f} account instead: {pooled_total_taken} total trades taken "
          f"({ma_taken_pooled} ma_crossover + {pairs_taken_pooled} pairs).")
    if solo_total_taken > 0:
        crowd_out_pct = round((1 - pooled_total_taken / solo_total_taken) * 100, 1)
        print(f"  Capital contention crowded out {crowd_out_pct}% of the trades that WOULD have been taken "
              "with unlimited/siloed capital -- a real, previously-unmeasured cost of running multiple "
              "strategies against one real account, exactly how the user actually trades today.")


if __name__ == "__main__":
    main()
