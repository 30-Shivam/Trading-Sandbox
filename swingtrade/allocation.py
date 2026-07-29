"""Pure capital-allocation logic. Operates on a Trade_Score-sorted DataFrame
with Signal/Est_Cost columns already populated -- no fetching, no UI.
"""

import pandas as pd


def allocate_capital(
    df: pd.DataFrame,
    total_cash: float,
    sector_lookup: dict[str, str] | None = None,
    max_sector_allocation_pct: float | None = None,
    existing_holdings: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Greedily allocate total_cash down the (already Trade_Score-sorted) list
    of Strong Buy / Buy signals, in order. A trade whose Est_Cost exceeds the
    remaining cash is relabeled Insufficient Funds -- without consuming any
    cash or stopping the walk, so a cheaper trade further down the list can
    still be funded.

    If `sector_lookup` (ticker -> sector) and a positive `max_sector_allocation_pct`
    are both given, a trade that would push its sector's cumulative spend past
    that fraction of total_cash is instead relabeled Sector Limit Reached --
    same non-consuming, keep-walking treatment as Insufficient Funds, just a
    different reason. This exists because Trade_Score-ranked greedy allocation
    has no notion of correlation: five Strong Buys in the same sector on the
    same day are one concentrated bet wearing five tickers, not five
    independent ones, and nothing else in this pipeline catches that (a
    per-ticker walk-forward backtest doesn't model simultaneous cross-ticker
    exposure either). A ticker missing from sector_lookup is never capped --
    there's no sector to check it against.

    `existing_holdings` (ticker -> dollars already committed, see
    storage.holdings) counts toward the sector cap on BOTH sides: it inflates
    the denominator (the cap is a fraction of total_cash + total holdings --
    your whole portfolio, not just today's fresh cash pool in isolation) and
    seeds each sector's already-spent total. A sector you're already
    overweight in from prior holdings will correctly get little or no new
    room today, even though existing_holdings never reduces `remaining_cash`
    itself -- holdings aren't fresh cash, they're already-deployed capital.

    Returns the updated DataFrame and total capital spent (today's fresh
    cash only, not counting existing holdings)."""
    df = df.copy()
    sector_lookup = sector_lookup or {}
    existing_holdings = existing_holdings or {}

    portfolio_value = total_cash + sum(existing_holdings.values())
    sector_cap_dollars = (
        max_sector_allocation_pct * portfolio_value
        if max_sector_allocation_pct and max_sector_allocation_pct > 0
        else None
    )

    remaining_cash = total_cash
    sector_spent: dict[str, float] = {}
    if sector_cap_dollars is not None:
        for ticker, amount in existing_holdings.items():
            sector = sector_lookup.get(ticker)
            if sector:
                sector_spent[sector] = sector_spent.get(sector, 0.0) + amount

    tickers = df["Ticker"].tolist()
    signals = df["Signal"].tolist()
    costs = df["Est_Cost"].tolist()

    for i in range(len(df)):
        if signals[i] not in ("Strong Buy", "Buy"):
            continue
        cost = costs[i]
        if cost > remaining_cash:
            signals[i] = "Insufficient Funds"
            continue

        sector = sector_lookup.get(tickers[i])
        if sector and sector_cap_dollars is not None:
            spent_in_sector = sector_spent.get(sector, 0.0)
            if spent_in_sector + cost > sector_cap_dollars:
                signals[i] = "Sector Limit Reached"
                continue

        remaining_cash -= cost
        if sector:
            sector_spent[sector] = sector_spent.get(sector, 0.0) + cost

    df["Signal"] = signals
    capital_allocated = round(total_cash - remaining_cash, 2)
    return df, capital_allocated
