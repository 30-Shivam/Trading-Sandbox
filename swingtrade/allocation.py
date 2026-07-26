"""Pure capital-allocation logic. Operates on a Trade_Score-sorted DataFrame
with Signal/Est_Cost columns already populated -- no fetching, no UI.
"""

import pandas as pd


def allocate_capital(df: pd.DataFrame, total_cash: float) -> tuple[pd.DataFrame, float]:
    """Greedily allocate total_cash down the (already Trade_Score-sorted) list
    of Strong Buy / Buy signals, in order. A trade whose Est_Cost exceeds the
    remaining cash is relabeled Insufficient Funds -- without consuming any
    cash or stopping the walk, so a cheaper trade further down the list can
    still be funded. Returns the updated DataFrame and total capital spent."""
    df = df.copy()
    remaining_cash = total_cash
    signals = df["Signal"].tolist()
    costs = df["Est_Cost"].tolist()
    for i in range(len(df)):
        if signals[i] not in ("Strong Buy", "Buy"):
            continue
        if costs[i] <= remaining_cash:
            remaining_cash -= costs[i]
        else:
            signals[i] = "Insufficient Funds"
    df["Signal"] = signals
    capital_allocated = round(total_cash - remaining_cash, 2)
    return df, capital_allocated
