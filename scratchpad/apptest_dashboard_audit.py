"""AppTest-driven audit of the Best Ideas tab (post-fix composite) and the
Daily Signals (experimental) tab (Value Rank / Accruals Rank live-wiring),
using Streamlit's own official testing framework -- the same tool this
project's history already established for dashboard verification (see
memory: "Verified via Streamlit AppTest"). Temporarily swaps watchlist.txt
for a small 10-ticker sample so the app's own default load (the FIRST
at.run(), before any widget override is possible) doesn't scan the full
415-ticker watchlist and time out. Restores the original file in finally.
"""
import json
import os
import sys

WATCHLIST_PATH = "watchlist.txt"
SMALLMID_WATCHLIST_PATH = "smallmid_watchlist.txt"
SMALL_SAMPLE = {
    "watchlist": [
        {"ticker": "AAPL", "sector": "Technology", "strategy": "Core/Stable"},
        {"ticker": "MSFT", "sector": "Technology", "strategy": "Core/Stable"},
        {"ticker": "NVDA", "sector": "Technology", "strategy": "Growth/AI"},
        {"ticker": "JPM", "sector": "Financials", "strategy": "Core/Stable"},
        {"ticker": "XOM", "sector": "Energy", "strategy": "Core/Stable"},
        {"ticker": "INTC", "sector": "Technology", "strategy": "Value"},
        {"ticker": "CSCO", "sector": "Technology", "strategy": "Value"},
        {"ticker": "KO", "sector": "Consumer Staples", "strategy": "Core/Stable"},
        {"ticker": "WMT", "sector": "Consumer Staples", "strategy": "Core/Stable"},
        {"ticker": "GE", "sector": "Industrials", "strategy": "Core/Stable"},
    ]
}
SMALLMID_SMALL_SAMPLE = {
    "watchlist": [
        {"ticker": "AAP", "sector": "Consumer Discretionary"},
        {"ticker": "AAT", "sector": "Real Estate"},
        {"ticker": "AAMI", "sector": "Financials"},
    ]
}

sys.path.insert(0, os.path.abspath("."))

with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
    original_watchlist_text = f.read()
with open(SMALLMID_WATCHLIST_PATH, "r", encoding="utf-8") as f:
    original_smallmid_text = f.read()

try:
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(SMALL_SAMPLE, f, indent=2)
    with open(SMALLMID_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(SMALLMID_SMALL_SAMPLE, f, indent=2)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dip_buy_analyzer.py", default_timeout=300)
    at.run()
    if at.exception:
        print("EXCEPTION ON INITIAL LOAD:")
        for e in at.exception:
            print(e)
        sys.exit(1)

    print("\n=== TABS FOUND ===")
    for tab in at.tabs:
        print(" -", tab.label if hasattr(tab, "label") else tab)

    print("\n=== ALL EXCEPTIONS ===", len(at.exception))
    for e in at.exception:
        print(e)

    # Best Ideas tab is index 2 (Daily Briefing, Mechanical Strategies, Best Ideas, Daily Signals, LLM Agent)
    best_ideas_tab = at.tabs[2]
    print("\n=== BEST IDEAS TAB CONTENT ===")
    print("dataframes:", len(best_ideas_tab.dataframe))
    print("headers/subheaders:", [h.value for h in best_ideas_tab.subheader] + [h.value for h in best_ideas_tab.header])
    print("captions (first 10):", [c.value[:150] for c in best_ideas_tab.caption][:10])
    print("markdown (first 5):", [m.value[:200] for m in best_ideas_tab.markdown][:5])
    if best_ideas_tab.dataframe:
        df = best_ideas_tab.dataframe[0].value
        print("first dataframe columns:", list(df.columns))
        print(df.head(5).to_string())

    daily_signals_tab = at.tabs[3]
    print("\n=== DAILY SIGNALS (EXPERIMENTAL) TAB CONTENT ===")
    print("subheaders:", [h.value for h in daily_signals_tab.subheader])
    print("dataframes:", len(daily_signals_tab.dataframe))
    for i, df_el in enumerate(daily_signals_tab.dataframe):
        df = df_el.value
        print(f"--- dataframe {i}: columns={list(df.columns)} rows={len(df)}")
        if "Value_Percentile" in df.columns or "Accruals_Percentile" in df.columns:
            print(df.head(5).to_string())
    print("captions (first 15):", [c.value[:200] for c in daily_signals_tab.caption][:15])

finally:
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        f.write(original_watchlist_text)
    with open(SMALLMID_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        f.write(original_smallmid_text)
    print("\n[watchlist.txt and smallmid_watchlist.txt restored to original content]")
