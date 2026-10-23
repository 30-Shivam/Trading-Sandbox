# Cheapest fix for the survivorship / point-in-time data problem (researched 2026-10-23)

> UPDATE 2026-10-23 (later): Tiingo free key tested - delisted history is served for some names (TWTR, ATVI, SIVB) but not all (FRC missing, BBBY empty); bulk pull of all 181 names in progress. Also found a worse bug than missing delisted prices: the ratio-model dataset included filings from BEFORE a stock joined the index (look-ahead). The 'promising' ridge result was mostly that artifact; see docs/BACKLOG.md correction. Probability table below pre-dates this and is optimistic for the ridge.

## What is actually missing (measured, not assumed)
- Point-in-time S&P 500 members 2010-2025 (hanshof CSV): **752 unique names**.
- Yahoo returns usable history for **571 (76%)**; **181 (24%) have none** (delisted/acquired/renamed). Coverage by membership year: 73% (2010) -> 96% (2025). Early years, where long-horizon tests live, are worst.
- Of the 181 missing, **only 1 has a current SEC ticker->CIK entry**, so fundamentals (EDGAR) are also unreachable for them. Combined effect: our ratio-model dataset sees **394 of 752 (52%)** of the historical index.
- Direction of bias: unknown per name (bankruptcies/forced exits are negative; acquisition targets get a positive premium before delisting), but earlier evidence says survivor/lookahead contamination can be huge: factor_portfolios momentum +6.0%/yr on today's constituents vs -1.5%/yr point-in-time (about 7.5%/yr of illusion).
- What we do NOT need to buy: fundamentals (EDGAR as-filed, free), index membership (hanshof CSV, free), macro (FRED, free). The gap is **delisted PRICES + a historical ticker->CIK bridge**.

## Options, cheapest first (prices verified from vendor pages 2026-10-23 unless marked)
| # | Option | Cost | Fixes | Residual gaps / risks | Confidence |
|---|---|---|---|---|---|
| 0 | **Tiingo free "Starter"** key (50 req/h, 1,000/day, **500 unique symbols/month**, 30+ yrs history, personal use) | $0 | 181 names fit in one month (~4 h of calls) IF delisted history is served | Docs I could read do NOT say delisted tickers are served; only a third-party comparison does. **Unverified.** 5-minute test: free key, request SIVB, TWTR, ATVI, FRC, BBBY. | Unverified |
| 1 | **EODHD "Historian"** EOD plan, pay ONE month, bulk-download the 181+ names, cancel | **$19.99** (annual $199) | Delisted EOD incl. dividend-adjusted close, 30+ yrs; their docs cite 60,191 delisted US symbols | Which plan can read delisted symbols is not stated on the pricing page (verify with one call after paying, or ask support first); users report reused tickers return the NEW company (resolve via their delisted symbol list + dates); 500 welcome calls, then 100k/day | Medium |
| 2 | **MarketParquet** "Complete Archive" | **$79 once** (optional $15/mo to keep current) | 16,000+ US stocks incl. **8,500 delisted**, 2000-now, daily + intraday, Parquet, perpetual use, personal backtesting + derived results allowed, raw redistribution forbidden | Split-adjusted ONLY (no dividends: matters for total-return comparisons, can be patched with EDGAR dividends or yfinance for survivors), starts 2000 (nothing for 1990s), no constituents, all sales final | High (vendor-stated; quality not independently verified) |
| 3 | Norgate Data US Stocks **Platinum** | ~$346 per 6 months / $630 per 12 (Gold has no delisted; check tier text) | Delisted + historical index constituents + total-return series, 1990-now | Costliest, subscription | Medium |
| 4 | Sharadar Core US Equities (Nasdaq Data Link) | not verified (page unreadable to me) | Prices + point-in-time fundamentals incl. delisted | Price unknown; likely more than #1/#2 | Low |
| 5 | FMP / Polygon(Massive) / Alpha Vantage | FMP free 250 req/day, 5y history; Polygon free 2y; paid $22-99/mo | partial | Free tiers too short; delisted coverage on free tiers unverified | Low |
| 6 | University WRDS/CRSP day-pass | $0 if you have a library affiliation | Gold-standard CRSP incl. delisting returns | Needs institutional access | Situational |

**The hidden second cost (any price source): historical ticker -> CIK for the ~180 delisted names.** Free route: SEC `company_tickers.json` is current-only, so map via company NAME using SEC bulk `submissions.zip`/`companyfacts.zip` entity names (+ formerNames). This is exactly the kind of glue an LLM subagent does well (ticker -> company name from the S&P changes history, then fuzzy-match to EDGAR names, with a manual spot-check of ~20). Cost: tokens + an hour, not dollars.

## Recommended path (cheapest credible)
1. **$0, 5 minutes (needs you):** make a free Tiingo key and test the 5 delisted tickers above. If they return history, we are done for $0 (500 symbols/month covers the 181).
2. **If not, $19.99 (EODHD Historian, one month):** after payment, call the delisted symbol list for a US exchange + one delisted ticker first; if it works, pull all 181 + a margin, cache locally, cancel.
3. **Fallback / small-cap unlock, $79 (MarketParquet):** also gives the whole 2000-now US universe (needed for small-cap tests, where survivorship is worst and a small account has a capacity edge).
Total worst case through step 3: ~$99. Skip Norgate/Sharadar unless these fail.

## Projected impact on future success (judgment, not measurement; every number is an estimate)
What cleaner data changes:
- **Trust, not edge.** Survivorship bias manufactures false positives; it does not hide real edges. Fixing it cannot create alpha, it tells us whether the leads we have (sec_ratio_ridge, accruals/value, factor ETFs overlays) are real. Today any positive test result is suspect by construction.
- **Coverage:** ratio-model universe 52% -> roughly 90-95% of historical index members (after the CIK bridge).
- **New test space (needs option 2):** small/mid-cap fundamentals strategies on a full-universe point-in-time basis. Anomalies are strongest there and a small account has no capacity problem, but survivorship is worst there, so it is currently untestable.
Subjective probabilities over the next ~3 months of work (rough, wide error bars):
| Event | Current data | With clean S&P 500 data | + clean small-cap data |
|---|---|---|---|
| sec_ratio_ridge passes ALL gates (DD <= SPY, p < Bonferroni, holdout) | 3-5% | 8-12% | 8-12% |
| ANY strategy passes all gates vs SPY | 3-5% | 8-12% | 12-18% |
| We avoid deploying a false positive that looks good only because of survivorship | not guaranteed | high | high |
Reading it: the data fix roughly doubles-to-triples a small chance and, more importantly, makes a "pass" believable. Expected cost $0-$99 vs the cost of acting on a biased result. The ratio model is the only lead with a plausible path; my robustness work shows its edge is thin (126-day only, tail-driven, fading since 2022), so even clean data most likely shrinks it.

## Not recommended
Paying for LLM/sentiment data (no point-in-time history exists; blind-LLM test already failed), real-time feeds (irrelevant to a daily swing system), or Norgate/Sharadar before steps 0-2 are exhausted.
