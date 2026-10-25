# Odd-lot tender offer feasibility screen -- FROZEN SPEC (written 2026-10-24, before any results)

Descriptive screen, NOT a hypothesis test: 0 trials registered in `validation_trials_registry.txt`. Payoffs are
near-deterministic accounting, so no t-stat applies. Purpose: does a tradable, profitable set of odd-lot tender
opportunities exist at all for a small account?

## Universe and window
- EDGAR quarterly `form.gz` indexes, 2021 Q1 through 2026 Q3. Form type `SC TO-I` only (originals; amendments excluded).
- Dedupe by accession number.
- Issuer must be on an exchange per SEC `company_tickers_exchange.json` (Nasdaq, NYSE, CBOE; OTC and unlisted excluded) by
  CIK, AND have Yahoo daily prices covering the filing date.
- Known limitation, stated up front: the ticker map is CURRENT-only, so issuers since delisted or acquired are missed
  (survivorship). Counts are therefore a lower bound; the number of exchange-unmapped filings is reported.

## Per-filing flags (parsed from the main Offer to Purchase / SC TO-I text)
- (a) ODD_LOT: text mentions an odd lot AND a priority/non-proration clause (regexes in `research/oddlot_tender/parse.py`).
- (b) PRICE_TYPE: `fixed` (single $ price per share), `range` (Dutch auction with a stated $ range; the LOW end is used,
  the conservative choice), `nav` (price tied to net asset value; NOT quantifiable from the filing), `unknown`.
- (c) Gross spread % = (offer price - last close strictly before the filing date) / that close. Offer price = fixed price,
  or low end of range.

## Qualifying event
ODD_LOT true AND PRICE_TYPE in {fixed, range} AND listed with Yahoo prices AND gross spread > 0.

## Payoff model
Net payoff = 99 x (offer price - entry price) - $25 broker tender fee. Assumes odd lots are accepted in full and the offer
completes (cancellations/extensions not modelled). Primary entry price = prior-day close (as specified).
SENSITIVITY (informational, not part of the verdict): entry at the close of filing day +1, and spread/payoff recomputed.
If the sensitivity flips a verdict, that is reported prominently.

## KILL RULES (verdict uses the primary entry price)
- Rule 1 (Volume): FAIL if the median of the yearly qualifying-event counts over the complete years 2021-2025 is < 12.
  Each year's count is shown; 2026 YTD is shown but not used.
- Rule 2 (Net payoff): FAIL if the pooled median net payoff per qualifying event (2021 through 2026 YTD) is < $75.
- Overall: PASS only if neither rule fails. Anything else is a stop for this avenue.

## Reported separately (never counted as qualifying, so they cannot rescue or sink the verdict silently)
- `nav` filings (listed funds whose tender price is a % of NAV): count per year. This is the main category the screen
  cannot quantify without NAV history; if the verdict is FAIL while this bucket is large, the result is "inconclusive
  for NAV-linked funds", not "no opportunity".
- Odd-lot fixed/range offers with a NEGATIVE spread, and listed filings with no parseable price.
- Parser validation: a random sample of classified filings is read by hand and the error rate reported.

---
# OUTCOME (run 2026-10-24) -- verdict: **FAIL (both kill rules)**. Kill rules and thresholds were NOT changed after seeing data.

Funnel: 2,831 SC TO-I originals (2021Q1-2026Q3) -> 417 on Nasdaq/NYSE/CBOE per the current SEC map (2,414 unmapped/unlisted:
non-traded funds, OTC, and issuers since delisted) -> 67 odd-lot + parseable fixed/range price -> 30 with positive spread.

| Tier | Events | Per year 2021/22/23/24/25 (2026 YTD) | Median spread | Median net $ | Rule 1 (>=12/yr) | Rule 2 (>=$75) |
|---|---|---|---|---|---|---|
| A mechanical (verdict basis) | 25 | 7/7/3/1/3 (4) | 9.85% | $58 | FAIL (median 3/yr) | FAIL |
| B hand-cleaned | 23 | 7/6/3/0/3 (4) | 7.27% | $58 | FAIL | FAIL |
| C strict | 22 | 6/6/3/0/3 (4) | 5.91% | $47 | FAIL | FAIL |

Upper bound with NO data-quality filters: 30 events = 5.2/yr. Classification errors therefore cannot rescue Rule 1.
Sensitivity (entry = first close after the filing): 8 of 23 tier-B events have a positive payoff, median net -$14.

## Post-hoc data fixes (made after reading results, applied uniformly to all rows; they fix measurement bugs, not thresholds)
1. Yahoo `Close` is adjusted for LATER splits (COKE/MNST/ELAB showed pre-split offers vs post-split prices) -> closes are un-split.
2. CAD-denominated offers (Imperial Oil) compared a CAD price to a USD quote -> excluded via a CAD flag.
3. Cross-check: if the filing's own stated market price differs >15% from the prior close, the row is excluded (catches WHLR).
4. Uuencoded graphics embedded in filings produced bogus CAD matches -> non-text documents are skipped.
Hand-read false positives (tier A -> B): WHLR (Series D preferred), NHP (preferred), PLTK (historical bidder range), ELAB (warrant
exercise price). Of 25 mechanical events, 4 were wrong (16%) and 1 (IVT) unverifiable. Zero odd-lot mentions were missed except
NAV-priced funds (22 filings, all in the NAV bucket).

## What this does NOT show (read before drawing conclusions)
- NAV-linked funds are unquantified: 42 listed odd-lot filings priced as a % of NAV (5-8 a year, many repeat quarterly filers).
  ILLUSTRATIVE arithmetic, not measured: a $10 fund trading 8-10% below NAV, tendered at ~98% of NAV, pays about $80-100 gross
  on 99 shares (~$55-75 net of the $25 fee), i.e. at or just below the $75 bar. Measuring it needs NAV history we do not have.
  Verdict for that bucket: INCONCLUSIVE, leaning not worth it.
- The ticker map is current-only, so delisted/acquired issuers are missing (survivorship). Counts are a lower bound, but the
  gap to 12/yr is ~4x.
- Dutch-auction offers use the LOW end of the range (conservative); fixed-price premium tenders (e.g. MTEX +26.8%) are the
  real opportunities, and there are only about 1-3 a year.
- Payoff assumes full odd-lot acceptance and completion; extensions, cancellations and broker deadlines are not modelled.

Code: research/oddlot_tender/{parse,classify}.py (resumable; 1 GB filing cache under ./cache, gitignored), tests/test_oddlot_parse.py.
