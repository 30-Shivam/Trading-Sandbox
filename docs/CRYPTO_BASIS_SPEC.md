# Crypto basis (cash-and-carry) funding-rate screen -- FROZEN SPEC (written 2026-10-24, before any funding data was pulled)

Descriptive yield-and-friction screen, NOT a hypothesis test: 0 trials registered in `validation_trials_registry.txt`
(no `--register`, no `audit_vs_cash` gate run). Question: before building any exchange infrastructure, is the *historical
funding carry* big enough above cash to be worth a deeper look? A PASS here means "worth a deeper look", never "deploy".

## Data
- Binance USD-M perpetual funding history, `BTCUSDT` and `ETHUSDT`, public endpoint `fapi/v1/fundingRate` (paginated, no key),
  2021-01-01 UTC through the last available record. 8-hour funding. Fallback if unavailable: `data.binance.vision` monthly
  `fundingRate` archives (same data). Cash: `^IRX` (13-week T-bill yield, percent), daily mean per calendar year.
- Convention: funding rate r per interval; longs pay shorts when r > 0. The carry position is LONG spot / SHORT perp, so it
  earns +r x notional per interval (and PAYS when r < 0). The always-on position is simulated -- no entry/exit timing, because
  timing is an optimization and would be a registered trial.

## Metrics (per asset, per calendar year 2021..2026; 2026 is year-to-date, annualized by days elapsed)
- (a) Gross annualized funding yield on notional = sum(r) x 365 / days_covered (simple, uncompounded, signed).
- (b) Net yield = (a) - fees. PRIMARY fee model (the user's assumption): 0.05% taker on each of the 4 legs (buy spot, short
  perp, close both) = 0.20% per round trip, ONE round trip per year. Sensitivities (informational, reported in full):
  maker 0.02% (0.08%/RT); 12 round trips a year (taker 2.4%, maker 0.96%); Binance base spot taker 0.10% + perp 0.05% (0.30%/RT).
  Negative-funding periods are already in (a) because it is signed ("8-hour funding costs").
- (c) Net spread over cash = (b) - mean ^IRX over the same calendar year.
- Capital view (secondary): the same net spread expressed on capital = notional x (1 + 20% short-leg margin), i.e. divided by 1.2.

## KILL RULES (evaluated on the PRIMARY fee model, on the NOTIONAL basis -- the more generous one)
- Rule 1 (yield): for each of BTC and ETH, average the yearly net spread over cash (c) across 2022, 2023, 2024, 2025 and
  2026-YTD (equal weight per year); then average BTC and ETH 50/50. KILL if that number is < 4.0 percentage points.
- Rule 2 (drawdown): build the cumulative funding-income curve (fees excluded, in % of notional) for each asset from
  2022-01-01. KILL if, for either asset, the longest underwater spell (days spent below a previous peak) exceeds 365 days.
  "Unrecoverable" is defined as that, fixed now. The maximum drawdown in %-points and the underwater-at-end flag are also reported.
- Overall: DROP the strategy (no trial registered) if either rule fires. Otherwise: "worth a deeper look" only, and the
  deeper look (venue/jurisdiction, margin and liquidation, counterparty) still stands between this and any deployment.
- Cross-check reported prominently: if the capital-basis or any fee sensitivity would flip the verdict, say so.

## What this screen deliberately does NOT measure
Liquidation/margin-call risk on the short leg in a fast rally; exchange failure (FTX-style loss of the whole balance);
withdrawal/ADL events; spot-perp basis mark-to-market; slippage; taxes; jurisdiction (Binance.com is restricted for US
persons -- the regulated alternative is spot ETF vs CME futures, a different instrument with its own basis history);
crowding from large delta-neutral products compressing future funding. These can only make real results WORSE than the
screen, so the screen is a ceiling on the carry, not an estimate of realized profit.

---
# OUTCOME (run 2026-10-24) -- verdict: **DROP** (Rule 1 fails; Rule 2 ok). 0 trials registered. Rules and thresholds unchanged.

Data: Binance USDT-M `BTCUSDT`/`ETHUSDT`, 6,370 eight-hour intervals each, 2021-01-01 -> 2026-10-25, no gaps > 9h. Cash: `^IRX`.
No data-quality fixes were needed after the first run (unlike the odd-lot screen).

| Net spread over cash, primary fees (pp, on notional) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD | mean 2022+ |
|---|---|---|---|---|---|---|---|
| BTC | 30.37 | 1.97 | 2.62 | 6.77 | 0.88 | -0.89 | **2.27** |
| ETH | 37.30 | -1.41 | 3.01 | 7.81 | 0.68 | -1.85 | **1.65** |
| BTC gross funding yield (%) | 30.61 | 4.16 | 7.87 | 11.92 | 5.13 | 3.00 | |
| ETH gross funding yield (%) | 37.54 | 0.79 | 8.26 | 12.96 | 4.93 | 2.04 | |
| Cash (^IRX, %) | 0.03 | 2.00 | 5.05 | 4.95 | 4.05 | 3.69 | |

- **Rule 1:** average (BTC, ETH) = **1.96 pp** vs the 4.0 bar -> FAIL. Only 2024 clears 4 pp; 2021 (the bull-era carry) is outside the
  frozen 2022+ window and is the only year with a large spread. 2026 YTD gross funding (3.0% BTC, 2.0% ETH) is BELOW T-bills.
- **Rule 2:** longest underwater spell 137 days (BTC) / 170 days (ETH) vs 365; max drawdown of cumulative funding 0.41 / 1.79 pp;
  neither is underwater at the end -> ok. Funding drawdowns were never the problem; the level is.
- **Fee sensitivity cannot flip it:** mean 2022+ spread is 1.96 (primary), 2.08 (maker), 1.86 (spot 0.10%+perp 0.05%), 1.20
  (maker, 12 round trips/yr) and -0.24 (taker, 12 round trips/yr). Fees are a rounding error next to a funding yield that
  already sits barely above cash. On a capital basis (20% short margin) the average is 1.63 pp.
- Between 4% and 34% of intervals per year had negative funding in 2022+ (BTC 8-26%, ETH 4-34%); the always-on carry pays in those windows.

Caveat on what this does not close: a funding-regime filter (trade only when funding is high) is a strategy variant with its own
selection bias, so it would be a registered trial and was NOT tested here. The stable conclusion is narrower: the *unfiltered*
Binance perp carry has earned ~2 pp over cash since 2022 on notional, with 2025-26 near zero, before any counterparty,
liquidation or jurisdiction cost (none of which this screen charges). Re-open only if funding regimes visibly change.

Code: research/crypto_basis/{screen,pull}.py (cache gitignored), tests/test_crypto_basis_screen.py.
