# PEAD (8-K Item 2.02 + SUE + reaction-day return): FROZEN SPEC (2026-10-23, committed BEFORE any returns computed)
Trial name in `validation_trials_registry.txt`: `pead_8k_sue_react` (45th trial). The last S&P 500-only trial: if it fails, S&P 500 alpha search is closed.

Universe: PIT S&P 500 members (hanshof table) as of the reaction day; prices = Yahoo/Tiingo adjusted close (`prices_pit.pkl`), delisted names exit at last close.
Events: for each fiscal quarter with a first-reported XBRL net income, the FIRST 8-K Item 2.02 accepted after quarter end and on/before the periodic report (10-Q/10-K filed date), within 45 days of it. No match -> no event.
SUE_q = (NI_q - NI_{q-4}) / std(prior 8 seasonal changes). NI = `NetIncomeLoss` (fallback `ProfitLoss`), standalone quarters, Q4 = 10-K annual - 9M YTD, FIRST-REPORTED values only (never restated). Dollars, so split-proof. Needs 12 quarters of history.
Reaction day: `acceptanceDateTime` is UTC -> converted to ET. Accepted before 16:00 ET -> that day (next session if not a session); at/after 16:00 ET -> next session. (A mid-day acceptance is typically a BMO release or the previous evening's release; either way the reaction session is that day.)
Reaction return Z = (close[react]/close[react-1] - SPY same day) / sd of the prior 60 trading days' daily excess return.
Trigger: SUE >= trailing-365-day 80th percentile AND Z >= trailing-365-day 80th percentile, cutoffs from earlier reaction dates only, at least 200 prior events.
Entry: PRIMARY = close of react+1 (the signal uses close[react], so entering at that same close would be circular). Sensitivity (reported, NOT decisive): close of react. Hold 20 sessions.
Sizing: 5% NAV per position, max 20 concurrent (skip new triggers when full; tie-break same day: higher SUE first), remainder in SPY. Cost 10 bps per side on the stock leg; SPY rebalancing free.
Decision metric: daily portfolio excess vs SPY, 21-day block-bootstrap t (research/etf_overlays/metrics.excess_t), plus maxDD vs SPY. PASS needs t >= 3.09 (p < 0.0011) AND maxDD no worse than SPY AND the 2019+ half must keep the same sign. No parameter is tuned; the whole sample from the first trigger date is the test (XBRL history means no pre-2013 design period exists).
Allowed afterwards: nothing. A fail is final for this specification.
