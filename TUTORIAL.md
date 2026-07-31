# Swing-Trading System — Usage Manual

Where this project actually stands today (Phases 0-7 of `ARCHITECTURE_PLAN.md`
are done; Phase 8 — Docker/Kubernetes — is not) and how to actually run it.
Nothing here needs containers: every piece runs as a plain Python script/app
on your machine, talking to a MongoDB Atlas cluster.

## 1. What this system does, in one paragraph

You maintain a watchlist of tickers. A rule-based scanner (RSI + ATR +
structural support, `swingtrade/`) decides which ones are a "Strong Buy" /
"Buy" today, with a Stop_Loss and take-profit target for each. Every such
signal gets logged to MongoDB. A nightly settlement job later checks what
actually happened to each logged trade (win/loss/expired, gap-aware). Once
enough history exists, an Optuna-based optimizer re-tunes the RSI/ATR/
stop-loss thresholds against that history (and years of backtested data,
recency-weighted toward what's working lately) and proposes a new parameter
set — which a human reviews and promotes before it ever affects live
signals. The Streamlit dashboard is just a window onto all of this.

**A real caveat worth knowing up front**: RSI-based "oversold" can stay true
for weeks during a genuine sustained decline, not just a brief dip — the
active config's tuned threshold has landed as high as ~52 (well above a
"classic" RSI<30 reading), and the structural support level recalculates on
a rolling window, so a ticker that's simply making new lows day after day
can keep generating fresh Strong Buy signals the whole way down. Every
result now carries `Oversold_Streak_Days` / `Extended_Decline_Warning`
(highlighted orange, 5+ consecutive days by default) specifically to
surface this — it does NOT change `Trade_Score`, so check it, don't just
trust a high score alone. See section 4's incident writeup for how this
played out on a real pair of trades.

## 2. One-time setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root (never committed — it's gitignored):

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGODB_DB_NAME=<your db name>
```

Use a free MongoDB Atlas M0 cluster if you don't already have one. Every
script in this repo degrades gracefully without Mongo (the dashboard falls
back to built-in defaults; the CLI scripts print a clear error and exit)
rather than crashing outright, so it's safe to try things before Mongo is
wired up — you just won't get logging/learning until it is.

Edit `watchlist.txt` to control which tickers get scanned. It accepts either
plain text (one ticker per line) or the JSON form already in the repo
(`{"watchlist": [{"ticker": "NVDA", ...}, ...]}` — the extra fields like
`sector`/`strategy` are just notes for you, not read by any code).

## 3. Command cheat sheet

Everything below is `py -3 <script>.py` (or `streamlit run` for the one
interactive app). Run from the repo root.

| Command | What it does |
|---|---|
| `streamlit run dip_buy_analyzer.py` | Interactive dashboard — look at today's picks, set position sizing, size real trades |
| `py -3 ingest.py` | Headless scan: fetch → score → log signals to Mongo, no browser needed |
| `py -3 ingest.py --watchlist other.txt --position-budget 500` | Same, with a different ticker list / per-trade sizing |
| `py -3 settle_trades.py` | Resolve unsettled signals to WIN/LOSS/EXPIRED against real subsequent price history |
| `py -3 run_backtest.py --start 2021-06-01 --end 2026-07-26` | Walk-forward backtest one config (default: `DEFAULT_CONFIG`) against history, no Mongo needed |
| `py -3 run_backtest.py --with-catalyst` | Same, plus a real (not always-False) Catalyst_Warning simulation and a catalyst-vs-non performance breakdown |
| `py -3 optimize.py --trials 50 --start 2021-06-01 --end 2026-07-26` | Optuna search for better RSI/ATR/stop-loss/streak-penalty params (5D); writes a `candidate`, never touches `active` |
| `py -3 optimize.py --trials 50 --recency-half-life-days 90` | Same, weighting recent conditions more heavily (0 = old uniform pooling) |
| `py -3 evaluate_config.py --tickers A,B,C --rsi-oversold-threshold 52 ...` | Backtest one specific hand-picked config before writing it as a candidate |
| `py -3 promote_config.py` | List every `System_Config` version (status, metrics, notes) |
| `py -3 promote_config.py --promote 4` | Promote version 4 to `active` (retires whatever was active) |
| `py -3 confirm_fill.py` | List logged signals you haven't confirmed as real fills yet |
| `py -3 confirm_fill.py --ticker INTC --date 2026-07-24 --price 89.60` | Mark a signal as an actual trade you made (price optional, defaults to the logged buy_price) |
| `py -3 check_survivorship_bias.py` | Compare today's watchlist vs. a genuine point-in-time S&P 500 sample, same config/window |

Add `--help` to any script for its full flag list and a detailed docstring
(e.g. `py -3 optimize.py --help`).

## 4. Walkthrough: from a cold start to a real order

Concrete example of the full loop, start to finish. Numbers below are one
real run's actual output — yours will differ day to day, but the shape and
the steps are exactly what to expect.

**Step 1 — Run the scan.** Data only updates on real trading days, so
running this on a weekend/holiday evening still works, it just reflects the
most recent close (e.g. running it Sunday night gives you Friday's numbers —
exactly what you want for placing a Monday order).

```bash
py -3 ingest.py
```
```
Using System_Config v3 (active).
Scanning 56 ticker(s)...
Analyzed 42/56 ticker(s): 5 Strong Buy, 18 Buy.
Logged 23 signal(s) to MongoDB.
```

That's it for logging — it already happened. No separate step needed.

**Step 2 — See the actual levels.** The dashboard is the easiest way
(`streamlit run dip_buy_analyzer.py`), or pull straight from Mongo:

```bash
py -3 -c "
import storage
db = storage.get_db()
docs = list(db['Trade_Signals'].find({'signal': 'Strong Buy'}).sort('trade_score', -1))
for d in docs:
    print(f\"{d['ticker']:6s} score={d['trade_score']:.1f}  buy={d['buy_price']:.2f}  stop={d['stop_loss']:.2f}  sell={d['sell_price']:.2f}  rrr={d['rrr']:.2f}  catalyst={d['catalyst_warning']}\")
"
```
```
GLW    score=88.1  buy=145.93  stop=138.49  sell=173.58  rrr=3.72  catalyst=True
ALAB   score=85.5  buy=289.60  stop=271.65  sell=356.33  rrr=3.72  catalyst=True
INTC   score=84.7  buy=89.59   stop=85.30   sell=105.55  rrr=3.72  catalyst=False
ADI    score=80.5  buy=366.72  stop=359.00  sell=395.43  rrr=3.72  catalyst=False
LRCX   score=80.1  buy=298.65  stop=286.79  sell=342.74  rrr=3.72  catalyst=True
```

`catalyst=True` means earnings fall within the warning window — check the
actual date before touching those; it's a flag, not an auto-disqualifier.

**Step 3 — Pick one and place the real order with your broker.** This tool
never touches your brokerage account. For a Strong Buy like INTC above:
a limit buy at 89.59 (day or GTC order) for the next session, and once/if
filled, a stop-loss at 85.30 and a take-profit limit sell at 105.55 (an OCO
bracket if your broker supports it).

**The one honest gap, and how to close it per-trade**: by default the system
tracks "what happens to a trade entered at exactly Buy_Price," independent
of whether your specific limit order actually filled — that's intentional
(it's scoring the *signal's* quality, not acting as your personal trade
blotter), but don't mistake "it's logged" for "I definitely own this." If
you actually place the order, tell the system:

```bash
py -3 confirm_fill.py --ticker INTC --date 2026-07-24 --price 89.60
```

`--price` is optional (your real fill, if it differed from the logged
Buy_Price) — Stop_Loss/Sell_Price never change, only the entry price used
for the eventual pnl_pct. This doesn't change how the signal settles, but it
lets reporting (`settle_trades.py`'s summary, `optimize.py`'s live-outcomes
context) separate "every mechanical signal's hypothetical outcome" from
"what actually happened to trades you made" — run `confirm_fill.py` with no
arguments any time to see what's still unconfirmed.

**Step 4 — Each following trading day, settle.**

```bash
py -3 settle_trades.py
```
```
Found 1 unsettled signal(s).
  INTC (2026-07-24): OPEN (still open)
```

Keeps printing OPEN until the stop or target actually gets hit (or 15
trading days pass), then resolves to WIN/LOSS/EXPIRED with the real
gap-aware exit price — no further action from you needed, just keep running
it daily (see the cheat sheet above for automating this via Task Scheduler).

**Step 5 — Let the loop compound.** Every day: `ingest.py` +
`settle_trades.py`. Whenever you want to look and size a position:
the dashboard. Every few weeks: `optimize.py` — by then, days like the one
above will fall inside the recency-weighted out-of-sample window and
actually pull weight in the next search (see section 6 below for why raw
live outcomes aren't blended in directly).

## 5. The two ways to run a scan, in more detail

**Interactive — the dashboard**, for actually looking at results, tweaking
your position budget, and eyeballing the watchlist:

```bash
streamlit run dip_buy_analyzer.py
```

Opens a browser tab. Sidebar shows which `System_Config` is currently active
and lets you set a per-trade "Position Budget" and a "Total Available Cash"
pool (greedily allocated down the ranked signal list). Every Strong Buy/Buy
it computes gets logged to Mongo automatically, same as the headless path
below.

Allocation also caps concentration per sector (`max_sector_allocation_pct`,
default 40%, read from `watchlist.txt`'s JSON `sector` field — see the
"Capital allocated by sector" expander on the page). Without this, a day
with several correlated Technology signals would greedily dump nearly all
your cash into one sector; trades that would breach the cap are labeled
`Sector Limit Reached` (distinct from `Insufficient Funds`) instead of
funded.

The cap is measured against your *whole portfolio*, not just today's cash —
the sidebar's "Current Holdings" box lets you type what you're actually
holding right now (`TICKER,AMOUNT[,AVG_COST]` per line, e.g. `INTC,8000,95.00`),
click Save to persist it (MongoDB's `Current_Holdings`), and the AMOUNT
counts toward both the cap's denominator and each sector's already-spent
total. It's manually maintained on purpose, not inferred from unsettled
`Trade_Signals` — a logged signal doesn't guarantee you actually got filled
(see the fill gap in section 4), so guessing your real holdings from it
would be unreliable. Holdings never reduce "Total Available Cash" itself —
they only tighten the sector cap.

**Position Review.** Any holding with an AVG_COST populates a "Position
Review" table: the same ATR-based stop/target math the scanner uses for new
candidates, anchored to your real entry price instead of a freshly computed
support level (`Stop_Loss = avg_cost - stop_loss_atr_multiplier × ATR`,
`Sell_Price = avg_cost + atr_take_profit_multiplier × ATR`), with a
HOLD / SELL (stop breached) / SELL (target hit) recommendation and your
unrealized P&L%. Same caveat as everything else here: informational, not a
guarantee — it uses whatever config is currently active, same trust level
as every live Buy signal.

**Screening without your portfolio state.** The "Apply capital allocation"
checkbox, when unchecked, shows raw Strong Buy/Buy/Watch/Ignore with Total
Available Cash and Current Holdings ignored entirely — useful when you just
want to see what the scanner found today without your cash/sector state
influencing which ones look fundable. Position Review is unaffected either
way.

**Headless — `ingest.py`**, for keeping signals accumulating even when
nobody has the dashboard open:

```bash
py -3 ingest.py
```

This is the Phase 7 addition. It runs the identical fetch → score → log
pipeline as the dashboard (they share `market_data.py`/`config_loader.py`,
so they can never silently disagree), prints a one-line summary, and exits.
Safe to re-run any time the same day — logging is an idempotent upsert, not
an insert, so you won't get duplicate signals.

Because Phase 8 (Kubernetes CronJob) isn't built yet, "on a schedule"
currently means *you* schedule it — e.g. Windows Task Scheduler running
`py -3 d:\Trading-Sandbox\ingest.py` once a day after market close, or just
running it by hand when you remember. Either way, this is the step that
actually matters for keeping the learning loop fed — the dashboard alone
only logs on days you happen to open it.

Useful flags: `--watchlist <path>` to scan a different file, `--position-budget <dollars>`
to change the per-trade sizing used for the logged `shares_to_buy`/`est_cost`
fields (defaults to $250, purely informational for logging — it doesn't
allocate real capital the way the dashboard's "Total Available Cash" does).

## 6. Letting Optuna search for better parameters

```bash
py -3 optimize.py --trials 50 --start 2021-06-01 --end 2026-07-26
```

Searches `rsi_oversold_threshold` / `atr_take_profit_multiplier` /
`stop_loss_atr_multiplier` / `extended_decline_penalty_per_day` /
`extended_decline_penalty_cap` via Bayesian optimization, scoring each trial
on its pooled out-of-sample `sharpe_like` (mean/stdev of pnl%, i.e. rewards
consistency, not just raw return) across every walk-forward fold. Trials
that don't produce enough trades to trust get penalized rather than scored
on a lucky small sample. Every trial's pnl already includes execution
realism (`slippage_pct`, `commission_pct_per_trade`) — those two are
deliberately NOT in the search space, since letting Optuna tune away the
friction they exist to model would defeat the point.

**Don't trust a hand-picked default just because it sounds reasonable.**
The `extended_decline_penalty_per_day` default (1.5) was a plausible guess
motivated by a real incident, and it actually *reduced* pooled out-of-sample
`sharpe_like` on a 15-ticker/5-year real-data test (0.97 → 0.892) versus no
penalty at all. A small-sample catalyst check told an equally misleading
story in the other direction (see `--with-catalyst` above). Always let the
search (and a large-enough sample) tell you, not intuition.

**It never touches your live config.** The winning trial is written to
`System_Config` as a `candidate` document only — nothing changes for the
dashboard or `ingest.py` until you deliberately promote it (section 8). This
champion/challenger gate exists specifically so a noisy search can't
silently degrade live signals.

If a parameter lands right at the edge of its search range (e.g.
`stop_loss_atr_multiplier` pinned at exactly its floor), that's a sign the
range itself may be cutting off the true optimum — worth widening the range
and rerunning, or capping it by hand and validating that by hand with the
next tool.

**Making the search respond to recent/live conditions.** Individual logged
trades aren't fed into the optimizer directly (a live outcome reflects one
specific historical config, not each trial's varying candidate — mixing it
in directly would compare apples to oranges across trials). Instead,
`optimize.py` weights recent out-of-sample folds more heavily than old ones
via `--recency-half-life-days` (default 180): a fold whose out-of-sample
window is 180 days before `--end` counts half as much as the most recent
one. Every candidate still gets re-simulated fairly over that same recent
window under its own parameters — only the weighting changes — so what's
been working in the current market regime (the one your real trades are
actually part of) has outsized influence on the winner, without corrupting
the cross-candidate comparison. Pass `--recency-half-life-days 0` for the
old uniform-pooling behavior.

## 7. Validating one specific config by hand

If you want to test a specific hand-picked value (e.g. capping something
Optuna pinned at a boundary) before ever writing it as a candidate:

```bash
py -3 evaluate_config.py --tickers NVDA,AMD,INTC,MU,AVGO \
    --start 2021-06-01 --end 2026-07-26 \
    --rsi-oversold-threshold 52.19 --atr-take-profit-multiplier 1.86 \
    --stop-loss-atr-multiplier 0.5 \
    --write-candidate --notes "manually capped stop-loss vs. boundary-pinned Optuna result"
```

Runs that exact config through the same walk-forward harness `optimize.py`
uses, prints its metrics against the `DEFAULT_CONFIG` baseline for
comparison, and — only if you pass `--write-candidate` — writes it to
`System_Config` as a new candidate. Never promotes anything itself.

## 8. Reviewing and promoting a candidate

```bash
py -3 promote_config.py
```

Lists every `System_Config` document (version, status, key metrics, notes)
so you can compare candidates against whatever's currently active. When
you've decided:

```bash
py -3 promote_config.py --promote 4
```

Promotes version 4 to `active` (retiring whatever was active before). The
dashboard and `ingest.py` pick this up automatically on their next run/cache
refresh — no restart needed for `ingest.py` (it's a one-shot process), and
within `SCAN_CACHE_TTL_SEC` (15 min) for a running dashboard.

## 9. A realistic weekly loop, today

Since Phase 8's scheduling automation isn't built yet, here's what "running
the system" actually looks like right now:

1. Once a day (manually, or via a Task Scheduler job you set up):
   `py -3 ingest.py` then `py -3 settle_trades.py`.
2. Open the dashboard (`streamlit run dip_buy_analyzer.py`) whenever you want
   to actually look at today's picks and size a real position.
3. Whenever you actually place an order: `py -3 confirm_fill.py --ticker X
   --date Y [--price Z]`, so `Trade_Outcomes` can eventually distinguish
   your real track record from every mechanical signal's hypothetical one.
4. Every few weeks, once `Trade_Outcomes` has grown, or just to re-tune
   against history: `py -3 optimize.py --trials 50 ...`, review with
   `promote_config.py`, promote if it looks like a genuine improvement (not
   just a boundary-pinned artifact — see section 6).

## 10. File map (what lives where)

| File / package | Role |
|---|---|
| `dip_buy_analyzer.py` | Streamlit dashboard — interactive entrypoint |
| `ingest.py` | Headless scan + signal logging (Phase 7) |
| `settle_trades.py` | Nightly settlement job |
| `run_backtest.py` | Walk-forward backtest CLI |
| `optimize.py` | Optuna parameter search |
| `evaluate_config.py` | Validate one manual config before promoting |
| `promote_config.py` | List / promote `System_Config` candidates |
| `confirm_fill.py` | Mark a logged signal as a real, confirmed trade |
| `check_survivorship_bias.py` | Cross-check today's watchlist vs. a point-in-time S&P 500 sample |
| `market_data.py` | All yfinance fetching, shared by the dashboard and `ingest.py` |
| `config_loader.py` | Loads the active `TradingConfig` from Mongo, shared the same way |
| `sp500_membership.py` | Point-in-time S&P 500 constituent lookup (cached CSV) |
| `watchlist.py` | Ticker-list parsing (`watchlist.txt` or pasted text) |
| `swingtrade/` | Pure calculation library — indicators, scoring, allocation, settlement math, backtest engine. No network/DB/UI dependency, so it's the one place this logic can drift out of sync from |
| `storage/` | MongoDB read/write for `Trade_Signals`, `Trade_Outcomes`, `System_Config`, `Current_Holdings` |

## 11. What's not here yet

Phase 8 (Docker + Kubernetes) hasn't been built: there's no container image,
no `docker-compose.yml`, no CronJob manifests. Everything above is a plain
local script — genuinely fine for a personal, ~60-ticker watchlist scanned a
few times a day, but it does mean scheduling is on you (Task Scheduler/cron)
until that phase lands.
