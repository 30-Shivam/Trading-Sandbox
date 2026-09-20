"""One-off SCOPING check, NOT a permanent feature.

Per the user's own follow-up ("the rate of improvement seems extremely slow
... anything we can do in terms of strategy refinement" -> "let's scope this
out" -> "universe expansion for ma_crossover/squeeze_breakout"): the ONE
genuinely real edge found recently wasn't a parameter refinement at all, it
was swapping the ticker UNIVERSE (small/mid-cap RSI, item 114/116, same v66
params unmodified). This script asks the same question of the two other live
strategies that haven't gotten that treatment yet: does ma_crossover (v71,
active) or squeeze_breakout (v53, last real version before its 2026-08-26
capital-eligibility removal) show a genuine, real-vs-random edge on the
already-built 992-ticker small/mid-cap universe (smallmid_watchlist.txt,
S&P 600 SmallCap + S&P 400 MidCap, zero overlap with the primary watchlist)?

Reuses benchmark_random_entry.py's real, unmodified CLI/methodology exactly
(same real-vs-random matched-count comparison, same multi-seed-averaged
ticker-holdout split used for the ORIGINAL small-cap RSI finding) --
monkeypatches its own WATCHLIST_FILE module constant to point at
smallmid_watchlist.txt (not just --tickers, which would leave sector_lookup
reading the PRIMARY watchlist's sectors -- the exact same real bug class
caught during the RSI small/mid-cap build, see dip_buy_analyzer.py's
cached_fetch_smallmid_bundle()). No new config, no new plumbing -- both
strategies' EXISTING live/last-promoted parameters, completely unmodified,
same "test the universe swap alone before touching anything else" discipline
as item 114's own original scoping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_random_entry

benchmark_random_entry.WATCHLIST_FILE = Path(__file__).resolve().parents[1] / "smallmid_watchlist.txt"

strategy = sys.argv[1] if len(sys.argv) > 1 else "ma_crossover"
extra_args = sys.argv[2:]

sys.argv = ["benchmark_random_entry.py", "--strategy", strategy] + extra_args

if __name__ == "__main__":
    benchmark_random_entry.main()
