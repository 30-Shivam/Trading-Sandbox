"""One-off verification run, NOT a permanent feature.

Per the user's own follow-up ("lets go with that" -- isolate whether v73/v74's
holdout improvement is really attributable to ma_crossover_skew_regime_min
specifically, vs. incidental re-tuning of the other 4 free dimensions both
searches also moved): pins EVERY ma_crossover search dimension except
ma_crossover_skew_regime_min to the exact values live in System_Config v71
(short/long window, sector_relative_strength_min, yield_curve_spread_max,
atr_take_profit_multiplier, stop_loss_atr_multiplier), then runs the real,
unmodified optimize.py CLI (same full pipeline -- holdout split, DSR,
real-vs-random check, candidate write) with only skew_regime_min free.

Monkeypatches optimize.py's own module-level range constants to (v71_value,
v71_value) rather than editing the committed file -- Optuna's suggest_int/
suggest_float both tolerate low==high (returns that fixed value every trial),
so this reuses the entire existing, already-verified search machinery
unchanged. tp/sl reuse the CLI's own existing --pin-atr-take-profit-multiplier/
--pin-stop-loss-atr-multiplier flags (no monkeypatching needed for those two).

If the winning trial's skew_regime_min lands close to v73/v74's independently-
found ~-21 to -22.5 AND clears a real holdout/DSR bar on its own, that's a
much cleaner promotion case: the improvement is attributable to the skew
filter alone, not entangled with 4 other simultaneously-tuned dimensions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage
import optimize

v71 = storage.system_config.get_config_by_version(71)["params"]
print("Pinning every ma_crossover dimension except skew_regime_min to v71's live values:")
for k in [
    "ma_crossover_short_window", "ma_crossover_long_window",
    "ma_crossover_sector_relative_strength_min", "ma_crossover_yield_curve_spread_max",
    "atr_take_profit_multiplier", "stop_loss_atr_multiplier",
]:
    print(f"  {k} = {v71[k]}")

optimize.MA_CROSSOVER_SHORT_WINDOW_RANGE = (
    v71["ma_crossover_short_window"], v71["ma_crossover_short_window"]
)
optimize.MA_CROSSOVER_LONG_WINDOW_RANGE = (
    v71["ma_crossover_long_window"], v71["ma_crossover_long_window"]
)
optimize.SECTOR_RELATIVE_STRENGTH_RANGE = (
    v71["ma_crossover_sector_relative_strength_min"], v71["ma_crossover_sector_relative_strength_min"]
)
optimize.YIELD_CURVE_SPREAD_MAX_RANGE = (
    v71["ma_crossover_yield_curve_spread_max"], v71["ma_crossover_yield_curve_spread_max"]
)
# optimize.SKEW_REGIME_MIN_RANGE deliberately left untouched -- the one free dimension.

sys.argv = [
    "optimize.py",
    "--strategy", "ma_crossover",
    "--trials", "20",
    "--pin-atr-take-profit-multiplier", str(v71["atr_take_profit_multiplier"]),
    "--pin-stop-loss-atr-multiplier", str(v71["stop_loss_atr_multiplier"]),
]

if __name__ == "__main__":
    optimize.main()
