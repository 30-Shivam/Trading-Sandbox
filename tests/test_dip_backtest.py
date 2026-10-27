"""Pure-simulator checks for llm_research/dip_backtest.py, run on synthetic arrays (no data files, no network)."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "llm_research"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dip_backtest import simulate, SLOT, COST, HOLD  # noqa: E402


def _panel(n=60, k=3, daily=0.01, spy=0.0):
    R = np.full((n, k), daily); R[0] = 0.0
    return R, np.full(n, spy), np.ones((n, k), bool), np.full(k, n - 1)


def test_single_event_excess_and_costs():
    R, rs, valid, last = _panel()
    ex, taken, skipped = simulate(np.array([0]), np.array([10]), [0], 1, R, rs, valid, last)
    assert (taken, skipped) == (1, 0)
    # entry row 11, exit row 31: 20 held days of +1% vs flat SPY, one cost per side on the stock leg
    assert np.isclose(ex.sum(), SLOT * HOLD * 0.01 - 2 * SLOT * COST)
    assert ex[:11].sum() == 0.0 and ex[11] == -SLOT * COST and ex[32:].sum() == 0.0


def test_slot_cap_skips_extra_events():
    R, rs, valid, last = _panel(k=4)
    ex, taken, skipped = simulate(np.arange(4), np.full(4, 10), [0, 1, 2, 3], 1, R, rs, valid, last, slots=2)
    assert (taken, skipped) == (2, 2)


def test_slot_frees_after_hold_ends():
    R, rs, valid, last = _panel(n=80)
    _, taken, skipped = simulate(np.array([0, 1]), np.array([5, 30]), [0, 1], 1, R, rs, valid, last, slots=1)
    assert (taken, skipped) == (2, 0)


def test_delisted_name_exits_at_last_priced_row():
    R, rs, valid, last = _panel()
    last = last.copy(); last[0] = 20; R[21:, 0] = 0.0
    ex, _, _ = simulate(np.array([0]), np.array([10]), [0], 1, R, rs, valid, last)
    assert np.isclose(ex.sum(), SLOT * 9 * 0.01 - 2 * SLOT * COST)      # held rows 12..20 only


def test_no_price_on_entry_row_means_no_trade_and_late_events_are_ignored():
    R, rs, valid, last = _panel(); valid = valid.copy(); valid[11, 0] = False
    _, taken, _ = simulate(np.array([0, 1]), np.array([10, 59]), [0, 1], 1, R, rs, valid, last)
    assert taken == 0


def test_same_day_priority_follows_order_argument():
    R, rs, valid, last = _panel(k=2); R[:, 0] = 0.02; R[0] = 0.0
    ex, taken, _ = simulate(np.array([0, 1]), np.array([10, 10]), [1, 0], 1, R, rs, valid, last, slots=1)
    assert taken == 1 and np.isclose(ex.sum(), SLOT * HOLD * 0.01 - 2 * SLOT * COST)   # ticker 1 was processed first


def test_entry_shift_zero_enters_on_signal_row():
    R, rs, valid, last = _panel()
    ex1, *_ = simulate(np.array([0]), np.array([10]), [0], 0, R, rs, valid, last)
    assert ex1[10] == -SLOT * COST and np.isclose(ex1.sum(), SLOT * HOLD * 0.01 - 2 * SLOT * COST)
