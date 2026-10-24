import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "llm_research"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pead_8k as P


def _units():
    rows = []
    for y in range(2010, 2014):
        rows += [dict(start=f"{y}-01-01", end=f"{y}-03-31", val=10 + y - 2010, form="10-Q", filed=f"{y}-05-01"),
                 dict(start=f"{y}-04-01", end=f"{y}-06-30", val=20, form="10-Q", filed=f"{y}-08-01"),
                 dict(start=f"{y}-01-01", end=f"{y}-09-30", val=90, form="10-Q", filed=f"{y}-11-01"),
                 dict(start=f"{y}-07-01", end=f"{y}-09-30", val=40, form="10-Q", filed=f"{y}-11-01"),
                 dict(start=f"{y}-01-01", end=f"{y}-12-31", val=130, form="10-K", filed=f"{y+1}-02-15")]
    return rows


def test_q4_is_annual_minus_9m_and_uses_first_reported():
    u = _units() + [dict(start="2010-01-01", end="2010-03-31", val=999, form="10-Q/A", filed="2012-01-01")]  # later restatement ignored
    q = P.quarter_series(u)
    assert q[q.end == "2010-03-31"].val.iloc[0] == 10
    assert q[q.end == "2010-12-31"].val.iloc[0] == 130 - 90


def test_assign_reaction_day_bmo_amc_weekend():
    days = pd.bdate_range("2024-05-01", "2024-05-31")
    assert P.assign_reaction_day("2024-05-02T11:30:00.000Z", days) == pd.Timestamp("2024-05-02")   # 07:30 ET
    assert P.assign_reaction_day("2024-05-02T15:50:00.000Z", days) == pd.Timestamp("2024-05-02")   # 11:50 ET, same day
    assert P.assign_reaction_day("2024-05-02T20:30:00.000Z", days) == pd.Timestamp("2024-05-03")   # 16:30 ET -> next session
    assert P.assign_reaction_day("2024-05-03T21:00:00.000Z", days) == pd.Timestamp("2024-05-06")   # Fri evening -> Mon


def test_match_8k_window():
    end, rep = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-05-03")
    assert P.match_8k(end, rep, ["2024-04-25T20:30:00.000Z"]) is not None
    assert P.match_8k(end, rep, ["2024-03-20T20:30:00.000Z"]) is None      # before quarter end
    assert P.match_8k(end, rep, ["2024-05-06T20:30:00.000Z"]) is None      # after the report
    assert P.match_8k(end, pd.Timestamp("2024-06-30"), ["2024-04-25T20:30:00.000Z"]) is None   # >45d before report


def test_trailing_cut_is_strictly_prior_and_needs_history():
    d = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=400, freq="D")); v = np.arange(400, dtype=float)
    assert P.trailing_cut(d, v, 10) is None
    c = P.trailing_cut(d, v, 399); assert c < 399 and c > 300       # uses only earlier events inside 365d


def test_simulate_slots_and_costs():
    days = pd.bdate_range("2024-01-01", periods=80); spy = pd.Series(100.0, index=days)
    px = pd.DataFrame({"AAA": np.linspace(100, 110, 80), "SPY": spy})
    trig = pd.DataFrame(dict(ticker=["AAA"], react=[days[10]], sue=[1.0]))
    ex, taken, skipped = P.simulate(trig, px, spy, 1)
    assert taken == 1 and skipped == 0
    assert ex.sum() > 0 and abs(ex.loc[days[11]] + P.SLOT * P.COST) < 1e-12   # entry-day cost only, no return before entry close
