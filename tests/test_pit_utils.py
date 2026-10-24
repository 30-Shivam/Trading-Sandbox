import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "llm_research"))
import pit_utils as pu


def test_member_asof_uses_latest_snapshot_on_or_before_date_and_excludes_pre_inclusion():
    t = pd.DataFrame({"date": pd.to_datetime(["2015-01-02", "2018-01-02", "2022-01-03"]),
                      "tickers": ["AAA,BBB", "AAA,BBB,NEW", "AAA,NEW"]})
    dates, sets = pu.make_membership(t)
    assert pu.member_asof("NEW", "2017-06-30", dates, sets) is False       # joined 2018: earlier filings must be excluded
    assert pu.member_asof("NEW", "2018-01-02", dates, sets) is True
    assert pu.member_asof("BBB", "2022-06-01", dates, sets) is False       # dropped after 2022 snapshot
    assert pu.member_asof("AAA", "2010-01-01", dates, sets) is True        # before data -> earliest snapshot


def _s(n, start="2020-01-01", step=1.0):
    return pd.Series(100 + np.arange(n) * step, index=pd.bdate_range(start, periods=n))


def test_label_full_window():
    s, spy = _s(300), _s(300, step=0.5)
    lab = pu.label_with_truncation(s, spy, "2020-02-03", hold=126, today="2022-01-01")
    assert lab["truncated"] is False and (lab["exit"] - lab["entry"]).days > 150


def test_label_truncates_at_delisting_and_pending_when_still_live():
    spy = _s(300, step=0.5)
    ended = _s(80)                                    # series stops after 80 bars -> delisted long ago
    lab = pu.label_with_truncation(ended, spy, "2020-01-06", hold=126, today="2023-01-01")
    assert lab["truncated"] is True and lab["exit"] == ended.index[-1]
    live = _s(80, start="2026-08-01")                 # ends near 'today' -> window simply incomplete
    assert pu.label_with_truncation(live, _s(80, start="2026-08-01"), "2026-08-03", hold=126, today=live.index[-1]) is None
