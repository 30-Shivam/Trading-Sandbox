"""edgar_features.rows_for: liabilities fallback (LiabilitiesAndStockholdersEquity - equity) when Liabilities is untagged."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "llm_research"))
import edgar_features as ef


def _fact(val, accn, end, start=None, filed="2020-02-20"):
    u = {"val": val, "accn": accn, "end": end, "form": "10-K", "filed": filed}
    if start: u["start"] = start
    return u


def _facts(with_liabilities):
    a1, a0 = "0001-20-000001", "0001-20-000001"
    d = {
        "Revenues": [_fact(120, a1, "2019-12-31", "2019-01-01"), _fact(100, a1, "2018-12-31", "2018-01-01")],
        "NetIncomeLoss": [_fact(12, a1, "2019-12-31", "2019-01-01"), _fact(10, a1, "2018-12-31", "2018-01-01")],
        "NetCashProvidedByUsedInOperatingActivities": [_fact(15, a1, "2019-12-31", "2019-01-01")],
        "Assets": [_fact(200, a1, "2019-12-31"), _fact(180, a1, "2018-12-31")],
        "LiabilitiesAndStockholdersEquity": [_fact(200, a1, "2019-12-31"), _fact(180, a1, "2018-12-31")],
        "StockholdersEquity": [_fact(80, a1, "2019-12-31"), _fact(90, a1, "2018-12-31")],
    }
    if with_liabilities:
        d["Liabilities"] = [_fact(120, a1, "2019-12-31"), _fact(90, a1, "2018-12-31")]
    return {k: {"units": {"USD": v}} for k, v in d.items()}


def test_liabilities_derived_when_untagged():
    rows = ef.rows_for("X", _facts(with_liabilities=False))
    assert len(rows) == 1 and abs(rows[0]["leverage"] - 120 / 200) < 1e-12 and abs(rows[0]["lev_change"] - (120 / 200 - 90 / 180)) < 1e-12


def test_reported_liabilities_take_precedence_and_match():
    a = ef.rows_for("X", _facts(with_liabilities=True))[0]; b = ef.rows_for("X", _facts(with_liabilities=False))[0]
    assert abs(a["leverage"] - b["leverage"]) < 1e-12
