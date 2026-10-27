"""research/oddlot_tender/parse.py -- pure parsing + frozen kill-rule arithmetic (docs/ODDLOT_SPEC.md). Offline."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "research" / "oddlot_tender"))
import parse as P  # noqa: E402


def test_idx_row_parses_misaligned_columns_and_rejects_noise():
    line = "SC TO-I          A&Q Long/Short Strategies Fund LLC        1186641     2023-03-22  edgar/data/1186641/0001104659-23-035511.txt   "
    rec = P.parse_form_idx_line(line)
    assert rec["form"] == "SC TO-I" and rec["cik"] == 1186641 and rec["filed"] == "2023-03-22"
    assert rec["accession"] == "0001104659-23-035511" and rec["company"] == "A&Q Long/Short Strategies Fund LLC"
    assert P.parse_form_idx_line("SC TO-I/A  X CORP  55  2023-03-22  edgar/data/55/0001-23-1.txt")["form"] == "SC TO-I/A"
    assert P.parse_form_idx_line("Form Type   Company Name   CIK   Date Filed   File Name") is None
    assert P.parse_form_idx_line("") is None


@pytest.mark.parametrize("text,expected", [
    ("Holders of fewer than 100 shares (odd lots) who tender all will be accepted before proration.", True),
    ("including the provisions relating to \"odd lot\" priority, proration and conditional tenders", True),
    ("odd lot holders who own 99 shares will not be subject to proration", True),
    ("The company may repurchase shares. Proration applies to all tenders.", False),
    ("An odd lot is a trade smaller than a round lot.", False),
    ("", False),
])
def test_detect_odd_lot(text, expected):
    assert P.detect_odd_lot(text) is expected


@pytest.mark.parametrize("text,ptype,price,high", [
    ("at a price of $12.50 per share in cash", "fixed", 12.5, None),
    ("at a purchase price of $1,250.00 per share", "fixed", 1250.0, None),
    ("at a price not greater than $15.00 nor less than $13.50 per share", "range", 13.5, 15.0),
    ("at a price per share of not less than $28.00 and not more than $32.00 in cash", "range", 28.0, 32.0),
    ("at a price equal to 98% of the net asset value per share", "nav", None, None),
    ("no price language at all", "unknown", None, None),
])
def test_extract_price_terms(text, ptype, price, high):
    assert P.extract_price_terms(text) == {"type": ptype, "price": price, "high": high}


def test_fixed_price_uses_most_common_amount():
    text = "at a price of $9.00 per share. The $9.00 per share in cash. Previously at a price of $8.00 per share."
    assert P.extract_price_terms(text)["price"] == 9.0


def test_stated_market_price_and_cad_detection():
    assert P.stated_market_price("the last reported sale price of the shares on Nasdaq was $20.68 per share") == 20.68
    assert P.stated_market_price("nothing here") is None
    assert P.is_cad("C$62.00 and C$78.00 and Cdn$5") is True
    assert P.is_cad("uuencoded noise \\C$$ >G55 and XC$HXQ4") is False


def test_payoff_arithmetic():
    assert P.gross_spread_pct(110, 100) == pytest.approx(10.0)
    assert P.net_payoff_usd(10.0, 9.0) == pytest.approx(99 * 1.0 - 25.0)
    assert P.net_payoff_usd(10.0, 10.0) == pytest.approx(-25.0)


def _ev(year, net, spread=5.0):
    return {"year": year, "net_usd": net, "spread_pct": spread}


def test_kill_rules_volume_and_payoff():
    many = [_ev(y, 100.0) for y in range(2021, 2026) for _ in range(12)]
    assert P.summarize(many)["passed"] is True
    assert P.summarize(many)["median_yearly_count"] == 12
    few = [_ev(y, 100.0) for y in range(2021, 2026) for _ in range(11)]
    assert P.summarize(few)["rule1_fail"] is True and P.summarize(few)["rule2_fail"] is False
    cheap = [_ev(y, 74.99) for y in range(2021, 2026) for _ in range(12)]
    assert P.summarize(cheap)["rule2_fail"] is True and P.summarize(cheap)["rule1_fail"] is False
    empty = P.summarize([])
    assert empty["rule1_fail"] and empty["rule2_fail"] and not empty["passed"]


def test_missing_year_counts_as_zero_in_the_median():
    events = [_ev(y, 100.0) for y in (2021, 2022) for _ in range(20)]   # 2023-25 empty -> counts 20,20,0,0,0
    assert P.summarize(events)["median_yearly_count"] == 0
