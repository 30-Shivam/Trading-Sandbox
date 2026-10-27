"""swingtrade.account_portfolio + storage.account_holdings + portfolio_view.py -- real-account holdings intake and valuation.
Offline: prices/FX are injected, Mongo is a fake collection."""
import pytest

import portfolio_view
import storage
from storage import account_holdings
from swingtrade import account_portfolio as ap

@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any test that reaches Yahoo fails loudly instead of silently going online."""
    import yfinance

    def boom(*a, **k):
        pytest.fail("test attempted a live Yahoo call")   # BaseException-derived: cannot be swallowed by `except Exception`

    monkeypatch.setattr(yfinance, "Ticker", boom)
    monkeypatch.setattr(yfinance, "download", boom)


CSV = """Symbol,Quantity,Currency,Average Cost
SPY,10,USD,480.25
VFV,25,CAD,112.40
BRK.B,2,USD,
CASH,350.00,CAD,
Total,,,
"""


# ---------------------------------------------------------------- parsing

def test_parse_number_formats():
    assert ap.parse_number("$1,234.56") == 1234.56
    assert ap.parse_number("(12.5)") == -12.5
    assert ap.parse_number(" 7 ") == 7.0
    for blank in ("", "--", None, "abc"):
        assert ap.parse_number(blank) is None


@pytest.mark.parametrize("raw,cur,exch,expected", [
    ("SPY", "USD", None, "SPY"),
    ("vfv", "CAD", None, "VFV.TO"),                  # CAD with no suffix -> TSX
    ("VFV.TO", "CAD", None, "VFV.TO"),                # explicit suffix kept
    ("BRK.B", "USD", None, "BRK-B"),                  # class-share dot -> dash
    ("BBD.B", "CAD", None, "BBD-B.TO"),
    ("ABC", "CAD", "TSXV", "ABC.V"),                  # exchange column wins over the CAD default
    ("XYZ", "CAD", "NEO", "XYZ.NE"),
    (" spy ", "USD", None, "SPY"),
])
def test_normalize_symbol(raw, cur, exch, expected):
    assert ap.normalize_symbol(raw, cur, exch) == expected


def test_parse_csv_rows_cash_totals_and_class_shares():
    pos = {p["symbol"]: p for p in ap.parse_holdings_csv(CSV)}
    assert set(pos) == {"SPY", "VFV.TO", "BRK-B", "CASH.CAD"}            # 'Total' row skipped
    assert pos["SPY"]["avg_cost"] == 480.25 and pos["SPY"]["currency"] == "USD" and pos["SPY"]["kind"] == "security"
    assert pos["VFV.TO"]["quantity"] == 25 and pos["BRK-B"]["avg_cost"] is None
    assert pos["CASH.CAD"]["kind"] == "cash" and pos["CASH.CAD"]["quantity"] == 350.0


def test_parse_csv_aliases_book_cost_and_default_currency():
    text = "Ticker,Shares,Book Cost\nXQQ,10,\"$1,200.00\"\n"
    (p,) = ap.parse_holdings_csv(text, default_currency="CAD")
    assert p["symbol"] == "XQQ.TO" and p["currency"] == "CAD" and p["avg_cost"] == pytest.approx(120.0)


def test_parse_csv_finds_header_below_preamble_and_handles_bom_free_text():
    text = "Scotia iTRADE export\nAccount: TFSA\n\nSymbol,Quantity\nSPY,3\n"
    assert [p["symbol"] for p in ap.parse_holdings_csv(text)] == ["SPY"]


@pytest.mark.parametrize("bad,msg", [
    ("Name,Value\nSPY,3\n", "no header"),
    ("Symbol,Quantity,Currency\nSPY,3,EUR\n", "unsupported currency"),
])
def test_parse_csv_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        ap.parse_holdings_csv(bad)


def test_parse_csv_skips_rows_without_quantity():
    assert ap.parse_holdings_csv("Symbol,Quantity\nSPY,\nQQQ,2\n")[0]["symbol"] == "QQQ"


# ---------------------------------------------------------------- valuation

def _positions():
    return ap.parse_holdings_csv(CSV)


def test_value_portfolio_fx_math_weights_and_pnl():
    prices = {"SPY": 500.0, "VFV.TO": 120.0, "BRK-B": 400.0}
    v = ap.value_portfolio(_positions(), prices, usdcad=1.40)
    by = {r["symbol"]: r for r in v["rows"]}
    assert by["SPY"]["value_cad"] == pytest.approx(10 * 500 * 1.40) and by["SPY"]["value_usd"] == pytest.approx(5000)
    assert by["VFV.TO"]["value_cad"] == pytest.approx(25 * 120) and by["VFV.TO"]["value_usd"] == pytest.approx(3000 / 1.40)
    assert by["CASH.CAD"]["value_cad"] == pytest.approx(350.0)
    total = 7000 + 3000 + 2 * 400 * 1.40 + 350
    assert v["total_cad"] == pytest.approx(total) and v["total_usd"] == pytest.approx(total / 1.40)
    assert sum(r["weight"] for r in v["rows"]) == pytest.approx(1.0)
    assert by["SPY"]["pnl_pct"] == pytest.approx((500 / 480.25 - 1) * 100)
    assert by["SPY"]["pnl_native"] == pytest.approx((500 - 480.25) * 10)
    assert "pnl_pct" not in by["BRK-B"] and "pnl_pct" not in by["CASH.CAD"]
    assert v["cad_denominated_share"] == pytest.approx((3000 + 350) / total)


def test_unpriced_security_is_flagged_not_valued_at_zero():
    v = ap.value_portfolio(_positions(), {"SPY": 500.0}, usdcad=1.40)
    assert set(v["unpriced"]) == {"VFV.TO", "BRK-B"}
    assert v["total_cad"] == pytest.approx(10 * 500 * 1.40 + 350)


def test_nan_price_counts_as_unpriced():
    v = ap.value_portfolio(_positions(), {"SPY": float("nan")}, usdcad=1.4)
    assert "SPY" in v["unpriced"]


def test_core_target_drift_other_and_cash():
    pos = ap.parse_holdings_csv("Symbol,Quantity,Currency\nSPY,8,USD\nQQQ,1,USD\nXYZ,10,USD\nCASH,100,USD\n")
    v = ap.value_portfolio(pos, {"SPY": 100.0, "QQQ": 100.0, "XYZ": 100.0}, usdcad=1.0,
                           core_targets={"SPY": 0.85, "QQQ": 0.15})
    # total 800 + 100 + 1000 + 100 = 2000
    c = v["core"]
    assert c["actual_weight"] == {"SPY": pytest.approx(0.4), "QQQ": pytest.approx(0.05)}
    assert c["cash_weight"] == pytest.approx(0.05) and c["other_holdings_weight"] == pytest.approx(0.5)


def test_bad_fx_rejected():
    with pytest.raises(ValueError):
        ap.value_portfolio([], {}, usdcad=0)


def test_empty_portfolio_is_zero_not_a_crash():
    v = ap.value_portfolio([], {}, usdcad=1.4, core_targets={"SPY": 1.0})
    assert v["total_cad"] == 0 and v["core"]["actual_weight"] == {"SPY": 0.0}


# ---------------------------------------------------------------- storage (fake collection)

class FakeCol:
    def __init__(self):
        self.docs = []

    def delete_many(self, q):
        self.docs = [d for d in self.docs if any(d.get(k) != v for k, v in q.items())]

    def insert_many(self, docs):
        self.docs += [dict(d) for d in docs]

    def find(self, q, proj=None):
        out = [d for d in self.docs if all(d.get(k) == v for k, v in q.items())]
        drop = {k for k, v in (proj or {}).items() if v == 0}
        return [{k: v for k, v in d.items() if k not in drop} for d in out]


@pytest.fixture
def fake_db(monkeypatch):
    col = FakeCol()
    monkeypatch.setattr(account_holdings, "get_db", lambda: {account_holdings.COLLECTION_NAME: col})
    return col


def test_set_account_holdings_replaces_only_that_account(fake_db):
    account_holdings.set_account_holdings("other", [{"symbol": "AAA", "quantity": 1, "kind": "security", "currency": "USD"}])
    n = account_holdings.set_account_holdings("scotia_tfsa", _positions())
    assert n == 4 and len(fake_db.docs) == 5
    account_holdings.set_account_holdings("scotia_tfsa", _positions()[:1])
    assert [d["symbol"] for d in fake_db.docs if d["account"] == "scotia_tfsa"] == ["SPY"]
    assert [d["symbol"] for d in fake_db.docs if d["account"] == "other"] == ["AAA"]      # untouched


def test_set_account_holdings_merges_duplicates_with_weighted_cost(fake_db):
    rows = [{"symbol": "SPY", "quantity": 10, "avg_cost": 100.0, "kind": "security", "currency": "USD"},
            {"symbol": "SPY", "quantity": 30, "avg_cost": 200.0, "kind": "security", "currency": "USD"}]
    account_holdings.set_account_holdings("a", rows)
    (doc,) = fake_db.docs
    assert doc["quantity"] == 40 and doc["avg_cost"] == pytest.approx(175.0)


def test_zero_quantity_rows_are_not_stored(fake_db):
    assert account_holdings.set_account_holdings("a", [{"symbol": "SPY", "quantity": 0, "kind": "security", "currency": "USD"}]) == 0


def test_get_account_holdings_strips_internal_fields(fake_db):
    account_holdings.set_account_holdings("a", _positions())
    got = account_holdings.get_account_holdings("a")
    assert len(got) == 4 and all("updated_at" not in d and "account" not in d for d in got)


# ---------------------------------------------------------------- CLI

def _stub_prices(monkeypatch, fx=1.40, prices=None):
    prices = prices or {"SPY": 500.0, "VFV.TO": 120.0, "BRK-B": 400.0}
    monkeypatch.setattr(portfolio_view, "last_price", lambda s: fx if s == portfolio_view.FX_SYMBOL else prices.get(s))


def test_cli_view_from_csv_without_database(tmp_path, monkeypatch, capsys):
    f = tmp_path / "h.csv"; f.write_text(CSV)
    _stub_prices(monkeypatch)
    monkeypatch.setattr(storage, "set_account_holdings", lambda *a: pytest.fail("must not save without --save"))
    rc = portfolio_view.main(["--csv", str(f)])
    out = capsys.readouterr().out
    assert rc == 0 and "scotia_tfsa: total C$" in out and "SPY" in out and "VFV.TO" in out
    assert "target 85%" in out and "cash" in out


def test_cli_save_persists_positions(tmp_path, monkeypatch, capsys):
    f = tmp_path / "h.csv"; f.write_text(CSV)
    _stub_prices(monkeypatch)
    saved = {}
    monkeypatch.setattr(storage, "ensure_indexes", lambda: None)
    monkeypatch.setattr(storage, "set_account_holdings", lambda acct, pos: saved.update(acct=acct, n=len(pos)) or len(pos))
    assert portfolio_view.main(["--csv", str(f), "--save", "--account", "mine"]) == 0
    assert saved == {"acct": "mine", "n": 4} and "Saved 4 position(s)" in capsys.readouterr().out


def test_cli_view_saved_account(monkeypatch, capsys):
    monkeypatch.setattr(storage, "get_account_holdings", lambda a: _positions())
    _stub_prices(monkeypatch)
    assert portfolio_view.main(["--account", "scotia_tfsa"]) == 0 and "SPY" in capsys.readouterr().out


def test_cli_nothing_saved_and_bad_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage, "get_account_holdings", lambda a: [])
    assert portfolio_view.main(["--account", "nope"]) == 1 and "No holdings saved" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"; bad.write_text("Name,Value\nx,1\n")
    assert portfolio_view.main(["--csv", str(bad)]) == 1 and "no header" in capsys.readouterr().err
    assert portfolio_view.main(["--csv", str(tmp_path / "missing.csv")]) == 1


def test_cli_fx_failure_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "h.csv"; f.write_text(CSV)
    monkeypatch.setattr(portfolio_view, "last_price", lambda s: None)
    assert portfolio_view.main(["--csv", str(f)]) == 1 and "USD/CAD" in capsys.readouterr().err


def test_cli_unpriced_symbol_is_reported(tmp_path, monkeypatch, capsys):
    f = tmp_path / "h.csv"; f.write_text(CSV)
    _stub_prices(monkeypatch, prices={"SPY": 500.0})
    assert portfolio_view.main(["--csv", str(f)]) == 0
    assert "NOT VALUED" in capsys.readouterr().out


def test_parse_targets():
    assert portfolio_view.parse_targets(None) == {"SPY": 0.85, "QQQ": 0.15}
    assert portfolio_view.parse_targets("vfv.to:0.85, xqq.to:0.15") == {"VFV.TO": 0.85, "XQQ.TO": 0.15}


def test_guard_actually_blocks_an_unstubbed_cli_run(tmp_path):
    """Proves the autouse guard works: with no stubs, the CLI would call Yahoo and must be stopped."""
    f = tmp_path / "h.csv"; f.write_text(CSV)
    from _pytest.outcomes import Failed
    with pytest.raises(Failed, match="live Yahoo"):
        portfolio_view.main(["--csv", str(f)])
