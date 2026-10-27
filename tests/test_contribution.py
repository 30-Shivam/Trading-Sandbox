"""swingtrade.contribution.plan_contribution -- whole-share contribution planner. Pure and offline."""
import itertools

import pytest

from swingtrade.contribution import plan_contribution

P = {"SPY": 500.0, "QQQ": 400.0}
T = {"SPY": 0.85, "QQQ": 0.15}


def _plan(cash_cad, **kw):
    args = dict(prices_usd=P, usdcad=1.4, sleeve_values_usd={}, targets=T, commission_usd=5.0, fx_spread=0.0)
    args.update(kw)
    return plan_contribution(cash_cad, **args)


def _brute_force_best(cash_cad, usdcad=1.4, spread=0.0, comm=5.0, cash_usd=0.0, sleeve=None):
    sleeve = sleeve or {}
    rate, best = usdcad * (1 + spread), None
    for ns, nq in itertools.product(range(0, 12), repeat=2):
        trades = (ns > 0) + (nq > 0)
        needed = ns * P["SPY"] + nq * P["QQQ"] + trades * comm
        if max(0.0, needed - cash_usd) * rate > cash_cad + 1e-9:
            continue
        vals = {"SPY": sleeve.get("SPY", 0) + ns * P["SPY"], "QQQ": sleeve.get("QQQ", 0) + nq * P["QQQ"]}
        tot = sum(vals.values())
        err = 1.0 + 1e-9 if tot <= 0 else sum(abs(vals[s] / tot - T[s]) for s in T)
        key = (round(err, 6), -(ns * P["SPY"] + nq * P["QQQ"]))
        if best is None or key < best[0]:
            best = (key, {"SPY": ns, "QQQ": nq})
    return {k: v for k, v in best[1].items() if v > 0}


@pytest.mark.parametrize("cash,spread,usd_cash", [(2800, 0.0, 0), (2800, 0.02, 0), (4500, 0.0, 0), (1500, 0.0, 600), (0, 0.0, 2000)])
def test_matches_brute_force_search(cash, spread, usd_cash):
    plan = _plan(cash, fx_spread=spread, cash_usd=usd_cash)
    assert plan["shares"] == _brute_force_best(cash, spread=spread, cash_usd=usd_cash)


def test_money_accounting_is_exact():
    plan = _plan(2800, fx_spread=0.02)
    rate = 1.4 * 1.02
    needed = plan["invested_usd"] + plan["commission_usd"]
    assert plan["usd_from_cad"] == pytest.approx(needed)
    assert plan["cad_converted"] == pytest.approx(needed * rate)
    assert plan["cad_left"] == pytest.approx(2800 - needed * rate) and plan["cad_left"] >= 0
    assert plan["fx_cost_cad"] == pytest.approx(needed * 1.4 * 0.02)
    assert plan["commission_usd"] == 5.0 * plan["trades"]
    assert plan["friction_cad"] == pytest.approx(plan["commission_usd"] * 1.4 + plan["fx_cost_cad"])


def test_usd_cash_is_spent_before_converting_cad():
    plan = _plan(0, cash_usd=2000)                                  # no CAD at all, US$2,000 on hand
    assert plan["shares"] and plan["usd_from_cad"] == 0 and plan["cad_converted"] == 0
    assert plan["usd_left"] == pytest.approx(2000 - plan["invested_usd"] - plan["commission_usd"])


def test_leftover_usd_when_cash_exceeds_purchase():
    plan = _plan(1400, cash_usd=300)
    assert plan["usd_left"] >= 0 and plan["cad_left"] >= 0


def test_existing_sleeve_overweight_steers_new_money_to_the_other_fund():
    plan = _plan(1200, sleeve_values_usd={"SPY": 20000.0, "QQQ": 0.0}, fx_spread=0.0)
    assert plan["shares"] == {"QQQ": 2}          # US$857 buys two QQQ (805 incl. commission) but not a 500 SPY on top


def test_too_small_to_buy_one_share_reports_the_cash_needed():
    plan = _plan(100)
    assert plan["recommendation"] == "WAIT_TOO_SMALL" and plan["shares"] == {}
    assert plan["min_cash_cad_for_one_share"] == pytest.approx((400 + 5) * 1.4)        # cheapest fund + one commission
    assert plan["invested_usd"] == 0


def test_single_small_trade_flags_commission_drag():
    plan = _plan(720)                                               # US$514 -> one QQQ or one SPY only
    assert plan["recommendation"] == "WAIT_COMBINE" and plan["commission_share"] > 0.005
    assert plan["cash_for_efficient_trade_cad"] == pytest.approx(5.0 / 0.005 * 1.4)


def test_large_contribution_is_buy_with_low_commission_share():
    plan = _plan(30000)
    assert plan["recommendation"] == "BUY" and plan["commission_share"] < 0.005
    assert plan["weights_after"]["SPY"] == pytest.approx(0.85, abs=0.02)


def test_higher_fx_spread_never_buys_more():
    cheap, dear = _plan(3000, fx_spread=0.0), _plan(3000, fx_spread=0.05)
    assert dear["invested_usd"] <= cheap["invested_usd"]


def test_never_spends_more_cad_than_available():
    for cash in (600, 1234.56, 2800, 9999):
        assert _plan(cash, fx_spread=0.015)["cad_converted"] <= cash + 1e-9


@pytest.mark.parametrize("kw", [dict(usdcad=0), dict(prices_usd={"SPY": 500.0}), dict(prices_usd={"SPY": 500.0, "QQQ": 0.0}),
                                dict(targets={"SPY": 0.8, "QQQ": 0.1})])
def test_bad_inputs_raise(kw):
    with pytest.raises(ValueError):
        _plan(1000, **kw)


# ---------------------------------------------------------------- CLI (network and Mongo stubbed)

import contribution_plan  # noqa: E402
import portfolio_view  # noqa: E402
import storage  # noqa: E402


@pytest.fixture
def stubbed(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: pytest.fail("live Yahoo call"))
    monkeypatch.setattr(storage, "get_account_holdings", lambda a: [
        {"symbol": "JPM", "quantity": 9, "kind": "security", "currency": "USD"}])
    monkeypatch.setattr(portfolio_view, "last_price",
                        lambda s: 1.4 if s == portfolio_view.FX_SYMBOL else {"SPY": 500.0, "QQQ": 400.0}.get(s))


def test_cli_buy_plan_output(stubbed, capsys):
    assert contribution_plan.main(["--cad", "30000", "--us-listed", "--fx-spread-pct", "1.5"]) == 0
    out = capsys.readouterr().out
    assert "RECOMMENDATION: BUY" in out and "x SPY" in out and "x QQQ" in out and "total friction" in out


def test_cli_wait_too_small(stubbed, capsys):
    assert contribution_plan.main(["--cad", "100", "--us-listed"]) == 0
    out = capsys.readouterr().out
    assert "RECOMMENDATION: WAIT" in out and "whole share" in out


def test_cli_wait_combine_note(stubbed, capsys):
    assert contribution_plan.main(["--cad", "740", "--us-listed", "--fx-spread-pct", "0"]) == 0
    out = capsys.readouterr().out
    assert "CONSIDER WAITING" in out and "percentage and does not shrink" in out


def test_cli_missing_price_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(storage, "get_account_holdings", lambda a: [])
    monkeypatch.setattr(portfolio_view, "last_price", lambda s: None)
    assert contribution_plan.main(["--cad", "1000", "--us-listed"]) == 1
    assert "could not fetch" in capsys.readouterr().err


def test_cli_ignores_non_index_holdings_in_the_sleeve(stubbed, capsys):
    contribution_plan.main(["--cad", "30000", "--us-listed"])
    assert "JPM" not in capsys.readouterr().out


def test_cli_default_delegates_to_cad_advisor(monkeypatch):
    from swingtrade import advisor
    seen = {}
    monkeypatch.setattr(advisor, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert contribution_plan.main(["--cad", "2500", "--commission", "5"]) == 0
    assert seen["argv"][:4] == ["--cash", "2500.0", "--commission", "5.0"] and "--targets" not in seen["argv"]


def test_cli_fx_flags_rejected_without_us_listed(capsys):
    assert contribution_plan.main(["--cad", "2500", "--fx-spread-pct", "1"]) == 1
    assert "--us-listed" in capsys.readouterr().err
