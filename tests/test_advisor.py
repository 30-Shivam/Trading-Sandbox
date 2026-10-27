"""swingtrade.advisor -- CAD TFSA rebalancing advisor: drift, cash-first routing, whole-share rounding, fill tracking.
Pure and offline: no network, no Mongo (the CLI's loaders and savers are patched)."""
import copy
import itertools
import random

import pytest

from swingtrade import advisor
from swingtrade.advisor import (allocate_cash_first, apply_fills, cash_to_fix_without_selling, compute_drift, parse_fills,
                                parse_targets, plan_purchase, sleeve_values)

V, Q = "VFV.TO", "XQQ.TO"
T = {V: 0.85, Q: 0.15}
P = {V: 100.0, Q: 50.0}


def _pos(sym, qty, avg=None):
    return {"symbol": sym, "raw_symbol": sym.split(".")[0], "quantity": float(qty), "currency": "CAD", "avg_cost": avg, "kind": "security"}


def _cash(amount):
    return {"symbol": "CASH.CAD", "raw_symbol": "CASH", "quantity": float(amount), "currency": "CAD", "avg_cost": None, "kind": "cash"}


# ---------------------------------------------------------------- targets / parsing
def test_default_targets_are_cad_listed_85_15():
    assert parse_targets(None) == {"VFV.TO": 0.85, "XQQ.TO": 0.15}


def test_parse_targets_accepts_qqc_alternative_and_upcases():
    assert parse_targets("vfv.to:0.85, qqc.to:0.15") == {"VFV.TO": 0.85, "QQC.TO": 0.15}


@pytest.mark.parametrize("bad", ["VFV.TO:0.8,XQQ.TO:0.1", "VFV.TO", "VFV.TO:1.2,XQQ.TO:-0.2"])
def test_parse_targets_rejects_bad_weights(bad):
    with pytest.raises(ValueError):
        parse_targets(bad)


def test_parse_fills_reads_several_fills():
    assert parse_fills("VFV.TO:15@135.20, xqq.to:7@52.1") == [
        {"symbol": "VFV.TO", "quantity": 15, "price": 135.2}, {"symbol": "XQQ.TO", "quantity": 7, "price": 52.1}]


@pytest.mark.parametrize("bad", ["", "VFV.TO:0@10", "VFV.TO:1.5@10", "VFV.TO:2@", "VFV.TO:2@0", "VFV.TO 2 10"])
def test_parse_fills_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_fills(bad)


# ---------------------------------------------------------------- valuation
def test_sleeve_values_ignores_cash_and_other_holdings():
    positions = [_pos(V, 10), _pos(Q, 4), _pos("ZAG.TO", 99), _cash(500)]
    assert sleeve_values(positions, {**P, "ZAG.TO": 14.0}, T) == {V: 1000.0, Q: 200.0}
    assert advisor.other_holdings(positions, T) == ["ZAG.TO"]


def test_sleeve_values_unheld_fund_is_zero_and_needs_no_price():
    assert sleeve_values([_pos(V, 2)], {V: 100.0}, T) == {V: 200.0, Q: 0.0}


def test_sleeve_values_never_values_a_held_fund_at_zero_silently():
    with pytest.raises(ValueError, match="no price"):
        sleeve_values([_pos(V, 2)], {}, T)


def test_sleeve_values_rejects_usd_held_fund():
    usd = {**_pos(V, 2), "currency": "USD"}
    with pytest.raises(ValueError, match="CAD"):
        sleeve_values([usd], P, T)


# ---------------------------------------------------------------- drift
def test_drift_weights_and_signs():
    d = compute_drift({V: 9000.0, Q: 1000.0}, T)
    assert d["total"] == 10000.0 and d["weights"] == {V: 0.9, Q: 0.1}
    assert d["drift"][V] == pytest.approx(0.05) and d["drift"][Q] == pytest.approx(-0.05)
    assert sorted(d["breached"]) == [V, Q]


def test_drift_exactly_on_band_is_not_a_breach():
    d = compute_drift({V: 8800.0, Q: 1200.0}, T, band=0.03)      # 88% vs 85%: exactly 3 points
    assert d["breached"] == []


def test_drift_just_past_band_is_a_breach():
    d = compute_drift({V: 8810.0, Q: 1190.0}, T, band=0.03)
    assert sorted(d["breached"]) == [V, Q]


def test_drift_within_band_not_breached():
    assert compute_drift({V: 8600.0, Q: 1400.0}, T)["breached"] == []


def test_drift_empty_sleeve_is_initial_build_not_a_breach():
    d = compute_drift({V: 0.0, Q: 0.0}, T)
    assert d["empty"] and d["breached"] == [] and d["total"] == 0.0


def test_drift_band_is_configurable():
    assert sorted(compute_drift({V: 8600.0, Q: 1400.0}, T, band=0.005)["breached"]) == [V, Q]


# ---------------------------------------------------------------- cash-first routing math
def test_cash_goes_all_to_the_underweight_fund_first():
    got = allocate_cash_first({V: 9000.0, Q: 1000.0}, T, 500.0)
    assert got[V] == 0.0 and got[Q] == pytest.approx(500.0)


def test_cash_beyond_the_shortfall_spills_by_target_weight():
    got = allocate_cash_first({V: 9000.0, Q: 1000.0}, T, 2000.0)     # sleeve 12000 -> 10200 / 1800
    assert got[V] == pytest.approx(1200.0) and got[Q] == pytest.approx(800.0)


def test_cash_exactly_at_the_cure_point_lands_on_target():
    cure = cash_to_fix_without_selling({V: 9000.0, Q: 1000.0}, T)
    assert cure == pytest.approx(9000 / 0.85 - 10000)
    got = allocate_cash_first({V: 9000.0, Q: 1000.0}, T, cure)
    assert got[V] == pytest.approx(0.0, abs=1e-6) and got[Q] == pytest.approx(cure)
    assert (1000 + got[Q]) / (10000 + cure) == pytest.approx(0.15)


def test_initial_build_splits_cash_by_target():
    got = allocate_cash_first({V: 0.0, Q: 0.0}, T, 1000.0)
    assert got == {V: pytest.approx(850.0), Q: pytest.approx(150.0)}


def test_zero_or_negative_cash_allocates_nothing():
    assert allocate_cash_first({V: 9000.0, Q: 1000.0}, T, 0.0) == {V: 0.0, Q: 0.0}
    assert allocate_cash_first({V: 9000.0, Q: 1000.0}, T, -5.0) == {V: 0.0, Q: 0.0}


def test_three_funds_most_overweight_leaves_first_then_the_next():
    t3 = {"A": 0.5, "B": 0.3, "C": 0.2}
    got = allocate_cash_first({"A": 6000.0, "B": 3000.0, "C": 0.0}, t3, 1000.0)
    assert got == {"A": 0.0, "B": pytest.approx(0.0, abs=1e-6), "C": pytest.approx(1000.0)}


def test_allocation_invariants_on_random_inputs():
    rng = random.Random(11)
    for _ in range(300):
        vals = {V: rng.uniform(0, 20000), Q: rng.uniform(0, 5000)}
        cash = rng.uniform(0.01, 8000)
        got = allocate_cash_first(vals, T, cash)
        assert sum(got.values()) == pytest.approx(cash)
        assert all(x >= 0 for x in got.values())
        worst = min(T, key=lambda s: vals[s] / T[s])           # the most underweight fund always receives money
        assert got[worst] > 0


def test_cash_to_fix_is_zero_when_already_on_target():
    assert cash_to_fix_without_selling({V: 8500.0, Q: 1500.0}, T) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- whole-share purchase planning
def _cost(shares, commission):
    return sum(n * P[s] for s, n in shares.items()) + commission * len(shares)


def test_buys_only_whole_shares_and_never_overspends():
    for cash in (60, 160, 333.33, 1000, 2750.5, 9999):
        plan = plan_purchase(P, {V: 5000.0, Q: 400.0}, T, cash, commission=5.0, max_commission_share=1.0)
        assert all(isinstance(n, int) and n > 0 for n in plan["shares"].values())
        assert _cost(plan["shares"], 5.0) <= cash + 1e-9
        assert plan["cash_left"] == pytest.approx(cash - _cost(plan["shares"], 5.0))
        assert plan["commission"] == pytest.approx(5.0 * len(plan["shares"]))


def test_underweight_fund_is_bought_first_and_the_overweight_one_is_left_alone():
    plan = plan_purchase(P, {V: 9000.0, Q: 1000.0}, T, 500.0, commission=5.0, max_commission_share=0.05)
    assert plan["shares"] == {Q: 9} and plan["trades"] == 1 and plan["recommendation"] == "BUY"
    assert plan["cash_left"] == pytest.approx(45.0)


def test_initial_build_is_close_to_85_15():
    plan = plan_purchase(P, {V: 0.0, Q: 0.0}, T, 10000.0, commission=5.0, max_commission_share=1.0)
    w = plan["weights_after"]
    assert w[V] == pytest.approx(0.85, abs=0.01) and w[Q] == pytest.approx(0.15, abs=0.01)
    assert _cost(plan["shares"], 5.0) <= 10000.0


def test_extra_commission_is_not_paid_for_a_negligible_improvement():
    plan = plan_purchase(P, {V: 85000.0, Q: 15000.0}, T, 160.0, commission=5.0, max_commission_share=1.0)
    assert plan["shares"] == {V: 1} and plan["trades"] == 1


def test_matches_brute_force_over_a_wider_search_space():
    for values, cash in [({V: 5000.0, Q: 400.0}, 1200.0), ({V: 0.0, Q: 0.0}, 2300.0), ({V: 9000.0, Q: 1000.0}, 1500.0),
                         ({V: 2000.0, Q: 2000.0}, 900.0)]:
        plan = plan_purchase(P, values, T, cash, commission=5.0, max_commission_share=1.0)
        best, best_key = None, None
        for nv, nq in itertools.product(range(0, 30), repeat=2):
            trades = (nv > 0) + (nq > 0)
            invested = nv * P[V] + nq * P[Q]
            cost = invested + 5.0 * trades
            if cost > cash + 1e-9:
                continue
            after = {V: values[V] + nv * P[V], Q: values[Q] + nq * P[Q]}
            err = advisor._weight_error(after, T, cash - cost)
            key = (round(err / advisor.ERROR_BUCKET), trades, -invested)
            if best_key is None or key < best_key:
                best, best_key = {s: n for s, n in ((V, nv), (Q, nq)) if n}, key
        assert plan["shares"] == best


def test_too_small_to_buy_one_share_waits():
    plan = plan_purchase(P, {V: 0.0, Q: 0.0}, T, 40.0, commission=5.0)
    assert plan["recommendation"] == "WAIT_TOO_SMALL" and plan["shares"] == {}
    assert plan["min_cash_for_one_share"] == pytest.approx(55.0)
    assert plan["cash_left"] == 40.0


def test_commission_heavy_order_inside_the_band_suggests_combining():
    held = {V: 8500.0, Q: 1500.0}                             # on target: nothing needs fixing, so batching is free
    plan = plan_purchase(P, held, T, 500.0, commission=5.0, max_commission_share=0.005)
    assert plan["recommendation"] == "WAIT_COMBINE"
    assert plan["cash_for_efficient_trade"] == pytest.approx(1000.0)
    assert plan["shares"]                                     # the order is still computed so you can buy anyway
    assert plan_purchase(P, held, T, 500.0, commission=5.0, max_commission_share=0.05)["recommendation"] == "BUY"


def test_commission_share_never_blocks_a_purchase_that_fixes_a_breach_or_builds_the_position():
    breached = plan_purchase(P, {V: 9000.0, Q: 1000.0}, T, 500.0, commission=5.0, max_commission_share=0.005)
    assert breached["recommendation"] == "BUY" and breached["commission_share"] > 0.005
    initial = plan_purchase(P, {V: 0.0, Q: 0.0}, T, 500.0, commission=5.0, max_commission_share=0.005)
    assert initial["recommendation"] == "BUY"
    # the same commission-heavy order inside a wide band is only a suggestion to combine
    assert plan_purchase(P, {V: 9000.0, Q: 1000.0}, T, 500.0, commission=5.0, max_commission_share=0.005,
                         band=0.10)["recommendation"] == "WAIT_COMBINE"


def test_no_cash_means_no_plan():
    plan = plan_purchase(P, {V: 9000.0, Q: 1000.0}, T, 0.0)
    assert plan["recommendation"] == "NO_CASH" and plan["shares"] == {}


def test_plan_requires_a_price_for_every_target():
    with pytest.raises(ValueError):
        plan_purchase({V: 100.0}, {V: 0.0, Q: 0.0}, T, 1000.0)


def test_plan_never_sells_and_never_buys_the_overweight_fund():
    rng = random.Random(5)
    for _ in range(100):
        values = {V: rng.uniform(5000, 20000), Q: rng.uniform(0, 1000)}
        plan = plan_purchase(P, values, T, rng.uniform(100, 3000), commission=5.0, max_commission_share=1.0)
        assert all(n > 0 for n in plan["shares"].values())
        if values[V] / T[V] > values[Q] / T[Q] and plan["ideal_dollars"][V] == 0.0:
            assert V not in plan["shares"]


# ---------------------------------------------------------------- fill tracking
def test_apply_fills_updates_shares_cash_and_cost_basis_including_commission():
    positions = [_pos(V, 10, avg=100.0), _cash(1000.0)]
    new, cash = apply_fills(positions, parse_fills("VFV.TO:5@110,XQQ.TO:3@50"), commission=5.0, targets=T)
    assert cash == pytest.approx(290.0)                       # 1000 - (550+5) - (150+5)
    by = {p["symbol"]: p for p in new}
    assert by[V]["quantity"] == 15 and by[V]["avg_cost"] == pytest.approx((100 * 10 + 555) / 15)
    assert by[Q]["quantity"] == 3 and by[Q]["avg_cost"] == pytest.approx(155 / 3) and by[Q]["currency"] == "CAD"
    assert by["CASH.CAD"]["quantity"] == pytest.approx(290.0)


def test_apply_fills_deposit_is_added_before_spending_and_creates_the_cash_row():
    new, cash = apply_fills([], parse_fills("VFV.TO:2@100"), commission=5.0, deposit=300.0, targets=T)
    assert cash == pytest.approx(95.0)
    assert {p["symbol"] for p in new} == {"VFV.TO", "CASH.CAD"}


def test_apply_fills_does_not_mutate_its_input():
    positions = [_pos(V, 10, avg=100.0), _cash(1000.0)]
    snapshot = copy.deepcopy(positions)
    apply_fills(positions, parse_fills("VFV.TO:1@100"), commission=5.0, targets=T)
    assert positions == snapshot


def test_apply_fills_rejects_overspending_and_records_nothing():
    positions = [_pos(V, 10, avg=100.0), _cash(100.0)]
    snapshot = copy.deepcopy(positions)
    with pytest.raises(ValueError, match="cash would go"):
        apply_fills(positions, parse_fills("VFV.TO:5@110"), commission=5.0, targets=T)
    assert positions == snapshot


def test_apply_fills_rejects_a_symbol_outside_the_targets():
    with pytest.raises(ValueError, match="not one of the target"):
        apply_fills([_cash(1000.0)], parse_fills("ZAG.TO:1@14"), commission=5.0, targets=T)


def test_apply_fills_keeps_unknown_cost_basis_unknown():
    new, _ = apply_fills([_pos(V, 10, avg=None), _cash(1000.0)], parse_fills("VFV.TO:1@100"), commission=0.0, targets=T)
    assert next(p for p in new if p["symbol"] == V)["avg_cost"] is None


# ---------------------------------------------------------------- CLI (offline)
@pytest.fixture
def cli(monkeypatch):
    state = {"positions": [], "saved": None}
    monkeypatch.setattr(advisor, "_load_holdings", lambda args: copy.deepcopy(state["positions"]))
    monkeypatch.setattr(advisor, "_save_holdings", lambda account, positions: state.__setitem__("saved", (account, positions)) or len(positions))
    return state


PRICE_ARGS = ["--price", "VFV.TO=135.20", "--price", "XQQ.TO=52.10"]


def test_cli_cash_plan_prints_manual_instructions(cli, capsys):
    cli["positions"] = [_pos(V, 60), _pos(Q, 20)]
    assert advisor.main(["--cash", "2500", *PRICE_ARGS]) == 0
    out = capsys.readouterr().out
    assert "ACTION: BUY" in out and "Scotia iTRADE" in out and "--confirm-fill" in out and "TSX" in out


def test_cli_drift_beyond_band_without_cash_says_hold_and_how_much_cash_fixes_it(cli, capsys):
    cli["positions"] = [_pos(V, 70), _pos(Q, 10)]
    assert advisor.main(PRICE_ARGS) == 0
    out = capsys.readouterr().out
    assert "BEYOND BAND" in out and "ACTION: HOLD" in out and "--cash" in out and "BUY" not in out.split("ACTION:")[1]


def test_cli_within_band_says_do_not_trade(cli, capsys):
    cli["positions"] = [_pos(V, 85), _pos(Q, 15 * 135.2 / 52.1)]
    assert advisor.main(PRICE_ARGS) == 0
    assert "ACTION: NONE. Every fund is within the band" in capsys.readouterr().out


def test_cli_custom_band_changes_the_verdict(cli, capsys):
    cli["positions"] = [_pos(V, 70), _pos(Q, 10)]
    assert advisor.main([*PRICE_ARGS, "--band-pct", "50"]) == 0
    assert "ACTION: NONE" in capsys.readouterr().out


def test_cli_empty_account_without_cash_asks_for_cash(cli, capsys):
    assert advisor.main(PRICE_ARGS) == 0
    assert "--cash" in capsys.readouterr().out


def test_cli_missing_price_is_an_error(cli, monkeypatch, capsys):
    cli["positions"] = [_pos(V, 10)]
    monkeypatch.setattr(advisor, "_fetch_prices", lambda symbols, overrides: dict(overrides))
    assert advisor.main(["--cash", "500"]) == 1
    assert "could not get a price" in capsys.readouterr().err


def test_cli_confirm_fill_saves_the_whole_account_and_needs_no_prices(cli, monkeypatch, capsys):
    cli["positions"] = [_pos(V, 10, avg=100.0), _pos("ZAG.TO", 5, avg=14.0), _cash(100.0)]
    monkeypatch.setattr(advisor, "_fetch_prices", lambda *a: pytest.fail("confirm-fill must not fetch prices"))
    assert advisor.main(["--confirm-fill", "XQQ.TO:9@50", "--cash", "500", "--commission", "5"]) == 0
    account, saved = cli["saved"]
    by = {p["symbol"]: p for p in saved}
    assert account == "scotia_tfsa" and set(by) == {V, Q, "ZAG.TO", "CASH.CAD"}      # other holdings survive the whole-account replace
    assert by[Q]["quantity"] == 9 and by["CASH.CAD"]["quantity"] == pytest.approx(100 + 500 - 455)
    assert "Recorded 1 fill" in capsys.readouterr().out


def test_cli_confirm_fill_dry_run_saves_nothing(cli, capsys):
    cli["positions"] = [_cash(1000.0)]
    assert advisor.main(["--confirm-fill", "VFV.TO:2@100", "--dry-run"]) == 0
    assert cli["saved"] is None and "DRY RUN" in capsys.readouterr().out


def test_cli_confirm_fill_overspend_saves_nothing(cli, capsys):
    cli["positions"] = [_cash(50.0)]
    assert advisor.main(["--confirm-fill", "VFV.TO:2@100"]) == 1
    assert cli["saved"] is None and "cash would go" in capsys.readouterr().err


def test_cli_confirm_fill_refuses_csv_source(cli, capsys):
    assert advisor.main(["--confirm-fill", "VFV.TO:1@100", "--holdings-csv", "x.csv"]) == 1
    assert cli["saved"] is None


def test_cli_bad_fill_and_bad_targets_are_clean_errors(cli, capsys):
    assert advisor.main(["--confirm-fill", "nonsense"]) == 1
    assert advisor.main(["--targets", "VFV.TO:0.5"]) == 1
    assert advisor.main(["--cash", "-5"]) == 1


def test_cli_loader_failure_is_a_clean_error(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("MONGODB_URI is not set")
    monkeypatch.setattr(advisor, "_load_holdings", boom)
    assert advisor.main(["--cash", "100"]) == 1
    assert "--holdings-csv" in capsys.readouterr().err
