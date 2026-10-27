"""Pure whole-share contribution planner: given new CAD cash (and any USD cash already in the account), decide how many
whole shares of each index fund to buy so the INDEX SLEEVE (e.g. SPY/QQQ) lands as close as possible to its target mix,
counting commissions and the CAD->USD conversion spread. No network, no files; prices and FX are passed in.

Assumptions (all parameters, none verified against a specific broker): shares are bought whole; commission is charged per
TRADE in USD; the conversion spread is a percentage markup over the mid USD/CAD rate applied only to the CAD actually
converted; any USD cash left over stays in the USD account for the next contribution.
"""
MAX_SHARES_SEARCHED = 400


def _sleeve_error(values: dict[str, float], targets: dict[str, float]) -> float:
    total = sum(values.values())
    if total <= 0:
        return 1.0 + 1e-9   # an empty sleeve is worse than any non-empty one
    return sum(abs(values.get(s, 0.0) / total - w) for s, w in targets.items())


def plan_contribution(cash_cad: float, prices_usd: dict[str, float], usdcad: float, sleeve_values_usd: dict[str, float],
                      targets: dict[str, float], cash_usd: float = 0.0, commission_usd: float = 4.99,
                      fx_spread: float = 0.015, max_commission_share: float = 0.005) -> dict:
    """Returns the best feasible whole-share purchase and the reasoning numbers.

    Best = lowest L1 distance between the sleeve's post-purchase weights and `targets`; ties go to the purchase that invests
    more, then to lower friction. `fx_spread` is a fraction (0.015 = 1.5%). `recommendation` is 'BUY' unless nothing is
    affordable ('WAIT_TOO_SMALL') or the commissions alone exceed `max_commission_share` of the amount invested ('WAIT_COMBINE':
    consider saving up and buying once)."""
    if usdcad <= 0 or any(prices_usd.get(s, 0) <= 0 for s in targets):
        raise ValueError("need a positive USD/CAD rate and a positive price for every target symbol")
    if abs(sum(targets.values()) - 1.0) > 1e-6:
        raise ValueError("targets must sum to 1.0 (they are weights WITHIN the index sleeve)")
    rate = usdcad * (1.0 + fx_spread)                    # CAD paid per USD received
    max_usd = cash_usd + max(cash_cad, 0.0) / rate
    syms = list(targets)
    limits = {s: min(int(max_usd // prices_usd[s]), MAX_SHARES_SEARCHED) for s in syms}

    best, best_key = None, None

    def search(i: int, chosen: dict[str, int]):
        nonlocal best, best_key
        if i == len(syms):
            cand = _evaluate(chosen)
            if cand is None:
                return
            key = (round(cand["error_after"], 6), -cand["invested_usd"], cand["friction_cad"])
            if best_key is None or key < best_key:
                best, best_key = cand, key
            return
        for n in range(limits[syms[i]] + 1):
            search(i + 1, {**chosen, syms[i]: n})

    def _evaluate(shares: dict[str, int]) -> dict | None:
        trades = sum(1 for n in shares.values() if n > 0)
        share_cost = sum(n * prices_usd[s] for s, n in shares.items())
        commissions = trades * commission_usd
        needed = share_cost + commissions
        usd_from_cad = max(0.0, needed - cash_usd)
        cad_spent = usd_from_cad * rate
        if cad_spent > cash_cad + 1e-9:
            return None
        fx_cost_cad = usd_from_cad * usdcad * fx_spread
        after = {s: sleeve_values_usd.get(s, 0.0) + shares.get(s, 0) * prices_usd[s] for s in syms}
        return {"shares": {s: n for s, n in shares.items() if n > 0}, "trades": trades, "invested_usd": share_cost,
                "commission_usd": commissions, "usd_from_cad": usd_from_cad, "cad_converted": cad_spent,
                "fx_cost_cad": fx_cost_cad, "friction_cad": commissions * usdcad + fx_cost_cad,
                "cad_left": cash_cad - cad_spent, "usd_left": cash_usd + usd_from_cad - needed,
                "weights_after": {s: (v / sum(after.values()) if sum(after.values()) else 0.0) for s, v in after.items()},
                "error_after": _sleeve_error(after, targets)}

    search(0, {})
    out = best or {"shares": {}, "trades": 0, "invested_usd": 0.0}
    invested = out["invested_usd"]
    out["commission_share"] = (out.get("commission_usd", 0.0) / invested) if invested else None
    if not out["shares"]:
        cheapest = min(prices_usd[s] for s in syms)
        out["recommendation"] = "WAIT_TOO_SMALL"
        out["min_cash_cad_for_one_share"] = (cheapest + commission_usd - cash_usd) * rate
    elif (out["commission_share"] or 0.0) > max_commission_share:
        out["recommendation"] = "WAIT_COMBINE"
        out["cash_for_efficient_trade_cad"] = (out["commission_usd"] / max_commission_share) * rate
    else:
        out["recommendation"] = "BUY"
    out["friction_share_of_invested"] = (out["friction_cad"] / (invested * usdcad)) if invested and "friction_cad" in out else None
    return out
