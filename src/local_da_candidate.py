"""A declared 0.50 MVAr candidate with matched bid and reactive-service assumptions."""

from dataclasses import replace
import json

import numpy as np
from gurobipy import GRB

from local_da_clearing import choose_directions, probe_prices, qualification_label
from local_da_contracts import Generator
from local_da_operating import OperatingModel, OperatingRules, audit_operating, build_historical_day, support_day
from run_local_da import ROOT, sha

EXTERNAL_Q_NAME = "EXTERNAL33"
EXTERNAL_Q_CAPACITY_MVAR = .50


def build_candidate_day():
    day, context, mapping = build_historical_day("2022-12-22")
    folder = ROOT / "reports/historical_mapping_da_2026-11-13/historical_pf_evidence"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha(folder / "loads.npz") != manifest["outputs"]["loads.npz"]:
        raise ValueError("Historical power-factor load evidence changed")
    pf_context = json.loads((folder / "context.json").read_text(encoding="utf-8"))
    if pf_context["load_power_factor"] != .8503:
        raise ValueError("The authorized historical power factor must remain 0.8503")
    with np.load(folder / "loads.npz", allow_pickle=False) as archive:
        p, q = archive["active_load_mw"].copy(), archive["reactive_load_mvar"].copy()
    np.testing.assert_allclose(p, day.active_load_mw, atol=1e-12, rtol=0)
    np.testing.assert_allclose(q, p * np.tan(np.arccos(.8503)), atol=1e-12, rtol=0)
    day = replace(day, active_load_mw=p, reactive_load_mvar=q)
    rules = OperatingRules()
    day, ratings = support_day(day, rules)
    resource = Generator(EXTERNAL_Q_NAME, 32, np.zeros(day.periods),
                         reactive_limit_mvar=EXTERNAL_Q_CAPACITY_MVAR)
    day = replace(day, generators=day.generators + (resource,))
    context.update(load_power_factor=.8503, q_basis="Historical uniform PF=0.8503 research assumption",
                   external_q_bus_one_based=33, external_q_capacity_mvar=EXTERNAL_Q_CAPACITY_MVAR,
                   external_q_control="Continuous capacitive injection 0..0.50 MVAr",
                   storage_zero_quantity="Active-market withdrawal with retained reactive-service obligation",
                   reactive_cost_status="Reactive service and capital costs not evidenced or included",
                   candidate_only=True)
    return day, rules, ratings, context, mapping


def stress_cases(day):
    cases = [("reference", day)]
    for unit in day.storage:
        bids = dict(day.bids)
        bids[unit.name] = replace(bids[unit.name], charge_quantity_mw=np.zeros(day.periods),
                                 discharge_quantity_mw=np.zeros(day.periods))
        cases.append((f"zero_{unit.name}", replace(day, bids=bids)))
    bids = {name: replace(bid, charge_quantity_mw=np.zeros(day.periods),
                         discharge_quantity_mw=np.zeros(day.periods)) for name, bid in day.bids.items()}
    cases.append(("all_zero_quantity", replace(day, bids=bids)))
    bids = {name: replace(bid, charge_price_cny_mwh=np.zeros(day.periods),
                         discharge_price_cny_mwh=np.full(day.periods, 600.)) for name, bid in day.bids.items()}
    cases.append(("high_discharge_low_charge", replace(day, bids=bids)))
    cases.append(("terminal_soc_equal", replace(day, terminal_soc_equal=True)))
    return cases


def candidate_model(day, rules, ratings, *, exact=True, directions=None, seconds=120,
                    log_path=None, minimum_shedding_mwh=None):
    model = OperatingModel(day, rules, ratings, exact=exact, directions=directions,
                           seconds=seconds, log_path=log_path,
                           minimum_shedding_mwh=minimum_shedding_mwh)
    for t in range(day.periods):
        model.generator_q[EXTERNAL_Q_NAME][t].LB = 0.
        model.generator_q[EXTERNAL_Q_NAME][t].UB = EXTERNAL_Q_CAPACITY_MVAR
    model.model.update()
    return model


def audit_candidate(day, rules, ratings, result):
    audit = audit_operating(day, rules, ratings, result)
    values = result["generator_q"][EXTERNAL_Q_NAME]
    audit["external_q_violation_mvar"] = max(0., float((-values).max()),
                                             float((values - EXTERNAL_Q_CAPACITY_MVAR).max()))
    audit["passed"] = audit["passed"] and audit["external_q_violation_mvar"] <= 1e-6
    return audit


def solve_candidate(day, rules, ratings, *, seconds=120, log_path=None):
    output = {"qualified_for_settlement": False, "price_qualification": "NOT_TESTED",
              "reactive_service_obligation_retained": True, "candidate_only": True}
    model = candidate_model(day, rules, ratings, seconds=seconds, log_path=log_path)
    try:
        if rules.allow_emergency_shedding:
            model.model.Params.MIPGapAbs = 1e-8
            model.model.setObjective(model.shed_energy, GRB.MINIMIZE)
            first = model.solve()
            output["minimum_shedding_stage"] = first
            if not first["optimal"]:
                output["failure"] = "Minimum emergency shedding not proven optimal"
                return output
            model.model.addConstr(model.shed_energy <= first["shed_mwh"] + rules.shedding_tolerance_mwh,
                                  name="minimum_emergency_shedding_cap")
            model.model.Params.MIPGapAbs = .001
            model.model.setObjective(model.market_objective, GRB.MINIMIZE)
        dispatch = model.solve()
        output["dispatch"] = dispatch
    finally:
        model.close()
    if dispatch["optimal"]:
        output["audit"] = audit_candidate(day, rules, ratings, dispatch)
        output["full_service"] = dispatch["shed_mwh"] <= 1e-5
        output["physical_qualified"] = output["audit"]["passed"]
    return output


def price_candidate(day, rules, ratings, dispatch, *, seconds=120, output_folder=None):
    """Price only the physically qualified full-service reference."""
    output = {"qualified_for_settlement": False, "price_qualification": "NOT_TESTED", "dispatch": dispatch}
    if rules.allow_emergency_shedding or not dispatch["optimal"]:
        output["failure"] = "Only optimal full-service dispatch can enter candidate pricing"
        return output
    original_audit = audit_candidate(day, rules, ratings, dispatch)
    output["dispatch_audit"] = original_audit
    if not original_audit["passed"] or dispatch["shed_mwh"] > 1e-5:
        output["failure"] = "Physical dispatch not qualified for pricing"
        return output
    directions, idle = choose_directions(dispatch)
    model = candidate_model(day, rules, ratings, exact=False, directions=directions, seconds=seconds,
                             log_path=output_folder / "conditional_pricing.log" if output_folder else None)
    try:
        pricing = model.solve(prices=True)
        output["pricing"] = pricing
        if not pricing["optimal"] or "lmp_cny_mwh" not in pricing:
            output["failure"] = "Conditional prices unavailable"
            return output
        tolerance = max(.01, abs(dispatch["clearing_objective"]) * 1e-6)
        gap = abs(pricing["clearing_objective"] - dispatch["clearing_objective"])
        output["conditional_objective_difference_cny"] = gap
        output["pricing_audit"] = audit_candidate(day, rules, ratings, pricing)
        if gap > tolerance or not output["pricing_audit"]["passed"]:
            output["failure"] = "Conditional objective or physical reconstruction failed"
            return output
        buses = sorted({day.network.slack, 32, *(unit.bus for unit in day.storage)})
        peak = int(np.argmax(pricing["pcc_mw"]))
        voltage = np.sqrt(pricing["voltage_sq"])
        periods = {peak}
        # A representative extra active-voltage period supplements the peak
        # probe without claiming that all 96 derivatives have been tested.
        bound = np.flatnonzero(((voltage <= day.network.voltage_min + 1e-6)
                               | (voltage >= day.network.voltage_max - 1e-6)).any(axis=1))
        if len(bound):
            periods.add(int(bound[np.argmin(voltage[bound].min(axis=1))]))
        probes = probe_prices(model, pricing, periods=sorted(periods), buses=buses)
        output["price_probes"] = probes
        output["price_qualification"] = qualification_label(probes)
    finally:
        model.close()
    alternate_directions = {name: np.where(idle[name], -1, values) for name, values in directions.items()}
    output["idle_direction_stable"] = True
    if any(mask.any() for mask in idle.values()):
        alternate = candidate_model(day, rules, ratings, exact=False, directions=alternate_directions,
                                     seconds=seconds, log_path=output_folder / "alternate_idle_pricing.log" if output_folder else None)
        try:
            other = alternate.solve(prices=True)
        finally:
            alternate.close()
        output["alternate_idle"] = {key: other[key] for key in ["solver_status", "solver_status_name", "optimal", "clearing_objective"] if key in other}
        stable = other["optimal"] and "lmp_cny_mwh" in other
        if stable:
            difference = abs(other["lmp_cny_mwh"] - pricing["lmp_cny_mwh"])
            output["idle_price_difference_max_cny_mwh"] = float(difference.max())
            stable = bool((difference <= np.maximum(.1, abs(pricing["lmp_cny_mwh"]) * .01)).all())
            stable = stable and abs(other["clearing_objective"] - pricing["clearing_objective"]) <= tolerance
        output["idle_direction_stable"] = bool(stable)
    output["qualified_for_settlement"] = (output["price_qualification"] in ("SMOOTH_PASS", "SUBGRADIENT_PASS")
        and output["idle_direction_stable"] and output["pricing_audit"]["passed"])
    return output
