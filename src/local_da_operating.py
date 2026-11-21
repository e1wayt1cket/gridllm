"""Historical load mapping, bounded inverter support and emergency service rules."""

from dataclasses import dataclass, replace
import csv
import json
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from local_da_accounting import ac_replay, physical_audit
from local_da_clearing import ClearingModel, choose_directions, probe_prices, qualification_label
from local_da_contracts import Generator
from run_local_da import ROOT, build_legacy_day, sha


@dataclass(frozen=True)
class OperatingRules:
    allow_emergency_shedding: bool = False
    bounded_reactive_support: bool = True
    inverter_smax_multiplier: float = 1.10
    shedding_tolerance_mwh: float = 1e-6


def remap_sources(active_load, selected, historical):
    """Move complete selected curves to their frozen user-level destination."""
    history = {row["user_id"]: row for row in historical}
    if len(history) != len(historical):
        raise ValueError("Historical users must be unique")
    result = np.zeros_like(active_load)
    kept, excluded, seen = [], [], set()
    for row in selected:
        source, user = row["source_id"], row["user_id"]
        if source in seen:
            raise ValueError("Selected source appears more than once")
        seen.add(source)
        if user not in history or source not in history[user]["meter_ids"].split(";"):
            raise ValueError("Selected source has no historical user-meter evidence")
        entry = history[user]
        if not entry["destination_bus"]:
            if entry["candidate_status"] != "unmapped" or not entry["mapping_reason"]:
                raise ValueError("An unmapped user requires an explicit historical reason")
            excluded.append(dict(row, reason=entry["mapping_reason"]))
            continue
        origin, destination = int(row["bus_zero_based"]), int(entry["destination_bus"])
        if not 1 <= destination < active_load.shape[1]:
            raise ValueError("Historical destination must be a non-slack load bus")
        result[:, destination] += active_load[:, origin]
        kept.append(dict(row, original_bus_zero_based=origin,
                         bus_zero_based=destination, bus_one_based=destination + 1))
    return result, kept, excluded


def build_historical_day(date="2022-12-22"):
    day, context = build_legacy_day(date)
    selected_path = ROOT / "reports/multi_user_load_2026-11-13/evidence/source_bus_mapping.csv"
    historical_path = ROOT / "data/processed/industry_meter_bus_map_v2.csv"
    historical_manifest = ROOT / "data/processed/industry_meter_bus_map_v2_manifest.json"
    metadata = json.loads(historical_manifest.read_text(encoding="utf-8"))
    if sha(historical_path) != metadata["mapping_sha256"]:
        raise ValueError("Historical mapping differs from its frozen manifest")
    with selected_path.open(encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    with historical_path.open(encoding="utf-8-sig", newline="") as stream:
        historical = list(csv.DictReader(stream))
    active, kept, excluded = remap_sources(day.active_load_mw, selected, historical)
    # The original map populated every destination once, so this recovers its
    # declared native nodal ratio without reading an unrelated load day.
    ratio = np.divide(day.reactive_load_mvar.sum(axis=0), day.active_load_mw.sum(axis=0),
                      out=np.zeros(day.network.buses), where=day.active_load_mw.sum(axis=0) > 0)
    reactive = active * ratio[None, :]
    removed = day.active_load_mw.sum(axis=1) - active.sum(axis=1)
    excluded_curves = sum((day.active_load_mw[:, int(row["bus_zero_based"])] for row in excluded),
                          np.zeros(day.periods))
    if not np.allclose(removed, excluded_curves, atol=1e-12, rtol=0):
        raise ValueError("Historical remapping changed more than explicitly excluded curves")
    context.update(mapping_version=metadata["dataset_version"], mapping_rule=metadata["rule"],
                   mapping_path=str(historical_path.relative_to(ROOT)),
                   historical_mapping_sha256=sha(historical_path), selected_sources=len(selected),
                   retained_sources=len(kept), excluded_sources=excluded,
                   populated_load_buses=int((active.max(axis=0) > 0).sum()),
                   native_background_added=False, active_load_scaled=False,
                   original_peak_mw=float(day.active_load_mw.sum(axis=1).max()),
                   historical_peak_mw=float(active.sum(axis=1).max()),
                   historical_energy_mwh=float(active.sum() * day.dt_hours),
                   excluded_energy_mwh=float(removed.sum() * day.dt_hours),
                   q_basis="Native destination-bus Q0/P0 assumption after historical remapping")
    return replace(day, active_load_mw=active, reactive_load_mvar=reactive), context, kept


def support_day(day, rules):
    if not np.isfinite(rules.inverter_smax_multiplier) or rules.inverter_smax_multiplier < 1:
        raise ValueError("An explicit apparent-power multiplier of at least one is required")
    if not rules.bounded_reactive_support:
        return day, {}
    if any(g.name.startswith("Q_") for g in day.generators):
        raise ValueError("Reactive support participants must not already be present")
    ratings = {g.name: (.3 if g.name.startswith("PV") else 1.) * rules.inverter_smax_multiplier
               for g in day.generators}
    generators = [replace(g, reactive_limit_mvar=ratings[g.name]) for g in day.generators]
    for unit in day.storage:
        name = f"Q_{unit.name}"
        rating = unit.power_mw * rules.inverter_smax_multiplier
        ratings[name] = rating
        generators.append(Generator(name, unit.bus, np.zeros(day.periods), reactive_limit_mvar=rating))
    return replace(day, generators=tuple(generators)), ratings


class OperatingModel(ClearingModel):
    """Keep service and inverter assumptions explicit across dispatch and pricing."""

    def __init__(self, day, rules, ratings, *, directions=None, exact=True,
                 seconds=120, log_path=None, minimum_shedding_mwh=None):
        super().__init__(day, directions=directions, exact=exact, seconds=seconds, log_path=log_path)
        self.rules, self.ratings = rules, ratings
        m = self.model
        self.market_objective = m.getObjective()
        self.shed = {}
        self.q_ratio = np.divide(day.reactive_load_mvar, day.active_load_mw,
                                 out=np.zeros_like(day.active_load_mw), where=day.active_load_mw > 0)
        if rules.allow_emergency_shedding:
            for t in range(day.periods):
                for b in range(day.network.buses):
                    if day.active_load_mw[t, b] <= 0:
                        continue
                    var = m.addVar(lb=0, ub=float(day.active_load_mw[t, b]), name=f"shed_{t}_{b}")
                    self.shed[t, b] = var
                    m.chgCoeff(self.balance[t, b], var, 1.)
                    m.chgCoeff(m.getConstrByName(f"reactive_balance_{t}_{b}"), var,
                               float(self.q_ratio[t, b]))
        self.shed_energy = gp.quicksum(day.dt_hours * var for var in self.shed.values())
        if minimum_shedding_mwh is not None:
            m.addConstr(self.shed_energy <= minimum_shedding_mwh + rules.shedding_tolerance_mwh,
                        name="minimum_emergency_shedding_cap")
        for gen in day.generators:
            if gen.name not in ratings:
                continue
            for t in range(day.periods):
                p, q = self.generation[gen.name][t], self.generator_q[gen.name][t]
                if gen.name.startswith("Q_"):
                    name = gen.name[2:]
                    p = self.discharge[name][t] - self.charge[name][t]
                m.addQConstr(p * p + q * q <= ratings[gen.name] ** 2,
                             name=f"inverter_circle_{gen.name}_{t}")
        m.update()

    def solve(self, *, prices=False):
        result = super().solve(prices=prices)
        result["solver_status_name"] = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
                                       9: "TIME_LIMIT", 12: "NUMERIC", 13: "SUBOPTIMAL"}.get(result["solver_status"], "OTHER")
        if result["solution_count"]:
            values = np.zeros_like(self.day.active_load_mw)
            for key, var in self.shed.items():
                values[key] = var.X
            result["shed_mw"] = values
            result["shed_q_mvar"] = values * self.q_ratio
            result["served_mw"] = self.day.active_load_mw - values
            result["shed_mwh"] = float(values.sum() * self.day.dt_hours)
        return result


def audit_operating(day, rules, ratings, result):
    served = replace(day, active_load_mw=result["served_mw"],
                     reactive_load_mvar=day.reactive_load_mvar - result["shed_q_mvar"])
    report = {"physical": physical_audit(served, result), "ac": ac_replay(served, result)}
    circle = 0.
    for gen in day.generators:
        if gen.name in ratings:
            p = result["generation"][gen.name]
            if gen.name.startswith("Q_"):
                name = gen.name[2:]
                p = result["discharge"][name] - result["charge"][name]
            circle = max(circle, float((p*p + result["generator_q"][gen.name]**2 - ratings[gen.name]**2).max()))
    report["inverter_circle_violation_max_mva2"] = max(0., circle)
    report["shed_bound_violation_mw"] = max(0., float((-result["shed_mw"]).max()),
                                            float((result["shed_mw"] - day.active_load_mw).max()))
    report["passed"] = (report["physical"]["passed"] and report["ac"]["passed"]
                         and circle <= 1e-6 and report["shed_bound_violation_mw"] <= 1e-6)
    return report


def clear_operating_day(day, rules, *, seconds=120, log_path=None, price_probes=False):
    day, ratings = support_day(day, rules)
    model = OperatingModel(day, rules, ratings, exact=True, seconds=seconds, log_path=log_path)
    output = {"qualified_for_settlement": False, "price_qualification": "NOT_TESTED",
              "thermal_limits_qualified": False, "reactive_ratings_mva": ratings,
              "reactive_support_basis": "Historical 1.10 multiplier research assumption; not device evidence"}
    minimum = None
    try:
        if rules.allow_emergency_shedding:
            model.model.Params.MIPGapAbs = 1e-8
            model.model.setObjective(model.shed_energy, GRB.MINIMIZE)
            first = model.solve()
            output["minimum_shedding_stage"] = first
            if not first["optimal"]:
                output["failure"] = "Minimum emergency shedding was not proven optimal"
                return output
            minimum = first["shed_mwh"]
            model.model.addConstr(model.shed_energy <= minimum + rules.shedding_tolerance_mwh,
                                  name="minimum_emergency_shedding_cap")
            model.model.setObjective(model.market_objective, GRB.MINIMIZE)
            model.model.Params.MIPGapAbs = .001
        dispatch = model.solve()
        output["dispatch"] = dispatch
    finally:
        model.close()
    if not dispatch["optimal"]:
        output["failure"] = "Exact operating dispatch was not proven optimal"
        return output
    output["dispatch_audit"] = audit_operating(day, rules, ratings, dispatch)
    output["full_service"] = dispatch["shed_mwh"] <= 1e-5
    output["physical_operating_qualified"] = output["dispatch_audit"]["passed"]
    if not price_probes or not output["full_service"] or not output["physical_operating_qualified"]:
        return output
    # Pricing reuses identical service and inverter conditions. Emergency
    # scarcity prices require a separate rule and are deliberately not issued.
    directions, idle = choose_directions(dispatch)
    pricing = OperatingModel(day, rules, ratings, directions=directions, exact=False,
                             seconds=seconds, minimum_shedding_mwh=minimum)
    try:
        result = pricing.solve(prices=True)
        output["pricing"] = result
        if not result["optimal"] or "lmp_cny_mwh" not in result:
            return output
        tolerance = max(.01, abs(dispatch["clearing_objective"]) * 1e-6)
        gap = abs(result["clearing_objective"] - dispatch["clearing_objective"])
        output["conditional_objective_difference_cny"] = gap
        buses = {day.network.slack, *(u.bus for u in day.storage)}
        peak = int(np.argmax(result["pcc_mw"]))
        v = np.sqrt(result["voltage_sq"][peak])
        # Probe the buses that sit on this day's own band, not on a literal one.
        lower = day.network.voltage_min + 1e-6
        upper = day.network.voltage_max - 1e-6
        buses.update(int(b) for b in np.flatnonzero((v <= lower) | (v >= upper)))
        probes = probe_prices(pricing, result, buses=sorted(buses))
        output["price_probes"] = probes
        output["price_qualification"] = qualification_label(probes)
        output["pricing_audit"] = audit_operating(day, rules, ratings, result)
    finally:
        pricing.close()
    alternative = {name: np.where(idle[name], -1, values) for name, values in directions.items()}
    output["idle_direction_stable"] = True
    if any(mask.any() for mask in idle.values()):
        alternate = OperatingModel(day, rules, ratings, directions=alternative, exact=False,
                                   seconds=seconds, minimum_shedding_mwh=minimum)
        try:
            other = alternate.solve(prices=True)
        finally:
            alternate.close()
        stable = other["optimal"] and "lmp_cny_mwh" in other
        if stable:
            difference = abs(result["lmp_cny_mwh"] - other["lmp_cny_mwh"])
            stable = bool((difference <= np.maximum(.1, abs(result["lmp_cny_mwh"]) * .01)).all())
            stable = stable and abs(result["clearing_objective"] - other["clearing_objective"]) <= tolerance
            output["idle_direction_max_price_difference_cny_mwh"] = float(difference.max())
        output["idle_direction_stable"] = bool(stable)
    output["qualified_for_settlement"] = (gap <= tolerance and output["pricing_audit"]["passed"]
        and output["idle_direction_stable"] and output["price_qualification"] in ("SMOOTH_PASS", "SUBGRADIENT_PASS"))
    return output
