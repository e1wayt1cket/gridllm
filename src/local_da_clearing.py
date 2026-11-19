"""Continuous complementary dispatch and direction-conditional SOCP prices."""

from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from local_da_contracts import validate_day


class ClearingModel:
    """A model owns its solver resources; all participants bid simultaneously."""

    def __init__(self, day, *, directions=None, exact=True, seconds=120,
                 log_path=None, mip_gap=1e-6):
        validate_day(day)
        self.day = day
        self.model = m = gp.Model("independent_local_da")
        m.Params.OutputFlag = int(log_path is not None)
        if log_path is not None:
            m.Params.LogFile = str(log_path)
            m.Params.LogToConsole = 0
        m.Params.Threads = 2
        m.Params.Seed = 0
        m.Params.TimeLimit = seconds
        m.Params.FeasibilityTol = 1e-8
        m.Params.OptimalityTol = 1e-8
        # The cones are the only quadratics in either model, and 1e-8 keeps them
        # inside the 1e-6 physical audit on the reference day; Gurobi's default
        # 1e-6 leaves that gap at 1.8e-6. A tighter barrier does not buy any
        # exclusivity now that the disjunction is structural, and at 1e-10 the
        # exact model stalls before certifying a window the settlement accepts.
        m.Params.BarQCPConvTol = 1e-8
        m.Params.NumericFocus = 1
        m.Params.SoftMemLimit = 2
        # The exact model's exclusivity is an explicit binary disjunction, so its
        # only quadratics are the convex cones and the default nonconvex setting
        # applies. Presolve and presolve's linearization stay on: they are what
        # turn each disjunction into a branchable cut rather than a continuous
        # product the solver has to separate by spatial branching.
        m.Params.NonConvex = 0
        m.Params.QCPDual = int(not exact)
        if exact:
            # A caller that already accepts a looser gap hands it in, so the
            # solver stops there instead of spending its limit closing a gap the
            # caller would not have used.
            m.Params.MIPGap = float(mip_gap)
            m.Params.MIPGapAbs = .001
        net, periods = day.network, day.periods
        buses, lines = range(net.buses), range(len(net.branches))
        self.v = m.addVars(periods, buses, lb=net.voltage_min ** 2,
                           ub=net.voltage_max ** 2, name="voltage_sq")
        self.p = m.addVars(periods, lines, lb=-GRB.INFINITY, name="branch_p")
        self.q = m.addVars(periods, lines, lb=-GRB.INFINITY, name="branch_q")
        self.i = m.addVars(periods, lines, lb=0, name="current_sq")
        self.pcc = m.addVars(periods, lb=-GRB.INFINITY if day.export_limit_mw is None
                            else -day.export_limit_mw,
                            ub=GRB.INFINITY if day.import_limit_mw is None else day.import_limit_mw,
                            name="pcc_net_import")
        self.qcc = m.addVars(periods, lb=-GRB.INFINITY, name="pcc_q")
        self.charge, self.discharge, self.energy = {}, {}, {}
        self.is_charge = {}
        objective = gp.LinExpr()
        self.idle_positions = []
        for unit in day.storage:
            name, bid = unit.name, day.bids[unit.name]
            self.charge[name] = m.addVars(periods, lb=0, name=f"charge_{name}")
            self.discharge[name] = m.addVars(periods, lb=0, name=f"discharge_{name}")
            # One disjunction per unit and period; a direction-fixed pricing model
            # already bounds one leg at zero and needs no binary.
            if exact and directions is None:
                self.is_charge[name] = m.addVars(
                    periods, vtype=GRB.BINARY, name=f"is_charge_{name}")
            self.energy[name] = m.addVars(periods + 1, lb=unit.soc_min * unit.energy_mwh,
                                        ub=unit.soc_max * unit.energy_mwh, name=f"energy_{name}")
            m.addConstr(self.energy[name][0] == unit.soc_initial * unit.energy_mwh)
            if day.terminal_soc_equal:
                m.addConstr(self.energy[name][periods] == unit.soc_initial * unit.energy_mwh)
            if directions is not None:
                if name not in directions or np.asarray(directions[name]).shape != (periods,):
                    raise ValueError("Conditional pricing requires a direction for each storage period")
                if not np.isin(directions[name], [-1, 1]).all():
                    raise ValueError("A conditional direction must open one leg, never lock both")
            for t in range(periods):
                ch, dis = self.charge[name][t], self.discharge[name][t]
                ch.UB, dis.UB = float(bid.charge_quantity_mw[t]), float(bid.discharge_quantity_mw[t])
                if directions is not None:
                    if directions[name][t] == 1:
                        dis.UB = 0
                    else:
                        ch.UB = 0
                m.addConstr(self.energy[name][t + 1] == self.energy[name][t]
                            + day.dt_hours * (unit.eta_charge * ch - dis / unit.eta_discharge))
                if name in self.is_charge:
                    # Exclusivity as a disjunction over the declared quantities:
                    # one leg is carried at zero each period. The bilinear
                    # equality this replaced left the solver to separate a
                    # product by spatial branching, which in a day of 20-period
                    # windows could consume the whole limit without an incumbent.
                    z = self.is_charge[name][t]
                    m.addConstr(ch <= float(bid.charge_quantity_mw[t]) * z,
                                name=f"charge_only_if_{name}_{t}")
                    m.addConstr(dis <= float(bid.discharge_quantity_mw[t]) * (1 - z),
                                name=f"discharge_only_if_{name}_{t}")
                objective += day.dt_hours * (float(bid.discharge_price_cny_mwh[t]) * dis
                                             - float(bid.charge_price_cny_mwh[t]) * ch)
        self.generation, self.generator_q = {}, {}
        for generator in day.generators:
            name = generator.name
            self.generation[name] = m.addVars(periods, lb=0, name=f"generation_{name}")
            self.generator_q[name] = m.addVars(periods, lb=-generator.reactive_limit_mvar,
                                              ub=generator.reactive_limit_mvar, name=f"reactive_{name}")
            for t in range(periods):
                self.generation[name][t].UB = float(generator.available_mw[t])
                objective += day.dt_hours * generator.offer_cny_mwh * self.generation[name][t]
        self.balance = {}
        incoming, outgoing = {}, {}
        for line, (start, end) in enumerate(net.branches):
            incoming[end] = line
            outgoing.setdefault(start, []).append(line)
        for t in range(periods):
            m.addConstr(self.v[t, net.slack] == net.slack_vm_pu ** 2)
            objective += day.dt_hours * float(day.external_price_cny_mwh[t]) * self.pcc[t]
            for bus in buses:
                active, reactive = gp.LinExpr(), gp.LinExpr()
                if bus == net.slack:
                    active += self.pcc[t]
                    reactive += self.qcc[t]
                for unit in day.storage:
                    if unit.bus == bus:
                        active += self.discharge[unit.name][t] - self.charge[unit.name][t]
                for generator in day.generators:
                    if generator.bus == bus:
                        active += self.generation[generator.name][t]
                        reactive += self.generator_q[generator.name][t]
                for shunt_bus, mvar in net.shunts:
                    if shunt_bus == bus:
                        reactive += float(mvar)
                if bus in incoming:
                    line = incoming[bus]
                    active += self.p[t, line] - net.r_mw_base[line] * self.i[t, line]
                    reactive += self.q[t, line] - net.x_mw_base[line] * self.i[t, line]
                for line in outgoing.get(bus, []):
                    active -= self.p[t, line]
                    reactive -= self.q[t, line]
                self.balance[t, bus] = m.addConstr(active == float(day.active_load_mw[t, bus]),
                                                   name=f"active_balance_{t}_{bus}")
                m.addConstr(reactive == float(day.reactive_load_mvar[t, bus]),
                            name=f"reactive_balance_{t}_{bus}")
            for line, (start, end) in enumerate(net.branches):
                r, x = float(net.r_mw_base[line]), float(net.x_mw_base[line])
                p, q, current, v = self.p[t, line], self.q[t, line], self.i[t, line], self.v[t, start]
                m.addConstr(self.v[t, end] == v - 2 * (r * p + x * q) + (r*r + x*x) * current)
                m.addQConstr(p*p + q*q <= v * current, name=f"cone_{t}_{line}")
                if net.current_sq_limit is not None:
                    current.UB = float(net.current_sq_limit[line])
        m.setObjective(objective, GRB.MINIMIZE)
        m.update()
        # A MIP carries no duals, so the pricing solve must stay continuous: its
        # directions are already fixed and every leg it does not open is zero.
        if directions is not None and m.NumIntVars != 0:
            raise RuntimeError("Conditional pricing must not declare integer variables")

    def close(self):
        self.model.dispose()

    def solve(self, *, prices=False):
        started = time.monotonic()
        self.model.optimize()
        m, day = self.model, self.day
        info = {"solver_status": int(m.Status), "optimal": m.Status == GRB.OPTIMAL,
                "solution_count": int(m.SolCount), "runtime_seconds": time.monotonic() - started,
                "declared_integer_variables": int(m.NumIntVars),
                "declared_binary_variables": int(m.NumBinVars)}
        if m.SolCount == 0:
            return info
        info["clearing_objective"] = float(m.ObjVal)
        if m.IsMIP:
            info.update(objective_bound=float(m.ObjBound), optimality_gap=float(m.MIPGap))
        periods = range(day.periods)

        def curve(variables):
            return np.array([variables[t].X for t in range(len(variables))])

        info.update(pcc_mw=curve(self.pcc), qcc_mvar=curve(self.qcc),
                    voltage_sq=np.array([[self.v[t, b].X for b in range(day.network.buses)] for t in periods]),
                    branch_p=np.array([[self.p[t, line].X for line in range(len(day.network.branches))] for t in periods]),
                    branch_q=np.array([[self.q[t, line].X for line in range(len(day.network.branches))] for t in periods]),
                    current_sq=np.array([[self.i[t, line].X for line in range(len(day.network.branches))] for t in periods]),
                    charge={name: curve(values) for name, values in self.charge.items()},
                    discharge={name: curve(values) for name, values in self.discharge.items()},
                    energy={name: curve(values) for name, values in self.energy.items()},
                    generation={name: curve(values) for name, values in self.generation.items()},
                    generator_q={name: curve(values) for name, values in self.generator_q.items()})
        if prices:
            try:
                info["lmp_cny_mwh"] = np.array([[self.balance[t, b].Pi / day.dt_hours
                                                  for b in range(day.network.buses)] for t in periods])
            except (AttributeError, gp.GurobiError) as error:
                info["price_extraction_error"] = str(error)
        return info


def choose_directions(result, *, tolerance=1e-6, idle_direction=1):
    directions, idle = {}, {}
    for name in result["charge"]:
        ch, dis = result["charge"][name], result["discharge"][name]
        if np.max(np.minimum(ch, dis)) > tolerance:
            raise ValueError("Cannot price a dispatch that violates charge/discharge exclusivity")
        idle[name] = (ch <= tolerance) & (dis <= tolerance)
        directions[name] = np.where(idle[name], idle_direction, np.where(ch >= dis, 1, -1))
    return directions, idle


def probe_prices(model, reference, *, periods=None, buses=None):
    """Demand derivatives are measured with Q and storage directions fixed."""
    day = model.day
    periods = [int(np.argmax(reference["pcc_mw"]))] if periods is None else periods
    buses = sorted({day.network.slack, *(u.bus for u in day.storage)}) if buses is None else buses
    records = []
    for t in periods:
        for bus in buses:
            constraint = model.balance[t, bus]
            original = constraint.RHS
            dual = float(reference["lmp_cny_mwh"][t, bus])
            for epsilon in [1e-4, 1e-5]:
                values = {}
                try:
                    for sign, name in [(1, "right"), (-1, "left")]:
                        constraint.RHS = original + sign * epsilon
                        result = model.solve()
                        if result["optimal"]:
                            values[name] = (result["clearing_objective"] - reference["clearing_objective"]) / (sign * epsilon * day.dt_hours)
                        else:
                            values[name] = None
                finally:
                    constraint.RHS = original
                tolerance = max(.1, abs(dual) * .01)
                smooth = all(value is not None and abs(value - dual) <= tolerance for value in values.values())
                bracket = (values["left"] is not None and values["right"] is not None
                           and values["left"] - tolerance <= dual <= values["right"] + tolerance)
                records.append({"period": int(t), "bus": int(bus), "epsilon_mw": epsilon,
                                "dual_cny_mwh": dual, "left_cny_mwh": values["left"],
                                "right_cny_mwh": values["right"], "smooth_pass": smooth,
                                "subgradient_bracket": bracket})
    return records


def qualification_label(probes):
    """The label the probes support. A corner is labelled as a corner, not smoothed.

    Every probe must satisfy the same rule: all smooth, or all bracketed by their
    one-sided derivatives. An empty probe list is untested, and a probe that fails
    both rules fails the day.
    """
    if not probes:
        return "NOT_TESTED"
    if all(p["smooth_pass"] for p in probes):
        return "SMOOTH_PASS"
    if all(p["subgradient_bracket"] for p in probes):
        return "SUBGRADIENT_PASS"
    return "NONSMOOTH_OR_FAILED"


def clear_day(day, *, seconds=120, price_probes=True, log_path=None):
    """Return explicit dispatch and conditional-price qualification states."""
    exact_model = ClearingModel(day, exact=True, seconds=seconds, log_path=log_path)
    try:
        dispatch = exact_model.solve()
    finally:
        exact_model.close()
    output = {"dispatch": dispatch, "qualified_for_settlement": False,
              "price_qualification": "NOT_TESTED", "thermal_limits_qualified": day.network.current_sq_limit is not None}
    if not dispatch["optimal"]:
        output["failure"] = "Exact complementary dispatch was not optimal"
        return output
    try:
        directions, idle = choose_directions(dispatch)
    except ValueError as error:
        output["failure"] = str(error)
        return output
    pricing_model = ClearingModel(day, directions=directions, exact=False, seconds=seconds)
    try:
        pricing = pricing_model.solve(prices=True)
        output["pricing"] = pricing
        if not pricing["optimal"] or "lmp_cny_mwh" not in pricing:
            output["failure"] = "Direction-conditional prices unavailable or solve not optimal"
            return output
        tolerance = max(.01, abs(dispatch["clearing_objective"]) * 1e-6)
        gap = abs(pricing["clearing_objective"] - dispatch["clearing_objective"])
        output["conditional_objective_difference_cny"] = gap
        output["conditional_schedule_difference_mw"] = max(
            [float(np.max(abs(pricing[leg][name] - dispatch[leg][name])))
             for leg in ["charge", "discharge"] for name in pricing[leg]] or [0.])
        if gap > tolerance:
            output["failure"] = "Conditional dispatch is incompatible with exact objective"
            return output
        if price_probes:
            buses = {day.network.slack, *(u.bus for u in day.storage)}
            peak = int(np.argmax(pricing["pcc_mw"]))
            voltage = np.sqrt(pricing["voltage_sq"][peak])
            buses.update(int(b) for b in np.flatnonzero((voltage <= day.network.voltage_min + 1e-6)
                                                        | (voltage >= day.network.voltage_max - 1e-6)))
            probes = probe_prices(pricing_model, pricing, buses=sorted(buses))
            output["price_probes"] = probes
            output["price_qualification"] = qualification_label(probes)
    finally:
        pricing_model.close()
    alternate = {name: np.where(idle[name], -1, values) for name, values in directions.items()}
    if any(mask.any() for mask in idle.values()):
        alternative_model = ClearingModel(day, directions=alternate, exact=False, seconds=seconds)
        try:
            alternative = alternative_model.solve(prices=True)
        finally:
            alternative_model.close()
        output["idle_direction_alternative"] = {key: alternative[key] for key in
                                               ["solver_status", "optimal", "clearing_objective"] if key in alternative}
        if alternative["optimal"] and "lmp_cny_mwh" in alternative:
            delta = abs(pricing["lmp_cny_mwh"] - alternative["lmp_cny_mwh"])
            relative = np.maximum(.1, abs(pricing["lmp_cny_mwh"]) * .01)
            output["idle_direction_price_max_difference_cny_mwh"] = float(delta.max())
            output["idle_direction_stable"] = bool((delta <= relative).all()) and abs(alternative["clearing_objective"] - pricing["clearing_objective"]) <= tolerance
        else:
            output["idle_direction_stable"] = False
    else:
        output["idle_direction_stable"] = True
    from local_da_accounting import ac_replay, physical_audit
    output["physical_audit"] = physical_audit(day, pricing)
    output["exact_physical_audit"] = physical_audit(day, dispatch)
    output["ac_replay"] = ac_replay(day, pricing)
    output["qualified_for_settlement"] = (output["physical_audit"]["passed"]
                                           and output["exact_physical_audit"]["passed"]
                                           and output["ac_replay"]["passed"]
                                           and output["idle_direction_stable"]
                                           and output["price_qualification"] in ("SMOOTH_PASS", "SUBGRADIENT_PASS"))
    return output
