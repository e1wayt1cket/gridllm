# dispatch_dc.py
"""DC-OPF solvers: Gurobi (primary) + HiGHS via ortools (fallback)."""

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False
    gp = None  # type: ignore
    GRB = None  # type: ignore

from dispatch_core import (
    _build_line_params,
    _build_agent_info,
    _classify_storage_mode,
)


def solve_dc_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                        action_params, config):
    if not _HAS_GUROBI:
        return _solve_dc_opf_highs(net, agents, t, stage, prev_soc,
                                    wholesale_t, action_params, config)
    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    x, limit = _build_line_params(net, base_kv)

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config)

    m = gp.Model("DC_OPF")
    m.setParam('OutputFlag', 0)
    theta = m.addVars(buses, lb=-GRB.INFINITY, name="theta")
    p_flow = m.addVars(lines, lb=-GRB.INFINITY, name="p_flow")
    m.addConstr(theta[slack_bus] == 0, "ref_angle")

    served = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    for a in agents:
        nm = a.name
        info = agent_info[nm]
        served[nm] = m.addVar(lb=0, ub=info['load'], name=f"served_{nm}")
        unserved[nm] = m.addVar(lb=0, ub=info['load'], name=f"unserv_{nm}")
        m.addConstr(served[nm] + unserved[nm] == info['load'], f"load_bal_{nm}")
        pv[nm] = m.addVar(lb=0, ub=info['pv_max'], name=f"pv_{nm}")
        wind[nm] = m.addVar(lb=0, ub=info['wind_max'], name=f"wind_{nm}")
        if info['has_storage'] and (info['ch_max'] > 0 or info['dis_max'] > 0):
            ch[nm] = m.addVar(lb=0, ub=info['ch_max'], name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=info['dis_max'], name=f"dis_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid_import = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_import")
    p_grid_export = m.addVar(lb=0, ub=config.reverse_power_limit_mw, name="p_grid_export")

    net_inj = {b: gp.LinExpr() for b in buses}
    net_inj[slack_bus] += p_grid_import - p_grid_export
    for a in agents:
        bus = agent_info[a.name]['bus']
        nm = a.name
        net_inj[bus] += pv[nm] + wind[nm] + dis[nm] - served[nm] - ch[nm]
    for b in buses:
        flow_out = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                flow_out += p_flow[l]
            elif net.line.at[l, 'to_bus'] == b:
                flow_out -= p_flow[l]
        m.addConstr(net_inj[b] == flow_out, f"p_balance_{b}")

    for l in lines:
        f = net.line.at[l, 'from_bus']
        to_bus = net.line.at[l, 'to_bus']
        m.addConstr(p_flow[l] == (theta[f] - theta[to_bus]) / x[l], f"dc_flow_{l}")
        m.addConstr(p_flow[l] <= limit[l], f"limit_pos_{l}")
        m.addConstr(p_flow[l] >= -limit[l], f"limit_neg_{l}")

    obj = gp.LinExpr()
    for a in agents:
        nm = a.name
        obj += agent_info[nm]['bid'] * served[nm]
        if a.storage is not None:
            obj += wholesale_t * (dis[nm] - ch[nm])
        else:
            obj -= agent_info[nm]['offer'] * (pv[nm] + wind[nm] + dis[nm])
        obj -= config.penalty_unserved * unserved[nm]
    obj -= wholesale_t * (p_grid_import - p_grid_export)
    if config.enable_multi_objective and config.lambda_carbon > 0:
        from dispatch_core import DT_HOURS
        obj -= config.lambda_carbon * config.emission_factor_grid * p_grid_import * DT_HOURS
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        return False, None, 0.0, {}, 0.0

    lmp = np.zeros(len(buses))
    for i, b in enumerate(buses):
        constr = m.getConstrByName(f"p_balance_{b}")
        if constr is not None:
            lmp[i] = -constr.Pi
        else:
            lmp[i] = wholesale_t

    agent_res = {}
    for a in agents:
        nm = a.name
        ch_val = ch[nm].X
        dis_val = dis[nm].X
        agent_res[nm] = {
            'served': served[nm].X, 'pv_used': pv[nm].X, 'wind_used': wind[nm].X,
            'p_ch': ch_val, 'p_dis': dis_val,
            'storage_mode': _classify_storage_mode(ch_val, dis_val),
        }
    return True, lmp, m.ObjVal, agent_res, p_grid_import.X - p_grid_export.X


def _solve_dc_opf_highs(net, agents, t, stage, prev_soc, wholesale_t,
                         action_params, config):
    """DC-OPF using ortools HiGHS (open-source fallback)."""
    from ortools.linear_solver import pywraplp

    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    x, limit = _build_line_params(net, base_kv)

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config)

    solver = pywraplp.Solver.CreateSolver("HIGHS")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return False, None, 0.0, {}, 0.0

    INF = solver.infinity()

    theta = {b: solver.NumVar(-INF, INF, f"theta_{b}") for b in buses}
    solver.Add(theta[slack_bus] == 0)

    p_flow = {}
    for l in lines:
        p_flow[l] = solver.NumVar(-limit[l], limit[l], f"p_{l}")
        f = net.line.at[l, 'from_bus']
        t_b = net.line.at[l, 'to_bus']
        solver.Add(p_flow[l] == (theta[f] - theta[t_b]) / x[l])

    p_grid_import = solver.NumVar(0, INF, "p_grid_import")
    p_grid_export = solver.NumVar(0, config.reverse_power_limit_mw, "p_grid_export")

    served = {}; unserved = {}; pv_v = {}; wind_v = {}; ch_v = {}; dis_v = {}
    for a in agents:
        nm = a.name
        info = agent_info[nm]
        served[nm] = solver.NumVar(0, info['load'], f"s_{nm}")
        unserved[nm] = solver.NumVar(0, info['load'], f"u_{nm}")
        solver.Add(served[nm] + unserved[nm] == info['load'])
        pv_v[nm] = solver.NumVar(0, info['pv_max'], f"pv_{nm}")
        wind_v[nm] = solver.NumVar(0, info['wind_max'], f"w_{nm}")
        if info['has_storage'] and (info['ch_max'] > 0 or info['dis_max'] > 0):
            ch_v[nm] = solver.NumVar(0, info['ch_max'], f"ch_{nm}")
            dis_v[nm] = solver.NumVar(0, info['dis_max'], f"dis_{nm}")
            is_ch = solver.IntVar(0, 1, f"ich_{nm}")
            is_dis = solver.IntVar(0, 1, f"idis_{nm}")
            solver.Add(is_ch + is_dis <= 1)
            solver.Add(ch_v[nm] <= info['ch_max'] * is_ch)
            solver.Add(dis_v[nm] <= info['dis_max'] * is_dis)
        else:
            ch_v[nm] = solver.NumVar(0, 0, f"ch_{nm}")
            dis_v[nm] = solver.NumVar(0, 0, f"dis_{nm}")

    # Power balance
    p_grid_net = p_grid_import - p_grid_export
    net_inj = {b: p_grid_net if b == slack_bus else solver.Sum() for b in buses}
    for a in agents:
        nm = a.name
        bus = agent_info[nm]['bus']
        net_inj[bus] += pv_v[nm] + wind_v[nm] + dis_v[nm] - served[nm] - ch_v[nm]

    pbal = {}
    for b in buses:
        flow_out = solver.Sum()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                flow_out += p_flow[l]
            elif net.line.at[l, 'to_bus'] == b:
                flow_out -= p_flow[l]
        pbal[b] = solver.Add(net_inj[b] == flow_out)

    # Objective: same structure as Gurobi version
    obj = solver.Objective()
    for a in agents:
        nm = a.name
        obj.SetCoefficient(served[nm], agent_info[nm]['bid'])
        if a.storage is not None:
            obj.SetCoefficient(dis_v[nm], wholesale_t)
            obj.SetCoefficient(ch_v[nm], -wholesale_t)
        else:
            obj.SetCoefficient(pv_v[nm], -agent_info[nm]['offer'])
            obj.SetCoefficient(wind_v[nm], -agent_info[nm]['offer'])
            obj.SetCoefficient(dis_v[nm], -agent_info[nm]['offer'])
        obj.SetCoefficient(unserved[nm], -config.penalty_unserved)

    grid_import_coef = -wholesale_t
    if config.enable_multi_objective and config.lambda_carbon > 0:
        from dispatch_core import DT_HOURS
        grid_import_coef -= config.lambda_carbon * config.emission_factor_grid * DT_HOURS
    obj.SetCoefficient(p_grid_import, grid_import_coef)
    obj.SetCoefficient(p_grid_export, wholesale_t)
    obj.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return False, None, 0.0, {}, 0.0

    lmp = np.zeros(len(buses))
    for i, b in enumerate(buses):
        try:
            lmp[i] = pbal[b].DualValue()
        except AttributeError:
            lmp[i] = wholesale_t

    agent_res = {}
    total_welfare = 0.0
    for a in agents:
        nm = a.name
        info = agent_info[nm]
        pv_val = pv_v[nm].SolutionValue()
        w_val = wind_v[nm].SolutionValue()
        s_val = served[nm].SolutionValue()
        ch_val = ch_v[nm].SolutionValue() if nm in ch_v else 0.0
        dis_val = dis_v[nm].SolutionValue() if nm in dis_v else 0.0
        bus = info['bus']
        bus_idx = buses.index(bus) if bus in buses else bus
        node_lmp = lmp[bus_idx] if bus_idx < len(lmp) else wholesale_t
        revenue = s_val * info['bid'] - (pv_val + w_val + dis_val) * info['offer']
        total_welfare += revenue
        agent_res[nm] = {
            'served': s_val, 'pv_used': pv_val, 'wind_used': w_val,
            'p_ch': ch_val, 'p_dis': dis_val,
        }

    return True, lmp, total_welfare, agent_res, p_grid_import.SolutionValue() - p_grid_export.SolutionValue()
