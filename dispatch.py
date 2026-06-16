# dispatch.py
"""Storage constraints + Gurobi OPF: DC / LinDistFlow dual-mode + multi-period joint optimization."""

# Pre-load ortools to work around DLL load order issue in pandapower
from ortools.linear_solver import pywraplp  # noqa: F401

from collections import deque

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False
    gp = None  # type: ignore
    GRB = None  # type: ignore

from models import MarketConfig, Agent

# ===========================================================================
# Storage constraint engine
# ===========================================================================
class StorageConstraints:
    @staticmethod
    def feasible_ranges(storage, soc, dt=0.25):
        p_ch_max = storage.max_charge_feasible(soc, dt)
        p_dis_max = storage.max_discharge_feasible(soc, dt)
        p_ch_min = storage.p_ch_min if storage.p_ch_min < p_ch_max else 0.0
        p_dis_min = storage.p_dis_min if storage.p_dis_min < p_dis_max else 0.0
        return p_ch_min, p_ch_max, p_dis_min, p_dis_max

    @staticmethod
    def apply_ramp(prev_p_ch, prev_p_dis, target_p_ch, target_p_dis, storage):
        prev_signed = prev_p_dis - prev_p_ch
        target_signed = target_p_dis - target_p_ch
        ramp_limit = storage.ramp_up_dis if storage.ramp_up_dis else float('inf')
        delta = target_signed - prev_signed
        if abs(delta) > ramp_limit:
            feasible_signed = prev_signed + np.sign(delta) * ramp_limit
        else:
            feasible_signed = target_signed
        if feasible_signed < -1e-6:
            return min(abs(feasible_signed), storage.p_ch_max), 0.0
        elif feasible_signed > 1e-6:
            return 0.0, min(feasible_signed, storage.p_dis_max)
        else:
            return 0.0, 0.0

    @staticmethod
    def execute_dispatch(storage, soc, p_ch, p_dis, dt=0.25):
        p_ch = max(0.0, min(p_ch, storage.p_ch_max))
        p_dis = max(0.0, min(p_dis, storage.p_dis_max))
        p_ch_min, p_ch_max, p_dis_min, p_dis_max = StorageConstraints.feasible_ranges(storage, soc, dt)
        if p_ch > p_ch_max: p_ch = p_ch_max
        if 0 < p_ch < p_ch_min: p_ch = 0.0
        if p_dis > p_dis_max: p_dis = p_dis_max
        if 0 < p_dis < p_dis_min: p_dis = 0.0
        energy_in = p_ch * storage.eta_ch * dt if p_ch > 0 else 0.0
        energy_out = p_dis / storage.eta_dis * dt if p_dis > 0 else 0.0
        soc_change = (energy_in - energy_out) / storage.e_max
        new_soc = soc + soc_change
        new_soc = max(storage.soc_min, min(storage.soc_max, new_soc))
        return p_ch, p_dis, new_soc


# ===========================================================================
# Shared constants
# ===========================================================================

DT_HOURS = 0.25  # 15-min period in hours
STORAGE_MODE_THRESHOLD = 1e-6  # below this, ch/dis is treated as zero


# ===========================================================================
# Shared OPF helpers
# ===========================================================================


def _build_line_params(net, base_kv):
    """Compute line reactance (pu) and thermal limits from network data.

    Returns (x, limit) dicts keyed by line index.
    For DC-OPF only: reactance x[l] = X_ohm / V2.
    """
    V2 = base_kv ** 2
    x = {}
    limit = {}
    for l in net.line.index:
        length = net.line.at[l, 'length_km']
        X_ohm = net.line.at[l, 'x_ohm_per_km'] * length
        x[l] = X_ohm / V2
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka
    return x, limit


def _build_line_params_full(net, base_kv):
    """Compute line r, x, limit dicts for LinDistFlow (includes resistance).

    Returns (r, x, limit) dicts keyed by line index. All values in per-unit.
    """
    V2 = base_kv ** 2
    r = {}
    x = {}
    limit = {}
    for l in net.line.index:
        length = net.line.at[l, 'length_km']
        r[l] = net.line.at[l, 'r_ohm_per_km'] * length / V2
        x[l] = net.line.at[l, 'x_ohm_per_km'] * length / V2
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka
    return r, x, limit


def _classify_storage_mode(ch_val, dis_val):
    """Return 'charge', 'discharge', or 'idle' from solved power values."""
    if dis_val > STORAGE_MODE_THRESHOLD:
        return 'discharge'
    elif ch_val > STORAGE_MODE_THRESHOLD:
        return 'charge'
    return 'idle'


def _build_agent_info(agents, t, stage, prev_soc, wholesale_t, action_params, config):
    """Build per-agent info dict for single-period solvers.

    Returns agent_info dict, with storage vars set up for MILP binary co-optimization
    (both ch_max and dis_max available; binary prevents simultaneous charge/discharge).
    """
    agent_info = {}
    for a in agents:
        load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
        pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
        wind_max = (a.get_wind_forecast()[t] if stage == "DA" else a.get_wind_real()[t]) if a.has_wind else 0.0

        target_ch, target_dis = 0.0, 0.0
        if a.storage is not None:
            cur_soc = prev_soc.get(a.name, a.storage.soc0)
            _, p_ch_max, _, p_dis_max = StorageConstraints.feasible_ranges(a.storage, cur_soc)
            target_ch, target_dis = float(p_ch_max), float(p_dis_max)

        bid = a.bid_value
        offer = a.offer_cost
        ap = action_params.get(a.name, {})
        if ap:
            bid *= ap["bid_mult"][t] if isinstance(ap.get("bid_mult"), np.ndarray) else ap.get("bid_mult", 1.0)
            offer += ap["offer_adder"][t] if isinstance(ap.get("offer_adder"), np.ndarray) else ap.get("offer_adder", 0.0)

        agent_info[a.name] = {
            'bus': a.bus, 'load': load_val, 'pv_max': pv_max, 'wind_max': wind_max,
            'ch_max': target_ch, 'dis_max': target_dis,
            'bid': bid, 'offer': offer, 'has_storage': a.storage is not None,
            'storage_mode': 'idle',
        }
    return agent_info


# ===========================================================================
# DC-OPF solver
# ===========================================================================
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
    p_grid_export = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_export")

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


# ===========================================================================
# LinDistFlow (single-period, for fallback or RT rolling)
# ===========================================================================
def solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                             action_params, config):
    if not _HAS_GUROBI:
        raise RuntimeError("LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")
    buses = list(net.bus.index)
    n_buses = len(buses)
    lines = list(net.line.index)[:n_buses - 1]  # radial backbone only
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv

    # Filter to radial lines only
    r, x_val, limit = _build_line_params_full(net, base_kv)
    r = {l: v for l, v in r.items() if l in lines}
    x_val = {l: v for l, v in x_val.items() if l in lines}
    limit = {l: v for l, v in limit.items() if l in lines}

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config)

    m = gp.Model("LinDistFlow")
    m.setParam('OutputFlag', 0)

    V = m.addVars(buses, lb=config.v_min_pu, ub=config.v_max_pu, name="V")
    P = m.addVars(lines, lb=-GRB.INFINITY, name="P")
    Q = m.addVars(lines, lb=-GRB.INFINITY, name="Q")
    q_grid = m.addVar(lb=-GRB.INFINITY, name="q_grid")
    p_loss = m.addVars(buses, lb=0, ub=GRB.INFINITY, name="p_loss")
    m.addConstr(V[slack_bus] == 1.0, "ref_voltage")

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
    p_grid_export = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_export")

    # Reactive power from DER inverters (single-period)
    q_re = {}
    if config.reactive_support:
        for a in agents:
            nm = a.name
            if a.is_prosumer or a.storage is not None:
                q_re[nm] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                    name=f"q_re_{nm}")

    pf = config.load_power_factor
    q_ratio = np.tan(np.arccos(pf))

    # Active power balance (with losses)
    inj_p = {b: gp.LinExpr() for b in buses}
    inj_p[slack_bus] += p_grid_import - p_grid_export
    for a in agents:
        nm = a.name
        bus = agent_info[nm]['bus']
        inj_p[bus] += pv[nm] + wind[nm] + dis[nm] - served[nm] - ch[nm]
    for b in buses:
        flow_out = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                flow_out += P[l]
            elif net.line.at[l, 'to_bus'] == b:
                flow_out -= P[l]
        m.addConstr(inj_p[b] - p_loss[b] == flow_out, f"p_balance_{b}")
        p_loss[b].UB = 0  # fixed to zero on first pass

    # Reactive power balance
    q_inj = {b: gp.LinExpr() for b in buses}
    q_inj[slack_bus] += q_grid
    for a in agents:
        nm = a.name
        bus = agent_info[nm]['bus']
        q_inj[bus] -= served[nm] * q_ratio
        if config.reactive_support and nm in q_re:
            q_inj[bus] += q_re[nm]
    for b in buses:
        q_flow = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                q_flow += Q[l]
            elif net.line.at[l, 'to_bus'] == b:
                q_flow -= Q[l]
        m.addConstr(q_inj[b] == q_flow, f"q_balance_{b}")

    # Voltage drop (P-Q coupled)
    for l in lines:
        f = net.line.at[l, 'from_bus']
        to_bus = net.line.at[l, 'to_bus']
        m.addConstr(V[f] - V[to_bus] == r[l] * P[l] + x_val[l] * Q[l], f"voltage_drop_{l}")

    # Line limits (P and Q)
    for l in lines:
        m.addConstr(P[l] <= limit[l], f"limitP_pos_{l}")
        m.addConstr(P[l] >= -limit[l], f"limitP_neg_{l}")
        m.addConstr(Q[l] <= limit[l], f"limitQ_pos_{l}")
        m.addConstr(Q[l] >= -limit[l], f"limitQ_neg_{l}")

    # Reactive power bounds from DER inverters
    # s_max uses INSTALLED capacity, not forecast, because inverters can
    # supply reactive power even when the resource is not generating.
    if config.reactive_support:
        for a in agents:
            nm = a.name
            if nm not in q_re:
                continue
            info = agent_info[nm]
            s_max = a.pv_capacity + a.wind_capacity
            if info['has_storage']:
                s_max += info['dis_max']
            if s_max > 0:
                q_re[nm].LB = -s_max
                q_re[nm].UB = s_max
                p_total = pv[nm] + wind[nm] + dis[nm]
                m.addConstr(q_re[nm] + p_total <= s_max, f"diamP_{nm}")
                m.addConstr(-q_re[nm] + p_total <= s_max, f"diamN_{nm}")

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
    m.setObjective(obj, GRB.MAXIMIZE)

    # I²R loss iteration
    for loss_iter in range(max(1, config.n_loss_iters)):
        m.optimize()

        if m.status != GRB.OPTIMAL:
            return False, None, 0.0, {}, 0.0

        if loss_iter < config.n_loss_iters - 1:
            for b in buses:
                p_loss[b].UB = 0
                p_loss[b].LB = 0
            for l in lines:
                p_val = P[l].X
                q_val = Q[l].X
                loss_val = r[l] * (p_val ** 2 + q_val ** 2)
                to_bus = net.line.at[l, 'to_bus']
                p_loss[to_bus].UB += loss_val
                p_loss[to_bus].LB += loss_val

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


# ===========================================================================
# Unified entry point (single-period)
# ===========================================================================
def solve_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                     action_params, config):
    if not _HAS_GUROBI:
        if config.opf_mode == "dc":
            return _solve_dc_opf_highs(net, agents, t, stage, prev_soc,
                                       wholesale_t, action_params, config)
        raise RuntimeError("Gurobi unavailable and no HiGHS fallback for LinDistFlow")
    if config.opf_mode == "dc":
        return solve_dc_opf_gurobi(net, agents, t, stage, prev_soc,
                                   wholesale_t, action_params, config)
    elif config.opf_mode == "lindistflow":
        return solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc,
                                        wholesale_t, action_params, config)
    else:
        raise ValueError(f"Unknown OPF mode: {config.opf_mode}")


# ===========================================================================
# HiGHS fallback solver (ortools, for when Gurobi is unavailable)
# ===========================================================================
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
    p_grid_export = solver.NumVar(0, INF, "p_grid_export")

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

    obj.SetCoefficient(p_grid_import, -wholesale_t)
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


# ===========================================================================
# Multi-period joint optimization (batch LinDistFlow)
# ===========================================================================
def solve_lindist_opf_batch(net, agents, T, stage, config, action_params, wholesale,
                            storage_units=None):
    """
    Multi-period joint LinDistFlow with storage SOC transition and terminal value.
    Returns a result dict matching the clear_market schema.
    """
    if not _HAS_GUROBI:
        raise RuntimeError("Batch LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")
    n_buses = len(net.bus.index)
    lmp = np.zeros((T, n_buses))
    buses = list(net.bus.index)
    lines = list(net.line.index)[:n_buses - 1]  # radial backbone only
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv

    # Line parameters (filter to radial lines)
    r, x_val, limit = _build_line_params_full(net, base_kv)
    r = {l: v for l, v in r.items() if l in lines}
    x_val = {l: v for l, v in x_val.items() if l in lines}
    limit = {l: v for l, v in limit.items() if l in lines}

    m = gp.Model("LinDistFlow_batch_T")
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 2)

    # Network variables: voltage, line flows, grid power
    V   = m.addVars(T, buses, lb=config.v_min_pu, ub=config.v_max_pu, name="V")
    P   = m.addVars(T, lines, lb=-GRB.INFINITY, name="P")
    Q   = m.addVars(T, lines, lb=-GRB.INFINITY, name="Q")
    p_grid_import = m.addVars(T, lb=0, ub=GRB.INFINITY, name="p_grid_import")
    p_grid_export = m.addVars(T, lb=0, ub=GRB.INFINITY, name="p_grid_export")
    q_grid = m.addVars(T, lb=-GRB.INFINITY, name="q_grid")
    p_loss = m.addVars(T, buses, lb=0, ub=GRB.INFINITY, name="p_loss")

    # Agent variables
    served   = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    q_re     = {}  # reactive power injection from DER inverters
    soc      = {}
    agents_list = list(agents)
    storage_units = storage_units or []
    storage_agents = [a for a in agents_list if a.storage is not None]

    for a in agents_list:
        nm = a.name
        served[nm]   = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"served_{nm}")
        unserved[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"unserv_{nm}")
        pv[nm]       = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"pv_{nm}")
        wind[nm]     = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"wind_{nm}")
        ch[nm]       = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
        dis[nm]      = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"dis_{nm}")

    if config.reactive_support:
        for a in agents_list:
            nm = a.name
            if a.is_prosumer or a.storage is not None:
                q_re[nm] = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"q_re_{nm}")

    for a in storage_agents:
        nm = a.name
        soc[nm] = m.addVars(T+1, lb=a.storage.soc_min, ub=a.storage.soc_max,
                                name=f"soc_{nm}")
        m.addConstr(soc[nm][0] == a.storage.soc0, f"init_soc_{nm}")

    for su in storage_units:
        nm = su.name
        ch[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
        dis[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"dis_{nm}")
        soc[nm] = m.addVars(T+1, lb=su.storage.soc_min, ub=su.storage.soc_max,
                            name=f"soc_{nm}")
        m.addConstr(soc[nm][0] == su.storage.soc0, f"init_soc_{nm}")

    # Non-storage agents must have ch=dis=0 — no phantom generation
    stor_names = {a.name for a in storage_agents} | {su.name for su in storage_units}
    for a in agents_list:
        if a.name not in stor_names:
            for t in range(T):
                ch[a.name][t].UB = 0
                dis[a.name][t].UB = 0

    pf = config.load_power_factor
    q_ratio = np.tan(np.arccos(pf))

    # Per-period constraints
    for t in range(T):
        m.addConstr(V[t, slack_bus] == 1.0, f"ref_voltage_{t}")

        for b in buses:
            inj = gp.LinExpr()
            if b == slack_bus:
                inj += p_grid_import[t] - p_grid_export[t]
            for a in agents_list:
                if a.bus == b:
                    nm = a.name
                    inj += pv[nm][t] + wind[nm][t] + dis[nm][t] - served[nm][t] - ch[nm][t]
            for su in storage_units:
                if su.bus == b:
                    nm = su.name
                    inj += dis[nm][t] - ch[nm][t]
            flow = gp.LinExpr()
            for l in lines:
                if net.line.at[l, 'from_bus'] == b:
                    flow += P[t, l]
                elif net.line.at[l, 'to_bus'] == b:
                    flow -= P[t, l]
            m.addConstr(inj - p_loss[t, b] == flow, f"p_bal_{t}_{b}")
            p_loss[t, b].UB = 0  # fixed to zero on first pass

        for b in buses:
            q_inj = gp.LinExpr()
            if b == slack_bus:
                q_inj += q_grid[t]
            for a in agents_list:
                if a.bus == b:
                    nm = a.name
                    q_inj -= served[nm][t] * q_ratio
                    if config.reactive_support and nm in q_re:
                        q_inj += q_re[nm][t]
            q_flow = gp.LinExpr()
            for l in lines:
                if net.line.at[l, 'from_bus'] == b:
                    q_flow += Q[t, l]
                elif net.line.at[l, 'to_bus'] == b:
                    q_flow -= Q[t, l]
            m.addConstr(q_inj == q_flow, f"q_bal_{t}_{b}")

        for l in lines:
            f = net.line.at[l, 'from_bus']
            to = net.line.at[l, 'to_bus']
            m.addConstr(V[t, f] - V[t, to] == r[l] * P[t, l] + x_val[l] * Q[t, l],
                        f"vdrop_{t}_{l}")

        for l in lines:
            m.addConstr(P[t, l] <= limit[l], f"limP_{t}_{l}")
            m.addConstr(P[t, l] >= -limit[l], f"limN_{t}_{l}")
            m.addConstr(Q[t, l] <= limit[l], f"limQP_{t}_{l}")
            m.addConstr(Q[t, l] >= -limit[l], f"limQN_{t}_{l}")

        for a in agents_list:
            nm = a.name
            load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
            m.addConstr(served[nm][t] + unserved[nm][t] == load_val, f"load_{t}_{nm}")

        for a in agents_list:
            nm = a.name
            pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
            wind_max = (a.get_wind_forecast()[t] if stage == "DA"
                        else a.get_wind_real()[t]) if a.has_wind else 0.0
            pv[nm][t].UB = pv_max
            wind[nm][t].UB = wind_max

        for a in storage_agents:
            nm = a.name
            stor = a.storage
            ch[nm][t].UB = stor.p_ch_max
            dis[nm][t].UB = stor.p_dis_max

            eta_ch = stor.eta_ch; eta_dis = stor.eta_dis
            e_max = stor.e_max
            soc_t = soc[nm][t]
            soc_next = soc[nm][t+1]
            m.addConstr(
                soc_next == soc_t + (eta_ch * ch[nm][t] - dis[nm][t] / eta_dis) * DT_HOURS / e_max,
                f"soctrans_{nm}_{t}"
            )

            # Ramp constraints (inter-period, t >= 1)
            if t > 0:
                if stor.ramp_up_ch is not None:
                    m.addConstr(ch[nm][t] - ch[nm][t-1] <= stor.ramp_up_ch,
                                f"ramp_up_ch_{nm}_{t}")
                if stor.ramp_down_ch is not None:
                    m.addConstr(ch[nm][t-1] - ch[nm][t] <= stor.ramp_down_ch,
                                f"ramp_down_ch_{nm}_{t}")
                if stor.ramp_up_dis is not None:
                    m.addConstr(dis[nm][t] - dis[nm][t-1] <= stor.ramp_up_dis,
                                f"ramp_up_dis_{nm}_{t}")
                if stor.ramp_down_dis is not None:
                    m.addConstr(dis[nm][t-1] - dis[nm][t] <= stor.ramp_down_dis,
                                f"ramp_down_dis_{nm}_{t}")

        # Reactive power support from DER inverters (PV + wind + storage)
        # Diamond constraint: |q_re| + p_total <= s_max (linear, tighter than box)
        # Reactive power bounds from DER inverters — use INSTALLED capacity,
        # not forecast, since inverters can supply Q even at zero P output.
        if config.reactive_support:
            for a in agents_list:
                nm = a.name
                if nm not in q_re:
                    continue
                s_max = a.pv_capacity + a.wind_capacity
                if a.storage:
                    s_max += a.storage.p_dis_max
                q_re[nm][t].LB = -s_max
                q_re[nm][t].UB = s_max
                p_total = pv[nm][t] + wind[nm][t] + dis[nm][t]
                # |q| + p <= s_max: when more active power is dispatched,
                # less inverter capacity remains for reactive power
                m.addConstr(q_re[nm][t] + p_total <= s_max,
                            f"diamP_{nm}_{t}")
                m.addConstr(-q_re[nm][t] + p_total <= s_max,
                            f"diamN_{nm}_{t}")

        # SOC transition + ramp for standalone storage units
        for su in storage_units:
            nm = su.name
            stor = su.storage
            ch[nm][t].UB = stor.p_ch_max
            dis[nm][t].UB = stor.p_dis_max

            eta_ch = stor.eta_ch; eta_dis = stor.eta_dis
            e_max = stor.e_max
            soc_t = soc[nm][t]
            soc_next = soc[nm][t+1]
            m.addConstr(
                soc_next == soc_t + (eta_ch * ch[nm][t] - dis[nm][t] / eta_dis) * DT_HOURS / e_max,
                f"soctrans_{nm}_{t}"
            )

            if t > 0:
                if stor.ramp_up_ch is not None:
                    m.addConstr(ch[nm][t] - ch[nm][t-1] <= stor.ramp_up_ch,
                                f"ramp_up_ch_{nm}_{t}")
                if stor.ramp_down_ch is not None:
                    m.addConstr(ch[nm][t-1] - ch[nm][t] <= stor.ramp_down_ch,
                                f"ramp_down_ch_{nm}_{t}")
                if stor.ramp_up_dis is not None:
                    m.addConstr(dis[nm][t] - dis[nm][t-1] <= stor.ramp_up_dis,
                                f"ramp_up_dis_{nm}_{t}")
                if stor.ramp_down_dis is not None:
                    m.addConstr(dis[nm][t-1] - dis[nm][t] <= stor.ramp_down_dis,
                                f"ramp_down_dis_{nm}_{t}")

    # Objective — precompute per-agent bid/offer arrays
    bid_arr = {}
    offer_arr = {}
    for a in agents_list:
        nm = a.name
        ap = action_params.get(a.name, {})
        bm = ap.get("bid_mult", 1.0)
        oa = ap.get("offer_adder", 0.0)
        if isinstance(bm, np.ndarray):
            bid_arr[nm] = a.bid_value * bm
        else:
            bid_arr[nm] = np.full(T, a.bid_value * bm)
        if isinstance(oa, np.ndarray):
            offer_arr[nm] = a.offer_cost + oa
        else:
            offer_arr[nm] = np.full(T, a.offer_cost + oa)

    obj = gp.LinExpr()
    if config.storage_terminal_value is not None:
        terminal_value = config.storage_terminal_value
    else:
        terminal_value = float(np.mean(wholesale[-8:]))  # last 2h avg: end-of-day valuation
    for t in range(T):
        for a in agents_list:
            nm = a.name
            bid = bid_arr[nm][t]
            offer = offer_arr[nm][t]

            obj += bid * served[nm][t]
            if a.storage is not None:
                obj += wholesale[t] * (dis[nm][t] - ch[nm][t])
                if config.lambda_cycle > 0:
                    obj -= config.lambda_cycle * (ch[nm][t] + dis[nm][t])
            else:
                obj -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
            obj -= config.penalty_unserved * unserved[nm][t]
        obj -= wholesale[t] * (p_grid_import[t] - p_grid_export[t])
    for a in storage_agents:
        obj += terminal_value * soc[a.name][T] * a.storage.e_max

    # -- standalone storage units: objective + terminal SOC
    su_bid = {}; su_offer = {}
    for su in storage_units:
        nm = su.name
        su_bid[nm] = np.full(T, su.bid_value)
        su_offer[nm] = np.full(T, su.offer_cost)
        for t in range(T):
            obj += su_bid[nm][t] * ch[nm][t]
            obj -= su_offer[nm][t] * dis[nm][t]
            if config.lambda_cycle > 0:
                obj -= config.lambda_cycle * (ch[nm][t] + dis[nm][t])
        obj += terminal_value * soc[nm][T] * su.storage.e_max

    m.setObjective(obj, GRB.MAXIMIZE)

    for loss_iter in range(max(1, config.n_loss_iters)):
        m.optimize()

        if m.status != GRB.OPTIMAL:
            print("Batch model solve failed, status:", m.status)
            return None

        if loss_iter < config.n_loss_iters - 1:
            for t in range(T):
                for b in buses:
                    p_loss[t, b].UB = 0
                    p_loss[t, b].LB = 0
                for l in lines:
                    p_val = P[t, l].X
                    q_val = Q[t, l].X
                    loss_val = r[l] * (p_val ** 2 + q_val ** 2)
                    to_bus = net.line.at[l, 'to_bus']
                    p_loss[t, to_bus].UB += loss_val
                    p_loss[t, to_bus].LB += loss_val

    all_storage = storage_agents + storage_units
    if all_storage and config.use_nodal_storage_price:
        bus_to_idx = {b: i for i, b in enumerate(buses)}

        # Extract first-pass nodal LMPs from power balance constraint duals
        nodal_lmp = np.zeros((T, n_buses))
        for t in range(T):
            for i, b in enumerate(buses):
                nodal_lmp[t, i] = -m.getConstrByName(f"p_bal_{t}_{b}").Pi

        # Iterative re-solve with nodal LMP until convergence
        max_iters = 4
        converged = False
        for nodal_iter in range(max_iters):
            obj2 = gp.LinExpr()
            for t in range(T):
                for a in agents_list:
                    nm = a.name
                    bid = bid_arr[nm][t]
                    offer = offer_arr[nm][t]
                    obj2 += bid * served[nm][t]
                    if a.storage is not None:
                        price_t = nodal_lmp[t, bus_to_idx[a.bus]]
                        obj2 += price_t * (dis[nm][t] - ch[nm][t])
                        if config.lambda_cycle > 0:
                            obj2 -= config.lambda_cycle * (ch[nm][t] + dis[nm][t])
                    else:
                        obj2 -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
                    obj2 -= config.penalty_unserved * unserved[nm][t]
                obj2 -= wholesale[t] * (p_grid_import[t] - p_grid_export[t])
            for a in storage_agents:
                obj2 += terminal_value * soc[a.name][T] * a.storage.e_max
            for su in storage_units:
                nm = su.name
                for t in range(T):
                    obj2 += su_bid[nm][t] * ch[nm][t]
                    obj2 -= su_offer[nm][t] * dis[nm][t]
                    if config.lambda_cycle > 0:
                        obj2 -= config.lambda_cycle * (ch[nm][t] + dis[nm][t])
                obj2 += terminal_value * soc[nm][T] * su.storage.e_max

            m.setObjective(obj2, GRB.MAXIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                break

            # Extract new LMPs and check convergence
            new_lmp = np.zeros((T, n_buses))
            for t in range(T):
                for i, b in enumerate(buses):
                    new_lmp[t, i] = -m.getConstrByName(f"p_bal_{t}_{b}").Pi

            max_change = np.max(np.abs(new_lmp - nodal_lmp))
            nodal_lmp = new_lmp
            if max_change < 1.0:  # 1 CNY/MWh tolerance
                converged = True
                break

    # Extract results
    from market import empty_schedules as _empty_schedules, split_power as _split_power
    schedules = _empty_schedules(agents_list, T)
    for su in storage_units:
        schedules[su.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'storage_mode': ['idle'] * T,
        }
    total_welfare = m.ObjVal
    total_re_avail = sum(
        (np.sum(a.pv_forecast if stage == "DA" else a.pv_real) +
         (np.sum(a.wind_forecast if stage == "DA" else a.wind_real) if a.has_wind else 0))
        * DT_HOURS
        for a in agents_list
    )
    total_re_used = 0.0
    total_curtailment = 0.0
    carbon_emissions = 0.0
    total_served_mwh = 0.0

    # Build radial path tree from slack bus for marginal loss factor computation
    parent_line = {}  # bus → (parent_bus, line_id)
    adj = {b: [] for b in buses}
    for l in lines:
        f = int(net.line.at[l, 'from_bus'])
        to = int(net.line.at[l, 'to_bus'])
        adj[f].append((to, l))
        adj[to].append((f, l))
    visited = {slack_bus}
    queue = deque([slack_bus])
    while queue:
        b = queue.popleft()
        for nb, l in adj[b]:
            if nb not in visited:
                visited.add(nb)
                parent_line[nb] = (b, l)
                queue.append(nb)

    for t in range(T):
        for i, b in enumerate(buses):
            constr = m.getConstrByName(f"p_bal_{t}_{b}")
            if constr is not None:
                lmp[t, i] = -constr.Pi
            else:
                lmp[t, i] = wholesale[t]

        # Apply marginal loss factors along radial path
        # Only correct uncongested buses — congested LMPs already capture
        # spatial price signals via binding line constraints.
        slack_idx = list(buses).index(slack_bus)
        base_lmp = lmp[t, slack_idx]
        for i, b in enumerate(buses):
            if b == slack_bus:
                continue
            # Skip congested buses: original dual already reflects true LMP
            if abs(lmp[t, i] - base_lmp) > 1.0:
                continue
            mlf = 0.0
            cur = b
            while cur != slack_bus:
                parent_b, line_l = parent_line[cur]
                p_flow = abs(P[t, line_l].X)
                mlf += 2.0 * r[line_l] * p_flow
                cur = parent_b
            lmp[t, i] = base_lmp * (1.0 + mlf)

        pgi = p_grid_import[t].X
        if pgi > 0:
            carbon_emissions += config.emission_factor_grid * pgi * DT_HOURS

        schedules['GRID']['g_grid'][t] = p_grid_import[t].X - p_grid_export[t].X

        for a in agents_list:
            nm = a.name
            s = schedules[nm]
            s['served'][t]    = served[nm][t].X
            s['unserved'][t]  = unserved[nm][t].X
            s['pv_used'][t]   = pv[nm][t].X
            s['wind_used'][t] = wind[nm][t].X
            s['p_ch'][t]      = ch[nm][t].X
            s['p_dis'][t]     = dis[nm][t].X
            s['q_re'][t]      = q_re[nm][t].X if config.reactive_support and nm in q_re else 0.0
            s['storage_mode'][t] = _classify_storage_mode(ch[nm][t].X, dis[nm][t].X)

            if a.storage:
                s['soc'][t]   = soc[nm][t].X
                if t == T-1:
                    s['soc_final'] = soc[nm][T].X
            else:
                s['soc'][t] = 0.0

            load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
            pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
            wind_max = (a.get_wind_forecast()[t] if stage == "DA" else a.get_wind_real()[t]) if a.has_wind else 0.0
            net_gen = s['pv_used'][t] + s['wind_used'][t] + s['p_dis'][t]
            net_con = s['served'][t] + s['p_ch'][t]
            s['p_buy'][t], s['p_sell'][t] = _split_power(net_gen, net_con)
            total_re_used += (s['pv_used'][t] + s['wind_used'][t]) * DT_HOURS
            total_curtailment += ((pv_max - s['pv_used'][t]) + (wind_max - s['wind_used'][t])) * DT_HOURS
            total_served_mwh += s['served'][t] * DT_HOURS

        for su in storage_units:
            nm = su.name
            s = schedules[nm]
            s['p_ch'][t]  = ch[nm][t].X
            s['p_dis'][t] = dis[nm][t].X
            s['soc'][t]   = soc[nm][t].X
            s['storage_mode'][t] = _classify_storage_mode(ch[nm][t].X, dis[nm][t].X)
            if t == T-1:
                s['soc_final'] = soc[nm][T].X
            s['p_buy'][t], s['p_sell'][t] = _split_power(s['p_dis'][t], s['p_ch'][t])
            total_served_mwh += s['served'][t] * DT_HOURS

    re_rate = (total_re_used / total_re_avail * 100) if total_re_avail > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    result = {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_avail,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
    }
    return result