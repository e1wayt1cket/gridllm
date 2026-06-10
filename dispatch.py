# dispatch.py
"""储能约束 + Gurobi OPF：DC / LinDistFlow 双模式 + 多时段联合优化"""

# Pre-load ortools to work around DLL load order issue in pandapower
from ortools.linear_solver import pywraplp  # noqa: F401

import numpy as np
import pandapower as pp
from typing import Dict, Tuple, List
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
# 储能约束引擎
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
        self_loss = soc * storage.self_discharge_rate * dt
        soc_change = (energy_in - energy_out) / storage.e_max
        new_soc = soc + soc_change - self_loss
        new_soc = max(storage.soc_min, min(storage.soc_max, new_soc))
        return p_ch, p_dis, new_soc


# ===========================================================================
# 储能模式决策（仅用于单时段回退，批量模型中由优化器自行决定）
# ===========================================================================
def determine_storage_mode(agent, wholesale_t, prev_soc, config,
                           action_params=None, prev_p_ch=0.0, prev_p_dis=0.0):
    """
    根据电价和SOC决定储能模式 (charge/discharge/idle)
    返回: (mode: str, target_ch: float, target_dis: float)
    """
    storage = agent.storage
    if storage is None:
        return "idle", 0.0, 0.0
    dt = 0.25
    p_ch_min, p_ch_max, p_dis_min, p_dis_max = StorageConstraints.feasible_ranges(storage, prev_soc, dt)
    if p_ch_max <= 0 and p_dis_max <= 0:
        return "idle", 0.0, 0.0

    ap = action_params or {}
    bid_mult = ap.get("bid_mult", 1.0)
    offer_adder = ap.get("offer_adder", 0.0)

    buf = config.storage_soc_buffer

    if prev_soc <= storage.soc_min + buf:
        if np.any(wholesale_t < agent.bid_value * bid_mult * 0.9) and p_ch_max > 0:
            mode, t_ch, t_dis = "charge", p_ch_max, 0.0
        else:
            return "idle", 0.0, 0.0
    elif prev_soc >= storage.soc_max - buf:
        if np.any(wholesale_t > agent.offer_cost + offer_adder) and p_dis_max > 0:
            mode, t_ch, t_dis = "discharge", 0.0, p_dis_max
        else:
            return "idle", 0.0, 0.0
    else:
        round_trip_eff = storage.eta_ch * storage.eta_dis
        charge_threshold = agent.bid_value * bid_mult * round_trip_eff * config.storage_charge_discount
        discharge_threshold = (agent.offer_cost + offer_adder) / round_trip_eff * config.storage_discharge_premium
        soc_range = storage.soc_max - storage.soc_min
        soc_ratio = (prev_soc - storage.soc_min) / soc_range if soc_range > 0 else 0.5
        charge_threshold *= (0.7 + 0.6 * soc_ratio)
        discharge_threshold *= (1.3 - 0.6 * soc_ratio)
        if np.any(wholesale_t < charge_threshold) and p_ch_max > 0:
            mode, t_ch, t_dis = "charge", p_ch_max, 0.0
        elif np.any(wholesale_t > discharge_threshold) and p_dis_max > 0:
            mode, t_ch, t_dis = "discharge", 0.0, p_dis_max
        else:
            return "idle", 0.0, 0.0

    t_ch, t_dis = StorageConstraints.apply_ramp(prev_p_ch, prev_p_dis, t_ch, t_dis, storage)
    if t_ch <= 1e-6 and t_dis <= 1e-6:
        mode = "idle"
    return mode, t_ch, t_dis


# ===========================================================================
# Shared OPF helpers
# ===========================================================================
def _add_multi_objective_terms(obj, config, agents, pv_vars, wind_vars,
                                pv_max_dict, wind_max_dict, p_grid_import):
    """Add multi-objective terms to a Gurobi objective expression.

    Used by all three OPF solvers (DC single, LinDistFlow single, LinDistFlow batch).
    """
    if not config.enable_multi_objective:
        if config.lambda_re > 0:
            for a in agents:
                obj += config.lambda_re * (pv_vars[a.name] + wind_vars.get(a.name, 0))
        return

    if config.lambda_re > 0:
        for a in agents:
            obj += config.lambda_re * (pv_vars[a.name] + wind_vars.get(a.name, 0))
    if config.lambda_curtail > 0:
        for a in agents:
            nm = a.name
            pv_curtail = pv_max_dict[nm] - pv_vars[nm]
            wind_curtail = wind_max_dict.get(nm, 0.0) - wind_vars.get(nm, 0)
            obj -= config.lambda_curtail * (pv_curtail + wind_curtail)
    if config.lambda_carbon > 0:
        obj -= config.lambda_carbon * config.emission_factor_grid * p_grid_import


def _add_multi_objective_constraints_batch(m, config, agents, pv, wind, p_grid_import,
                                             T, stage, dt=0.25):
    """Add global hard constraints for carbon and RE (batch solver only).

    Returns a dict with constraint objects for shadow price extraction,
    or None if constraint mode is not active.
    """
    if not (config.enable_multi_objective and config.use_constraint_multi_obj):
        return None

    constrs = {}
    if config.carbon_cap_tco2 is not None:
        carbon_expr = gp.quicksum(
            config.emission_factor_grid * p_grid_import[t] * dt for t in range(T)
        )
        constrs['carbon_cap'] = m.addConstr(carbon_expr <= config.carbon_cap_tco2, "carbon_cap")

    if config.re_min_rate is not None:
        re_used = gp.LinExpr()
        re_avail = 0.0
        for t in range(T):
            for a in agents:
                nm = a.name
                re_used += (pv[nm][t] + wind[nm][t]) * dt
                pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
                wind_max = (a.get_wind_forecast()[t] if stage == "DA"
                            else a.get_wind_real()[t]) if a.has_wind else 0.0
                re_avail += (pv_max + wind_max) * dt
        constrs['re_min_rate'] = m.addConstr(
            re_used >= config.re_min_rate / 100.0 * re_avail, "re_min_rate"
        )
    return constrs


def _add_multi_objective_constraints_single(m, config, agents, pv_vars, wind_vars,
                                              pv_max_dict, wind_max_dict,
                                              p_grid_import, dt=0.25):
    """Add per-period hard constraints for single-period solvers.

    Global caps are divided evenly across T=96 periods as an approximation.
    """
    if not (config.enable_multi_objective and config.use_constraint_multi_obj):
        return None

    constrs = {}
    T_default = 96
    if config.carbon_cap_tco2 is not None:
        per_period_cap = config.carbon_cap_tco2 / T_default
        carbon_period = config.emission_factor_grid * p_grid_import * dt
        constrs['carbon_cap'] = m.addConstr(carbon_period <= per_period_cap, "carbon_cap")

    if config.re_min_rate is not None:
        re_used = gp.LinExpr()
        re_avail = 0.0
        for a in agents:
            nm = a.name
            re_used += pv_vars[nm] + wind_vars.get(nm, 0)
            re_avail += pv_max_dict[nm] + wind_max_dict.get(nm, 0.0)
        if re_avail > 1e-6:
            constrs['re_min_rate'] = m.addConstr(
                re_used >= config.re_min_rate / 100.0 * re_avail, "re_min_rate"
            )
    return constrs


def _build_agent_info(agents, t, stage, prev_soc, wholesale_t, action_params, config, prev_power):
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
# DC-OPF 求解
# ===========================================================================
def solve_dc_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                        action_params, config, prev_power):
    if not _HAS_GUROBI:
        return _solve_dc_opf_highs(net, agents, t, stage, prev_soc,
                                    wholesale_t, action_params, config, prev_power)
    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    base_mva = 1.0
    Z_base = (base_kv ** 2) / base_mva
    x = {}; limit = {}
    for l in lines:
        length = net.line.at[l, 'length_km']
        x_ohm_per_km = net.line.at[l, 'x_ohm_per_km']
        X_ohm = x_ohm_per_km * length
        x[l] = X_ohm / Z_base
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka * 1e3 / 1e6

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config, prev_power)

    m = gp.Model("DC_OPF")
    m.setParam('OutputFlag', 0)
    theta = m.addVars(buses, lb=-GRB.INFINITY, name="theta")
    p_flow = m.addVars(lines, lb=-GRB.INFINITY, name="p_flow")
    m.addConstr(theta[slack_bus] == 0, "ref_angle")

    served = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    is_ch_bin = {}; is_dis_bin = {}
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
            is_ch_bin[nm] = m.addVar(vtype=GRB.BINARY, name=f"is_ch_{nm}")
            is_dis_bin[nm] = m.addVar(vtype=GRB.BINARY, name=f"is_dis_{nm}")
            m.addConstr(is_ch_bin[nm] + is_dis_bin[nm] <= 1, f"ch_dis_excl_{nm}")
            m.addConstr(ch[nm] <= info['ch_max'] * is_ch_bin[nm], f"ch_bin_{nm}")
            m.addConstr(dis[nm] <= info['dis_max'] * is_dis_bin[nm], f"dis_bin_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid = m.addVar(lb=-GRB.INFINITY, name="p_grid")
    p_grid_import = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_import")
    m.addConstr(p_grid_import >= p_grid, "grid_import_def")

    net_inj = {b: gp.LinExpr() for b in buses}
    net_inj[slack_bus] += p_grid
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
        t = net.line.at[l, 'to_bus']
        m.addConstr(p_flow[l] == (theta[f] - theta[t]) / x[l], f"dc_flow_{l}")
        m.addConstr(p_flow[l] <= limit[l], f"limit_pos_{l}")
        m.addConstr(p_flow[l] >= -limit[l], f"limit_neg_{l}")

    obj = gp.LinExpr()
    for a in agents:
        nm = a.name
        obj += agent_info[nm]['bid'] * served[nm]
        obj -= agent_info[nm]['offer'] * (pv[nm] + wind[nm] + dis[nm])
        obj -= config.penalty_unserved * unserved[nm]
    pv_max_dict = {a.name: agent_info[a.name]['pv_max'] for a in agents}
    wind_max_dict = {a.name: agent_info[a.name]['wind_max'] for a in agents}
    if not (config.enable_multi_objective and config.use_constraint_multi_obj):
        _add_multi_objective_terms(obj, config, agents, pv, wind,
                                    pv_max_dict, wind_max_dict, p_grid_import)
    obj -= wholesale_t * p_grid
    m.setObjective(obj, GRB.MAXIMIZE)

    _add_multi_objective_constraints_single(m, config, agents, pv, wind,
                                              pv_max_dict, wind_max_dict, p_grid_import)
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
        agent_res[nm] = {
            'served': served[nm].X, 'pv_used': pv[nm].X, 'wind_used': wind[nm].X,
            'p_ch': ch[nm].X, 'p_dis': dis[nm].X,
            'storage_mode': agent_info[nm]['storage_mode']
        }
    return True, lmp, m.ObjVal, agent_res, p_grid.X


# ===========================================================================
# LinDistFlow（单时段，用于回退或 RT 滚动）
# ===========================================================================
def solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                             action_params, config, prev_power):
    if not _HAS_GUROBI:
        raise RuntimeError("LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")
    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    base_mva = 1.0
    Z_base = (base_kv ** 2) / base_mva

    r = {}; x_val = {}; limit = {}
    for l in lines:
        length = net.line.at[l, 'length_km']
        r_ohm = net.line.at[l, 'r_ohm_per_km'] * length
        x_ohm = net.line.at[l, 'x_ohm_per_km'] * length
        r[l] = r_ohm / Z_base
        x_val[l] = x_ohm / Z_base
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka * 1e3 / 1e6

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config, prev_power)

    m = gp.Model("LinDistFlow")
    m.setParam('OutputFlag', 0)

    V = m.addVars(buses, lb=0.85, ub=1.15, name="V")
    P = m.addVars(lines, lb=-GRB.INFINITY, name="P")
    m.addConstr(V[slack_bus] == 1.0, "ref_voltage")

    served = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    is_ch_bin = {}; is_dis_bin = {}
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
            is_ch_bin[nm] = m.addVar(vtype=GRB.BINARY, name=f"is_ch_{nm}")
            is_dis_bin[nm] = m.addVar(vtype=GRB.BINARY, name=f"is_dis_{nm}")
            m.addConstr(is_ch_bin[nm] + is_dis_bin[nm] <= 1, f"ch_dis_excl_{nm}")
            m.addConstr(ch[nm] <= info['ch_max'] * is_ch_bin[nm], f"ch_bin_{nm}")
            m.addConstr(dis[nm] <= info['dis_max'] * is_dis_bin[nm], f"dis_bin_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid = m.addVar(lb=-GRB.INFINITY, name="p_grid")
    p_grid_import = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_import")
    m.addConstr(p_grid_import >= p_grid, "grid_import_def")

    # 有功功率平衡
    inj_p = {b: gp.LinExpr() for b in buses}
    inj_p[slack_bus] += p_grid
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
        m.addConstr(inj_p[b] == flow_out, f"p_balance_{b}")

    # 电压降（忽略无功）
    for l in lines:
        f = net.line.at[l, 'from_bus']
        t = net.line.at[l, 'to_bus']
        m.addConstr(V[f] - V[t] == r[l] * P[l], f"voltage_drop_{l}")

    # 线路容量
    for l in lines:
        m.addConstr(P[l] <= limit[l], f"limit_pos_{l}")
        m.addConstr(P[l] >= -limit[l], f"limit_neg_{l}")

    obj = gp.LinExpr()
    for a in agents:
        nm = a.name
        obj += agent_info[nm]['bid'] * served[nm]
        obj -= agent_info[nm]['offer'] * (pv[nm] + wind[nm] + dis[nm])
        obj -= config.penalty_unserved * unserved[nm]
    pv_max_dict = {a.name: agent_info[a.name]['pv_max'] for a in agents}
    wind_max_dict = {a.name: agent_info[a.name]['wind_max'] for a in agents}
    if not (config.enable_multi_objective and config.use_constraint_multi_obj):
        _add_multi_objective_terms(obj, config, agents, pv, wind,
                                    pv_max_dict, wind_max_dict, p_grid_import)
    obj -= wholesale_t * p_grid
    m.setObjective(obj, GRB.MAXIMIZE)

    _add_multi_objective_constraints_single(m, config, agents, pv, wind,
                                              pv_max_dict, wind_max_dict, p_grid_import)
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
        agent_res[nm] = {
            'served': served[nm].X, 'pv_used': pv[nm].X, 'wind_used': wind[nm].X,
            'p_ch': ch[nm].X, 'p_dis': dis[nm].X,
            'storage_mode': agent_info[nm]['storage_mode']
        }
    return True, lmp, m.ObjVal, agent_res, p_grid.X


# ===========================================================================
# 统一入口（单时段用）
# ===========================================================================
def solve_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                     action_params, config, prev_power):
    if not _HAS_GUROBI:
        if config.opf_mode == "dc":
            return _solve_dc_opf_highs(net, agents, t, stage, prev_soc,
                                       wholesale_t, action_params, config, prev_power)
        raise RuntimeError("Gurobi unavailable and no HiGHS fallback for LinDistFlow")
    if config.opf_mode == "dc":
        return solve_dc_opf_gurobi(net, agents, t, stage, prev_soc,
                                   wholesale_t, action_params, config, prev_power)
    elif config.opf_mode == "lindistflow":
        return solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc,
                                        wholesale_t, action_params, config, prev_power)
    else:
        raise ValueError(f"未知 OPF 模式: {config.opf_mode}")


# ===========================================================================
# HiGHS fallback solver (ortools, for when Gurobi is unavailable)
# ===========================================================================
def _solve_dc_opf_highs(net, agents, t, stage, prev_soc, wholesale_t,
                         action_params, config, prev_power):
    """DC-OPF using ortools HiGHS (open-source fallback)."""
    from ortools.linear_solver import pywraplp

    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    base_mva = 1.0
    Z_base = (base_kv ** 2) / base_mva
    x = {}; limit = {}
    for l in lines:
        length = net.line.at[l, 'length_km']
        x_ohm_per_km = net.line.at[l, 'x_ohm_per_km']
        X_ohm = x_ohm_per_km * length
        x[l] = X_ohm / Z_base
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka * 1e3 / 1e6

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config, prev_power)

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

    p_grid = solver.NumVar(-INF, INF, "p_grid")
    p_grid_import = solver.NumVar(0, INF, "p_grid_import")
    solver.Add(p_grid_import >= p_grid)

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
    net_inj = {b: p_grid if b == slack_bus else solver.Sum() for b in buses}
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
        obj.SetCoefficient(pv_v[nm], -agent_info[nm]['offer'])
        obj.SetCoefficient(wind_v[nm], -agent_info[nm]['offer'])
        obj.SetCoefficient(dis_v[nm], -agent_info[nm]['offer'])
        obj.SetCoefficient(unserved[nm], -config.penalty_unserved)

    use_constraint = config.enable_multi_objective and config.use_constraint_multi_obj
    if not use_constraint:
        if config.enable_multi_objective:
            if config.lambda_re > 0:
                for a in agents:
                    obj.SetCoefficient(pv_v[a.name], obj.GetCoefficient(pv_v[a.name]) + config.lambda_re)
                    obj.SetCoefficient(wind_v[a.name], obj.GetCoefficient(wind_v[a.name]) + config.lambda_re)
            if config.lambda_curtail > 0:
                for a in agents:
                    nm = a.name
                    obj.SetCoefficient(pv_v[nm], obj.GetCoefficient(pv_v[nm]) + config.lambda_curtail)
                    obj.SetCoefficient(wind_v[nm], obj.GetCoefficient(wind_v[nm]) + config.lambda_curtail)
            if config.lambda_carbon > 0:
                obj.SetCoefficient(p_grid_import, -config.lambda_carbon * config.emission_factor_grid)
        else:
            if config.lambda_re > 0:
                for a in agents:
                    obj.SetCoefficient(pv_v[a.name], obj.GetCoefficient(pv_v[a.name]) + config.lambda_re)
                    obj.SetCoefficient(wind_v[a.name], obj.GetCoefficient(wind_v[a.name]) + config.lambda_re)

    obj.SetCoefficient(p_grid, -wholesale_t)
    obj.SetMaximization()

    if use_constraint:
        dt = 0.25
        T_default = 96
        if config.carbon_cap_tco2 is not None:
            per_period_cap = config.carbon_cap_tco2 / T_default
            solver.Add(config.emission_factor_grid * p_grid_import * dt <= per_period_cap)
        if config.re_min_rate is not None:
            re_used = solver.Sum()
            re_avail = 0.0
            for a in agents:
                re_used += pv_v[a.name] + wind_v[a.name]
                re_avail += agent_info[a.name]['pv_max'] + agent_info[a.name]['wind_max']
            if re_avail > 1e-6:
                solver.Add(re_used >= config.re_min_rate / 100.0 * re_avail)

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
        node_lmp = lmp[bus] if bus < len(lmp) else wholesale_t
        revenue = s_val * info['bid'] - (pv_val + w_val + dis_val) * info['offer']
        total_welfare += revenue
        agent_res[nm] = {
            'served': s_val, 'pv_used': pv_val, 'wind_used': w_val,
            'p_ch': ch_val, 'p_dis': dis_val,
        }

    return True, lmp, total_welfare, agent_res, p_grid.SolutionValue()


# ===========================================================================
# 多时段联合优化（核心修复）
# ===========================================================================
def solve_lindist_opf_batch(net, agents, T, stage, config, action_params, wholesale):
    """
    批量求解 LinDistFlow，包含储能 SOC 转移约束和终端价值。
    返回与 clear_market 一致的字典。
    """
    if not _HAS_GUROBI:
        raise RuntimeError("Batch LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")
    n_buses = len(net.bus.index)
    lmp = np.zeros((T, n_buses))
    buses = list(net.bus.index)
    lines = list(net.line.index)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.base_kv
    base_mva = 1.0
    Z_base = (base_kv ** 2) / base_mva

    # 线路参数
    r = {}; x_val = {}; limit = {}
    for l in lines:
        length = net.line.at[l, 'length_km']
        r_ohm = net.line.at[l, 'r_ohm_per_km'] * length
        x_ohm = net.line.at[l, 'x_ohm_per_km'] * length
        r[l] = r_ohm / Z_base
        x_val[l] = x_ohm / Z_base
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka * 1e3 / 1e6

    m = gp.Model("LinDistFlow_batch_T")
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 2)

    # 变量：电压、线路潮流、主网功率
    V   = m.addVars(T, buses, lb=0.85, ub=1.15, name="V")
    P   = m.addVars(T, lines, lb=-GRB.INFINITY, name="P")
    p_grid = m.addVars(T, lb=-GRB.INFINITY, name="p_grid")
    p_grid_import = m.addVars(T, lb=0, ub=GRB.INFINITY, name="p_grid_import")
    for t in range(T):
        m.addConstr(p_grid_import[t] >= p_grid[t], f"grid_import_def_{t}")

    # 智能体变量
    served   = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    soc      = {}
    agents_list = list(agents)
    storage_agents = [a for a in agents_list if a.storage is not None]

    for a in agents_list:
        nm = a.name
        served[nm]   = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"served_{nm}")
        unserved[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"unserv_{nm}")
        pv[nm]       = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"pv_{nm}")
        wind[nm]     = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"wind_{nm}")
        ch[nm]       = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
        dis[nm]      = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"dis_{nm}")

    for a in storage_agents:
        soc[a.name] = m.addVars(T+1, lb=a.storage.soc_min, ub=a.storage.soc_max,
                                name=f"soc_{a.name}")
        m.addConstr(soc[a.name][0] == a.storage.soc0, f"init_soc_{a.name}")

    # 逐时段约束
    for t in range(T):
        m.addConstr(V[t, slack_bus] == 1.0, f"ref_voltage_{t}")

        for b in buses:
            inj = gp.LinExpr()
            if b == slack_bus:
                inj += p_grid[t]
            for a in agents_list:
                if a.bus == b:
                    nm = a.name
                    inj += pv[nm][t] + wind[nm][t] + dis[nm][t] - served[nm][t] - ch[nm][t]
            flow = gp.LinExpr()
            for l in lines:
                if net.line.at[l, 'from_bus'] == b:
                    flow += P[t, l]
                elif net.line.at[l, 'to_bus'] == b:
                    flow -= P[t, l]
            m.addConstr(inj == flow, f"p_bal_{t}_{b}")

        for l in lines:
            f = net.line.at[l, 'from_bus']
            to = net.line.at[l, 'to_bus']
            m.addConstr(V[t, f] - V[t, to] == r[l] * P[t, l], f"vdrop_{t}_{l}")

        for l in lines:
            m.addConstr(P[t, l] <= limit[l], f"limP_{t}_{l}")
            m.addConstr(P[t, l] >= -limit[l], f"limN_{t}_{l}")

        for a in agents_list:
            nm = a.name
            load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
            m.addConstr(served[nm][t] + unserved[nm][t] == load_val, f"load_{t}_{nm}")

        for a in agents_list:
            nm = a.name
            pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
            wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
            pv[nm][t].UB = pv_max
            wind[nm][t].UB = wind_max

        for a in storage_agents:
            nm = a.name
            stor = a.storage
            ch[nm][t].UB = stor.p_ch_max
            dis[nm][t].UB = stor.p_dis_max

            dt = 0.25
            eta_ch = stor.eta_ch; eta_dis = stor.eta_dis
            e_max = stor.e_max; sigma = stor.self_discharge_rate
            soc_t = soc[nm][t]
            soc_next = soc[nm][t+1]
            m.addConstr(
                soc_next == soc_t + (eta_ch * ch[nm][t] - dis[nm][t] / eta_dis) * dt / e_max
                           - sigma * soc_t * dt,
                f"soctrans_{nm}_{t}"
            )

    # 目标函数 — 预计算每个 agent 的 bid/offer 数组
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
    terminal_value = 300.0
    for t in range(T):
        pv_dict_t = {}; wind_dict_t = {}
        pv_max_dict_t = {}; wind_max_dict_t = {}
        for a in agents_list:
            nm = a.name
            bid = bid_arr[nm][t]
            offer = offer_arr[nm][t]

            obj += bid * served[nm][t]
            obj -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
            obj -= config.penalty_unserved * unserved[nm][t]
            pv_dict_t[nm] = pv[nm][t]
            wind_dict_t[nm] = wind[nm][t]
            pv_max_dict_t[nm] = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
            wind_max_dict_t[nm] = (a.get_wind_forecast()[t] if stage == "DA" else a.get_wind_real()[t]) if a.has_wind else 0.0
        obj -= wholesale[t] * p_grid[t]
        if not (config.enable_multi_objective and config.use_constraint_multi_obj):
            _add_multi_objective_terms(obj, config, agents_list, pv_dict_t, wind_dict_t,
                                        pv_max_dict_t, wind_max_dict_t, p_grid_import[t])

    constraint_objs = _add_multi_objective_constraints_batch(m, config, agents_list, pv, wind,
                                             p_grid_import, T, stage)

    for a in storage_agents:
        obj += terminal_value * soc[a.name][T] * a.storage.e_max

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("批量模型求解失败，状态：", m.status)
        return None

    # 提取结果
    from market import empty_schedules as _empty_schedules
    schedules = _empty_schedules(agents_list, T)
    dt = 0.25

    total_welfare = m.ObjVal
    total_re_avail = sum(
        (np.sum(a.pv_forecast if stage == "DA" else a.pv_real) +
         (np.sum(a.wind_forecast if stage == "DA" else a.wind_real) if a.has_wind else 0))
        * dt
        for a in agents_list
    )
    total_re_used = 0.0
    total_curtailment = 0.0
    carbon_emissions = 0.0
    total_served_mwh = 0.0

    for t in range(T):
        for i, b in enumerate(buses):
            constr = m.getConstrByName(f"p_bal_{t}_{b}")
            if constr is not None:
                lmp[t, i] = -constr.Pi
            else:
                lmp[t, i] = wholesale[t]

        pgi = p_grid_import[t].X
        if pgi > 0:
            carbon_emissions += config.emission_factor_grid * pgi * dt

        schedules['GRID']['g_grid'][t] = p_grid[t].X

        for a in agents_list:
            nm = a.name
            s = schedules[nm]
            s['served'][t]    = served[nm][t].X
            s['unserved'][t]  = unserved[nm][t].X
            s['pv_used'][t]   = pv[nm][t].X
            s['wind_used'][t] = wind[nm][t].X
            s['p_ch'][t]      = ch[nm][t].X
            s['p_dis'][t]     = dis[nm][t].X

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
            from market import split_power as _split_power
            s['p_buy'][t], s['p_sell'][t] = _split_power(net_gen, net_con)
            total_re_used += (s['pv_used'][t] + s['wind_used'][t]) * dt
            total_curtailment += ((pv_max - s['pv_used'][t]) + (wind_max - s['wind_used'][t])) * dt
            total_served_mwh += s['served'][t] * dt

    re_rate = (total_re_used / total_re_avail * 100) if total_re_avail > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    shadow_prices = {}
    if constraint_objs:
        for key, constr in constraint_objs.items():
            try:
                shadow_prices[key] = constr.Pi
            except AttributeError:
                shadow_prices[key] = None
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
    if shadow_prices:
        result["shadow_prices"] = shadow_prices
    return result