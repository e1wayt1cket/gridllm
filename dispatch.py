# dispatch.py
"""储能约束 + Gurobi OPF：DC / LinDistFlow 双模式 + 多时段联合优化"""

import numpy as np
import pandapower as pp
from typing import Dict, Tuple, List
import gurobipy as gp
from gurobipy import GRB
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

    if prev_soc <= storage.soc_min + 0.02:
        if np.any(wholesale_t < agent.bid_value * bid_mult * 0.9) and p_ch_max > 0:
            mode, t_ch, t_dis = "charge", p_ch_max, 0.0
        else:
            return "idle", 0.0, 0.0
    elif prev_soc >= storage.soc_max - 0.02:
        if np.any(wholesale_t > agent.offer_cost + offer_adder) and p_dis_max > 0:
            mode, t_ch, t_dis = "discharge", 0.0, p_dis_max
        else:
            return "idle", 0.0, 0.0
    else:
        round_trip_eff = storage.eta_ch * storage.eta_dis
        charge_threshold = agent.bid_value * bid_mult * round_trip_eff * 0.85
        discharge_threshold = (agent.offer_cost + offer_adder) / round_trip_eff * 1.15
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
# DC-OPF 求解
# ===========================================================================
def solve_dc_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                        action_params, config, prev_power):
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

    agent_info = {}
    for a in agents:
        load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
        pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
        wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
        mode, target_ch, target_dis = "idle", 0.0, 0.0
        if a.storage:
            cur_soc = prev_soc.get(a.name, a.storage.soc0)
            pch, pdis = prev_power.get(a.name, (0.0, 0.0))
            mode, target_ch, target_dis = determine_storage_mode(
                a, wholesale_t, cur_soc, config, action_params.get(a.name), pch, pdis)
        ap = action_params.get(a.name, {})
        bid = a.bid_value * (ap["bid_mult"][t] if isinstance(ap.get("bid_mult"), np.ndarray)
                             else ap.get("bid_mult", 1.0))
        offer = a.offer_cost + (ap["offer_adder"][t] if isinstance(ap.get("offer_adder"), np.ndarray)
                                else ap.get("offer_adder", 0.0))
        agent_info[a.name] = {
            'bus': a.bus, 'load': load_val, 'pv_max': pv_max, 'wind_max': wind_max,
            'ch_max': target_ch, 'dis_max': target_dis, 'storage_mode': mode,
            'bid': bid, 'offer': offer,
        }

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
        if info['storage_mode'] == "charge":
            ch[nm] = m.addVar(lb=0, ub=info['ch_max'], name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
        elif info['storage_mode'] == "discharge":
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=info['dis_max'], name=f"dis_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid = m.addVar(lb=-GRB.INFINITY, name="p_grid")

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
    if config.lambda_re > 0:
        for a in agents:
            obj += config.lambda_re * (pv[a.name] + wind[a.name])
    obj -= wholesale_t * p_grid
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

    agent_info = {}
    for a in agents:
        load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
        pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
        wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
        mode, target_ch, target_dis = "idle", 0.0, 0.0
        if a.storage:
            cur_soc = prev_soc.get(a.name, a.storage.soc0)
            pch, pdis = prev_power.get(a.name, (0.0, 0.0))
            mode, target_ch, target_dis = determine_storage_mode(
                a, wholesale_t, cur_soc, config, action_params.get(a.name), pch, pdis)
        ap = action_params.get(a.name, {})
        bid = a.bid_value * (ap["bid_mult"][t] if isinstance(ap.get("bid_mult"), np.ndarray)
                             else ap.get("bid_mult", 1.0))
        offer = a.offer_cost + (ap["offer_adder"][t] if isinstance(ap.get("offer_adder"), np.ndarray)
                                else ap.get("offer_adder", 0.0))
        agent_info[a.name] = {
            'bus': a.bus, 'load': load_val, 'pv_max': pv_max, 'wind_max': wind_max,
            'ch_max': target_ch, 'dis_max': target_dis, 'storage_mode': mode,
            'bid': bid, 'offer': offer,
        }

    m = gp.Model("LinDistFlow")
    m.setParam('OutputFlag', 0)

    V = m.addVars(buses, lb=0.85, ub=1.15, name="V")
    P = m.addVars(lines, lb=-GRB.INFINITY, name="P")
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
        if info['storage_mode'] == "charge":
            ch[nm] = m.addVar(lb=0, ub=info['ch_max'], name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
        elif info['storage_mode'] == "discharge":
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=info['dis_max'], name=f"dis_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid = m.addVar(lb=-GRB.INFINITY, name="p_grid")

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
    if config.lambda_re > 0:
        for a in agents:
            obj += config.lambda_re * (pv[a.name] + wind[a.name])
    obj -= wholesale_t * p_grid
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
    if config.opf_mode == "dc":
        return solve_dc_opf_gurobi(net, agents, t, stage, prev_soc,
                                   wholesale_t, action_params, config, prev_power)
    elif config.opf_mode == "lindistflow":
        return solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc,
                                        wholesale_t, action_params, config, prev_power)
    else:
        raise ValueError(f"未知 OPF 模式: {config.opf_mode}")


# ===========================================================================
# 多时段联合优化（核心修复）
# ===========================================================================
def solve_lindist_opf_batch(net, agents, T, stage, config, action_params, wholesale):
    """
    批量求解 LinDistFlow，包含储能 SOC 转移约束和终端价值。
    返回与 clear_market 一致的字典。
    """
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

    # 目标函数
    obj = gp.LinExpr()
    terminal_value = 300.0
    for t in range(T):
        for a in agents_list:
            nm = a.name
            ap = action_params.get(a.name, {})
            bid_mult = ap.get("bid_mult", 1.0)
            offer_adder = ap.get("offer_adder", 0.0)
            if isinstance(bid_mult, np.ndarray):
                bid_mult = bid_mult[t]
            if isinstance(offer_adder, np.ndarray):
                offer_adder = offer_adder[t]
            bid = a.bid_value * bid_mult
            offer = a.offer_cost + offer_adder

            obj += bid * served[nm][t]
            obj -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
            obj -= config.penalty_unserved * unserved[nm][t]
            if config.lambda_re > 0:
                obj += config.lambda_re * (pv[nm][t] + wind[nm][t])
        obj -= wholesale[t] * p_grid[t]

    for a in storage_agents:
        obj += terminal_value * soc[a.name][T] * a.storage.e_max

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print("批量模型求解失败，状态：", m.status)
        return None

    # 提取结果
    schedules = {}
    for a in agents_list:
        schedules[a.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'storage_mode': ['idle'] * T,
        }
    schedules['GRID'] = {'g_grid': np.zeros(T)}

    total_welfare = m.ObjVal
    total_re_avail = sum(
        np.sum(a.pv_forecast if stage == "DA" else a.pv_real) +
        (np.sum(a.wind_forecast if stage == "DA" else a.wind_real) if a.has_wind else 0)
        for a in agents_list
    )
    total_re_used = 0.0

    for t in range(T):
        for i, b in enumerate(buses):
            constr = m.getConstrByName(f"p_bal_{t}_{b}")
            if constr is not None:
                lmp[t, i] = -constr.Pi
            else:
                lmp[t, i] = wholesale[t]

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
            net_gen = s['pv_used'][t] + s['wind_used'][t] + s['p_dis'][t]
            net_con = s['served'][t] + s['p_ch'][t]
            if net_gen > net_con:
                s['p_sell'][t] = net_gen - net_con
                s['p_buy'][t] = 0.0
            else:
                s['p_buy'][t] = net_con - net_gen
                s['p_sell'][t] = 0.0
            total_re_used += s['pv_used'][t] + s['wind_used'][t]

    re_rate = (total_re_used / total_re_avail * 100) if total_re_avail > 0 else 100.0
    return {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_avail,
    }