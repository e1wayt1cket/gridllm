# dispatch.py
"""
储能物理约束引擎 + Gurobi DC-OPF 求解器（社会福利最大化 → LMP）
pandapower 仅用于读取线路参数，不再参与任何 OPF 求解。
"""
import numpy as np
import pandapower as pp
from typing import Dict, Tuple, List, Optional
import gurobipy as gp
from gurobipy import GRB

from models import MarketConfig, Agent, StorageSpec


# =================== 储能约束引擎 ===================
class StorageConstraints:
    """ASSUME 风格的储能物理约束引擎"""

    @staticmethod
    def feasible_ranges(storage: StorageSpec, soc: float, dt: float = 0.25):
        """返回 (p_ch_min, p_ch_max, p_dis_min, p_dis_max)"""
        p_ch_max = storage.max_charge_feasible(soc, dt)
        p_dis_max = storage.max_discharge_feasible(soc, dt)
        p_ch_min = storage.p_ch_min if storage.p_ch_min < p_ch_max else 0.0
        p_dis_min = storage.p_dis_min if storage.p_dis_min < p_dis_max else 0.0
        return p_ch_min, p_ch_max, p_dis_min, p_dis_max

    @staticmethod
    def apply_ramp(prev_p_ch: float, prev_p_dis: float,
                   target_p_ch: float, target_p_dis: float,
                   storage: StorageSpec) -> Tuple[float, float]:
        """爬坡约束"""
        prev_signed = prev_p_dis - prev_p_ch
        target_signed = target_p_dis - target_p_ch
        ramp_limit = storage.ramp_up_dis if storage.ramp_up_dis is not None else float('inf')
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
    def execute_dispatch(storage: StorageSpec, soc: float,
                         p_ch: float, p_dis: float, dt: float = 0.25) -> Tuple[float, float, float]:
        """执行调度，返回 (actual_ch, actual_dis, new_soc)"""
        p_ch = max(0.0, min(p_ch, storage.p_ch_max))
        p_dis = max(0.0, min(p_dis, storage.p_dis_max))
        p_ch_min, p_ch_max, p_dis_min, p_dis_max = StorageConstraints.feasible_ranges(storage, soc, dt)
        if p_ch > p_ch_max:
            p_ch = p_ch_max
        if 0 < p_ch < p_ch_min:
            p_ch = 0.0
        if p_dis > p_dis_max:
            p_dis = p_dis_max
        if 0 < p_dis < p_dis_min:
            p_dis = 0.0
        energy_in = p_ch * storage.eta_ch * dt if p_ch > 0 else 0.0
        energy_out = p_dis / storage.eta_dis * dt if p_dis > 0 else 0.0
        self_discharge_loss = soc * storage.self_discharge_rate * dt
        soc_change = (energy_in - energy_out) / storage.e_max
        new_soc = soc + soc_change - self_discharge_loss
        new_soc = max(storage.soc_min, min(storage.soc_max, new_soc))
        return p_ch, p_dis, new_soc


# =================== 储能模式决策 ===================
def determine_storage_mode(
    agent: Agent,
    wholesale_t: float,
    prev_soc: float,
    config: MarketConfig,
    action_params: Optional[Dict] = None,
    prev_p_ch: float = 0.0,
    prev_p_dis: float = 0.0,
) -> Tuple[str, float, float]:
    """返回 (mode, target_p_ch, target_p_dis)"""
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
        if p_ch_max > 0 and wholesale_t < agent.bid_value * bid_mult * 0.9:
            mode, t_ch, t_dis = "charge", p_ch_max, 0.0
        else:
            return "idle", 0.0, 0.0
    elif prev_soc >= storage.soc_max - 0.02:
        if p_dis_max > 0 and wholesale_t > agent.offer_cost + offer_adder:
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
        if wholesale_t < charge_threshold and p_ch_max > 0:
            mode, t_ch, t_dis = "charge", p_ch_max, 0.0
        elif wholesale_t > discharge_threshold and p_dis_max > 0:
            mode, t_ch, t_dis = "discharge", 0.0, p_dis_max
        else:
            return "idle", 0.0, 0.0
    if mode != "idle":
        t_ch, t_dis = StorageConstraints.apply_ramp(prev_p_ch, prev_p_dis, t_ch, t_dis, storage)
        if t_ch <= 1e-6 and t_dis <= 1e-6:
            mode = "idle"
    return mode, t_ch, t_dis


# =================== Gurobi DC-OPF 求解器 ===================
def solve_dc_opf_gurobi(
    net: pp.pandapowerNet,
    agents: List[Agent],
    t: int,
    stage: str,
    prev_soc: Dict[str, float],
    wholesale_t: float,
    action_params: Dict[str, Dict],
    config: MarketConfig,
    prev_power: Dict[str, Tuple[float, float]],
) -> Tuple[bool, np.ndarray, float, Dict[str, Dict[str, float]], float]:
    """
    使用 Gurobi 求解单时段 DC-OPF，目标：最大化社会福利。
    返回: (成功标志, 节点LMP (每节点), 社会福利值, 智能体调度结果, 主网注入p_grid)
    """
    # 1. 从 pandapower 网络提取数据
    buses = net.bus.index.tolist()
    lines = net.line.index.tolist()
    n_buses = len(buses)
    base_kv = config.base_kv       # 12.66 kV
    # 计算线路电抗标幺值（基准：1 MVA / 12.66 kV）
    X = {}
    Limit = {}
    for l in lines:
        length = net.line.at[l, 'length_km']
        x_per_km = net.line.at[l, 'x_ohm_per_km']
        Z_base = (base_kv ** 2) / 1.0         # 1 MVA 基准阻抗 (Ω)
        x_ohm = x_per_km * length
        X[l] = x_ohm / Z_base
        max_i_ka = net.line.at[l, 'max_i_ka']
        # 容量限制 (MVA) ≈ √3 * kV * kA，简化为有功上限
        Limit[l] = np.sqrt(3) * base_kv * max_i_ka * 1e3 / 1e6   # 转换为 MW

    # 2. 整理智能体该时段的数据
    agent_info = {}
    for a in agents:
        # 负荷
        load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
        # 光伏/风电
        pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
        wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
        # 储能模式决策
        mode, target_ch, target_dis = "idle", 0.0, 0.0
        if a.storage:
            current_soc = prev_soc.get(a.name, a.storage.soc0)
            prev_ch, prev_dis = prev_power.get(a.name, (0.0, 0.0))
            mode, target_ch, target_dis = determine_storage_mode(
                a, wholesale_t, current_soc, config,
                action_params.get(a.name), prev_ch, prev_dis
            )
        # 报价系数
        ap = action_params.get(a.name, {})
        bid = a.bid_value * ap.get("bid_mult", 1.0)
        offer = a.offer_cost + ap.get("offer_adder", 0.0)

        agent_info[a.name] = {
            'bus': a.bus,
            'load': load_val,
            'pv_max': pv_max,
            'wind_max': wind_max,
            'ch_max': target_ch,
            'dis_max': target_dis,
            'storage_mode': mode,
            'bid': bid,
            'offer': offer,
        }

    # 3. 构建 Gurobi 模型
    m = gp.Model("DC_OPF")
    m.setParam('OutputFlag', 0)   # 关闭求解器日志

    # 变量
    theta = m.addVars(buses, lb=-GRB.INFINITY, name="theta")        # 电压相角
    p_flow = m.addVars(lines, lb=-GRB.INFINITY, name="p_flow")     # 线路有功潮流
    slack_bus = net.ext_grid.at[0, 'bus']
    m.addConstr(theta[slack_bus] == 0, "ref_angle")

    served = {}
    unserved = {}
    pv_gen = {}
    wind_gen = {}
    ch_load = {}
    dis_gen = {}
    for a in agents:
        name = a.name
        info = agent_info[name]
        served[name] = m.addVar(lb=0, ub=info['load'], name=f"served_{name}")
        unserved[name] = m.addVar(lb=0, ub=info['load'], name=f"unserv_{name}")
        m.addConstr(served[name] + unserved[name] == info['load'], f"load_balance_{name}")
        pv_gen[name] = m.addVar(lb=0, ub=info['pv_max'], name=f"pv_{name}")
        wind_gen[name] = m.addVar(lb=0, ub=info['wind_max'], name=f"wind_{name}")
        if info['storage_mode'] == "charge":
            ch_load[name] = m.addVar(lb=0, ub=info['ch_max'], name=f"ch_{name}")
            dis_gen[name] = m.addVar(lb=0, ub=0, name=f"dis_{name}")
        elif info['storage_mode'] == "discharge":
            ch_load[name] = m.addVar(lb=0, ub=0, name=f"ch_{name}")
            dis_gen[name] = m.addVar(lb=0, ub=info['dis_max'], name=f"dis_{name}")
        else:
            ch_load[name] = m.addVar(lb=0, ub=0, name=f"ch_{name}")
            dis_gen[name] = m.addVar(lb=0, ub=0, name=f"dis_{name}")

    p_grid = m.addVar(lb=-GRB.INFINITY, name="p_grid")   # 主网注入（正=购电）

    # 节点净注入功率
    net_inj = {b: gp.LinExpr() for b in buses}
    net_inj[slack_bus] += p_grid
    for a in agents:
        bus = a.bus
        name = a.name
        net_inj[bus] += pv_gen[name] + wind_gen[name] + dis_gen[name] - served[name] - ch_load[name]

    # 节点功率平衡约束
    for b in buses:
        flow_out = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                flow_out += p_flow[l]
            elif net.line.at[l, 'to_bus'] == b:
                flow_out -= p_flow[l]
        m.addConstr(net_inj[b] == flow_out, f"p_balance_{b}")

    # DC 潮流方程
    for l in lines:
        f = net.line.at[l, 'from_bus']
        t = net.line.at[l, 'to_bus']
        m.addConstr(p_flow[l] == (theta[f] - theta[t]) / X[l], f"dc_flow_{l}")

    # 线路容量约束
    for l in lines:
        m.addConstr(p_flow[l] <= Limit[l], f"limit_pos_{l}")
        m.addConstr(p_flow[l] >= -Limit[l], f"limit_neg_{l}")

    # 目标函数：最大化社会福利
    obj = gp.LinExpr()
    for a in agents:
        name = a.name
        obj += agent_info[name]['bid'] * served[name]
        obj -= agent_info[name]['offer'] * (pv_gen[name] + wind_gen[name] + dis_gen[name])
        obj -= config.penalty_unserved * unserved[name]

    # ---- 多目标：可再生消纳奖励 ----
    if config.lambda_re > 0:
        for a in agents:
            obj += config.lambda_re * (pv_gen[a.name] + wind_gen[a.name])
    # 若 lambda_re=0 但 w_re_consume>0（旧版兼容），也可按旧逻辑
    elif config.w_re_consume > 0:
        for a in agents:
            obj += config.w_re_consume * 10.0 * (pv_gen[a.name] + wind_gen[a.name])

    obj -= wholesale_t * p_grid
    m.setObjective(obj, GRB.MAXIMIZE)

    # 求解
    m.optimize()

    if m.status != GRB.OPTIMAL:
        return False, None, 0.0, {}, 0.0 #type: ignore

    # 提取节点 LMP（功率平衡约束的对偶变量）
    lmp = np.zeros(n_buses)
    for b in buses:
        constr = m.getConstrByName(f"p_balance_{b}")
        if constr is not None:
            lmp[b] = constr.Pi
        else:
            lmp[b] = wholesale_t  # 使用默认价格作为后备方案

    # 提取智能体调度结果
    agent_results = {}
    for a in agents:
        name = a.name
        agent_results[name] = {
            'served': served[name].X,
            'pv_used': pv_gen[name].X,
            'wind_used': wind_gen[name].X,
            'p_ch': ch_load[name].X,
            'p_dis': dis_gen[name].X,
            'storage_mode': agent_info[name]['storage_mode']
        }

    social_welfare = m.ObjVal
    actual_p_grid = p_grid.X

    return True, lmp, social_welfare, agent_results, actual_p_grid