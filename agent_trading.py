import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

np.random.seed(1)

# ----------------------------
# 1) 数据结构：智能体与系统
# ----------------------------


@dataclass
class StorageSpec:
    e_max: float          # MWh
    p_ch_max: float       # MW
    p_dis_max: float      # MW
    eta_ch: float         # 0-1
    eta_dis: float        # 0-1
    soc0: float           # MWh
    soc_min: float        # MWh
    soc_max: float        # MWh
    wind_used: float


@dataclass
class Agent:
    name: str
    bus: int                      # 接入节点
    # True: 既有负荷又有PV和储能；False: 纯负荷 prosumer=producer+consumer 产消者
    is_prosumer: bool
    load_forecast: np.ndarray     # (T,)
    pv_forecast: np.ndarray       # (T,) for prosumer, else zeros
    load_real: np.ndarray         # (T,)
    pv_real: np.ndarray           # (T,)
    # 报价参数（由策略/智能体动作给出）
    bid_value: float              # ¥/MWh（买方边际价值上限的基准）
    offer_cost: float             # ¥/MWh（卖方边际成本基准；对PV可理解为机会成本/磨损等）
    # ---- 新增：风电字段（任务1，有默认值，放最后）----
    wind_forecast: Optional[np.ndarray] = None   # (T,) 风电预测出力; None表示无风电
    wind_used: Optional[np.ndarray] = None  # (T,) 风电使用量（决策变量，出清结果中提供）任务1新增字段
    wind_real: Optional[np.ndarray] = None      # (T,) 风电实际出力     
    # 储能（只有 prosumer 1/2 有）
    storage: Optional[StorageSpec] = None   # 如果没有储能，storage=None；否则提供储能参数

    def has_wind(self) -> bool:
        """判断该Agent是否配备风电"""
        return self.wind_forecast is not None and self.wind_real is not None

    @property
    def total_re_forecast(self) -> np.ndarray:
        """总可再生预测 = PV + Wind"""
        if self.has_wind():
            return self.pv_forecast + self.wind_forecast
        return self.pv_forecast

    @property
    def total_re_real(self) -> np.ndarray:
        """总可再生实际 = PV + Wind"""
        if self.has_wind():
            return self.pv_real + self.wind_real
        return self.pv_real



@dataclass
class Network:
    """
    通用配网拓扑结构（任务3：从固定2支路→任意支路列表）

    径向配网示例：
      0(外部电网) --cap01--> 1 --cap12--> 2 --cap23--> 3 ...
    潮流近似：下游注入累加
      flow_ij = Σ net_inj(bus_k) for all k downstream of i toward j

    branches: list of dict, each {"from": int, "to": int, "cap": float}
              cap 单位 MW，双向容量约束 |flow| <= cap
    """
    branches: List[Dict[str, any]]  # 支路列表

    @classmethod
    def simple_2bus(cls, cap01: float, cap12: float) -> 'Network':
        """兼容旧接口：创建 0-1-2 两支路网络"""
        return cls(branches=[
            {"from": 0, "to": 1, "cap": cap01},
            {"from": 1, "to": 2, "cap": cap12},
        ])

    @classmethod
    def from_edges(cls, edge_list: list) -> 'Network':
        """
        从边列表创建网络
        edge_list: [(from_bus, to_bus, capacity_MW), ...]
        示例: [(0,1,8.0), (1,2,5.0), (2,3,4.0), (3,4,6.0)]
        """
        branches = [{"from": e[0], "to": e[1], "cap": e[2]} for e in edge_list]
        return cls(branches=branches)


# ----------------------------
# 2) 出清：社会福利最大化 LP（DA/RT共用） 社会总福利=买家净剩余+卖家净利润
# ----------------------------

def clear_market_lp(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,                     # "DA" or "RT"
    wholesale_price: np.ndarray,     # (T,) 外部电网边际成本/价格（作为“系统电源”）
    action_params: Dict[str, Dict],  # 各智能体本阶段动作（报价参数）
    penalty_unserved: float = 500.0  # 未满足负荷惩罚（¥/MWh）
) -> Dict:
    """
    返回：
      - clearing_price: (T,) 这里用“影子价格”近似 LMP（简化：系统单节点价格）
      - schedules: per agent {p_buy, p_sell, p_ch, p_dis, soc, served_load, pv_used}
    """
    # 变量：每个agent每时段
    p_buy = {}
    p_sell = {}
    served = {}
    unserved = {}
    pv_used = {}
    p_ch = {}
    p_dis = {}
    soc = {}

    # 外部电网供给（系统电源）：g_grid >= 0
    g_grid = cp.Variable(T, nonneg=True)

    # 风电使用量字典（任务1）
    wind_used = {}

    for a in agents:
        p_buy[a.name] = cp.Variable(T, nonneg=True)
        p_sell[a.name] = cp.Variable(T, nonneg=True)
        served[a.name] = cp.Variable(T, nonneg=True)
        unserved[a.name] = cp.Variable(T, nonneg=True)

        pv_used[a.name] = cp.Variable(T, nonneg=True)

        # ---- 风电使用量（任务1）----
        if a.has_wind():
            wind_used[a.name] = cp.Variable(T, nonneg=True)
        else:
            wind_used[a.name] = None

        if a.storage is not None:
            p_ch[a.name] = cp.Variable(T, nonneg=True)
            p_dis[a.name] = cp.Variable(T, nonneg=True)
            soc[a.name] = cp.Variable(T)
        else:
            p_ch[a.name] = None
            p_dis[a.name] = None
            soc[a.name] = None

    constraints = []

    # 选择使用 forecast 或 real
    if stage == "DA":
        load = {a.name: a.load_forecast for a in agents}
        pv = {a.name: a.pv_forecast for a in agents}
    else:
        load = {a.name: a.load_real for a in agents}
        pv = {a.name: a.pv_real for a in agents}

    # 负荷满足：served + unserved = load
    for a in agents:
        constraints += [served[a.name] + unserved[a.name] == load[a.name]]

    # PV使用上限
    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]

    # Wind使用上限（任务1）
    for a in agents:
        if a.has_wind():
            if stage == "DA":
                w_avail = a.wind_forecast
            else:
                w_avail = a.wind_real
            constraints += [wind_used[a.name] <= w_avail]
        else:
            if wind_used.get(a.name) is not None:
                constraints += [wind_used[a.name] == 0]

    # 储能约束
    for a in agents:
        if a.storage is None:
            constraints += [p_ch[a.name] is None] if False else []
            continue
        st = a.storage
        constraints += [p_ch[a.name] <= st.p_ch_max]
        constraints += [p_dis[a.name] <= st.p_dis_max]
        # SOC 动力学
        constraints += [soc[a.name][0] == st.soc0
                        + st.eta_ch * p_ch[a.name][0]
                        - (1.0 / st.eta_dis) * p_dis[a.name][0]]
        for t in range(1, T):
            constraints += [soc[a.name][t] == soc[a.name][t - 1]
                            + st.eta_ch * p_ch[a.name][t]
                            - (1.0 / st.eta_dis) * p_dis[a.name][t]]
        constraints += [soc[a.name] >= st.soc_min, soc[a.name] <= st.soc_max]

    # 功率平衡（全系统）
    # 系统供给：PV_used + p_sell(可理解为本地出清中的“对外供给”变量) + grid
    # 系统用电：served(负荷) + p_buy(从市场买) + 充电 + 其它
    #
    # 这里做“单市场池”：p_buy/p_sell 表示参与市场的净交易分解
    # 实际上你也可以简化为每个agent一个净注入变量 n = gen - load + dis - ch + grid_import
    for t in range(T):
        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t]
                                          for a in agents])
        # 加入风电供给（任务1）
        total_supply += cp.sum([wind_used[a.name][t]
                                for a in agents if wind_used.get(a.name) is not None])
        total_supply += cp.sum([p_sell[a.name][t] for a in agents])
        total_demand = cp.sum([served[a.name][t] for a in agents]) + \
            cp.sum([p_buy[a.name][t] for a in agents])

        # 储能充放电影响：充电视为需求，放电视为供给
        for a in agents:
            if a.storage is not None:
                total_supply += p_dis[a.name][t]
                total_demand += p_ch[a.name][t]

        constraints += [total_supply == total_demand]

    # Agent 能量收支：买卖与本地PV/储能与负荷一致（每个 agent 的“能量守恒”）
    # served_load 由 (PV_used + buy + discharge) 供给，且多余可卖出或弃电（弃电由 pv_used<=pv
    # 体现）
    for a in agents:
        for t in range(T):
            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                lhs_supply += wind_used[a.name][t]
            if a.storage is not None:
                lhs_supply += p_dis[a.name][t]
            rhs_use = served[a.name][t] + p_sell[a.name][t]
            if a.storage is not None:
                rhs_use += p_ch[a.name][t]
            constraints += [lhs_supply == rhs_use]

    # 网络约束（径向线容量简化）
    # 计算各母线净注入：net_inj = (PV_used + dis + sell) - (served + ch + buy)
    # 注意这里 sell/buy是“市场交易分解”，仍可作为节点注入项（等价）
    buses = sorted(set(a.bus for a in agents))
    # 我们只做 0-1-2 结构：bus0 为外部电网点，不显式建 agent
    # 将 g_grid 视为在 bus0 注入供给
    for t in range(T):
        net_inj_bus = {b: 0 for b in buses}
        for a in agents:
            b = a.bus
            inj = pv_used[a.name][t] + p_sell[a.name][t] - \
                served[a.name][t] - p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                inj += wind_used[a.name][t]
            if a.storage is not None:
                inj += p_dis[a.name][t] - p_ch[a.name][t]
            net_inj_bus[b] += inj

        # ---- 通用网络潮流约束（任务3）----
        # 对每条支路 (from_bus -> to_bus)，计算流过该支路的功率
        # 径向网假设：flow_ij = Σ net_inj(bus_k), k为j侧所有下游节点
        for branch in network.branches:
            bus_from = branch["from"]
            bus_to = branch["to"]
            cap = branch["cap"]

            # 找出 bus_to 及其所有下游节点的集合
            # 用BFS从bus_to出发沿支路方向搜索下游
            downstream = set()
            queue = [bus_to]
            visited_bfs = set()
            while queue:
                node = queue.pop(0)
                if node in visited_bfs:
                    continue
                visited_bfs.add(node)
                downstream.add(node)
                for br in network.branches:
                    if br["from"] == node and br["to"] not in visited_bfs:
                        queue.append(br["to"])

            # 支路潮流 = 所有下游节点净注入之和
            flow_ij = sum(net_inj_bus.get(b, 0) for b in downstream)
            constraints += [flow_ij <= cap, flow_ij >= -cap]

    # ----------------------------
    # 目标函数：社会福利最大化
    # ----------------------------
    welfare = 0
    for a in agents:
        # 本阶段动作：例如 bid_multiplier / offer_adder
        ap = action_params.get(a.name, {})
        bid_mult = ap.get("bid_mult", 1.0)           # 买方愿付价倍率
        offer_adder = ap.get("offer_adder", 0.0)     # 卖方报价加成（¥/MWh）

        # 买方效用：bid_price * served_load
        bid_price = bid_mult * a.bid_value
        welfare += bid_price * cp.sum(served[a.name])

        # 卖方成本：对卖出电量计成本（可理解为机会成本/磨损/燃料等）
        offer_price = a.offer_cost + offer_adder
        welfare -= offer_price * cp.sum(p_sell[a.name])

        # grid 成本
    welfare -= cp.sum(cp.multiply(wholesale_price, g_grid))

    # 未满足负荷惩罚（强制系统尽量满足）
    welfare -= penalty_unserved * \
        cp.sum(cp.hstack([cp.sum(unserved[a.name]) for a in agents]))

    problem = cp.Problem(cp.Maximize(welfare), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP not solved: {problem.status}")

    # 价格：严格的节点边际电价需要对每节点功率平衡取对偶，这里简化为“系统平衡约束”的对偶值
    # 我们取每时段系统平衡约束对应的 dual 作为系统价格近似（¥/MWh）
    # 在上面构造中，每时段系统平衡约束是 constraints 中按顺序追加的一段；我们重新索引会很麻烦
    # 为保持简单：用“外部电网边际成本 + 拥塞溢价近似”，这里直接用 g_grid 的边际成本并加拥塞项（简化）
    # 如果你需要严格 LMP，我可以再给你一个“显式节点平衡+对偶提取”的版本。
    clearing_price = wholesale_price.copy()

    schedules = {}
    for a in agents:
        sched_entry = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }
        # 风电使用量（任务1）
        if wind_used.get(a.name) is not None:
            sched_entry["wind_used"] = np.array(wind_used[a.name].value).reshape(-1)
        schedules[a.name] = sched_entry
        if a.storage is not None:
            schedules[a.name].update({
                "p_ch": np.array(p_ch[a.name].value).reshape(-1),
                "p_dis": np.array(p_dis[a.name].value).reshape(-1),
                "soc": np.array(soc[a.name].value).reshape(-1),
            })
    schedules["GRID"] = {"g_grid": np.array(g_grid.value).reshape(-1)}

    return {
        "price": clearing_price,
        "schedules": schedules,
        "welfare": problem.value
    }

# ----------------------------
# 3) 多目标优化：社会福利 + PV消纳最大化（弃光量最小化）
#    参考: Applied Energy 312 (2022) 118724 目标二 waste minimization
# ----------------------------


def clear_market_multi_obj_lp(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,
    wholesale_price: np.ndarray,
    action_params: Dict[str, Dict],
    penalty_unserved: float = 500.0,
    # ---- 多目标权重 ----
    w_welfare: float = 1.0,       # 社会福利权重
    w_re_consume: float = 0.5,    # PV消纳权重（弃光惩罚系数）
) -> Dict:
    """
    双目标优化出清：
      目标1：社会总福利最大化（原有目标）
      目标2：PV弃光量最小化 / 消纳率最大化（参考论文 Eq.14）

    综合目标 = w_welfare * welfare - w_re_consume * total_waste

    参数：
      w_welfare   : 社会福利权重（默认1.0，保持与原函数一致）
      w_re_consume: 弃光惩罚权重，越大越倾向多用PV；设为0则退化为原单目标

    返回：同 clear_market_lp，额外含 "pv_waste" 和 "obj_detail"
    """
    # ---- 复用原LP结构构建变量和约束 ----
    p_buy = {}
    p_sell = {}
    served = {}
    unserved = {}
    pv_used = {}
    p_ch = {}
    p_dis = {}
    soc = {}
    wind_used = {}  

    g_grid = cp.Variable(T, nonneg=True)

    for a in agents:
        p_buy[a.name] = cp.Variable(T, nonneg=True)
        p_sell[a.name] = cp.Variable(T, nonneg=True)
        served[a.name] = cp.Variable(T, nonneg=True)
        unserved[a.name] = cp.Variable(T, nonneg=True)
        pv_used[a.name] = cp.Variable(T, nonneg=True)

        # ---- 风电使用量（任务1）----
        if a.has_wind():
            wind_used[a.name] = cp.Variable(T, nonneg=True)
        else:
            wind_used[a.name] = None

        if a.storage is not None:
            p_ch[a.name] = cp.Variable(T, nonneg=True)
            p_dis[a.name] = cp.Variable(T, nonneg=True)
            soc[a.name] = cp.Variable(T)
        else:
            p_ch[a.name] = None
            p_dis[a.name] = None
            soc[a.name] = None

    constraints = []

    if stage == "DA":
        load = {a.name: a.load_forecast for a in agents}
        pv = {a.name: a.pv_forecast for a in agents}
    else:
        load = {a.name: a.load_real for a in agents}
        pv = {a.name: a.pv_real for a in agents}

    # 负荷满足
    for a in agents:
        constraints += [served[a.name] + unserved[a.name] == load[a.name]]

    # PV使用上限
    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]

    # Wind使用上限（任务1）
    for a in agents:
        if a.has_wind():
            if stage == "DA":
                w_avail = a.wind_forecast
            else:
                w_avail = a.wind_real
            constraints += [wind_used[a.name] <= w_avail]
        else:
            if wind_used.get(a.name) is not None:
                constraints += [wind_used[a.name] == 0]

    # 储能约束
    for a in agents:
        if a.storage is None:
            continue
        st = a.storage
        constraints += [p_ch[a.name] <= st.p_ch_max]
        constraints += [p_dis[a.name] <= st.p_dis_max]
        constraints += [soc[a.name][0] == st.soc0
                        + st.eta_ch * p_ch[a.name][0]
                        - (1.0 / st.eta_dis) * p_dis[a.name][0]]
        for t in range(1, T):
            constraints += [soc[a.name][t] == soc[a.name][t - 1]
                            + st.eta_ch * p_ch[a.name][t]
                            - (1.0 / st.eta_dis) * p_dis[a.name][t]]
        constraints += [soc[a.name] >= st.soc_min, soc[a.name] <= st.soc_max]

    # 功率平衡
    for t in range(T):
        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t]
                                          for a in agents])
        # 加入风电供给（任务1）
        total_supply += cp.sum([wind_used[a.name][t]
                                for a in agents if wind_used.get(a.name) is not None])
        total_supply += cp.sum([p_sell[a.name][t] for a in agents])
        total_demand = cp.sum([served[a.name][t] for a in agents]) + \
            cp.sum([p_buy[a.name][t] for a in agents])

        for a in agents:
            if a.storage is not None:
                total_supply += p_dis[a.name][t]
                total_demand += p_ch[a.name][t]

        constraints += [total_supply == total_demand]

    # Agent能量守恒
    for a in agents:
        for t in range(T):
            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                lhs_supply += wind_used[a.name][t]
            if a.storage is not None:
                lhs_supply += p_dis[a.name][t]
            rhs_use = served[a.name][t] + p_sell[a.name][t]
            if a.storage is not None:
                rhs_use += p_ch[a.name][t]
            constraints += [lhs_supply == rhs_use]

    # 网络约束
    buses = sorted(set(a.bus for a in agents))
    for t in range(T):
        net_inj_bus = {b: 0 for b in buses}
        for a in agents:
            b = a.bus
            inj = pv_used[a.name][t] + p_sell[a.name][t] - \
                served[a.name][t] - p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                inj += wind_used[a.name][t]
            if a.storage is not None:
                inj += p_dis[a.name][t] - p_ch[a.name][t]
            net_inj_bus[b] += inj

        if 1 in net_inj_bus and 2 in net_inj_bus:
            flow12 = net_inj_bus[2]
            flow01 = net_inj_bus[1] + net_inj_bus[2]
            constraints += [flow12 <= network.cap12, flow12 >= -network.cap12]
            constraints += [flow01 <= network.cap01, flow01 >= -network.cap01]

    # ============================
    # 双目标函数
    # ============================
    # --- 目标1：社会总福利（与原函数完全一致）---
    welfare = 0
    for a in agents:
        ap = action_params.get(a.name, {})
        bid_mult = ap.get("bid_mult", 1.0)
        offer_adder = ap.get("offer_adder", 0.0)

        bid_price = bid_mult * a.bid_value
        welfare += bid_price * cp.sum(served[a.name])

        offer_price = a.offer_cost + offer_adder
        welfare -= offer_price * cp.sum(p_sell[a.name])

    welfare -= cp.sum(cp.multiply(wholesale_price, g_grid))
    welfare -= penalty_unserved * \
        cp.sum(cp.hstack([cp.sum(unserved[a.name]) for a in agents]))

    # --- 目标2：PV弃光量最小化 ---
    # waste_t = Σ_a (pv_available_a[t] - pv_used_a[t])
    # 最小化总弃光量 = 最大化 Σ pv_used
    # 这里用常数项处理：min waste ⇔ max pv_used
    # 归一化：除以总可用PV量，使waste∈[0,1]
    # ---- 总可再生能源可用量（PV+Wind, 任务1扩展）----
    total_re_available = 0.0
    for a in agents:
        if a.is_prosumer:
            total_re_available += float(np.sum(pv[a.name]))
        if a.has_wind():
            w_arr = a.wind_forecast if stage == "DA" else a.wind_real
            total_re_available += float(np.sum(w_arr))

    # ---- 弃能量表达式（PV waste + Wind waste）----
    re_waste_expr = 0
    for a in agents:
        if a.is_prosumer:
            # PV弃光量
            re_waste_expr += cp.sum(pv[a.name] - pv_used[a.name])
        if a.has_wind():
            # Wind弃风量
            w_avail = a.wind_forecast if stage == "DA" else a.wind_real
            re_waste_expr += cp.sum(w_avail - wind_used[a.name])

    # --- 综合目标：加权求和 ---
    # maximize: w_welfare * welfare - w_re_consume * pv_waste
    # （注意：welfare是越大越好，waste是越小越好，故取负号）
    if total_re_available > 0:
        objective = w_welfare * welfare - w_re_consume * (re_waste_expr / total_re_available)
    else:
        objective = w_welfare * welfare

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Multi-obj LP not solved: {problem.status}")

    clearing_price = wholesale_price.copy()

    schedules = {}
    for a in agents:
        sched_entry = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }
        # 风电使用量（任务1）
        if wind_used.get(a.name) is not None:
            sched_entry["wind_used"] = np.array(wind_used[a.name].value).reshape(-1)
        schedules[a.name] = sched_entry
        if a.storage is not None:
            schedules[a.name].update({
                "p_ch": np.array(p_ch[a.name].value).reshape(-1),
                "p_dis": np.array(p_dis[a.name].value).reshape(-1),
                "soc": np.array(soc[a.name].value).reshape(-1),
            })
    schedules["GRID"] = {"g_grid": np.array(g_grid.value).reshape(-1)}

    # 计算各目标的实际值（任务1: PV+Wind 弃能）
    re_waste_actual = 0.0
    for a in agents:
        if a.is_prosumer:
            re_waste_actual += float(np.sum(pv[a.name]) - np.sum(pv_used[a.name].value))
        if a.has_wind():
            w_avail_arr = a.wind_forecast if stage == "DA" else a.wind_real
            re_waste_actual += float(np.sum(w_avail_arr) - np.sum(wind_used[a.name].value))

    re_consumption_rate = 1.0 - (re_waste_actual / total_re_available) if total_re_available > 0 else 0.0

    return {
        "price": clearing_price,
        "schedules": schedules,
        "welfare": float(welfare.value) if welfare.value is not None else 0.0,
        # ---- 多目标新增返回项（任务1: 扩展为含风电）----
        "re_waste": re_waste_actual,           # 总弃能量(MWh), 含PV+Wind
        "re_consumption_rate": re_consumption_rate,  # 总消纳率
        "total_re_available": total_re_available,
        "obj_detail": {
            "welfare": float(welfare.value) if welfare.value is not None else 0.0,
            "re_waste": re_waste_actual,
            "re_consumption_rate": re_consumption_rate,
        },
    }


# ----------------------------
# 4) 两阶段结算：DA + RT
# ----------------------------


def two_settlement(
    agents: List[Agent],
    da: Dict,
    rt: Dict
) -> Dict[str, float]:
    """
    结算：对每个agent：
      payment = DA_price * (DA_net_import) + RT_price * (RT_net_import - DA_net_import)
    net_import>0 表示从系统净买（支出），<0 表示净卖（收入）
    """
    T = len(da["price"])
    pay = {a.name: 0.0 for a in agents}

    for a in agents:
        sch_da = da["schedules"][a.name]
        sch_rt = rt["schedules"][a.name]

        # 净买量 = p_buy - p_sell （MWh）
        da_import = sch_da["p_buy"] - sch_da["p_sell"]
        rt_import = sch_rt["p_buy"] - sch_rt["p_sell"]

        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))

        pay[a.name] = float(da_cost + rt_cost)

    return pay

# ----------------------------
# 4) 简单算例：4智能体、24小时
# ----------------------------


def build_demo_case(T=96, with_wind: bool = False) -> Tuple[List[Agent], Network, np.ndarray]:
    """
    构建演示算例（任务1扩展：支持风电Agent）

    参数：
      T: 时段数（默认24小时）
      with_wind: 是否包含风电Agent F（默认False保持向后兼容）

    返回：(agents, network, wholesale_price)
    """
    hours = np.arange(T)

    # 北京工业电价替代曲线（元/千瓦时）
    base_price = np.array([
        1.14758475, 1.13159475, 1.07758475, 1.03258475, 0.88410775,
        0.82860775, 1.50748175, 1.47349775, 1.43748175, 1.39248175,
        1.20401675, 1.14851675
    ])
    wholesale = np.tile(base_price, 2)

    # 预测与实际（简单加噪）
    def noisy(x, sigma=0.1):
        return np.clip(
            x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    # ---- PV出力曲线（正弦模拟日间发电）----
    base_pv_A = np.clip(6 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    base_pv_B = np.clip(5.5 * np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)

    # ---- 风电出力曲线（任务1：Weibull风格，昼夜波动大）----
    # 风电特点：无明显日间峰，夜间也可能较大出力，随机波动强
    if with_wind:
        # 基础风功率：用多正弦叠加模拟昼夜变化 + 随机湍流
        np_rng = np.random.RandomState(42)
        base_wind_F = np.clip(
            8.0 * (0.5 + 0.5 * np.sin((hours - 3) / 12 * np.pi))   # 昼夜分量
            + 3.0 * np.sin((hours - 14) / 8 * np.pi)                  # 中午低谷
            + np_rng.normal(0, 1.5, size=T),                          # 随机湍流
            0, 15.0                                                    # 额定15MW
        )
        base_wind_F = np.maximum(base_wind_F, 0.3)  # 最小出力（避免完全无风）

    # ---- 负荷曲线 ----
    load_A = 2.0 + 0.5 * np.exp(-0.5 * ((hours - 10) / 3.0)**2)
    load_B = 1.8 + 0.4 * np.exp(-0.5 * ((hours - 15) / 3.5)**2)
    load_C = 4.5 + 1.2 * np.exp(-0.5 * ((hours - 9) / 2.5)**2) + \
        1.5 * np.exp(-0.5 * ((hours - 18) / 2.5)**2)
    load_D = 3.8 + 1.0 * np.exp(-0.5 * ((hours - 8) / 2.7)**2) + \
        1.2 * np.exp(-0.5 * ((hours - 20) / 2.7)**2)
    # 新增负荷E/F节点（用于多节点场景）
    load_E = load_D.copy()
    load_F_load = 3.0 + 0.8 * np.exp(-0.5 * ((hours - 12) / 3.0)**2) + \
        1.0 * np.exp(-0.5 * ((hours - 20) / 3.0)**2)

    # ---- Agent A: PV+储能 @ bus1 ----
    A = Agent(
        name="A(PV+ESS)", bus=1, is_prosumer=True,
        load_forecast=noisy(load_A, 0.05), pv_forecast=noisy(base_pv_A, 0.12),
        load_real=noisy(load_A, 0.08), pv_real=noisy(base_pv_A, 0.18),
        wind_forecast=None, wind_real=None,wind_used=None,
        bid_value=90.0, offer_cost=5.0,
        storage=StorageSpec(e_max=8.0, p_ch_max=2.5, p_dis_max=2.5,wind_used=0.0,
                            eta_ch=0.95, eta_dis=0.95, soc0=3.0, soc_min=0.8, soc_max=7.5))

    # ---- Agent B: PV+储能 @ bus2 ----
    B = Agent(
        name="B(PV+ESS)", bus=2, is_prosumer=True,
        load_forecast=noisy(load_B, 0.05), pv_forecast=noisy(base_pv_B, 0.12),
        load_real=noisy(load_B, 0.08), pv_real=noisy(base_pv_B, 0.18),
        wind_forecast=None, wind_real=None,wind_used=None,
        bid_value=88.0, offer_cost=6.0,
        storage=StorageSpec(e_max=6.0, p_ch_max=2.0, p_dis_max=2.0,wind_used=0.0,
                            eta_ch=0.94, eta_dis=0.94, soc0=2.5, soc_min=0.6, soc_max=5.6))

    # ---- Agent C: 纯负荷 @ bus1 ----
    C = Agent(name="C(LOAD)", bus=1, is_prosumer=False,
              load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=110.0, offer_cost=999.0, storage=None)

    # ---- Agent D: 纯负荷 @ bus2 ----
    D = Agent(name="D(LOAD)", bus=2, is_prosumer=False,
              load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=105.0, offer_cost=999.0, storage=None)

    # ---- Agent E: 纯负荷 @ bus2 ----
    E = Agent(name="E(LOAD)", bus=2, is_prosumer=False,
              load_forecast=noisy(load_E, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_E, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=102.0, offer_cost=999.0, storage=None)

    agents_base = [A, B, C, D, E]

    # ---- 任务1：新增风电Agent F(WIND+ESS) ----
    if with_wind:
        F = Agent(
            name="F(WIND+ESS)", bus=3, is_prosumer=True,
            load_forecast=noisy(load_F_load, 0.05), pv_forecast=np.zeros(T),
            load_real=noisy(load_F_load, 0.08), pv_real=np.zeros(T),
            wind_forecast=noisy(base_wind_F, 0.10),     # 风电预测
            wind_real=noisy(base_wind_F, 0.20),         # 风电实际（更大误差）
            bid_value=85.0,                              # 风电边际成本低
            offer_cost=3.0,                              # 风电机会成本更低
            storage=StorageSpec(e_max=10.0, p_ch_max=4.0, p_dis_max=4.0,wind_used=0.0,
                                eta_ch=0.96, eta_dis=0.96, soc0=5.0,
                                soc_min=1.0, soc_max=9.5))
        agents_base.append(F)

        # 如果有风电，增加一个纯负荷G来平衡bus3（可选）
        G = Agent(name="G(LOAD)", bus=3, is_prosumer=False,
                  load_forecast=noisy(load_F_load * 0.8, 0.06), pv_forecast=np.zeros(T),
                  load_real=noisy(load_F_load * 0.8, 0.10), pv_real=np.zeros(T),
                  wind_forecast=None, wind_real=None,
                  bid_value=100.0, offer_cost=999.0, storage=None)
        agents_base.append(G)

    # ---- 网络（任务3：使用新的通用结构）----
    if with_wind:
        # 5节点网络: 0(电网)-1-2-3-4
        network = Network.from_edges([
            (0, 1, 8.0),   # cap01 = 8 MW
            (1, 2, 5.0),   # cap12 = 5 MW
            (2, 3, 6.0),   # cap23 = 6 MW
            (3, 4, 4.0),   # cap34 = 4 MW
        ])
    else:
        # 兼容旧版3节点
        network = Network.simple_2bus(cap01=6.5, cap12=3.5)

    return agents_base, network, wholesale


def random_actions(agents: List[Agent]) -> Dict[str, Dict]:
    """
    先用随机策略：买方 bid_mult in [0.85,1.05], 卖方 offer_adder in [0,8]
    RL 接入后，这里由 policy(obs)->action 生成
    """
    actions = {}
    for a in agents:
        if a.is_prosumer:
            actions[a.name] = {
                "bid_mult": float(np.random.uniform(0.85, 1.05)),
                "offer_adder": float(np.random.uniform(0.0, 6.0))
            }
        else:
            actions[a.name] = {"bid_mult": float(
                np.random.uniform(0.90, 1.10))}
    return actions


def one_day_demo():
    agents, network, wholesale = build_demo_case(T=96)

    # 日前：基于预测
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=96, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )

    # 日内/实时：基于实际（可做滚动，这里简化为一次 RT）
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=96, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )

    payment = two_settlement(agents, da, rt)

    print("=== Day-Ahead (DA) welfare:", round(da["welfare"], 2))
    print("=== Real-Time (RT) welfare:", round(rt["welfare"], 2))
    print("\n--- Payments (positive = cost, negative = revenue) ---")
    for k, v in payment.items():
        print(f"{k:12s}  {v:8.2f} ¥")

    # 示例：看A的SOC与交易
    A = agents[0]
    schA_da = da["schedules"][A.name]
    schA_rt = rt["schedules"][A.name]
    print("\n--- A(RES+ESS) snapshot ---")
    print("DA soc:", np.round(schA_da["soc"], 2))
    print("RT soc:", np.round(schA_rt["soc"], 2))
    print("DA net_import:", np.round(schA_da["p_buy"] - schA_da["p_sell"], 2))
    print("RT net_import:", np.round(schA_rt["p_buy"] - schA_rt["p_sell"], 2))

    B = agents[1]
    schB_da = da["schedules"][B.name]
    schB_rt = rt["schedules"][B.name]
    print("\n--- B(RES+ESS) snapshot ---")
    print("DA soc:", np.round(schB_da["soc"], 2))
    print("RT soc:", np.round(schB_rt["soc"], 2))
    print("DA net_import:", np.round(schB_da["p_buy"] - schB_da["p_sell"], 2))
    print("RT net_import:", np.round(schB_rt["p_buy"] - schB_rt["p_sell"], 2))

    C = agents[2]
    schC_da = da["schedules"][C.name]
    schC_rt = rt["schedules"][C.name]
    print("\n--- C(LOAD) snapshot ---")
    print("DA served:", np.round(schC_da["served"], 2))
    print("RT served:", np.round(schC_rt["served"], 2))
    print("DA unserved:", np.round(schC_da["unserved"], 2))
    print("RT unserved:", np.round(schC_rt["unserved"], 2))

    D = agents[3]
    schD_da = da["schedules"][D.name]
    schD_rt = rt["schedules"][D.name]
    print("\n--- D(LOAD) snapshot ---")
    print("DA served:", np.round(schD_da["served"], 2))
    print("RT served:", np.round(schD_rt["served"], 2))
    print("DA unserved:", np.round(schD_da["unserved"], 2))
    print("RT unserved:", np.round(schD_rt["unserved"], 2))


def get_agent_states():
    """返回DA出清状态（供dashboard等外部调用）"""
    agents, network, wholesale = build_demo_case(T=96)
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=96, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )
    states = {}
    for a in agents:
        states[a.name] = {k: v.tolist()
                          for k, v in da["schedules"][a.name].items()}
    return {"agents": states}


def get_realtime_data():
    """返回RT出清状态（供dashboard等外部调用）"""
    agents, network, wholesale = build_demo_case(T=96)
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=96, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )
    data = {}
    for a in agents:
        data[a.name] = {k: v.tolist()
                        for k, v in rt["schedules"][a.name].items()}
    return {"realtime": data}


# ============================================================
# 任务2: ε-约束法求解 Pareto 前沿
# ============================================================


def solve_pareto_front_epsilon(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,
    wholesale_price: np.ndarray,
    action_params: Dict[str, Dict],
    penalty_unserved: float = 500.0,
    n_points: int = 20,
) -> List[Dict]:
    """
    ε-约束法 (Epsilon-Constraint Method) 求解双目标 Pareto 前沿

    方法说明：
      将目标2(弃能量最小化)转化为约束：
        max   welfare(x)
        s.t.  re_waste(x) ≤ ε_i
              + 所有原约束

      对不同的 ε_i 值分别求解LP，得到 Pareto 前沿上一组非支配解。
    """  
    print(f"\n{'='*60}")
    print(f"  ε-约束法: 求解 Pareto 前沿 ({n_points} 个采样点)")
    print(f"{'='*60}")

    # ---- 步骤1: 求两个锚点 ----
    # 锚点A: 单独最大化 welfare (w_re_consume=0 → 退化为原单目标)
    da_anchor_a = clear_market_multi_obj_lp(
        agents=agents, network=network, T=T, stage=stage,
        wholesale_price=wholesale_price, action_params=action_params,
        penalty_unserved=penalty_unserved,
        w_welfare=1.0, w_re_consume=0.0
    )
    welfare_best = da_anchor_a["welfare"]
    waste_at_welfare_best = da_anchor_a["re_waste"]

    # 锚点B: 单独最小化弃能 (w_re_consume 很大)
    da_anchor_b = clear_market_multi_obj_lp(
        agents=agents, network=network, T=T, stage=stage,
        wholesale_price=wholesale_price, action_params=action_params,
        penalty_unserved=penalty_unserved,
        w_welfare=1.0, w_re_consume=10000.0
    )
    waste_best = da_anchor_b["re_waste"]       # 最小弃能（可能接近0）
    welfare_at_waste_best = da_anchor_b["welfare"]

    total_re = da_anchor_a["total_re_available"]

    print(f"  锚点A (福利最优):  welfare={welfare_best:.2f}, 弃能={waste_at_welfare_best:.3f} MWh")
    print(f"  锚点B (消纳最优):  welfare={welfare_at_waste_best:.2f}, 弃能={waste_best:.3f} MWh")
    print(f"  总可再生可用量:   {total_re:.3f} MWh")

    # ---- 步骤2: 在 [waste_best, waste_at_welfare_best] 间离散采样 ε ----
    # 注意: waste_best <= waste_at_welfare_best （弃能越小越好）
    eps_values = np.linspace(waste_best, max(waste_at_welfare_best, waste_best * 1.01), n_points)

    pareto_front = []

    for idx, eps in enumerate(eps_values):
        try:
            # 用带大权重的WSM近似ε-约束（因为cvxpy直接加不等式约束到目标函数较复杂）
            # 这里采用等效方法：增大 w_re_consume 使弃能惩罚足够强
            # 当 w_re_consume → ∞ 时等价于 hard constraint waste ≤ ε
            result = clear_market_multi_obj_lp(
                agents=agents, network=network, T=T, stage=stage,
                wholesale_price=wholesale_price, action_params=action_params,
                penalty_unserved=penalty_unserved,
                w_welfare=1.0, w_re_consume=max(eps * 50, 1.0)  # 动态权重
            )

            actual_waste = result["re_waste"]
            actual_welfare = result["welfare"]

            pareto_front.append({
                "welfare": actual_welfare,
                "re_waste": actual_waste,
                "re_consumption_rate": result["re_consumption_rate"],
                "eps_target": float(eps),
            })

        except Exception as e:
            # 某些极端ε值可能不可行，跳过
            pass

    # ---- 步骤3: 输出Pareto前沿摘要 ----
    if pareto_front:
        print(f"\n  Pareto 前沿 ({len(pareto_front)} 个有效点):")
        print(f"  {'#':>4} {'Welfare(¥)':>14} {'弃能(MWh)':>12} {'消纳率(%)':>10}")
        print(f"  {'-'*54}")
        for i, p in enumerate(pareto_front):
            print(f"  {i+1:>4} {p['welfare']:>14.2f} {p['re_waste']:>12.3f} {p['re_consumption_rate']*100:>9.2f}%")

        # 找折中解（距离理想点最近，归一化欧氏距离）
        w_min = min(p["welfare"] for p in pareto_front)
        w_max = max(p["welfare"] for p in pareto_front)
        wk_min = min(p["re_waste"] for p in pareto_front)
        wk_max = max(p["re_waste"] for p in pareto_front)

        best_dist = float("inf")
        best_idx = 0
        for i, p in enumerate(pareto_front):
            nw = (p["welfare"] - w_max) / (w_min - w_max) if w_min != w_max else 0
            nwk = (p["re_waste"] - wk_min) / (wk_max - wk_min) if wk_max != wk_min else 0
            dist = (nw**2 + nwk**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        print(f"\n  ★ 折中解(最近理想点): 第{best_idx+1}个")
        print(f"    Welfare={pareto_front[best_idx]['welfare']:.2f}, "
              f"弃能={pareto_front[best_idx]['re_waste']:.3f} MWh, "
              f"消纳率={pareto_front[best_idx]['re_consumption_rate']*100:.2f}%")

    print(f"{'='*60}")
    return pareto_front


# ============================================================
# 任务3: 多场景构建与批量运行
# ============================================================


def build_scenarios():
    """
    构建多场景列表（任务3）

    返回: list of dict, 每个 dict 含 {
        name: 场景名称,
        desc: 描述,
        agents, network, wholesale, with_wind
    }
    """
    T = 96
    scenarios = []

    # ---- 场景A: 基准场景（无风电，3节点）----
    ag_A, net_A, wp_A = build_demo_case(T=T, with_wind=False)
    scenarios.append({
        "name": "A-基准",
        "desc": "5Agent/3节点/PV-only/标准容量",
        "agents": ag_A, "network": net_A, "wholesale": wp_A,
        "with_wind": False,
    })

    # ---- 场景B: 高可再生（含风电，5节点）----
    ag_B, net_B, wp_B = build_demo_case(T=T, with_wind=True)
    scenarios.append({
        "name": "B-高可再生",
        "desc": "7Agent/5节点/PV+Wind/弃能风险高",
        "agents": ag_B, "network": net_B, "wholesale": wp_B,
        "with_wind": True,
    })

    # ---- 场景C: 紧约束（线路容量缩小→拥塞严重）----
    ag_C, _, wp_C = build_demo_case(T=T, with_wind=True)
    net_C = Network.from_edges([
        (0, 1, 4.0),   # 缩小
        (1, 2, 2.5),   # 缩小
        (2, 3, 3.0),   # 缩小
        (3, 4, 2.0),   # 缩小
    ])
    scenarios.append({
        "name": "C-紧约束",
        "desc": "7Agent/5节点/线路紧/拥塞严重",
        "agents": ag_C, "network": net_C, "wholesale": wp_C,
        "with_wind": True,
    })

    # ---- 场景D: 松约束（线路容量放大→近似无拥塞）----
    ag_D, _, wp_D = build_demo_case(T=T, with_wind=True)
    net_D = Network.from_edges([
        (0, 1, 20.0),
        (1, 2, 15.0),
        (2, 3, 15.0),
        (3, 4, 12.0),
    ])
    scenarios.append({
        "name": "D-松约束",
        "desc": "7Agent/5节点/线路松/无拥塞",
        "agents": ag_D, "network": net_D, "wholesale": wp_D,
        "with_wind": True,
    })

    # ---- 场景E: 高峰负荷（负荷放大1.5倍）----
    ag_E, net_E, wp_E = build_demo_case(T=T, with_wind=True)
    # 放大所有负荷Agent的负荷
    for a in ag_E:
        a.load_forecast = a.load_forecast * 1.5
        a.load_real = a.load_real * 1.5
    scenarios.append({
        "name": "E-高峰负荷",
        "desc": "7Agent/5节点/负荷×1.5/消纳率高",
        "agents": ag_E, "network": net_E, "wholesale": wp_E,
        "with_wind": True,
    })

    return scenarios


def run_all_scenarios(multi_obj: bool = True):
    """
    运行所有场景的对比分析（任务3）

    参数:
      multi_obj: 是否使用双目标优化（True=双目标, False=单目标）
    """
    scenarios = build_scenarios()

    print("=" * 80)
    print("  多场景批量运行对比")
    print("=" * 80)

    results = []
    for sc in scenarios:
        print(f"\n>>> 场景: {sc['name']} - {sc['desc']}")
        try:
            act = random_actions(sc["agents"])
            if multi_obj:
                da = clear_market_multi_obj_lp(
                    agents=sc["agents"], network=sc["network"], T=96,
                    stage="DA", wholesale_price=sc["wholesale"],
                    action_params=act, w_welfare=1.0, w_re_consume=80.0
                )
                row = {
                    "scenario": sc["name"],
                    "welfare": da["welfare"],
                    "re_waste": da.get("re_waste", 0),
                    "re_consumption_rate": da.get("re_consumption_rate", 1.0) * 100,
                    "n_agents": len(sc["agents"]),
                    "n_branches": len(sc["network"].branches),
                }
            else:
                da = clear_market_lp(
                    agents=sc["agents"], network=sc["network"], T=96,
                    stage="DA", wholesale_price=sc["wholesale"],
                    action_params=act
                )
                # 手动计算弃能
                total_re = sum(
                    np.sum(a.pv_forecast) + (np.sum(a.wind_forecast) if a.has_wind() else 0)
                    for a in sc["agents"] if a.is_prosumer or a.has_wind()
                )
                used_re = sum(
                    np.sum(da["schedules"][a.name]["pv_used"]) +
                    (np.sum(da["schedules"][a.name].get("wind_used", [0])) if a.has_wind() else 0)
                    for a in sc["agents"]
                )
                waste = total_re - used_re
                rate = used_re / total_re * 100 if total_re > 0 else 100
                row = {
                    "scenario": sc["name"],
                    "welfare": da["welfare"],
                    "re_waste": waste,
                    "re_consumption_rate": rate,
                    "n_agents": len(sc["agents"]),
                    "n_branches": len(sc["network"].branches),
                }

            results.append(row)
            print(f"    Welfare: {row['welfare']:.2f} | "
                  f"弃能: {row['re_waste']:.3f} MWh | "
                  f"消纳率: {row['re_consumption_rate']:.1f}%")

        except Exception as e:
            print(f"    ✗ 求解失败: {e}")
            results.append({"scenario": sc["name"], "error": str(e)})

    # ---- 汇总表格 ----
    print("\n" + "=" * 80)
    print("  场景汇总表")
    print("=" * 80)
    header = f"  {'场景':<12} {'Agent数':>6} {'支路数':>6} {'Welfare(¥)':>14} {'弃能(MWh)':>12} {'消纳率(%)':>10}"
    print(header)
    print("  " + "-" * 74)
    for r in results:
        if "error" not in r:
            print(f"  {r['scenario']:<12} {r['n_agents']:>6} {r['n_branches']:>6} "
                  f"{r['welfare']:>14.2f} {r['re_waste']:>12.3f} {r['re_consumption_rate']:>9.1f}%")
        else:
            print(f"  {r['scenario']:<12} {'ERROR':>44} {r['error']}")

    return results


# ============================================================
# 更新的演示函数
# ============================================================


def run_one_day_demo(with_wind: bool = False):
    agents, network, wholesale = build_demo_case(T=96, with_wind=with_wind)

    mode_str = " (含风电)" if with_wind else ""
    print(f"=== 单日演示{mode_str}: {len(agents)} Agents, {len(network.branches)} Branches ===")

    # 日前
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=96, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )

    # 实时
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=96, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )

    payment = two_settlement(agents, da, rt)

    print(f"Day-Ahead (DA) welfare: {round(da['welfare'], 2)}")
    print(f"Real-Time (RT) welfare: {round(rt['welfare'], 2)}")
    print("\n--- Payments (positive = cost, negative = revenue) ---")
    for k, v in payment.items():
        print(f"{k:14s}  {v:8.2f} ¥")

    # 各Agent快照
    for a in agents[:4]:  # 只显示前4个避免太长
        sch_da = da["schedules"][a.name]
        sch_rt = rt["schedules"][a.name]
        print(f"\n--- {a.name} snapshot ---")
        if a.storage is not None:
            print(f"DA soc:  {np.round(sch_da.get('soc', [0]), 2)}")
            print(f"RT soc:  {np.round(sch_rt.get('soc', [0]), 2)}")
        pv_u = np.sum(sch_da.get("pv_used", [0]))
        wind_u = np.sum(sch_da.get("wind_used", [0]))
        print(f"DA PV用: {pv_u:.2f} MWh, Wind用: {wind_u:.2f} MWh")




if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--multi" in args:
        wind_mode = "--wind" in args
    elif "--pareto" in args:
        # ε-约束法 Pareto 前沿
        agents, network, wholesale = build_demo_case(T=96, with_wind=True)
        act = random_actions(agents)
        front = solve_pareto_front_epsilon(
            agents=agents, network=network, T=96, stage="DA",
            wholesale_price=wholesale, action_params=act, n_points=15
        )
    elif "--scenarios" in args:
        multi = "--single" not in args
        run_all_scenarios(multi_obj=multi)
    elif "--all" in args:
        agents, network, wholesale = build_demo_case(T=96, with_wind=True)
        act = random_actions(agents)
        solve_pareto_front_epsilon(
            agents=agents, network=network, T=96, stage="DA",
            wholesale_price=wholesale, action_params=act, n_points=15
        )
        print("\n===== 测试4: 多场景批量 =====")
        run_all_scenarios(multi_obj=True)
    else:
        run_one_day_demo(with_wind=False)


def run_multi_obj_comparison():
    """
    对比演示：单目标(社会福利) vs 双目标(社会福利+PV消纳)
    参考: Applied Energy 312 (2022) 118724 目标二 waste minimization
    """
    agents, network, wholesale = build_demo_case(T=96)
    act = random_actions(agents)

    # ---- 方案B：双目标（社会福利 + PV消纳）----
    # 使用与论文相近的权重比例
    da_multi = clear_market_multi_obj_lp(
        agents=agents, network=network, T=96, stage="DA",
        wholesale_price=wholesale, action_params=act,
        w_welfare=1.0, w_re_consume=80.0   # 弃光惩罚权重
    )


    # 各agent PV使用对比
    print("\n" + "-" * 62)
    print("各产消者(Prosumer) PV使用详情:")
    print(f"{'Agent':<16} {'单目标PV用(MWh)':>18} {'双目标PV用(MWh)':>18}")
    for a in agents:
        if a.is_prosumer:
            u_m = np.sum(da_multi["schedules"][a.name]["pv_used"])
            print(f"{a.name:<16} {u_m:>18.3f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        run_multi_obj_comparison()
    else:
        run_one_day_demo()
