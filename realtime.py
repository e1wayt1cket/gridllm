# realtime_server.py
import asyncio
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import json

import socket
from contextlib import asynccontextmanager
from pathlib import Path

np.random.seed(1)

# ----------------------------
# 复用原有数据结构（略，与原代码相同）
# ----------------------------


@dataclass
class StorageSpec:
    e_max: float
    p_ch_max: float
    p_dis_max: float
    eta_ch: float
    eta_dis: float
    soc0: float
    soc_min: float
    soc_max: float


@dataclass
class Agent:
    name: str
    bus: int
    is_prosumer: bool
    load_forecast: np.ndarray
    pv_forecast: np.ndarray
    load_real: np.ndarray
    pv_real: np.ndarray
    bid_value: float
    offer_cost: float
    storage: Optional[StorageSpec] = None


@dataclass
class Network:
    cap01: float
    cap12: float


def clear_market_lp(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,                     # "DA" or "RT"
    wholesale_price: np.ndarray,     # (T,) 外部电网边际成本/价格（作为“系统电源”）
    action_params: Dict[str, Dict],  # 各智能体本阶段动作（报价参数）
    penalty_unserved: float = 500.0  # 未满足负荷惩罚（£/MWh）
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

    for a in agents:
        p_buy[a.name] = cp.Variable(T, nonneg=True)
        p_sell[a.name] = cp.Variable(T, nonneg=True)
        served[a.name] = cp.Variable(T, nonneg=True)
        unserved[a.name] = cp.Variable(T, nonneg=True)

        pv_used[a.name] = cp.Variable(T, nonneg=True)

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
                                          for a in agents]) + cp.sum([p_sell[a.name][t] for a in agents])
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
            if a.storage is not None:
                inj += p_dis[a.name][t] - p_ch[a.name][t]
            net_inj_bus[b] += inj

        # 这里假设 bus 分别为 1,2；bus0 为上级电网
        # flow12 = net_inj(bus2)
        # flow01 = net_inj(bus1) + net_inj(bus2)
        if 1 in net_inj_bus and 2 in net_inj_bus:
            flow12 = net_inj_bus[2]
            flow01 = net_inj_bus[1] + net_inj_bus[2]
            constraints += [flow12 <= network.cap12, flow12 >= -network.cap12]
            constraints += [flow01 <= network.cap01, flow01 >= -network.cap01]

    # ----------------------------
    # 目标函数：社会福利最大化
    # ----------------------------
    welfare = 0
    for a in agents:
        # 本阶段动作：例如 bid_multiplier / offer_adder
        ap = action_params.get(a.name, {})
        bid_mult = ap.get("bid_mult", 1.0)           # 买方愿付价倍率
        offer_adder = ap.get("offer_adder", 0.0)     # 卖方报价加成（£/MWh）

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
    # 我们取每时段系统平衡约束对应的 dual 作为系统价格近似（£/MWh）
    # 在上面构造中，每时段系统平衡约束是 constraints 中按顺序追加的一段；我们重新索引会很麻烦
    # 为保持简单：用“外部电网边际成本 + 拥塞溢价近似”，这里直接用 g_grid 的边际成本并加拥塞项（简化）
    # 如果你需要严格 LMP，我可以再给你一个“显式节点平衡+对偶提取”的版本。
    clearing_price = wholesale_price.copy()

    schedules = {}
    for a in agents:
        schedules[a.name] = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }
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
# 滚动时域出清函数（只执行第一个时段）
# ----------------------------


def rolling_clearing(
    agents: List[Agent],
    network: Network,
    current_hour: int,
    horizon: int,
    wholesale_price_func,
) -> Dict:
    """
    求解从 current_hour 开始的 horizon 小时市场出清，返回第一个小时的结果及完整调度（用于SOC更新）
    """
    T = horizon
    # 构造未来T小时的预测数据（用真实值+噪声模拟，实际可用预测）
    # 为简单，这里用各agent的 load_real 和 pv_real 的切片，加上小噪声
    # 实际应用中应使用预测模型
    hours = np.arange(current_hour, current_hour + T) % 24
    wholesale = np.array([wholesale_price_func(h) for h in hours])

    # 深拷贝agent，但替换其load/pv为未来时段的值（仅用于优化）
    temp_agents = []
    for a in agents:
        # 获取未来T小时的实际负荷（循环取模）
        load_win = np.array([a.load_real[h % 24] for h in hours])
        pv_win = np.array([a.pv_real[h % 24] for h in hours]
                          ) if a.is_prosumer else np.zeros(T)
        storage = None
        if a.storage is not None:
            storage = StorageSpec(
                e_max=a.storage.e_max,
                p_ch_max=a.storage.p_ch_max,
                p_dis_max=a.storage.p_dis_max,
                eta_ch=a.storage.eta_ch,
                eta_dis=a.storage.eta_dis,
                soc0=current_soc[a.name],
                soc_min=a.storage.soc_min,
                soc_max=a.storage.soc_max,
            )

        temp_a = Agent(
            name=a.name, bus=a.bus, is_prosumer=a.is_prosumer,
            load_forecast=load_win, pv_forecast=pv_win,
            load_real=load_win, pv_real=pv_win,  # 优化中用实际值（或预测）
            bid_value=a.bid_value, offer_cost=a.offer_cost,
            storage=storage
        )
        temp_agents.append(temp_a)

    # 调用原有的clear_market_lp，但只使用T小时，stage="RT"，动作参数随机（或可学习）
    # 为简化，动作参数使用默认（bid_mult=1.0, offer_adder=0）
    action_params = {a.name: {"bid_mult": 1.0, "offer_adder": 0.0}
                     for a in temp_agents}
    # 为避免循环导入，建议把clear_market_lp复制到本文件（略，见原代码）
    # 这里直接使用原函数（需要将其定义复制到本文件）
    result = clear_market_lp(
        agents=temp_agents, network=network, T=T, stage="RT",
        wholesale_price=wholesale, action_params=action_params
    )
    # 提取第一个时段的结果
    t0_schedules = {}
    for name in result["schedules"]:
        if name == "GRID":
            t0_schedules["GRID"] = {
                "g_grid": result["schedules"]["GRID"]["g_grid"][0]}
        else:
            t0_schedules[name] = {k: v[0]
                                  for k, v in result["schedules"][name].items()}
    # 更新全局SOC（用于下次滚动）
    for a in temp_agents:
        if a.storage is not None:
            new_soc = result["schedules"][a.name]["soc"][0]  # 第一个时段结束的SOC
            current_soc[a.name] = new_soc
    return {
        "hour": current_hour,
        "price": result["price"][0],
        "schedules": t0_schedules,
        "welfare": result["welfare"]
    }

# ----------------------------
# 实时市场模拟器（带WebSocket广播）
# ----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler：启动时创建广播任务，关闭时取消任务。"""
    task = asyncio.create_task(broadcast_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)
clients = set()
current_hour = 0
horizon = 4  # 滚动窗口4小时
global current_soc  # 存储每个产消者当前SOC


def wholesale_price_func(hour):
    # 与原始相同的价格曲线
    h = hour % 24
    return 40 + 15 * np.exp(-0.5 * ((h - 8) / 2.5)**2) + \
        20 * np.exp(-0.5 * ((h - 19) / 2.5)**2)

# 构建案例（与原始相同，但需要从原始代码中提取agents, network）


def build_agents_network():
    T = 24
    hours = np.arange(T)

    def noisy(x, sigma=0.1):
        return np.clip(
            x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)
    base_pv_A = np.clip(6 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    base_pv_B = np.clip(5.5 * np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)
    load_A = 2.0 + 0.5 * np.exp(-0.5 * ((hours - 10) / 3.0)**2)
    load_B = 1.8 + 0.4 * np.exp(-0.5 * ((hours - 15) / 3.5)**2)
    load_C = 4.5 + 1.2 * np.exp(-0.5 * ((hours - 9) / 2.5)**2) + \
        1.5 * np.exp(-0.5 * ((hours - 18) / 2.5)**2)
    load_D = 3.8 + 1.0 * np.exp(-0.5 * ((hours - 8) / 2.7)**2) + \
        1.2 * np.exp(-0.5 * ((hours - 20) / 2.7)**2)

    A = Agent(
        name="A(RES+ESS)",
        bus=1,
        is_prosumer=True,
        load_forecast=noisy(
            load_A,
            0.05),
        pv_forecast=noisy(
            base_pv_A,
            0.12),
        load_real=noisy(
            load_A,
            0.08),
        pv_real=noisy(
            base_pv_A,
            0.18),
        bid_value=90.0,
        offer_cost=5.0,
        storage=StorageSpec(
            e_max=8.0,
            p_ch_max=2.5,
            p_dis_max=2.5,
            eta_ch=0.95,
            eta_dis=0.95,
            soc0=3.0,
            soc_min=0.8,
            soc_max=7.5))
    B = Agent(
        name="B(RES+ESS)",
        bus=1,
        is_prosumer=True,
        load_forecast=noisy(
            load_B,
            0.05),
        pv_forecast=noisy(
            base_pv_B,
            0.12),
        load_real=noisy(
            load_B,
            0.08),
        pv_real=noisy(
            base_pv_B,
            0.18),
        bid_value=88.0,
        offer_cost=6.0,
        storage=StorageSpec(
            e_max=6.0,
            p_ch_max=2.0,
            p_dis_max=2.0,
            eta_ch=0.94,
            eta_dis=0.94,
            soc0=2.5,
            soc_min=0.6,
            soc_max=5.6))
    C = Agent(name="C(LOAD)", bus=1, is_prosumer=False,
              load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
              bid_value=110.0, offer_cost=999.0, storage=None)
    D = Agent(name="D(LOAD)", bus=1, is_prosumer=False,
              load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
              bid_value=105.0, offer_cost=999.0, storage=None)
    E = Agent(name="E(LOAD)", bus=2, is_prosumer=False,
              load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
              bid_value=102.0, offer_cost=999.0, storage=None)
    network = Network(cap01=6.5, cap12=3.5)
    return [A, B, C, D, E], network


agents, network = build_agents_network()
# 初始化全局SOC
current_soc = {}
for a in agents:
    if a.storage:
        current_soc[a.name] = a.storage.soc0

# 注意：需要把 clear_market_lp 函数完整复制到此文件（因篇幅省略，请从原代码复制）
# 下面假设已经定义了 clear_market_lp

# 使用 lifespan 管理广播任务（见文件顶部的 lifespan 实现）


async def broadcast_loop():
    global current_hour
    while True:
        await asyncio.sleep(5)  # 每5秒更新一次
        # 执行滚动优化
        result = rolling_clearing(
            agents,
            network,
            current_hour,
            horizon,
            wholesale_price_func)
        # 构建推送数据
        snapshot = {
            "timestamp": current_hour,
            "price": float(result["price"]),
            "soc": {name: float(current_soc.get(name, 0)) for name in current_soc},
            "grid_supply": float(result["schedules"]["GRID"]["g_grid"]),
            "line_flow": {
                # 需实现
                "flow01": float(compute_line_flow(result["schedules"], agents, 0)),
                "flow12": float(compute_line_flow(result["schedules"], agents, 1))
            },
            "load_served": {
                name: float(result["schedules"][name]["served"]) for name in result["schedules"] if name != "GRID"
            }
        }
        # 广播给所有客户端
        for client in clients.copy():
            try:
                await client.send_json(snapshot)
            except BaseException:
                clients.remove(client)
        current_hour = (current_hour + 1) % 24


def compute_line_flow(schedules, agents, line_id):
    # 计算各母线净注入
    buses = sorted(set(a.bus for a in agents))
    net_inj_bus = {b: 0.0 for b in buses}
    for a in agents:
        b = a.bus
        sched = schedules[a.name]
        inj = (
            sched["pv_used"] +
            sched["p_sell"] -
            sched["served"] -
            sched["p_buy"])
        if "p_dis" in sched:
            inj += sched["p_dis"] - sched["p_ch"]
        net_inj_bus[b] += inj
    # 计算线潮流
    if line_id == 0:  # flow01
        return net_inj_bus.get(1, 0) + net_inj_bus.get(2, 0)
    elif line_id == 1:  # flow12
        return net_inj_bus.get(2, 0)
    else:
        return 0.0


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # 保持连接
    except BaseException:
        clients.remove(websocket)


@app.get("/")
def get():
    # 从脚本目录读取 index.html，避免工作目录不同导致找不到文件
    idx = Path(__file__).parent / "index.html"
    if not idx.exists():
        return HTMLResponse("<h3>index.html 未找到</h3>", status_code=404)
    return HTMLResponse(idx.read_text(encoding='utf-8'))


def find_free_port(start: int = 8000, end: int = 8100) -> int:
    """在指定范围内查找空闲端口；找不到则抛出错误。"""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


if __name__ == "__main__":
    try:
        port = find_free_port(8000, 8100)
        if port != 8000:
            print(f"port 8000 busy, using free port {port}")

        # 使用 127.0.0.1 而不是 localhost，避免DNS解析问题
        url = f"http://127.0.0.1:{port}/"
        print(f"Server starting. Open your browser to: {url}")
        print(f"Alternative: http://localhost:{port}/")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start server: {e}")
