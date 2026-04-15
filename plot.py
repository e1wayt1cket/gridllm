from typing import Dict, Tuple, List, Optional
import cvxpy as cp
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# If matplotlib triggers C-extension import errors due to NumPy ABI mismatch,
# avoid importing it and use Plotly (pure-python) for visualization instead.
USE_PLOTLY = True
try:
    np_ver = tuple(int(x) for x in np.__version__.split('.')[:2])
except Exception:
    np_ver = (0, 0)
if np_ver[0] >= 2:
    # NumPy 2.x detected — prefer Plotly to avoid C-extension incompatibility
    USE_PLOTLY = True
else:
    # Try to import matplotlib but keep fallback
    try:
        import matplotlib.pyplot as plt
        USE_PLOTLY = False
    except Exception:
        USE_PLOTLY = True
        plt = None

# ================================
# 以下复制原代码中的数据结构与函数
# 保持模拟逻辑完全一致
# ================================

np.random.seed(1)


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
    stage: str,
    wholesale_price: np.ndarray,
    action_params: Dict[str, Dict],
    penalty_unserved: float = 500.0
) -> Dict:
    p_buy = {}
    p_sell = {}
    served = {}
    unserved = {}
    pv_used = {}
    p_ch = {}
    p_dis = {}
    soc = {}

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

    if stage == "DA":
        load = {a.name: a.load_forecast for a in agents}
        pv = {a.name: a.pv_forecast for a in agents}
    else:
        load = {a.name: a.load_real for a in agents}
        pv = {a.name: a.pv_real for a in agents}

    for a in agents:
        constraints += [served[a.name] + unserved[a.name] == load[a.name]]

    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]

    for a in agents:
        if a.storage is None:
            continue
        st = a.storage
        constraints += [p_ch[a.name] <= st.p_ch_max]
        constraints += [p_dis[a.name] <= st.p_dis_max]
        constraints += [soc[a.name][0] == st.soc0 + st.eta_ch * \
            p_ch[a.name][0] - (1.0 / st.eta_dis) * p_dis[a.name][0]]
        for t in range(1, T):
            constraints += [soc[a.name][t] == soc[a.name][t - 1] + st.eta_ch *
                            p_ch[a.name][t] - (1.0 / st.eta_dis) * p_dis[a.name][t]]
        constraints += [soc[a.name] >= st.soc_min, soc[a.name] <= st.soc_max]

    for t in range(T):
        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t]
                                          for a in agents]) + cp.sum([p_sell[a.name][t] for a in agents])
        total_demand = cp.sum([served[a.name][t] for a in agents]) + \
            cp.sum([p_buy[a.name][t] for a in agents])
        for a in agents:
            if a.storage is not None:
                total_supply += p_dis[a.name][t]
                total_demand += p_ch[a.name][t]
        constraints += [total_supply == total_demand]

    for a in agents:
        for t in range(T):
            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]
            if a.storage is not None:
                lhs_supply += p_dis[a.name][t]
            rhs_use = served[a.name][t] + p_sell[a.name][t]
            if a.storage is not None:
                rhs_use += p_ch[a.name][t]
            constraints += [lhs_supply == rhs_use]

    buses = sorted(set(a.bus for a in agents))
    for t in range(T):
        net_inj_bus = {b: 0 for b in buses}
        for a in agents:
            b = a.bus
            inj = pv_used[a.name][t] + p_sell[a.name][t] - \
                served[a.name][t] - p_buy[a.name][t]
            if a.storage is not None:
                inj += p_dis[a.name][t] - p_ch[a.name][t]
            net_inj_bus[b] += inj
        if 1 in net_inj_bus and 2 in net_inj_bus:
            flow12 = net_inj_bus[2]
            flow01 = net_inj_bus[1] + net_inj_bus[2]
            constraints += [flow12 <= network.cap12, flow12 >= -network.cap12]
            constraints += [flow01 <= network.cap01, flow01 >= -network.cap01]

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

    problem = cp.Problem(cp.Maximize(welfare), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP not solved: {problem.status}")

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


def two_settlement(
    agents: List[Agent],
    da: Dict,
    rt: Dict
) -> Dict[str, float]:
    T = len(da["price"])
    pay = {a.name: 0.0 for a in agents}
    for a in agents:
        sch_da = da["schedules"][a.name]
        sch_rt = rt["schedules"][a.name]
        da_import = sch_da["p_buy"] - sch_da["p_sell"]
        rt_import = sch_rt["p_buy"] - sch_rt["p_sell"]
        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))
        pay[a.name] = float(da_cost + rt_cost)
    return pay


def build_demo_case(T=24):
    hours = np.arange(T)
    wholesale = 40 + 15 * np.exp(-0.5 * ((hours - 8) / 2.5)**2) + \
        20 * np.exp(-0.5 * ((hours - 19) / 2.5)**2)
    wholesale = np.round(wholesale, 2)

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
    C = Agent(
        name="C(LOAD)", bus=1, is_prosumer=False,
        load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
        bid_value=110.0, offer_cost=999.0,
        storage=None
    )
    D = Agent(
        name="D(LOAD)", bus=1, is_prosumer=False,
        load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
        bid_value=105.0, offer_cost=999.0,
        storage=None
    )
    E = Agent(
        name="E(LOAD)", bus=2, is_prosumer=False,
        load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
        bid_value=102.0, offer_cost=999.0,
        storage=None
    )
    network = Network(cap01=6.5, cap12=3.5)
    return [A, B, C, D, E], network, wholesale


def random_actions(agents: List[Agent]) -> Dict[str, Dict]:
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


def run_simulation():
    """运行完整模拟并返回所有结果"""
    agents, network, wholesale = build_demo_case(T=24)
    act_DA = random_actions(agents)
    da = clear_market_lp(agents, network, 24, "DA", wholesale, act_DA)
    act_RT = random_actions(agents)
    rt = clear_market_lp(agents, network, 24, "RT", wholesale, act_RT)
    payment = two_settlement(agents, da, rt)
    return agents, network, wholesale, da, rt, payment

# ================================
# 可视化部分
# ================================


def visualize_all(agents, wholesale, da, rt, payment, out_dir="plots"):
    """Generate interactive HTML visualizations (using Plotly when available)
    and save raw data backups (.npz) in `out_dir`. This avoids importing
    matplotlib when C-extension/NumPy ABI mismatches exist.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    T = len(wholesale)
    hours = np.arange(T)

    # raw price arrays
    da_price = da["price"]
    rt_price = rt["price"]
    np.savez(
        outp / 'prices_data.npz',
        da_price=da_price,
        rt_price=rt_price,
        wholesale=wholesale)

    # compute flows
    def compute_flows(schedules, agents, T):
        flow01 = np.zeros(T)
        flow12 = np.zeros(T)
        for t in range(T):
            net_inj_bus1 = 0.0
            net_inj_bus2 = 0.0
            for a in agents:
                sch = schedules[a.name]
                inj = sch["pv_used"][t] + sch["p_sell"][t] - \
                    sch["served"][t] - sch["p_buy"][t]
                if a.storage is not None:
                    inj += sch["p_dis"][t] - sch["p_ch"][t]
                if a.bus == 1:
                    net_inj_bus1 += inj
                elif a.bus == 2:
                    net_inj_bus2 += inj
            flow12[t] = net_inj_bus2
            flow01[t] = net_inj_bus1 + net_inj_bus2
        return flow01, flow12

    flow01_da, flow12_da = compute_flows(da["schedules"], agents, T)
    flow01_rt, flow12_rt = compute_flows(rt["schedules"], agents, T)

    # Save raw arrays for downstream plotting
    np.savez(
        outp / 'flows.npz',
        flow01_da=flow01_da,
        flow12_da=flow12_da,
        flow01_rt=flow01_rt,
        flow12_rt=flow12_rt)

    # Attempt to use Plotly (pure Python) for interactive HTML
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=3, subplot_titles=[
            'Market Prices', 'Storage SOC', 'Load Serving: Agent C',
            'Line 0-1 Flow', 'Line 1-2 Flow', 'Net Energy Import',
            'Final Settlement', 'Social Welfare', 'External Grid Supply'
        ])

        # 1 Market Prices
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=wholesale,
                mode='lines',
                name='Wholesale'),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=da_price,
                mode='lines',
                name='DA price'),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=rt_price,
                mode='lines',
                name='RT price'),
            row=1,
            col=1)

        # 2 SOC
        for a in agents:
            if a.storage is not None:
                soc_da = da["schedules"][a.name]["soc"]
                soc_rt = rt["schedules"][a.name]["soc"]
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=soc_da,
                        mode='lines',
                        name=f'{
                            a.name} DA'),
                    row=1,
                    col=2)
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=soc_rt,
                        mode='lines',
                        name=f'{
                            a.name} RT'),
                    row=1,
                    col=2)

        # 3 Load C
        load_c = agents[2].load_real
        served_c_da = da["schedules"]["C(LOAD)"]["served"]
        served_c_rt = rt["schedules"]["C(LOAD)"]["served"]
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=load_c,
                mode='lines',
                name='Actual load'),
            row=1,
            col=3)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=served_c_da,
                mode='lines',
                name='DA served'),
            row=1,
            col=3)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=served_c_rt,
                mode='lines',
                name='RT served'),
            row=1,
            col=3)

        # 4 Flow 0-1
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=flow01_da,
                mode='lines',
                name='DA flow01'),
            row=2,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=flow01_rt,
                mode='lines',
                name='RT flow01'),
            row=2,
            col=1)

        # 5 Flow 1-2
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=flow12_da,
                mode='lines',
                name='DA flow12'),
            row=2,
            col=2)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=flow12_rt,
                mode='lines',
                name='RT flow12'),
            row=2,
            col=2)

        # 6 Net imports summary bars
        agents_plot = [agents[0], agents[2], agents[4]]
        da_net = [np.sum(da["schedules"][a.name]["p_buy"] -
                         da["schedules"][a.name]["p_sell"]) for a in agents_plot]
        rt_net = [np.sum(rt["schedules"][a.name]["p_buy"] -
                         rt["schedules"][a.name]["p_sell"]) for a in agents_plot]
        x = [a.name for a in agents_plot]
        fig.add_trace(
            go.Bar(
                x=x,
                y=da_net,
                name='DA net import'),
            row=2,
            col=3)
        fig.add_trace(
            go.Bar(
                x=x,
                y=rt_net,
                name='RT net import'),
            row=2,
            col=3)

        # 7 Settlement
        names = list(payment.keys())
        amounts = list(payment.values())
        colors = ['green' if x < 0 else 'red' for x in amounts]
        fig.add_trace(
            go.Bar(
                x=names,
                y=amounts,
                marker_color=colors,
                name='Payment'),
            row=3,
            col=1)

        # 8 Welfare
        fig.add_trace(go.Bar(x=['DA', 'RT'], y=[
                      da['welfare'], rt['welfare']], name='Welfare'), row=3, col=2)

        # 9 Grid supply
        grid_da = da["schedules"]["GRID"]["g_grid"]
        grid_rt = rt["schedules"]["GRID"]["g_grid"]
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=grid_da,
                mode='lines',
                name='DA grid supply'),
            row=3,
            col=3)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=grid_rt,
                mode='lines',
                name='RT grid supply'),
            row=3,
            col=3)

        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Two-Settlement Electricity Market Results")
        htmlf = outp / 'market_results.html'
        fig.write_html(str(htmlf), include_plotlyjs='cdn')
        print('Wrote interactive visualization to', htmlf)
        # Try to export static PNG using kaleido (pure-python runtime)
        try:
            pngf = outp / 'market_results.png'
            fig.write_image(str(pngf))
            print('Wrote PNG visualization to', pngf)
            return {
                'html': str(htmlf),
                'png': str(pngf),
                'data_dir': str(outp)}
        except Exception as e:
            print('PNG export failed (kaleido may be missing):', e)
            return {'html': str(htmlf), 'png': None, 'data_dir': str(outp)}

    except Exception as e:
        # Fallback: save raw data only
        print('Plotly not available or failed:', e)
        return {'html': None, 'data_dir': str(outp)}


if __name__ == "__main__":
    agents, network, wholesale, da, rt, payment = run_simulation()
    visualize_all(agents, wholesale, da, rt, payment)
