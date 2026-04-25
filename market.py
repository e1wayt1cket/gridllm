import numpy as np
from typing import Dict, List, Optional
from models import Agent, MarketConfig
from grid import day_ahead_price_china

# ---------- 辅助函数：从历史结果中提取每个agent的节点LMP ----------
def extract_lmp_history(result: Dict, agents: List[Agent]) -> Dict[str, np.ndarray]:
    """
    从一次 market clearing 结果中，
    为每个 agent 提取其所在 bus 的全时段 LMP 序列 (T,)
    """
    lmp = result["lmp"]  # shape (T, n_bus)
    lmp_dict = {}
    for a in agents:
        lmp_dict[a.name] = lmp[:, a.bus].copy()
    return lmp_dict


# ---------- 报价策略 ----------
def random_actions(agents: List[Agent], config: MarketConfig, T: int = 96) -> Dict:
    """随机报价（分时段）"""
    actions = {}
    for a in agents:
        if a.is_prosumer:
            actions[a.name] = {
                "bid_mult": np.random.uniform(*config.bid_mult_range, size=T),
                "offer_adder": np.random.uniform(*config.offer_adder_range, size=T)
            }
        else:
            actions[a.name] = {
                "bid_mult": np.random.uniform(*config.bid_mult_range, size=T)
            }
    return actions


def best_response_bidding(
    agents: List[Agent],
    config: MarketConfig,
    market_history: Optional[Dict] = None,
    T: int = 96
) -> Dict:
    """
    最佳响应报价（分时段）
    如果有 market_history['lmp_per_agent'][agent_name] 则使用节点电价，
    否则退化为外部日前电价。
    """
    base_price = day_ahead_price_china(T)   # fallback 外部电价

    actions = {}
    for a in agents:
        # ---- 确定该智能体使用的全时段电价 (T,) ----
        if (market_history and 'lmp_per_agent' in market_history
                and a.name in market_history['lmp_per_agent']):
            my_lmp = market_history['lmp_per_agent'][a.name]   # 历史节点电价
        else:
            my_lmp = base_price

        if a.is_prosumer:
            bid_mult = np.full(T, np.mean(config.bid_mult_range))
            offer_adder = np.full(T, np.mean(config.offer_adder_range))
            for t in range(T):
                if my_lmp[t] > a.offer_cost + 50:          # 高价 → 多卖少买
                    bid_mult[t] = np.clip(bid_mult[t] - 0.1, *config.bid_mult_range)
                    offer_adder[t] = np.clip(offer_adder[t] - 10, *config.offer_adder_range)
                elif my_lmp[t] < a.bid_value - 50:         # 低价 → 多买少卖
                    bid_mult[t] = np.clip(bid_mult[t] + 0.1, *config.bid_mult_range)
                    offer_adder[t] = np.clip(offer_adder[t] + 10, *config.offer_adder_range)
            actions[a.name] = {"bid_mult": bid_mult, "offer_adder": offer_adder}
        else:
            bid_mult = np.full(T, np.mean(config.bid_mult_range))
            for t in range(T):
                if my_lmp[t] > a.bid_value * 0.9:
                    bid_mult[t] = np.clip(bid_mult[t] - 0.1, *config.bid_mult_range)
                elif my_lmp[t] < a.bid_value * 0.7:
                    bid_mult[t] = np.clip(bid_mult[t] + 0.1, *config.bid_mult_range)
            actions[a.name] = {"bid_mult": bid_mult}
    return actions


def adaptive_bidding(
    agents: List[Agent],
    config: MarketConfig,
    strategy: str = "best_response",
    market_history: Optional[Dict] = None,
    T: int = 96
) -> Dict:
    """策略调度器"""
    if strategy == "random":
        return random_actions(agents, config, T)
    elif strategy == "best_response":
        return best_response_bidding(agents, config, market_history, T)
    else:
        raise ValueError(f"未知策略类型: {strategy}")


# ---------- 两阶段结算 ----------
def two_settlement(agents: List[Agent], da: Dict, rt: Dict) -> Dict[str, float]:
    T_da = len(da["price"])
    payments = {}
    for a in agents:
        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
    return payments


# ---------- 市场出清核心 ----------
def clear_market(
    agents: List[Agent],
    T: int,
    stage: str,
    action_params: Dict,
    config: MarketConfig,
) -> Dict:
    """
    市场出清（时序 OPF，使用 Gurobi DC-OPF）
    """
    from grid import build_base_network
    from dispatch import solve_dc_opf_gurobi, StorageConstraints

    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    lmp = np.zeros((T, n_buses))
    wholesale = day_ahead_price_china(T)

    schedules = {}
    for a in agents:
        schedules[a.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'storage_mode': ['idle'] * T,
        }
    schedules['GRID'] = {'g_grid': np.zeros(T)}

    total_welfare = 0.0
    total_re_available = 0.0
    total_re_used = 0.0

    for a in agents:
        if stage == "DA":
            total_re_available += float(np.sum(a.pv_forecast))
            if a.has_wind and a.wind_forecast is not None:
                total_re_available += float(np.sum(a.wind_forecast))
        else:
            total_re_available += float(np.sum(a.pv_real))
            if a.has_wind and a.wind_real is not None:
                total_re_available += float(np.sum(a.wind_real))

    prev_soc: Dict[str, float] = {}
    prev_power: Dict[str, tuple] = {}

    for t in range(T):
        success, lmp_t, welfare_t, agent_res, p_grid = solve_dc_opf_gurobi(
            base_net, agents, t, stage, prev_soc, wholesale[t],
            action_params, config, prev_power
        )

        if not success:
            print(f"⚠️ 时段 {t} 失败，沿用上一时段")
            if t > 0:
                lmp[t] = lmp[t-1]
                for a in agents:
                    for key in ['p_buy', 'p_sell', 'served', 'unserved',
                               'pv_used', 'wind_used', 'p_ch', 'p_dis']:
                        schedules[a.name][key][t] = schedules[a.name][key][t-1]
                    schedules[a.name]['storage_mode'][t] = schedules[a.name]['storage_mode'][t-1]
                    if a.storage and a.name in prev_soc:
                        schedules[a.name]['soc'][t] = prev_soc[a.name]
            continue

        lmp[t] = lmp_t
        schedules['GRID']['g_grid'][t] = p_grid
        total_welfare += welfare_t

        for a in agents:
            sched = schedules[a.name]
            res = agent_res[a.name]
            sched['served'][t] = res['served']
            sched['pv_used'][t] = res['pv_used']
            sched['wind_used'][t] = res['wind_used']
            sched['storage_mode'][t] = res['storage_mode']

            if a.storage:
                soc0 = prev_soc.get(a.name, a.storage.soc0) if t > 0 else a.storage.soc0
                ch, dis, new_soc = StorageConstraints.execute_dispatch(
                    a.storage, soc0, res['p_ch'], res['p_dis'], dt=0.25
                )
                sched['p_ch'][t] = ch
                sched['p_dis'][t] = dis
                sched['soc'][t] = new_soc
                prev_soc[a.name] = new_soc
                prev_power[a.name] = (ch, dis)
            else:
                sched['p_ch'][t] = 0.0
                sched['p_dis'][t] = 0.0

            load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
            net_gen = res['pv_used'] + res['wind_used'] + sched['p_dis'][t]
            net_con = res['served'] + sched['p_ch'][t]
            if net_gen > net_con:
                sched['p_sell'][t] = net_gen - net_con
                sched['p_buy'][t] = 0.0
            else:
                sched['p_buy'][t] = net_con - net_gen
                sched['p_sell'][t] = 0.0
            sched['unserved'][t] = load_val - res['served']
            total_re_used += res['pv_used'] + res['wind_used']

    re_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    return {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_waste": total_re_available - total_re_used,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_available,
    }