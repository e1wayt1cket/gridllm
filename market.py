# market.py
"""
市场出清主逻辑（时序 OPF，仅使用 Gurobi DC-OPF）
包含报价策略函数：random, best_response, adaptive, lmp_based
"""
import copy
import numpy as np
from typing import Dict, Tuple, List, Optional

from models import MarketConfig, Agent
from grid import build_base_network, day_ahead_price_china
from dispatch import (
    StorageConstraints,
    solve_dc_opf_gurobi,
)


# ================= 报价策略部分 =================
def random_actions(agents: List[Agent], config: MarketConfig) -> Dict[str, Dict]:
    """随机报价策略：购电乘子和售电加价在允许范围内随机取值"""
    actions = {}
    for a in agents:
        if a.is_prosumer:
            actions[a.name] = {
                "bid_mult": float(np.random.uniform(*config.bid_mult_range)),
                "offer_adder": float(np.random.uniform(*config.offer_adder_range))
            }
        else:
            actions[a.name] = {
                "bid_mult": float(np.random.uniform(*config.bid_mult_range))
            }
    return actions


def best_response_bidding(
    agents: List[Agent],
    config: MarketConfig,
    market_history: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    最佳响应报价策略。
    根据市场历史价格及波动性调整购电/售电意愿。
    """
    if market_history:
        avg_price = market_history.get('avg_price', 400.0)
        volatility = market_history.get('volatility', 0.2)
    else:
        avg_price = 400.0
        volatility = 0.2

    actions = {}
    for a in agents:
        if a.is_prosumer:
            if avg_price > a.offer_cost + 50:
                # 高价区：多卖少买
                bid_mult = max(config.bid_mult_range[0],
                               np.mean(config.bid_mult_range) - volatility * 0.2)
                offer_adder = min(config.offer_adder_range[1],
                                  np.mean(config.offer_adder_range) + volatility * 10)
            elif avg_price < a.bid_value - 50:
                # 低价区：多买少卖
                bid_mult = min(config.bid_mult_range[1],
                               np.mean(config.bid_mult_range) + volatility * 0.3)
                offer_adder = max(config.offer_adder_range[0],
                                  np.mean(config.offer_adder_range) - volatility * 10)
            else:
                # 中性
                bid_mult = np.mean(config.bid_mult_range)
                offer_adder = np.mean(config.offer_adder_range)

            actions[a.name] = {
                "bid_mult": float(bid_mult),
                "offer_adder": float(offer_adder)
            }
        else:
            # 纯负荷：根据电价调整购买意愿
            if avg_price > a.bid_value * 0.9:
                bid_mult = max(config.bid_mult_range[0],
                               np.mean(config.bid_mult_range) - volatility * 0.2)
            elif avg_price < a.bid_value * 0.7:
                bid_mult = min(config.bid_mult_range[1],
                               np.mean(config.bid_mult_range) + volatility * 0.3)
            else:
                bid_mult = np.mean(config.bid_mult_range)
            actions[a.name] = {
                "bid_mult": float(bid_mult)
            }
    return actions


def lmp_based_bidding(
    agents: List[Agent],
    config: MarketConfig,
    market_history: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    基于LMP（节点边际电价）的报价策略。
    根据预测的LMP差异和自身位置调整报价策略。
    """
    if market_history:
        # 获取LMP信息
        lmp_matrix = market_history.get('lmp', None)
        avg_lmp = market_history.get('avg_price', 400.0)
        
        # 如果有LMP矩阵，可以根据节点位置和LMP差异调整策略
        if lmp_matrix is not None and lmp_matrix.size > 0:
            # 计算平均LMP和LMP波动性
            avg_node_lmps = np.mean(lmp_matrix, axis=0)  # 每个节点的时间平均LMP
            node_lmp_std = np.std(lmp_matrix, axis=0)    # 每个节点的LMP波动
            
            # 计算全网LMP波动性
            overall_volatility = np.mean(node_lmp_std)
        else:
            avg_node_lmps = np.full(len(agents), 400.0)  # 默认LMP
            node_lmp_std = np.full(len(agents), 0.2)
            overall_volatility = 0.2
            avg_lmp = 400.0
    else:
        # 没有历史数据时使用默认值
        avg_node_lmps = np.full(len(agents), 400.0)
        node_lmp_std = np.full(len(agents), 0.2)
        overall_volatility = 0.2
        avg_lmp = 400.0

    actions = {}
    for i, a in enumerate(agents):
        # 获取该智能体所在节点的LMP信息
        agent_node_lmp = avg_node_lmps[a.bus] if a.bus < len(avg_node_lmps) else avg_lmp
        agent_node_volatility = node_lmp_std[a.bus] if a.bus < len(node_lmp_std) else overall_volatility
        
        if a.is_prosumer:
            # 对于产消者，根据所在节点的LMP和波动性调整报价
            if agent_node_lmp > a.offer_cost + 50:
                # 该节点LMP高：积极出售，保守购买
                bid_mult = max(config.bid_mult_range[0],
                               np.mean(config.bid_mult_range) - agent_node_volatility * 0.3)
                offer_adder = min(config.offer_adder_range[1],
                                  np.mean(config.offer_adder_range) + agent_node_volatility * 15)
            elif agent_node_lmp < a.bid_value - 50:
                # 该节点LMP低：积极购买，保守出售
                bid_mult = min(config.bid_mult_range[1],
                               np.mean(config.bid_mult_range) + agent_node_volatility * 0.4)
                offer_adder = max(config.offer_adder_range[0],
                                  np.mean(config.offer_adder_range) - agent_node_volatility * 15)
            else:
                # 中等LMP水平：保持中性
                bid_mult = np.mean(config.bid_mult_range)
                offer_adder = np.mean(config.offer_adder_range)
                
            # 进一步根据LMP波动性调整：波动大时更保守
            if agent_node_volatility > 0.3:  # 高波动
                # 减少激进报价
                bid_mult = np.clip(bid_mult, 
                                   np.mean(config.bid_mult_range),
                                   config.bid_mult_range[1])
                offer_adder = np.clip(offer_adder,
                                      config.offer_adder_range[0],
                                      np.mean(config.offer_adder_range))
            
            actions[a.name] = {
                "bid_mult": float(bid_mult),
                "offer_adder": float(offer_adder)
            }
        else:
            # 纯负荷：根据所在节点LMP调整购买意愿
            if agent_node_lmp > a.bid_value * 0.9:
                # 节点电价过高：降低购买意愿
                bid_mult = max(config.bid_mult_range[0],
                               np.mean(config.bid_mult_range) - agent_node_volatility * 0.2)
            elif agent_node_lmp < a.bid_value * 0.7:
                # 节点电价较低：提高购买意愿
                bid_mult = min(config.bid_mult_range[1],
                               np.mean(config.bid_mult_range) + agent_node_volatility * 0.3)
            else:
                bid_mult = np.mean(config.bid_mult_range)
            
            actions[a.name] = {
                "bid_mult": float(bid_mult)
            }
    return actions


def adaptive_bidding(
    agents: List[Agent],
    config: MarketConfig,
    strategy: str = "best_response"
) -> Dict[str, Dict]:
    """自适应报价策略调度器，目前支持 random, best_response, lmp_based 三种"""
    if strategy == "random":
        return random_actions(agents, config)
    elif strategy == "best_response":
        return best_response_bidding(agents, config)
    elif strategy == "lmp_based":
        return lmp_based_bidding(agents, config)
    else:
        raise ValueError(f"未知策略类型: {strategy}，可选: random, best_response, lmp_based")


# ================= 两阶段结算 =================
def two_settlement(agents: List[Agent], da: Dict, rt: Dict) -> Dict[str, float]:
    """日前与实时市场的两阶段结算（按日前价结算日前计划量，实时偏差按实时价结算）"""
    T = len(da["price"])
    payments = {}
    for a in agents:
        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
    return payments


# ================= 市场出清核心 =================
def clear_market(
    agents: List[Agent],
    T: int,
    stage: str,
    action_params: Dict[str, Dict],
    config: MarketConfig,
) -> Dict:
    """
    市场出清（时序 OPF）。
    - 使用 Gurobi 求解每个时段的 DC-OPF
    - 目标为社会福利最大化，自动获得节点 LMP
    - 外部循环更新储能 SOC
    """
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    lmp = np.zeros((T, n_buses))
    wholesale = day_ahead_price_china(T)

    # 初始化各智能体调度记录
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

    # 计算可用可再生能源总量
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
    prev_power: Dict[str, Tuple[float, float]] = {}   # (prev_ch, prev_dis)

    # 逐时段求解
    for t in range(T):
        success, lmp_t, welfare_t, agent_res, p_grid = solve_dc_opf_gurobi(
            base_net, agents, t, stage, prev_soc, wholesale[t],
            action_params, config, prev_power
        )

        if not success:
            print(f"⚠️ 时段 {t} OPF 失败，沿用上一时段结果")
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

            # 储能：物理约束校验并更新 SOC
            if a.storage:
                current_soc = prev_soc.get(a.name, a.storage.soc0) if t > 0 else a.storage.soc0
                actual_ch, actual_dis, new_soc = StorageConstraints.execute_dispatch(
                    a.storage, current_soc, res['p_ch'], res['p_dis'], dt=0.25
                )
                sched['p_ch'][t] = actual_ch
                sched['p_dis'][t] = actual_dis
                sched['soc'][t] = new_soc
                prev_soc[a.name] = new_soc
                prev_power[a.name] = (actual_ch, actual_dis)
            else:
                sched['p_ch'][t] = 0.0
                sched['p_dis'][t] = 0.0

            # 计算净购/售电量
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

    re_consumption_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    re_waste = total_re_available - total_re_used

    return {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_waste": re_waste,
        "re_consumption_rate": re_consumption_rate,
        "total_re_available": total_re_available,
    }