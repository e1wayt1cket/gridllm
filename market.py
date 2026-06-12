# market.py
import numpy as np
from typing import Dict, List, Tuple
from models import Agent, MarketConfig
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints, solve_lindist_opf_batch

# ---------------------------------------------------------------------------
# Shared utilities (used by dispatch.py and market.py)
# ---------------------------------------------------------------------------

def empty_schedules(agents, T):
    """Create a zero-filled schedules dict for the given agents and periods."""
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
    return schedules


def split_power(net_gen, net_con):
    """Return (p_buy, p_sell) given net generation and consumption."""
    if net_gen > net_con:
        return 0.0, net_gen - net_con
    else:
        return net_con - net_gen, 0.0


def random_actions(agents, config, T=96):
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

def best_response_bidding(agents, config, market_history=None, T=96):
    base_price = day_ahead_price_china(T)
    actions = {}
    for a in agents:
        if market_history and 'lmp_per_agent' in market_history and a.name in market_history['lmp_per_agent']:
            my_lmp = market_history['lmp_per_agent'][a.name]
        else:
            my_lmp = base_price
        if a.is_prosumer:
            bid_mult = np.full(T, np.mean(config.bid_mult_range))
            offer_adder = np.full(T, np.mean(config.offer_adder_range))
            for t in range(T):
                if my_lmp[t] > a.offer_cost + 50:
                    bid_mult[t] = np.clip(bid_mult[t] - 0.1, *config.bid_mult_range)
                    offer_adder[t] = np.clip(offer_adder[t] - 10, *config.offer_adder_range)
                elif my_lmp[t] < a.bid_value - 50:
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

_STRATEGIES = {
    "random": (lambda agents, config, T, _: random_actions(agents, config, T)),
    "best_response": (lambda agents, config, T, mh: best_response_bidding(agents, config, mh, T)),
    "mpc": (lambda agents, config, T, mh: _mpc_bidding(agents, config, mh, T)),
}


def _mpc_bidding(agents, config, market_history, T):
    from mpc_storage import mpc_storage_bidding
    return mpc_storage_bidding(agents, config, market_history, T)


def adaptive_bidding(agents, config, strategy="best_response", market_history=None, T=96):
    if strategy.startswith("stackelberg"):
        from stackelberg import stackelberg_bidding, stackelberg_nash
        parts = strategy.split(":", 1)
        if strategy.startswith("stackelberg_nash"):
            actions, history = stackelberg_nash(agents, config, T=T)
            return actions
        if len(parts) > 1:
            leader_name = parts[1]
        else:
            storage_agents = [a for a in agents if a.storage is not None]
            if not storage_agents:
                raise ValueError("No storage agent for Stackelberg leader")
            leader_name = storage_agents[0].name
        actions, info = stackelberg_bidding(agents, config, leader_name, T=T)
        if config.verbose:
            print(f"Stackelberg leader={info['leader']} "
                  f"payoff={info['optimal_payoff']:.1f}")
        return actions

    if strategy in _STRATEGIES:
        return _STRATEGIES[strategy](agents, config, T, market_history)
    raise ValueError(f"unknown strategy: {strategy}")

def two_settlement(agents, da, rt):
    T = len(da["price"])
    payments = {}
    for a in agents:
        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
    return payments

def clear_market(agents, T, stage, action_params, config, storage_units=None):
    base_net = build_base_network(config)
    wholesale = day_ahead_price_china(T)

    # ---- 多时段联合优化（仅 LinDistFlow）----
    if config.opf_mode == "lindistflow":
        result = solve_lindist_opf_batch(base_net, agents, T, stage, config,
                                         action_params, wholesale, storage_units)
        if result is not None:
            return result
        else:
            print("批量模型失败，回退至逐时段求解...")

    # ---- 原有的逐时段求解（DC-OPF 或 LinDistFlow 回退）----
    n_buses = len(base_net.bus)
    lmp = np.zeros((T, n_buses))
    schedules = empty_schedules(agents, T)

    total_welfare = 0.0
    total_re_available = 0.0; total_re_used = 0.0
    total_curtailment = 0.0
    carbon_emissions = 0.0
    total_served_mwh = 0.0
    for a in agents:
        if stage == "DA":
            total_re_available += np.sum(a.pv_forecast)
            if a.has_wind and a.wind_forecast is not None:
                total_re_available += np.sum(a.wind_forecast)
        else:
            total_re_available += np.sum(a.pv_real)
            if a.has_wind and a.wind_real is not None:
                total_re_available += np.sum(a.wind_real)

    prev_soc: Dict[str, float] = {}
    prev_power: Dict[str, Tuple[float, float]] = {}

    for t in range(T):
        success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
            base_net, agents, t, stage, prev_soc, wholesale[t],
            action_params, config, prev_power
        )
        if not success:
            if t > 0:
                lmp[t] = lmp[t-1]
                for a in agents:
                    for key in ['p_buy','p_sell','served','unserved','pv_used','wind_used','p_ch','p_dis']:
                        schedules[a.name][key][t] = schedules[a.name][key][t-1]
                    if a.storage and a.name in prev_soc:
                        schedules[a.name]['soc'][t] = prev_soc[a.name]
            continue

        lmp[t] = lmp_t
        schedules['GRID']['g_grid'][t] = p_grid
        total_welfare += welfare_t
        if p_grid > 0:
            carbon_emissions += config.emission_factor_grid * p_grid * 0.25

        for a in agents:
            sched = schedules[a.name]
            res = agent_res[a.name]
            sched['served'][t] = res['served']
            sched['pv_used'][t] = res['pv_used']
            sched['wind_used'][t] = res['wind_used']

            if a.storage:
                soc0 = prev_soc.get(a.name, a.storage.soc0) if t > 0 else a.storage.soc0
                ch_val, dis_val, new_soc = StorageConstraints.execute_dispatch(
                    a.storage, soc0, res['p_ch'], res['p_dis'], dt=0.25
                )
                sched['p_ch'][t] = ch_val
                sched['p_dis'][t] = dis_val
                sched['soc'][t] = new_soc
                prev_soc[a.name] = new_soc
                prev_power[a.name] = (ch_val, dis_val)
            else:
                sched['p_ch'][t] = 0.0
                sched['p_dis'][t] = 0.0

            load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
            pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
            wind_max = (a.get_wind_forecast()[t] if stage == "DA" else a.get_wind_real()[t]) if a.has_wind else 0.0
            net_gen = res['pv_used'] + res['wind_used'] + sched['p_dis'][t]
            net_con = res['served'] + sched['p_ch'][t]
            sched['p_buy'][t], sched['p_sell'][t] = split_power(net_gen, net_con)
            sched['unserved'][t] = load_val - res['served']
            total_re_used += res['pv_used'] + res['wind_used']
            total_curtailment += (pv_max - res['pv_used']) + (wind_max - res['wind_used'])
            total_served_mwh += res['served']

    re_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    return {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_available,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
    }


def _make_window_agents(agents, t_start, t_end, prev_soc, action_params):
    """Create copies of agents with sliced data arrays and updated storage SOC for
    a rolling-horizon window [t_start, t_end)."""
    from copy import deepcopy
    from models import Agent as AgentCls
    window_agents = []
    for a in agents:
        wa = AgentCls(
            name=a.name, bus=a.bus, is_prosumer=a.is_prosumer,
            load_forecast=a.load_forecast[t_start:t_end].copy(),
            pv_forecast=a.pv_forecast[t_start:t_end].copy(),
            load_real=a.load_real[t_start:t_end].copy() if a.load_real is not None else None,
            pv_real=a.pv_real[t_start:t_end].copy() if a.pv_real is not None else None,
            bid_value=a.bid_value, offer_cost=a.offer_cost,
            wind_forecast=a.wind_forecast[t_start:t_end].copy() if a.has_wind else None,
            wind_real=a.wind_real[t_start:t_end].copy() if a.has_wind else None,
            storage=deepcopy(a.storage) if a.storage else None,
            load_type=a.load_type,
        )
        if wa.storage is not None and wa.name in prev_soc:
            wa.storage.soc0 = prev_soc[wa.name]
        window_agents.append(wa)
    return window_agents


def clear_rt_rolling_mpc(agents, T, action_params_base, config, storage_units=None):
    """Rolling real-time market with multi-period MPC and price forecasting.

    At each step, forecasts prices for the look-ahead window, solves a joint
    multi-period OPF, and commits only the first step's dispatch.
    """
    from price_forecaster import PriceForecaster

    rt_horizon = config.rt_horizon
    rt_step = config.rt_step
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    wholesale = day_ahead_price_china(T)
    forecaster = PriceForecaster(wholesale, mode=config.rt_forecast_mode,
                                  noise_pct=config.rt_forecast_noise_pct)

    lmp_rt = np.zeros((T, n_buses))
    schedules_rt = empty_schedules(agents, T)
    for su in (storage_units or []):
        schedules_rt[su.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'storage_mode': ['idle'] * T,
        }

    prev_soc = {}
    prev_power = {}
    total_welfare_rt = 0.0
    carbon_emissions = 0.0
    total_curtailment = 0.0
    total_served_mwh = 0.0
    total_re_available = 0.0
    total_re_used = 0.0

    window_config = MarketConfig(
        opf_mode=config.opf_mode, verbose=config.verbose,
        lambda_cycle=config.lambda_cycle,
        penalty_unserved=config.penalty_unserved,
        line_capacity_multiplier=config.line_capacity_multiplier,
        emission_factor_grid=config.emission_factor_grid,
        enable_multi_objective=False,
        use_constraint_multi_obj=False,
        storage_terminal_value=config.storage_terminal_value,
        rt_horizon=config.rt_horizon, rt_step=config.rt_step,
    )
    idx = 0
    while idx < T:
        window_end = min(idx + rt_horizon, T)
        window_T = window_end - idx
        forecast = forecaster.forecast(idx, window_T)

        window_agents = _make_window_agents(agents, idx, window_end,
                                            prev_soc, action_params_base)
        window_storage_units = None
        if storage_units:
            from copy import deepcopy
            window_storage_units = []
            for su in storage_units:
                su_copy = deepcopy(su)
                if su.name in prev_soc:
                    su_copy.storage.soc0 = prev_soc[su.name]
                window_storage_units.append(su_copy)

        result = solve_lindist_opf_batch(
            base_net, window_agents, window_T, "RT", window_config,
            action_params_base, forecast, window_storage_units
        )

        n_commit = min(rt_step, window_T)
        if result is not None:
            for d in range(n_commit):
                t_abs = idx + d
                dt = 0.25
                lmp_rt[t_abs] = result["lmp"][d]
                schedules_rt['GRID']['g_grid'][t_abs] = result['schedules']['GRID']['g_grid'][d]

                for a in window_agents:
                    nm = a.name
                    s = schedules_rt[nm]
                    ws = result["schedules"][nm]
                    s['served'][t_abs] = ws['served'][d]
                    s['unserved'][t_abs] = ws['unserved'][d]
                    s['pv_used'][t_abs] = ws['pv_used'][d]
                    s['wind_used'][t_abs] = ws['wind_used'][d]
                    s['p_ch'][t_abs] = ws['p_ch'][d]
                    s['p_dis'][t_abs] = ws['p_dis'][d]
                    s['p_buy'][t_abs] = ws['p_buy'][d]
                    s['p_sell'][t_abs] = ws['p_sell'][d]
                    if a.storage:
                        s['soc'][t_abs] = ws['soc'][d]
                        next_soc = ws.get('soc_final', ws['soc'][d])
                        if d + 1 < window_T:
                            next_soc = ws['soc'][d + 1]
                        prev_soc[nm] = next_soc
                        prev_power[nm] = (ws['p_ch'][d], ws['p_dis'][d])

                    pv_max = a.pv_real[d] if a.pv_real is not None else 0.0
                    wind_max = a.wind_real[d] if a.has_wind else 0.0
                    total_re_available += pv_max + wind_max
                    total_re_used += ws['pv_used'][d] + ws['wind_used'][d]
                    total_curtailment += (pv_max - ws['pv_used'][d]) + (wind_max - ws['wind_used'][d])
                    total_served_mwh += ws['served'][d]

                for su in (window_storage_units or []):
                    nm = su.name
                    ws = result["schedules"][nm]
                    s = schedules_rt[nm]
                    s['p_ch'][t_abs] = ws['p_ch'][d]
                    s['p_dis'][t_abs] = ws['p_dis'][d]
                    s['soc'][t_abs] = ws['soc'][d]
                    s['p_buy'][t_abs] = ws['p_buy'][d]
                    s['p_sell'][t_abs] = ws['p_sell'][d]
                    next_soc = ws.get('soc_final', ws['soc'][d])
                    if d + 1 < window_T:
                        next_soc = ws['soc'][d + 1]
                    prev_soc[nm] = next_soc
                    prev_power[nm] = (ws['p_ch'][d], ws['p_dis'][d])

                pgi = schedules_rt['GRID']['g_grid'][t_abs]
                if pgi > 0:
                    carbon_emissions += config.emission_factor_grid * pgi * dt

            total_welfare_rt += result["welfare"] * (n_commit / window_T)
        else:
            if idx > 0:
                lmp_rt[idx] = lmp_rt[idx - 1]

        idx += rt_step

    re_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    return {
        "price": lmp_rt.mean(axis=1),
        "lmp": lmp_rt,
        "schedules": schedules_rt,
        "welfare": total_welfare_rt,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_available,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
    }
def clear_rt_rolling(agents, T, action_params_base, config):
    """滚动实时市场 (MPC) – 可按需启用"""
    rt_horizon = config.rt_horizon
    rt_step = config.rt_step
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    lmp_rt = np.zeros((T, n_buses))
    schedules_rt = empty_schedules(agents, T)
    prev_soc = {}
    prev_power = {}
    total_welfare_rt = 0.0
    carbon_emissions = 0.0
    total_curtailment = 0.0
    total_served_mwh = 0.0
    total_re_available = 0.0
    total_re_used = 0.0
    wholesale = day_ahead_price_china(T)

    idx = 0
    while idx < T:
        horizon_end = min(idx + rt_horizon, T)
        sub_T = horizon_end - idx
        for t_rel in range(sub_T):
            t_abs = idx + t_rel
            success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
                base_net, agents, t_abs, "RT", prev_soc, wholesale[t_abs],
                action_params_base, config, prev_power
            )
            if success:
                lmp_rt[t_abs] = lmp_t
                total_welfare_rt += welfare_t
                schedules_rt['GRID']['g_grid'][t_abs] = p_grid
                if p_grid > 0:
                    carbon_emissions += config.emission_factor_grid * p_grid * 0.25
                for a in agents:
                    s = schedules_rt[a.name]
                    res = agent_res[a.name]
                    s['served'][t_abs] = res['served']
                    s['pv_used'][t_abs] = res['pv_used']
                    s['wind_used'][t_abs] = res['wind_used']
                    pv_max = a.pv_real[t_abs]
                    wind_max = a.get_wind_real()[t_abs] if a.has_wind else 0.0
                    total_curtailment += (pv_max - res['pv_used']) + (wind_max - res['wind_used'])
                    total_re_available += pv_max + wind_max
                    total_re_used += res['pv_used'] + res['wind_used']
                    total_served_mwh += res['served']
                    if a.storage:
                        soc0 = prev_soc.get(a.name, a.storage.soc0)
                        ch_val, dis_val, new_soc = StorageConstraints.execute_dispatch(
                            a.storage, soc0, res['p_ch'], res['p_dis'])
                        s['p_ch'][t_abs] = ch_val
                        s['p_dis'][t_abs] = dis_val
                        s['soc'][t_abs] = new_soc
                        prev_soc[a.name] = new_soc
                        prev_power[a.name] = (ch_val, dis_val)
                    load_val = a.load_real[t_abs]
                    net_gen = res['pv_used'] + res['wind_used'] + s['p_dis'][t_abs]
                    net_con = res['served'] + s['p_ch'][t_abs]
                    s['p_buy'][t_abs], s['p_sell'][t_abs] = split_power(net_gen, net_con)
                    s['unserved'][t_abs] = load_val - res['served']
            else:
                if t_abs > 0:
                    lmp_rt[t_abs] = lmp_rt[t_abs-1]
        idx += rt_step
    re_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    return {
        "price": lmp_rt.mean(axis=1),
        "lmp": lmp_rt,
        "schedules": schedules_rt,
        "welfare": total_welfare_rt,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_available,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
    }