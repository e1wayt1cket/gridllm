# market.py
import dataclasses
from copy import deepcopy

import numpy as np

from models import Agent
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
            'q_re': np.zeros(T),
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

def learning_bidding(agents, config, market_history=None, T=96):
    """LMP-aware bidding with exponential moving average of historical prices.

    Each agent tracks the LMP at its own bus and adjusts bid/offer
    based on whether its local price is above or below its valuation.
    """
    base_price = day_ahead_price_china(T)
    actions = {}
    for a in agents:
        if market_history and 'lmp_per_agent' in market_history and a.name in market_history['lmp_per_agent']:
            hist_lmp = market_history['lmp_per_agent'][a.name]
            # EMA-smoothed LMP signal
            alpha = 0.3
            lmp_signal = np.zeros(T)
            lmp_signal[0] = hist_lmp[0] if len(hist_lmp) > 0 else base_price[0]
            for t in range(1, T):
                lmp_signal[t] = alpha * hist_lmp[t] + (1 - alpha) * lmp_signal[t - 1]
        else:
            lmp_signal = base_price

        if a.is_prosumer:
            bid_mult = np.full(T, config.default_bid_mult)
            offer_adder = np.full(T, config.default_offer_adder)
            for t in range(T):
                # If local LMP >> offer_cost, lower offer to sell more
                margin = lmp_signal[t] - a.offer_cost
                if margin > 100:
                    offer_adder[t] = max(0, offer_adder[t] - 20)
                elif margin > 0:
                    offer_adder[t] = max(0, offer_adder[t] - 5)
                elif margin < -50:
                    offer_adder[t] = min(50, offer_adder[t] + 15)
                # If local LMP >> bid_value, bid higher to ensure load is served
                if lmp_signal[t] > a.bid_value * 1.1:
                    bid_mult[t] = min(config.bid_mult_range[1], bid_mult[t] + 0.1)
                elif lmp_signal[t] < a.bid_value * 0.8:
                    bid_mult[t] = max(config.bid_mult_range[0], bid_mult[t] - 0.1)
            actions[a.name] = {"bid_mult": bid_mult, "offer_adder": offer_adder}
        else:
            bid_mult = np.full(T, config.default_bid_mult)
            for t in range(T):
                # Consumer: bid higher when local LMP exceeds willingness-to-pay
                if lmp_signal[t] > a.bid_value:
                    bid_mult[t] = min(config.bid_mult_range[1], bid_mult[t] + 0.15)
                elif lmp_signal[t] < a.bid_value * 0.6:
                    bid_mult[t] = max(config.bid_mult_range[0], bid_mult[t] - 0.1)
            actions[a.name] = {"bid_mult": bid_mult}
    return actions


_STRATEGIES = {
    "random": (lambda agents, config, T, _: random_actions(agents, config, T)),
    "best_response": (lambda agents, config, T, mh: best_response_bidding(agents, config, mh, T)),
    "mpc": (lambda agents, config, T, mh: _mpc_bidding(agents, config, mh, T)),
    "learning": (lambda agents, config, T, mh: learning_bidding(agents, config, mh, T)),
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
    # Build bus→position map for nodal LMP lookup
    da_lmp = da.get("lmp", None)
    rt_lmp = rt.get("lmp", None) if rt else None
    if da_lmp is not None and da_lmp.shape[1] >= max(a.bus for a in agents) + 1:
        bus_to_idx = {b: b for b in range(da_lmp.shape[1])}
    else:
        bus_to_idx = {}

    payments = {}
    for a in agents:
        bus = a.bus
        lmp_da = da_lmp[:, bus_to_idx.get(bus, 0)] if da_lmp is not None else da["price"]
        lmp_rt = rt_lmp[:, bus_to_idx.get(bus, 0)] if rt_lmp is not None and rt is not None else rt["price"] if rt else np.zeros(1)

        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(lmp_da * da_import)
        rt_cost = np.sum(lmp_rt * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
    return payments

def clear_market(agents, T, stage, action_params, config, storage_units=None):
    base_net = build_base_network(config)
    wholesale = day_ahead_price_china(T)

    # ---- Multi-period joint optimization (LinDistFlow only) ----
    if config.opf_mode == "lindistflow":
        result = solve_lindist_opf_batch(base_net, agents, T, stage, config,
                                         action_params, wholesale, storage_units)
        if result is not None:
            return result
        else:
            print("Batch solve failed, falling back to single-period solver...")

    # ---- Per-period fallback (DC-OPF or LinDistFlow single-period) ----
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
            total_re_available += np.sum(a.pv_forecast) * 0.25
            if a.has_wind and a.wind_forecast is not None:
                total_re_available += np.sum(a.wind_forecast) * 0.25
        else:
            total_re_available += np.sum(a.pv_real) * 0.25
            if a.has_wind and a.wind_real is not None:
                total_re_available += np.sum(a.wind_real) * 0.25

    prev_soc: dict[str, float] = {}

    for t in range(T):
        success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
            base_net, agents, t, stage, prev_soc, wholesale[t],
            action_params, config
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
            total_re_used += (res['pv_used'] + res['wind_used']) * 0.25
            total_curtailment += ((pv_max - res['pv_used']) + (wind_max - res['wind_used'])) * 0.25
            total_served_mwh += res['served'] * 0.25

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
    total_welfare_rt = 0.0
    carbon_emissions = 0.0
    total_curtailment = 0.0
    total_served_mwh = 0.0
    total_re_available = 0.0
    total_re_used = 0.0

    window_config = dataclasses.replace(config, verbose=False,
                                        enable_multi_objective=False,
                                        use_constraint_multi_obj=False)
    idx = 0
    while idx < T:
        window_end = min(idx + rt_horizon, T)
        window_T = window_end - idx
        forecast = forecaster.forecast(idx, window_T)

        window_agents = _make_window_agents(agents, idx, window_end,
                                            prev_soc, action_params_base)
        window_storage_units = None
        if storage_units:
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

                    pv_max = a.pv_real[d] if a.pv_real is not None else 0.0
                    wind_max = a.wind_real[d] if a.has_wind else 0.0
                    total_re_available += (pv_max + wind_max) * dt
                    total_re_used += (ws['pv_used'][d] + ws['wind_used'][d]) * 0.25
                    total_curtailment += ((pv_max - ws['pv_used'][d]) + (wind_max - ws['wind_used'][d])) * 0.25
                    total_served_mwh += ws['served'][d] * 0.25

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
    """Rolling real-time market (MPC) — enable on demand."""
    rt_horizon = config.rt_horizon
    rt_step = config.rt_step
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    lmp_rt = np.zeros((T, n_buses))
    schedules_rt = empty_schedules(agents, T)
    prev_soc = {}
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
                action_params_base, config
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
                    load_val = a.load_real[t_abs]
                    s['served'][t_abs] = res['served']
                    s['pv_used'][t_abs] = res['pv_used']
                    s['wind_used'][t_abs] = res['wind_used']
                    pv_max = a.pv_real[t_abs] if a.pv_real is not None else 0.0
                    wind_max = a.get_wind_real()[t_abs] if a.has_wind else 0.0
                    total_curtailment += ((pv_max - res['pv_used']) + (wind_max - res['wind_used'])) * 0.25
                    total_re_available += (pv_max + wind_max) * 0.25
                    total_re_used += (res['pv_used'] + res['wind_used']) * 0.25
                    total_served_mwh += res['served'] * 0.25
                    if a.storage:
                        soc0 = prev_soc.get(a.name, a.storage.soc0)
                        ch_val, dis_val, new_soc = StorageConstraints.execute_dispatch(
                            a.storage, soc0, res['p_ch'], res['p_dis'])
                        s['p_ch'][t_abs] = ch_val
                        s['p_dis'][t_abs] = dis_val
                        s['soc'][t_abs] = new_soc
                        prev_soc[a.name] = new_soc
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