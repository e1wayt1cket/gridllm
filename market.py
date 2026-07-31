# market.py
import dataclasses
from copy import deepcopy

import numpy as np

from models import Agent
from grid import build_base_network, day_ahead_price_china
from dispatch import solve_opf_gurobi, StorageConstraints, solve_lindist_opf_batch
from dispatch_core import empty_schedules, split_power
from strategies import adaptive_bidding  # noqa: F401 — re-export


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
    breakdown = {}
    for a in agents:
        bus = a.bus
        lmp_da = da_lmp[:, bus_to_idx.get(bus, 0)] if da_lmp is not None else da["price"]
        lmp_rt = rt_lmp[:, bus_to_idx.get(bus, 0)] if rt_lmp is not None and rt is not None else rt["price"] if rt else np.zeros(1)

        da_import = da["schedules"][a.name]["p_buy"] - da["schedules"][a.name]["p_sell"]
        rt_import = rt["schedules"][a.name]["p_buy"] - rt["schedules"][a.name]["p_sell"]
        da_cost = np.sum(lmp_da * da_import)
        rt_cost = np.sum(lmp_rt * (rt_import - da_import))
        payments[a.name] = float(da_cost + rt_cost)
        breakdown[a.name] = {"da": float(da_cost), "rt": float(rt_cost)}
    return payments, breakdown

def clear_market(agents, T, stage, action_params, config, storage_units=None):
    if stage == "DA" and config.rt.da_rolling_enabled:
        return clear_da_rolling(agents, T, action_params, config, storage_units)

    base_net = build_base_network(config)
    wholesale = day_ahead_price_china(T, agents=agents, config=config)

    # ---- Multi-period joint optimization (LinDistFlow / SOCP) ----
    if config.opf_mode in ("lindistflow", "socp"):
        from dispatch import solve_socp_opf_batch
        solve_fn = solve_socp_opf_batch if config.opf_mode == "socp" else solve_lindist_opf_batch
        result = solve_fn(base_net, agents, T, stage, config,
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
                        # Compute consistent SOC from copied p_ch/p_dis
                        soc0 = prev_soc[a.name]
                        ch = schedules[a.name]['p_ch'][t]
                        dis = schedules[a.name]['p_dis'][t]
                        e_max = a.storage.e_max
                        eta_ch = a.storage.eta_ch
                        eta_dis = a.storage.eta_dis
                        schedules[a.name]['soc'][t] = soc0 + (eta_ch * ch - dis / eta_dis) * 0.25 / e_max
            continue

        lmp[t] = lmp_t
        schedules['GRID']['g_grid'][t] = p_grid
        total_welfare += welfare_t
        if p_grid > 0:
            carbon_emissions += config.market_design.emission_factor_grid * p_grid * 0.25

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
        "shadow_prices": {},
    }


def clear_da_rolling(agents, T, action_params, config, storage_units=None):
    """Rolling-horizon DA market clearing with limited price foresight.

    Replaces the single T-period batch solve with overlapping windows.
    Storage sees prices only within the current window, producing realistic
    intraday charge/discharge cycles instead of holding SOC all day.
    """
    window_len = config.rt.da_window_length
    step = config.rt.da_window_step
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    wholesale = day_ahead_price_china(T, agents=agents, config=config)

    lmp_da = np.zeros((T, n_buses))
    schedules_da = empty_schedules(agents, T)
    if storage_units:
        for su in storage_units:
            schedules_da[su.name] = {
                'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
                'served': np.zeros(T), 'unserved': np.zeros(T),
                'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
                'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
                'q_re': np.zeros(T),
                'storage_mode': ['idle'] * T,
            }

    prev_soc: dict[str, float] = {}
    total_welfare = 0.0
    total_re_available = 0.0
    total_re_used = 0.0
    total_curtailment = 0.0
    carbon_emissions = 0.0
    total_served_mwh = 0.0

    for a in agents:
        total_re_available += np.sum(a.pv_forecast) * 0.25
        if a.has_wind and a.wind_forecast is not None:
            total_re_available += np.sum(a.wind_forecast) * 0.25
    all_agents_list = list(agents)
    if storage_units:
        all_agents_list.extend(storage_units)
        for su in storage_units:
            if su.pv_forecast is not None:
                total_re_available += np.sum(su.pv_forecast) * 0.25
            if su.has_wind and su.wind_forecast is not None:
                total_re_available += np.sum(su.wind_forecast) * 0.25

    window_config = dataclasses.replace(config, verbose=False)

    idx = 0
    while idx < T:
        window_end = min(idx + window_len, T)
        window_T = window_end - idx

        window_agents = _make_window_agents(agents, idx, window_end,
                                            prev_soc, action_params)
        window_storage_units = None
        if storage_units:
            window_storage_units = []
            for su in storage_units:
                su_copy = deepcopy(su)
                su_copy.load_forecast = su.load_forecast[idx:window_end].copy() if su.load_forecast is not None else None
                su_copy.pv_forecast = su.pv_forecast[idx:window_end].copy() if su.pv_forecast is not None else None
                su_copy.load_real = su.load_real[idx:window_end].copy() if su.load_real is not None else None
                su_copy.pv_real = su.pv_real[idx:window_end].copy() if su.pv_real is not None else None
                if su.has_wind:
                    su_copy.wind_forecast = su.wind_forecast[idx:window_end].copy()
                    su_copy.wind_real = su.wind_real[idx:window_end].copy()
                if su.name in prev_soc:
                    su_copy.storage.soc0 = prev_soc[su.name]
                window_storage_units.append(su_copy)

        price_slice = wholesale[idx:window_end].copy()
        if config.rt.da_forecast_noise_pct > 0:
            rng = np.random.RandomState(idx)
            noise = rng.normal(0, config.rt.da_forecast_noise_pct / 100.0 * price_slice)
            price_slice = np.maximum(0, price_slice + noise)

        # Continuation value for storage at window boundary:
        # without this, storage dumps SOC at window end since the OPF sees no future.
        future_slice = wholesale[window_end:]
        continuation_price = float(np.mean(future_slice)) if len(future_slice) > 0 else 0.0
        window_config.storage = dataclasses.replace(
            window_config.storage, terminal_value=continuation_price)

        if config.opf_mode == "socp":
            from dispatch_socp import solve_socp_opf_batch
            result = solve_socp_opf_batch(
                base_net, window_agents, window_T, "DA", window_config,
                action_params, price_slice, window_storage_units)
        else:
            result = solve_lindist_opf_batch(
                base_net, window_agents, window_T, "DA", window_config,
                action_params, price_slice, window_storage_units)

        n_commit = min(step, window_T)
        if result is not None:
            for d in range(n_commit):
                t_abs = idx + d
                dt = 0.25
                lmp_da[t_abs] = result["lmp"][d]
                schedules_da['GRID']['g_grid'][t_abs] = result['schedules']['GRID']['g_grid'][d]

                for a in window_agents:
                    nm = a.name
                    s = schedules_da[nm]
                    ws = result["schedules"][nm]
                    s['served'][t_abs] = ws['served'][d]
                    s['unserved'][t_abs] = ws['unserved'][d]
                    s['pv_used'][t_abs] = ws['pv_used'][d]
                    s['wind_used'][t_abs] = ws['wind_used'][d]
                    s['p_ch'][t_abs] = ws['p_ch'][d]
                    s['p_dis'][t_abs] = ws['p_dis'][d]
                    s['p_buy'][t_abs] = ws['p_buy'][d]
                    s['p_sell'][t_abs] = ws['p_sell'][d]

                    pv_max = a.pv_forecast[d]
                    wind_max = a.wind_forecast[d] if a.has_wind else 0.0
                    total_re_used += (ws['pv_used'][d] + ws['wind_used'][d]) * dt
                    total_curtailment += ((pv_max - ws['pv_used'][d]) + (wind_max - ws['wind_used'][d])) * dt
                    total_served_mwh += ws['served'][d] * dt

                    if a.storage:
                        s['soc'][t_abs] = ws['soc'][d]
                        next_soc = ws.get('soc_final', ws['soc'][d])
                        if d + 1 < window_T:
                            next_soc = ws['soc'][d + 1]
                        prev_soc[nm] = next_soc

                for su in (window_storage_units or []):
                    nm = su.name
                    ws = result["schedules"][nm]
                    s = schedules_da[nm]
                    s['p_ch'][t_abs] = ws['p_ch'][d]
                    s['p_dis'][t_abs] = ws['p_dis'][d]
                    s['soc'][t_abs] = ws['soc'][d]
                    s['p_buy'][t_abs] = ws['p_buy'][d]
                    s['p_sell'][t_abs] = ws['p_sell'][d]
                    next_soc = ws.get('soc_final', ws['soc'][d])
                    if d + 1 < window_T:
                        next_soc = ws['soc'][d + 1]
                    prev_soc[nm] = next_soc

            total_welfare += result["welfare"] * (n_commit / window_T)
            carbon_emissions += result.get("carbon_emissions", 0) * (n_commit / window_T)
        else:
            if idx > 0:
                lmp_da[idx] = lmp_da[idx - 1]
                for a in agents:
                    for key in ['p_buy', 'p_sell', 'served', 'unserved',
                                'pv_used', 'wind_used', 'p_ch', 'p_dis', 'soc']:
                        schedules_da[a.name][key][idx] = schedules_da[a.name][key][idx - 1]

        idx += step

    re_rate = (total_re_used / total_re_available * 100) if total_re_available > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    return {
        "price": lmp_da.mean(axis=1),
        "lmp": lmp_da,
        "schedules": schedules_da,
        "welfare": total_welfare,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_available,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
        "shadow_prices": {},
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

    rt_horizon = config.rt.rt_horizon
    rt_step = config.rt.rt_step
    base_net = build_base_network(config)
    n_buses = len(base_net.bus)
    wholesale = day_ahead_price_china(T, agents=agents, config=config)
    forecaster = PriceForecaster(wholesale, mode=config.rt.rt_forecast_mode,
                                  noise_pct=config.rt.rt_forecast_noise_pct)

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

    window_config = dataclasses.replace(config, verbose=False)
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

        if config.opf_mode == "socp":
            from dispatch_socp import solve_socp_opf_batch
            result = solve_socp_opf_batch(
                base_net, window_agents, window_T, "RT", window_config,
                action_params_base, forecast, window_storage_units
            )
        else:
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
                    carbon_emissions += config.market_design.emission_factor_grid * pgi * dt

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
        "shadow_prices": {},
    }
