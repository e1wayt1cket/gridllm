# mpc_storage.py
"""MPC-based storage dispatch: look-ahead LP replaces rule-based heuristic.

Each storage agent solves a small LP over a rolling horizon to decide
optimal charge/discharge, then converts the plan into bid parameters.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_storage_mpc(storage, soc0, price_forecast, dt=0.25):
    """Solve a small LP for optimal storage schedule over the given horizon.

    The LP naturally avoids simultaneous charge+discharge because
    round-trip efficiency < 1 makes it strictly suboptimal.

    Args:
        storage: StorageSpec
        soc0: initial SOC (p.u.)
        price_forecast: array of LMP forecast [$/MWh] for next H periods
        dt: time step in hours

    Returns:
        (ch, dis, soc): arrays of length H (ch/dis) and H+1 (soc)
    """
    H = len(price_forecast)
    m = gp.Model("storage_mpc")
    m.setParam('OutputFlag', 0)

    ch = m.addVars(H, lb=0, ub=storage.p_ch_max, name="ch")
    dis = m.addVars(H, lb=0, ub=storage.p_dis_max, name="dis")
    soc = m.addVars(H + 1, lb=storage.soc_min, ub=storage.soc_max, name="soc")

    m.addConstr(soc[0] == soc0)
    for t in range(H):
        m.addConstr(
            soc[t + 1] == soc[t]
            + (storage.eta_ch * ch[t] - dis[t] / storage.eta_dis) * dt / storage.e_max
        )

    # Maximize arbitrage revenue + terminal value
    revenue = gp.quicksum(
        (dis[t] - ch[t]) * price_forecast[t] * dt for t in range(H)
    )
    # Terminal value: energy kept at end of horizon can be sold later.
    # Use max forecast price as a proxy for future opportunity value,
    # so the LP doesn't discharge at low prices just before horizon end.
    terminal_price = np.max(price_forecast)
    terminal_value = terminal_price * soc[H] * storage.e_max * storage.eta_dis
    m.setObjective(revenue + terminal_value, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        return (
            np.zeros(H), np.zeros(H),
            np.full(H + 1, soc0)
        )

    ch_vals = np.array([ch[t].X for t in range(H)])
    dis_vals = np.array([dis[t].X for t in range(H)])
    soc_vals = np.array([soc[t].X for t in range(H + 1)])
    return ch_vals, dis_vals, soc_vals


def mpc_storage_bidding(agents, config, market_history=None, T=96, H=8):
    """MPC-based bidding: storage agents use look-ahead LP, others use best_response.

    Each storage agent:
      1. Forecasts LMP for next H periods
      2. Solves MPC LP for optimal charge/discharge
      3. Converts MPC schedule to bid_mult/offer_adder

    Args:
        agents: list of Agent
        config: MarketConfig
        market_history: optional previous market result for LMP forecast
        T: total periods
        H: MPC look-ahead horizon (periods)

    Returns:
        actions dict
    """
    from market import best_response_bidding, day_ahead_price_china

    # Base actions for all agents (fallback)
    base_actions = best_response_bidding(agents, config, market_history, T)

    # LMP forecast: use previous market LMP if available, else base price curve
    if market_history is not None:
        lmp_forecast = market_history.get("price", day_ahead_price_china(T))
    else:
        lmp_forecast = day_ahead_price_china(T)

    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return base_actions

    dt = 0.25

    for a in storage_agents:
        bid_mult = np.ones(T)
        offer_adder = np.full(T, np.mean(config.offer_adder_range))

        # Rolling MPC: at each period, look ahead H periods, take first action
        soc_now = a.storage.soc0
        mpc_ch = np.zeros(T)
        mpc_dis = np.zeros(T)

        for t in range(T):
            # Build price forecast for next H periods
            remaining = min(H, T - t)
            if remaining == 0:
                break
            # Wrap around or pad: use forecast[t:t+H] with fallback to mean
            if t + remaining <= T:
                price_fwd = lmp_forecast[t:t + remaining]
            else:
                price_fwd = np.pad(
                    lmp_forecast[t:], (0, remaining - (T - t)),
                    mode='mean'
                )

            ch_sched, dis_sched, soc_sched = solve_storage_mpc(
                a.storage, soc_now, price_fwd, dt
            )

            # Take first-period action
            mpc_ch[t] = ch_sched[0]
            mpc_dis[t] = dis_sched[0]
            soc_now = soc_sched[1]

            # Convert to bid parameters
            ch_power = ch_sched[0]
            dis_power = dis_sched[0]

            if dis_power > 0.001:
                # MPC wants to discharge: lower offer_adder to sell more
                offer_adder[t] = np.clip(
                    config.offer_adder_range[0] + 5.0,
                    *config.offer_adder_range
                )
                bid_mult[t] = np.clip(0.85, *config.bid_mult_range)
            elif ch_power > 0.001:
                # MPC wants to charge: raise bid_mult to buy more
                bid_mult[t] = np.clip(1.15, *config.bid_mult_range)
                offer_adder[t] = np.clip(
                    config.offer_adder_range[1] - 5.0,
                    *config.offer_adder_range
                )
            else:
                # Idle: moderate parameters
                bid_mult[t] = np.mean(config.bid_mult_range)
                offer_adder[t] = np.mean(config.offer_adder_range)

        if a.is_prosumer:
            base_actions[a.name] = {
                "bid_mult": bid_mult, "offer_adder": offer_adder
            }
        else:
            base_actions[a.name] = {"bid_mult": bid_mult}

    return base_actions


def mpc_rt_rolling(agents, T, action_params_base, config, H=8):
    """MPC-based rolling RT market with short-horizon batch optimization.

    Replaces the single-period loop in clear_rt_rolling with a rolling
    batch OPF over horizon H periods. Only the first rt_step periods
    are executed before re-optimizing.

    Args:
        agents: list of Agent
        T: total periods
        action_params_base: base action parameters
        config: MarketConfig
        H: MPC horizon (periods, default 8 = 2 hours)

    Returns:
        result dict matching clear_market schema
    """
    from market import empty_schedules, split_power
    from dispatch import StorageConstraints
    from grid import build_base_network
    import copy

    base_net = build_base_network(config)
    wholesale = __import__('grid').day_ahead_price_china(T)

    rt_step = max(1, config.rt_step)
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

    idx = 0
    while idx < T:
        horizon_end = min(idx + H, T)
        sub_T = horizon_end - idx
        if sub_T < 1:
            break

        # Build mini batch for this horizon
        try:
            from dispatch import solve_lindist_opf_batch
            # Create a slice of agents with data for the sub-horizon
            sub_agents = []
            for a in agents:
                import copy as cp
                sub_a = cp.deepcopy(a)
                # Slice forecasts/actuals to the sub-horizon
                sub_a.load_forecast = a.load_real[idx:horizon_end]
                sub_a.pv_forecast = a.pv_real[idx:horizon_end]
                if a.has_wind:
                    sub_a.wind_forecast = a.wind_real[idx:horizon_end]
                sub_a.load_real = a.load_real[idx:horizon_end]
                sub_a.pv_real = a.pv_real[idx:horizon_end]
                if a.has_wind:
                    sub_a.wind_real = a.wind_real[idx:horizon_end]
                # Update storage SOC
                if a.storage is not None and a.name in prev_soc:
                    sub_a.storage.soc0 = prev_soc[a.name]
                sub_agents.append(sub_a)

            sub_actions = {}
            for name, act in action_params_base.items():
                sub_actions[name] = {}
                for k, v in act.items():
                    if isinstance(v, np.ndarray) and len(v) == T:
                        sub_actions[name][k] = v[idx:horizon_end]
                    else:
                        sub_actions[name][k] = v

            sub_wholesale = wholesale[idx:horizon_end]
            sub_result = solve_lindist_opf_batch(
                base_net, sub_agents, sub_T, "RT", config,
                sub_actions, sub_wholesale
            )

            if sub_result is not None:
                # Apply only the first rt_step periods
                apply_steps = min(rt_step, sub_T)
                for t_rel in range(apply_steps):
                    t_abs = idx + t_rel
                    lmp_rt[t_abs] = sub_result["lmp"][t_rel]

                    sched_rt = schedules_rt
                    sched_rt['GRID']['g_grid'][t_abs] = \
                        sub_result["schedules"]['GRID']['g_grid'][t_rel]

                    for a in agents:
                        nm = a.name
                        s_sub = sub_result["schedules"][nm]
                        s_rt = sched_rt[nm]
                        for key in ['served', 'unserved', 'pv_used', 'wind_used',
                                    'p_ch', 'p_dis']:
                            s_rt[key][t_abs] = s_sub[key][t_rel]
                        if a.storage:
                            s_rt['soc'][t_abs] = s_sub['soc'][t_rel]
                            prev_soc[nm] = s_sub['soc'][t_rel + 1] \
                                if t_rel + 1 < len(s_sub['soc']) else s_sub['soc'][t_rel]

                        load_val = a.load_real[t_abs]
                        pv_max = a.pv_real[t_abs]
                        wind_max = a.get_wind_real()[t_abs] if a.has_wind else 0.0
                        net_gen = s_rt['pv_used'][t_abs] + s_rt['wind_used'][t_abs] \
                                  + s_rt['p_dis'][t_abs]
                        net_con = s_rt['served'][t_abs] + s_rt['p_ch'][t_abs]
                        s_rt['p_buy'][t_abs], s_rt['p_sell'][t_abs] = \
                            split_power(net_gen, net_con)
                        s_rt['unserved'][t_abs] = load_val - s_rt['served'][t_abs]
                        total_re_used += (s_rt['pv_used'][t_abs] + s_rt['wind_used'][t_abs]) * 0.25
                        total_curtailment += ((pv_max - s_rt['pv_used'][t_abs])
                                              + (wind_max - s_rt['wind_used'][t_abs])) * 0.25
                        total_re_available += (pv_max + wind_max) * 0.25
                        total_served_mwh += s_rt['served'][t_abs] * 0.25
                        if a.storage:
                            prev_power[nm] = (s_rt['p_ch'][t_abs], s_rt['p_dis'][t_abs])

                    p_grid = sub_result["schedules"]['GRID']['g_grid'][t_rel]
                    if p_grid > 0:
                        carbon_emissions += config.emission_factor_grid * p_grid * 0.25
                    total_welfare_rt += sub_result.get("welfare", 0) / sub_T

                idx += apply_steps
            else:
                # Fallback: single-period solve
                _mpc_fallback_step(
                    agents, base_net, config, idx, T, wholesale,
                    action_params_base, prev_soc, prev_power,
                    lmp_rt, schedules_rt,
                )
                idx += 1
        except Exception:
            _mpc_fallback_step(
                agents, base_net, config, idx, T, wholesale,
                action_params_base, prev_soc, prev_power,
                lmp_rt, schedules_rt,
            )
            idx += 1

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


def _mpc_fallback_step(agents, base_net, config, t_abs, T, wholesale,
                       action_params_base, prev_soc, prev_power,
                       lmp_rt, schedules_rt):
    """Single-period fallback for when batch MPC fails."""
    from dispatch import solve_opf_gurobi, StorageConstraints
    from market import split_power

    success, lmp_t, welfare_t, agent_res, p_grid = solve_opf_gurobi(
        base_net, agents, t_abs, "RT", prev_soc, wholesale[t_abs],
        action_params_base, config, prev_power
    )
    if not success:
        if t_abs > 0:
            lmp_rt[t_abs] = lmp_rt[t_abs - 1]
        return

    lmp_rt[t_abs] = lmp_t
    schedules_rt['GRID']['g_grid'][t_abs] = p_grid
    for a in agents:
        s = schedules_rt[a.name]
        res = agent_res[a.name]
        s['served'][t_abs] = res['served']
        s['pv_used'][t_abs] = res['pv_used']
        s['wind_used'][t_abs] = res['wind_used']
        if a.storage:
            soc0 = prev_soc.get(a.name, a.storage.soc0)
            ch_val, dis_val, new_soc = StorageConstraints.execute_dispatch(
                a.storage, soc0, res['p_ch'], res['p_dis']
            )
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
