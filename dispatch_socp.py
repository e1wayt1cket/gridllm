# dispatch_socp.py
"""SOCP (Second-Order Cone Programming) OPF for radial distribution networks.

Solves the DistFlow equations with exact SOCP relaxation, which is
computationally exact for radial networks under mild conditions.

Key differences from LinDistFlow:
  - Variables: v = V^2, I_sq = I^2 (voltage/current magnitude squared)
  - SOCP cone: P_ij^2 + Q_ij^2 <= v_i * I_sq_ij (rotated second-order cone)
  - Exact voltage drop: v_j = v_i - 2*(rP + xQ) + (r^2+x^2)*I_sq
  - No iterative loss linearization — losses captured naturally as r*I_sq
  - Can optionally include reactive power support from DERs.
"""

import numpy as np
import pandapower as pp
from typing import List, Optional

from models import Agent, MarketConfig
from dispatch_core import _select_radial_lines, _compute_mpc_schedules, empty_schedules, split_power, DT_HOURS

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

DT_HOURS = 0.25


def solve_socp_opf_batch(net, agents, T, stage, config, action_params, wholesale,
                         storage_units=None):
    """Multi-period SOCP-OPF with storage SOC transition and carbon cost.

    Uses rotated second-order cone relaxation of the DistFlow equations.
    Returns a result dict matching the clear_market schema, identical to
    solve_lindist_opf_batch output structure.
    """
    if not _HAS_GUROBI:
        raise RuntimeError("SOCP OPF requires Gurobi")
    n_buses = len(net.bus.index)
    buses = list(net.bus.index)
    lines = _select_radial_lines(net)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.network.base_kv

    # Line parameters (filter to radial lines)
    r, x_val, limit = _build_line_params(net, base_kv)
    r = {l: v for l, v in r.items() if l in lines}
    x_val = {l: v for l, v in x_val.items() if l in lines}
    limit = {l: v for l, v in limit.items() if l in lines}

    # Precompute line topology
    line_from_to = {}
    for l in lines:
        f = int(net.line.at[l, 'from_bus'])
        t = int(net.line.at[l, 'to_bus'])
        line_from_to[l] = (f, t)

    mpc_schedules = _compute_mpc_schedules(net, agents, T, wholesale, config, storage_units)

    m = gp.Model("SOCP_OPF_batch_T")
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 2)  # barrier (interior point)
    m.setParam('QCPDual', 1)  # enable dual values for QCP constraints

    # --- Network variables ---
    v = m.addVars(T, buses, lb=config.network.v_min_pu ** 2, ub=config.network.v_max_pu ** 2,
                   name="v")  # V^2
    P = m.addVars(T, lines, lb=-GRB.INFINITY, name="P")
    Q = m.addVars(T, lines, lb=-GRB.INFINITY, name="Q")
    I_sq = m.addVars(T, lines, lb=0, ub=GRB.INFINITY, name="I_sq")
    p_grid_import = m.addVars(T, lb=0, ub=GRB.INFINITY, name="p_grid_import")
    p_grid_export = m.addVars(T, lb=0, ub=config.network.reverse_power_limit_mw, name="p_grid_export")
    q_grid = m.addVars(T, lb=-GRB.INFINITY, name="q_grid")

    # --- Agent variables ---
    served = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    q_re = {}
    soc = {}
    agents_list = list(agents)
    storage_units = storage_units or []
    storage_agents = [a for a in agents_list if a.storage is not None]

    for a in agents_list:
        nm = a.name
        served[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"served_{nm}")
        unserved[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"unserv_{nm}")
        pv[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"pv_{nm}")
        wind[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"wind_{nm}")
        ch[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
        dis[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"dis_{nm}")

    if config.network.reactive_support:
        for a in agents_list:
            nm = a.name
            if a.is_prosumer or a.storage is not None:
                q_re[nm] = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"q_re_{nm}")

    for a in storage_agents:
        nm = a.name
        soc[nm] = m.addVars(T + 1, lb=a.storage.soc_min, ub=a.storage.soc_max,
                            name=f"soc_{nm}")
        m.addConstr(soc[nm][0] == a.storage.soc0, f"init_soc_{nm}")

    for su in storage_units:
        nm = su.name
        ch[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
        dis[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"dis_{nm}")
        soc[nm] = m.addVars(T + 1, lb=su.storage.soc_min, ub=su.storage.soc_max,
                            name=f"soc_{nm}")
        m.addConstr(soc[nm][0] == su.storage.soc0, f"init_soc_{nm}")

    # Non-storage agents: ch = dis = 0
    stor_names = {a.name for a in storage_agents} | {su.name for su in storage_units}
    for a in agents_list:
        if a.name not in stor_names:
            for t in range(T):
                ch[a.name][t].UB = 0
                dis[a.name][t].UB = 0

    # Fix storage variables to MPC pre-computed schedules
    if mpc_schedules:
        for nm, (mpc_ch, mpc_dis, mpc_soc_arr) in mpc_schedules.items():
            for t in range(T):
                ch[nm][t].LB = mpc_ch[t]
                ch[nm][t].UB = mpc_ch[t]
                dis[nm][t].LB = mpc_dis[t]
                dis[nm][t].UB = mpc_dis[t]
                soc[nm][t].LB = mpc_soc_arr[t]
                soc[nm][t].UB = mpc_soc_arr[t]
            soc[nm][T].LB = mpc_soc_arr[T]
            soc[nm][T].UB = mpc_soc_arr[T]

    pf = config.network.load_power_factor
    q_ratio = np.tan(np.arccos(pf))
    p_bal_constr = {}  # save refs for LMP extraction

    # --- Per-period constraints ---
    for t in range(T):
        # Reference voltage
        m.addConstr(v[t, slack_bus] == 1.0, f"ref_voltage_{t}")

        for b in buses:
            # Power balance: injection + flows = 0
            inj = gp.LinExpr()
            if b == slack_bus:
                inj += p_grid_import[t] - p_grid_export[t]
            for a in agents_list:
                if a.bus == b:
                    nm = a.name
                    inj += pv[nm][t] + wind[nm][t] + dis[nm][t] \
                           - served[nm][t] - ch[nm][t]
            for su in storage_units:
                if su.bus == b:
                    nm = su.name
                    inj += dis[nm][t] - ch[nm][t]

            # Line flows into/out of bus b
            for l, (f, to) in line_from_to.items():
                if f == b:
                    inj -= P[t, l]
                if to == b:
                    inj += P[t, l]
                # Losses allocated to 'to' bus
                if to == b:
                    inj -= r[l] * I_sq[t, l]

            c = m.addConstr(inj == 0, f"p_bal_{t}_{b}")
            p_bal_constr[(t, b)] = c

        # Reactive power balance
        for b in buses:
            q_inj = gp.LinExpr()
            if b == slack_bus:
                q_inj += q_grid[t]
            for a in agents_list:
                if a.bus == b:
                    nm = a.name
                    # Load reactive power demand
                    q_inj -= served[nm][t] * q_ratio
                    # DER reactive power support
                    if nm in q_re:
                        q_inj += q_re[nm][t]

            for l, (f, to) in line_from_to.items():
                if f == b:
                    q_inj -= Q[t, l]
                if to == b:
                    q_inj += Q[t, l]
                if to == b:
                    q_inj -= x_val[l] * I_sq[t, l]

            m.addConstr(q_inj == 0, f"q_bal_{t}_{b}")

        # DistFlow voltage drop + cone constraints per line
        for l in lines:
            f, to = line_from_to[l]
            r_l = r[l]
            x_l = x_val[l]

            # Exact voltage drop: v_to = v_from - 2(rP + xQ) + (r^2+x^2)*I_sq
            m.addConstr(
                v[t, to] == v[t, f]
                - 2 * (r_l * P[t, l] + x_l * Q[t, l])
                + (r_l ** 2 + x_l ** 2) * I_sq[t, l],
                f"v_drop_{t}_{l}"
            )

            # Rotated second-order cone: P^2 + Q^2 <= v_from * I_sq
            # Gurobi handles this natively as a quadratic constraint
            m.addQConstr(
                P[t, l] * P[t, l] + Q[t, l] * Q[t, l]
                <= v[t, f] * I_sq[t, l],
                f"socp_cone_{t}_{l}"
            )

            # Thermal limit: I_sq <= I_max^2
            m.addConstr(I_sq[t, l] <= limit[l] ** 2, f"thermal_{t}_{l}")

        # Agent RE and load constraints (same as LinDistFlow)
        for a in agents_list:
            nm = a.name
            if stage == "DA":
                pv_max = a.pv_forecast[t]
                wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
                ld = a.load_forecast[t]
            else:
                pv_max = a.pv_real[t]
                wind_max = a.get_wind_real()[t] if a.has_wind else 0.0
                ld = a.load_real[t]

            m.addConstr(pv[nm][t] <= max(pv_max, 0), f"pv_ub_{nm}_{t}")
            m.addConstr(wind[nm][t] <= max(wind_max, 0), f"wind_ub_{nm}_{t}")
            m.addConstr(served[nm][t] + unserved[nm][t] == max(ld, 0),
                        f"load_bal_{nm}_{t}")

            # Storage power caps
            if a.storage is not None:
                ch_max = a.storage.p_ch_max
                dis_max = a.storage.p_dis_max
                m.addConstr(ch[nm][t] <= ch_max, f"ch_ub_{nm}_{t}")
                m.addConstr(dis[nm][t] <= dis_max, f"dis_ub_{nm}_{t}")

            # Reactive power limits per DER inverter
            if config.network.reactive_support and nm in q_re:
                s_max = a.pv_capacity + a.wind_capacity
                if a.storage:
                    s_max += a.storage.p_dis_max
                q_re[nm][t].LB = -s_max
                q_re[nm][t].UB = s_max
                p_total = pv[nm][t] + wind[nm][t] + dis[nm][t]
                m.addConstr(q_re[nm][t] + p_total <= s_max,
                            f"diamP_{nm}_{t}")
                m.addConstr(-q_re[nm][t] + p_total <= s_max,
                            f"diamN_{nm}_{t}")

    # Storage SOC transition and ramp constraints
    for a in storage_agents:
        nm = a.name
        stor = a.storage
        e_max = stor.e_max
        eta_ch = stor.eta_ch
        eta_dis = stor.eta_dis
        for t in range(T):
            soc_t = soc[nm][t]
            soc_next = soc[nm][t + 1]
            m.addConstr(
                soc_next == soc_t + (eta_ch * ch[nm][t]
                                     - dis[nm][t] / eta_dis) * DT_HOURS / e_max,
                f"soc_trans_{nm}_{t}"
            )
            if stor.ramp_up_ch is not None and t > 0:
                m.addConstr(ch[nm][t] - ch[nm][t - 1] <= stor.ramp_up_ch,
                            f"ramp_up_ch_{nm}_{t}")
            if stor.ramp_down_ch is not None and t > 0:
                m.addConstr(ch[nm][t - 1] - ch[nm][t] <= stor.ramp_down_ch,
                            f"ramp_down_ch_{nm}_{t}")
            if stor.ramp_up_dis is not None and t > 0:
                m.addConstr(dis[nm][t] - dis[nm][t - 1] <= stor.ramp_up_dis,
                            f"ramp_up_dis_{nm}_{t}")
            if stor.ramp_down_dis is not None and t > 0:
                m.addConstr(dis[nm][t - 1] - dis[nm][t] <= stor.ramp_down_dis,
                            f"ramp_down_dis_{nm}_{t}")

    # Grid ramping constraint (peak shaving): limit net exchange change rate
    if config.network.ramp_limit_mw_per_period is not None:
        for t in range(1, T):
            ramp = config.network.ramp_limit_mw_per_period
            net_t = p_grid_import[t] - p_grid_export[t]
            net_tm1 = p_grid_import[t-1] - p_grid_export[t-1]
            m.addConstr(net_t - net_tm1 <= ramp, f"ramp_up_grid_{t}")
            m.addConstr(net_tm1 - net_t <= ramp, f"ramp_down_grid_{t}")

    # Precompute total RE available for constraint-based multi-objective
    total_re_avail_mwh = 0.0
    for a in agents_list:
        pv_arr = a.pv_forecast if stage == "DA" else a.pv_real
        wind_arr = (a.wind_forecast if stage == "DA" else a.wind_real) if a.has_wind else np.zeros(T)
        total_re_avail_mwh += float(np.sum(np.maximum(pv_arr, 0)) + np.sum(np.maximum(wind_arr, 0))) * DT_HOURS

    # Constraint-based multi-objective: carbon cap and RE minimum rate
    shadow_prices = {}
    slack_carbon = m.addVar(lb=0, ub=GRB.INFINITY, name="slack_carbon")
    slack_re = m.addVar(lb=0, ub=GRB.INFINITY, name="slack_re")
    if config.market_design.use_constraint_multi_obj:
        if config.market_design.carbon_cap_tco2 is not None:
            carbon_expr = gp.LinExpr()
            for t in range(T):
                carbon_expr += config.market_design.emission_factor_grid * p_grid_import[t] * DT_HOURS
            carbon_constr = m.addConstr(carbon_expr - slack_carbon
                                        <= config.market_design.carbon_cap_tco2,
                                        "carbon_cap")
        else:
            slack_carbon.UB = 0
        if config.market_design.re_min_rate is not None and total_re_avail_mwh > 0:
            re_expr = gp.LinExpr()
            for t in range(T):
                for a in agents_list:
                    nm = a.name
                    re_expr += (pv[nm][t] + wind[nm][t]) * DT_HOURS
            re_target = config.market_design.re_min_rate * total_re_avail_mwh
            re_constr = m.addConstr(re_expr + slack_re >= re_target, "re_min_rate")
        else:
            slack_re.UB = 0

    # --- Objective ---
    bid_arr = {}; offer_arr = {}
    pv_max_arr = {}; wind_max_arr = {}
    for a in agents_list:
        nm = a.name
        ap = action_params.get(a.name, {})
        bm = ap.get("bid_mult", 1.0)
        oa = ap.get("offer_adder", 0.0)
        if isinstance(bm, np.ndarray):
            bid_arr[nm] = a.bid_value * bm
        else:
            bid_arr[nm] = np.full(T, a.bid_value * bm)
        if isinstance(oa, np.ndarray):
            offer_arr[nm] = a.offer_cost + oa
        else:
            offer_arr[nm] = np.full(T, a.offer_cost + oa)
        pv_max_arr[nm] = a.pv_forecast if stage == "DA" else a.pv_real
        wind_max_arr[nm] = (a.wind_forecast if stage == "DA" else a.wind_real) if a.has_wind else np.zeros(T)

    if config.storage.terminal_value is not None:
        terminal_value = config.storage.terminal_value
    else:
        terminal_value = float(np.mean(wholesale))

    gamma = config.storage.discount_factor
    obj = gp.LinExpr()
    for t in range(T):
        discount_t = gamma ** t
        for a in agents_list:
            nm = a.name
            bid = bid_arr[nm][t]
            offer = offer_arr[nm][t]
            obj += bid * served[nm][t]
            if a.storage is not None:
                obj += discount_t * wholesale[t] * (dis[nm][t] - ch[nm][t])
                if config.storage.cycle_cost > 0:
                    obj -= config.storage.cycle_cost * (ch[nm][t] + dis[nm][t])
            else:
                obj -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
            obj -= config.market_design.penalty_unserved * unserved[nm][t]
        obj -= wholesale[t] * (p_grid_import[t] - p_grid_export[t])
        if config.market_design.enable_multi_objective:
            if config.market_design.lambda_carbon > 0:
                obj -= config.market_design.lambda_carbon * config.market_design.emission_factor_grid \
                       * p_grid_import[t] * DT_HOURS
            if config.market_design.lambda_re > 0:
                for a in agents_list:
                    nm = a.name
                    obj += config.market_design.lambda_re * (pv[nm][t] + wind[nm][t]) * DT_HOURS
            if config.market_design.lambda_curtail > 0:
                for a in agents_list:
                    nm = a.name
                    curtail = (pv_max_arr[nm][t] - pv[nm][t]) + (wind_max_arr[nm][t] - wind[nm][t])
                    obj -= config.market_design.lambda_curtail * curtail * DT_HOURS
    # Constraint-slack penalties
    if config.market_design.use_constraint_multi_obj:
        if config.market_design.carbon_cap_tco2 is not None:
            obj -= config.market_design.penalty_carbon_slack * slack_carbon
        if config.market_design.re_min_rate is not None:
            obj -= config.market_design.penalty_re_slack * slack_re
    disc_T = gamma ** T
    for a in storage_agents:
        obj += disc_T * terminal_value * soc[a.name][T] * a.storage.e_max

    for su in storage_units:
        nm = su.name
        for t in range(T):
            obj += su.bid_value * ch[nm][t]
            obj -= su.offer_cost * dis[nm][t]
            if config.storage.cycle_cost > 0:
                obj -= config.storage.cycle_cost * (ch[nm][t] + dis[nm][t])
        obj += disc_T * terminal_value * soc[nm][T] * su.storage.e_max

    m.setObjective(obj, GRB.MAXIMIZE)
    # --- Solve ---
    m.optimize()

    if m.status != GRB.OPTIMAL:
        if config.verbose:
            print("SOCP model solve failed, status:", m.status)
        if m.status == GRB.INFEASIBLE:
            m.computeIIS()
            m.write("socp_infeasible.ilp")
        return None

    # --- Nodal storage pricing: re-solve with storage at nodal LMP ---
    all_storage = storage_agents + storage_units
    if all_storage and config.storage.use_nodal_price:
        bus_to_idx = {b: i for i, b in enumerate(buses)}

        # Extract first-pass nodal LMPs
        nodal_lmp = np.zeros((T, n_buses))
        for t in range(T):
            for i, b in enumerate(buses):
                try:
                    nodal_lmp[t, i] = -p_bal_constr[(t, b)].Pi
                except AttributeError:
                    nodal_lmp[t, i] = wholesale[t]

        max_iters = 4
        for nodal_iter in range(max_iters):
            obj2 = gp.LinExpr()
            for t in range(T):
                discount_t = gamma ** t
                for a in agents_list:
                    nm = a.name
                    bid = bid_arr[nm][t]
                    offer = offer_arr[nm][t]
                    obj2 += bid * served[nm][t]
                    if a.storage is not None:
                        price_t = nodal_lmp[t, bus_to_idx[a.bus]]
                        obj2 += discount_t * price_t * (dis[nm][t] - ch[nm][t])
                        if config.storage.cycle_cost > 0:
                            obj2 -= config.storage.cycle_cost * (ch[nm][t] + dis[nm][t])
                    else:
                        obj2 -= offer * (pv[nm][t] + wind[nm][t] + dis[nm][t])
                    obj2 -= config.market_design.penalty_unserved * unserved[nm][t]
                obj2 -= wholesale[t] * (p_grid_import[t] - p_grid_export[t])
                if config.market_design.enable_multi_objective:
                    if config.market_design.lambda_carbon > 0:
                        obj2 -= config.market_design.lambda_carbon * config.market_design.emission_factor_grid \
                                * p_grid_import[t] * DT_HOURS
                    if config.market_design.lambda_re > 0:
                        for a in agents_list:
                            nm = a.name
                            obj2 += config.market_design.lambda_re * (pv[nm][t] + wind[nm][t]) * DT_HOURS
                    if config.market_design.lambda_curtail > 0:
                        for a in agents_list:
                            nm = a.name
                            curtail = (pv_max_arr[nm][t] - pv[nm][t]) + (wind_max_arr[nm][t] - wind[nm][t])
                            obj2 -= config.market_design.lambda_curtail * curtail * DT_HOURS
            for a in storage_agents:
                obj2 += disc_T * terminal_value * soc[a.name][T] * a.storage.e_max
            for su in storage_units:
                nm = su.name
                for t in range(T):
                    obj2 += su.bid_value * ch[nm][t]
                    obj2 -= su.offer_cost * dis[nm][t]
                    if config.storage.cycle_cost > 0:
                        obj2 -= config.storage.cycle_cost * (ch[nm][t] + dis[nm][t])
                obj2 += disc_T * terminal_value * soc[nm][T] * su.storage.e_max

            m.setObjective(obj2, GRB.MAXIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                break

            new_lmp = np.zeros((T, n_buses))
            for t in range(T):
                for i, b in enumerate(buses):
                    try:
                        new_lmp[t, i] = -p_bal_constr[(t, b)].Pi
                    except AttributeError:
                        new_lmp[t, i] = wholesale[t]

            max_change = np.max(np.abs(new_lmp - nodal_lmp))
            nodal_lmp = new_lmp
            if max_change < 1.0:
                break

    # --- Extract LMP from duals ---
    lmp = np.zeros((T, n_buses))
    for t in range(T):
        for i, b in enumerate(buses):
            try:
                lmp[t, i] = -p_bal_constr[(t, b)].Pi
            except AttributeError:
                lmp[t, i] = wholesale[t]  # dual unavailable (presolve eliminated constraint)

    # Extract shadow prices from constraint-based multi-objective
    shadow_prices = {}
    if config.market_design.use_constraint_multi_obj:
        try:
            carbon_c = m.getConstrByName("carbon_cap")
            if carbon_c is not None:
                shadow_prices["carbon_cap"] = carbon_c.Pi
        except Exception:
            pass
        try:
            re_c = m.getConstrByName("re_min_rate")
            if re_c is not None:
                shadow_prices["re_min_rate"] = -re_c.Pi
        except Exception:
            pass
        # Report slack usage
        if slack_carbon.X > 1e-6:
            shadow_prices["carbon_slack_tco2"] = float(slack_carbon.X)
        if slack_re.X > 1e-6:
            shadow_prices["re_slack_mwh"] = float(slack_re.X)

    # --- Extract results ---
    schedules = empty_schedules(agents_list, T)
    for su in storage_units:
        schedules[su.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'storage_mode': ['idle'] * T,
        }
    total_welfare = m.ObjVal
    total_re_avail = sum(
        (np.sum(a.pv_forecast if stage == "DA" else a.pv_real)
         + (np.sum(a.wind_forecast if stage == "DA" else a.wind_real)
            if a.has_wind else 0))
        * DT_HOURS for a in agents_list
    )
    total_re_used = 0.0
    total_curtailment = 0.0
    carbon_emissions = 0.0
    total_served_mwh = 0.0

    for t in range(T):
        p_import_val = p_grid_import[t].X
        if p_import_val > 0:
            carbon_emissions += config.market_design.emission_factor_grid * p_import_val * DT_HOURS

        for a in agents_list:
            nm = a.name
            s = schedules[nm]
            s['served'][t] = served[nm][t].X
            s['unserved'][t] = unserved[nm][t].X
            s['pv_used'][t] = pv[nm][t].X
            s['wind_used'][t] = wind[nm][t].X
            s['p_ch'][t] = ch[nm][t].X
            s['p_dis'][t] = dis[nm][t].X
            s['soc'][t] = soc[nm][t].X if nm in soc else 0.0

            if stage == "DA":
                pv_max = a.pv_forecast[t]
                wind_max = a.get_wind_forecast()[t] if a.has_wind else 0.0
            else:
                pv_max = a.pv_real[t]
                wind_max = a.get_wind_real()[t] if a.has_wind else 0.0

            total_re_used += (s['pv_used'][t] + s['wind_used'][t]) * DT_HOURS
            total_curtailment += max(0, (max(pv_max, 0) - s['pv_used'][t])
                                     + (max(wind_max, 0) - s['wind_used'][t])) * DT_HOURS
            net_gen = s['pv_used'][t] + s['wind_used'][t] + s['p_dis'][t]
            net_con = s['served'][t] + s['p_ch'][t]
            s['p_buy'][t], s['p_sell'][t] = split_power(net_gen, net_con)
            total_served_mwh += s['served'][t] * DT_HOURS

    # SOC for final period
    for a in storage_agents:
        schedules[a.name]['soc'][T - 1] = soc[a.name][T].X
    for su in storage_units:
        schedules[su.name]['soc'][T - 1] = soc[su.name][T].X

    re_rate = (total_re_used / total_re_avail * 100) if total_re_avail > 0 else 100.0
    carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
    result = {
        "price": lmp.mean(axis=1),
        "lmp": lmp,
        "schedules": schedules,
        "welfare": total_welfare,
        "re_consumption_rate": re_rate,
        "total_re_available": total_re_avail,
        "carbon_emissions": carbon_emissions,
        "carbon_intensity": carbon_intensity,
        "total_curtailment": total_curtailment,
        "shadow_prices": shadow_prices,
    }
    return result


def _build_line_params(net, base_kv):
    """Extract per-unit line parameters for SOCP formulation."""
    r = {}; x_val = {}; limit = {}
    v_base = base_kv
    z_base = v_base ** 2
    for idx in net.line.index:
        r_ohm = net.line.at[idx, 'r_ohm_per_km'] * net.line.at[idx, 'length_km']
        x_ohm = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km']
        r[idx] = r_ohm / z_base
        x_val[idx] = x_ohm / z_base
        # Current limit in pu: I_pu = I_ka * kV / MVA_base * sqrt(3)
        # For 1 MVA base: I_base = 1 / (sqrt(3) * V_base_kV) in kA
        i_base = 1.0 / (np.sqrt(3) * base_kv)
        limit[idx] = net.line.at[idx, 'max_i_ka'] / i_base
    return r, x_val, limit
