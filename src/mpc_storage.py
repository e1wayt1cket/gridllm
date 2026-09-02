# mpc_storage.py
"""MPC-based storage dispatch: look-ahead LP replaces rule-based heuristic.

Each storage agent solves a small LP over a rolling horizon to decide
optimal charge/discharge, then converts the plan into bid parameters.

Supports:
  - Deterministic LP (solve_storage_mpc)
  - 2-stage stochastic LP with scenario-tree prices (solve_storage_mpc_stochastic)
  - SOC gap reserve for forecasted RE shortfall (compute_re_gap_reserve)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, List


def compute_re_gap_reserve(load_forecast, pv_forecast, wind_forecast,
                           soc_min, soc_max, gap_factor=0.5, dt=0.25):
    """Precompute SOC reserve needed for forecasted RE shortfalls.

    For each period t, computes the cumulative net load surplus
    (load - RE) over the remaining horizon. Storage should reserve
    enough SOC to cover at least gap_factor * worst-case gap.

    Args:
        load_forecast: array of load forecasts [MW] for H periods
        pv_forecast: array of PV forecasts [MW] for H periods
        wind_forecast: array of wind forecasts [MW] for H periods
        soc_min: minimum SOC (p.u.)
        soc_max: maximum SOC (p.u.)
        gap_factor: fraction of future RE gap to reserve (0=disabled, 1=full)
        dt: time step in hours

    Returns:
        soc_reserve: array of length H, minimum SOC threshold per period
    """
    H = len(load_forecast)
    re_gen = np.maximum(pv_forecast, 0) + np.maximum(wind_forecast, 0)
    net_load = load_forecast - re_gen  # positive = deficit, negative = surplus

    # Cumulative net load from t+1 to H (remaining horizon)
    # If cumulative is positive, storage needs enough energy to cover
    soc_reserve = np.zeros(H)
    if gap_factor <= 0 or H == 0:
        return soc_reserve

    cum_deficit = 0.0
    for t in range(H - 1, -1, -1):
        cum_deficit += max(0, net_load[t]) * dt
        soc_reserve[t] = min(soc_min + gap_factor * cum_deficit, soc_max)

    return np.maximum(soc_reserve, soc_min)


def solve_storage_mpc(storage, soc0, price_forecast, dt=0.25,
                      terminal_price=None, soc_reserve=None):
    """Solve a small LP for optimal storage schedule over the given horizon.

    The LP naturally avoids simultaneous charge+discharge because
    round-trip efficiency < 1 makes it strictly suboptimal.

    Args:
        storage: StorageSpec
        soc0: initial SOC (p.u.)
        price_forecast: array of LMP forecast [$/MWh] for next H periods
        dt: time step in hours
        terminal_price: optional override for terminal value price.
            If None, uses max(price_forecast). For DA self-scheduling,
            pass the DA average price to limit "hold" behavior.

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
        # SOC reserve for future RE shortfall
        if soc_reserve is not None and t < H:
            m.addConstr(soc[t] >= float(soc_reserve[t]),
                        f"soc_reserve_{t}")

    # Maximize arbitrage revenue + terminal value
    revenue = gp.quicksum(
        (dis[t] - ch[t]) * price_forecast[t] * dt for t in range(H)
    )
    # Terminal value: energy kept at end of horizon can be sold later.
    # Default uses max forecast price as a proxy for future opportunity value,
    # so the LP doesn't discharge at low prices just before horizon end.
    if terminal_price is None:
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


def solve_storage_mpc_stochastic(storage, soc0, price_scenarios,
                                 scenario_probs=None, dt=0.25,
                                 terminal_price=None, soc_reserve=None):
    """2-stage stochastic LP for storage scheduling under price uncertainty.

    Stage 1 (t=0, here-and-now): ch and dis decisions are identical across
    all scenarios (non-anticipativity).

    Stage 2 (t>0, wait-and-see): each scenario has its own trajectory that
    can diverge based on the realized price path.

    Args:
        storage: StorageSpec
        soc0: initial SOC (p.u.)
        price_scenarios: list of arrays, each shape (H,), one per scenario
        scenario_probs: probabilities per scenario (default: uniform)
        dt: time step in hours
        terminal_price: override for terminal value (default: mean of scenario max)
        soc_reserve: optional array of minimum SOC thresholds per period

    Returns:
        (ch_plan, dis_plan, soc_exp):
            ch_plan, dis_plan: stage-1 decision for period 0 (length H, t>0=0)
            soc_exp: expected SOC trajectory (length H+1)
    """
    n_scenarios = len(price_scenarios)
    H = len(price_scenarios[0])
    if scenario_probs is None:
        scenario_probs = np.full(n_scenarios, 1.0 / n_scenarios)

    m = gp.Model("storage_mpc_stochastic")
    m.setParam('OutputFlag', 0)

    # Variables per scenario
    ch_s = {}
    dis_s = {}
    soc_s = {}
    for s in range(n_scenarios):
        ch_s[s] = m.addVars(H, lb=0, ub=storage.p_ch_max,
                            name=f"ch_s{s}")
        dis_s[s] = m.addVars(H, lb=0, ub=storage.p_dis_max,
                             name=f"dis_s{s}")
        soc_s[s] = m.addVars(H + 1, lb=storage.soc_min, ub=storage.soc_max,
                             name=f"soc_s{s}")

    # Non-anticipativity: first-period decisions must be identical
    ch0 = m.addVar(lb=0, ub=storage.p_ch_max, name="ch0")
    dis0 = m.addVar(lb=0, ub=storage.p_dis_max, name="dis0")
    for s in range(n_scenarios):
        m.addConstr(ch_s[s][0] == ch0, f"na_ch_{s}")
        m.addConstr(dis_s[s][0] == dis0, f"na_dis_{s}")

    # SOC dynamics and reserve per scenario
    for s in range(n_scenarios):
        m.addConstr(soc_s[s][0] == soc0)
        for t in range(H):
            m.addConstr(
                soc_s[s][t + 1] == soc_s[s][t]
                + (storage.eta_ch * ch_s[s][t]
                   - dis_s[s][t] / storage.eta_dis) * dt / storage.e_max
            )
            if soc_reserve is not None and t < H:
                m.addConstr(soc_s[s][t] >= float(soc_reserve[t]),
                            f"soc_reserve_s{s}_{t}")

    # Expected revenue + terminal value
    if terminal_price is None:
        terminal_price = float(np.mean([np.max(ps) for ps in price_scenarios]))
    expected_obj = gp.QuadExpr() if False else gp.LinExpr()
    for s in range(n_scenarios):
        prob = scenario_probs[s]
        price = price_scenarios[s]
        revenue = gp.quicksum(
            (dis_s[s][t] - ch_s[s][t]) * price[t] * dt for t in range(H)
        )
        term_val = terminal_price * soc_s[s][H] * storage.e_max * storage.eta_dis
        expected_obj += prob * (revenue + term_val)

    m.setObjective(expected_obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        return (
            np.zeros(H), np.zeros(H),
            np.full(H + 1, soc0)
        )

    # Extract stage-1 decision and expected SOC
    ch_plan = np.zeros(H)
    dis_plan = np.zeros(H)
    ch_plan[0] = ch0.X
    dis_plan[0] = dis0.X
    # For t>0, use the probability-weighted average
    soc_exp = np.zeros(H + 1)
    for s in range(n_scenarios):
        prob = scenario_probs[s]
        soc_exp += prob * np.array([soc_s[s][t].X for t in range(H + 1)])
    soc_exp[0] = soc0

    return ch_plan, dis_plan, soc_exp


def solve_storage_mpc_batch(storage_data_list, dt=0.25):
    """Solve MPC for multiple storage units in a single Gurobi LP.

    Merges independent storage MPC problems into one model to avoid
    repeated model-creation overhead. Each unit has its own variables,
    constraints, and objective term — they are separable but solved
    jointly for efficiency.

    Args:
        storage_data_list: list of (storage, soc0, price_forecast,
            terminal_price) tuples, one per storage unit.
            price_forecast: array of length H.
        dt: time step in hours.

    Returns:
        list of (ch_vals, dis_vals, soc_vals) tuples, one per input.
        ch_vals/dis_vals length H, soc_vals length H+1.
    """
    if not storage_data_list:
        return []

    m = gp.Model("storage_mpc_batch")
    m.setParam('OutputFlag', 0)

    meta = []  # (ch_vars, dis_vars, soc_vars, H, storage)
    obj_terms = []

    for idx, (storage, soc0, price_fwd, term_price) in enumerate(storage_data_list):
        H = len(price_fwd)
        pref = f"u{idx}"
        ch = m.addVars(H, lb=0, ub=storage.p_ch_max, name=f"{pref}_ch")
        dis = m.addVars(H, lb=0, ub=storage.p_dis_max, name=f"{pref}_dis")
        soc = m.addVars(H + 1, lb=storage.soc_min, ub=storage.soc_max,
                        name=f"{pref}_soc")

        m.addConstr(soc[0] == float(soc0), name=f"{pref}_soc0")
        for t in range(H):
            delta = (storage.eta_ch * ch[t]
                     - dis[t] / storage.eta_dis) * dt / storage.e_max
            m.addConstr(soc[t + 1] == soc[t] + delta,
                        name=f"{pref}_dyn{t}")

        revenue = gp.quicksum(
            (dis[t] - ch[t]) * float(price_fwd[t]) * dt for t in range(H))
        tp = term_price if term_price is not None else float(np.max(price_fwd))
        terminal = tp * soc[H] * storage.e_max * storage.eta_dis
        obj_terms.append(revenue + terminal)
        meta.append((ch, dis, soc, H))

    m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        results = []
        for _, (storage, soc0, price_fwd, _) in enumerate(storage_data_list):
            H = len(price_fwd)
            results.append(
                (np.zeros(H), np.zeros(H), np.full(H + 1, float(soc0))))
        return results

    results = []
    for (ch, dis, soc, H) in meta:
        ch_vals = np.array([ch[t].X for t in range(H)])
        dis_vals = np.array([dis[t].X for t in range(H)])
        soc_vals = np.array([soc[t].X for t in range(H + 1)])
        results.append((ch_vals, dis_vals, soc_vals))
    return results


class PersistentMPCModel:
    """Reusable Gurobi LP for rolling-horizon MPC across time steps.

    Creates the model once at maximum horizon, then updates coefficients
    and bounds each time step — eliminating per-step model-creation overhead.

    Usage::
        pm = PersistentMPCModel(storage_list, max_horizon, dt)
        for t in range(T):
            results = pm.solve_step(soc_now_list, price_list, term_price)
            # commit first-period actions, advance soc_now
    """

    def __init__(self, storage_list, max_horizon, dt=0.25):
        self.storage_list = list(storage_list)
        self.max_H = max_horizon
        self.dt = dt
        self.n_units = len(storage_list)

        self._m = gp.Model("storage_mpc_persistent")
        self._m.setParam('OutputFlag', 0)

        # Allocate variables for max horizon × all units
        self._ch = []
        self._dis = []
        self._soc = []
        self._dyn_constrs = []  # per-unit list of constraint lists
        self._obj_terms_per_unit = []  # reusable linear expressions

        for i, s in enumerate(storage_list):
            pref = f"u{i}"
            ch = self._m.addVars(max_horizon, lb=0, ub=s.p_ch_max,
                                 name=f"{pref}_ch")
            dis = self._m.addVars(max_horizon, lb=0, ub=s.p_dis_max,
                                  name=f"{pref}_dis")
            soc = self._m.addVars(max_horizon + 1, lb=s.soc_min,
                                  ub=s.soc_max, name=f"{pref}_soc")

            # SOC dynamics: soc[t+1] == soc[t] + eta_ch*ch[t]*dt/e_max
            #                         - dis[t]/eta_dis*dt/e_max
            constrs = []
            for t in range(max_horizon):
                delta = (s.eta_ch * ch[t]
                         - dis[t] / s.eta_dis) * dt / s.e_max
                c = self._m.addConstr(
                    soc[t + 1] == soc[t] + delta,
                    name=f"{pref}_dyn{t}")
                constrs.append(c)

            self._ch.append(ch)
            self._dis.append(dis)
            self._soc.append(soc)
            self._dyn_constrs.append(constrs)
            self._obj_terms_per_unit.append(None)  # built per-step

    def _rebuild_objective(self, price_lists, term_price):
        """Replace objective with current price forecasts."""
        obj = gp.LinExpr()
        for i, s in enumerate(self.storage_list):
            prices = price_lists[i]
            H = len(prices)
            ch = self._ch[i]
            dis = self._dis[i]
            soc = self._soc[i]
            revenue = gp.quicksum(
                (dis[t] - ch[t]) * float(prices[t]) * self.dt
                for t in range(H))
            tp = term_price if term_price is not None else float(np.max(prices))
            terminal = tp * soc[H] * s.e_max * s.eta_dis
            obj += revenue + terminal
        self._m.setObjective(obj, GRB.MAXIMIZE)

    def _update_bounds(self, soc0_list, horizons, term_price):
        """Update variable bounds for current time step.

        For periods beyond the horizon, fix variables to zero.
        The SOC initial constraint is updated via lb/ub on soc[0].
        """
        for i, s in enumerate(self.storage_list):
            H = horizons[i]
            soc0 = float(soc0_list[i])
            ch = self._ch[i]
            dis = self._dis[i]
            soc = self._soc[i]

            # Fix SOC initial value
            soc[0].lb = soc0
            soc[0].ub = soc0

            # Active horizon: set storage bounds
            for t in range(self.max_H):
                if t < H:
                    ch[t].lb = 0
                    ch[t].ub = s.p_ch_max
                    dis[t].lb = 0
                    dis[t].ub = s.p_dis_max
                    soc[t + 1].lb = s.soc_min
                    soc[t + 1].ub = s.soc_max
                else:
                    # Beyond horizon: fix to zero / last SOC
                    ch[t].lb = 0
                    ch[t].ub = 0
                    dis[t].lb = 0
                    dis[t].ub = 0
                    soc[t + 1].lb = soc0
                    soc[t + 1].ub = soc0

            # Update SOC dynamics constraints for active horizon
            for t in range(self.max_H):
                if t < H:
                    # Keep constraint active: adjust delta coefficients
                    delta = (s.eta_ch * ch[t]
                             - dis[t] / s.eta_dis) * self.dt / s.e_max
                    self._m.chgCoef(
                        self._dyn_constrs[i][t], soc[t + 1], -1.0)
                    self._m.chgCoef(
                        self._dyn_constrs[i][t], soc[t], 1.0)
                    self._m.chgCoef(
                        self._dyn_constrs[i][t], ch[t],
                        s.eta_ch * self.dt / s.e_max)
                    self._m.chgCoef(
                        self._dyn_constrs[i][t], dis[t],
                        -1.0 / s.eta_dis * self.dt / s.e_max)
                    # Remove old RHS by setting to 0 (constraint is soc[t+1] == soc[t] + delta)
                    self._m.chgCoeff(
                        self._dyn_constrs[i][t],
                        self._dyn_constrs[i][t].getAttr('RHS'), 0.0)

    def solve_step(self, soc0_list, price_lists, term_price):
        """Solve one MPC step: update bounds, rebuild objective, optimize.

        Args:
            soc0_list: list of initial SOC per unit (length n_units)
            price_lists: list of price arrays per unit (each length ≤ max_H)
            term_price: terminal value price or None

        Returns:
            list of (ch, dis, soc) arrays per unit
        """
        horizons = [len(p) for p in price_lists]
        self._update_bounds(soc0_list, horizons, term_price)
        self._rebuild_objective(price_lists, term_price)
        self._m.optimize()

        if self._m.status != GRB.OPTIMAL:
            results = []
            for i, s in enumerate(self.storage_list):
                H = horizons[i]
                results.append(
                    (np.zeros(H), np.zeros(H),
                     np.full(H + 1, float(soc0_list[i]))))
            return results

        results = []
        for i, s in enumerate(self.storage_list):
            H = horizons[i]
            ch_vals = np.array([self._ch[i][t].X for t in range(H)])
            dis_vals = np.array([self._dis[i][t].X for t in range(H)])
            soc_vals = np.array([self._soc[i][t].X for t in range(H + 1)])
            results.append((ch_vals, dis_vals, soc_vals))
        return results


def solve_storage_mpc_heuristic(storage, soc0, price_forecast, dt=0.25,
                                terminal_price=None,
                                charge_discount=0.75, discharge_premium=1.30):
    """Fast threshold-based MPC — O(H) lookup instead of LP.

    Charge when the period's price is below ``charge_discount × mean``
    and discharge when above ``discharge_premium × mean``.  This
    replaces the LP solver for scenarios where millisecond latency
    matters more than optimality.

    Args:
        storage: StorageSpec
        soc0: initial SOC (p.u.)
        price_forecast: length-H price array
        dt: time step in hours
        terminal_price: unused (kept for API compatibility with solve_storage_mpc)
        charge_discount: price / mean below which to charge
        discharge_premium: price / mean above which to discharge

    Returns:
        (ch, dis, soc): arrays of length H (ch/dis) and H+1 (soc)
    """
    H = len(price_forecast)
    ch = np.zeros(H)
    dis = np.zeros(H)
    soc = np.zeros(H + 1)
    soc[0] = float(soc0)

    mean_price = float(np.mean(price_forecast))
    ch_threshold = charge_discount * mean_price
    dis_threshold = discharge_premium * mean_price

    for t in range(H):
        s = soc[t]
        # Compute feasible range at current SOC
        ch_max_t = storage.max_charge_feasible(s, dt)
        dis_max_t = storage.max_discharge_feasible(s, dt)

        price = float(price_forecast[t])
        if price < ch_threshold and ch_max_t > 0:
            ch[t] = ch_max_t
        elif price > dis_threshold and dis_max_t > 0:
            dis[t] = dis_max_t
        else:
            ch[t] = 0.0
            dis[t] = 0.0

        # Update SOC
        delta = (storage.eta_ch * ch[t] - dis[t] / storage.eta_dis) * dt / storage.e_max
        soc[t + 1] = float(np.clip(s + delta, storage.soc_min, storage.soc_max))

    return ch, dis, soc


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
    from strategies.random_bidding import RandomStrategy
    from grid import day_ahead_price_china

    # Base actions for all agents (fallback)
    base_actions = RandomStrategy().formulate(agents, config, T=T)

    # LMP forecast: use previous market LMP if available, else base price curve
    if market_history is not None:
        lmp_forecast = market_history.get("price", day_ahead_price_china(T, agents=agents, config=config))
    else:
        lmp_forecast = day_ahead_price_china(T, agents=agents, config=config)

    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return base_actions

    dt = 0.25

    for a in storage_agents:
        bid_mult = np.ones(T)
        offer_adder = np.full(T, np.mean(config.market_design.offer_adder_range))

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
                    config.market_design.offer_adder_range[0] + 5.0,
                    *config.market_design.offer_adder_range
                )
                bid_mult[t] = np.clip(0.85, *config.market_design.bid_mult_range)
            elif ch_power > 0.001:
                # MPC wants to charge: raise bid_mult to buy more
                bid_mult[t] = np.clip(1.15, *config.market_design.bid_mult_range)
                offer_adder[t] = np.clip(
                    config.market_design.offer_adder_range[1] - 5.0,
                    *config.market_design.offer_adder_range
                )
            else:
                # Idle: moderate parameters
                bid_mult[t] = np.mean(config.market_design.bid_mult_range)
                offer_adder[t] = np.mean(config.market_design.offer_adder_range)

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
    wholesale = __import__('grid').day_ahead_price_china(T, agents=agents, config=config)

    rt_step = max(1, config.rt.rt_step)
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
                        carbon_emissions += config.market_design.emission_factor_grid * p_grid * 0.25
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
        action_params_base, config
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
