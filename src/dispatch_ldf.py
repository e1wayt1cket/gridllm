# dispatch_ldf.py
"""LinDistFlow OPF solvers: single-period + multi-period batch with storage SOC coupling."""

from collections import deque

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False
    gp = None  # type: ignore
    GRB = None  # type: ignore

from dispatch_core import (
    DT_HOURS,
    _build_line_params_full,
    _build_agent_info,
    _classify_storage_mode,
    _select_radial_lines,
    _compute_mpc_schedules,
    empty_schedules,
    split_power,
)


# ===========================================================================
# LinDistFlow (single-period, for fallback or RT rolling)
# ===========================================================================
def solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                             action_params, config):
    if not _HAS_GUROBI:
        raise RuntimeError("LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")
    buses = list(net.bus.index)
    n_buses = len(buses)
    lines = _select_radial_lines(net)
    slack_bus = net.ext_grid.at[0, 'bus']
    base_kv = config.network.base_kv

    # Filter to radial lines only
    r, x_val, limit = _build_line_params_full(net, base_kv)
    r = {l: v for l, v in r.items() if l in lines}
    x_val = {l: v for l, v in x_val.items() if l in lines}
    limit = {l: v for l, v in limit.items() if l in lines}

    agent_info = _build_agent_info(agents, t, stage, prev_soc, wholesale_t,
                                    action_params, config)

    m = gp.Model("LinDistFlow")
    m.setParam('OutputFlag', 0)

    V = m.addVars(buses, lb=config.network.v_min_pu, ub=config.network.v_max_pu, name="V")
    P = m.addVars(lines, lb=-GRB.INFINITY, name="P")
    Q = m.addVars(lines, lb=-GRB.INFINITY, name="Q")
    q_grid = m.addVar(lb=-GRB.INFINITY, name="q_grid")
    p_loss = m.addVars(buses, lb=0, ub=GRB.INFINITY, name="p_loss")
    m.addConstr(V[slack_bus] == 1.0, "ref_voltage")

    served = {}; unserved = {}; pv = {}; wind = {}; ch = {}; dis = {}
    for a in agents:
        nm = a.name
        info = agent_info[nm]
        served[nm] = m.addVar(lb=0, ub=info['load'], name=f"served_{nm}")
        unserved[nm] = m.addVar(lb=0, ub=info['load'], name=f"unserv_{nm}")
        m.addConstr(served[nm] + unserved[nm] == info['load'], f"load_bal_{nm}")
        pv[nm] = m.addVar(lb=0, ub=info['pv_max'], name=f"pv_{nm}")
        wind[nm] = m.addVar(lb=0, ub=info['wind_max'], name=f"wind_{nm}")
        if info['has_storage'] and (info['ch_max'] > 0 or info['dis_max'] > 0):
            ch[nm] = m.addVar(lb=0, ub=info['ch_max'], name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=info['dis_max'], name=f"dis_{nm}")
        else:
            ch[nm] = m.addVar(lb=0, ub=0, name=f"ch_{nm}")
            dis[nm] = m.addVar(lb=0, ub=0, name=f"dis_{nm}")
    p_grid_import = m.addVar(lb=0, ub=GRB.INFINITY, name="p_grid_import")
    p_grid_export = m.addVar(lb=0, ub=config.network.reverse_power_limit_mw, name="p_grid_export")

    # Reactive power from DER inverters (single-period)
    q_re = {}
    if config.network.reactive_support:
        for a in agents:
            nm = a.name
            if a.is_prosumer or a.storage is not None:
                q_re[nm] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                    name=f"q_re_{nm}")

    pf = config.network.load_power_factor
    q_ratio = np.tan(np.arccos(pf))

    # Active power balance (with losses)
    inj_p = {b: gp.LinExpr() for b in buses}
    inj_p[slack_bus] += p_grid_import - p_grid_export
    for a in agents:
        nm = a.name
        bus = agent_info[nm]['bus']
        inj_p[bus] += pv[nm] + wind[nm] + dis[nm] - served[nm] - ch[nm]
    for b in buses:
        flow_out = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                flow_out += P[l]
            elif net.line.at[l, 'to_bus'] == b:
                flow_out -= P[l]
        m.addConstr(inj_p[b] - p_loss[b] == flow_out, f"p_balance_{b}")
        p_loss[b].UB = 0  # fixed to zero on first pass

    # Reactive power balance
    q_inj = {b: gp.LinExpr() for b in buses}
    q_inj[slack_bus] += q_grid
    for a in agents:
        nm = a.name
        bus = agent_info[nm]['bus']
        q_inj[bus] -= served[nm] * q_ratio
        if config.network.reactive_support and nm in q_re:
            q_inj[bus] += q_re[nm]
    for b in buses:
        q_flow = gp.LinExpr()
        for l in lines:
            if net.line.at[l, 'from_bus'] == b:
                q_flow += Q[l]
            elif net.line.at[l, 'to_bus'] == b:
                q_flow -= Q[l]
        m.addConstr(q_inj[b] == q_flow, f"q_balance_{b}")

    # Voltage drop (P-Q coupled)
    for l in lines:
        f = net.line.at[l, 'from_bus']
        to_bus = net.line.at[l, 'to_bus']
        m.addConstr(V[f] - V[to_bus] == r[l] * P[l] + x_val[l] * Q[l], f"voltage_drop_{l}")

    # Line limits — diamond constraint |P|+|Q| <= S_max (linear inner
    # approximation of apparent-power circle, conservatively ~30% tighter
    # than the box P<=S, Q<=S which allows S up to sqrt(2)*limit).
    for l in lines:
        m.addConstr(P[l] + Q[l] <= limit[l], f"diamond_PpQp_{l}")
        m.addConstr(P[l] - Q[l] <= limit[l], f"diamond_PpQn_{l}")
        m.addConstr(-P[l] + Q[l] <= limit[l], f"diamond_PnQp_{l}")
        m.addConstr(-P[l] - Q[l] <= limit[l], f"diamond_PnQn_{l}")

    # Reactive power bounds from DER inverters
    # s_max uses INSTALLED capacity, not forecast, because inverters can
    # supply reactive power even when the resource is not generating.
    if config.network.reactive_support:
        for a in agents:
            nm = a.name
            if nm not in q_re:
                continue
            info = agent_info[nm]
            s_max = a.pv_capacity + a.wind_capacity
            if info['has_storage']:
                s_max += info['dis_max']
            if s_max > 0:
                q_re[nm].LB = -s_max
                q_re[nm].UB = s_max
                p_total = pv[nm] + wind[nm] + dis[nm]
                m.addConstr(q_re[nm] + p_total <= s_max, f"diamP_{nm}")
                m.addConstr(-q_re[nm] + p_total <= s_max, f"diamN_{nm}")

    obj = gp.LinExpr()
    for a in agents:
        nm = a.name
        obj += agent_info[nm]['bid'] * served[nm]
        if a.storage is not None:
            obj += wholesale_t * (dis[nm] - ch[nm])
        else:
            obj -= agent_info[nm]['offer'] * (pv[nm] + wind[nm] + dis[nm])
        obj -= config.market_design.penalty_unserved * unserved[nm]
    obj -= wholesale_t * (p_grid_import - p_grid_export)
    if config.market_design.enable_multi_objective and config.market_design.lambda_carbon > 0:
        obj -= config.market_design.lambda_carbon * config.market_design.emission_factor_grid * p_grid_import * DT_HOURS
    m.setObjective(obj, GRB.MAXIMIZE)

    # I2R loss iteration
    for loss_iter in range(max(1, config.network.n_loss_iters)):
        m.optimize()

        if m.status != GRB.OPTIMAL:
            return False, None, 0.0, {}, 0.0

        if loss_iter < config.network.n_loss_iters - 1:
            for b in buses:
                p_loss[b].UB = 0
                p_loss[b].LB = 0
            for l in lines:
                p_val = P[l].X
                q_val = Q[l].X
                loss_val = r[l] * (p_val ** 2 + q_val ** 2)
                to_bus = net.line.at[l, 'to_bus']
                p_loss[to_bus].UB += loss_val
                p_loss[to_bus].LB += loss_val

    lmp = np.zeros(len(buses))
    for i, b in enumerate(buses):
        constr = m.getConstrByName(f"p_balance_{b}")
        if constr is not None:
            lmp[i] = -constr.Pi
        else:
            lmp[i] = wholesale_t

    agent_res = {}
    for a in agents:
        nm = a.name
        ch_val = ch[nm].X
        dis_val = dis[nm].X
        agent_res[nm] = {
            'served': served[nm].X, 'pv_used': pv[nm].X, 'wind_used': wind[nm].X,
            'p_ch': ch_val, 'p_dis': dis_val,
            'storage_mode': _classify_storage_mode(ch_val, dis_val),
        }
    return True, lmp, m.ObjVal, agent_res, p_grid_import.X - p_grid_export.X


# ===========================================================================
# Multi-period joint optimization (batch LinDistFlow)
# ===========================================================================
def solve_lindist_opf_batch(net, agents, T, stage, config, action_params, wholesale,
                            storage_units=None):
    """
    Multi-period joint LinDistFlow with storage SOC transition and terminal value.
    Uses a persistent model cache to avoid rebuilding Gurobi model on repeated calls.
    """
    if not _HAS_GUROBI:
        raise RuntimeError("Batch LinDistFlow requires Gurobi; only DC-OPF has HiGHS fallback")

    mpc_schedules_ldf = _compute_mpc_schedules(net, agents, T, wholesale, config, storage_units)

    # Try cached model for repeated simulations with same topology
    su_list = storage_units or []
    e_maxes = tuple(a.storage.e_max for a in agents
                    if a.storage is not None)
    lm = config.network.line_capacity_multiplier
    key = _opf_cache_key(len(net.bus.index), T,
                         [a.name for a in agents],
                         [su.name for su in su_list],
                         e_maxes, lm)
    if key in _OPF_CACHE:
        cached = _OPF_CACHE[key]
        return cached.solve(agents, T, stage, config, action_params,
                           wholesale, su_list,
                           mpc_schedules=mpc_schedules_ldf)

    # First call: build and cache the model
    cached = _CachedOPFModel(net, agents, T, config, su_list)
    _OPF_CACHE[key] = cached
    return cached.solve(agents, T, stage, config, action_params,
                       wholesale, su_list,
                       mpc_schedules=mpc_schedules_ldf)



# ===========================================================================
# Persistent OPF model cache — eliminates repeated model-building overhead
# ===========================================================================

_OPF_CACHE: dict = {}  # key -> _CachedOPFModel
# Persistent OPF model cache — eliminates repeated model-building overhead
# ===========================================================================

_OPF_CACHE: dict = {}  # key -> _CachedOPFModel


def _opf_cache_key(n_buses, T, agent_names, su_names,
                   storage_e_maxes=(), line_mult=1.0):
    """Stable cache key including parameters that affect constraint structure."""
    return (n_buses, T, tuple(sorted(agent_names)),
            tuple(sorted(su_names)),
            tuple(sorted(storage_e_maxes)), line_mult)


class _CachedOPFModel:
    """Stores a built Gurobi OPF model and allows fast re-solve with updated data.

    Model building (variables + constraints) dominates OPF runtime (~5s).
    This class builds once, then reuses the model across simulations by
    updating only variable bounds and rebuilding the objective expression.
    """

    def __init__(self, net, agents, T, config, storage_units):
        self.net = net
        self.T = T
        self.n_buses = len(net.bus.index)
        self.buses = list(net.bus.index)
        self.lines = _select_radial_lines(net)
        self.slack_bus = net.ext_grid.at[0, 'bus']
        self.base_kv = config.network.base_kv

        r_all, x_all, limit_all = _build_line_params_full(net, self.base_kv)
        self.r = {l: v for l, v in r_all.items() if l in self.lines}
        self.x_val = {l: v for l, v in x_all.items() if l in self.lines}
        self.limit = {l: v for l, v in limit_all.items() if l in self.lines}

        self.m = gp.Model("LinDistFlow_batch_T_cached")
        self.m.setParam('OutputFlag', 0)
        self.m.setParam('Method', 1)  # dual simplex: deterministic + fast warm-start

        self._build_variables(agents, T, config, storage_units)
        self._build_constraints(agents, T, config, storage_units)

    def _build_variables(self, agents, T, config, storage_units):
        m = self.m
        buses = self.buses
        lines = self.lines

        self.V = m.addVars(T, buses, lb=config.network.v_min_pu,
                           ub=config.network.v_max_pu, name="V")
        self.P = m.addVars(T, lines, lb=-GRB.INFINITY, name="P")
        self.Q = m.addVars(T, lines, lb=-GRB.INFINITY, name="Q")
        self.p_grid_import = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                       name="p_grid_import")
        self.p_grid_export = m.addVars(T, lb=0,
                                       ub=config.network.reverse_power_limit_mw,
                                       name="p_grid_export")
        self.q_grid = m.addVars(T, lb=-GRB.INFINITY, name="q_grid")
        self.p_loss = m.addVars(T, buses, lb=0, ub=GRB.INFINITY, name="p_loss")

        self.served = {}; self.unserved = {}; self.pv = {}; self.wind = {}
        self.ch = {}; self.dis = {}; self.q_re = {}; self.soc = {}

        agents_list = list(agents)
        storage_agents = [a for a in agents_list if a.storage is not None]
        su_list = storage_units or []

        for a in agents_list:
            nm = a.name
            self.served[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                        name=f"served_{nm}")
            self.unserved[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                          name=f"unserv_{nm}")
            self.pv[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"pv_{nm}")
            self.wind[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                      name=f"wind_{nm}")
            self.ch[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY, name=f"ch_{nm}")
            self.dis[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                     name=f"dis_{nm}")

        if config.network.reactive_support:
            for a in agents_list:
                nm = a.name
                if a.is_prosumer or a.storage is not None:
                    self.q_re[nm] = m.addVars(T, lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              name=f"q_re_{nm}")

        for a in storage_agents:
            nm = a.name
            self.soc[nm] = m.addVars(T + 1, lb=a.storage.soc_min,
                                     ub=a.storage.soc_max, name=f"soc_{nm}")
            self.soc[nm][0].LB = a.storage.soc0
            self.soc[nm][0].UB = a.storage.soc0

        for su in su_list:
            nm = su.name
            self.ch[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                    name=f"ch_{nm}")
            self.dis[nm] = m.addVars(T, lb=0, ub=GRB.INFINITY,
                                     name=f"dis_{nm}")
            self.soc[nm] = m.addVars(T + 1, lb=su.storage.soc_min,
                                     ub=su.storage.soc_max, name=f"soc_{nm}")
            self.soc[nm][0].LB = su.storage.soc0
            self.soc[nm][0].UB = su.storage.soc0

        # Zero-out ch/dis for non-storage agents
        stor_names = {a.name for a in storage_agents} | {su.name for su in su_list}
        for a in agents_list:
            if a.name not in stor_names:
                for t in range(T):
                    self.ch[a.name][t].UB = 0
                    self.dis[a.name][t].UB = 0

        # Slack variables for constraint-based multi-objective —
        # created once so they persist across cached-model re-solves.
        self.slack_carbon = m.addVar(lb=0, ub=GRB.INFINITY, name="slack_carbon")
        self.slack_re = m.addVar(lb=0, ub=GRB.INFINITY, name="slack_re")

        self._storage_agents = storage_agents
        self._su_list = su_list
        self._agent_names = [a.name for a in agents_list]

    def _build_constraints(self, agents, T, config, storage_units):
        m = self.m
        buses = self.buses
        lines = self.lines
        slack_bus = self.slack_bus
        r = self.r
        x_val = self.x_val
        limit = self.limit
        agents_list = list(agents)
        storage_agents = self._storage_agents
        su_list = self._su_list
        served = self.served; unserved = self.unserved
        pv = self.pv; wind = self.wind; ch = self.ch; dis = self.dis
        soc = self.soc; q_re = self.q_re
        V = self.V; P = self.P; Q = self.Q
        p_grid_import = self.p_grid_import
        p_grid_export = self.p_grid_export
        q_grid_var = self.q_grid
        p_loss = self.p_loss

        pf = config.network.load_power_factor
        q_ratio = np.tan(np.arccos(pf))

        self._load_constrs = {}  # (t, nm) -> constraint obj
        self._carbon_constr = None
        self._re_constr = None

        for t in range(T):
            m.addConstr(V[t, slack_bus] == 1.0, f"ref_voltage_{t}")

            for b in buses:
                inj = gp.LinExpr()
                if b == slack_bus:
                    inj += p_grid_import[t] - p_grid_export[t]
                for a in agents_list:
                    if a.bus == b:
                        nm = a.name
                        inj += pv[nm][t] + wind[nm][t] + dis[nm][t] \
                               - served[nm][t] - ch[nm][t]
                for su in su_list:
                    if su.bus == b:
                        nm = su.name
                        inj += dis[nm][t] - ch[nm][t]
                flow = gp.LinExpr()
                for l in lines:
                    if self.net.line.at[l, 'from_bus'] == b:
                        flow += P[t, l]
                    elif self.net.line.at[l, 'to_bus'] == b:
                        flow -= P[t, l]
                m.addConstr(inj - p_loss[t, b] == flow, f"p_bal_{t}_{b}")
                p_loss[t, b].UB = 0

            for b in buses:
                q_inj = gp.LinExpr()
                if b == slack_bus:
                    q_inj += q_grid_var[t]
                for a in agents_list:
                    if a.bus == b:
                        nm = a.name
                        q_inj -= served[nm][t] * q_ratio
                        if config.network.reactive_support and nm in q_re:
                            q_inj += q_re[nm][t]
                q_flow = gp.LinExpr()
                for l in lines:
                    if self.net.line.at[l, 'from_bus'] == b:
                        q_flow += Q[t, l]
                    elif self.net.line.at[l, 'to_bus'] == b:
                        q_flow -= Q[t, l]
                m.addConstr(q_inj == q_flow, f"q_bal_{t}_{b}")

            for l in lines:
                f = self.net.line.at[l, 'from_bus']
                to = self.net.line.at[l, 'to_bus']
                m.addConstr(
                    V[t, f] - V[t, to] == r[l] * P[t, l] + x_val[l] * Q[t, l],
                    f"vdrop_{t}_{l}")

            for l in lines:
                m.addConstr(P[t, l] + Q[t, l] <= limit[l],
                            f"diam_PpQp_{t}_{l}")
                m.addConstr(P[t, l] - Q[t, l] <= limit[l],
                            f"diam_PpQn_{t}_{l}")
                m.addConstr(-P[t, l] + Q[t, l] <= limit[l],
                            f"diam_PnQp_{t}_{l}")
                m.addConstr(-P[t, l] - Q[t, l] <= limit[l],
                            f"diam_PnQn_{t}_{l}")

            for a in agents_list:
                nm = a.name
                load_val = (a.load_forecast[t] if True else a.load_real[t])
                c = m.addConstr(served[nm][t] + unserved[nm][t] == load_val,
                                f"load_{t}_{nm}")
                self._load_constrs[(t, nm)] = c

            for a in storage_agents:
                nm = a.name
                stor = a.storage
                eta_ch = stor.eta_ch; eta_dis = stor.eta_dis
                e_max = stor.e_max
                m.addConstr(
                    soc[nm][t + 1] == soc[nm][t]
                    + (eta_ch * ch[nm][t] - dis[nm][t] / eta_dis)
                    * DT_HOURS / e_max,
                    f"soctrans_{nm}_{t}")
                if t > 0:
                    if stor.ramp_up_ch is not None:
                        m.addConstr(ch[nm][t] - ch[nm][t - 1]
                                    <= stor.ramp_up_ch,
                                    f"ramp_up_ch_{nm}_{t}")
                    if stor.ramp_down_ch is not None:
                        m.addConstr(ch[nm][t - 1] - ch[nm][t]
                                    <= stor.ramp_down_ch,
                                    f"ramp_down_ch_{nm}_{t}")
                    if stor.ramp_up_dis is not None:
                        m.addConstr(dis[nm][t] - dis[nm][t - 1]
                                    <= stor.ramp_up_dis,
                                    f"ramp_up_dis_{nm}_{t}")
                    if stor.ramp_down_dis is not None:
                        m.addConstr(dis[nm][t - 1] - dis[nm][t]
                                    <= stor.ramp_down_dis,
                                    f"ramp_down_dis_{nm}_{t}")

            for su in su_list:
                nm = su.name
                stor = su.storage
                eta_ch = stor.eta_ch; eta_dis = stor.eta_dis
                e_max = stor.e_max
                m.addConstr(
                    soc[nm][t + 1] == soc[nm][t]
                    + (eta_ch * ch[nm][t] - dis[nm][t] / eta_dis)
                    * DT_HOURS / e_max,
                    f"soctrans_{nm}_{t}")
                if t > 0:
                    if stor.ramp_up_ch is not None:
                        m.addConstr(ch[nm][t] - ch[nm][t - 1]
                                    <= stor.ramp_up_ch,
                                    f"ramp_up_ch_{nm}_{t}")
                    if stor.ramp_down_ch is not None:
                        m.addConstr(ch[nm][t - 1] - ch[nm][t]
                                    <= stor.ramp_down_ch,
                                    f"ramp_down_ch_{nm}_{t}")
                    if stor.ramp_up_dis is not None:
                        m.addConstr(dis[nm][t] - dis[nm][t - 1]
                                    <= stor.ramp_up_dis,
                                    f"ramp_up_dis_{nm}_{t}")
                    if stor.ramp_down_dis is not None:
                        m.addConstr(dis[nm][t - 1] - dis[nm][t]
                                    <= stor.ramp_down_dis,
                                    f"ramp_down_dis_{nm}_{t}")

            if config.network.ramp_limit_mw_per_period is not None and t > 0:
                ramp = config.network.ramp_limit_mw_per_period
                net_t = p_grid_import[t] - p_grid_export[t]
                net_tm1 = p_grid_import[t - 1] - p_grid_export[t - 1]
                m.addConstr(net_t - net_tm1 <= ramp,
                            f"ramp_up_grid_{t}")
                m.addConstr(net_tm1 - net_t <= ramp,
                            f"ramp_down_grid_{t}")

    def solve(self, agents, T, stage, config, action_params, wholesale,
              storage_units, mpc_schedules=None):
        """Update bounds, rebuild objective, optimize, extract results."""
        m = self.m
        buses = self.buses
        lines = self.lines
        slack_bus = self.slack_bus
        n_buses = self.n_buses
        r = self.r
        agents_list = list(agents)
        storage_agents = self._storage_agents
        su_list = storage_units or []
        all_storage_names = {a.name for a in storage_agents} | {su.name for su in su_list}

        served = self.served; unserved = self.unserved
        pv = self.pv; wind = self.wind; ch = self.ch; dis = self.dis
        soc = self.soc; q_re = self.q_re
        V = self.V; P_line = self.P; Q_line = self.Q
        p_grid_import = self.p_grid_import
        p_grid_export = self.p_grid_export
        p_loss = self.p_loss

        # ---- Update variable bounds ----
        for a in agents_list:
            nm = a.name
            for t in range(T):
                load_val = (a.load_forecast[t] if stage == "DA"
                            else a.load_real[t])
                c = self._load_constrs.get((t, nm))
                if c is not None:
                    c.RHS = load_val

                pv_max = (a.pv_forecast[t] if stage == "DA"
                          else a.pv_real[t])
                wind_max = ((a.get_wind_forecast()[t] if stage == "DA"
                             else a.get_wind_real()[t])
                            if a.has_wind else 0.0)
                pv[nm][t].UB = max(pv_max, 0)
                wind[nm][t].UB = max(wind_max, 0)

        for a in storage_agents:
            nm = a.name
            stor = a.storage
            for t in range(T):
                ch[nm][t].UB = stor.p_ch_max
                dis[nm][t].UB = stor.p_dis_max
            soc[nm][0].LB = stor.soc0
            soc[nm][0].UB = stor.soc0

        for su in su_list:
            nm = su.name
            stor = su.storage
            for t in range(T):
                ch[nm][t].UB = stor.p_ch_max
                dis[nm][t].UB = stor.p_dis_max
            soc[nm][0].LB = stor.soc0
            soc[nm][0].UB = stor.soc0

        if config.network.reactive_support:
            for a in agents_list:
                nm = a.name
                if nm not in q_re:
                    continue
                s_max = a.pv_capacity + a.wind_capacity
                if a.storage:
                    s_max += a.storage.p_dis_max
                for t in range(T):
                    q_re[nm][t].LB = -s_max
                    q_re[nm][t].UB = s_max

        # ---- Fix storage variables to MPC schedules (must be AFTER general bounds) ----
        if mpc_schedules:
            for nm, (mpc_ch_vals, mpc_dis_vals, mpc_soc_arr) in mpc_schedules.items():
                if nm in ch:
                    for t in range(T):
                        ch[nm][t].LB = mpc_ch_vals[t]
                        ch[nm][t].UB = mpc_ch_vals[t]
                        dis[nm][t].LB = mpc_dis_vals[t]
                        dis[nm][t].UB = mpc_dis_vals[t]
                        soc[nm][t].LB = mpc_soc_arr[t]
                        soc[nm][t].UB = mpc_soc_arr[t]
                    soc[nm][T].LB = mpc_soc_arr[T]
                    soc[nm][T].UB = mpc_soc_arr[T]

        # ---- Constraint-based multi-objective ----
        total_re_avail_mwh = 0.0
        for a in agents_list:
            pv_arr = (a.pv_forecast if stage == "DA" else a.pv_real)
            wind_arr = ((a.wind_forecast if stage == "DA"
                         else a.wind_real) if a.has_wind
                        else np.zeros(T))
            total_re_avail_mwh += float(
                np.sum(np.maximum(pv_arr, 0))
                + np.sum(np.maximum(wind_arr, 0))) * DT_HOURS

        shadow_prices = {}
        if config.market_design.use_constraint_multi_obj:
            if config.market_design.carbon_cap_tco2 is not None:
                self.slack_carbon.UB = GRB.INFINITY
                carbon_expr = gp.LinExpr()
                for t in range(T):
                    carbon_expr += (config.market_design.emission_factor_grid
                                    * p_grid_import[t] * DT_HOURS)
                try:
                    cc = m.getConstrByName("carbon_cap")
                    if cc is not None:
                        m.remove(cc)
                except Exception:
                    pass
                m.addConstr(carbon_expr - self.slack_carbon
                            <= config.market_design.carbon_cap_tco2,
                            "carbon_cap")
            else:
                self.slack_carbon.UB = 0
            if (config.market_design.re_min_rate is not None
                    and total_re_avail_mwh > 0):
                self.slack_re.UB = GRB.INFINITY
                re_expr = gp.LinExpr()
                for t in range(T):
                    for a in agents_list:
                        nm = a.name
                        re_expr += (pv[nm][t] + wind[nm][t]) * DT_HOURS
                re_target = config.market_design.re_min_rate * total_re_avail_mwh
                try:
                    rc = m.getConstrByName("re_min_rate")
                    if rc is not None:
                        m.remove(rc)
                except Exception:
                    pass
                m.addConstr(re_expr + self.slack_re >= re_target,
                            "re_min_rate")
            else:
                self.slack_re.UB = 0

        # ---- Build objective ----
        bid_arr = {}; offer_arr = {}; pv_max_arr = {}; wind_max_arr = {}
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
            pv_max_arr[nm] = (a.pv_forecast if stage == "DA"
                              else a.pv_real)
            wind_max_arr[nm] = ((a.wind_forecast if stage == "DA"
                                 else a.wind_real)
                                if a.has_wind else np.zeros(T))

        if config.storage.terminal_value is not None:
            terminal_value = config.storage.terminal_value
        else:
            terminal_value = float(np.mean(wholesale))
        gamma = config.storage.discount_factor
        disc_T = gamma ** T

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
                    obj -= (config.market_design.lambda_carbon
                            * config.market_design.emission_factor_grid
                            * p_grid_import[t] * DT_HOURS)
                if config.market_design.lambda_re > 0:
                    for a in agents_list:
                        nm = a.name
                        obj += (config.market_design.lambda_re
                                * (pv[nm][t] + wind[nm][t]) * DT_HOURS)
                if config.market_design.lambda_curtail > 0:
                    for a in agents_list:
                        nm = a.name
                        curtail = ((pv_max_arr[nm][t] - pv[nm][t])
                                   + (wind_max_arr[nm][t] - wind[nm][t]))
                        obj -= (config.market_design.lambda_curtail
                                * curtail * DT_HOURS)
        # Constraint-slack penalties — charged only when constraints are active
        # and slack > 0, so the penalty cost equals the shadow price.
        if config.market_design.use_constraint_multi_obj:
            if config.market_design.carbon_cap_tco2 is not None:
                obj -= config.market_design.penalty_carbon_slack * self.slack_carbon
            if config.market_design.re_min_rate is not None:
                obj -= config.market_design.penalty_re_slack * self.slack_re
        for a in storage_agents:
            obj += disc_T * terminal_value * soc[a.name][T] * a.storage.e_max

        su_bid = {}; su_offer = {}
        for su in su_list:
            nm = su.name
            su_bid[nm] = np.full(T, su.bid_value)
            su_offer[nm] = np.full(T, su.offer_cost)
            for t in range(T):
                obj += su_bid[nm][t] * ch[nm][t]
                obj -= su_offer[nm][t] * dis[nm][t]
                if config.storage.cycle_cost > 0:
                    obj -= config.storage.cycle_cost * (ch[nm][t] + dis[nm][t])
            obj += disc_T * terminal_value * soc[nm][T] * su.storage.e_max

        m.setObjective(obj, GRB.MAXIMIZE)

        # ---- Loss iteration ----
        # Reset p_loss bounds to zero before starting
        for t in range(T):
            for b in buses:
                p_loss[t, b].UB = 0
                p_loss[t, b].LB = 0
        for loss_iter in range(max(1, config.network.n_loss_iters)):
            m.optimize()
            if m.status != GRB.OPTIMAL:
                return None
            if loss_iter < config.network.n_loss_iters - 1:
                for t in range(T):
                    for b in buses:
                        p_loss[t, b].UB = 0
                        p_loss[t, b].LB = 0
                    for l in lines:
                        p_val = P_line[t, l].X
                        q_val = Q_line[t, l].X
                        loss_val = r[l] * (p_val ** 2 + q_val ** 2)
                        to_bus = self.net.line.at[l, 'to_bus']
                        p_loss[t, to_bus].UB += loss_val
                        p_loss[t, to_bus].LB += loss_val

        # ---- Nodal price re-solve ----
        all_storage = storage_agents + su_list
        lmp = np.zeros((T, n_buses))
        if all_storage and config.storage.use_nodal_price:
            bus_to_idx = {b: i for i, b in enumerate(buses)}
            nodal_lmp = np.zeros((T, n_buses))
            for t in range(T):
                for i, b in enumerate(buses):
                    nodal_lmp[t, i] = -m.getConstrByName(f"p_bal_{t}_{b}").Pi

            for nodal_iter in range(4):
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
                            obj2 -= (config.market_design.lambda_carbon
                                     * config.market_design.emission_factor_grid
                                     * p_grid_import[t] * DT_HOURS)
                        if config.market_design.lambda_re > 0:
                            for a in agents_list:
                                nm = a.name
                                obj2 += (config.market_design.lambda_re
                                         * (pv[nm][t] + wind[nm][t]) * DT_HOURS)
                        if config.market_design.lambda_curtail > 0:
                            for a in agents_list:
                                nm = a.name
                                curtail = ((pv_max_arr[nm][t] - pv[nm][t])
                                           + (wind_max_arr[nm][t] - wind[nm][t]))
                                obj2 -= (config.market_design.lambda_curtail
                                         * curtail * DT_HOURS)
                for a in storage_agents:
                    obj2 += disc_T * terminal_value * soc[a.name][T] * a.storage.e_max
                for su in su_list:
                    nm = su.name
                    for t in range(T):
                        obj2 += su_bid[nm][t] * ch[nm][t]
                        obj2 -= su_offer[nm][t] * dis[nm][t]
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
                        new_lmp[t, i] = -m.getConstrByName(f"p_bal_{t}_{b}").Pi
                max_change = np.max(np.abs(new_lmp - nodal_lmp))
                nodal_lmp = new_lmp
                if max_change < 1.0:
                    break

        # ---- Extract results ----
        if config.market_design.use_constraint_multi_obj:
            try:
                cc = m.getConstrByName("carbon_cap")
                if cc is not None:
                    shadow_prices["carbon_cap"] = cc.Pi
            except Exception:
                pass
            try:
                rc = m.getConstrByName("re_min_rate")
                if rc is not None:
                    shadow_prices["re_min_rate"] = -rc.Pi
            except Exception:
                pass
            # Report slack usage: positive slack means the constraint was violated
            # and the penalty was paid.
            slack_c = self.slack_carbon.X if hasattr(self, 'slack_carbon') else 0.0
            slack_r = self.slack_re.X if hasattr(self, 'slack_re') else 0.0
            if slack_c > 1e-6:
                shadow_prices["carbon_slack_tco2"] = float(slack_c)
            if slack_r > 1e-6:
                shadow_prices["re_slack_mwh"] = float(slack_r)

        schedules = empty_schedules(agents_list, T)
        for su in su_list:
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
             + (np.sum(a.wind_forecast if stage == "DA"
                       else a.wind_real) if a.has_wind else 0))
            * DT_HOURS for a in agents_list)
        total_re_used = 0.0
        total_curtailment = 0.0
        carbon_emissions = 0.0
        total_served_mwh = 0.0

        # Build radial path tree
        parent_line = {}
        adj = {b: [] for b in buses}
        for l in lines:
            f = int(self.net.line.at[l, 'from_bus'])
            to = int(self.net.line.at[l, 'to_bus'])
            adj[f].append((to, l))
            adj[to].append((f, l))
        visited = {slack_bus}
        queue = deque([slack_bus])
        while queue:
            b = queue.popleft()
            for nb, l in adj[b]:
                if nb not in visited:
                    visited.add(nb)
                    parent_line[nb] = (b, l)
                    queue.append(nb)

        for t in range(T):
            for i, b in enumerate(buses):
                constr = m.getConstrByName(f"p_bal_{t}_{b}")
                if constr is not None:
                    lmp[t, i] = -constr.Pi
                else:
                    lmp[t, i] = wholesale[t]

            slack_idx = list(buses).index(slack_bus)
            base_lmp = lmp[t, slack_idx]
            for i, b in enumerate(buses):
                if b == slack_bus:
                    continue
                if abs(lmp[t, i] - base_lmp) > 1.0:
                    continue
                mlf = 0.0
                cur = b
                while cur != slack_bus:
                    parent_b, line_l = parent_line[cur]
                    p_flow = abs(P_line[t, line_l].X)
                    mlf += 2.0 * r[line_l] * p_flow
                    cur = parent_b
                lmp[t, i] = base_lmp * (1.0 + mlf)

            pgi = p_grid_import[t].X
            if pgi > 0:
                carbon_emissions += (config.market_design.emission_factor_grid
                                     * pgi * DT_HOURS)
            schedules['GRID']['g_grid'][t] = (p_grid_import[t].X
                                              - p_grid_export[t].X)

            for a in agents_list:
                nm = a.name
                s = schedules[nm]
                s['served'][t] = served[nm][t].X
                s['unserved'][t] = unserved[nm][t].X
                s['pv_used'][t] = pv[nm][t].X
                s['wind_used'][t] = wind[nm][t].X
                s['p_ch'][t] = ch[nm][t].X
                s['p_dis'][t] = dis[nm][t].X
                s['q_re'][t] = (q_re[nm][t].X
                                if config.network.reactive_support
                                and nm in q_re else 0.0)
                s['storage_mode'][t] = _classify_storage_mode(
                    ch[nm][t].X, dis[nm][t].X)
                if a.storage:
                    s['soc'][t] = soc[nm][t].X
                    if t == T - 1:
                        s['soc_final'] = soc[nm][T].X
                else:
                    s['soc'][t] = 0.0

                load_val = (a.load_forecast[t] if stage == "DA"
                            else a.load_real[t])
                pv_max = (a.pv_forecast[t] if stage == "DA"
                          else a.pv_real[t])
                wind_max = ((a.get_wind_forecast()[t] if stage == "DA"
                             else a.get_wind_real()[t])
                            if a.has_wind else 0.0)
                net_gen = s['pv_used'][t] + s['wind_used'][t] + s['p_dis'][t]
                net_con = s['served'][t] + s['p_ch'][t]
                s['p_buy'][t], s['p_sell'][t] = split_power(net_gen, net_con)
                total_re_used += (s['pv_used'][t] + s['wind_used'][t]) * DT_HOURS
                total_curtailment += ((pv_max - s['pv_used'][t])
                                      + (wind_max - s['wind_used'][t])) * DT_HOURS
                total_served_mwh += s['served'][t] * DT_HOURS

            for su in su_list:
                nm = su.name
                s = schedules[nm]
                s['p_ch'][t] = ch[nm][t].X
                s['p_dis'][t] = dis[nm][t].X
                s['soc'][t] = soc[nm][t].X
                s['storage_mode'][t] = _classify_storage_mode(
                    ch[nm][t].X, dis[nm][t].X)
                if t == T - 1:
                    s['soc_final'] = soc[nm][T].X
                net_gen_su = s['pv_used'][t] + s['wind_used'][t] + s['p_dis'][t]
                net_con_su = s['served'][t] + s['p_ch'][t]
                s['p_buy'][t], s['p_sell'][t] = split_power(net_gen_su,
                                                            net_con_su)
                total_served_mwh += s['served'][t] * DT_HOURS

        re_rate = ((total_re_used / total_re_avail * 100)
                   if total_re_avail > 0 else 100.0)
        carbon_intensity = carbon_emissions / max(total_served_mwh, 1e-6)
        return {
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
