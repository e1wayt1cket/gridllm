# dispatch_core.py
"""Storage constraints, shared helpers, and constants used by OPF solvers."""

# Pre-load ortools to work around DLL load order issue in pandapower
from ortools.linear_solver import pywraplp  # noqa: F401

from collections import deque

import numpy as np

from models import MarketConfig, Agent

# ===========================================================================
# Constants
# ===========================================================================

DT_HOURS = 0.25  # 15-min period in hours
STORAGE_MODE_THRESHOLD = 1e-6  # below this, ch/dis is treated as zero


# ===========================================================================
# Storage constraint engine
# ===========================================================================
class StorageConstraints:
    @staticmethod
    def feasible_ranges(storage, soc, dt=0.25):
        p_ch_max = storage.max_charge_feasible(soc, dt)
        p_dis_max = storage.max_discharge_feasible(soc, dt)
        p_ch_min = storage.p_ch_min if storage.p_ch_min < p_ch_max else 0.0
        p_dis_min = storage.p_dis_min if storage.p_dis_min < p_dis_max else 0.0
        return p_ch_min, p_ch_max, p_dis_min, p_dis_max

    @staticmethod
    def apply_ramp(prev_p_ch, prev_p_dis, target_p_ch, target_p_dis, storage):
        prev_signed = prev_p_dis - prev_p_ch
        target_signed = target_p_dis - target_p_ch
        ramp_limit = storage.ramp_up_dis if storage.ramp_up_dis else float('inf')
        delta = target_signed - prev_signed
        if abs(delta) > ramp_limit:
            feasible_signed = prev_signed + np.sign(delta) * ramp_limit
        else:
            feasible_signed = target_signed
        if feasible_signed < -1e-6:
            return min(abs(feasible_signed), storage.p_ch_max), 0.0
        elif feasible_signed > 1e-6:
            return 0.0, min(feasible_signed, storage.p_dis_max)
        else:
            return 0.0, 0.0

    @staticmethod
    def execute_dispatch(storage, soc, p_ch, p_dis, dt=0.25):
        p_ch = max(0.0, min(p_ch, storage.p_ch_max))
        p_dis = max(0.0, min(p_dis, storage.p_dis_max))
        p_ch_min, p_ch_max, p_dis_min, p_dis_max = StorageConstraints.feasible_ranges(storage, soc, dt)
        if p_ch > p_ch_max: p_ch = p_ch_max
        if 0 < p_ch < p_ch_min: p_ch = 0.0
        if p_dis > p_dis_max: p_dis = p_dis_max
        if 0 < p_dis < p_dis_min: p_dis = 0.0
        energy_in = p_ch * storage.eta_ch * dt if p_ch > 0 else 0.0
        energy_out = p_dis / storage.eta_dis * dt if p_dis > 0 else 0.0
        soc_change = (energy_in - energy_out) / storage.e_max
        new_soc = soc + soc_change
        new_soc = max(storage.soc_min, min(storage.soc_max, new_soc))
        return p_ch, p_dis, new_soc


# ===========================================================================
# Shared OPF helpers
# ===========================================================================


def _build_line_params(net, base_kv):
    """Compute line reactance (pu) and thermal limits from network data.

    Returns (x, limit) dicts keyed by line index.
    For DC-OPF only: reactance x[l] = X_ohm / V2.
    """
    V2 = base_kv ** 2
    x = {}
    limit = {}
    for l in net.line.index:
        length = net.line.at[l, 'length_km']
        X_ohm = net.line.at[l, 'x_ohm_per_km'] * length
        x[l] = X_ohm / V2
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka
    return x, limit


def _build_line_params_full(net, base_kv):
    """Compute line r, x, limit dicts for LinDistFlow (includes resistance).

    Returns (r, x, limit) dicts keyed by line index. All values in per-unit.
    """
    V2 = base_kv ** 2
    r = {}
    x = {}
    limit = {}
    for l in net.line.index:
        length = net.line.at[l, 'length_km']
        r[l] = net.line.at[l, 'r_ohm_per_km'] * length / V2
        x[l] = net.line.at[l, 'x_ohm_per_km'] * length / V2
        max_i_ka = net.line.at[l, 'max_i_ka']
        limit[l] = np.sqrt(3) * base_kv * max_i_ka
    return r, x, limit


def _classify_storage_mode(ch_val, dis_val):
    """Return 'charge', 'discharge', or 'idle' from solved power values."""
    if dis_val > STORAGE_MODE_THRESHOLD:
        return 'discharge'
    elif ch_val > STORAGE_MODE_THRESHOLD:
        return 'charge'
    return 'idle'


def _select_radial_lines(net):
    """Select the radial backbone lines via BFS from the slack bus.

    Avoids relying on line index ordering (e.g. list(net.line.index)[:n_buses-1])
    which breaks if the network is reordered or non-standard. Returns a list of
    line indices that form the radial spanning tree.
    """
    slack_bus = net.ext_grid.at[0, 'bus']
    adj = {}
    for idx in net.line.index:
        f = int(net.line.at[idx, 'from_bus'])
        t = int(net.line.at[idx, 'to_bus'])
        adj.setdefault(f, []).append((t, idx))
        adj.setdefault(t, []).append((f, idx))
    radial_lines = []
    visited = {slack_bus}
    queue = deque([slack_bus])
    while queue:
        bus = queue.popleft()
        for nb, lidx in adj.get(bus, []):
            if nb not in visited:
                visited.add(nb)
                radial_lines.append(lidx)
                queue.append(nb)
    return radial_lines


def _build_agent_info(agents, t, stage, prev_soc, wholesale_t, action_params, config):
    """Build per-agent info dict for single-period solvers.

    Returns agent_info dict, with storage vars set up for MILP binary co-optimization
    (both ch_max and dis_max available; binary prevents simultaneous charge/discharge).
    """
    agent_info = {}
    for a in agents:
        load_val = a.load_forecast[t] if stage == "DA" else a.load_real[t]
        pv_max = a.pv_forecast[t] if stage == "DA" else a.pv_real[t]
        wind_max = (a.get_wind_forecast()[t] if stage == "DA" else a.get_wind_real()[t]) if a.has_wind else 0.0

        target_ch, target_dis = 0.0, 0.0
        if a.storage is not None:
            cur_soc = prev_soc.get(a.name, a.storage.soc0)
            _, p_ch_max, _, p_dis_max = StorageConstraints.feasible_ranges(a.storage, cur_soc)
            target_ch, target_dis = float(p_ch_max), float(p_dis_max)

        bid = a.bid_value
        offer = a.offer_cost
        ap = action_params.get(a.name, {})
        if ap:
            bid *= ap["bid_mult"][t] if isinstance(ap.get("bid_mult"), np.ndarray) else ap.get("bid_mult", 1.0)
            offer += ap["offer_adder"][t] if isinstance(ap.get("offer_adder"), np.ndarray) else ap.get("offer_adder", 0.0)

        agent_info[a.name] = {
            'bus': a.bus, 'load': load_val, 'pv_max': pv_max, 'wind_max': wind_max,
            'ch_max': target_ch, 'dis_max': target_dis,
            'bid': bid, 'offer': offer, 'has_storage': a.storage is not None,
            'storage_mode': 'idle',
        }
    return agent_info


# ===========================================================================
# Shared schedule utilities (used by batch solvers and market.py)
# ===========================================================================

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


# ===========================================================================
# MPC self-scheduling helpers (shared by LDF and SOCP batch solvers)
# ===========================================================================

def _compute_bus_electrical_distance(net):
    """Compute cumulative line resistance from slack bus to each bus."""
    import collections
    slack = int(net.ext_grid.at[0, 'bus'])
    adj = collections.defaultdict(list)
    for _, row in net.line.iterrows():
        f = int(row['from_bus'])
        t = int(row['to_bus'])
        r = float(row['r_ohm_per_km']) * float(row['length_km'])
        adj[f].append((t, r))
    r_cum = {slack: 0.0}
    stack = [slack]
    while stack:
        u = stack.pop()
        for v, r in adj.get(u, []):
            if v not in r_cum:
                r_cum[v] = r_cum[u] + r
                stack.append(v)
    return r_cum


def _run_mpc_pass(all_storage_agents, T, H, wholesale, term_price,
                  noise_pct, r_cum, r_max, config, mpc_schedules,
                  congestion_prices):
    """Run one pass of MPC for all storage agents.

    If congestion_prices is provided, it overrides the base wholesale
    price forecast with congestion-aware nodal prices.
    """
    use_heuristic = getattr(config.storage, 'mpc_fast_heuristic', False)
    if use_heuristic:
        from mpc_storage import solve_storage_mpc_heuristic as _mpc_solve
    else:
        from mpc_storage import solve_storage_mpc as _mpc_solve

    if not all_storage_agents:
        return

    # Pre-compute per-agent noise arrays (single RNG call per agent)
    noise = {}
    for a in all_storage_agents:
        rng = np.random.RandomState(hash(a.name) % 2**31)
        noise[a.name] = 1.0 + rng.normal(0, noise_pct, T)

    charge_discount = getattr(config.storage, 'charge_discount', 0.75)
    discharge_premium = getattr(config.storage, 'discharge_premium', 1.30)

    for a in all_storage_agents:
        mpc_ch = np.zeros(T)
        mpc_dis = np.zeros(T)
        mpc_soc_arr = np.zeros(T + 1)
        soc_now = a.storage.soc0
        mpc_soc_arr[0] = soc_now
        price_src = (congestion_prices.get(a.bus, wholesale)
                     if congestion_prices else wholesale)
        for t in range(T):
            remaining = min(H, T - t)
            if remaining <= 0:
                mpc_soc_arr[t + 1] = soc_now
                break
            price_fwd = price_src[t:t + remaining] * noise[a.name][t:t + remaining]
            if use_heuristic:
                ch_sched, dis_sched, soc_sched = _mpc_solve(
                    a.storage, soc_now, price_fwd, DT_HOURS,
                    terminal_price=term_price,
                    charge_discount=charge_discount,
                    discharge_premium=discharge_premium)
            else:
                ch_sched, dis_sched, soc_sched = _mpc_solve(
                    a.storage, soc_now, price_fwd, DT_HOURS,
                    terminal_price=term_price)
            mpc_ch[t] = ch_sched[0]
            mpc_dis[t] = dis_sched[0]
            soc_now = soc_sched[1]
            mpc_soc_arr[t + 1] = soc_now
        mpc_schedules[a.name] = (mpc_ch, mpc_dis, mpc_soc_arr)


def _compute_mpc_schedules(net, agents, T, wholesale, config, storage_units):
    """Two-pass MPC self-scheduling for storage agents.

    First pass estimates congestion, second pass adjusts with
    congestion-aware nodal prices. Returns dict of MPC schedules.
    """
    all_storage = [a for a in agents if a.storage is not None]
    if storage_units:
        all_storage.extend(storage_units)
    schedules = {}
    if not config.storage.self_schedule or not all_storage:
        return schedules
    H = config.storage.mpc_horizon
    noise_pct = config.storage.mpc_price_noise_pct / 100.0
    term_price = float(np.mean(wholesale))
    r_cum = _compute_bus_electrical_distance(net)
    r_max = max(r_cum.values()) if r_cum else 1.0
    _run_mpc_pass(all_storage, T, H, wholesale, term_price,
                  noise_pct, r_cum, r_max, config, schedules, None)
    # Second congestion-aware pass — disabled by default.
    # Storage arbitrage is driven by temporal price differences;
    # spatial congestion markups have negligible impact in radial
    # networks with adequate line capacity.  Enable via:
    #   config.storage.mpc_congestion_pass = True
    if getattr(config.storage, 'mpc_congestion_pass', False):
        total_ch_pass1 = np.zeros(T)
        for sched in schedules.values():
            total_ch_pass1 += sched[0]
        if total_ch_pass1.sum() > 1e-6:
            congestion_prices = {}
            for b in range(len(net.bus.index)):
                bus_r = r_cum.get(b, 0.0)
                markup = 1.0 + 0.30 * (bus_r / r_max) * (
                    total_ch_pass1 / max(total_ch_pass1.max(), 0.01))
                congestion_prices[b] = wholesale * markup
            schedules.clear()
            _run_mpc_pass(all_storage, T, H, wholesale, term_price,
                          noise_pct, r_cum, r_max, config, schedules,
                          congestion_prices)
    return schedules
