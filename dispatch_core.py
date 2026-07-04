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
