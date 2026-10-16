# dispatch_core.py
"""Storage constraints, shared helpers, and constants used by OPF solvers."""

# Pre-load ortools to work around DLL load order issue in pandapower
from ortools.linear_solver import pywraplp  # noqa: F401

from collections import deque

import numpy as np

import money

from agent_seed import seed_for_agent
from models import MarketConfig, Agent
from rl_spec import ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2

# ===========================================================================
# Constants
# ===========================================================================

DT_HOURS = money.DT_HOURS  # 15-min period in hours; see money.py for the unit rule
STORAGE_MODE_THRESHOLD = 1e-6  # below this, ch/dis is treated as zero


# ===========================================================================
# Horizon semantics
# ===========================================================================
HORIZON_TYPES = ("full_day", "window")


def pins_terminal_soc(config, horizon_type: str) -> bool:
    """Whether this clear pins every storage unit's final state of charge.

    Pinning makes a comparison like for like: a day that finishes with a fuller
    battery has banked energy it never sold, so cash alone would reward
    whichever policy ran the battery down. It belongs to the settled horizon
    and not to a rolling window, where the pin would forbid the trajectory the
    window was opened to choose.

    The horizon is stated, never inferred. It used to be guessed from the
    period count, which is wrong in both directions: a rolling window that
    happens to be a whole day long was pinned, and a whole day cleared at any
    other period length was not. The guess is gone, so a caller cannot buy the
    wrong semantics by passing a familiar length.
    """
    if horizon_type not in HORIZON_TYPES:
        raise ValueError(
            f"unknown horizon_type {horizon_type!r}; expected one of "
            f"{HORIZON_TYPES}")
    if not config.storage.terminal_soc_equal:
        return False
    return horizon_type == "full_day"


# ===========================================================================
# Storage: one power flow, decomposed into charging and discharging
# ===========================================================================
def add_storage_net_power(m, nm: str, storage, T: int, ch, dis):
    """Give a storage unit one net power flow and tie charge/discharge to it.

    A battery has one power flow. Modelling it as two independent variables
    needs something extra to stop both being positive at once, and every such
    device so far has been an artificial rule bolted onto the quotes: a charge
    bid capped at the discharge offer, or a binary per unit-period. The cap
    worked, but it worked by forbidding exactly the price spread a battery uses
    to express arbitrage, so under it storage charged and never discharged.

    Here the two directions are the parts of one variable. `net` is free over
    the unit's own envelope, positive meaning discharge, and charge and
    discharge are held at or above its negative and positive parts. Both are
    pushed down to those parts by their objective coefficients, which are
    negative on every leg the clearing pays (the declared charge bid, the
    declared discharge offer, and the degradation charge), so at any optimum
    `ch = max(-net, 0)` and `dis = max(net, 0)`. Two non-negative numbers
    cannot both be the parts of one number, so overlapping charge and discharge
    is not forbidden by a rule -- it cannot be written down.

    This also makes the unit's arbitrage expressible: charging is costed at the
    declared bid and discharging at the declared offer, so the two quotes are
    the prices between which the unit will cycle, and they may cross. The cost
    is that a higher charge bid makes a unit *less* willing to charge, which is
    the opposite of what the same quote meant when charging was rewarded.

    Returns the net variable, a Gurobi tupledict indexed by period. Callers keep
    their existing charge/discharge expressions: at an optimum `dis - ch` is
    exactly `net`, so the power balance and the state-of-charge transition need
    no rewriting.
    """
    net = m.addVars(T, lb=-storage.p_ch_max, ub=storage.p_dis_max,
                    name=f"net_{nm}")
    for t in range(T):
        m.addConstr(ch[nm][t] >= -net[t], f"net_ch_{nm}_{t}")
        m.addConstr(dis[nm][t] >= net[t], f"net_dis_{nm}_{t}")
    return net


def check_storage_quote_signs(bid: float, offer: float, cycle_cost: float,
                              where: str):
    """Refuse quotes under which the decomposition above is not exact.

    The parts are pinned to the net flow by their objective coefficients being
    negative. A negative quote flips one of them positive, and the optimizer
    then drives that auxiliary to its *upper* bound instead -- buying objective
    value with a dispatch that cancels in the power balance. A zero quote is
    only safe while something else on the same leg stays negative.
    """
    if bid < 0 or offer < 0:
        raise ValueError(
            f"{where}: storage quotes must be non-negative, got bid={bid}, "
            f"offer={offer}")
    if bid + cycle_cost <= 0 or offer + cycle_cost <= 0:
        raise ValueError(
            f"{where}: storage needs a strictly negative objective coefficient "
            f"on both legs to pin charge and discharge to the net flow; with "
            f"bid={bid}, offer={offer}, cycle_cost={cycle_cost} one leg is not")


# ===========================================================================
# Grid exchange
# ===========================================================================
def grid_import_export(net_mw):
    """Split a net exchange into non-negative import and export.

    Positive draws from the upper grid, negative is a reverse flow. The two are
    the positive and negative parts of one quantity, so they cannot both be
    non-zero -- that used to be a property the solution happened to have, and is
    now a property of the representation.

    Returns (import_mw, export_mw), each matching the shape of the input.
    """
    net = np.asarray(net_mw, dtype=float)
    return np.maximum(net, 0.0), np.maximum(-net, 0.0)


def grid_carbon_tco2(net_mw, emission_factor: float) -> float:
    """Carbon over a horizon from the import part of the net exchange.

    Charging on the gross import of a model that carries import and export as
    two variables overstates this by the amount by which the solver inflated
    both together, which it was free to do because they shared a price.
    """
    imported, _ = grid_import_export(net_mw)
    return float(np.sum(imported) * DT_HOURS * emission_factor)


def add_grid_exchange(m, T, config, prefix: str = "p_grid"):
    """A single free net-exchange variable plus the import auxiliary.

    Import is unbounded and reverse flow is capped by
    `reverse_power_limit_mw`, which for one variable is the same statement as
    bounding it below. Pricing a single variable at the wholesale price removes
    the direction the objective used to be flat in: with import and export
    separate and equally priced, raising both by the same amount changed
    nothing, so the solver parked export at its cap in every period and
    inflated import to match.

    The auxiliary carries the carbon charge. It is bounded below by the
    exchange and by zero, and every objective coefficient it appears under is
    non-positive (a carbon price, or the penalty on a carbon-cap slack that is
    driven to zero), so the optimizer drives it down to exactly
    `max(exchange, 0)` without needing a binary variable. Where no carbon term
    is active at all the auxiliary is unused and may sit anywhere at or above
    its lower bound, which is why reported carbon is computed from the exchange
    in `grid_carbon_tco2` rather than read back off this variable.

    Returns (p_grid, imp_aux), both Gurobi tupledicts indexed by period.
    """
    # Imported here rather than at module scope: this module is also loaded on
    # paths that run without Gurobi, and the callers of this helper are the only
    # ones that need it.
    import gurobipy as gp

    p_grid = m.addVars(T, lb=-config.network.reverse_power_limit_mw,
                       ub=gp.GRB.INFINITY, name=prefix)
    imp_aux = m.addVars(T, lb=0, ub=gp.GRB.INFINITY, name=f"{prefix}_import")
    for t in range(T):
        m.addConstr(imp_aux[t] >= p_grid[t], f"{prefix}_import_lb_{t}")
    return p_grid, imp_aux


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


def quote_arrays(agent, action_params, T, config):
    """The prices the clearing costs this agent's charge and discharge at.

    Two ontologies, chosen by `participant_type` and never by whether the agent
    carries a battery. A prosumer with a battery keeps the declared-price pair
    it has always had: `bid_value` is what its load is worth to it and
    `offer_cost` what its generation must recover, both meaningful for that
    participant, shaded by `bid_mult` and `offer_adder`. A storage unit alone
    has neither of those things, so it declares its two reservation prices
    directly, in CNY/MWh.

    A storage agent whose action is missing a price raises rather than falling
    back. The old defaults (`ap.get("bid_mult", 1.0)`) are exactly how a unit
    would end up bidding a legacy multiplier where a price belongs -- silently,
    and only on the paths nobody re-checked.

    Returns `(bid_arr, offer_arr)`, each a length-T array.
    """
    nm = agent.name
    ap = action_params.get(nm, {}) or {}
    is_storage = getattr(agent, "participant_type", "prosumer") == "storage"

    if is_storage:
        # The retired pair is an error rather than a fallback. A caller still
        # passing `bid_mult` believes it is shading a price that no longer
        # exists, and would otherwise get the truthful declaration and never
        # learn that its action had stopped doing anything.
        legacy = [k for k in ("bid_mult", "offer_adder") if k in ap]
        if legacy:
            raise KeyError(
                f"storage agent {nm!r} was given {legacy}; a storage unit "
                f"declares its reservation prices as "
                f"'charge_price'/'discharge_price' in CNY/MWh, and "
                f"'bid_mult'/'offer_adder' only mean something for a load or a "
                f"prosumer")
        names = ACTION_STORAGE_QUOTE_V2.action_names
        charge_name, discharge_name = names
        declared = [k for k in names if k in ap]
        if not declared:
            # No declaration at all is the truthful one: a battery that names
            # no premium is asking for none. This is the reference point the
            # baselines are measured from, so it is the right thing for an
            # absent action to mean -- unlike a *partial* declaration, which
            # can only be a mistake.
            ap = dict(ACTION_STORAGE_QUOTE_V2.truthful)
        elif len(declared) != len(names):
            raise KeyError(
                f"storage agent {nm!r} declared only {declared}; both "
                f"reservation prices are required")
        bid_arr = _broadcast(ap[charge_name], T)
        offer_arr = _broadcast(ap[discharge_name], T)
    else:
        # The default is this space's own truthful point rather than a literal:
        # a multiplier of 1 and no adder are what "declare no premium" means
        # for a participant whose prices live in bid_value/offer_cost.
        bm = ap.get("bid_mult", ACTION_BID_OFFER_V1.truthful["bid_mult"])
        oa = ap.get("offer_adder", ACTION_BID_OFFER_V1.truthful["offer_adder"])
        bid_arr = _broadcast(bm, T) * agent.bid_value
        offer_arr = _broadcast(oa, T) + agent.offer_cost

    # Any agent carrying a battery is modelled with charge and discharge as the
    # parts of one net flow, and that decomposition is exact only while both
    # objective coefficients stay negative. A prosumer's declared prices are
    # positive by construction so this never fires for one, but the invariant
    # is a property of the mechanism rather than of the ontology.
    if getattr(agent, "storage", None) is not None:
        check_storage_quote_signs(float(np.min(bid_arr)),
                                  float(np.min(offer_arr)),
                                  config.storage.cycle_cost, f"storage {nm}")
    return bid_arr, offer_arr


def _broadcast(value, T):
    """A scalar or a per-period array, as a length-T float array."""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float)[:T]
    return np.full(T, float(value))


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

        bid_arr, offer_arr = quote_arrays(a, action_params, t + 1, config)
        bid, offer = float(bid_arr[t]), float(offer_arr[t])

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
    """Create a zero-filled schedules dict for the given agents and periods.

    `soc` is the per-period readout and `soc_boundary` the state *between*
    periods: `soc_boundary[0]` is the initial state and `soc_boundary[t + 1]`
    the state after period `t`. The two differ in length by one and the second
    is the one a transition can be verified against -- `soc[t + 1]` is not
    `soc[t]` after the transition of period `t`, it is a different quantity.

    `soc_terminal_target` is what the horizon was pinned to, if anything, and
    `soc_terminal_actual` what the solve actually reached. They are stored
    separately so a pinned horizon cannot make the target masquerade as a
    result.
    """
    schedules = {}
    for a in agents:
        schedules[a.name] = {
            'p_buy': np.zeros(T), 'p_sell': np.zeros(T),
            'served': np.zeros(T), 'unserved': np.zeros(T),
            'pv_used': np.zeros(T), 'wind_used': np.zeros(T),
            'p_ch': np.zeros(T), 'p_dis': np.zeros(T), 'soc': np.zeros(T),
            'q_re': np.zeros(T),
            'storage_mode': ['idle'] * T,
            'soc_boundary': np.zeros(T + 1),
            'soc_terminal_target': None,
            'soc_terminal_actual': None,
        }
    schedules['GRID'] = {'g_grid': np.zeros(T)}
    return schedules


# ===========================================================================
# Solver provenance
# ===========================================================================

#: Gurobi status codes, spelled. A bare integer is not a report: its meaning
#: belongs to the library version, and a reader cannot tell `3` from `9`
#: without one.
GUROBI_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
    12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS",
    15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}

#: Status codes that mean the solver ran out of something rather than that the
#: problem has no answer.
_LIMIT_CODES = (7, 8, 9, 10, 11, 15, 16, 17)


def categorize_termination(code) -> str:
    """Which kind of ending a status code is, in words a report can use.

    Distinguishing "there is no feasible answer" from "the solver stopped
    early" matters: the first is a fact about the model, the second is a fact
    about the run and is usually fixable by allowing more of the resource.
    """
    if not isinstance(code, int):
        return "unknown"
    if code == 2:
        return "optimal"
    if code in (3, 4):
        return "infeasible_or_unbounded"
    if code in _LIMIT_CODES:
        return "limit_reached"
    if code == 12:
        return "numerical_trouble"
    if code == 13:
        return "suboptimal"
    if code == 5:
        return "unbounded"
    if code in (1, 14):
        return "in_progress"
    return "no_solution"


def gurobi_solver_record(model, name: str = "gurobi") -> dict:
    """One Gurobi solve's own outcome, as serializable data.

    Deliberately does NOT report whether duals were usable. A model can return
    OPTIMAL and still hand back no power-balance duals -- that is the whole
    reason `BarHomogeneous` is set -- so a caller that needs prices must ask
    about duals, not about the status. Conflating the two would let a
    price-less result pass as a successful one.
    """
    code = int(getattr(model, "status", -1))
    return {
        "name": name,
        "status_code": code,
        "status_name": GUROBI_STATUS_NAMES.get(code, f"UNKNOWN_{code}"),
        "termination": categorize_termination(code),
        "optimal": code == 2,
    }


def fallback_solver_record(name: str, status_code=None, status_name=None,
                           message=None) -> dict:
    """A solver that is not Gurobi, described through the same fields.

    The per-period fallback is a different engine solving a different problem
    -- DC-OPF has no voltage variables and does not model the independent fleet
    -- so its status is recorded in its own right rather than folded into
    Gurobi's.
    """
    return {
        "name": str(name),
        "status_code": status_code,
        "status_name": status_name,
        "termination": categorize_termination(status_code),
        "optimal": bool(status_name == "OPTIMAL") if status_name else None,
        "message": message,
    }


#: The most recent batch solve's own outcome, from whichever engine ran.
#:
#: A batch engine returns None when it cannot clear, so its caller has no result
#: to read a status from -- and "the batch returned nothing" is not a diagnosis.
#: Rather than make the caller guess, the engine records what its solver said
#: here. This is a last-call cache, not configuration: overwritten on every
#: solve, never read to decide anything, and it exists only so a failure can be
#: reported with its own status code.
_LAST_SOLVE: dict = {}


def record_last_solve(engine: str, record: dict) -> None:
    """Remember what the most recent batch solve returned."""
    _LAST_SOLVE.clear()
    _LAST_SOLVE.update(record)
    _LAST_SOLVE["engine"] = str(engine)


def last_solve_metadata() -> dict:
    """A copy of the most recent batch solve's outcome, or an empty dict."""
    return dict(_LAST_SOLVE)


#: Why a period's price is not a nodal one. Named per cause, because "some
#: prices fell back" is not a finding a reader can act on: a period priced off
#: the wholesale curve and a period whose solve never returned need different
#: responses, and the price column alone cannot tell them apart.
LMP_FALLBACK_NO_DUAL = (
    "no power-balance dual was available for the period; it is priced off the "
    "wholesale curve")
LMP_FALLBACK_UNSOLVED = (
    "the period did not solve; its prices were carried forward from the "
    "previous period")


def lmp_fallback_report(n_fallbacks: int, periods, reason: str) -> dict:
    """The fields that say whether a price field is nodal or partly not.

    A day whose prices came off the wholesale curve for some periods is not a
    nodal price field, and every reader of `lmp` has to be able to tell. The
    count alone could not say so: it reports how many cells fell back rather
    than which periods, it was absent altogether on the paths that never
    extracted a dual, and a caller that summed a day's prices had no way to
    know part of it was not that day's market.

    `n_fallbacks` counts cells, one per (period, bus) that fell back, which is
    what the field has always meant. `periods` lists the distinct periods that
    contain at least one such cell, which is the unit a reader reasons in.
    """
    periods = sorted({int(t) for t in periods})
    return {
        "lmp_fallbacks": int(n_fallbacks),
        "lmp_fallback_periods": periods,
        "lmp_fallback_reason": reason if periods else "",
        "has_wholesale_fallback": bool(periods),
        "lmp_price_source": "mixed_wholesale_fallback" if periods else "nodal",
    }


def solver_metadata(engine: str, primary: dict, used_fallback: bool = False,
                    fallback: dict = None, duals_available=None,
                    duals_note: str = None) -> dict:
    """The `solver` block a clearing result carries.

    `final` names whichever solver actually produced the returned numbers. It
    is not an alias for `primary`: after a fallback the primary failed, and
    reporting its status as the run's outcome would describe a solve whose
    answer was thrown away.
    """
    final = fallback if (used_fallback and fallback is not None) else primary
    return {
        "engine": str(engine),
        "primary": primary,
        "used_fallback": bool(used_fallback),
        "fallback": fallback,
        "final": final,
        "duals_available": duals_available,
        "duals_note": duals_note or (
            "whether a power-balance dual was available for every period; a "
            "dual is what an LMP is, so False means some price fell back to "
            "the wholesale curve rather than being nodal"),
    }


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
        rng = np.random.RandomState(seed_for_agent(a.name))
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
