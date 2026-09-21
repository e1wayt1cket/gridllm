# diagnose_typical_day.py
"""Physical baseline diagnostic for the typical day.

Clears one whole day jointly and reports what the day physically did: how big
the load was, where the renewable output went, how hard each line worked, how
far each bus voltage moved, and what each independent battery bought, sold and
cycled. The point is to be able to say whether the base case is sound before any
bidding policy is trained on it, and to be told when it is not.

Two quantities get particular attention because they are what a storage study
rests on and neither is visible from the cleared schedules alone:

  * Charge/discharge exclusivity. A battery cannot charge and discharge in the
    same period. The clearing relies on a quoting rule rather than an explicit
    exclusive-or, so this reports the overlap directly and treats any of it as a
    failure rather than assuming the rule held.

  * The arbitrage window. A battery only earns from a price spread when the
    peak-to-valley ratio covers the round trip, which for a 0.92/0.92 battery at
    cycle_cost 100 needs a ratio near 1.7. The window is reported per battery bus
    next to that breakeven so a day where every battery charges and none
    discharges is visible as what it is rather than as a successful clear.

Settlement is not re-derived here. Each battery's cash is taken from
`participant_payoff`, the single implementation the trainer and evaluator also
route through, and the funds-flow cross-check is `surplus_metrics.reconciliation`.

The day is cleared once for the whole horizon rather than through the training
environment: the environment drives a rolling window and stitches the committed
slices, which is a different object from a settled day.

Usage:
  PYTHONPATH=src python src/diagnose_typical_day.py
  PYTHONPATH=src python src/diagnose_typical_day.py --json outputs/typical_day.json
  PYTHONPATH=src python src/diagnose_typical_day.py --sweep-load-scale 1.6,1.75,1.914
  PYTHONPATH=src python src/diagnose_typical_day.py --sweep-elasticity 0.35,1.0,1.4,1.8
  PYTHONPATH=src python src/diagnose_typical_day.py --sweep-ess-buses all
  PYTHONPATH=src python src/diagnose_typical_day.py --search-ess-buses all

The last two are the placement experiment: `--sweep-ess-buses` clears one unit
at every candidate bus, and `--search-ess-buses` chooses four by hill-climbing
the share of the action box each keeps tradeable *in the fleet's company*. The
placement in `config/defaults.yaml` is the output of the second; the first is
the per-bus landscape and is not the selection metric (see `siting_sweep`).

Exits non-zero when a hard gate fails, so it can be wired into a check later.
"""

import argparse
import json
import sys

import numpy as np

import config_loader
import money
import participant_payoff
import surplus_metrics
from config_loader import (get_default, reload_defaults, reload_scenarios,
                           set_default, set_scenario_override)
from grid import build_base_network, day_ahead_price_china
from market import clear_market
from rl_physics import pairwise_resistance_distance
from rl_spec import ACTION_STORAGE_QUOTE_V2
from scenarios import get_scenario, typical_day_config

# The typical day's targets, in one place so the report and the gates agree.
# The load level is the scenario's own declaration now rather than a constant
# here, and rather than a global default: `native_peak` means the network's
# rated demand and `explicit_peak` a declared figure, and the profiles are
# scaled to hit whichever the scenario chose. Read at call time so a sweep that overrides the config is reported
# against the level it actually ran at.
# ---------------------------------------------------------------------------
# Siting eligibility
# ---------------------------------------------------------------------------
# A battery declares two reservation prices and the clearing costs charging at
# the first and discharging at the second, on top of `cycle_cost`. So the unit
# still cycles only while `charge_price + discharge_price` stays under the
# margin its own price window leaves over the degradation floor. That region is
# a triangle in the corner of the box, and its share of the box is the unit's
# tradeable strategy space: the fraction of its own action on which anything
# happens at all.
#
# This threshold is fixed **before** any siting measurement is run, so which
# buses turn out to be eligible cannot be reverse-engineered from the threshold
# afterwards. 25% is the plan's number; it is a judgement about how much of a
# strategy space a policy needs to be learnable, not a measured quantity.
TRADEABLE_BOX_SHARE = 0.25


def round_trip_margin_cny_mwh(node, eta_ch: float, eta_dis: float,
                              cycle_cost: float) -> float:
    """Margin over the degradation floor, per MWh *stored*, in CNY/MWh.

    One MWh stored costs `valley / eta_ch` to put in and releases `eta_dis` MWh
    at the peak price, so it earns `eta_dis * peak`; degradation is charged on
    both legs. Expressed per MWh stored rather than per MWh discharged so it
    compares directly against the `2 * cycle_cost` the pair of quotes has to
    cover inside the action box.
    """
    valley = float(np.min(node))
    peak = float(np.max(node))
    return eta_dis * peak - valley / eta_ch - 2.0 * cycle_cost


def tradeable_box_share(margin: float, box) -> float:
    """Share of the action box in which the unit still cycles.

    The tradeable region is `charge + discharge < margin` clipped to the box.
    Integrating the slice height `clamp(M - u, 0, H)` along the charge axis
    gives the area exactly, in the two pieces the clamp splits it into: a
    constant-height part where the slice has already hit the box's discharge
    side, and a linear part where it has not.
    """
    (c_lo, c_hi), (d_lo, d_hi) = box
    width, height = c_hi - c_lo, d_hi - d_lo
    if width <= 0 or height <= 0:
        return 0.0
    # Distance along the charge axis, measured from the box's lower corner,
    # over which charge + discharge still fits under the margin.
    m = margin - c_lo - d_lo
    if m <= 0:
        return 0.0
    full = min(width, max(0.0, m - height))   # slice already at full height
    linear = min(width, m)                    # slice still growing
    area = height * full + m * (linear - full) \
        - (linear ** 2 - full ** 2) / 2.0
    return area / (width * height)


def action_box():
    """The storage action box, as ((charge_lo, charge_hi), (disch_lo, disch_hi)).

    Read from the spec rather than restated: the siting rule is a statement
    about the space a policy actually declares in, so a change to the bounds
    has to move this with it.
    """
    spec = ACTION_STORAGE_QUOTE_V2
    return (spec.bounds[spec.action_names[0]],
            spec.bounds[spec.action_names[1]])


def price_source(scenario: str = "typical_day") -> str:
    """Which wholesale market prices this scenario.

    Read from the scenario's own declaration, because that is what
    `_build_scenario` writes into the price layer -- the defaults-level setting
    only applies to paths that never build a scenario.
    """
    from config_loader import get_scenario_cfg
    return str(get_scenario_cfg(scenario).get("wholesale_source", "legacy"))


def _target_peak_mw(scenario: str = "typical_day") -> float:
    """The coincident peak this scenario declares.

    Read from the scenario rather than from a default, because the load level
    is a scenario's own choice: `native_peak` means the network's rated demand
    and `explicit_peak` means the declared figure.
    """
    from config_loader import get_scenario_cfg
    cfg = get_scenario_cfg(scenario)
    mode = cfg.get("load_scale_mode")
    if mode == "native_peak":
        from grid import native_nominal_demand_mw
        from scenarios import typical_day_config
        from grid import build_base_network
        return native_nominal_demand_mw(
            build_base_network(typical_day_config(money.PERIODS_PER_DAY)))
    if mode == "explicit_peak":
        return float(cfg["peak_load_mw"])
    raise ValueError(f"scenario {scenario!r} declares no load scale")


def _declared_renewables(scenario: str = "typical_day"):
    """(pv_mw, wind_mw) this scenario declares, summed over its buses."""
    from config_loader import get_scenario_cfg
    renewable = get_scenario_cfg(scenario).get("renewable") or {}
    pv = sum(float(v) for v in (renewable.get("pv_bus_mw") or {}).values())
    wind = sum(float(v) for v in (renewable.get("wind_bus_mw") or {}).values())
    return pv, wind
ESS_POWER_TO_PEAK_BAND = (0.30, 0.42)
ESS_ENERGY_TO_LOAD_BAND = (0.08, 0.10)
MIN_RE_RATE = 90.0
CYCLES_BAND = (0.2, 0.9)
# The feeder is voltage-limited above roughly 8.3 MW of peak load at the stated
# +/-7% band, so serving 11 MW sheds a little load. Bounded, not zero: the point
# of the gate is to catch the day degrading, not to pretend nothing is shed.
# The shedding the day is allowed, armed at the level this baseline was accepted
# at. Zero is not reachable at a physical inverter rating: over the headroom
# sweep, shedding falls from 1.22 MWh at parity to nothing only at 2.50, i.e. a
# 1 MW array behind a 2.5 MVA inverter. The alternative was to cut the peak the
# day serves from 10 MW to about 6.1 MW. Both were rejected in favour of
# accepting roughly one megawatt-hour of shed load at a 1.10 rating, so the day
# is gated at what was accepted instead of at what would be convenient.
# `--unserved-limit` overrides it, including with 0.
UNSERVED_LIMIT_MWH = 1.1
INSTALLED_MW_TOL = 1e-6
SOC_CLOSURE_TOL = 1e-6
# Solver tolerance on the charge/discharge overlap; see the gate for why this is
# not an exact zero.
CHURN_TOL_MWH = 1e-4


def run_day(scenario="typical_day", T=money.PERIODS_PER_DAY, opf_mode="socp",
            load_scale=None, price_base=None, elasticity=None,
            inverter_headroom=None, seed_note=None, ess_buses=None,
            der_moves=None):
    """Clear one whole typical day and return everything the report reads.

    The keyword overrides are applied to the in-memory configuration caches, so
    calibration can sweep values without editing config/defaults.yaml. Both
    caches are reset first, so each run starts from what is on disk and a
    previous sweep cannot leak into this one.
    """
    reload_defaults()
    reload_scenarios()
    if load_scale is not None:
        set_default("profiles.load.load_scale", float(load_scale))
    if price_base is not None:
        set_default("price_curve.base", float(price_base))
    if elasticity is not None:
        set_default("price_curve.merit_order_price_elasticity", float(elasticity))
    if inverter_headroom is not None:
        # Written to the scenario cache, not to `config`, because the scenario
        # owns this key: `get_scenario` re-applies `override_config` onto any
        # config it is handed, so a value set on the config object alone is
        # overwritten by config/scenarios.yaml one line later and the sweep
        # reads as flat. Expressed where the file is read, the override is seen
        # by both readers -- `typical_day_config` here and `_build_scenario`
        # below -- and the two cannot disagree.
        set_scenario_override(scenario, "network.inverter_smax_multiplier",
                              float(inverter_headroom))

    config = typical_day_config(T)
    config.opf_mode = opf_mode
    agents, wholesale = get_scenario(scenario, T, config, ess_buses=ess_buses,
                                     der_moves=der_moves)
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale,
                          horizon_type="full_day" if T == money.PERIODS_PER_DAY
                          else "window")
    return {
        "config": config,
        "agents": agents,
        "wholesale": wholesale,
        # The level the day was built to, whichever way it was specified: an
        # explicit scale when the sweep gave one, otherwise the target peak the
        # profiles were scaled to reach.
        "load_level": ("load_scale %.4g" % load_scale if load_scale is not None
                       else "target peak %.4g MW" % _target_peak_mw(scenario)),
        "result": result,
        "scenario": scenario,
        "T": T,
        "load_scale": get_default("profiles.load.load_scale", 1.0),
        "price_base": get_default("price_curve.base"),
        "elasticity": get_default("price_curve.merit_order_price_elasticity"),
        # Which market priced this day. The two merit-order numbers above are
        # still returned because the legacy sweeps set them, but they describe
        # nothing unless this says `legacy`.
        "price_source": price_source(scenario),
    }


# --------------------------------------------------------------------------
# shed diagnostics
#
# Shed load is not evidence of an energy shortage. It can be any of: the day is
# short of energy; the network cannot carry what would be needed; a voltage or
# line limit binds; or the clearing simply chose not to serve a unit the objective
# valued below its cost. These two runs separate those, and neither replaces the
# economic market run -- they answer a different question about the same physics.
# --------------------------------------------------------------------------

#: The value-of-lost-load weight that makes the clearing lexicographic in shed.
#: Every other objective coefficient is a price or a declared bid, all of them
#: below 1e4 CNY/MWh in this study, so a weight this large puts shed strictly
#: ahead of every other term while leaving every constraint untouched.
MIN_SHED_VOLL = 1e7


def min_shed_run(scenario="typical_day", T=money.PERIODS_PER_DAY,
                 opf_mode="socp", ess_buses=None, **kwargs):
    """The same day solved with total shed ahead of every economic term.

    DIAGNOSTIC ONLY. The constraints are the ones the market clears against --
    same network, same line capacities, same voltage band -- and only the
    objective's ordering changes, so the residual shed this reports is what the
    *network* makes unservable rather than what the market chose not to buy.
    The dispatch it produces is not a market outcome and its prices are not
    nodal prices for any economic purpose.

    Compare it with the economic run: `shed_economic - shed_min` is the load the
    clearing declined to serve at the prices it faced, and `shed_min` is the
    part no price could have bought.
    """
    day = run_day(scenario=scenario, T=T, opf_mode=opf_mode,
                  ess_buses=ess_buses, **kwargs)
    config = day["config"]
    agents = day["agents"]
    wholesale = day["wholesale"]

    config.market_design.penalty_unserved = MIN_SHED_VOLL
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale,
                          horizon_type="full_day"
                          if T == money.PERIODS_PER_DAY else "window")

    shed_mwh = sum(float(np.sum(result["schedules"][a.name]["unserved"]))
                   for a in agents) * money.DT_HOURS
    return {
        "diagnostic_mode": "MIN_SHED_RUN",
        "not_a_market_outcome": True,
        "voll_used": MIN_SHED_VOLL,
        "scenario": scenario,
        "shed_mwh": shed_mwh,
        "curtailment_mwh": float(result.get("total_curtailment", 0.0)),
        "shed_economic_mwh": float(
            sum(np.sum(day["result"]["schedules"][a.name]["unserved"])
                for a in agents) * money.DT_HOURS),
        "result": result,
        "agents": agents,
        "config": config,
    }


def no_shed_voltage_diagnostic(scenario="typical_day",
                               T=money.PERIODS_PER_DAY, opf_mode="socp",
                               band=None, ess_buses=None, **kwargs):
    """Force every unit served and report what the network violates.

    DIAGNOSTIC ONLY, and not a market run at all: there is no clearing here.
    Each period is a power flow on the day's must-serve injections -- every
    unit's load served in full, every renewable at its available output -- so
    the answer is the network's, not the objective's. Whichever limit breaks
    first is what has to give if shed is to be zero.

    Shed is fixed at zero by construction. Nothing is tuned to reach it.
    """
    import pandapower as pp

    from dispatch_core import _select_radial_lines
    from grid import build_base_network

    day = run_day(scenario=scenario, T=T, opf_mode=opf_mode,
                  ess_buses=ess_buses, **kwargs)
    config = day["config"]
    agents = day["agents"]
    lo, hi = band if band else (config.network.v_min_pu,
                                config.network.v_max_pu)

    # A copy, because `build_base_network` hands back the same cached object to
    # every caller. This diagnostic rewrites every bus's load and adds a
    # generator per renewable bus, so working on the cache in place would leave
    # those edits in the network every later run in this process clears against.
    import copy as _copy

    net = _copy.deepcopy(build_base_network(config))
    pf = float(config.network.load_power_factor)
    tan_phi = float(np.tan(np.arccos(pf)))

    # The per-bus injections this day actually has: every unit's load, served
    # in full, and every renewable at the output it had. Aggregated by bus,
    # because the power flow's elements are per bus and the agents are not.
    load_bus: Dict[int, np.ndarray] = {}
    gen_bus: Dict[int, np.ndarray] = {}
    for a in agents:
        b = int(a.bus)
        arr = getattr(a, "load_real", None)
        if arr is not None:
            load_bus.setdefault(b, np.zeros(T))
            load_bus[b] = load_bus[b] + np.asarray(arr, dtype=float)
        for attr in ("pv_real", "wind_real"):
            arr = getattr(a, attr, None)
            if arr is None:
                continue
            gen_bus.setdefault(b, np.zeros(T))
            gen_bus[b] = gen_bus[b] + np.maximum(
                np.asarray(arr, dtype=float), 0.0)

    # One element per bus, rewritten each period. The published case33bw loads
    # are replaced rather than added to: this is the scenario's day, not the
    # benchmark's.
    load_buses = sorted(set(int(x) for x in net.load.bus))
    for b in load_buses:
        net.load.loc[net.load.bus == b, "p_mw"] = load_bus.get(b, np.zeros(T))[0]
        net.load.loc[net.load.bus == b, "q_mvar"] = \
            load_bus.get(b, np.zeros(T))[0] * tan_phi
    # A bus carrying load and generation both is the normal case, so the
    # generators go on every bus that has one, not only on the load-free ones.
    for b in sorted(gen_bus):
        pp.create_sgen(net, bus=b, p_mw=float(gen_bus[b][0]), q_mvar=0.0,
                       name=f"re_{b}")

    rows = []
    converged = True
    for t in range(T):
        for b, series in load_bus.items():
            net.load.loc[net.load.bus == b, "p_mw"] = series[t]
            net.load.loc[net.load.bus == b, "q_mvar"] = series[t] * tan_phi
        for b, series in gen_bus.items():
            mask = net.sgen.bus == b
            net.sgen.loc[mask, "p_mw"] = series[t]
        try:
            pp.runpp(net, algorithm="bfsw")
        except Exception:
            converged = False
            break
        vm = net.res_bus.vm_pu
        rows.append({
            "period": t,
            "vmin": float(vm.min()),
            "vmin_bus": int(vm.idxmin()),
            "vmax": float(vm.max()),
            "buses_below_lo": int((vm < lo).sum()),
            "buses_above_hi": int((vm > hi).sum()),
            "max_line_loading": float(net.res_line.loading_percent.max()) / 100.0,
        })

    if not converged:
        return {
            "diagnostic_mode": "NO_SHED_VOLTAGE_DIAGNOSTIC",
            "not_a_market_outcome": True,
            "shed": 0.0,
            "shed_forced_zero": True,
            "converged": False,
            "note": "the forced-serve case does not even solve; the violation "
                    "is beyond a power flow's reach",
        }

    vmin_row = min(rows, key=lambda r: r["vmin"])
    overloaded = [r for r in rows if r["max_line_loading"] > 1.0]
    voltage_rows = [r for r in rows if r["buses_below_lo"] or r["buses_above_hi"]]
    return {
        "diagnostic_mode": "NO_SHED_VOLTAGE_DIAGNOSTIC",
        "not_a_market_outcome": True,
        "shed": 0.0,
        "shed_forced_zero": True,
        "converged": True,
        "band": [float(lo), float(hi)],
        "worst_voltage": vmin_row,
        "periods_violating_voltage": len(voltage_rows),
        "bus_periods_below_lo": int(sum(r["buses_below_lo"] for r in rows)),
        "bus_periods_above_hi": int(sum(r["buses_above_hi"] for r in rows)),
        "periods_overloading_a_line": len(overloaded),
        "worst_line_loading": max(r["max_line_loading"] for r in rows),
        # Which network limit breaks first, stated plainly.
        "binding_constraint": (
            "voltage" if voltage_rows else
            ("line_capacity" if overloaded else "none")),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------

def _storage_agents(agents):
    return [a for a in agents if a.storage is not None]


def _churn(schedule):
    """Overlap between charging and discharging (MW, summed over the day)."""
    ch = np.asarray(schedule["p_ch"], dtype=float)
    dis = np.asarray(schedule["p_dis"], dtype=float)
    return float(np.sum(np.minimum(ch, dis)) * money.DT_HOURS)


# Candidate DER layouts for the zero-shedding search (plan §3.3 S-PV1/S-PV2/
# S-WIND/S-SPLIT). Each is a set of moves against the placement in
# config/defaults.yaml, so the installed MW are conserved by construction and
# the candidate is a question about location alone.
#
# The plan fixes what each one probes: whether moving injection toward the
# voltage-limited end of the feeder buys shedding (the deep main feeder is
# buses 14-17), whether putting it next to the largest load does, whether the
# end of the longest spur does, and whether splitting a site beats one large
# injection.
DER_LAYOUTS = {
    "none": [],
    # 2.0 MW of residential PV from bus 21 to bus 17, the deep main feeder.
    "S-PV1": [(21, 17, "pv")],
    # 1.5 MW of industrial PV from bus 23 to bus 24, beside the largest load.
    "S-PV2": [(23, 24, "pv")],
    # 1.5 MW of wind from bus 30 to bus 32, the electrical end of the spur.
    "S-WIND": [(30, 32, "wind")],
    # The residential site split in half across two distant buses, against the
    # same 2.0 MW delivered at one point. The second fraction is of what is
    # left, so the site empties: 1.0 MW to each of buses 17 and 2.
    "S-SPLIT": [(21, 17, "pv", 0.5), (21, 2, "pv", 1.0)],
    # The combined move the individual candidates are each one half of.
    "S-COMB": [(21, 17, "pv"), (23, 24, "pv"), (30, 32, "wind")],
}


def system_report(day):
    agents, result, T = day["agents"], day["result"], day["T"]

    load_energy = sum(float(np.sum(a.load_forecast)) * money.DT_HOURS
                      for a in agents)
    coincident = max(float(sum(a.load_forecast[t] for a in agents))
                     for t in range(T))
    sum_of_maxima = sum(float(np.max(a.load_forecast)) for a in agents)

    # Keyed by agent name: the schedule dict also carries entries that are not
    # participants (the slack, for instance) and those have no agent columns.
    scheds = [result["schedules"][a.name] for a in agents]
    pv_energy = sum(float(np.sum(s["pv_used"])) * money.DT_HOURS for s in scheds)
    wind_energy = sum(float(np.sum(s["wind_used"])) * money.DT_HOURS for s in scheds)
    pv_installed = sum(a.pv_capacity for a in agents)
    wind_installed = sum(a.wind_capacity for a in agents)

    unserved = sum(float(np.sum(s["unserved"])) * money.DT_HOURS for s in scheds)

    # Net exchange is the physical draw on the upper grid. Import and export are
    # the positive and negative parts of one variable now, so gross import is
    # the draw in the periods the network is importing and nothing more. It
    # exceeds the net by the network's own losses, which the slack supplies and
    # which no participant's schedule carries.
    net_mwh = sum(float(np.sum(s["p_buy"] - s["p_sell"])) * money.DT_HOURS
                  for s in scheds)
    gross_import = gross_export = None
    if "grid_import_mw" in result:
        gross_import = float(np.sum(result["grid_import_mw"]) * money.DT_HOURS)
        gross_export = float(np.sum(result["grid_export_mw"]) * money.DT_HOURS)

    return {
        "peak_load_coincident_mw": coincident,
        "peak_load_sum_of_maxima_mw": sum_of_maxima,
        "load_energy_mwh": load_energy,
        "pv_energy_mwh": pv_energy,
        "wind_energy_mwh": wind_energy,
        "pv_installed_mw": pv_installed,
        "wind_installed_mw": wind_installed,
        "grid_net_mwh": net_mwh,
        "grid_gross_import_mwh": gross_import,
        "grid_gross_export_mwh": gross_export,
        "curtailment_mwh": result["total_curtailment"],
        "unserved_mwh": unserved,
        "re_consumption_rate": result["re_consumption_rate"],
        "lmp_fallbacks": result.get("lmp_fallbacks"),
        # The canonical name. This read `result["objective"]` until the clearing
        # stopped emitting that spelling; `legacy_schema` translates the retired
        # name for artifacts already on disk, but a freshly cleared result
        # carries only the canonical one, so reading the old name here failed
        # outright rather than being translated.
        "objective": result["clearing_objective"],
        "carbon_emissions_tco2": result["carbon_emissions"],
    }


def ess_report(day):
    config, agents, result = day["config"], day["agents"], day["result"]
    lmp = result["lmp"]
    rows = []
    for a in _storage_agents(agents):
        sched = result["schedules"][a.name]
        lmp_node = lmp[:, a.bus]
        ledger = participant_payoff.participant_payoff(
            sched, lmp_node, a, config).as_dict()
        ch = np.asarray(sched["p_ch"], dtype=float)
        dis = np.asarray(sched["p_dis"], dtype=float)
        rows.append({
            "name": a.name,
            "bus": a.bus,
            "e_max_mwh": a.storage.e_max,
            "p_ch_max_mw": a.storage.p_ch_max,
            "p_dis_max_mw": a.storage.p_dis_max,
            "eta_ch": a.storage.eta_ch,
            "eta_dis": a.storage.eta_dis,
            "soc_initial": a.storage.soc0,
            "soc_final": float(sched["soc"][-1]),
            "soc_min": a.storage.soc_min,
            "soc_max": a.storage.soc_max,
            "charge_mwh": float(np.sum(ch) * money.DT_HOURS),
            "discharge_mwh": float(np.sum(dis) * money.DT_HOURS),
            "churn_mwh": _churn(sched),
            "cycles": float(ledger["equivalent_cycles"]),
            "revenue": float(ledger["revenue"]),
            "purchase_cost": float(ledger["purchase_cost"]),
            "degradation_cost": float(ledger["degradation_cost"]),
            "net_profit": float(ledger["total"]),
        })
    return rows


def network_report(day):
    result = day["result"]
    if "line_utilization" not in result:
        return {"available": False, "reason": "engine reports no line current"}
    net = build_base_network(day["config"])
    util = result["line_utilization"]
    peak = util.max(axis=0)
    labels = []
    for line_id in result["line_indices"]:
        f = int(net.line.at[line_id, "from_bus"])
        t = int(net.line.at[line_id, "to_bus"])
        labels.append({"line_id": int(line_id), "from_bus": f, "to_bus": t,
                       "label": "%d->%d" % (f, t)})
    order = np.argsort(-peak)
    top = [dict(labels[i], max_utilization=float(peak[i])) for i in order[:5]]
    volts = result["bus_voltage"]
    return {
        "available": True,
        "max_line_utilization": float(peak.max()),
        "mean_line_max_utilization": float(peak.mean()),
        "lines_over_085": int((peak > 0.85).sum()),
        "lines_over_095": int((peak > 0.95).sum()),
        "lines_over_limit": int((peak > 1.0 + 1e-6).sum()),
        "top_lines": top,
        "min_bus_voltage_pu": float(volts.min()),
        "max_bus_voltage_pu": float(volts.max()),
        "binding_thermal": int((util > 1.0 - 1e-6).sum()),
    }


def critical_lines(day, n=3):
    """The lines to target first when a congestion case is built."""
    report = network_report(day)
    if not report.get("available"):
        return []
    return report["top_lines"][:n]


def _window(node, eta_ch, eta_dis, cycle_cost):
    """Temporal price ratio at one node against the round-trip breakeven.

    To deliver one MWh the unit charges 1/(eta_ch*eta_dis) MWh, so the energy it
    buys costs that multiple of the valley price, and degradation is charged on
    both legs. The window is real only when the ratio clears that breakeven.
    """
    valley = float(node.min())
    peak = float(node.max())
    if valley <= 0:
        return {"lmp_valley": valley, "lmp_peak": peak,
                "ratio": float("inf"), "breakeven_ratio": float("inf"),
                "margin": float("-inf")}
    breakeven = ((1.0 / (eta_ch * eta_dis)) * valley + 2.0 * cycle_cost) / valley
    return {"lmp_valley": valley, "lmp_peak": peak, "ratio": peak / valley,
            "breakeven_ratio": breakeven, "margin": peak / valley - breakeven}


def price_report(day):
    config, agents, result = day["config"], day["agents"], day["result"]
    lmp = result["lmp"]
    price = result["price"]
    cycle_cost = float(config.storage.cycle_cost)
    rows = []
    for a in _storage_agents(agents):
        node = lmp[:, a.bus]
        row = {"name": a.name, "bus": a.bus,
               "lmp_mean": float(node.mean())}
        row.update(_window(node, a.storage.eta_ch, a.storage.eta_dis, cycle_cost))
        # The margin in CNY/MWh and the share of the unit's own action box it
        # leaves tradeable. The ratio above says whether a cycle pays at all;
        # these say how much of what the unit can declare still does something.
        margin = round_trip_margin_cny_mwh(node, a.storage.eta_ch,
                                           a.storage.eta_dis, cycle_cost)
        row["round_trip_margin_cny_mwh"] = margin
        row["tradeable_box_share"] = tradeable_box_share(
            margin, action_box())
        row["siting_eligible"] = (
            row["tradeable_box_share"] >= TRADEABLE_BOX_SHARE)
        rows.append(row)

    # Every bus, not just the ones carrying a battery: this is what makes a
    # siting claim checkable, and it is how a bus whose spread is too narrow to
    # pay for a cycle gets noticed before a battery is placed on it.
    by_bus = []
    for i, bus in enumerate(result.get("bus_indices", [])):
        node = lmp[:, i]
        row = {"bus": int(bus), "lmp_mean": float(node.mean())}
        row.update(_window(node, 0.92, 0.92, cycle_cost))
        # Same two quantities as the per-unit rows, on the standard 0.92/0.92
        # unit, so the pool a layout is searched over is read from one place.
        margin = round_trip_margin_cny_mwh(node, 0.92, 0.92, cycle_cost)
        row["round_trip_margin_cny_mwh"] = margin
        row["tradeable_box_share"] = tradeable_box_share(margin, action_box())
        by_bus.append(row)
    by_bus.sort(key=lambda r: -r["margin"])
    viable = [r["bus"] for r in by_bus if r["margin"] > 0]
    # The wholesale curve with its noise switched off, so the reader can tell
    # how much of the spread is the load/renewable shape and how much is the
    # forecast noise riding on top of it.
    quiet = None
    try:
        saved = get_default("price_curve.noise_sigma")
        set_default("price_curve.noise_sigma", 0.0)
        quiet = day_ahead_price_china(day["T"], agents=agents, config=config)
        set_default("price_curve.noise_sigma", saved)
    except Exception:
        quiet = None
    return {
        "lmp_min": float(lmp.min()),
        "lmp_mean": float(lmp.mean()),
        "lmp_max": float(lmp.max()),
        "price_min": float(price.min()),
        "price_mean": float(price.mean()),
        "price_max": float(price.max()),
        "price_peak_period": int(np.argmax(price)),
        "price_valley_period": int(np.argmin(price)),
        "system_ratio": float(price.max() / price.min()) if price.min() > 0 else None,
        "wholesale_ratio": (float(np.max(day["wholesale"]) / np.min(day["wholesale"]))
                            if np.min(day["wholesale"]) > 0 else None),
        "wholesale_ratio_no_noise": (
            float(np.max(quiet) / np.min(quiet)) if quiet is not None and np.min(quiet) > 0
            else None),
        "per_ess": rows,
        "by_bus": by_bus,
        "viable_buses": viable,
    }


def distance_report(day):
    """Re-verify the siting claim from the topology on every run.

    Note the metric: `pairwise_resistance_distance` runs a shortest path over
    every line in the network, including the five tie switches that the radial
    OPF leaves open. It is a topological spread, not the electrical distance the
    clearing actually prices.
    """
    buses = [a.bus for a in _storage_agents(day["agents"])]
    net = build_base_network(day["config"])
    all_buses = list(range(len(net.bus.index)))
    D = pairwise_resistance_distance(net, all_buses)
    slack_d = {b: float(D[0][b]) for b in buses}
    pairwise = {}
    for i, bi in enumerate(buses):
        for bj in buses[i + 1:]:
            pairwise["%d-%d" % (min(bi, bj), max(bi, bj))] = float(D[bi][bj])
    return {
        "buses": buses,
        "slack_distance_ohm": slack_d,
        "pairwise_ohm": pairwise,
        "min_pairwise_ohm": min(pairwise.values()) if pairwise else None,
    }


def reconciliation_check(day):
    try:
        rec = surplus_metrics.reconciliation(
            day["result"]["schedules"], day["result"]["lmp"],
            day["wholesale"], day["agents"])
    except Exception as exc:                      # cross-check only
        return {"available": False, "reason": str(exc)[:120]}
    out = {"available": True}
    for key in ("rent_total", "bill_total", "cp_total", "bc_total",
                "import_total", "max_identity_residual"):
        if key in rec:
            out[key] = rec[key]
    out["rent_min"] = float(np.min(rec["rent"])) if "rent" in rec else None
    return out


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def evaluate_gates(day, system, ess, network, price, distances):
    """Hard gates plus the deviations this baseline knowingly carries.

    Two properties of this day are outcomes rather than defects, and are
    reported as accepted rather than folded into a tolerance that would hide a
    real regression:

      * Some load is shed. The feeder is voltage-limited from about 8.3 MW of
        peak load upward, and the stated +/-7% band is kept, so serving 11 MW
        means giving up a little load at the evening peak.

      * The arbitrage window lands on the round-trip breakeven rather than
        clearly above it. With degradation at 100 CNY/MWh per leg and a 0.92/0.92
        round trip, a battery needs a peak-to-valley ratio above roughly 1.5, and
        the day's spread is about that; four batteries also arbitrage away part
        of the spread they feed on. The batteries still cycle with the price.
    """
    gates = []
    accepted = []
    diagnostics = []

    def add(name, ok, value, target):
        gates.append({"gate": name, "ok": bool(ok), "value": value, "target": target})

    def note(name, value, reason):
        accepted.append({"item": name, "value": value, "reason": reason})

    def observe(name, value, target):
        """A measurement the baseline reports but does not fail on.

        A gate is a property the model must have. These are properties of this
        particular day: how much the fleet cycles, how large it is relative to
        the load, which units earned. Pinning them as pass/fail turns one
        realization into a contract and makes the next honest change look like a
        regression, so they are reported and left alone.
        """
        diagnostics.append({"item": name, "value": value, "target": target})

    fallbacks = system["lmp_fallbacks"]
    add("nodal prices are real",
        fallbacks == 0, fallbacks, "lmp_fallbacks == 0")

    # Nothing downstream should have to guard against a non-finite number; if
    # one appears it is a solver or extraction failure, not a market outcome.
    lmp_arr = np.asarray(day["result"]["lmp"], dtype=float)
    finite = bool(np.all(np.isfinite(lmp_arr)))
    for r in ess:
        finite = finite and all(
            np.isfinite(np.asarray(day["result"]["schedules"][r["name"]][k],
                                   dtype=float)).all()
            for k in ("p_ch", "p_dis", "soc", "served", "unserved"))
    add("every reported quantity is finite", finite, finite, "no NaN or inf")

    # The state of charge is bounded by construction, so a breach means the
    # bound was set somewhere other than where the schedule was read -- worth
    # failing on rather than reporting.
    soc_breach = []
    for r in ess:
        soc = np.asarray(day["result"]["schedules"][r["name"]]["soc"], dtype=float)
        if soc.min() < r["soc_min"] - 1e-6 or soc.max() > r["soc_max"] + 1e-6:
            soc_breach.append(
                "%s: [%.4f, %.4f] vs [%.2f, %.2f]"
                % (r["name"], soc.min(), soc.max(), r["soc_min"], r["soc_max"]))
    add("state of charge respects its bounds",
        not soc_breach, soc_breach or "within bounds",
        "soc_min <= soc <= soc_max, per unit")

    # Import and export are the positive and negative parts of one variable, so
    # this is an identity rather than a property of the solution. It is still
    # asserted, because the readout that recovers the two directions from that
    # variable is the thing that could get it wrong.
    imp = np.asarray(day["result"].get("grid_import_mw", []), dtype=float)
    exp = np.asarray(day["result"].get("grid_export_mw", []), dtype=float)
    if imp.size and exp.size:
        overlap = float(np.max(imp * exp))
        add("the network never imports and exports at once",
            overlap <= 1e-9, overlap,
            "max(import * export) == 0, by construction")

    # Tolerance, not an exact zero: the clear is a barrier-solved QCP, so the
    # overlap lands a few parts in ten million of a MWh rather than on the nose.
    # A megawatt-hour is eight orders of magnitude above that, so this still
    # means "no physically meaningful overlap".
    fleet_churn = sum(r["churn_mwh"] for r in ess)
    add("charge/discharge exclusive",
        fleet_churn <= CHURN_TOL_MWH, fleet_churn,
        "churn <= %.0e MWh" % CHURN_TOL_MWH)

    # What the day exists to demonstrate: a battery the price can move. That the
    # *fleet* has a viable opportunity is the claim; which of the four units
    # takes it, and how far, is an outcome of the day's prices at its bus. A
    # deep-feeder unit that sits out a day is not a defect.
    idle = [r["name"] for r in ess if r["discharge_mwh"] <= 0.0]
    cycling = [r["name"] for r in ess if r["charge_mwh"] > 0 and r["discharge_mwh"] > 0]
    add("the fleet has a viable arbitrage opportunity",
        bool(cycling), cycling or "none cycled",
        "at least one unit both charges and discharges")

    worst_soc_gap = max(abs(r["soc_final"] - r["soc_initial"]) for r in ess)
    add("day closes at its starting state of charge",
        worst_soc_gap <= SOC_CLOSURE_TOL, worst_soc_gap,
        "|soc[T] - soc[0]| <= 1e-6")

    if network.get("available"):
        add("network within limits",
            network["max_line_utilization"] <= 1.0 + 1e-6
            and network["min_bus_voltage_pu"] >= 0.93 - 1e-6
            and network["max_bus_voltage_pu"] <= 1.05 + 1e-6,
            {"max_util": network["max_line_utilization"],
             "v_min": network["min_bus_voltage_pu"],
             "v_max": network["max_bus_voltage_pu"]},
            "util <= 1, 0.93 <= v <= 1.05")

    # The shedding gate is armed only when the caller names the limit, because
    # what the limit should be is a trade-off between the peak load the day is
    # asked to serve, the inverter headroom it is given and the voltage band it
    # is held to -- not a property of the model. Unarmed, the run still
    # measures and reports the shed load, and says loudly that it did not fail
    # on it, so a green report can never be read as a day that shed nothing.
    unserved_frac = (system["unserved_mwh"] / system["load_energy_mwh"]
                     if system["load_energy_mwh"] else 0.0)
    if UNSERVED_LIMIT_MWH is None:
        note("load shedding is NOT gated",
             {"mwh": system["unserved_mwh"], "fraction": unserved_frac},
             "no --unserved-limit was given, so nothing failed on this; see the "
             "inverter-headroom sweep before treating the day as shed-free")
    else:
        add("shed load within the declared limit",
            system["unserved_mwh"] <= UNSERVED_LIMIT_MWH,
            {"mwh": system["unserved_mwh"], "fraction": unserved_frac},
            "unserved <= %.4g MWh" % UNSERVED_LIMIT_MWH)
    if unserved_frac > 0:
        note("some load is shed at the evening peak",
             {"mwh": system["unserved_mwh"], "fraction": unserved_frac},
             "feeder is voltage-limited above ~8.3 MW peak at the +/-7% band")

    observe("renewable consumption rate", system["re_consumption_rate"],
            "percent of available RE used")

    # The coincident peak is what the network is asked to carry, and it is what
    # the load level is configured as; the scale that produces it is derived
    # from these profiles, so this holds to solver noise rather than to a band.
    peak = system["peak_load_sum_of_maxima_mw"]
    coincident = system["peak_load_coincident_mw"]
    target_peak = _target_peak_mw()
    add("coincident peak load on target",
        abs(coincident - target_peak) <= 1e-6, coincident,
        "== %.4g MW, the peak this scenario declares" % target_peak)
    observe("peak load (sum of per-agent maxima)", peak,
            "MW; larger than the coincident peak because the buses do not all "
            "peak at the same period, and not drawn at any single instant")

    ess_power = sum(r["p_ch_max_mw"] for r in ess)
    ess_energy = sum(r["e_max_mwh"] for r in ess)
    ratio_power = ess_power / peak if peak else 0.0
    ratio_energy = ess_energy / system["load_energy_mwh"] if system["load_energy_mwh"] else 0.0
    observe("fleet power over peak load", ratio_power,
            "the band this used to be gated on was %.2f-%.2f"
            % ESS_POWER_TO_PEAK_BAND)
    observe("fleet energy over daily load", ratio_energy,
            "the band this used to be gated on was %.2f-%.2f"
            % ESS_ENERGY_TO_LOAD_BAND)

    target_pv, target_wind = _declared_renewables()
    add("installed renewables on target",
        abs(system["pv_installed_mw"] - target_pv) <= INSTALLED_MW_TOL
        and abs(system["wind_installed_mw"] - target_wind) <= INSTALLED_MW_TOL,
        {"pv_mw": system["pv_installed_mw"], "wind_mw": system["wind_installed_mw"]},
        "PV %.2f MW, wind %.2f MW, as the scenario declares" % (target_pv,
                                                                target_wind))

    observe("cycles per unit", [round(c, 4) for c in (r["cycles"] for r in ess)],
            "how hard the day's prices worked each unit; was gated on %.1f-%.1f"
            % CYCLES_BAND)

    if distances["min_pairwise_ohm"] is not None:
        add("battery buses are distinct",
            len(set(distances["buses"])) == len(distances["buses"]),
            {"buses": distances["buses"],
             "min_pairwise_ohm": distances["min_pairwise_ohm"]},
            "distinct buses")
        observe("minimum pairwise electrical separation",
                distances["min_pairwise_ohm"],
                "ohm; was gated on > 0.5")

    worst_margin = min((r["margin"] for r in price["per_ess"]), default=None)
    note("arbitrage window sits on the round-trip breakeven",
         {"worst_margin": worst_margin,
          "per_bus": {r["name"]: round(r["margin"], 3) for r in price["per_ess"]},
          "fleet_net_cny": sum(r["net_profit"] for r in ess)},
         "cycle_cost 100/leg with a 0.92/0.92 round trip needs a peak/valley "
         "ratio near 1.5, which is about what the day offers; the batteries "
         "cycle on price but earn near breakeven")

    observe("LMP peak-to-valley ratio", price["system_ratio"],
            "was never gated; a ratio is not a property the model must have")
    observe("fleet net profit", sum(r["net_profit"] for r in ess),
            "CNY/day; not gated, and not a target to tune towards")
    observe("units that neither charged nor discharged", idle or "none",
            "not gated; a unit may sit out a day its bus does not pay for")

    return gates, accepted, diagnostics


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fmt(value):
    if isinstance(value, float):
        return "%.4g" % value
    if isinstance(value, dict):
        return ", ".join("%s=%s" % (k, _fmt(v)) for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def render(day, system, ess, network, price, distances, gates, recon,
           accepted=None, diagnostics=None):
    out = []
    line = out.append
    line("=" * 78)
    line("Typical day physical baseline")
    line("=" * 78)
    # The price source is named rather than assumed. This header used to print
    # the merit-order map's base and elasticity unconditionally, which stopped
    # describing the day the moment the native scenario was priced by an
    # external market -- the numbers stayed on the page and referred to
    # parameters no longer in the path.
    line("scenario %s | T=%d | engine %s | %s | price source %s"
         % (day["scenario"], day["T"], day["config"].opf_mode,
            day.get("load_level", "load level n/a"),
            day.get("price_source", "unknown")))

    line("")
    line("-- system " + "-" * 68)
    line("  peak load (coincident)               %.4g MW   target %.4g"
         % (system["peak_load_coincident_mw"], _target_peak_mw(day["scenario"])))
    line("  peak load (sum of per-agent maxima)  %.4g MW   (diagnostic)"
         % system["peak_load_sum_of_maxima_mw"])
    line("  load energy                          %.4g MWh" % system["load_energy_mwh"])
    line("  PV  installed %.4g MW / used %.4g MWh" % (system["pv_installed_mw"], system["pv_energy_mwh"]))
    line("  Wind installed %.4g MW / used %.4g MWh" % (system["wind_installed_mw"], system["wind_energy_mwh"]))
    line("  grid exchange (net)                  %.4g MWh" % system["grid_net_mwh"])
    if system["grid_gross_import_mwh"] is not None:
        line("  grid import periods / export periods %.4g / %.4g  MWh"
             % (system["grid_gross_import_mwh"], system["grid_gross_export_mwh"]))
        line("      one net-exchange variable, so import times export is identically zero;"
             "\n      import exceeds the net above by the network's own losses)")
    line("  curtailment                          %.4g MWh" % system["curtailment_mwh"])
    line("  unserved energy                      %.4g MWh" % system["unserved_mwh"])
    line("  RE consumption rate                  %.4g %%" % system["re_consumption_rate"])
    line("  LMP fallbacks                        %s (0 = every price is nodal)" % system["lmp_fallbacks"])
    line("  clearing objective                   %.6g CNY" % system["objective"])
    line("  carbon on net import                 %.4g tCO2"
         % system["carbon_emissions_tco2"])

    line("")
    line("-- batteries " + "-" * 65)
    line("  %-8s %4s %7s %7s %8s %8s %8s %7s %10s %11s %10s"
         % ("unit", "bus", "Pmax", "Emax", "soc0", "socT", "ch", "dis",
            "churn", "cycles", "net profit"))
    for r in ess:
        line("  %-8s %4d %7.2f %7.2f %8.3f %8.3f %8.2f %7.2f %10.4f %11.3f %10.1f"
             % (r["name"], r["bus"], r["p_ch_max_mw"], r["e_max_mwh"],
                r["soc_initial"], r["soc_final"], r["charge_mwh"],
                r["discharge_mwh"], r["churn_mwh"], r["cycles"], r["net_profit"]))
    line("  revenue / purchase / degradation per unit, CNY:")
    for r in ess:
        line("    %-8s buy %9.2f  sell %9.2f  degrade %8.2f  net %9.2f"
             % (r["name"], r["purchase_cost"], r["revenue"],
                r["degradation_cost"], r["net_profit"]))
    fleet_net = sum(r["net_profit"] for r in ess)
    line("  fleet net %+.2f CNY/day" % fleet_net)

    line("")
    line("-- price " + "-" * 68)
    line("  LMP  min %.4g  mean %.4g  max %.4g  ratio %.3g"
         % (price["lmp_min"], price["lmp_mean"], price["lmp_max"], price["system_ratio"]))
    line("  wholesale ratio %.3g | without forecast noise %.3g"
         % (price["wholesale_ratio"], price["wholesale_ratio_no_noise"]))
    line("  peak period t=%d  valley period t=%d  (t=%d is %.2f h)"
         % (price["price_peak_period"], price["price_valley_period"],
            price["price_peak_period"], price["price_peak_period"] * money.DT_HOURS))
    line("  arbitrage window per battery bus against the round-trip breakeven:")
    line("    %-8s %4s %9s %9s %7s %10s %9s"
         % ("unit", "bus", "valley", "peak", "ratio", "breakeven", "margin"))
    for r in price["per_ess"]:
        line("    %-8s %4d %9.1f %9.1f %7.2f %10.2f %+9.2f"
             % (r["name"], r["bus"], r["lmp_valley"], r["lmp_peak"],
                r["ratio"], r["breakeven_ratio"], r["margin"]))
    if price.get("by_bus"):
        line("  buses with a positive window: %d of %d"
             % (len(price["viable_buses"]), len(price["by_bus"])))
        line("  widest windows (all buses, siting evidence):")
        for r in price["by_bus"][:6]:
            line("    bus %2d  valley %7.1f  peak %7.1f  ratio %5.2f  breakeven %5.2f  margin %+6.2f"
                 % (r["bus"], r["lmp_valley"], r["lmp_peak"], r["ratio"],
                    r["breakeven_ratio"], r["margin"]))
        line("  narrowest windows:")
        for r in price["by_bus"][-3:]:
            line("    bus %2d  valley %7.1f  peak %7.1f  ratio %5.2f  breakeven %5.2f  margin %+6.2f"
                 % (r["bus"], r["lmp_valley"], r["lmp_peak"], r["ratio"],
                    r["breakeven_ratio"], r["margin"]))

    line("")
    line("-- network " + "-" * 66)
    if network.get("available"):
        line("  max line utilization %.4g | mean of per-line maxima %.4g"
             % (network["max_line_utilization"], network["mean_line_max_utilization"]))
        line("  lines over 0.85: %d | over 0.95: %d | over limit: %d"
             % (network["lines_over_085"], network["lines_over_095"],
                network["lines_over_limit"]))
        line("  bus voltage  min %.4g pu  max %.4g pu"
             % (network["min_bus_voltage_pu"], network["max_bus_voltage_pu"]))
        line("  most utilized lines:")
        for t in network["top_lines"]:
            line("    line %-7s (id %2d)  rho_max %.3f"
                 % (t["label"], t["line_id"], t["max_utilization"]))
    else:
        line("  n/a (%s)" % network.get("reason", "unavailable"))

    line("")
    line("-- siting " + "-" * 67)
    line("  buses %s" % distances["buses"])
    line("  distance from slack (ohm): %s"
         % ", ".join("bus %d %.3f" % (b, d) for b, d in distances["slack_distance_ohm"].items()))
    line("  min pairwise separation %.4g ohm" % distances["min_pairwise_ohm"])
    line("  metric note: shortest path over all lines including the tie switches")
    line("  critical lines for a congestion case: %s"
         % ", ".join(t["label"] for t in critical_lines(day)))

    line("")
    line("-- cross-checks " + "-" * 62)
    if recon.get("available"):
        line("  funds-flow: %s" % _fmt({k: v for k, v in recon.items() if k != "available"}))
    else:
        line("  funds-flow n/a (%s)" % recon.get("reason"))

    line("")
    line("-- gates " + "-" * 68)
    for g in gates:
        line("  %-4s %-38s %-34s %s"
             % ("PASS" if g["ok"] else "FAIL", g["gate"], _fmt(g["value"]), g["target"]))
    failed = [g["gate"] for g in gates if not g["ok"]]
    line("")
    line("  %d/%d gates passed%s"
         % (len(gates) - len(failed), len(gates),
            "" if not failed else "; failing: " + "; ".join(failed)))

    if diagnostics:
        line("")
        line("-- diagnostics (reported, not gated) " + "-" * 40)
        for d in diagnostics:
            line("  %-42s %s" % (d["item"], _fmt(d["value"])))
            line("  %-42s   %s" % ("", d["target"]))

    if accepted:
        line("")
        line("-- accepted deviations " + "-" * 55)
        line("  These are properties of this day, not defects. They are reported")
        line("  rather than tuned away, and they do not fail the run.")
        for a in accepted:
            line("  * %s" % a["item"])
            line("      value  %s" % _fmt(a["value"]))
            line("      why    %s" % a["reason"])
    return "\n".join(out)


def build_report(day):
    system = system_report(day)
    ess = ess_report(day)
    network = network_report(day)
    price = price_report(day)
    distances = distance_report(day)
    recon = reconciliation_check(day)
    gates, accepted, diagnostics = evaluate_gates(
        day, system, ess, network, price, distances)
    return {
        "config": {"scenario": day["scenario"], "T": day["T"],
                   "opf_mode": day["config"].opf_mode,
                   "load_scale": day["load_scale"],
                   "price_base": day["price_base"],
                   "elasticity": day["elasticity"]},
        "system": system,
        "batteries": ess,
        "network": network,
        "price": price,
        "siting": distances,
        "critical_lines": critical_lines(day),
        "reconciliation": recon,
        "gates": gates,
        "diagnostics": diagnostics,
        "accepted_deviations": accepted,
        "passed": all(g["ok"] for g in gates),
    }


def _sweep_axis(label, values, run_kwargs, axis):
    """Clear the day once per value of one knob and show the gates that matter.

    Returns the rows, so a caller that is writing an artifact gets the table
    rather than only the printout.
    """
    print("\n" + "=" * 78)
    print("sweep: %s over %s" % (label, values))
    print("=" * 78)
    print("%-9s %8s %7s %8s %9s %7s %7s %8s"
          % (label, "peak MW", "re %", "curt MWh", "unserv MWh",
             "v_min", "v_max", "maxUtil"))
    rows = []
    for v in values:
        kwargs = dict(run_kwargs)
        kwargs[axis] = v
        try:
            day = run_day(**kwargs)
        except Exception as exc:
            print("%-9.4g FAILED: %s" % (v, str(exc)[:44]))
            continue
        system = system_report(day)
        network = network_report(day)
        rows.append({
            axis: v,
            "peak_load_coincident_mw": system["peak_load_coincident_mw"],
            "peak_load_sum_of_maxima_mw": system["peak_load_sum_of_maxima_mw"],
            "re_consumption_rate": system["re_consumption_rate"],
            "curtailment_mwh": system["curtailment_mwh"],
            "unserved_mwh": system["unserved_mwh"],
            "min_bus_voltage_pu": network.get("min_bus_voltage_pu"),
            "max_bus_voltage_pu": network.get("max_bus_voltage_pu"),
            "max_line_utilization": network.get("max_line_utilization"),
        })
        print("%-9.4g %8.3f %7.2f %8.2f %9.4f %7.4f %7.4f %8.4f"
              % (v, system["peak_load_sum_of_maxima_mw"],
                 system["re_consumption_rate"], system["curtailment_mwh"],
                 system["unserved_mwh"],
                 network.get("min_bus_voltage_pu", float("nan")),
                 network.get("max_bus_voltage_pu", float("nan")),
                 network.get("max_line_utilization", float("nan"))))
    return rows


def der_layout_sweep(run_kwargs, headroom, names=None):
    """Clear each candidate DER layout and show what it does to the shedding.

    Every candidate is checked at one inverter headroom so the layouts are
    compared against each other rather than against different multipliers, and
    the invariants the plan requires are asserted on every one: PV 3.5 MW,
    wind 1.5 MW, exactly four batteries, at least two PV buses, and no bus
    carrying both.
    """
    names = list(names or DER_LAYOUTS)
    print("\n" + "=" * 78)
    print("DER layouts at headroom %.2f" % headroom)
    print("=" * 78)
    print("%-9s %7s %7s %9s %10s %9s %8s"
          % ("layout", "pv MW", "wind MW", "unserv MWh", "curt MWh", "v_min", "peak MW"))
    rows = []
    for name in names:
        moves = DER_LAYOUTS[name]
        try:
            day = run_day(**dict(run_kwargs, inverter_headroom=headroom,
                                 der_moves=moves))
        except Exception as exc:
            print("%-9s FAILED: %s" % (name, str(exc)[:56]))
            continue
        agents, result = day["agents"], day["result"]
        pv = sum(a.pv_capacity for a in agents)
        wind = sum(a.wind_capacity for a in agents)
        fleet = [a for a in agents if getattr(a, "participant_type", "") == "storage"]
        pv_buses = {a.bus for a in agents if a.pv_capacity > 0}
        wind_buses = {a.bus for a in agents if a.wind_capacity > 0}
        assert abs(pv - 3.5) < 1e-6 and abs(wind - 1.5) < 1e-6, \
            f"{name}: PV {pv} / wind {wind} are not the required 3.5 / 1.5 MW"
        assert len(fleet) == 4, f"{name}: {len(fleet)} batteries, expected 4"
        assert len(pv_buses) >= 2, f"{name}: PV on {sorted(pv_buses)}, need >= 2"
        assert not (pv_buses & wind_buses), \
            f"{name}: {sorted(pv_buses & wind_buses)} carries both PV and wind"
        system = system_report(day)
        network = network_report(day)
        rows.append({
            "layout": name, "moves": [list(m) for m in moves],
            "pv_mw": pv, "wind_mw": wind,
            "pv_buses": sorted(pv_buses), "wind_buses": sorted(wind_buses),
            "unserved_mwh": system["unserved_mwh"],
            "curtailment_mwh": system["curtailment_mwh"],
            "re_consumption_rate": system["re_consumption_rate"],
            "min_bus_voltage_pu": network.get("min_bus_voltage_pu"),
            "max_line_utilization": network.get("max_line_utilization"),
            "peak_load_sum_of_maxima_mw":
                system["peak_load_sum_of_maxima_mw"],
        })
        print("%-9s %7.2f %7.2f %9.4f %10.4f %9.4f %8.4f"
              % (name, pv, wind, system["unserved_mwh"],
                 system["curtailment_mwh"],
                 network.get("min_bus_voltage_pu", float("nan")),
                 system["peak_load_sum_of_maxima_mw"]))
    return rows


def _margin_for_share(share, box):
    """The margin a given tradeable share corresponds to, for reporting."""
    from scipy.optimize import brentq
    span = (box[0][1] - box[0][0]) + (box[1][1] - box[1][0])
    return float(brentq(lambda m: tradeable_box_share(m, box) - share,
                        -1.0, span))


def _slack_bus(day):
    """The bus the external grid connects at, which cannot host a battery."""
    net = build_base_network(day["config"])
    return int(net.ext_grid.at[0, "bus"])


def _non_slack_buses(day):
    """Every bus a battery could be placed on, in network order."""
    slack = _slack_bus(day)
    return [int(b) for b in day["result"].get("bus_indices", [])
            if int(b) != slack]


def siting_sweep(run_kwargs, buses=None):
    """Measure one battery alone at each candidate bus (the plan's M2).

    **This is a diagnostic, not the selection metric, and it was measured to be
    misleading as one.** A single-unit clear answers "what would this bus be
    worth to a unit arriving into today's market", but four units arbitrage the
    same daily spread, so the price field a candidate is evaluated on is one
    the other three have already flattened. Bus 6 leaves a lone unit 186
    CNY/MWh and 17 once the fleet bids; the layout this scan selects
    (10/15/19/26) leaves *none* of the four with a usable action space, while
    the previous layout leaves one.

    So read it as a per-bus price landscape -- which buses have a wide window,
    which are narrow, where the crowding bites -- and take the eligibility a
    layout is judged on from `price_report`, whose margin is read off the joint
    day. `layout_search` is what selects.

    Eligibility in both places is the share of the action box the unit can
    still trade in, not the sign of its margin: "profitable" and "has a
    strategy space" are different questions, and a unit can be the first
    without being the second.
    """
    day = run_day(**run_kwargs)
    candidates = buses if buses is not None else _non_slack_buses(day)
    box = action_box()
    cost = float(day["config"].storage.cycle_cost)

    print("\n" + "=" * 78)
    print("siting: one unit alone at each of %d buses" % len(candidates))
    print("=" * 78)
    print("%4s %9s %9s %10s %9s %8s %9s"
          % ("bus", "valley", "peak", "margin", "share", "cycles", "net CNY"))
    rows = []
    for bus in candidates:
        try:
            solo = run_day(**dict(run_kwargs, ess_buses=[bus]))
        except Exception as exc:
            print("%4d FAILED: %s" % (bus, str(exc)[:52]))
            continue
        unit = _storage_agents(solo["agents"])[0]
        sched = solo["result"]["schedules"][unit.name]
        node = solo["result"]["lmp"][:, unit.bus]
        ledger = participant_payoff.participant_payoff(
            sched, node, unit, solo["config"]).as_dict()
        margin = round_trip_margin_cny_mwh(node, unit.storage.eta_ch,
                                           unit.storage.eta_dis, cost)
        share = tradeable_box_share(margin, box)
        rows.append({
            "bus": int(bus),
            "lmp_valley": float(np.min(node)),
            "lmp_peak": float(np.max(node)),
            "round_trip_margin_cny_mwh": margin,
            "tradeable_box_share": share,
            "eligible": bool(share >= TRADEABLE_BOX_SHARE),
            "charge_mwh": float(np.sum(sched["p_ch"]) * money.DT_HOURS),
            "discharge_mwh": float(np.sum(sched["p_dis"]) * money.DT_HOURS),
            "cycles": float(ledger["equivalent_cycles"]),
            "net_profit": float(ledger["total"]),
        })
        print("%4d %9.1f %9.1f %10.1f %9.6f %8.3f %9.2f"
              % (bus, rows[-1]["lmp_valley"], rows[-1]["lmp_peak"], margin,
                 share, rows[-1]["cycles"], rows[-1]["net_profit"]))

    eligible = [r for r in rows if r["eligible"]]
    print("-" * 78)
    print("eligible (share >= %.2f, i.e. margin >= %.1f CNY/MWh): %s"
          % (TRADEABLE_BOX_SHARE, _margin_for_share(TRADEABLE_BOX_SHARE, box),
             [r["bus"] for r in eligible] or "none"))
    print("of those, %d cycled: %s"
          % (sum(1 for r in eligible if r["cycles"] > 0),
             [r["bus"] for r in eligible if r["cycles"] > 0] or "none"))
    if not eligible:
        print("note: no bus clears the eligibility bar. That is a finding "
              "about the day, not a reason to lower the bar after the fact.")
    return rows


def joint_tradeable_shares(run_kwargs, buses):
    """Per-unit tradeable box shares in a layout cleared jointly.

    The criterion a layout is judged on. Read off the day the four units
    actually produced together, which is the only day any of them will trade
    in.
    """
    day = run_day(**dict(run_kwargs, ess_buses=list(buses)))
    cost = float(day["config"].storage.cycle_cost)
    lmp = day["result"]["lmp"]
    shares = {}
    for a in _storage_agents(day["agents"]):
        margin = round_trip_margin_cny_mwh(lmp[:, a.bus], a.storage.eta_ch,
                                           a.storage.eta_dis, cost)
        shares[int(a.bus)] = tradeable_box_share(margin, action_box())
    return shares


def siting_pool(run_kwargs):
    """Every bus a battery is allowed to sit on, in network order.

    One condition, and it is a constraint rather than a score: the bus carries
    no renewable, which preserves the spatial separation the network has always
    had (PV at 21/23, wind at 30, batteries elsewhere). Otherwise a siting
    search could buy its margin by putting a battery on top of a generator it
    would then be paid to absorb.

    Deliberately **not** filtered by any margin. A pool is a search bound, and
    a bound that excludes the optimum cannot be repaired by the objective --
    while the two available margin readings disagree sharply and each is wrong
    on its own. Read jointly, a deep-feeder bus carries a voltage-constraint
    shadow price that a battery sited there would itself relieve (bus 17: 1405
    CNY/MWh jointly against 45.8 with a single unit); read in isolation, the
    bus shows the value a lone unit would find before the other three flatten
    it. Both are in the artifact as `by_bus` and `siting_sweep`. The criterion
    that decides is the *realized* joint share, which `layout_search`
    evaluates for each candidate.
    """
    day = run_day(**run_kwargs)
    re_buses = {a.bus for a in day["agents"]
                if getattr(a, "pv_capacity", 0.0) > 0
                or getattr(a, "wind_capacity", 0.0) > 0}
    return sorted(b for b in day["result"].get("bus_indices", [])
                  if int(b) not in re_buses and int(b) != _slack_bus(day))


def layout_search(run_kwargs, pool=None, start=None):
    """Choose four buses by hill-climbing the joint tradeable-share criterion.

    The objective is lexicographic: how many units clear the eligibility bar
    first, then the worst-placed unit's share, then the total. The floor comes
    first because the round's goal is four units that each have a strategy
    space, not one unit with a large one -- and the layouts that maximise the
    total do so by giving one unit almost everything.

    Reproducible: the same pool and starting layout search the same
    neighbourhood in the same order, so the run that selected the committed
    placement can be repeated from the CLI.
    """
    if pool is None:
        pool = siting_pool(run_kwargs)
    pool = sorted(set(pool))
    if len(pool) < 4:
        raise ValueError(
            f"only {len(pool)} buses qualify for placement, so four units "
            f"cannot be sited: {pool}")
    current = tuple(start or pool[:4])
    best = _layout_score(joint_tradeable_shares(run_kwargs, current))

    print("\n" + "=" * 78)
    print("layout search: hill-climb on the joint tradeable-share criterion")
    print("=" * 78)
    print("start %-22s eligible=%d worst=%.5f sum=%.3f"
          % (str(current), best[0], best[1], best[2]))
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            for bus in pool:
                if bus in current:
                    continue
                cand = tuple(sorted(current[:i] + (bus,) + current[i + 1:]))
                shares = joint_tradeable_shares(run_kwargs, cand)
                score = _layout_score(shares)
                if score > best:
                    best, current, improved = score, cand, True
                    print("  -> %-22s eligible=%d worst=%.5f sum=%.3f  %s"
                          % (str(cand), score[0], score[1], score[2],
                             " ".join("%.3f" % shares[b]
                                      for b in sorted(shares))))
    print("-" * 78)
    print("chosen %s" % (list(current),))
    print("  eligible=%d of %d, worst share=%.5f, sum=%.3f"
          % (best[0], len(current), best[1], best[2]))
    return list(current), joint_tradeable_shares(run_kwargs, current)


def _layout_score(shares):
    """(units clearing the bar, worst share, total) -- maximised in that order."""
    values = list(shares.values())
    return (sum(1 for s in values if s >= TRADEABLE_BOX_SHARE),
            min(values), sum(values))


def _parse_buses(text):
    """`all`, or a comma-separated list of bus numbers; None means all."""
    text = str(text).strip()
    if text.lower() == "all":
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="typical_day")
    parser.add_argument("--T", type=int, default=money.PERIODS_PER_DAY)
    parser.add_argument("--opf-mode", default="socp")
    parser.add_argument("--load-scale", type=float, default=None)
    parser.add_argument("--base", type=float, default=None,
                        help="price_curve.base override")
    parser.add_argument("--elasticity", type=float, default=None,
                        help="merit_order_price_elasticity override")
    parser.add_argument("--sweep-load-scale", default=None,
                        help="comma separated load_scale values")
    parser.add_argument("--sweep-elasticity", default=None,
                        help="comma separated elasticity values")
    parser.add_argument("--sweep-headroom", default=None,
                        help="comma separated inverter apparent-power "
                             "multipliers (1.0 = no headroom above the real "
                             "rating). SOCP only: the LinDistFlow batch solver "
                             "carries no inverter diamond, so the sweep reads "
                             "as no effect there.")
    parser.add_argument("--sweep-ess-buses", default=None,
                        help="clear the day once per candidate bus with a "
                             "single-unit fleet ('all' for every non-slack "
                             "bus, or a comma separated list). Slow: one full "
                             "clear per bus.")
    parser.add_argument("--ess-buses", default=None,
                        help="place the fleet at these buses for the report "
                             "run ('all' is not meaningful here)")
    parser.add_argument("--sweep-der-layouts", default=None,
                        help="candidate renewable layouts to clear, comma "
                             "separated, or 'all'. The candidates are the "
                             "plan's S-PV1/S-PV2/S-WIND/S-SPLIT; each is a set "
                             "of moves of an existing site, so installed MW "
                             "are conserved.")
    parser.add_argument("--der-headroom", type=float, default=1.10,
                        help="inverter multiplier the DER layouts are compared "
                             "at (default 1.10, the configured value)")
    parser.add_argument("--search-ess-buses", default=None,
                        help="hill-climb the joint tradeable-share criterion "
                             "to choose the fleet's placement. Give the "
                             "eligible pool, comma separated, or 'all' to "
                             "search every bus. Slow: one full clear per "
                             "neighbour evaluated.")
    parser.add_argument("--search-start", default=None,
                        help="comma separated four buses to start the layout "
                             "search from (default: the first four of the pool)")
    parser.add_argument("--unserved-limit", type=float, default=None,
                        help="MWh of load shedding to fail on; omitted, shedding "
                             "is reported and not gated")
    parser.add_argument("--min-shed", action="store_true",
                        help="MIN_SHED_RUN: solve the same day with total shed "
                             "ahead of every economic term, to separate what the "
                             "network cannot serve from what the market declined "
                             "to buy. Diagnostic only; not a market outcome.")
    parser.add_argument("--no-shed-voltage", action="store_true",
                        help="NO_SHED_VOLTAGE_DIAGNOSTIC: force every unit served "
                             "and report which network limit breaks first. No "
                             "market clearing runs.")
    parser.add_argument("--json", default=None, help="write the report as JSON here")
    parser.add_argument("--quiet", action="store_true", help="JSON only")
    args = parser.parse_args(argv)

    # The two shed diagnostics answer a different question from the market run,
    # so they replace it rather than being appended to it.
    if args.min_shed or args.no_shed_voltage:
        if args.min_shed:
            out = min_shed_run(scenario=args.scenario, T=args.T,
                               opf_mode=args.opf_mode)
            out.pop("result", None)
            out.pop("agents", None)
            out.pop("config", None)
        else:
            out = no_shed_voltage_diagnostic(scenario=args.scenario, T=args.T,
                                             opf_mode=args.opf_mode)
        if args.json:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(out, fh, indent=2, sort_keys=True)
                fh.write("\n")
        if not args.quiet:
            print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0

    global UNSERVED_LIMIT_MWH
    UNSERVED_LIMIT_MWH = args.unserved_limit

    run_kwargs = {"scenario": args.scenario, "T": args.T,
                  "opf_mode": args.opf_mode}

    headroom_rows = None
    if args.sweep_load_scale:
        _sweep_axis("load_scale", _parse_floats(args.sweep_load_scale),
                    dict(run_kwargs, price_base=args.base,
                         elasticity=args.elasticity), "load_scale")
    if args.sweep_elasticity:
        _sweep_axis("elasticity", _parse_floats(args.sweep_elasticity),
                    dict(run_kwargs, load_scale=args.load_scale,
                         price_base=args.base), "elasticity")
    if args.sweep_headroom:
        if args.opf_mode != "socp":
            print("note: the headroom sweep is SOCP-only; the LinDistFlow "
                  "batch solver has no inverter diamond to widen, so a run "
                  "there reports no effect for a reason that is not physics.")
        headroom_rows = _sweep_axis(
            "headroom", _parse_floats(args.sweep_headroom),
            dict(run_kwargs, load_scale=args.load_scale,
                 price_base=args.base, elasticity=args.elasticity),
            "inverter_headroom")
    der_rows = None
    if args.sweep_der_layouts:
        text = args.sweep_der_layouts.strip()
        names = list(DER_LAYOUTS) if text.lower() == "all" \
            else [x.strip() for x in text.split(",") if x.strip()]
        unknown = [n for n in names if n not in DER_LAYOUTS]
        if unknown:
            parser.error(f"unknown DER layout(s) {unknown}; "
                         f"available: {list(DER_LAYOUTS)}")
        der_rows = der_layout_sweep(
            dict(run_kwargs, load_scale=args.load_scale, price_base=args.base,
                 elasticity=args.elasticity),
            args.der_headroom, names)
        if args.json is None:
            return 0

    search_result = None
    if args.search_ess_buses:
        _buses = _parse_buses(args.search_ess_buses)
        pool = _buses if _buses is not None else None
        start = (_parse_buses(args.search_start)
                 if args.search_start else None)
        chosen, shares = layout_search(
            dict(run_kwargs, load_scale=args.load_scale, price_base=args.base,
                 elasticity=args.elasticity),
            pool=pool, start=start)
        search_result = {
            "chosen": chosen,
            "pool": sorted(set(pool)) if pool is not None
                    else siting_pool(dict(run_kwargs, load_scale=args.load_scale,
                                          price_base=args.base,
                                          elasticity=args.elasticity)),
            "start": start,
            "shares": {str(b): s for b, s in shares.items()},
            "threshold": TRADEABLE_BOX_SHARE,
        }
        if args.json is None:
            return 0

    siting_rows = None
    if args.sweep_ess_buses:
        siting_rows = siting_sweep(
            dict(run_kwargs, load_scale=args.load_scale, price_base=args.base,
                 elasticity=args.elasticity),
            _parse_buses(args.sweep_ess_buses))
        if args.json is None:
            return 0

    if args.sweep_load_scale or args.sweep_elasticity or args.sweep_headroom:
        if args.json is None:
            return 0

    day = run_day(**dict(run_kwargs, load_scale=args.load_scale,
                         price_base=args.base, elasticity=args.elasticity,
                         ess_buses=_parse_buses(args.ess_buses)
                         if args.ess_buses else None))
    report = build_report(day)
    if headroom_rows is not None:
        report["headroom_sweep"] = headroom_rows
    if der_rows is not None:
        report["der_layouts"] = der_rows
    if search_result is not None:
        report["layout_search"] = search_result
    if siting_rows is not None:
        # Recorded next to the layout it argues about, so the siting claim is
        # re-derivable from the artifact rather than from a terminal scrollback.
        report["siting_sweep"] = siting_rows
        report["siting_rule"] = {
            "tradeable_box_share_threshold": TRADEABLE_BOX_SHARE,
            "margin_cny_mwh_threshold": _margin_for_share(TRADEABLE_BOX_SHARE,
                                                          action_box()),
            "rule": "eligibility is the share of the action box the unit can "
                    "still trade in; the threshold was fixed before the sweep "
                    "was run",
        }
    if not args.quiet:
        system = report["system"]
        print(render(day, system, report["batteries"], report["network"],
                     report["price"], report["siting"], report["gates"],
                     report["reconciliation"], report["accepted_deviations"],
                     report.get("diagnostics")))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=False)
        if not args.quiet:
            print("\nwrote %s" % args.json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
