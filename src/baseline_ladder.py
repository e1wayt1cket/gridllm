# baseline_ladder.py
"""The B0-B3 physical baseline ladder, from the bare benchmark to the ESS fleet.

Four rungs, each one addition to the one before:

    B0  the original IEEE 33-bus at its own nominal load. A static power-flow
        snapshot: no market, no DER, no storage, no shedding mechanism.
    B1  the same spatial structure and the same nominal demand, spread over 96
        periods by one shared, peak-normalized profile, and no distributed
        generation.
    B2  B1 plus the configured PV and wind.
    B3  B2 plus the independent storage fleet under a deterministic rule.

B0 is a *benchmark reference* and B1 is the *scenario day*: one is a snapshot
at the load the network was published with, the other is that same load varying
over a day. They share the nodal distribution and the total, which is what
makes B0's voltage result the thing B1 starts from rather than an unrelated
measurement.

Two voltage bands are run for every rung. `PRIMARY_BAND` is the operational one
the main results are reported at, and `STRICT_BAND` is a sensitivity case:

    primary 0.93 <= V <= 1.05   GB/T 12325 +/-7%
    strict  0.95 <= V <= 1.05   the band the enhanced-IEEE33 literature uses

They differ only in the lower bound, which is the one this network binds on.
Neither is reached by tuning, and no rung's scenario parameters differ between
them -- the difference is the band's own.

The pressure is the network's, not the study's. Measured, the original IEEE
33-bus at its own nominal load has Vmin 0.9131 pu at bus 17: 21 of its 33 buses
are below 0.95 and 14 are below 0.93. The 0.93 band therefore binds at the
benchmark's own demand, before anything is added to it, which is why no load
scale had to be chosen to make the day interesting.

Every rung is written as one JSON record. The same configuration, seed and
scenario reproduce it byte for byte.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

import money
from models import MarketConfig, MarketDesignConfig

# The scenario every B1-B3 rung is built from.
BASE_SCENARIO = "typical_day"

RUNGS = ("B0", "B1", "B2", "B3")

#: The band the main results are reported at: GB/T 12325, the project's own
#: operational limits and the ones `defaults.yaml` already carries.
PRIMARY_BAND = (0.93, 1.05)
#: A sensitivity band. It differs from the primary one only in its lower bound,
#: which is the quantity this network actually binds on.
STRICT_BAND = (0.95, 1.05)
BANDS: Dict[str, tuple] = {"primary": PRIMARY_BAND, "strict": STRICT_BAND}

#: Which rung carries which fleet. B1 has no distributed generation, B2 has no
#: battery, B3 has both.
_RUNG_FLEETS = {
    "B1": {"der": False, "ess": False},
    "B2": {"der": True, "ess": False},
    "B3": {"der": True, "ess": True},
}


def _jsonable(obj):
    """Plain Python for anything numpy or numpy-scalar shaped."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# B0 -- the original benchmark
# ---------------------------------------------------------------------------

def run_b0(band_name: str, band) -> dict:
    """The original IEEE 33-bus at its own nominal load.

    A static power flow and nothing else. There is no market clearing here, so
    the absence of shed load is a statement about the *model*, not about the
    network: B0 has no shedding mechanism to shed with. It is reported as
    `shed_mechanism: "absent"` so no reader mistakes the zero for a feasibility
    result. Likewise there is no horizon, so the energy figures are null rather
    than zero.
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case33bw()
    total_p = float(net.load.p_mw.sum())
    total_q = float(net.load.q_mvar.sum())
    try:
        pp.runpp(net, algorithm="bfsw")
        converged = True
    except Exception:
        converged = False

    if not converged:
        return {
            "rung": "B0", "voltage_band": band_name, "convergence": False,
            "network": {"vmin": None, "vmax": None, "vmin_bus": None,
                        "max_line_utilization": None,
                        "voltage_violation_bus_periods": None,
                        "voltage_binding": None,
                        "shed": None, "shed_mechanism": "absent",
                        "curtailment": None},
            "system": {"peak_load": None, "energy_load": None,
                       "metric_not_applicable": True},
            "market": None, "ess": [],
        }

    vm = net.res_bus.vm_pu
    loading = net.res_line.loading_percent / 100.0

    # The `case33bw` that pandapower ships carries `max_i_ka = 99999` on every
    # line -- no thermal limit at all -- so `loading_percent` against it is a
    # current divided by infinity and reads as ~2e-6 whatever the flow is. The
    # project derives a rating from each line's own resistance instead (see
    # `grid.build_base_network`); the same rule is applied here so this rung's
    # line loading is a quantity, and is comparable with B1-B3's.
    from config_loader import get_default
    base_ka = float(get_default("network.base_ampacity_ka", 0.25))
    min_ka = float(get_default("network.min_line_ka", 0.08))
    max_ka = float(get_default("network.max_line_ka", 0.80))
    r_ohm = (net.line.r_ohm_per_km * net.line.length_km).to_numpy(dtype=float)
    derived_ka = np.clip(base_ka / np.sqrt(np.maximum(r_ohm, 0.005)),
                         min_ka, max_ka)
    i_ka = net.res_line.i_ka.to_numpy(dtype=float)
    native_ka = net.line.max_i_ka.to_numpy(dtype=float)
    utilization = float(np.max(i_ka / derived_ka))

    return {
        "rung": "B0",
        "voltage_band": band_name,
        "voltage_band_bounds": list(band),
        # B0 is a benchmark reference: the original network at the load it was
        # published with, not this study's operating point.
        "reference": "native_benchmark",
        "convergence": True,
        "network": {
            "vmin": float(vm.min()),
            "vmax": float(vm.max()),
            "vmin_bus": int(vm.idxmin()),
            "vmax_bus": int(vm.idxmax()),
            # Against the project's derived ampacity at the native multiplier
            # of 1.0 -- B0 is the unmodified network, so no scenario capacity
            # multiplier applies to it.
            "max_line_utilization": utilization,
            "max_line_utilization_rating_used": "derived_ampacity_x1.0",
            # The same flow against the ratings the file actually ships, which
            # are unbounded. Kept so the substitution above is visible.
            "max_line_utilization_vs_native_rating":
                float((i_ka / native_ka).max()),
            "native_line_rating_ka": float(native_ka.min()),
            "native_line_rating_is_unbounded": bool(native_ka.min() >= 9999.0),
            "buses_below_0_95": int((vm < 0.95).sum()),
            "buses_below_0_93": int((vm < 0.93).sum()),
            # No market clears here, so no voltage constraint is enforced and
            # nothing can be violated. Zero, stated as a mechanism's absence.
            "voltage_violation_bus_periods": 0,
            "voltage_binding": False,
            "shed": 0.0,
            "shed_mechanism": "absent",
            "curtailment": None,
        },
        "system": {
            "total_p_mw": total_p,
            "total_q_mvar": total_q,
            "n_buses": int(len(net.bus)),
            "n_lines": int(len(net.line)),
            # A snapshot has no horizon and no energy. Null, not zero.
            "peak_load": None,
            "energy_load": None,
            "pv_energy": None,
            "wind_energy": None,
            "renewable_consumption": None,
            "metric_not_applicable": True,
        },
        "market": None,
        "ess": [],
    }


# ---------------------------------------------------------------------------
# B1-B3 -- the project scenario
# ---------------------------------------------------------------------------

def _config(band, opf_mode: str = "socp") -> MarketConfig:
    """The scenario's own model, at this rung's voltage band.

    Built by `scenarios.scenario_model_config`, which is the same construction
    the scenario itself uses. This used to assemble a `MarketConfig` from the
    dataclass defaults, which agree with `config/defaults.yaml` on most fields
    and not on the ones the scenario declares: the rung cleared at a load power
    factor of 0.90 against the scenario's 0.8503, and with an inverter rated at
    1.0 against the scenario's 1.10. Both are choices about this study's
    facilities, and a rung that does not carry them is not a rung of that
    scenario.

    The band and the rung's own silence about storage are the only things set
    here -- `storage.use_nodal_price` in particular is left as configured,
    because its two-pass re-solve is part of the market model the ladder is
    measuring.
    """
    from scenarios import scenario_model_config

    config = scenario_model_config(BASE_SCENARIO, opf_mode=opf_mode)
    config.verbose = False
    config.network.v_min_pu = float(band[0])
    config.network.v_max_pu = float(band[1])
    return config


def _zero_der(agents) -> None:
    """Remove the distributed generation from a built population.

    Done on the agents rather than through a scenario multiplier because the
    fleet is what has to be absent: a zeroed profile and an absent array are
    the same thing to every consumer, and this way no capacity is left behind
    for a device rating to be derived from.
    """
    for a in agents:
        for attr in ("pv_forecast", "pv_real", "wind_forecast", "wind_real"):
            if getattr(a, attr, None) is not None:
                setattr(a, attr, np.zeros_like(np.asarray(getattr(a, attr),
                                                          dtype=float)))
        for attr in ("pv_capacity", "wind_capacity"):
            if hasattr(a, attr):
                setattr(a, attr, 0.0)


def build_rung_agents(rung: str, band, seed: Optional[int] = None,
                      opf_mode: str = "socp"):
    """(agents, wholesale, config) for one rung, built the same way each time.

    Both config caches are cleared first. They are process-global, so a
    per-run override another caller wrote -- a sweep setting an inverter
    multiplier, say -- would otherwise be inherited by this rung and quietly
    change the day. That leak is not hypothetical: it is what makes a rung
    depend on the order it was built in, and it is the same trap
    `diagnose_typical_day.run_day` clears its caches for.
    """
    from config_loader import reload_defaults, reload_scenarios
    from scenarios import get_scenario

    reload_defaults()
    reload_scenarios()

    if rung not in _RUNG_FLEETS:
        raise ValueError(f"rung {rung!r} has no fleet definition; "
                         f"expected one of {sorted(_RUNG_FLEETS)}")
    fleets = _RUNG_FLEETS[rung]
    config = _config(band, opf_mode)

    if seed is not None:
        np.random.seed(seed)
    # B3 keeps the configured fleet; B1 and B2 clear the day with no battery.
    ess_buses = None if fleets["ess"] else []
    agents, wholesale = get_scenario(BASE_SCENARIO, T=money.PERIODS_PER_DAY,
                                     config=copy.deepcopy(config),
                                     ess_buses=ess_buses)
    if not fleets["der"]:
        _zero_der(agents)
    return agents, wholesale, config


def _network_block(result, band) -> dict:
    """Network state from a cleared day."""
    if result is None:
        return {"vmin": None, "vmax": None, "vmin_bus": None,
                "max_line_utilization": None,
                "voltage_violation_bus_periods": None,
                "voltage_binding": None, "shed": None,
                "shed_mechanism": "market_clearing", "curtailment": None}

    v = result.get("bus_voltage")
    util = result.get("line_utilization")
    buses = result.get("bus_indices") or []

    if v is not None:
        v = np.asarray(v, dtype=float)
        vmin, vmax = float(v.min()), float(v.max())
        vmin_bus = int(buses[int(np.unravel_index(v.argmin(), v.shape)[1])]) \
            if buses else None
        # Bus-periods outside the band, in either direction.
        outside = int(((v < band[0]) | (v > band[1])).sum())
        # The constraint binds when the solver parked a voltage on one of its
        # bounds. This is the quantity that distinguishes "the market chose not
        # to serve" from "the network made service infeasible".
        binding = bool(np.isclose(v.min(), band[0], atol=1e-6)
                       or np.isclose(v.max(), band[1], atol=1e-6))
    else:
        vmin = vmax = vmin_bus = outside = None
        binding = None

    shed_mwh = 0.0
    for sched in (result.get("schedules") or {}).values():
        if "unserved" in sched:
            shed_mwh += float(np.sum(sched["unserved"])) * money.DT_HOURS

    return {
        "vmin": vmin,
        "vmax": vmax,
        "vmin_bus": vmin_bus,
        "max_line_utilization": float(np.asarray(util).max())
        if util is not None else None,
        "voltage_violation_bus_periods": outside,
        "voltage_binding": binding,
        "shed": shed_mwh,
        "shed_mechanism": "market_clearing",
        "curtailment": float(result.get("total_curtailment", 0.0)),
    }


def _system_block(agents, result) -> dict:
    """Load and renewable energy over the day."""
    loads = [np.asarray(a.load_real, dtype=float) for a in agents
             if getattr(a, "load_real", None) is not None]
    load_shape = np.sum(loads, axis=0) if loads else np.zeros(1)

    def _energy(arr):
        return float(np.sum(arr)) * money.DT_HOURS

    pv = sum(_energy(np.asarray(a.pv_real, dtype=float)) for a in agents
             if getattr(a, "pv_real", None) is not None) if agents else 0.0
    wind = sum(_energy(np.asarray(a.wind_real, dtype=float)) for a in agents
               if getattr(a, "wind_real", None) is not None) if agents else 0.0

    re_used = 0.0
    for sched in (result.get("schedules") or {}).values():
        if "pv_used" in sched:
            re_used += _energy(np.asarray(sched["pv_used"], dtype=float))
        if "wind_used" in sched:
            re_used += _energy(np.asarray(sched["wind_used"], dtype=float))

    return {
        "peak_load": float(load_shape.max()),
        "energy_load": _energy(load_shape),
        "pv_energy": pv,
        "wind_energy": wind,
        "renewable_consumption": re_used,
        "metric_not_applicable": False,
    }


def _market_block(result, band) -> Optional[dict]:
    if result is None:
        return None
    lmp = np.asarray(result["lmp"], dtype=float)
    return {
        "clearing_objective": float(result["clearing_objective"]),
        "lmp_mean": float(lmp.mean()),
        "lmp_min": float(lmp.min()),
        "lmp_max": float(lmp.max()),
        "voltage_band": list(band),
    }


def _ess_fleet_block(units: Sequence[dict], activity_tol: float = 1e-6) -> dict:
    """The fleet's own totals, summed from the per-unit rows.

    Summed rather than measured separately, so the fleet figure and the
    per-unit figures cannot disagree. `active` counts units that actually
    moved energy in either direction: a unit that sat out the day is what a
    fleet-level profit hides, and it is the first thing to look at when the
    aggregate looks reasonable.
    """
    if not units:
        return {"n_units": 0, "active_units": 0,
                "charge_energy_mwh": 0.0, "discharge_energy_mwh": 0.0,
                "equivalent_cycles": 0.0, "profit": 0.0,
                "inactive_buses": []}
    active = [u for u in units
              if u["charge_energy"] > activity_tol
              or u["discharge_energy"] > activity_tol]
    return {
        "n_units": len(units),
        "active_units": len(active),
        "charge_energy_mwh": float(sum(u["charge_energy"] for u in units)),
        "discharge_energy_mwh": float(sum(u["discharge_energy"]
                                          for u in units)),
        "equivalent_cycles": float(sum(u["cycles"] for u in units)),
        "profit": float(sum(u["profit"] for u in units)),
        "inactive_buses": sorted(u["bus"] for u in units
                                 if u not in active),
    }


def _ess_block(result, agents, config) -> List[dict]:
    """Per-unit economics, from the endpoint-pinned settlement."""
    if result is None:
        return []
    import participant_payoff

    out = []
    for a in agents:
        if getattr(a, "participant_type", "") != "storage":
            continue
        sched = (result.get("schedules") or {}).get(a.name)
        if sched is None:
            continue
        bd = participant_payoff.participant_payoff(
            sched, np.asarray(result["lmp"])[:, a.bus], a, config)
        out.append({
            "bus": int(a.bus),
            "charge_energy": bd.charge_mwh,
            "discharge_energy": bd.discharge_mwh,
            "cycles": bd.equivalent_cycles,
            "degradation_cost": bd.degradation_cost,
            "revenue": bd.revenue,
            "purchase_cost": bd.purchase_cost,
            "profit": bd.total,
        })
    return out


def _declared_actions(agents, config, names, policy: str, seed: int) -> dict:
    """The quotes a policy emits causally, block by block, over one day.

    The rollout is causal on purpose: the policy sees only what a real
    participant would have seen at the time each block was committed. Only the
    *settlement* is done jointly afterwards, so the quotes driven here are the
    ones a day of operating this rule would actually have produced.
    """
    from baselines import make_baseline_policy
    from rl_env import BiddingEnv, N_BLOCKS

    np.random.seed(seed)
    env = BiddingEnv(agents, config, rl_agent_names=list(names))
    pol = make_baseline_policy(policy, config)
    obs = env.reset()
    for _ in range(N_BLOCKS):
        acts = {nm: np.asarray(pol(obs[nm]), dtype=np.float32) for nm in names}
        next_obs, _rewards, done, _info = env.step(acts)
        if done:
            break
        obs = next_obs
    return {nm: {k: np.asarray(v, dtype=float)
                 for k, v in env.current_actions[nm].items()}
            for nm in names}


def run_market_rung(rung: str, band_name: str, band, seed: int,
                    opf_mode: str = "socp", policy: str = "rule") -> dict:
    """Clear one B1-B3 day and report it.

    The day is settled by the same endpoint-pinned settlement the paired
    evaluator uses, so B3's storage profit is the quantity the research is
    about and not a rolling-window cash sum that rewards running the battery
    down. B3 additionally reports the truthful arm, because a profit only
    means something against the counterfactual the rest of the study measures
    against.
    """
    from counterfactual_eval import settle_day
    from rl_env import BiddingEnv

    agents, wholesale, config = build_rung_agents(rung, band, seed, opf_mode)
    names = [a.name for a in agents
             if getattr(a, "participant_type", "") == "storage"]

    if names:
        declared = _declared_actions(agents, config, names, policy, seed)
    else:
        # No battery on this rung: nothing declares anything, and one truthful
        # clear is the whole day.
        declared = {}

    settled, clean, result = settle_day(
        agents, config, names, declared, wholesale, money.PERIODS_PER_DAY)

    day = {
        "rung": rung,
        "voltage_band": band_name,
        "voltage_band_bounds": list(band),
        "seed": seed,
        "engine": opf_mode,
        "scenario_id": BASE_SCENARIO,
        "reference": "project_scenario",
        "convergence": result is not None,
        "settled_clean": bool(clean),
        "policy": policy if names else None,
        "network": _network_block(result, band),
        "system": _system_block(agents, result),
        "market": _market_block(result, band),
        "ess": _ess_block(result, agents, config),
        "ess_fleet": _ess_fleet_block(_ess_block(result, agents, config)),
    }

    if names:
        # The fleet's own counterfactual: the same day with every unit
        # declaring no premium.
        np.random.seed(seed)
        truthful_env = BiddingEnv(agents, config, rl_agent_names=list(names))
        truthful_env.reset()
        base_declared = truthful_env._default_actions()
        base_profits, base_clean, _ = settle_day(
            agents, config, names, base_declared, wholesale,
            money.PERIODS_PER_DAY)
        pi_ai = sum(settled.get(nm, 0.0) for nm in names)
        pi_baseline = sum(base_profits.get(nm, 0.0) for nm in names)
        day["pi_ai"] = pi_ai
        day["pi_baseline"] = pi_baseline
        day["delta_pi"] = pi_ai - pi_baseline
        day["delta_pi_percent"] = (
            (pi_ai - pi_baseline) / (abs(pi_baseline) + 1e-6))
        day["baseline_settled_clean"] = bool(base_clean)
    return day


# ---------------------------------------------------------------------------
# The ladder, and its band sensitivity
# ---------------------------------------------------------------------------

def run_rung(rung: str, band_name: str, seed: int, opf_mode: str = "socp",
             policy: str = "rule") -> dict:
    band = BANDS[band_name]
    if rung == "B0":
        record = run_b0(band_name, band)
        record["seed"] = seed
        record["engine"] = "power_flow"
        record["scenario_id"] = "ieee33_native"
        return record
    return run_market_rung(rung, band_name, band, seed, opf_mode, policy)


def band_sensitivity(records: Sequence[dict]) -> dict:
    """What the strict band costs against the primary one, rung by rung.

    Only meaningful where a market clears: B0 has no clearing, so its
    difference is reported as absent rather than as zero.
    """
    by_rung: Dict[str, dict] = {}
    for rung in RUNGS:
        p = next((r for r in records
                  if r["rung"] == rung and r["voltage_band"] == "primary"),
                 None)
        s = next((r for r in records
                  if r["rung"] == rung and r["voltage_band"] == "strict"),
                 None)
        if p is None or s is None or p.get("market") is None \
                or s.get("market") is None:
            by_rung[rung] = {"applicable": False,
                             "reason": "no market clearing at this rung"}
            continue
        by_rung[rung] = {
            "applicable": True,
            "delta_shed_95_vs_93": s["network"]["shed"] - p["network"]["shed"],
            "delta_curtailment_95_vs_93":
                s["network"]["curtailment"] - p["network"]["curtailment"],
            "delta_lmp_95_vs_93":
                s["market"]["lmp_mean"] - p["market"]["lmp_mean"],
            "delta_profit_95_vs_93":
                (s.get("pi_ai") - p.get("pi_ai"))
                if s.get("pi_ai") is not None
                and p.get("pi_ai") is not None else None,
        }
    return by_rung


def build_ladder(seed: int = 123, bands=("primary", "strict"),
                 rungs=RUNGS, opf_mode: str = "socp",
                 policy: str = "rule") -> dict:
    """Every requested rung at every requested band, as one JSON-able doc."""
    records = []
    for rung in rungs:
        for band_name in bands:
            records.append(run_rung(rung, band_name, seed, opf_mode, policy))
    return _jsonable({
        "seed": seed,
        "engine": opf_mode,
        "policy": policy,
        "voltage_band": {
            "primary": list(PRIMARY_BAND),
            "strict": list(STRICT_BAND),
        },
        "rungs": records,
        "band_sensitivity": band_sensitivity(records),
    })


def physical_vs_strategic(records: Sequence[dict],
                          matd3: Optional[dict] = None) -> dict:
    """Two different questions, reported side by side and never summed.

    **Physical** is B2 -> B3: the effect of installing the batteries, with no
    change in how anyone bids. It is a fact about the equipment.

    **Strategic** is the bidding effect at *fixed* facilities -- B3 throughout,
    truthful against the rule against MATD3. It is the only one of the two that
    anything can be attributed to AI bidding.

    Adding them would credit the policy with the batteries. They are returned
    under separate keys for that reason, and the physical block carries the rung
    names it compares so a reader cannot mistake its scope.
    """
    def _row(rung, band="primary"):
        return next((r for r in records
                     if r["rung"] == rung and r["voltage_band"] == band), None)

    b2, b3 = _row("B2"), _row("B3")
    physical = {"comparison": "B2 -> B3 (facility effect)",
                "attributable_to_bidding": False}
    if b2 and b3:
        physical.update({
            "delta_shed_mwh": b3["network"]["shed"] - b2["network"]["shed"],
            "shed_b2_mwh": b2["network"]["shed"],
            "shed_b3_mwh": b3["network"]["shed"],
            "delta_curtailment_mwh": (b3["network"]["curtailment"]
                                      - b2["network"]["curtailment"]),
            "delta_lmp_mean": (b3["market"]["lmp_mean"]
                               - b2["market"]["lmp_mean"]),
            "delta_clearing_objective":
                (b3["market"]["clearing_objective"]
                 - b2["market"]["clearing_objective"]),
        })

    strategic = {"comparison": "truthful -> rule -> MATD3 at B3 facilities",
                 "attributable_to_bidding": True}
    if b3:
        strategic.update({
            "pi_baseline_truthful": b3.get("pi_baseline"),
            "pi_rule": b3.get("pi_ai"),
            "delta_pi_rule_vs_truthful": b3.get("delta_pi"),
            "delta_pi_percent_rule": b3.get("delta_pi_percent"),
        })
    if matd3 is not None:
        rows = matd3.get("rows") or []
        if rows:
            strategic["matd3_seeds"] = [r["seed"] for r in rows]
            strategic["matd3_episodes"] = matd3.get("episodes")
            strategic["matd3_smoke_passed"] = matd3.get("all_passed")
    return {"physical_b2_to_b3": physical, "strategic_at_b3": strategic}


def scenario_inventory(seed: int = 123) -> List[dict]:
    """Every scenario's declared scale, renewable capacity and storage.

    Built so that "no scenario inherits another's scale" is checkable rather
    than asserted: each row shows what the scenario declares *and* what that
    actually produces, so a declaration that does not reach the agents is
    visible as a mismatch. `scenario_scale_source` says where the scale came
    from -- a row reading `legacy_explicit` is at the old study scale and is
    not the benchmark's own demand, which is the distinction the field keeps.
    """
    import copy as _copy

    from config_loader import get_scenario_cfg
    from scenarios import get_scenario, list_scenarios

    rows = []
    for name in list_scenarios():
        cfg = get_scenario_cfg(name)
        renewable = cfg.get("renewable") or {}
        row = {
            "scenario": name,
            "scenario_scale_source": cfg.get("scenario_scale_source", "MISSING"),
            "load_scale_mode": cfg.get("load_scale_mode", "MISSING"),
            "load_shape": cfg.get("load_shape", "MISSING"),
            "load_power_factor_declared": cfg.get("load_power_factor"),
            "peak_load_mw_declared": cfg.get("peak_load_mw"),
            "pv_bus_mw_declared": {int(k): float(v) for k, v in
                                   (renewable.get("pv_bus_mw") or {}).items()},
            "wind_bus_mw_declared": {int(k): float(v) for k, v in
                                     (renewable.get("wind_bus_mw") or {}).items()},
        }
        # What the declaration actually builds.
        try:
            agents, _wholesale = get_scenario(
                name, T=money.PERIODS_PER_DAY,
                config=_copy.deepcopy(_config(PRIMARY_BAND)))
            P = np.sum([np.asarray(a.load_real, dtype=float) for a in agents],
                       axis=0)
            row.update({
                "realized_peak_mw": float(P.max()),
                "realized_energy_mwh": float(np.sum(P)) * money.DT_HOURS,
                "realized_pv_mw": sum(float(getattr(a, "pv_capacity", 0.0) or 0.0)
                                      for a in agents),
                "realized_wind_mw": sum(float(getattr(a, "wind_capacity", 0.0) or 0.0)
                                        for a in agents),
                "n_ess": sum(1 for a in agents
                             if getattr(a, "participant_type", "") == "storage"),
            })
        except Exception as exc:                      # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
        rows.append(row)
    return rows


def write_ladder(doc: dict, path: str) -> str:
    """Write the ladder with sorted keys, so it is byte-for-byte stable."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv=None) -> int:
    # A non-formal entry point: this measures the control path against a
    # market with a known shape, not against a measured one.
    from wholesale_price import enable_non_formal_market
    enable_non_formal_market()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", default="all",
                        help="B0 | B1 | B2 | B3 | all")
    parser.add_argument("--bands", default="both",
                        help="primary | strict | both")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opf-mode", default="socp")
    parser.add_argument("--policy", default="rule",
                        help="the deterministic storage policy for B3")
    parser.add_argument("--inventory", action="store_true",
                        help="list every scenario's declared and realized "
                             "scale, renewable capacity and storage, then exit")
    parser.add_argument("--json", default=None,
                        help="write the ladder as JSON here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.inventory:
        rows = scenario_inventory(args.seed)
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                        exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(rows, fh, indent=2, sort_keys=True)
                fh.write("\n")
        if not args.quiet:
            print(f"{'scenario':<17} {'source':<17} {'mode':<14} "
                  f"{'peak MW':>8} {'PV MW':>6} {'wind MW':>8} {'ESS':>4}")
            for r in rows:
                print(f"{r['scenario']:<17} {r['scenario_scale_source']:<17} "
                      f"{r['load_scale_mode']:<14} "
                      f"{r.get('realized_peak_mw', float('nan')):>8.4f} "
                      f"{r.get('realized_pv_mw', 0.0):>6.2f} "
                      f"{r.get('realized_wind_mw', 0.0):>8.2f} "
                      f"{r.get('n_ess', 0):>4d}")
        return 0

    rungs = RUNGS if args.rung == "all" else (args.rung,)
    bands = (("primary", "strict") if args.bands == "both"
             else (args.bands,))

    doc = build_ladder(seed=args.seed, bands=bands, rungs=rungs,
                       opf_mode=args.opf_mode, policy=args.policy)
    if args.json:
        write_ladder(doc, args.json)
        if not args.quiet:
            print(f"wrote {args.json}")
    if not args.quiet:
        for r in doc["rungs"]:
            net = r["network"]
            print(f"  {r['rung']} [{r['voltage_band']:7s}] "
                  f"vmin={net['vmin'] if net['vmin'] is None else format(net['vmin'], '.4f')} "
                  f"shed={net['shed'] if net['shed'] is None else format(net['shed'], '.4f')} "
                  f"curt={net['curtailment']}")
        sep = physical_vs_strategic(doc["rungs"])
        print()
        print("Physical (B2 -> B3): the batteries, not the bidding")
        for k, v in sep["physical_b2_to_b3"].items():
            print(f"  {k}: "
                  f"{v if not isinstance(v, float) else format(v, '+.4f')}")
        print("Strategic (at B3): the bidding, at fixed facilities")
        for k, v in sep["strategic_at_b3"].items():
            print(f"  {k}: "
                  f"{v if not isinstance(v, float) else format(v, '+.2f')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

