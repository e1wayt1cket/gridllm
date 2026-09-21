# ess_siting.py
"""Where an independent battery belongs on the IEEE 33-bus feeder.

The method is **B0 operating-point-based finite-difference voltage
self-sensitivity screening**. Each bus's score is how far its own voltage moves
when a small real injection is placed at that same bus, measured on the
benchmark network at the load it was published with:

    S_i = dV_i / dP_i  ~=  [V_i(P_i + eps) - V_i(P_i)] / eps

That is the standard distribution-generation placement criterion: an injection
helps the network most where the network's own voltage is weakest, so the buses
with the largest `S` are the ones worth putting a battery on. It is a property
of the network and its load, and it involves no market quantity at all.

Three things this deliberately is not:

* **Not electrical distance.** `dispatch_core._compute_bus_electrical_distance`
  (cumulative line resistance to the slack) is reported alongside as the
  diagnostic column `electrical_distance`, but it is a proxy for sensitivity
  rather than sensitivity, it ignores the load distribution entirely, and it
  takes no part in the screen.
* **Not a profit ranking.** A bus is never chosen, kept or dropped because of
  what a battery there would earn. Market quantities appear only in the
  screening table, as a feasibility diagnostic.
* **Not the top four by sensitivity.** Four units on the four most sensitive
  buses would sit on top of one another and arbitrage the same spread away.
  See `select_buses` for the rule that is used instead, which is declared
  before any measurement is taken.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

# Pre-load ortools before pandapower is reached, for the DLL load-order reason
# `dispatch_core` documents: pandapower's estimation subpackage imports ortools
# itself, and on Windows that import fails if it is the first one to run. This
# module reaches pandapower from inside a function, so without this line it
# works under pytest -- where an earlier module has already loaded ortools --
# and crashes when run as a script.
from ortools.linear_solver import pywraplp  # noqa: F401

import money

# ---------------------------------------------------------------------------
# The method's two declarations. Both are fixed here, before any measurement,
# so that neither can be adjusted to produce a preferred answer.
# ---------------------------------------------------------------------------

#: The injection used for the finite difference, in MW. Small enough to stay in
#: the linear regime and large enough that the voltage response is well above
#: the power flow's own tolerance. `check_eps_stability` re-runs the whole
#: screen at twice this and compares, so the choice is shown to not matter
#: rather than asserted to not matter.
SENSITIVITY_EPS = 0.01

#: A bus is a candidate when its sensitivity reaches this share of the largest
#: one. Expressed as a share rather than an absolute threshold so it carries to
#: another network. 25% is a judgement about how far down the ranking still
#: means "electrically weak", not a measured quantity.
SENSITIVITY_FLOOR = 0.25

#: How many units the fleet has.
FLEET_SIZE = 4


def voltage_self_sensitivity(eps: float = SENSITIVITY_EPS) -> Dict[int, float]:
    """`S_i` for every load bus, at the B0 native operating point.

    The benchmark case is rebuilt from `pandapower.networks.case33bw` for every
    bus rather than reused, because the injection has to be applied to a clean
    network each time; the unperturbed voltages are taken once from a single
    run of the same case.
    """
    import pandapower as pp
    import pandapower.networks as pn

    base = pn.case33bw()
    pp.runpp(base, algorithm="bfsw")
    v0 = base.res_bus.vm_pu.copy()

    out: Dict[int, float] = {}
    for bus in base.load.bus.astype(int):
        net = pn.case33bw()
        pp.create_sgen(net, bus=int(bus), p_mw=float(eps), q_mvar=0.0)
        pp.runpp(net, algorithm="bfsw")
        out[int(bus)] = float((net.res_bus.vm_pu[int(bus)]
                               - v0[int(bus)]) / eps)
    return out


def renewable_buses(scenario: str = "typical_day") -> set:
    """The buses the scenario declares renewable capacity on."""
    from config_loader import get_scenario_cfg

    renewable = get_scenario_cfg(scenario).get("renewable") or {}
    out = set()
    for key in ("pv_bus_mw", "wind_bus_mw"):
        for bus, mw in (renewable.get(key) or {}).items():
            if float(mw) > 0.0:
                out.add(int(bus))
    return out


def slack_bus() -> int:
    import pandapower.networks as pn

    from grid import build_base_network
    from models import MarketConfig
    net = build_base_network(MarketConfig(opf_mode="socp", verbose=False))
    return int(net.ext_grid.at[0, "bus"])


def electrical_distance() -> Dict[int, float]:
    """Cumulative line resistance from the slack to each bus.

    A DIAGNOSTIC COLUMN ONLY. It is an electrical distance, not a voltage
    sensitivity: it says how far a bus is from the source along the conductors,
    not how much its voltage responds to an injection. It takes no part in the
    screen and must not be described as the criterion.
    """
    from dispatch_core import _compute_bus_electrical_distance
    from grid import build_base_network
    from models import MarketConfig

    net = build_base_network(MarketConfig(opf_mode="socp", verbose=False))
    return {int(b): float(r)
            for b, r in _compute_bus_electrical_distance(net).items()}


def check_eps_stability(eps: float = SENSITIVITY_EPS) -> dict:
    """Re-run the screen at twice `eps` and compare.

    A finite difference is only a derivative if the step is in the linear
    regime. Comparing the ranking and the values at two step sizes shows the
    choice is not doing any work, rather than assuming it.
    """
    s1 = voltage_self_sensitivity(eps)
    s2 = voltage_self_sensitivity(2.0 * eps)
    buses = sorted(s1)
    rank1 = sorted(buses, key=lambda b: -s1[b])
    rank2 = sorted(buses, key=lambda b: -s2[b])
    scale = max(abs(v) for v in s1.values()) or 1.0
    return {
        "eps": eps,
        "eps_doubled": 2.0 * eps,
        "ranking_identical": rank1 == rank2,
        "max_relative_difference": max(
            abs(s1[b] - s2[b]) for b in buses) / scale,
        "ranking": rank1,
    }


def sensitivity_table(scenario: str = "typical_day",
                      eps: float = SENSITIVITY_EPS) -> List[dict]:
    """One row per bus: its score, whether it is a candidate, and why not."""
    s = voltage_self_sensitivity(eps)
    max_s = max(s.values())
    floor = SENSITIVITY_FLOOR * max_s
    re_buses = renewable_buses(scenario)
    slack = slack_bus()
    dist = electrical_distance()

    rows = []
    for bus in sorted(s):
        renewable_conflict = bus in re_buses
        passes_floor = s[bus] >= floor
        is_slack = bus == slack
        rows.append({
            "bus": int(bus),
            "dV_dP_pu_per_mw": s[bus],
            "normalized_sensitivity": s[bus] / max_s,
            "passes_sensitivity_floor": bool(passes_floor),
            "renewable_conflict": bool(renewable_conflict),
            "is_slack": bool(is_slack),
            "candidate": bool(passes_floor and not renewable_conflict
                              and not is_slack),
            # Diagnostic only; never used to screen.
            "electrical_distance": dist.get(int(bus)),
        })
    return rows


def candidate_buses(rows: Sequence[dict]) -> List[int]:
    """The pool the selection rule draws from, in network order."""
    return [r["bus"] for r in rows if r["candidate"]]


def select_buses(rows: Sequence[dict], k: int = FLEET_SIZE) -> List[int]:
    """Choose `k` buses from the candidate pool, without using any market data.

    The rule, declared before any measurement:

      1. The seed is the candidate with the largest sensitivity -- the single
         bus whose voltage is most responsive to a local injection.
      2. Each further pick is the candidate whose minimum pairwise series
         resistance to the buses already chosen is largest (max-min
         dispersion).

    Two things follow, and both are the point. The fleet is not the top `k` by
    sensitivity: only the seed is chosen that way, and the rest are spread out.
    And the spread is measured in ohms of shared conductor, so it is a
    statement about the network rather than about the market.

    Spread matters because units that share a conductor arbitrage the same
    daily spread and flatten it for one another. That was measured on this
    network at the old scale -- a bus that leaves a lone unit 186 CNY/MWh
    leaves it 17 once the fleet is bidding -- and nothing about the mechanism
    depends on the load level.
    """
    from models import MarketConfig
    from rl_physics import pairwise_resistance_distance
    from grid import build_base_network

    pool = candidate_buses(rows)
    if len(pool) < k:
        raise ValueError(
            f"only {len(pool)} candidate buses for a fleet of {k}; lower "
            f"SENSITIVITY_FLOOR or revisit the pool")
    by_bus = {r["bus"]: r for r in rows}

    net = build_base_network(MarketConfig(opf_mode="socp", verbose=False))
    D = pairwise_resistance_distance(net, sorted(pool))
    idx = {b: i for i, b in enumerate(sorted(pool))}

    chosen = [max(pool, key=lambda b: by_bus[b]["normalized_sensitivity"])]
    while len(chosen) < k:
        def spread(bus):
            if bus in chosen:
                return -np.inf
            return min(D[idx[bus], idx[c]] for c in chosen)
        chosen.append(max(pool, key=spread))
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Screening table: physical feasibility / market opportunity / economic outcome
# ---------------------------------------------------------------------------

def screen_bus(bus: int, scenario: str = "typical_day",
               policy: str = "rule") -> dict:
    """Measure one candidate bus with a single unit placed there.

    A DIAGNOSTIC, NOT A SELECTION METRIC. It answers what a unit arriving at
    this bus would find in today's market, which is the honest way to measure
    an *opportunity* -- but four units arbitrage the same daily spread, so the
    price field a candidate shows alone is one the others would have already
    flattened. `market_opportunity` is therefore measured here, and the
    fleet's realized opportunity is measured separately on the settled day.

    Both arms are reported: the unit declaring no premium, and the same unit
    driven by the deterministic rule. A bus where the two are identical is a
    bus where bidding cannot matter.
    """
    import copy

    from baseline_ladder import (PRIMARY_BAND, _declared_actions, _config)
    from counterfactual_eval import settle_day
    from scenarios import get_scenario

    # One unit, at this bus and nowhere else: the fleet is built with a single
    # placement rather than filtered afterwards, so there is no other battery
    # in the clearing to arbitrage the spread away.
    config = _config(PRIMARY_BAND)
    agents, wholesale = get_scenario(
        "typical_day", money.PERIODS_PER_DAY, copy.deepcopy(config),
        ess_buses=[int(bus)])
    names = [a.name for a in agents
             if getattr(a, "participant_type", "") == "storage"]
    if len(names) != 1:
        raise ValueError(f"expected one screened unit at bus {bus}; got {names}")
    target = names[0]
    agent = next(a for a in agents if a.name == target)

    truthful, clean, result = settle_day(
        agents, config, [target], {}, wholesale, money.PERIODS_PER_DAY)
    if result is None:
        return {"bus": int(bus), "error": "no result"}

    declared = _declared_actions(agents, config, [target], policy, 123)
    rule_profits, rule_clean, _ = settle_day(
        agents, config, [target], declared, wholesale, money.PERIODS_PER_DAY)

    import participant_payoff
    from diagnose_typical_day import (action_box, round_trip_margin_cny_mwh,
                                      tradeable_box_share)

    sched = result["schedules"][target]
    lmp_node = np.asarray(result["lmp"], dtype=float)[:, agent.bus]
    bd = participant_payoff.participant_payoff(sched, lmp_node, agent, config)

    cycle_cost = float(config.storage.cycle_cost)
    margin = round_trip_margin_cny_mwh(lmp_node, agent.storage.eta_ch,
                                       agent.storage.eta_dis, cycle_cost)
    charge = np.asarray(sched["p_ch"], dtype=float)
    discharge = np.asarray(sched["p_dis"], dtype=float)

    return {
        "bus": int(bus),
        # --- PHYSICAL FEASIBILITY ---
        "lmp_valley": float(lmp_node.min()),
        "lmp_peak": float(lmp_node.max()),
        # --- MARKET OPPORTUNITY ---
        "charge_opportunity_count": int(np.count_nonzero(charge > 1e-6)),
        "discharge_opportunity_count": int(np.count_nonzero(discharge > 1e-6)),
        "chargeable_energy_mwh": float(np.sum(charge)) * money.DT_HOURS,
        "dischargeable_energy_mwh": float(np.sum(discharge)) * money.DT_HOURS,
        "equivalent_cycles": bd.equivalent_cycles,
        # --- ECONOMIC OUTCOME ---
        "peak_valley_spread": float(lmp_node.max() - lmp_node.min()),
        "degradation_adjusted_spread": margin,
        "truthful_profit": bd.total,
        "rule_profit": rule_profits.get(target),
        "strategic_action_space_width": tradeable_box_share(margin,
                                                            action_box()),
        "settled_clean": bool(clean and rule_clean),
    }


def screening_table(buses: Sequence[int], scenario: str = "typical_day") -> List[dict]:
    return [screen_bus(int(b), scenario) for b in buses]


def feasible(row: dict) -> bool:
    """Whether a bus has at least one chance to charge and one to discharge.

    The only hard condition in this screen, and it is a statement about the
    mechanism rather than about money: a unit with no period in which charging
    is possible, or none in which discharging is, cannot participate at all.
    Profit and cycles are reported and never gate.
    """
    return (row.get("charge_opportunity_count", 0) > 0
            and row.get("discharge_opportunity_count", 0) > 0)


def build_report(scenario: str = "typical_day") -> dict:
    rows = sensitivity_table(scenario)
    pool = candidate_buses(rows)
    selected = select_buses(rows)
    # Screen every candidate, not only the chosen four: the table is the
    # evidence the selection was made from, and a reader has to be able to see
    # the buses the rule passed over.
    screened = screening_table(pool, scenario)

    sel_rows = [r for r in screened if r["bus"] in set(selected)]
    inactive = [r["bus"] for r in sel_rows if not feasible(r)]
    return {
        "method": "B0 operating-point-based finite-difference voltage "
                  "self-sensitivity screening",
        "scenario": scenario,
        "eps": SENSITIVITY_EPS,
        "sensitivity_floor": SENSITIVITY_FLOOR,
        "eps_stability": check_eps_stability(),
        "buses": rows,
        "candidate_pool": pool,
        "selected_buses": selected,
        "screening": screened,
        "structurally_inactive": inactive,
        "feasibility_ok": not inactive,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="typical_day")
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(args.scenario)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    if not args.quiet:
        st = report["eps_stability"]
        print(f"eps={st['eps']} vs 2*eps: ranking identical "
              f"{st['ranking_identical']}, max relative difference "
              f"{st['max_relative_difference']:.2e}")
        print()
        print(f"{'bus':>4} {'S':>9} {'norm':>7} {'cand':>5} {'RE':>4} "
              f"{'elec dist':>10}")
        for r in report["buses"]:
            print(f"{r['bus']:>4} {r['dV_dP_pu_per_mw']:>9.5f} "
                  f"{r['normalized_sensitivity']:>7.4f} "
                  f"{'yes' if r['candidate'] else 'no':>5} "
                  f"{'yes' if r['renewable_conflict'] else 'no':>4} "
                  f"{r['electrical_distance']:>10.4f}")
        r17 = next(r for r in report["buses"] if r["bus"] == 17)
        print()
        print(f"bus 17: S={r17['dV_dP_pu_per_mw']:.5f} "
              f"norm={r17['normalized_sensitivity']:.4f} "
              f"candidate={'yes' if r17['candidate'] else 'no'}")
        print(f"candidate pool ({len(report['candidate_pool'])}): "
              f"{report['candidate_pool']}")
        print(f"selected: {report['selected_buses']}")
        print()
        print(f"{'bus':>4} {'ch#':>5} {'dis#':>5} {'ch MWh':>8} {'dis MWh':>8} "
              f"{'cyc':>7} {'spread':>8} {'deg-adj':>9} {'profit':>8} {'box':>6}")
        for r in report["screening"]:
            print(f"{r['bus']:>4} {r['charge_opportunity_count']:>5} "
                  f"{r['discharge_opportunity_count']:>5} "
                  f"{r['chargeable_energy_mwh']:>8.4f} "
                  f"{r['dischargeable_energy_mwh']:>8.4f} "
                  f"{r['equivalent_cycles']:>7.4f} "
                  f"{r['peak_valley_spread']:>8.1f} "
                  f"{r['degradation_adjusted_spread']:>9.1f} "
                  f"{r['truthful_profit']:>8.2f} "
                  f"{r['strategic_action_space_width']:>6.4f}")
        if report["structurally_inactive"]:
            print()
            print(f"STRUCTURALLY INACTIVE: {report['structurally_inactive']} "
                  f"-- return to the candidate pool for review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
