# phase9_sensitivity.py
"""Load-shape, storage-capacity, siting and price-scenario sensitivity.

The phase runs in two steps on purpose. A **feasibility screen** comes first and
decides how much of the experiment is worth running at all; only what survives
it gets a full 96-period clearing. The screen is not a substitute for physical
verification -- the runs below still check voltage, lines, SOC and feasibility --
it only decides whether a *revenue* comparison could possibly say anything.

The screen's argument, which is what makes the small experiment defensible:

    For any battery, the margin it can capture is at most

        eta_dis * max_t price[t]  -  min_t price[t] / eta_ch

    which charges at the day's cheapest instant and discharges at its dearest,
    ignoring both the power limit and the energy limit. It is an optimistic
    bound on every real battery, and it is therefore an optimistic bound on
    every capacity and every location at once.

    On all four price curves that bound is below the required margin -- which
    `arbitrage_lp.required_margin` derives from the clearing's own objective as
    `cycle_cost * (1/eta_ch + eta_dis)`, not the lossless-round-trip
    `2 * cycle_cost` -- at **every one of the 33 buses**. So no capacity and no siting can open the window, and
    running the grid of idle clears would produce the same zero many times over.

What the phase therefore reports for capacity and siting is the part that IS
capable of varying: the physical constraints. Revenue is reported as
indistinguishable, with the reason, rather than as a table of noise.

Nothing was tuned. `cycle_cost`, efficiency, the price limits and the greedy
thresholds are exactly as Phases 5-8 left them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from arbitrage_lp import required_margin

SCENARIO = "typical_day_industry"
BASELINE_SCENARIO = "typical_day"

#: Efficiencies the margin bound uses, read from configuration at run time so
#: this file cannot disagree with the fleet it describes.
def _fleet_unit():
    from ess import build_ess_fleet
    from models import MarketConfig
    from arbitrage_lp import spec_from_storage

    config = MarketConfig.from_yaml()
    fleet = build_ess_fleet(96, config)
    if not fleet:
        raise RuntimeError("no independent storage fleet is configured")
    return spec_from_storage(fleet[0].storage), float(config.storage.cycle_cost)


def _fresh():
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults

    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


def _clear(day_id: int, scenario: str = SCENARIO, ess_buses=None,
           strategy: str = "fixed", capacity_mwh: Optional[float] = None,
           power_ratio: Optional[float] = None, T: int = 96) -> dict:
    """One full clearing under an explicitly configured fleet.

    Every knob is set here and reset by the caller's `_fresh()`, so a run cannot
    inherit the previous configuration's fleet through the config cache.
    """
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import set_default
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies import adaptive_bidding

    from accept_typical_day_industry import _physics

    _fresh()
    if capacity_mwh is not None:
        set_default("independent_storage.capacity_mwh", float(capacity_mwh))
    if power_ratio is not None:
        set_default("independent_storage.power_ratio", float(power_ratio))

    config = MarketConfig.from_yaml()
    agents, wholesale = get_scenario(scenario, T=T, config=config,
                                     day_id=int(day_id), ess_buses=ess_buses)
    declared = adaptive_bidding(agents, config, strategy=strategy, T=T)
    names = [a.name for a in agents if a.participant_type == "storage"]
    profits, clean, result = ce.settle_day(agents, config, names, declared,
                                           wholesale, T)
    if not clean or result is None:
        return {"settled_clean": False,
                "reason": "the whole-day settlement degraded; no payoff reported"}

    loads = [a for a in agents if a.load_type != "storage"]
    total = np.sum([np.asarray(a.load_real) for a in loads], axis=0)
    physics = _physics(result, agents, config, T)
    return {
        "settled_clean": True,
        "fleet_profit_cny": float(sum(profits.values())),
        "per_unit_profit": {k: float(v) for k, v in profits.items()},
        "storage_buses": [int(a.bus) for a in agents
                          if a.participant_type == "storage"],
        "storage_power_mw": [float(a.storage.p_ch_max) for a in agents
                             if a.participant_type == "storage"],
        "storage_energy_mwh": [float(a.storage.e_max) for a in agents
                               if a.participant_type == "storage"],
        "peaks": {
            "coincident_peak_mw": float(total.max()),
            "daily_energy_mwh": float(total.sum() * 0.25),
            "peak_slot": int(np.argmax(total)),
            "load_factor": (float(total.mean() / total.max())
                            if total.max() > 0 else None),
        },
        "physics": physics,
        "lmp": np.asarray(result["lmp"], dtype=float),
        "wholesale": np.asarray(wholesale, dtype=float),
    }


def _nodal_margins(lmp: np.ndarray, unit, cycle_cost: float) -> dict:
    """The round-trip margin at every bus, and whether any of them clears.

    Reports the best bus rather than a ranking: with nothing trading, a ranking
    would be a ranking of floating-point residue.
    """
    eta_ch, eta_dis = float(unit.eta_ch), float(unit.eta_dis)
    need = required_margin(cycle_cost, eta_ch, eta_dis)
    rows = []
    for bus in range(lmp.shape[1]):
        node = lmp[:, bus]
        margin = eta_dis * node.max() - node.min() / eta_ch
        rows.append({"bus": int(bus), "lmp_min": float(node.min()),
                     "lmp_max": float(node.max()), "margin": float(margin),
                     "shortfall": float(margin - need)})
    best = max(rows, key=lambda r: r["margin"])
    return {
        "required_margin": float(need),
        "best_bus": best["bus"],
        "best_margin": best["margin"],
        "best_shortfall": best["shortfall"],
        "n_buses_clearing": int(sum(1 for r in rows if r["margin"] > need)),
        "per_bus": rows,
    }


# ---------------------------------------------------------------------------
# Step 1: the screen
# ---------------------------------------------------------------------------

def feasibility_screen(T: int = 96) -> dict:
    """Is any revenue comparison possible at all, on any curve or bus?"""
    from dianyicha_price import SHEETS, read_curves

    unit, cycle_cost = _fleet_unit()
    need = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)
    curves = read_curves()
    out = {"cycle_cost_per_leg": cycle_cost, "required_margin": need,
           "unit": {"p_max_mw": unit.p_ch_max, "e_max_mwh": unit.e_max,
                    "eta_ch": unit.eta_ch, "eta_dis": unit.eta_dis},
           "curves": {}}

    for index, sheet in enumerate(SHEETS):
        run = _clear(index)
        if not run.get("settled_clean"):
            out["curves"][sheet] = {"clean": False, "reason": run["reason"]}
            continue
        price = run["wholesale"]
        # The optimistic bound: the day's extremes, ignoring every limit.
        raw = unit.eta_dis * price.max() - price.min() / unit.eta_ch
        nodal = _nodal_margins(run["lmp"], unit, cycle_cost)
        out["curves"][sheet] = {
            "clean": True,
            "external": {"min": float(price.min()), "max": float(price.max()),
                         "available_spread": float(price.max() - price.min()),
                         "unconstrained_margin": float(raw),
                         "shortfall": float(raw - need),
                         "clears": bool(raw > need)},
            "nodal": nodal,
            "nodal_vs_external": {
                "nodal_min": float(run["lmp"].min()),
                "nodal_max": float(run["lmp"].max()),
                "premium_at_deepest_bus": float(
                    nodal["per_bus"][nodal["best_bus"]]["lmp_max"]
                    - price.max()),
            },
        }
    out["conclusion"] = {
        "any_curve_clears": any(c.get("external", {}).get("clears")
                                for c in out["curves"].values()),
        "any_bus_clears": any(c.get("nodal", {}).get("n_buses_clearing", 0) > 0
                              for c in out["curves"].values()),
    }
    return out


def capacity_screen(T: int = 96) -> dict:
    """Does any battery size open a window the baseline one cannot?

    Answered on the day's extremes, which bound every battery, so the answer
    does not depend on the solver or on a particular clearing.
    """
    import dataclasses
    from arbitrage_lp import arbitrage_lp

    unit, cycle_cost = _fleet_unit()
    need = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)
    run = _clear(0)
    lmp = run["lmp"]
    out = {"required_margin": need, "configs": []}
    for p_mw, e_mwh in ((0.2, 1.6), (0.4, 0.8), (0.4, 1.6), (0.4, 3.2),
                        (0.8, 1.6), (2.0, 8.0), (4.0, 32.0), (10.0, 100.0)):
        u = dataclasses.replace(unit, p_ch_max=p_mw, p_dis_max=p_mw, e_max=e_mwh)
        best = None
        for bus in range(lmp.shape[1]):
            node = lmp[:, bus]
            raw = u.eta_dis * node.max() - node.min() / u.eta_ch
            if best is None or raw > best[1]:
                best = (bus, raw)
        margin = max(arbitrage_lp(lmp[:, b], u, cycle_cost).margin_cny
                     for b in range(lmp.shape[1]))
        out["configs"].append({
            "p_ch_max_mw": p_mw, "e_max_mwh": e_mwh,
            "best_unconstrained_margin": float(best[1]),
            "best_bus": int(best[0]),
            "clears": bool(best[1] > need),
            "best_lp_margin_over_all_buses": float(margin),
        })
    out["any_config_clears"] = any(c["clears"] for c in out["configs"])
    return out


# ---------------------------------------------------------------------------
# Step 2: what is worth running
# ---------------------------------------------------------------------------

def load_shape_comparison(T: int = 96) -> dict:
    """`typical_day` against `typical_day_industry`, same network and scale."""
    unit, cycle_cost = _fleet_unit()
    out = {}
    for name in (BASELINE_SCENARIO, SCENARIO):
        run = _clear(0, scenario=name)
        if not run.get("settled_clean"):
            out[name] = {"clean": False, "reason": run["reason"]}
            continue
        nodal = _nodal_margins(run["lmp"], unit, cycle_cost)
        out[name] = {
            "coincident_peak_mw": run["peaks"]["coincident_peak_mw"],
            "daily_energy_mwh": run["peaks"]["daily_energy_mwh"],
            "load_factor": run["peaks"]["load_factor"],
            "peak_slot": run["peaks"]["peak_slot"],
            "voltage": run["physics"]["bus_voltage"],
            "line_utilization": run["physics"]["line_utilization"],
            "energy": run["physics"]["energy"],
            "storage": run["physics"]["storage"],
            "fleet_profit_cny": run["fleet_profit_cny"],
            "nodal_best_margin": nodal["best_margin"],
            "nodal_best_bus": nodal["best_bus"],
            "nodal_buses_clearing": nodal["n_buses_clearing"],
            # The whole per-bus margin table, not just the best bus. On the
            # native day the configured fleet's own buses sit within a few
            # CNY/MWh of the requirement, and the report quotes them; keeping
            # the rows here is what makes that table a reading of an artefact
            # rather than a figure transcribed once and never re-derivable.
            "required_margin": nodal["required_margin"],
            "nodal_per_bus": nodal["per_bus"],
        }
    return out


def capacity_scan(T: int = 96) -> dict:
    """Vary power at fixed energy, then energy at fixed power."""
    cases = [
        ("baseline", 1.6, 0.25),
        ("power /2", 1.6, 0.125),
        ("power x2", 1.6, 0.5),
        ("energy /2", 0.8, 0.5),
        ("energy x2", 3.2, 0.125),
        ("joint x2", 3.2, 0.5),
    ]
    out = {}
    for label, e_max, ratio in cases:
        run = _clear(0, capacity_mwh=e_max, power_ratio=ratio)
        if not run.get("settled_clean"):
            out[label] = {"clean": False, "reason": run["reason"]}
            continue
        out[label] = {
            "energy_mwh_each": run["storage_energy_mwh"][0],
            "power_mw_each": run["storage_power_mw"][0],
            "fleet_profit_cny": run["fleet_profit_cny"],
            "charge_mwh": sum(s["charge_mwh"]
                              for s in run["physics"]["storage"].values()),
            "discharge_mwh": sum(s["discharge_mwh"]
                                 for s in run["physics"]["storage"].values()),
            "soc_all_verified": all(s["all_transitions_verified"]
                                    for s in run["physics"]["storage"].values()),
            "soc_within_bounds": all(s["within_bounds"]
                                     for s in run["physics"]["storage"].values()),
            "voltage_min": run["physics"]["bus_voltage"]["min"],
            "voltage_max": run["physics"]["bus_voltage"]["max"],
            "voltage_violations": (run["physics"]["bus_voltage"]["periods_below_min"]
                                   + run["physics"]["bus_voltage"]["periods_above_max"]),
            "line_max_utilization": run["physics"]["line_utilization"]["max"],
            "line_over_limit": run["physics"]["line_utilization"]["lines_over_1"],
        }
    return out


def siting_scan(T: int = 96) -> dict:
    """Move the fleet, keeping its count and its ratings.

    The placements include the configured one, the two the project previously
    used, and sets chosen to put units on the deepest-feeder buses -- where the
    nodal price is furthest above the wholesale curve -- so the scan covers the
    most favourable locations the network has rather than an arbitrary sample.
    """
    cases = [
        ("configured 9/14/17/27", None),
        ("previous 6/20/24/31", [6, 20, 24, 31]),
        ("previous 10/15/19/26", [10, 15, 19, 26]),
        ("deep feeder 16/17/18/19", [16, 17, 18, 19]),
        ("spur ends 21/24/30/31", [21, 24, 30, 31]),
    ]
    unit, cycle_cost = _fleet_unit()
    out = {}
    for label, buses in cases:
        run = _clear(0, ess_buses=buses)
        if not run.get("settled_clean"):
            out[label] = {"clean": False, "reason": run["reason"]}
            continue
        nodal = _nodal_margins(run["lmp"], unit, cycle_cost)
        own = [nodal["per_bus"][b] for b in run["storage_buses"]]
        out[label] = {
            "buses": run["storage_buses"],
            "fleet_profit_cny": run["fleet_profit_cny"],
            "best_margin_at_own_bus": max(r["margin"] for r in own),
            "own_buses_clearing": sum(1 for r in own
                                      if r["margin"] > nodal["required_margin"]),
            "network_best_margin": nodal["best_margin"],
            "network_best_bus": nodal["best_bus"],
            "voltage_min": run["physics"]["bus_voltage"]["min"],
            "voltage_max": run["physics"]["bus_voltage"]["max"],
            "voltage_violations": (run["physics"]["bus_voltage"]["periods_below_min"]
                                   + run["physics"]["bus_voltage"]["periods_above_max"]),
            "line_max_utilization": run["physics"]["line_utilization"]["max"],
            "line_over_limit": run["physics"]["line_utilization"]["lines_over_1"],
            "soc_all_verified": all(s["all_transitions_verified"]
                                    for s in run["physics"]["storage"].values()),
        }
    return out


def price_scenarios(T: int = 96) -> dict:
    """The four curves, run at the configured fleet."""
    from dianyicha_price import ROLES, SHEETS, read_curves

    unit, cycle_cost = _fleet_unit()
    need = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)
    curves = read_curves()
    out = {}
    for index, sheet in enumerate(SHEETS):
        run = _clear(index)
        if not run.get("settled_clean"):
            out[sheet] = {"clean": False, "reason": run["reason"]}
            continue
        price = run["wholesale"]
        nodal = _nodal_margins(run["lmp"], unit, cycle_cost)
        out[sheet] = {
            "role": ROLES.get(sheet, "unspecified"),
            "source_kind": curves[sheet].source_kind,
            "spatial_statistic": curves[sheet].spatial_statistic,
            "temporal_statistic": curves[sheet].temporal_statistic,
            "window_days": curves[sheet].window_days,
            "is_single_day": curves[sheet].is_single_day,
            "externally_confirmed": curves[sheet].externally_confirmed,
            # Deprecated compatibility copy; not a single-day statement.
            "is_measured": curves[sheet].is_measured,
            "workbook_sha256": curves[sheet].provenance.get("workbook_sha256"),
            "header_label": curves[sheet].provenance.get("header_label"),
            "mean": float(price.mean()),
            "min": float(price.min()),
            "max": float(price.max()),
            "available_spread": float(price.max() - price.min()),
            "unconstrained_margin": float(
                unit.eta_dis * price.max() - price.min() / unit.eta_ch),
            "required_margin": float(need),
            "clears": bool(unit.eta_dis * price.max()
                           - price.min() / unit.eta_ch > need),
            "nodal_best_margin": nodal["best_margin"],
            "nodal_best_bus": nodal["best_bus"],
            "nodal_buses_clearing": nodal["n_buses_clearing"],
            "fleet_profit_cny": run["fleet_profit_cny"],
            "curtailment_mwh": run["physics"]["energy"]["curtailment_mwh"],
            "voltage_min": run["physics"]["bus_voltage"]["min"],
            "voltage_max": run["physics"]["bus_voltage"]["max"],
            "line_max_utilization": run["physics"]["line_utilization"]["max"],
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(T: int = 96, run_full: bool = True) -> dict:
    """Screen first, then run only what the screen leaves open."""
    screen = feasibility_screen(T)
    report = {
        "scenario": SCENARIO, "T": int(T),
        "cycle_cost_per_leg": screen["cycle_cost_per_leg"],
        "required_margin": screen["required_margin"],
        "screen": screen,
    }
    if not run_full:
        return report

    report["capacity_screen"] = capacity_screen(T)
    report["load_shape"] = load_shape_comparison(T)
    report["capacity"] = capacity_scan(T)
    report["siting"] = siting_scan(T)
    report["price_scenarios"] = price_scenarios(T)
    report["scope_note"] = (
        "Capacity and siting were run to establish physical feasibility and to "
        "confirm the screen, not to rank configurations by revenue. The screen "
        "shows no bus on any curve clears the required margin even with the limits "
        "removed, so no capacity or placement can open the window, and a "
        "revenue ranking would be a ranking of floating-point residue.")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-9 sensitivity analysis.")
    p.add_argument("--json", default="outputs/phase9_sensitivity.json")
    p.add_argument("--screen-only", action="store_true")
    p.add_argument("--T", type=int, default=96)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_all(T=args.T, run_full=not args.screen_only)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"written to {args.json}")

    s = report["screen"]
    print(f"\ncycle_cost {s['cycle_cost_per_leg']:.0f}/leg -> a cycle needs "
          f"margin > {s['required_margin']:.0f} CNY/MWh")
    print(f"{'curve':<10}{'exo margin':>12}{'best nodal':>12}{'at bus':>8}"
          f"{'buses clearing':>16}")
    for sheet, c in s["curves"].items():
        if not c.get("clean"):
            print(f"{sheet:<10}  NOT CLEAN"); continue
        print(f"{sheet:<10}{c['external']['unconstrained_margin']:>12.2f}"
              f"{c['nodal']['best_margin']:>12.2f}{c['nodal']['best_bus']:>8}"
              f"{c['nodal']['n_buses_clearing']:>16}")
    print(f"\nany curve clears: {s['conclusion']['any_curve_clears']}   "
          f"any bus clears: {s['conclusion']['any_bus_clears']}")
    if not args.screen_only:
        print(f"any capacity clears: {report['capacity_screen']['any_config_clears']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
