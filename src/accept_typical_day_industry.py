# accept_typical_day_industry.py
"""One end-to-end acceptance run of the industry-shape research day.

Runs a single representative price scenario through the whole chain -- industry
load shapes, IEEE 33-bus network, the independent storage fleet, a full-day SOCP
clearing, and the shared settlement -- and reports what it measured.

Two things make this a check rather than a demonstration.

First, the day is run **twice through different paths** and the two are
compared. Path A settles directly from the `fixed` declarations; path B runs
the rolling episode the evaluation CLI uses and settles through
`eval_agents.settle_capture`. Both must produce the same per-unit payoff. That
is the property Phase 5 exists to establish, and it is checked here on a real
day rather than on a fixture.

Second, the fleet is diagnosed rather than merely reported. On this price
scenario every unit is idle, and the run explains why: the round-trip margin at
each unit's own bus is compared against what the clearing charges it to cycle.
An idle fleet that is not explained is indistinguishable from a broken one.

Nothing here tunes anything. Prices, `cycle_cost`, the network and the
thresholds are read from configuration and reported as found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCENARIO = "typical_day_industry"
STORAGE_NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(a) -> bool:
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


def _load_industry_shapes(path: str = "data/processed/industry_typical_profiles.csv"):
    from industry_load import N_SLOTS, load_shapes_csv

    shapes = load_shapes_csv(path)
    check = {"path": path, "n_industries": len(shapes), "n_slots": N_SLOTS}
    bad = []
    for name in shapes.industries():
        arr = shapes.shape_of(name)
        if arr.shape != (N_SLOTS,) or not _finite(arr):
            bad.append(name)
    check["industries_failing_96_finite"] = bad
    check["reliable"] = sorted(n for n in shapes.industries()
                               if shapes.shapes[n].reliable)
    check["limited"] = sorted(n for n in shapes.industries()
                              if not shapes.shapes[n].reliable)
    return shapes, check


def _load_price_curves(workbook: Optional[str] = None):
    from dianyicha_price import N_SLOTS, describe, read_curves

    curves = read_curves(workbook)
    check = {"curves": {}, "n_slots": N_SLOTS}
    for name, c in curves.items():
        arr = c.array()
        ok = arr.shape == (N_SLOTS,) and _finite(arr)
        check["curves"][name] = {
            "role": c.role,
            "source_kind": c.source_kind,
            "spatial_statistic": c.spatial_statistic,
            "temporal_statistic": c.temporal_statistic,
            "window_days": c.window_days,
            "is_single_day": c.is_single_day,
            "externally_confirmed": c.externally_confirmed,
            # Deprecated compatibility copy. It says the source is real measured
            # data -- true for all four -- NOT that the curve is one day; see
            # dianyicha_price.DEPRECATED_FIELDS.
            "is_measured": c.is_measured,
            "header_label": c.provenance.get("header_label"),
            "workbook_sha256": c.provenance.get("workbook_sha256"),
            "interval_convention": c.provenance.get("interval_convention"),
            "units": c.provenance.get("units"),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "n_slots": int(arr.size),
            "valid": bool(ok),
        }
    check["describe"] = describe(curves)
    return curves, check


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _physics(result: dict, agents, config, T: int) -> dict:
    """What the cleared day actually satisfies.

    Reported as measured. A constraint violation is a finding, not something to
    explain away, so nothing here is clipped, filtered or tolerance-hidden.
    """
    import money

    out: Dict[str, object] = {}
    v = np.asarray(result["bus_voltage"], dtype=float)
    util = np.asarray(result["line_utilization"], dtype=float)
    vmin, vmax = float(config.network.v_min_pu), float(config.network.v_max_pu)

    out["bus_voltage"] = {
        "min": float(v.min()), "max": float(v.max()),
        "band": [vmin, vmax],
        "periods_below_min": int(np.sum(np.any(v < vmin - 1e-6, axis=1))),
        "periods_above_max": int(np.sum(np.any(v > vmax + 1e-6, axis=1))),
        "worst_bus": int(result["bus_indices"][int(np.argmin(v.min(axis=0)))]),
    }
    out["line_utilization"] = {
        "max": float(util.max()),
        "lines_over_1": int(np.sum(util > 1.0 + 1e-6)),
        "periods_over_1": int(np.sum(np.any(util > 1.0 + 1e-6, axis=1))),
        "worst_line": int(result["line_indices"][int(np.argmax(util.max(axis=0)))]),
    }

    grid_in = np.asarray(result["grid_import_mw"], dtype=float)
    grid_out = np.asarray(result["grid_export_mw"], dtype=float)
    out["grid"] = {
        "import_mwh": float(grid_in.sum() * money.DT_HOURS),
        "export_mwh": float(grid_out.sum() * money.DT_HOURS),
        "periods_importing": int(np.sum(grid_in > 1e-9)),
        "periods_exporting": int(np.sum(grid_out > 1e-9)),
        "simultaneous_import_export": int(np.sum((grid_in > 1e-9) & (grid_out > 1e-9))),
    }

    loads = [a for a in agents if a.load_type != "storage"]
    unserved = sum(float(np.sum(result["schedules"][a.name]["unserved"]))
                   for a in loads)
    served = sum(float(np.sum(result["schedules"][a.name]["served"]))
                 for a in loads)
    # The day-ahead clearing dispatches against `load_forecast`, not
    # `load_real` -- so `served` is comparable to the forecast and not to the
    # realized demand. Reporting unserved against the realized load would
    # manufacture a shortfall that no constraint produced: the gap between the
    # two is the forecast error, which the real-time settlement exists to
    # handle and which this day-ahead acceptance does not settle.
    forecast = sum(float(np.sum(a.load_forecast)) for a in loads)
    realized = sum(float(np.sum(a.load_real)) for a in loads)
    out["energy"] = {
        "da_basis": "load_forecast",
        "forecast_mwh": float(forecast * money.DT_HOURS),
        "realized_mwh": float(realized * money.DT_HOURS),
        "forecast_error_mwh": float((realized - forecast) * money.DT_HOURS),
        "forecast_error_pct": (float(100.0 * (realized - forecast) / forecast)
                               if forecast else None),
        "served_mwh": float(served * money.DT_HOURS),
        "unserved_mwh": float(unserved * money.DT_HOURS),
        "unserved_fraction_of_forecast": (float(unserved / forecast)
                                          if forecast else 0.0),
        "served_covers_forecast": bool(abs(served + unserved - forecast) < 1e-6),
        "curtailment_mwh": float(result.get("total_curtailment", 0.0)),
        "re_consumption_rate_pct": float(result.get("re_consumption_rate", 0.0)),
    }

    out["prices"] = {
        "lmp_fallbacks": int(result.get("lmp_fallbacks", -1)),
        "nodal_min": float(np.asarray(result["lmp"]).min()),
        "nodal_max": float(np.asarray(result["lmp"]).max()),
        "clearing_objective": float(result.get("clearing_objective", float("nan"))),
        "carbon_tco2": float(result.get("carbon_emissions", float("nan"))),
    }

    # Which solver produced the day, and whether its prices are duals. Reported
    # as two separate answers: an OPTIMAL model can still yield no balance dual,
    # and it is the dual that an LMP is.
    solver = result.get("solver") or {}
    out["solver"] = {
        "present": bool(solver),
        "engine": solver.get("engine"),
        "primary_name": (solver.get("primary") or {}).get("name"),
        "primary_status_code": (solver.get("primary") or {}).get("status_code"),
        "primary_status_name": (solver.get("primary") or {}).get("status_name"),
        "primary_termination": (solver.get("primary") or {}).get("termination"),
        "used_fallback": solver.get("used_fallback"),
        "fallback": solver.get("fallback"),
        "final_name": (solver.get("final") or {}).get("name"),
        "final_status_name": (solver.get("final") or {}).get("status_name"),
        "final_is_primary": bool((solver.get("final") or {}).get("name")
                                 == (solver.get("primary") or {}).get("name")),
        "duals_available": solver.get("duals_available"),
        "fallback_key_present": "fell_back" in result,
    }

    # The state-of-charge recursion, verified on the boundary trajectory.
    #
    # `soc_boundary` holds the model's own T+1 states: [0] is the initial state
    # and [t+1] the state after period t. Every one of the T transitions is
    # therefore checkable. The per-period `soc` readout is not usable for this:
    # its last entry holds the terminal state rather than the T-1th, so a
    # residual computed on it silently skips the final transition.
    TOL = 1e-6
    soc_checks = {}
    for a in agents:
        if a.storage is None:
            continue
        s = result["schedules"][a.name]
        boundary = np.asarray(s["soc_boundary"], dtype=float)
        ch = np.asarray(s["p_ch"], dtype=float)
        dis = np.asarray(s["p_dis"], dtype=float)
        flow = (a.storage.eta_ch * ch - dis / a.storage.eta_dis) \
            * money.DT_HOURS / a.storage.e_max
        resid = (boundary[1:] - boundary[:-1]) - flow
        worst = int(np.argmax(np.abs(resid))) if resid.size else None
        target = s.get("soc_terminal_target")
        actual = s.get("soc_terminal_actual")
        soc_checks[a.name] = {
            "n_boundary_states": int(boundary.size),
            "transitions_expected": int(T),
            "transitions_checked": int(resid.size),
            "all_transitions_checked": bool(resid.size == T),
            "max_recursion_residual": float(np.max(np.abs(resid))) if resid.size else None,
            "worst_transition_slot": worst,
            "worst_transition_residual": float(np.abs(resid).max()) if resid.size else None,
            "tolerance": TOL,
            "all_transitions_verified": bool(resid.size == T
                                             and np.max(np.abs(resid)) <= TOL),
            "soc0_boundary": float(boundary[0]),
            "soc0_configured": float(a.storage.soc0),
            "soc_terminal_target": None if target is None else float(target),
            "soc_terminal_actual": None if actual is None else float(actual),
            "terminal_actual_is_the_boundary_end": bool(
                actual is not None
                and abs(float(actual) - float(boundary[-1])) < 1e-12),
            "within_bounds": bool(np.all(boundary >= a.storage.soc_min - 1e-9)
                                  and np.all(boundary <= a.storage.soc_max + 1e-9)),
            "charge_mwh": float(np.sum(ch) * money.DT_HOURS),
            "discharge_mwh": float(np.sum(dis) * money.DT_HOURS),
            "periods_charging": int(np.sum(ch > 1e-9)),
            "periods_discharging": int(np.sum(dis > 1e-9)),
        }
    out["storage"] = soc_checks
    return out


def _arbitrage_diagnosis(result: dict, agents, config) -> dict:
    """Why the fleet did or did not cycle, in the clearing's own arithmetic.

    A unit charges at its declared bid and discharges at its declared offer,
    with `cycle_cost` charged on both legs, so the day must cover
    `eta_dis * peak - valley / eta_ch > required_margin` at the unit's own bus,
    where the requirement is derived from the clearing's objective (see
    `arbitrage_lp.required_margin`) rather than approximated.
    Reporting the shortfall turns "the fleet is idle" from an observation into
    a measured result.
    """
    from arbitrage_lp import required_margin

    lmp = np.asarray(result["lmp"], dtype=float)
    cc = float(config.storage.cycle_cost)
    rows = []
    for a in agents:
        if a.storage is None:
            continue
        node = lmp[:, int(a.bus)]
        eta = float(a.storage.eta_dis)
        eta_ch = float(a.storage.eta_ch)
        required = required_margin(cc, eta_ch, eta)
        margin = float(eta * node.max() - node.min() / eta_ch)
        rows.append({
            "unit": a.name, "bus": int(a.bus),
            "lmp_min": float(node.min()), "lmp_max": float(node.max()),
            "trough_slot": int(np.argmin(node)), "peak_slot": int(np.argmax(node)),
            "round_trip_margin": margin,
            "required": required,
            "shortfall": margin - required,
        })
    return {
        "cycle_cost_per_leg": cc,
        "required_margin": required,
        "units": rows,
        "all_below_threshold": all(r["shortfall"] < 0 for r in rows) if rows else None,
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_acceptance(workbook: Optional[str] = None,
                   shapes_path: str = "data/processed/industry_typical_profiles.csv",
                   T: int = 96, repeats: int = 2) -> dict:
    """Run one acceptance and return everything it measured."""
    import counterfactual_eval as ce
    import eval_agents as ea
    import money
    import wholesale_price as wp
    from config_loader import get_scenario_cfg, reload_defaults
    from grid import native_nominal_demand_mw
    from models import MarketConfig
    from rl_env import BiddingEnv
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    report: Dict[str, object] = {"scenario": SCENARIO, "T": int(T)}

    reload_defaults()
    wp.reset_cache()

    shapes, shape_check = _load_industry_shapes(shapes_path)
    report["industry_shapes"] = shape_check

    curves, price_check = _load_price_curves(workbook)
    report["price_curves"] = price_check

    scenario_cfg = get_scenario_cfg(SCENARIO)
    curve_index = int(scenario_cfg["wholesale_curve_index"])
    from dianyicha_price import SHEETS
    report["declared_curve"] = {
        "wholesale_curve_index": curve_index,
        "resolved_sheet": SHEETS[curve_index],
        "declared_source": scenario_cfg.get("wholesale_source"),
        "declared_by": "config/scenarios.yaml:typical_day_industry.wholesale_curve_index",
        "global_day_id_used": False,
    }

    config = MarketConfig.from_yaml()
    agents, wholesale = get_scenario(SCENARIO, T=T, config=config)

    # -- scenario and load -------------------------------------------------
    loads = [a for a in agents if a.load_type != "storage"]
    storage = [a for a in agents if a.participant_type == "storage"]
    total = np.sum([np.asarray(a.load_real) for a in loads], axis=0)
    map_cfg = scenario_cfg.get("industry_map", {})
    report["scenario"] = {
        "name": SCENARIO,
        "load_scale_mode": scenario_cfg.get("load_scale_mode"),
        "load_shape": scenario_cfg.get("load_shape"),
        "n_load_agents": len(loads),
        "n_storage_agents": len(storage),
        "storage_names": [a.name for a in storage],
        "storage_buses": [int(a.bus) for a in storage],
        "storage_power_mw": [float(a.storage.p_ch_max) for a in storage],
        "storage_energy_mwh": [float(a.storage.e_max) for a in storage],
        "pv_mw": float(sum(float(a.pv_capacity or 0.0) for a in agents)),
        "wind_mw": float(sum(float(a.wind_capacity or 0.0) for a in agents)),
        "mapping_is_simulation": bool(map_cfg.get("simulation_mapping")),
        "mapping_excluded": list(map_cfg.get("excluded_industries", [])),
        "mapping_industries": sorted(set(map_cfg.get("buses", {}).values())),
    }
    target = native_nominal_demand_mw(
        __import__("grid").build_base_network(config))
    realized_peak = float(total.max())
    report["load_check"] = {
        "coincident_peak_mw": realized_peak,
        "network_nominal_demand_mw": float(target),
        "absolute_error_mw": abs(realized_peak - float(target)),
        "daily_energy_mwh": float(total.sum() * money.DT_HOURS),
        "all_finite": _finite(total),
        "all_non_negative": bool(np.all(total >= 0.0)),
        "uses_pv_industry_as_load": "分布式光伏" in set(map_cfg.get("buses", {}).values()),
    }
    report["price_check"] = {
        "resolved_sheet": SHEETS[curve_index],
        "wholesale_matches_declared_curve": bool(np.allclose(
            wholesale, curves[SHEETS[curve_index]].array())),
        "min": float(wholesale.min()), "max": float(wholesale.max()),
        "mean": float(wholesale.mean()),
        "matches_workbook_min": abs(float(wholesale.min())
                                    - price_check["curves"][SHEETS[curve_index]]["min"]) < 1e-9,
        "matches_workbook_max": abs(float(wholesale.max())
                                    - price_check["curves"][SHEETS[curve_index]]["max"]) < 1e-9,
        "matches_workbook_mean": abs(float(wholesale.mean())
                                     - price_check["curves"][SHEETS[curve_index]]["mean"]) < 1e-9,
    }

    # -- path A: settle the fixed declarations directly --------------------
    declared = FixedStrategy().formulate(agents, config, None, T)
    names = [a.name for a in storage]
    profits_a, clean_a, result = ce.settle_day(agents, config, names,
                                               declared, wholesale, T)
    report["settlement_path_a"] = {
        "entry": "counterfactual_eval.settle_day",
        "clean": bool(clean_a),
        "per_unit_profit": {k: float(v) for k, v in profits_a.items()},
        "fleet_profit": float(sum(profits_a.values())),
    }

    # -- path B: the rolling episode the CLI uses, then settle -------------
    reload_defaults()
    wp.reset_cache()
    config_b = MarketConfig.from_yaml()
    agents_b, wholesale_b = get_scenario(SCENARIO, T=T, config=config_b)
    np.random.seed(1007)
    env = BiddingEnv(agents_b, config_b)
    episode = ea.run_episode(env, policy=None, capture=True)
    profits_b, clean_b = ea.settle_capture(agents_b, config_b, names, episode)
    report["settlement_path_b"] = {
        "entry": "eval_agents.run_episode + settle_capture",
        "capture_usable": bool(episode.get("usable")),
        "clean": bool(clean_b),
        "per_unit_profit": {k: float(v) for k, v in profits_b.items()},
        "rolling_reward": {k: float(episode["profits"].get(k, 0.0)) for k in names},
    }
    report["paths_agree"] = {
        "prices_identical": bool(np.allclose(wholesale, wholesale_b)),
        "per_unit_max_abs_diff": float(max(
            abs(profits_a[k] - profits_b[k]) for k in names)),
        "agree": bool(all(abs(profits_a[k] - profits_b[k]) < 1e-6 for k in names)),
    }

    # -- physics, breakdown, arbitrage ------------------------------------
    report["physics"] = _physics(result, agents, config, T)

    import participant_payoff
    breakdown = {}
    for a in storage:
        sched = result["schedules"][a.name]
        bd = participant_payoff.participant_payoff(
            sched, result["lmp"][:, int(a.bus)], a, config)
        parts = (bd.revenue - bd.purchase_cost - bd.generation_cost
                 - bd.degradation_cost - bd.other_cost - bd.penalty)
        breakdown[a.name] = {
            "revenue": float(bd.revenue),
            "purchase_cost": float(bd.purchase_cost),
            "generation_cost": float(bd.generation_cost),
            "degradation_cost": float(bd.degradation_cost),
            "other_cost": float(bd.other_cost),
            "penalty": float(bd.penalty),
            "total": float(bd.total),
            "reconciles": bool(abs(bd.total - parts) < 1e-6),
            "matches_settlement": bool(abs(bd.total - profits_a[a.name]) < 1e-6),
        }
    report["breakdown"] = breakdown
    # The fleet total is checked against a sum of independently computed unit
    # payoffs, not against itself: the settlement returns per-unit figures, so
    # summing them twice would prove nothing.
    report["fleet_reconciliation"] = {
        "sum_of_settled_unit_profits": float(sum(profits_a.values())),
        "sum_of_independent_breakdowns": float(
            sum(b["total"] for b in breakdown.values())),
        "agree": bool(abs(sum(profits_a.values())
                           - sum(b["total"] for b in breakdown.values())) < 1e-6),
        "every_unit_reconciles": bool(all(b["reconciles"]
                                          for b in breakdown.values())),
        "every_unit_matches_settlement": bool(all(b["matches_settlement"]
                                                  for b in breakdown.values())),
    }
    report["arbitrage"] = _arbitrage_diagnosis(result, agents, config)
    declared = price_check["curves"][SHEETS[curve_index]]
    report["provenance"] = {
        "workbook_sha256": declared["workbook_sha256"],
        "shapes_file_sha256": _sha256(Path(shapes_path)),
        "curve_sheet": SHEETS[curve_index],
        "curve_role": declared["role"],
        "curve_source_kind": declared["source_kind"],
        "curve_spatial_statistic": declared["spatial_statistic"],
        "curve_temporal_statistic": declared["temporal_statistic"],
        "curve_window_days": declared["window_days"],
        "curve_is_single_day": declared["is_single_day"],
        "curve_externally_confirmed": declared["externally_confirmed"],
        # Deprecated compatibility copy; not a single-day statement.
        "curve_is_measured": declared["is_measured"],
        "dt_hours": money.DT_HOURS,
        "units": "CNY/MWh",
        "profit_definition": (
            "modelled storage day-ahead energy profit: settlement revenue less "
            "purchase, degradation and penalty; no out-of-model charges"),
    }

    # -- repeatability -----------------------------------------------------
    repeat = _repeat_run(repeats, T)
    report["repeatability"] = repeat
    return report


def _repeat_run(repeats: int, T: int) -> dict:
    """Run the frozen configuration again and compare the headline numbers."""
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    runs = []
    for _ in range(max(2, repeats)):
        reload_defaults()
        wp.reset_cache()
        ce.reset_baseline_cache()
        config = MarketConfig.from_yaml()
        agents, wholesale = get_scenario(SCENARIO, T=T, config=config)
        declared = FixedStrategy().formulate(agents, config, None, T)
        names = [a.name for a in agents if a.participant_type == "storage"]
        profits, clean, _ = ce.settle_day(agents, config, names, declared,
                                          wholesale, T)
        loads = [a for a in agents if a.load_type != "storage"]
        total = np.sum([np.asarray(a.load_real) for a in loads], axis=0)
        runs.append({"wholesale": np.asarray(wholesale, dtype=float),
                     "peak": float(total.max()),
                     "profits": dict(profits), "clean": bool(clean)})

    base = runs[0]
    return {
        "n_runs": len(runs),
        "prices_identical": all(np.array_equal(base["wholesale"], r["wholesale"])
                                for r in runs[1:]),
        "peak_identical": all(abs(base["peak"] - r["peak"]) < 1e-9
                              for r in runs[1:]),
        "profits_identical": all(
            all(abs(base["profits"][k] - r["profits"][k]) < 1e-6
                for k in base["profits"]) for r in runs[1:]),
        "all_clean": all(r["clean"] for r in runs),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end acceptance of the industry-shape research day.")
    p.add_argument("--workbook", default=None)
    p.add_argument("--shapes", default="data/processed/industry_typical_profiles.csv")
    p.add_argument("--json", default=None, help="write the full report here")
    p.add_argument("--repeats", type=int, default=2)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_acceptance(workbook=args.workbook, shapes_path=args.shapes,
                            repeats=args.repeats)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print(f"report written to {args.json}")

    sp = report["settlement_path_a"]
    ph = report["physics"]
    print(f"scenario            : {report['scenario']['name']}")
    print(f"curve               : {report['declared_curve']['resolved_sheet']}"
          f" (index {report['declared_curve']['wholesale_curve_index']}, "
          f"temporal={report['provenance']['curve_temporal_statistic']}"
          f"/days={report['provenance']['curve_window_days']}, "
          f"spatial={report['provenance']['curve_spatial_statistic']}, "
          f"single_day={report['provenance']['curve_is_single_day']})")
    print(f"coincident peak     : {report['load_check']['coincident_peak_mw']:.6f} MW"
          f"  (target {report['load_check']['network_nominal_demand_mw']:.6f})")
    print(f"settled (clean={sp['clean']}) : "
          + "  ".join(f"{k}={v:+.4f}" for k, v in sp["per_unit_profit"].items()))
    print(f"fleet profit        : {sp['fleet_profit']:+.4f}")
    print(f"paths agree         : {report['paths_agree']['agree']}"
          f" (max diff {report['paths_agree']['per_unit_max_abs_diff']:.2e})")
    print(f"voltage             : {ph['bus_voltage']['min']:.4f}"
          f" .. {ph['bus_voltage']['max']:.4f}"
          f"  band {ph['bus_voltage']['band']}"
          f"  violations {ph['bus_voltage']['periods_below_min']} low /"
          f" {ph['bus_voltage']['periods_above_max']} high")
    print(f"line utilisation    : max {ph['line_utilization']['max']:.4f}"
          f"  over-limit periods {ph['line_utilization']['periods_over_1']}")
    print(f"energy (DA basis)   : forecast {ph['energy']['forecast_mwh']:.2f} MWh"
          f"  realized {ph['energy']['realized_mwh']:.2f} MWh"
          f"  (forecast error {ph['energy']['forecast_error_pct']:+.2f}%)")
    print(f"                      unserved {ph['energy']['unserved_mwh']:.2e} MWh"
          f"  curtailment {ph['energy']['curtailment_mwh']:.4f} MWh"
          f"  RE rate {ph['energy']['re_consumption_rate_pct']:.2f}%")
    print(f"lmp fallbacks       : {ph['prices']['lmp_fallbacks']}")
    sv = ph["solver"]
    print(f"solver              : {sv['engine']} / primary "
          f"{sv['primary_status_name']} (code {sv['primary_status_code']}, "
          f"{sv['primary_termination']})  fallback={sv['used_fallback']}  "
          f"final={sv['final_name']}  duals={sv['duals_available']}")
    for unit, s in ph["storage"].items():
        print(f"SOC {unit:<6}        : {s['transitions_checked']}/"
              f"{s['transitions_expected']} transitions verified, max residual "
              f"{s['max_recursion_residual']:.2e} (slot "
              f"{s['worst_transition_slot']}), target "
              f"{s['soc_terminal_target']} actual {s['soc_terminal_actual']}")
    arb = report["arbitrage"]
    print(f"arbitrage           : required {arb['required_margin']:.0f} CNY/MWh, "
          f"all below threshold = {arb['all_below_threshold']}")
    print(f"repeatable          : {report['repeatability']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
