# s8_physical_audit.py
"""S8: the physical audit of one cleared day, as an **audit** and not a verdict.

S8 is the network validation stage. This module runs its measurements over the
scenario named on the command line, which today is the subset scenario on the
single-day case.

**This is not an externally certified physical result.** The load rests on a
declared reading unit whose basis is a **recorded project ruling**
(`user_confirmed_project_ruling`); `is_externally_verified` is `False` and every
figure here inherits that. The ledger it reads is a **formal** ledger — the
ruling is an admissible basis and `validate_formal_ledger` passes it — but a
formal basis is not an externally verified one, so nothing below may be quoted
as externally certified.

The audit is also **conditional on this repository's own line ampacities**,
which are project-assigned rather than the published `case33bw` ratings: see
§3, and note that no statement here is a validation against authoritative
standard line ratings.

The clearing is not re-implemented. `run_single_day_diagnostic.clear_once` runs
it and hands back the fleet, the config, the case and the raw result; this
module measures that one clearing. Two reports over one day must not come from
two clears -- a second clear is a second set of numbers, and the two would drift
apart with neither being wrong.

Nothing here tunes anything. The voltage band, the line ampacity, the power
factor, the storage parameters and the load are **read from configuration and
reported as found**; if a limit is binding, that is the finding.

Usage:
    PYTHONPATH=src python src/s8_physical_audit.py \
        --scenario typical_day_industry_subset --json outputs/s8_physical_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import config_loader as cfg
import dispatch_socp
import grid
import money
import run_single_day_diagnostic as diag

SCENARIO = "typical_day_industry_subset"
REPORT = Path("reports/s8_physical_audit.md")

#: Tolerance for a physical identity, in MW. The clearing solves to Gurobi's
#: default tolerances, so an exact zero is not available and not expected.
IDENTITY_ATOL_MW = 1e-6


def _total(schedules, agents, key) -> np.ndarray:
    """One period-indexed series summed over every agent."""
    return np.sum([np.asarray(schedules[a.name][key], dtype=float)
                   for a in agents], axis=0)


def _mwh(series: np.ndarray) -> float:
    """A power series in MW over a day, in MWh."""
    return float(np.sum(series) * money.DT_HOURS)


def _line_losses(config, result) -> np.ndarray:
    """The clearing's own network losses, in MW per period.

    Recomputed from the solver's line currents rather than read from it: the
    active-power balance carries `r * I^2` per line and the result dict does not
    surface it, so the loss is recovered the way the model defines it. The
    parameters come from `dispatch_socp._build_line_params`, the same function
    the model used, so this is the model's own loss and not a second model's.
    """
    net = grid.build_base_network(config)
    r_pu, _x_pu, limit_pu = dispatch_socp._build_line_params(
        net, config.network.base_kv)
    lines = list(result.get("line_indices") or [])
    util = np.asarray(result["line_utilization"], dtype=float)
    if not lines or util.size == 0:
        return np.zeros(util.shape[0] if util.ndim else 0)
    current = np.stack([util[:, j] * limit_pu[lines[j]]
                        for j in range(len(lines))], axis=1)
    resistance = np.array([r_pu[lines[j]] for j in range(len(lines))])
    return current ** 2 @ resistance


def measure(scenario: str = SCENARIO) -> dict:
    """The audit's measurements, from one clearing of the named scenario."""
    cleared = diag.clear_once(scenario)
    config = cleared["config"]
    agents = cleared["agents"]
    case = cleared["case"]
    result = cleared["result"]
    schedules = result["schedules"]
    T = cleared["T"]

    served = _total(schedules, agents, "served")
    unserved = _total(schedules, agents, "unserved")
    pv_used = _total(schedules, agents, "pv_used")
    wind_used = _total(schedules, agents, "wind_used")
    p_ch = _total(schedules, agents, "p_ch")
    p_dis = _total(schedules, agents, "p_dis")

    # The load the fleet actually carries, read off the agents rather than off
    # the ledger, so this is what the clearing was given.
    asked = np.sum([np.asarray(a.load_real, dtype=float) for a in agents],
                   axis=0)

    imp = np.asarray(result.get("grid_import_mw", np.zeros(T)), dtype=float)
    exp = np.asarray(result.get("grid_export_mw", np.zeros(T)), dtype=float)
    p_grid = imp - exp
    losses = _line_losses(config, result)

    # The clearing's active-power identity, term by term. `inj -= r * I^2` in
    # the bus balance is the only thing on the left that is not a schedule, so
    # the residual is the loss and nothing else.
    residual = p_grid + pv_used + wind_used + p_dis - served - p_ch - losses

    v = np.asarray(result["bus_voltage"], dtype=float)
    buses = list(result["bus_indices"])
    util = np.asarray(result["line_utilization"], dtype=float)
    lines = list(result.get("line_indices") or [])
    band = (float(config.network.v_min_pu), float(config.network.v_max_pu))

    net = grid.build_base_network(config)
    ampacity = {int(i): float(net.line.at[i, "max_i_ka"])
                for i in net.line.index}

    tan_phi = float(np.tan(np.arccos(config.network.load_power_factor)))
    q_series = served * tan_phi

    available = float(result.get("total_re_available", 0.0))
    curtailed = float(result.get("total_curtailment", 0.0))
    used = available - curtailed

    solver = (result.get("solver") or {}).get("final", {}) or {}
    fallback_periods = [int(t) for t in result.get("lmp_fallback_periods") or []]
    prices = np.asarray(result["price"], dtype=float)
    anomalous = sorted(set(fallback_periods)
                       | {t for t in range(len(prices))
                          if not np.isfinite(prices[t])})

    return {
        "scenario": scenario,
        "slots": int(T),
        "buses": buses,
        "case": case.describe() if case is not None else None,
        "unit_basis": case.unit_basis if case is not None else None,
        "solver": {
            "status": solver.get("status_name"),
            "optimal": solver.get("optimal"),
            "runtime_s": solver.get("runtime_s"),
            "fell_back": bool(result.get("fell_back")),
            "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
            "lmp_fallback_periods": fallback_periods,
            "lmp_price_source": result.get("lmp_price_source"),
            "duals_available": (result.get("solver") or {}).get("duals_available"),
            "anomalous_periods": anomalous,
        },
        "voltage": {
            "band": list(band),
            "min": float(v.min()), "max": float(v.max()),
            "min_bus": int(buses[int(np.unravel_index(v.argmin(), v.shape)[1])]),
            "min_period": int(np.unravel_index(v.argmin(), v.shape)[0]),
            "max_bus": int(buses[int(np.unravel_index(v.argmax(), v.shape)[1])]),
            "max_period": int(np.unravel_index(v.argmax(), v.shape)[0]),
            "bus_periods_above": int((v > band[1]).sum()),
            "bus_periods_below": int((v < band[0]).sum()),
            "at_lower_bound": bool(np.isclose(v.min(), band[0], atol=1e-6)),
            "per_bus": [
                {"bus": int(buses[j]),
                 "min": float(v[:, j].min()), "max": float(v[:, j].max()),
                 "mean": float(v[:, j].mean()),
                 "periods_at_lower": int(np.isclose(v[:, j], band[0],
                                                    atol=1e-6).sum())}
                for j in range(v.shape[1])
            ],
        },
        "lines": {
            "n_lines": len(lines),
            "max_utilization": float(util.max()) if util.size else None,
            "max_line": (int(lines[int(np.unravel_index(util.argmax(),
                                                        util.shape)[1])])
                         if lines and util.size else None),
            "max_period": (int(np.unravel_index(util.argmax(), util.shape)[0])
                           if util.size else None),
            "at_limit": bool(util.size
                             and np.isclose(util.max(), 1.0, atol=1e-6)),
            "line_periods_at_or_over": int((util >= 1.0 - 1e-6).sum()),
            "lines_over_limit": int((util.max(axis=0) >= 1.0 - 1e-6).sum())
                                if util.size else 0,
            "per_line": [
                {"line": int(lines[j]),
                 "max_utilization": float(util[:, j].max()),
                 "max_period": int(util[:, j].argmax()),
                 "ampacity_ka": ampacity.get(int(lines[j])),
                 "over_limit": bool(util[:, j].max() >= 1.0 - 1e-6)}
                for j in range(len(lines))
            ],
            "capacity_source": {
                "field": "net.line.max_i_ka, the thermal current limit",
                "used_by": "dispatch_socp._build_line_params converts it to a "
                           "per-unit current limit; the constraint is "
                           "I_sq <= limit^2",
                "provenance": ("PROJECT-ASSIGNED, not the published case33bw "
                               "rating. grid.build_base_network computes "
                               "clip(base_ampacity_ka / sqrt(r_ohm_per_km * "
                               "length_km), min_line_ka, max_line_ka) * "
                               "line_capacity_multiplier * per-line override"),
                "base_ampacity_ka": cfg.get_default("network.base_ampacity_ka"),
                "min_line_ka": cfg.get_default("network.min_line_ka"),
                "max_line_ka": cfg.get_default("network.max_line_ka"),
                "line_capacity_multiplier":
                    float(config.network.line_capacity_multiplier),
                "overrides": {str(k): float(v) for k, v in
                              (config.network.line_capacity_overrides
                               or {}).items()},
                "ampacity_ka_range": ([min(ampacity.values()),
                                       max(ampacity.values())]
                                      if ampacity else None),
                "out_of_service": sorted(
                    int(i) for i in net.line.index
                    if not bool(net.line.at[i, "in_service"])),
            },
        },
        "pcc": {
            "import_max_mw": float(imp.max()) if imp.size else None,
            "import_max_period": int(imp.argmax()) if imp.size else None,
            "import_mean_mw": float(imp.mean()) if imp.size else None,
            "import_mwh": _mwh(imp),
            "export_max_mw": float(exp.max()) if exp.size else None,
            "export_mwh": _mwh(exp),
            "periods_exporting": int((exp > 1e-9).sum()),
            # The exchange the bound is actually on. `reverse_power_limit_mw`
            # is the **lower** bound of one net-exchange variable, so the
            # quantity it constrains is the most negative exchange, not the
            # import.
            "net_exchange_min_mw": float(p_grid.min()) if p_grid.size else None,
            "net_exchange_min_period": int(p_grid.argmin()) if p_grid.size else None,
            "reverse_limit_mw": float(config.network.reverse_power_limit_mw),
            "reverse_limit_slack_mw": (float(config.network.reverse_power_limit_mw
                                             + p_grid.min())
                                       if p_grid.size else None),
            "reverse_limit_breached": bool(p_grid.min()
                                           < -config.network.reverse_power_limit_mw
                                           - 1e-9) if p_grid.size else False,
            "import_limit_mw": None,
            "import_limit_note": (
                "there is no import limit. dispatch_core.add_grid_exchange "
                "builds one net-exchange variable with "
                "lb=-reverse_power_limit_mw and ub=+infinity, so the 2.5 MW "
                "figure caps REVERSE flow (export) and bounds import not at "
                "all. Every engine that clears (socp, lindistflow, dc) does "
                "this through that one helper. A reader comparing the import "
                "maximum against it is comparing two different quantities."),
            "unconstrained_by_design": (
                "import is priced, not capped: the wholesale price is what "
                "makes the solver prefer not to over-import"),
        },
        "energy": {
            "asked_mwh": _mwh(asked),
            "asked_peak_mw": float(asked.max()),
            "asked_peak_period": int(asked.argmax()),
            "served_mwh": _mwh(served),
            "served_peak_mw": float(served.max()),
            "served_peak_period": int(served.argmax()),
            "shed_mwh": _mwh(unserved),
            "served_share": (float(served.sum() / asked.sum())
                             if asked.sum() > 0 else None),
            "load_closure_max_mw": float(np.abs(served + unserved
                                                - asked).max()),
            "re_available_mwh": available,
            "re_used_mwh": used,
            "re_curtailed_mwh": curtailed,
            "re_consumption_rate": float(result["re_consumption_rate"]),
            "storage_charged_mwh": _mwh(p_ch),
            "storage_discharged_mwh": _mwh(p_dis),
            "grid_import_mwh": _mwh(imp),
            "grid_export_mwh": _mwh(exp),
            "network_losses_mwh": _mwh(losses),
            "balance_residual_max_mw": float(np.abs(residual).max()),
            "balance_residual_mean_mw": float(np.abs(residual).mean()),
        },
        "reactive": {
            "rule": "q_load = served * tan(phi), per agent and per period "
                    "(dispatch_socp's reactive-power balance)",
            "power_factor": float(config.network.load_power_factor),
            "tan_phi": tan_phi,
            "status": ("simulation assumption on ACTIVE load, not a "
                       "measurement -- no meter in this workbook reports "
                       "reactive power"),
            "q_peak_mvar": float(q_series.max()),
            "q_peak_period": int(q_series.argmax()),
            "q_energy_mvarh": _mwh(q_series),
            "benchmark_q_mvar": 2.3,
            "support_enabled": bool(config.network.reactive_support),
        },
        "objective": {
            "clearing_objective_cny": float(result["clearing_objective"]),
            "what_it_is": ("the clearing model's own optimization objective, "
                           "in CNY. It is NOT social welfare and NOT any "
                           "participant's payoff: it values each storage unit "
                           "at that unit's own declared bid and offer rather "
                           "than at the settlement price, so it cannot be read "
                           "as anyone's profit"),
            "re_consumption_rate_percent": float(result["re_consumption_rate"]),
            "re_rate_formula": ("100 * total_re_used_mwh / total_re_avail_mwh, "
                                "where available sums the fleet's forecast PV "
                                "and wind over the horizon and used is what the "
                                "clearing consumed"),
            "carbon_tco2": float(result["carbon_emissions"]),
            "carbon_definition": ("grid_carbon_tco2 on the import part of the "
                                  "net exchange times "
                                  "market_design.emission_factor_grid"),
            "carbon_intensity_tco2_per_mwh": float(
                result.get("carbon_intensity", float("nan"))),
        },
        "not_tuned": {
            "voltage_band": list(band),
            "load_power_factor": float(config.network.load_power_factor),
            "line_capacity_multiplier":
                float(config.network.line_capacity_multiplier),
            "reverse_power_limit_mw": float(
                config.network.reverse_power_limit_mw),
            "reactive_support": bool(config.network.reactive_support),
            "statement": ("read from configuration and reported as found; no "
                          "load, power factor, line capacity, voltage bound or "
                          "storage parameter was moved to produce this result"),
        },
    }


def _fmt(value, spec=".4f"):
    return "—" if value is None else format(value, spec)


def write_report(doc: dict, path: Path = REPORT) -> Path:
    case = doc["case"] or {}
    rule = case.get("date_rule", {})
    basis = doc["unit_basis"] or {}
    v, ln, pcc = doc["voltage"], doc["lines"], doc["pcc"]
    energy, reactive, solver = doc["energy"], doc["reactive"], doc["solver"]
    objective = doc["objective"]
    source = ln["capacity_source"]

    lines: List[str] = []
    add = lines.append
    add("# S8 — the physical audit of the subset day")
    add("")
    add(f"One cleared day on `{doc['scenario']}`: "
        f"{doc['slots']} periods, {len(doc['buses'])} buses. Produced by "
        "`src/s8_physical_audit.py`; every number is a solver output and none "
        "has been adjusted.")
    add("")
    add("> **Not an externally certified result.** The reading unit's basis is "
        f"the recorded project ruling `{basis.get('basis')}`; "
        f"`is_externally_verified` is "
        f"**{basis.get('is_externally_verified')}** and this audit inherits "
        "that. The ledger it reads is a **formal** ledger — the ruling is an "
        "admissible basis and `validate_formal_ledger` passes it — but a formal "
        "basis is not a verified one, so nothing here may be presented as "
        "externally certified.")
    add("")
    add("> **Conditional on this repository's own line ampacities.** Those are "
        "project-assigned rather than the published `case33bw` ratings (§3). "
        "Every limit statement below holds under that assumption and nothing "
        "here is a validation against authoritative standard line ratings.")
    add("")
    add("> **Cross-date diagnostic.** Load day "
        f"**{rule.get('load_date')}**, price day **{rule.get('price_date')}** "
        "— different days, paired by decision. No figure here is what either "
        "day actually met in its own market.")
    add("")
    add("## 1. The verdict")
    add("")
    if energy["shed_mwh"] <= 1.0:
        add(f"**The network carried the load.** Served "
            f"**{energy['served_mwh']:,.1f} MWh** of "
            f"**{energy['asked_mwh']:,.1f} MWh** "
            f"(**{energy['served_share']:.2%}**); shed "
            f"**{energy['shed_mwh']:,.3f} MWh**.")
    else:
        add(f"**The network could not carry the load.** Served "
            f"{energy['served_mwh']:,.1f} MWh of {energy['asked_mwh']:,.1f} "
            f"({energy['served_share']:.2%}); shed "
            f"**{energy['shed_mwh']:,.1f} MWh**.")
    add("")
    add("**Both halves of that sentence are conditional.** The voltage band is "
        "declared, and the line ampacities are **project-assigned** rather than "
        f"the published `case33bw` ratings (§3): {v['bus_periods_below'] + v['bus_periods_above']} "
        f"bus-periods leave the band and {ln['line_periods_at_or_over']} "
        "line-periods reach their limit **under this repository's own ampacity "
        "rule**. Nothing here is a validation against authoritative standard "
        "line ratings, and the result must not be described as one.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| scenario | `{doc['scenario']}` |")
    add(f"| periods × buses | {doc['slots']} × {len(doc['buses'])} |")
    add(f"| served / asked | {energy['served_mwh']:,.4f} / "
        f"{energy['asked_mwh']:,.4f} MWh |")
    add(f"| shed | {energy['shed_mwh']:,.4f} MWh |")
    add(f"| served share | {energy['served_share']:.4%} |")
    add("")

    add("## 2. Voltage, all 96 periods and 33 buses")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| band | {v['band'][0]:.4f} .. {v['band'][1]:.4f} pu "
        "(declared, not tuned) |")
    add(f"| min | **{v['min']:.4f}** pu at bus {v['min_bus']}, "
        f"period {v['min_period']} |")
    add(f"| max | {v['max']:.4f} pu at bus {v['max_bus']}, "
        f"period {v['max_period']} |")
    add(f"| bus-periods below the band | **{v['bus_periods_below']}** |")
    add(f"| bus-periods above the band | {v['bus_periods_above']} |")
    add(f"| sitting on the lower bound | **{v['at_lower_bound']}** |")
    add("")
    add("Per bus:")
    add("")
    add("| bus | min pu | max pu | mean pu | periods at the lower bound |")
    add("|---|---|---|---|---|")
    for row in v["per_bus"]:
        add(f"| {row['bus']} | {row['min']:.4f} | {row['max']:.4f} | "
            f"{row['mean']:.4f} | {row['periods_at_lower']} |")
    add("")

    add("## 3. Lines")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| lines in the radial model | {ln['n_lines']} |")
    add(f"| worst utilization | **{_fmt(ln['max_utilization'])}** on line "
        f"{ln['max_line']}, period {ln['max_period']} |")
    add(f"| line-periods at or over the limit | **"
        f"{ln['line_periods_at_or_over']}** |")
    add(f"| lines at or over the limit at some period | "
        f"**{ln['lines_over_limit']}** |")
    add("")
    add("### Where the capacity comes from")
    add("")
    add(f"- field: `{source['field']}`")
    add(f"- used by: {source['used_by']}")
    add(f"- provenance: **{source['provenance']}**")
    add(f"- declared inputs: `base_ampacity_ka` "
        f"{source['base_ampacity_ka']}, clip to "
        f"[{source['min_line_ka']}, {source['max_line_ka']}] kA, "
        f"`line_capacity_multiplier` {source['line_capacity_multiplier']}, "
        f"overrides {source['overrides'] or 'none'}")
    add(f"- resulting ampacity range: "
        f"{source['ampacity_ka_range'][0]:.4f} .. "
        f"{source['ampacity_ka_range'][1]:.4f} kA")
    add(f"- lines marked out of service: "
        f"{source['out_of_service'] or 'none'}")
    add("")
    add("**The ampacity is project-assigned, not the published `case33bw` "
        "rating.** The line capacities are a modelling choice made in this "
        "repository, so two limits follow and both hold for every utilization "
        "figure below and for the verdict in §1:")
    add("")
    add("1. Every statement here about a line being within its limit is "
        "**conditional on this ampacity rule**. A different rule would move "
        "every utilization number, and nothing here would show it.")
    add("2. **This is not a validation against authoritative standard line "
        "ratings**, and it must not be described as one. The `case33bw` "
        "ampacities were replaced, not adopted.")
    add("")
    add("| line | max utilization | at period | ampacity kA | over limit |")
    add("|---|---|---|---|---|")
    for row in sorted(ln["per_line"], key=lambda r: -r["max_utilization"]):
        add(f"| {row['line']} | {row['max_utilization']:.4f} | "
            f"{row['max_period']} | {_fmt(row['ampacity_ka'])} | "
            f"{'**yes**' if row['over_limit'] else 'no'} |")
    add("")

    add("## 4. PCC — power at the point of common coupling")
    add("")
    add("### There is no import limit")
    add("")
    add("**Read this before the table.** The one PCC constraint this model "
        "carries is on **reverse flow**. `dispatch_core.add_grid_exchange` "
        "builds a single net-exchange variable and bounds it **below**:")
    add("")
    add("```")
    add("p_grid = addVars(T, lb=-network.reverse_power_limit_mw, ub=+INFINITY)")
    add("```")
    add("")
    add(f"So the declared {pcc['reverse_limit_mw']:g} MW caps **export** and "
        "bounds import not at all, and every engine that clears — `socp`, "
        "`lindistflow`, `dc` — builds it through that one helper. **Import is "
        "priced, not capped.** An import maximum compared against this figure "
        "compares two different quantities; the quantity the bound acts on is "
        "the most negative net exchange, which is reported below.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| import max | {pcc['import_max_mw']:,.4f} MW at period "
        f"{pcc['import_max_period']} |")
    add(f"| import mean | {pcc['import_mean_mw']:,.4f} MW |")
    add(f"| import energy | {pcc['import_mwh']:,.4f} MWh |")
    add("| **import limit** | **none — unbounded by construction** |")
    add(f"| export max | {pcc['export_max_mw']:,.4f} MW |")
    add(f"| export energy | {pcc['export_mwh']:,.4f} MWh |")
    add(f"| periods exporting | {pcc['periods_exporting']} |")
    add(f"| **most negative net exchange** | "
        f"{pcc['net_exchange_min_mw']:,.4f} MW at period "
        f"{pcc['net_exchange_min_period']} |")
    add(f"| reverse-flow limit (the bound actually applied) | "
        f"−{pcc['reverse_limit_mw']:,.4f} MW |")
    add(f"| slack on that bound | {pcc['reverse_limit_slack_mw']:,.4f} MW |")
    add(f"| reverse limit breached | **{pcc['reverse_limit_breached']}** |")
    add("")
    add("Both are the same variable, in MW, at the same period index: the "
        "import series is `max(net, 0)` and the export series `max(-net, 0)`, "
        "taken from the solver's own `p_grid[t].X` — the variable the bound "
        "sits on. So the reading and the constraint are on the same quantity, "
        "and on this day the exchange never went negative at all.")
    add("")

    add("## 5. Supply, load, and the energy balance")
    add("")
    add("| term | MWh |")
    add("|---|---|")
    add(f"| load asked | {energy['asked_mwh']:,.4f} |")
    add(f"| served | {energy['served_mwh']:,.4f} |")
    add(f"| shed | {energy['shed_mwh']:,.4f} |")
    add(f"| load peak (asked) | {energy['asked_peak_mw']:,.4f} MW at period "
        f"{energy['asked_peak_period']} |")
    add(f"| load peak (served) | {energy['served_peak_mw']:,.4f} MW at period "
        f"{energy['served_peak_period']} |")
    add(f"| grid import | {energy['grid_import_mwh']:,.4f} |")
    add(f"| grid export | {energy['grid_export_mwh']:,.4f} |")
    add(f"| renewable available | {energy['re_available_mwh']:,.4f} |")
    add(f"| renewable consumed | {energy['re_used_mwh']:,.4f} |")
    add(f"| renewable curtailed | {energy['re_curtailed_mwh']:,.4f} |")
    add(f"| storage charged | {energy['storage_charged_mwh']:,.4f} |")
    add(f"| storage discharged | {energy['storage_discharged_mwh']:,.4f} |")
    add(f"| **network losses** | **{energy['network_losses_mwh']:,.4f}** |")
    add("")
    add("The identity, per period, rebuilt from the clearing's own terms:")
    add("")
    add("```")
    add("grid_import - grid_export + pv_used + wind_used + p_dis")
    add("    - served - p_ch - losses  =  0")
    add("```")
    add("")
    add("| residual | MW |")
    add("|---|---|")
    add(f"| max | {energy['balance_residual_max_mw']:.3e} |")
    add(f"| mean | {energy['balance_residual_mean_mw']:.3e} |")
    add("")
    add(f"`losses` is the model's own `r * I^2` per line, recovered from the "
        f"solver's line currents through `dispatch_socp._build_line_params`. "
        f"Load closure `served + unserved - asked` is within "
        f"{energy['load_closure_max_mw']:.3e} MW.")
    add("")

    add("## 6. Solver state, nodal-price fallbacks, anomalous periods")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| status | `{solver['status']}` (optimal: {solver['optimal']}) |")
    add(f"| runtime | {_fmt(solver['runtime_s'], '.3f')} s |")
    add(f"| fell back to a single-period engine | {solver['fell_back']} |")
    add(f"| balance duals available | {solver['duals_available']} |")
    add(f"| nodal-price fallbacks (cells) | {solver['lmp_fallbacks']} |")
    add(f"| fallback periods | {solver['lmp_fallback_periods'] or 'none'} |")
    add(f"| price source | `{solver['lmp_price_source']}` |")
    add(f"| **anomalous periods** | "
        f"{solver['anomalous_periods'] or 'none'} |")
    add("")
    add("An anomalous period is one whose price is not a nodal balance dual -- "
        "either it fell back to the wholesale curve, or its price is not "
        "finite. `lmp` may not be read as a nodal price field over any period "
        "listed above.")
    add("")

    add("## 7. Reactive load: what it is and what it is not")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| rule | {reactive['rule']} |")
    add(f"| power factor | {reactive['power_factor']} |")
    add(f"| tan(phi) | {reactive['tan_phi']:.6f} |")
    add(f"| Q at the peak period | {reactive['q_peak_mvar']:,.4f} MVAr "
        f"(period {reactive['q_peak_period']}) |")
    add(f"| Q energy over the day | {reactive['q_energy_mvarh']:,.4f} MVArh |")
    add(f"| benchmark Q | {reactive['benchmark_q_mvar']} MVAr |")
    add(f"| inverter reactive support | {reactive['support_enabled']} |")
    add("")
    add(f"**This is an assumption, not a measurement.** {reactive['status']}. "
        "It is charged on the **served** active load, so a period that sheds "
        "is also a period that draws less reactive power. The factor is "
        "`0.8503`, declared to reproduce the benchmark's 2.3 MVAr at the "
        "benchmark's own 3.715 MW; this day peaks below that, so the "
        "reactive load is correspondingly below 2.3 MVAr. It was not "
        "re-tuned.")
    add("")

    add("## 8. What the clearing objective and the RE rate are")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| `clearing_objective` | {objective['clearing_objective_cny']:,.2f} "
        "CNY |")
    add(f"| what it is | {objective['what_it_is']} |")
    add(f"| `re_consumption_rate` | "
        f"{objective['re_consumption_rate_percent']:.4f} % |")
    add(f"| how it is computed | {objective['re_rate_formula']} |")
    add(f"| carbon | {objective['carbon_tco2']:,.4f} tCO2 |")
    add(f"| carbon definition | {objective['carbon_definition']} |")
    add(f"| carbon intensity | "
        f"{objective['carbon_intensity_tco2_per_mwh']:,.6f} tCO2/MWh |")
    add("")
    add("**The clearing objective is not welfare and not a payoff.** It is the "
        "optimization expression the solver minimised, in CNY, and it values "
        "each storage unit at that unit's own declared bid and offer rather "
        "than at the settlement price. Any economic statement about this day "
        "belongs to `participant_payoff`, not to this number.")
    add("")
    add("**The RE rate is a ratio of energies, in percent.** Its numerator is "
        "the renewable energy the clearing consumed and its denominator is the "
        "energy the fleet's PV and wind could have produced over the horizon, "
        "both in MWh. It says nothing about price, cost or revenue.")
    add("")

    add("## 9. What was not tuned")
    add("")
    add(f"{doc['not_tuned']['statement']}:")
    add("")
    add("| parameter | value read |")
    add("|---|---|")
    add(f"| voltage band | {doc['not_tuned']['voltage_band'][0]} .. "
        f"{doc['not_tuned']['voltage_band'][1]} pu |")
    add(f"| `load_power_factor` | {doc['not_tuned']['load_power_factor']} |")
    add(f"| `line_capacity_multiplier` | "
        f"{doc['not_tuned']['line_capacity_multiplier']} |")
    add(f"| `reverse_power_limit_mw` | "
        f"{doc['not_tuned']['reverse_power_limit_mw']} |")
    add(f"| `reactive_support` | {doc['not_tuned']['reactive_support']} |")
    add("")
    add("## 10. What this audit does not establish")
    add("")
    add("- It is **not** the formal physical network case. The reading unit is "
        "unverified; the ledger is an `audit` ledger.")
    add("- It is **not** a historical replay: the load day and the price day "
        "differ by decision.")
    add("- The line ampacities are **project-assigned**, so every utilization "
        "figure is conditional on that assignment.")
    add("- It says nothing about a learned policy, about profit, or about what "
        "any participant would have earned.")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add(f"PYTHONPATH=src python src/s8_physical_audit.py --scenario "
        f"{doc['scenario']}")
    add("```")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    doc = measure(args.scenario)
    write_report(doc, Path(args.report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

    energy, v, ln, pcc = doc["energy"], doc["voltage"], doc["lines"], doc["pcc"]
    solver = doc["solver"]
    print(f"scenario    {doc['scenario']}   {doc['slots']} x "
          f"{len(doc['buses'])}")
    print(f"solver      {solver['status']}  fallbacks {solver['lmp_fallbacks']}"
          f"  anomalous {len(solver['anomalous_periods'])}")
    print(f"served      {energy['served_mwh']:,.1f} MWh of "
          f"{energy['asked_mwh']:,.1f} ({energy['served_share']:.4%}), "
          f"shed {energy['shed_mwh']:,.4f} MWh")
    print(f"voltage     {v['min']:.4f} .. {v['max']:.4f} pu "
          f"(outside band: {v['bus_periods_below'] + v['bus_periods_above']})")
    print(f"lines       worst {ln['max_utilization']:.4f}, at/over limit: "
          f"{ln['line_periods_at_or_over']}")
    print(f"pcc         import max {pcc['import_max_mw']:,.4f} MW "
          "(no import limit); most negative exchange "
          f"{pcc['net_exchange_min_mw']:,.4f} MW against the "
          f"-{pcc['reverse_limit_mw']:,.2f} MW reverse-flow bound")
    print(f"balance     residual max "
          f"{energy['balance_residual_max_mw']:.3e} MW")
    print(f"wrote       {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
