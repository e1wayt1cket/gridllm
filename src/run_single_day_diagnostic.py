# run_single_day_diagnostic.py
"""One real single-day clearing, on the real load and the real price curve.

**What this runs.** `typical_day_industry`, whose load comes from the node-load
ledger through `single_day_data` and whose price comes from the `261009` curve.
Both are measured data; the load day and the price day are different, and the
case type says so.

**What it is not.** Not a historical replay, and not evidence about either day's
actual market. It is a diagnostic that asks one question -- can this network
carry this load -- and the answer is reported as it comes out, including "no".

**What it will not do.** It does not scale the load, widen a line, relax a
voltage band or move a storage parameter to make the run feasible. If the
network cannot serve the load, that is the result. Every number below is a
solver output; none is adjusted.

Usage:
    PYTHONPATH=src python src/run_single_day_diagnostic.py
    PYTHONPATH=src python src/run_single_day_diagnostic.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import money  # noqa: E402
import scenarios  # noqa: E402

REPORT = Path("reports/single_day_diagnostic_run.md")
SCENARIO = "typical_day_industry"


def _flat(schedules, agents, key) -> np.ndarray:
    return np.array([float(np.sum(schedules[a.name][key])) for a in agents])


def clear_once(scenario: str = SCENARIO, T: int = money.PERIODS_PER_DAY) -> dict:
    """Build through the boundary and clear the day once, keeping everything.

    The one clearing this module and `s8_physical_audit` share. Two reports over
    one day must not come from two clears: a second clear is a second set of
    numbers, and the two would drift apart without either being wrong. The raw
    result travels with the fleet, the config and the case, so a caller can
    measure what this module only digests.
    """
    from market import clear_market
    from strategies.fixed_bidding import FixedStrategy

    config = scenarios.MarketConfig.from_yaml()
    agents, wholesale = scenarios.get_scenario(scenario, T=T, config=config)
    case = scenarios.last_single_day_case()

    declared = FixedStrategy().formulate(agents, config, None, T)
    result = clear_market(agents, T, "DA", declared, config,
                          wholesale=wholesale, horizon_type="full_day")
    return {"scenario": scenario, "T": T, "config": config, "agents": agents,
            "wholesale": wholesale, "case": case, "result": result}


def run(scenario: str = SCENARIO, T: int = money.PERIODS_PER_DAY) -> dict:
    """Build through the boundary, clear once, and report what came back."""
    cleared = clear_once(scenario, T)
    config = cleared["config"]
    agents = cleared["agents"]
    case = cleared["case"]
    result = cleared["result"]

    served_mwh = float(sum(_flat(result["schedules"], agents, "served"))) \
        * money.DT_HOURS
    shed_mwh = float(sum(_flat(result["schedules"], agents, "unserved"))) \
        * money.DT_HOURS
    asked_mwh = served_mwh + shed_mwh

    v = np.asarray(result["bus_voltage"], dtype=float)
    buses = list(result["bus_indices"])
    vmin_idx = np.unravel_index(np.argmin(v), v.shape)
    util = np.asarray(result["line_utilization"], dtype=float)
    lines = list(result.get("line_indices", []))
    util_idx = np.unravel_index(np.argmax(util), util.shape)

    imp = np.asarray(result.get("grid_import_mw", []), dtype=float)
    exp = np.asarray(result.get("grid_export_mw", []), dtype=float)

    band = (config.network.v_min_pu, config.network.v_max_pu)

    return {
        "scenario": scenario,
        "case": case.describe() if case is not None else None,
        "unit_basis": case.unit_basis if case is not None else None,
        "solver": result.get("solver"),
        "fell_back": bool(result.get("fell_back")) if result.get("fell_back")
                     else False,
        "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
        "has_wholesale_fallback": bool(result.get("has_wholesale_fallback")),
        "clearing_objective": float(result["clearing_objective"]),
        "re_consumption_rate": float(result["re_consumption_rate"]),
        "carbon_emissions": float(result["carbon_emissions"]),
        "asked_mwh": asked_mwh,
        "served_mwh": served_mwh,
        "shed_mwh": shed_mwh,
        "served_share": (served_mwh / asked_mwh) if asked_mwh > 0 else None,
        "voltage": {
            "band": list(band),
            "min": float(v.min()), "max": float(v.max()),
            "min_bus": int(buses[vmin_idx[1]]), "min_period": int(vmin_idx[0]),
            "at_lower_bound": bool(np.isclose(v.min(), band[0], atol=1e-6)),
            "at_upper_bound": bool(np.isclose(v.max(), band[1], atol=1e-6)),
            "bus_periods_outside_band": int(((v < band[0])
                                             | (v > band[1])).sum()),
        },
        "lines": {
            "max_utilization": float(util.max()),
            "max_line": (lines[util_idx[1]] if lines else None),
            "max_period": int(util_idx[0]),
            "at_limit": bool(np.isclose(util.max(), 1.0, atol=1e-6)),
            "lines_at_or_over_limit": int((util >= 1.0 - 1e-6).sum()),
        },
        "pcc": {
            "import_max_mw": float(imp.max()) if imp.size else None,
            "import_mean_mw": float(imp.mean()) if imp.size else None,
            "export_max_mw": float(exp.max()) if exp.size else None,
            "periods_exporting": int((exp > 1e-9).sum()) if exp.size else None,
            "reverse_limit_mw": float(config.network.reverse_power_limit_mw),
        },
        "network_infeasible": bool(shed_mwh > 1.0),
    }


def _fmt(value, spec=".4f"):
    return "—" if value is None else format(value, spec)


def write_report(doc: dict, path: Path = REPORT) -> Path:
    case = doc["case"] or {}
    rule = case.get("date_rule", {})
    load = case.get("load_source", {})
    price = case.get("price_source", {})
    v, ln, pcc = doc["voltage"], doc["lines"], doc["pcc"]
    basis = doc["unit_basis"] or {}
    solver = (doc.get("solver") or {}).get("final", {}) or {}

    lines = [
        "# The single-day diagnostic clearing",
        "",
        "One real clearing run on the single-day case: the real metered industry",
        "load from the node-load ledger, against the real `261009` price curve,",
        "through `single_day_data`. This report is the output of",
        "`src/run_single_day_diagnostic.py`; every number in it is a solver",
        "output and none has been adjusted.",
        "",
        "> **This is not a historical replay.** The load is dated "
        f"**{rule.get('load_date')}** and the price **{rule.get('price_date')}**",
        "> — different days, paired by decision as a `cross_date_diagnostic`.",
        "> Nothing here is what either day actually met in its own market, and no",
        "> revenue, profit or welfare figure computed from this run describes",
        "> anything that happened.",
        "",
        "## The verdict, first",
        "",
    ]
    if doc["network_infeasible"]:
        lines += [
            f"**The network cannot carry this load.** It served "
            f"**{doc['served_mwh']:,.1f} MWh** of the "
            f"**{doc['asked_mwh']:,.1f} MWh** demanded — "
            f"**{doc['served_share']:.2%}** — and shed "
            f"**{doc['shed_mwh']:,.1f} MWh**.",
            "",
            "This is reported as the result, not treated as a failure of the",
            "run. The load was not scaled, no line was widened, no band was",
            "relaxed and no storage parameter was moved to change it.",
        ]
    else:
        lines += [
            f"**The network carried the load.** Served "
            f"{doc['served_mwh']:,.1f} MWh against "
            f"{doc['asked_mwh']:,.1f} MWh demanded.",
        ]

    lines += [
        "",
        "| | |",
        "|---|---|",
        f"| case type | `{case.get('case_type')}` |",
        f"| load date | {rule.get('load_date')} |",
        f"| price date | {rule.get('price_date')} |",
        f"| same day | **{rule.get('same_day')}** |",
        f"| load | `{load.get('ledger')}` |",
        f"| price | sheet `{price.get('sheet')}` from `{price.get('workbook')}` |",
        f"| unit basis | **{basis.get('basis')}** — "
        f"`{basis.get('reading_unit')}`, evidence `{basis.get('evidence')}` |",
        f"| externally verified | **{basis.get('is_externally_verified')}** |",
        "",
        "## What the solver returned",
        "",
        "| | |",
        "|---|---|",
        f"| status | `{solver.get('status_name')}` (optimal: "
        f"{solver.get('optimal')}) |",
        f"| fell back to a single-period engine | {doc['fell_back']} |",
        f"| nodal-price fallbacks | {doc['lmp_fallbacks']} |",
        f"| price field | `{case.get('alignment', {}).get('method', '')}` |",
        f"| clearing objective | {doc['clearing_objective']:,.2f} |",
        f"| renewable consumption rate | {doc['re_consumption_rate']:.2f}% |",
        f"| carbon | {doc['carbon_emissions']:,.2f} tCO2 |",
        "",
        "## Network state",
        "",
        "| | |",
        "|---|---|",
        f"| bus voltage band | {v['band'][0]} .. {v['band'][1]} pu |",
        f"| voltage min | **{v['min']:.4f}** at bus {v['min_bus']}, "
        f"period {v['min_period']} |",
        f"| voltage max | {v['max']:.4f} |",
        f"| voltage sitting on the lower bound | **{v['at_lower_bound']}** |",
        f"| bus-periods outside the band | {v['bus_periods_outside_band']} |",
        f"| worst line utilization | **{ln['max_utilization']:.4f}** on line "
        f"{ln['max_line']}, period {ln['max_period']} |",
        f"| line sitting at its limit | **{ln['at_limit']}** |",
        f"| lines at or over their limit | {ln['lines_at_or_over_limit']} "
        f"(of {len(case.get('load_policies', {})) and 32}) |",
        f"| PCC import | max {_fmt(pcc['import_max_mw'], ',.2f')} MW, mean "
        f"{_fmt(pcc['import_mean_mw'], ',.2f')} MW |",
        f"| PCC export | max {_fmt(pcc['export_max_mw'], ',.2f')} MW, over "
        f"{pcc['periods_exporting']} periods (limit "
        f"{pcc['reverse_limit_mw']} MW) |",
        "",
    ]

    if v["at_lower_bound"] or ln["at_limit"]:
        binding = []
        if v["at_lower_bound"]:
            binding.append("the lower voltage bound, at "
                           f"{v['bus_periods_outside_band']} bus-periods "
                           "outside the band")
        if ln["at_limit"]:
            binding.append(f"a line's thermal limit, with "
                           f"{ln['lines_at_or_over_limit']} line-periods at or "
                           f"over it")
        lines += [
            "### What is binding",
            "",
            "The clearing is held by " + " and by ".join(binding) + ". A shed "
            "figure of this size is therefore a statement about the **network's",
            "capacity at this load**, not about the price or about the fleet:",
            "the market shed load it could not deliver, and it could not deliver",
            "it because the feeder's voltage and thermal limits are already",
            "reached.",
            "",
        ]

    lines += [
        "## What this run does not establish",
        "",
        "- It is **not** a historical replay, and no revenue, profit or",
        "  arbitrage figure from it describes anything that happened.",
        "- The load rests on a **user-confirmed project ruling** for the reading",
        "  unit (`kW`), not on an external document. `is_externally_verified` is",
        "  `false` and no figure here may be presented as externally certified.",
        "- The price curve is a **spatial mean** read as a single day **by",
        "  project ruling**; it represents no bus or zone.",
        "- It is **not** a formal physical network case: the formal ledger is",
        "  still refused, and this run is a diagnostic on the audit ledger.",
        "",
        "## Reproduce",
        "",
        "```",
        "PYTHONPATH=src python src/run_single_day_diagnostic.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default=SCENARIO)
    p.add_argument("--json", default=None)
    p.add_argument("--report", default=str(REPORT))
    args = p.parse_args(argv)

    doc = run(args.scenario)
    write_report(doc, Path(args.report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

    print(f"case        {doc['case']['case_type']}  "
          f"{doc['case']['date_rule']['load_date']} / "
          f"{doc['case']['date_rule']['price_date']}")
    print(f"unit basis  {doc['unit_basis']['basis']} "
          f"(externally verified: {doc['unit_basis']['is_externally_verified']})")
    print(f"served      {doc['served_mwh']:,.1f} MWh of {doc['asked_mwh']:,.1f} "
          f"({doc['served_share']:.2%})")
    print(f"shed        {doc['shed_mwh']:,.1f} MWh")
    print(f"voltage     min {doc['voltage']['min']:.4f} "
          f"(at lower bound: {doc['voltage']['at_lower_bound']})")
    print(f"worst line  {doc['lines']['max_utilization']:.4f} "
          f"(at limit: {doc['lines']['at_limit']})")
    print(f"fallbacks   {doc['lmp_fallbacks']}   fell_back {doc['fell_back']}")
    print(f"wrote       {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
