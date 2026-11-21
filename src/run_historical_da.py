"""Run a paired full-day service and reactive-support comparison on the old map."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from local_da_accounting import settle_day
from local_da_operating import (OperatingRules, audit_operating, build_historical_day,
                                clear_operating_day, support_day)
from run_local_da import ROOT, archive_result, sha, write_json


def case_summary(name, day, result):
    summary = {"case": name, "qualified_for_settlement": result["qualified_for_settlement"],
               "price_qualification": result["price_qualification"],
               "physical_operating_qualified": result.get("physical_operating_qualified", False)}
    dispatch = result.get("dispatch", result.get("minimum_shedding_stage", {}))
    summary.update(solver_status=dispatch.get("solver_status"),
                   solver_status_name=dispatch.get("solver_status_name"),
                   optimal=dispatch.get("optimal", False), solution_count=dispatch.get("solution_count", 0))
    if dispatch.get("solution_count", 0):
        shed = dispatch["shed_mw"]
        q = sum((abs(v) for v in dispatch["generator_q"].values()), np.zeros(day.periods))
        summary.update(shed_mwh=dispatch["shed_mwh"], shed_peak_mw=float(shed.sum(axis=1).max()),
                       served_fraction=1. - dispatch["shed_mwh"] / (day.active_load_mw.sum() * day.dt_hours),
                       shed_periods=int((shed.sum(axis=1) > 1e-6).sum()),
                       shed_nodes_one_based=[int(b + 1) for b in np.flatnonzero(shed.max(axis=0) > 1e-6)],
                       voltage_min_pu=float(np.sqrt(dispatch["voltage_sq"]).min()),
                       voltage_max_pu=float(np.sqrt(dispatch["voltage_sq"]).max()),
                       q_abs_peak_mvar=float(q.max()),
                       q_abs_integral_mvarh=float(q.sum() * day.dt_hours),
                       integer_variables=dispatch["declared_integer_variables"],
                       binary_variables=dispatch["declared_binary_variables"])
    if "dispatch_audit" in result:
        summary["audit"] = result["dispatch_audit"]
    if "failure" in result:
        summary["failure"] = result["failure"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2022-12-22")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--cases", nargs="+", choices=["full_q0", "full_qbounded", "emergency_q0", "emergency_qbounded"],
                        default=["full_q0", "full_qbounded", "emergency_q0", "emergency_qbounded"])
    parser.add_argument("--price-probes", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Solve budget must be finite and positive")
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Run output must stay within the repository")
    day, context, mapping = build_historical_day(args.date)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "context.json", context)
    write_json(output / "historical_source_mapping.json", mapping)
    np.savez_compressed(output / "loads.npz", active_load_mw=day.active_load_mw,
                        reactive_load_mvar=day.reactive_load_mvar)
    summaries = []
    for case in args.cases:
        rules = OperatingRules(allow_emergency_shedding=case.startswith("emergency"),
                               bounded_reactive_support=case.endswith("qbounded"))
        write_json(output / f"{case}_rules.json", asdict(rules))
        print(f"Starting {case}", flush=True)
        result = clear_operating_day(day, rules, seconds=args.seconds,
                                     log_path=output / f"{case}.log",
                                     price_probes=args.price_probes and case == "full_qbounded")
        archive_result(output / case, result)
        summary = case_summary(case, day, result)
        if result["qualified_for_settlement"]:
            supplied, _ = support_day(day, rules)
            ledger = settle_day(supplied, result, inventory_price_cny_mwh=float(day.external_price_cny_mwh[-1]))
            write_json(output / f"{case}_settlement.json", ledger)
        summaries.append(summary)
        write_json(output / "summary.json", {"date": args.date, "context": context, "cases": summaries,
                   "thermal_limits_qualified": False, "new_training_started": False})
        print({key: value for key, value in summary.items() if key != "audit"}, flush=True)
    inputs = [ROOT / name for name in ["src/local_da_operating.py", "src/run_historical_da.py",
              "src/local_da_clearing.py", "src/local_da_accounting.py", "src/local_da_contracts.py",
              "src/run_local_da.py", "src/grid.py", "data/processed/industry_meter_bus_map_v2.csv",
              "data/processed/industry_meter_bus_map_v2_manifest.json",
              "reports/multi_user_load_2026-11-13/evidence/source_bus_mapping.csv",
              "reports/multi_user_load_2026-11-13/evidence/bus_active_load_mw.npz",
              "data/processed/v2_cy_jn_2025/common_split.json", "data/processed/v2_cy_jn_2025/price_curves.npz",
              "reports/historical_mapping_da_2026-11-13/SPEC.md"]]
    write_json(output / "manifest.json", {"inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
               "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
               "claims": "Historical map, explicit research Q capability, four independent ESS; no formal Jiangsu compliance claim"})


if __name__ == "__main__":
    main()
