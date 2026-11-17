"""Run one frozen training day through the new independent DA mechanism."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandapower.networks as networks

from local_da_contracts import (DayAheadInput, Generator, RadialNetwork, StorageBid,
                                StorageUnit)
from local_da_clearing import ClearingModel, clear_day
from local_da_accounting import settle_day

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/multi_user_load_2026-11-13/evidence"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
                          + "\n", encoding="utf-8")


def archive_result(path, record):
    arrays = {}

    def flatten(value, prefix):
        if isinstance(value, np.ndarray):
            arrays[prefix] = value
            return {"npz_key": prefix, "shape": list(value.shape)}
        if isinstance(value, dict):
            return {key: flatten(item, f"{prefix}/{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [flatten(item, f"{prefix}/{index}") for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    summary = flatten(record, "result")
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write_json(path.with_suffix(".json"), summary)


def production_renewable_curves():
    """Evaluate existing pure profile formulas without importing the old market."""
    path = ROOT / "src/grid.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"pv_profile", "wind_profile"}]
    if len(functions) != 2:
        raise ValueError("Existing renewable profile definitions are unavailable")
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    hours = np.arange(96) * .25
    pv = namespace["pv_profile"](hours, amplitude=1., cloud_prob=.15,
                                  cloud_attenuation_min=.55, cloud_attenuation_max=.85,
                                  cloud_persistence=.85, cloud_seed=42, dt_h=1.)
    wind = namespace["wind_profile"](hours, seed=72, ar_coef=.9, noise_std=.15,
                                      peak_hour=3., dt_h=1.)
    generators = []
    for bus in [21, 23]:
        noise = np.random.RandomState(bus * 10 + 2).normal(1., .35, 96)
        values = np.clip(np.maximum(pv * .3 * noise, 0), 0, .3)
        generators.append(Generator(f"PV{bus}", bus, values))
    noise = np.random.RandomState(304).normal(1., .12, 96)
    generators.append(Generator("WIND30", 30, np.clip(np.maximum(wind * noise, 0), 0, 1.)))
    return tuple(generators)


def build_day(date):
    split = json.loads((ROOT / "data/processed/v2_cy_jn_2025/common_split.json").read_text())
    pairs = [p for p in split["near"]["train"] if p.split("|")[0] == date]
    if len(pairs) != 1:
        raise ValueError("Only one explicitly authorized train pair may be loaded")
    metadata = json.loads((DATA / "manifest.json").read_text())
    for name, expected in metadata["outputs"].items():
        if sha(DATA / name) != expected:
            raise ValueError(f"Changed frozen load output: {name}")
    if date not in metadata["retained_dates"]:
        raise ValueError("The requested date is outside the accepted real-load dataset")
    with np.load(DATA / "bus_active_load_mw.npz", allow_pickle=False) as archive:
        p = archive[date].copy()
    price_date = pairs[0].split("|")[1]
    price_path = ROOT / "data/processed/v2_cy_jn_2025/price_curves.npz"
    with np.load(price_path, allow_pickle=False) as archive:
        price = archive[f"south/{price_date}"].copy()
    net = networks.case33bw()
    ratio = np.zeros(33)
    for _, row in net.load.iterrows():
        ratio[int(row.bus)] = row.q_mvar / row.p_mw
    opened = set(int(row.element) for _, row in net.switch.iterrows()
                 if row.et == "l" and not row.closed)
    available = [(int(idx), int(row.from_bus), int(row.to_bus)) for idx, row in net.line.iterrows()
                 if row.in_service and int(idx) not in opened]
    reached, ordered = {0}, []
    while available:
        progress = False
        for entry in available.copy():
            idx, start, end = entry
            if (start in reached) != (end in reached):
                if end in reached:
                    start, end = end, start
                ordered.append((idx, start, end))
                reached.add(end)
                available.remove(entry)
                progress = True
        if not progress:
            raise ValueError("Original case33bw topology is not a connected radial tree")
    if len(ordered) != 32 or len(reached) != 33:
        raise ValueError("Unexpected original IEEE33 topology")
    zbase = 12.66 ** 2
    r = np.array([net.line.at[idx, "r_ohm_per_km"] * net.line.at[idx, "length_km"] / zbase
                  for idx, _, _ in ordered])
    x = np.array([net.line.at[idx, "x_ohm_per_km"] * net.line.at[idx, "length_km"] / zbase
                  for idx, _, _ in ordered])
    network = RadialNetwork(33, 0, tuple((start, end) for _, start, end in ordered), r, x)
    storage = tuple(StorageUnit(f"ESS{bus}", bus) for bus in [9, 14, 17, 27])
    bids = {unit.name: StorageBid(np.full(96, .4), np.full(96, .4), np.zeros(96), np.zeros(96))
            for unit in storage}
    day = DayAheadInput(network, p, p * ratio[None, :], price, storage,
                        production_renewable_curves(), bids, 0., 600.)
    return day, {"load_date": date, "price_date": price_date, "pair_key": pairs[0],
                 "scenario_kind": "Cross-year perfect-forecast research diagnostic",
                 "bid_price_bounds": "0..600 CNY/MWh research interval, not verified Jiangsu formal limits",
                 "price_unit_status": "Historical inferred CNY/MWh; not externally confirmed",
                 "q_basis": "Native bus Q0/P0 assumption, no ideal compensation",
                 "renewables": "Existing synthetic profile formulas; PV .3 MW at 21/23, wind 1 MW at 30; clipped to declared nameplate",
                 "generator_q_limits_mvar": 0., "thermal_limits_qualified": False,
                 "network": "Native case33bw R/X, slack 1.0 pu, no thermal rating inference",
                 "terminal_soc": "free", "quote_reference": "zero-price, full-quantity reference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2022-12-22")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("A finite positive solve budget is required")
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Evidence output must stay within the repository")
    day, context = build_day(args.date)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "context.json", context)
    arrays = {"active_load_mw": day.active_load_mw, "reactive_load_mvar": day.reactive_load_mvar,
              "external_price_cny_mwh": day.external_price_cny_mwh,
              "branch_r": day.network.r_mw_base, "branch_x": day.network.x_mw_base}
    arrays.update({g.name: g.available_mw for g in day.generators})
    np.savez_compressed(output / "inputs.npz", **arrays)
    result = clear_day(day, seconds=args.seconds, log_path=output / "exact_solver.log")
    archive_result(output / "baseline", result)
    if result["qualified_for_settlement"]:
        ledger = settle_day(day, result, inventory_price_cny_mwh=float(day.external_price_cny_mwh[-1]))
        write_json(output / "settlement.json", ledger)
    if not result["dispatch"]["optimal"]:
        relaxed_model = ClearingModel(day, exact=False, seconds=args.seconds)
        try:
            relaxed = relaxed_model.solve()
            if relaxed["solver_status"] == 3:
                relaxed_model.model.computeIIS()
                relaxed_model.model.write(str(output / "infeasible_relaxation.ilp"))
                conflict = {
                    "linear_constraints": [c.ConstrName for c in relaxed_model.model.getConstrs() if c.IISConstr],
                    "quadratic_constraints": [c.QCName for c in relaxed_model.model.getQConstrs() if c.IISQConstr],
                    "variable_lower_bounds": [v.VarName for v in relaxed_model.model.getVars() if v.IISLB],
                    "variable_upper_bounds": [v.VarName for v in relaxed_model.model.getVars() if v.IISUB],
                    "proof_scope": "A conflict in the relaxed fixed-input model; not a minimum load-shedding result",
                }
                write_json(output / "infeasibility_conflict.json", conflict)
        finally:
            relaxed_model.close()
        archive_result(output / "continuous_relaxation_diagnostic", relaxed)
    inputs = [ROOT / "src" / f for f in ["local_da_contracts.py", "local_da_clearing.py",
              "local_da_accounting.py", "run_local_da.py", "grid.py"]]
    inputs += [DATA / "manifest.json", DATA / "bus_active_load_mw.npz",
               ROOT / "data/processed/v2_cy_jn_2025/common_split.json",
               ROOT / "data/processed/v2_cy_jn_2025/price_curves.npz",
               ROOT / "docs/2026-11-13-local-da-clearing-spec.md"]
    status = {"qualified_for_settlement": result["qualified_for_settlement"],
              "dispatch_status": result["dispatch"]["solver_status"],
              "price_qualification": result["price_qualification"],
              "declared_integer_variables": result["dispatch"]["declared_integer_variables"],
              "research_diagnostic_only": True,
              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
              "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", status)
    print(json.dumps({key: value for key, value in status.items() if key not in {"inputs", "outputs"}}), flush=True)


if __name__ == "__main__":
    main()
