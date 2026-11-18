"""Clear every v3 library day and report whether the market can serve it.

Feasibility and settlement qualification are separate claims. A day is
feasible when the exact dispatch solves and every bus-period stays inside the
network's own voltage band. `qualified_for_settlement` additionally requires
the physical audits, an idle-direction-stable price and smooth prices. Both
are reported, and neither edits the library.
"""

import argparse
from collections import Counter
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import grid  # noqa: F401  ortools must load before pandas on this platform

from local_da_clearing import clear_day
from run_local_da import ROOT, V3, archive_result, build_day, sha, write_json

BLOCKS = ("train", "validation", "test", "holiday")
COLUMNS = ["date", "block", "price_date", "optimal", "solver_status", "solution_count",
           "runtime_seconds", "voltage_min_pu", "voltage_max_pu", "voltage_band_ok",
           "voltage_at_lower_bound", "voltage_at_upper_bound", "pcc_peak_mw",
           "clearing_objective", "price_qualification", "qualified_for_settlement",
           "feasible", "failure"]


def library_dates():
    """Block membership and the paired price day, both from the frozen split."""
    split = json.loads((V3 / "split.json").read_text(encoding="utf-8"))
    block_of = {date: name for name, rows in split["blocks"].items() for date in rows}
    price_of = {pair.split("|")[0]: pair.split("|")[1]
                for rows in split["pairs"].values() for pair in rows}
    return block_of, price_of


def row(date, block, price_date, day, result):
    """One library day's verdict; the band is the day's own, never a constant."""
    dispatch = result["dispatch"]
    summary = {"date": date, "block": block, "price_date": price_date,
               "optimal": bool(dispatch["optimal"]),
               "solver_status": dispatch["solver_status"],
               "solution_count": dispatch["solution_count"],
               "runtime_seconds": dispatch.get("runtime_seconds"),
               "price_qualification": result["price_qualification"],
               "qualified_for_settlement": bool(result["qualified_for_settlement"]),
               "voltage_band_ok": False, "feasible": False,
               "failure": result.get("failure", "")}
    if dispatch.get("voltage_sq") is not None:
        volts = np.sqrt(np.asarray(dispatch["voltage_sq"]))
        low, high = day.network.voltage_min, day.network.voltage_max
        summary.update(voltage_min_pu=float(volts.min()), voltage_max_pu=float(volts.max()),
                       voltage_band_ok=bool(volts.min() >= low - 1e-9
                                            and volts.max() <= high + 1e-9),
                       voltage_at_lower_bound=bool(volts.min() <= low + 1e-6),
                       voltage_at_upper_bound=bool(volts.max() >= high - 1e-6))
    if dispatch.get("pcc_mw") is not None:
        summary["pcc_peak_mw"] = float(np.max(dispatch["pcc_mw"]))
    if "clearing_objective" in dispatch:
        summary["clearing_objective"] = dispatch["clearing_objective"]
    summary["feasible"] = summary["optimal"] and summary["voltage_band_ok"]
    return summary


def block_counts(rows):
    return {block: {"days": len(subset),
                    "optimal": sum(r["optimal"] for r in subset),
                    "band_ok": sum(r["voltage_band_ok"] for r in subset),
                    "feasible": sum(r["feasible"] for r in subset),
                    "qualified_for_settlement": sum(r["qualified_for_settlement"]
                                                    for r in subset),
                    "price_qualification": dict(Counter(r["price_qualification"]
                                                        for r in subset))}
            for block in BLOCKS
            for subset in [[r for r in rows if r["block"] == block]]}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--slack", type=float, default=1.,
                        help="Substation voltage the market holds, in pu.")
    parser.add_argument("--capacitors", nargs="+", default=[], metavar="BUS:MVAR",
                        help="Fixed compensation the feeder declares, as BUS:MVAR pairs.")
    parser.add_argument("--generator-q", type=float, default=0., metavar="MVAR",
                        help="Reactive capability the market declares for every generator.")
    parser.add_argument("--storage-energy", type=float, default=None, metavar="MWH",
                        help="Energy capacity the market declares for every storage unit.")
    parser.add_argument("--no-probes", dest="probes", action="store_false",
                        help="Skip the price probes; qualification then stays NOT_TESTED.")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("A finite positive solve budget is required")
    if not np.isfinite(args.slack) or not .9 < args.slack <= 1.07:
        raise ValueError("The declared substation voltage must lie inside the band")
    if not np.isfinite(args.generator_q) or args.generator_q < 0:
        raise ValueError("The declared reactive capability must be finite and non-negative")
    if args.storage_energy is not None and (not np.isfinite(args.storage_energy)
                                            or args.storage_energy <= 0):
        raise ValueError("The declared storage energy must be finite and positive")
    try:
        capacitors = tuple((int(bus), float(mvar))
                           for bus, mvar in (item.split(":") for item in args.capacitors))
    except ValueError:
        raise ValueError("A capacitor must be declared as BUS:MVAR") from None
    output = Path(args.out).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Evidence output must stay within the repository")
    block_of, price_of = library_dates()
    dates = args.dates or sorted(block_of)
    unknown = [date for date in dates if date not in block_of]
    if unknown:
        raise ValueError(f"Dates outside the v3 library: {unknown}")
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((V3 / "manifest.json").read_text(encoding="utf-8"))
    hashes = lambda: {name: sha(V3 / name) for name in manifest["outputs"]}
    before = hashes()
    band = None
    rows = []
    for date in dates:
        day, context = build_day(date)
        day = dataclasses.replace(day, network=dataclasses.replace(day.network,
                                                                  slack_vm_pu=args.slack,
                                                                  shunts=capacitors),
                                  generators=tuple(dataclasses.replace(
                                      g, reactive_limit_mvar=args.generator_q)
                                      for g in day.generators),
                                  storage=tuple(dataclasses.replace(
                                      u, energy_mwh=args.storage_energy
                                      if args.storage_energy is not None else u.energy_mwh)
                                      for u in day.storage))
        if band is None:
            band = [day.network.voltage_min, day.network.voltage_max]
            # Declared before the first solve, so the rule cannot follow the result.
            write_json(output / "rule.json", {
                "declared_before_first_solve": True,
                "feasible": ("exact complementary dispatch optimal and every bus-period "
                             "inside the network's own voltage band"),
                "band": band,
                "slack_vm_pu": args.slack,
                "shunts_mvar": [list(shunt) for shunt in day.network.shunts],
                "generator_q_mvar": args.generator_q,
                "storage_energy_mwh": [u.energy_mwh for u in day.storage],
                "generators": [[g.name, g.bus, g.reactive_limit_mvar]
                               for g in day.generators],
                "qualified_for_settlement": ("separate and stricter: physical audits, "
                                             "idle-direction stability and smooth prices"),
                "clearing": f"clear_day(price_probes={args.probes}) on build_day(date)"})
        result = clear_day(day, seconds=args.seconds, price_probes=args.probes,
                           log_path=output / f"{date}.log")
        archive_result(output / date, result)
        rows.append(row(date, block_of[date], context["price_date"], day, result))
        write_csv(output / "summary.csv", rows)
        print(json.dumps({k: v for k, v in rows[-1].items()
                          if k in {"date", "block", "optimal", "voltage_min_pu",
                                   "feasible", "qualified_for_settlement", "failure"}}),
              flush=True)

    write_json(output / "summary.json", {
        "rule_file": "rule.json",
        "cells": "one clearing per library day, price probes enabled",
        "counts": block_counts(rows),
        "runtime_seconds_total": sum(r["runtime_seconds"] or 0 for r in rows),
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in
                   [V3 / "manifest.json", V3 / "split.json", V3 / "bus_active_load_mw.npz",
                    V3 / "bus_reactive_load_mvar.npz", ROOT / "src/run_local_da.py",
                    ROOT / "src/local_da_clearing.py", ROOT / "src/local_da_contracts.py"]},
        "library_hashes_before": before,
        "library_hashes_after": hashes(),
        "library_unchanged": before == hashes()})


if __name__ == "__main__":
    main()
