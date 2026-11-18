"""Re-measure the price qualification of a named sample of library days.

The old screen ran before the market declared its reactive asset, and it left
two different failures: a direction-conditional price solve that returned
suboptimal, and a price solve that was optimal but failed every probe. This
script re-clears the sample on the current default and records both classes
separately, so the two are never added into one "qualified" count.
"""

import argparse
import json
from pathlib import Path

import grid  # noqa: F401  ortools must load before pandas on this platform

from local_da_clearing import clear_day
from run_local_da import build_day, write_json

def measure(date):
    day, context = build_day(date)
    output = clear_day(day, price_probes=True)
    probes = output.get("price_probes") or []
    dispatch = output["dispatch"]
    pricing = output.get("pricing") or {}
    return {
        "date": date,
        "price_date": context["price_date"],
        "dispatch_optimal": bool(dispatch["optimal"]),
        "dispatch_status": int(dispatch["solver_status"]),
        "pricing_optimal": bool(pricing.get("optimal")),
        "pricing_status": (int(pricing["solver_status"])
                           if "solver_status" in pricing else None),
        "objective_difference_cny": output.get("conditional_objective_difference_cny"),
        "objective_tolerance_cny": max(.01, abs(dispatch["clearing_objective"]) * 1e-6),
        "price_qualification": output["price_qualification"],
        "qualified_for_settlement": bool(output["qualified_for_settlement"]),
        "idle_direction_stable": output.get("idle_direction_stable"),
        "failure": output.get("failure"),
        "smooth_all": bool(probes) and all(p["smooth_pass"] for p in probes),
        "bracket_all": bool(probes) and all(p["subgradient_bracket"] for p in probes),
        "probe_count": len(probes),
        "probes": probes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", required=True, help="comma-separated load dates")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    records = [measure(date) for date in args.dates.split(",")]
    write_json(args.out, {"sample": records})
    for record in records:
        print(record["date"], record["price_qualification"],
              record["pricing_status"], record["smooth_all"],
              record["bracket_all"], record["failure"])


if __name__ == "__main__":
    main()
