"""What the fleet's own reactive capability would do to the voltage corner.

The clearing model already prices generator reactive power: `Generator` carries
`reactive_limit_mvar` and the reactive balance includes it. This probe sets that
limit from a power factor on each unit's nameplate and clears the same days.
"""

import dataclasses

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np

from local_da_clearing import clear_day
from run_local_da import build_day

DAYS = ["2022-10-14", "2021-09-20", "2021-06-04", "2021-10-14"]
CASES = [("Q = 0", None), ("PF 0.95", .95), ("PF 0.90", .90), ("PF 0.85", .85)]


def with_reactive_capability(day, power_factor):
    """A constant limit at nameplate times tan(acos(pf)), independent of output."""
    tan = float(np.tan(np.arccos(power_factor)))
    generators = tuple(dataclasses.replace(g, reactive_limit_mvar=tan * float(np.max(g.available_mw)))
                       for g in day.generators)
    return dataclasses.replace(day, generators=generators)


def main():
    print("day         case      Vmin   bus-periods at the bound   LMP max   cells>3x   objective")
    for date in DAYS:
        for label, power_factor in CASES:
            day, _ = build_day(date)
            if power_factor is not None:
                day = with_reactive_capability(day, power_factor)
            result = clear_day(day, seconds=120, price_probes=False)
            dispatch = result["dispatch"]
            if dispatch.get("voltage_sq") is None:
                print(f"{date}  {label:9s}  -- solver status {dispatch['solver_status']}, infeasible")
                continue
            volts = np.sqrt(dispatch["voltage_sq"])
            lmp = (result.get("pricing") or {}).get("lmp_cny_mwh")
            pinned = int((volts <= day.network.voltage_min + 1e-6).sum())
            hot = int((lmp > 3 * day.external_price_cny_mwh[:, None]).sum()) if lmp is not None else -1
            peak = lmp.max() if lmp is not None else float("nan")
            print(f"{date}  {label:9s} {volts.min():.4f}   {pinned:4d}/{volts.size}"
                  f"            {peak:7.1f}  {hot:8d}   {dispatch['clearing_objective']:.0f}")
        print()


if __name__ == "__main__":
    main()
