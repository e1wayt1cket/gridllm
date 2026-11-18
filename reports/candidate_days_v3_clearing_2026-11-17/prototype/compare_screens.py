"""Compare two screened evidence directories on feasibility, voltage and prices."""

import csv
import sys
from collections import Counter
from pathlib import Path

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np

from run_local_da import PRICE_CURVES


def load(directory):
    rows = list(csv.DictReader((directory / "summary.csv").open(encoding="utf-8")))
    prices = np.load(PRICE_CURVES)
    for row in rows:
        with np.load(directory / f"{row['date']}.npz") as archive:
            row["lmp"] = archive["result/pricing/lmp_cny_mwh"] if "result/pricing/lmp_cny_mwh" in archive else None
            row["external"] = prices[f"south/{row['price_date']}"]
    return rows


def report(directory):
    rows = load(directory)
    print(f"\n{directory.name}: {len(rows)} days, "
          f"{sum(r['feasible'] == 'True' for r in rows)} feasible")
    blocks = Counter(r["block"] for r in rows)
    feasible = Counter(r["block"] for r in rows if r["feasible"] == "True")
    print("  per block:", {b: f"{feasible[b]}/{blocks[b]}" for b in sorted(blocks)})
    print("  pinned at the lower bound:", sum(r["voltage_at_lower_bound"] == "True" for r in rows),
          "| pinned at the upper bound:", sum(r["voltage_at_upper_bound"] == "True" for r in rows))
    print("  price qualification:", dict(Counter(r["price_qualification"] for r in rows)))
    ratios, hot = [], 0
    for row in rows:
        if row["lmp"] is None:
            continue
        ratios.append(row["lmp"].max() / row["external"].max())
        hot += int((row["lmp"] > 3 * row["external"][:, None]).sum())
    ratios = np.array(ratios)
    print(f"  day-max LMP / day-max external: median {np.median(ratios):.2f}, "
          f"max {ratios.max():.2f}, days above 2x {(ratios > 2).sum()}")
    print(f"  bus-periods above 3x external: {hot}")


def main():
    roots = [Path(arg) for arg in sys.argv[1:]]
    for directory in roots:
        report(directory)


if __name__ == "__main__":
    main()
