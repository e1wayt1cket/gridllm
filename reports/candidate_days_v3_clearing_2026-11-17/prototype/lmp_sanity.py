"""Summarize the cleared LMPs of the screened days and where they go extreme.

Reads only the archived evidence. The price is the active-balance dual per
period, so the slack bus must reproduce the exogenous price and no price may
be negative; those two are checked. The rest is reported, not gated.
"""

import argparse
import csv
from pathlib import Path

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np

from run_local_da import PRICE_CURVES, ROOT, build_day

EVIDENCE = ROOT / "reports/candidate_days_v3_clearing_2026-11-17/evidence"


def load(evidence=EVIDENCE):
    prices = np.load(PRICE_CURVES)
    records = []
    for row in csv.DictReader((Path(evidence) / "summary.csv").open(encoding="utf-8")):
        with np.load(Path(evidence) / f"{row['date']}.npz") as archive:
            if "result/pricing/lmp_cny_mwh" not in archive:
                records.append((row, None, None, None))
                continue
            lmp = archive["result/pricing/lmp_cny_mwh"]
            volts = np.sqrt(archive["result/dispatch/voltage_sq"])
        records.append((row, lmp, prices[f"south/{row['price_date']}"], volts))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=str(EVIDENCE))
    records = load(parser.parse_args().evidence)
    priced = [(r, l, p, v) for r, l, p, v in records if l is not None]
    print(f"days {len(records)}, with prices {len(priced)}, without "
          f"{[r['date'] for r, l, _, _ in records if l is None]}")
    lmp = np.stack([l for _, l, _, _ in priced])
    ext = np.stack([p for _, _, p, _ in priced])
    volts = np.stack([v for _, _, _, v in priced])
    external = np.repeat(ext[:, :, None], lmp.shape[2], axis=2)

    slack = np.abs(lmp[:, :, 0] - ext)
    flat = lmp.ravel()
    print(f"slack vs external: median |d| {np.median(slack):.4f}, max {slack.max():.4f} CNY/MWh")
    print(f"LMP: min {flat.min():.2f}, median {np.median(flat):.2f}, p99 "
          f"{np.percentile(flat, 99):.2f}, max {flat.max():.2f}; negative share "
          f"{100 * (flat < -1e-6).mean():.4f}%")
    print(f"external: min {ext.min():.0f}, median {np.median(ext):.0f}, max {ext.max():.0f}")
    markup = lmp.mean(axis=(0, 1)) / ext.mean() - 1
    print(f"mean markup over slack by bus: min {100 * markup.min():.2f}% (bus "
          f"{markup.argmin() + 1}), max {100 * markup.max():.2f}% (bus {markup.argmax() + 1})")

    spread = lmp.max(axis=2) - lmp.min(axis=2)
    print(f"cross-bus spread per period: median {np.median(spread):.2f}, p99 "
          f"{np.percentile(spread, 99):.2f}, max {spread.max():.2f}")

    hot = lmp > 3 * external
    hot_period = hot.any(axis=2)
    bound_period = (volts <= 0.93 + 1e-6).any(axis=2)
    print(f"cells above 3x external: {hot.sum()} of {hot.size} "
          f"({100 * hot.mean():.2f}%); periods {hot_period.sum()} of {hot_period.size} "
          f"({100 * hot_period.mean():.2f}%)")
    print(f"hot periods with a bus on 0.93: {100 * (hot_period & bound_period).sum() / hot_period.sum():.1f}%")
    pinned = volts <= 0.93 + 1e-6
    print(f"bus-periods on 0.93: {pinned.sum()} of {pinned.size} "
          f"({100 * pinned.mean():.2f}%); periods {bound_period.sum()} of "
          f"{bound_period.size} ({100 * bound_period.mean():.2f}%)")
    ratio = (lmp / external).max(axis=2)
    for label, mask in (("pinned", bound_period), ("free", ~bound_period)):
        values = ratio[mask]
        print(f"{label}: periods {mask.sum()}, markup median {np.median(values):.2f}, "
              f"p90 {np.percentile(values, 90):.2f}, max {values.max():.2f}, "
              f"share above 1.5x {100 * (values > 1.5).mean():.1f}%")
    hot_ext = external[hot]
    print(f"external at hot cells: median {np.median(hot_ext):.0f}, p90 "
          f"{np.percentile(hot_ext, 90):.0f}; share at or below 60 CNY/MWh "
          f"{100 * (hot_ext <= 60).mean():.1f}%")

    day_max = lmp.max(axis=(1, 2))
    print(f"days with a price above 800: {(day_max > 800).sum()}; above 600: "
          f"{(day_max > 600).sum()}")

    branches = np.array(build_day(priced[0][0]["date"])[0].network.branches)
    drop = lmp[:, :, branches[:, 1]] - lmp[:, :, branches[:, 0]]
    print(f"price falls of more than 20 CNY across a branch: {(drop < -20).sum()} of "
          f"{drop.size} branch-periods, worst {drop.min():.2f}")


if __name__ == "__main__":
    main()
