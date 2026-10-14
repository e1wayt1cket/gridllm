# screen_industry_blocks.py
"""Report every metered block's screening outcome, and what each unit counts.

This is evidence, not a verdict. It re-runs the validators `industry_load`
already ships, states the rules and tolerances those validators actually apply
(read from the module, not restated here), and emits one row per
(meter, date) block with the flags it carries and the reason it would be kept
or dropped.

The reason the units are printed next to every count: the load pool's filters
act on three different things -- a *record* is one workbook row, a *block* is
one meter on one date, and the ratio/capacity check is a *meter* verdict. A
figure quoted without its unit is not reproducible, and "N qualified blocks"
has been read as N meters before.

Three readings of "qualified" are emitted side by side, because the project has
not frozen one:

* `qualified_A` -- the block survives the shape pool: it carries none of
  `SHAPE_DROP_FLAGS`. This is the rule in force today and it says nothing about
  the ratio or the capacity.
* `qualified_B` -- `A`, and the block's meter passes the ratio/capacity check.
  The check is per meter and is applied to every block that meter metered.
* `qualified_C` -- `B`, and the block itself carries no `ratio_suspect` or
  `capacity_inconsistent` advisory flag.

Whichever reading is chosen, none of them normalizes, imputes or edits a
reading, and none of them converts to MW.

Usage:
    PYTHONPATH=src python src/screen_industry_blocks.py --out data/processed
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import industry_load as il


def rules_in_force() -> List[Tuple[str, str]]:
    """The rules and tolerances the shipped validators apply, as they apply them."""
    return [
        ("pool drop -- identity", ", ".join(il.EXCLUDING_FLAGS)),
        ("pool drop -- repeated block", il.FLAG_DUPLICATE_ROW),
        ("pool drop -- total", ", ".join(il.SHAPE_DROP_FLAGS)),
        ("advisory, never dropping", ", ".join(il.ADVISORY_FLAGS)),
        ("capacity tolerance", f"implied peak <= {il.CAPACITY_PEAK_TOLERANCE}x "
                               f"运行容量 (peak = max|slot| * 综合倍率 * "
                               f"60/{il.SLOT_MINUTES})"),
        ("ratio outlier factor", f"departs from the industry median by more "
                                 f"than {il.RATIO_OUTLIER_FACTOR}x (descriptive)"),
        ("reliable industry floor", f">= {il.MIN_USERS_FOR_RELIABLE} users and "
                                    f">= {il.MIN_DAYS_FOR_RELIABLE} meter-days"),
    ]


def screen(records: Sequence[il.MeterDay]):
    """One row per (meter, date) block, plus the meter-level ratio verdict."""
    evidence = il.verify_ratios(records)

    # A meter's ratio/capacity verdict is one bit, decided once and applied to
    # every block that meter metered -- never recomputed per block.
    meter_ok: Dict[str, bool] = {}
    for entry in evidence["per_meter"].values():
        meter_ok[entry["meter"]] = (
            float(entry["peak_ratio"]) <= il.CAPACITY_PEAK_TOLERANCE)

    pool_drop = set(il.SHAPE_DROP_FLAGS)
    advisory = set(il.ADVISORY_FLAGS)
    blocks: Dict[Tuple[str, str], dict] = {}
    for md in records:
        key = (md.meter_key, md.date)
        row = blocks.setdefault(key, {
            "meter": md.meter_key, "account": md.account,
            "industry": md.industry, "date": md.date, "flags": set(),
            "in_pool": False,
        })
        row["flags"].update(md.flags)
        if not (set(md.flags) & pool_drop):
            row["in_pool"] = True
    return evidence, meter_ok, blocks


def classify(row: dict, meter_ok: Dict[str, bool], checkable: set) -> dict:
    """The three readings, and the first reason that decides the strictest one."""
    flags = row["flags"]
    a = row["in_pool"]
    m = row["meter"]
    b = a and m in checkable and meter_ok.get(m, False)
    c = b and not (flags & {"ratio_suspect", "capacity_inconsistent"})

    if not a:
        reason = "dropped from the shape pool: " + ";".join(
            sorted(flags & set(il.SHAPE_DROP_FLAGS)))
    elif m not in checkable:
        reason = ("meter not checkable: 综合倍率 or 运行容量 absent or "
                  "unparseable")
    elif not meter_ok.get(m, False):
        reason = ("meter implied peak exceeds "
                  f"{il.CAPACITY_PEAK_TOLERANCE}x its 运行容量")
    elif not c:
        reason = "advisory only: " + ";".join(
            sorted(flags & {"ratio_suspect", "capacity_inconsistent"}))
    else:
        reason = ""
    return {"qualified_A": a, "qualified_B": b, "qualified_C": c,
            "reason": reason, "checkable": m in checkable,
            "meter_ok": meter_ok.get(m) if m in checkable else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen every metered block and count the units separately.")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--csv-name", default="industry_block_screening.csv")
    args = parser.parse_args(argv)

    print("Rules and tolerances in force")
    for name, text in rules_in_force():
        print(f"  {name:<28} {text}")

    raw = list(il.iter_meter_days(args.workbook, args.sheet))
    records = il.flag_records(raw)
    evidence, meter_ok, blocks = screen(records)
    checkable = set(meter_ok)

    print()
    print("Ratio and capacity evidence (per meter, from verify_ratios)")
    print(f"  meters checked                {evidence['meters_checked']}")
    print(f"  meters whose peak fits        {evidence['meters_consistent']}")
    print(f"  meters whose peak contradicts {evidence['meters_inconsistent']}")

    classified = {k: classify(v, meter_ok, checkable) for k, v in blocks.items()}
    print()
    print("Units, counted separately")
    print(f"  records (workbook rows)  {len(records)}")
    print(f"  blocks  (meter x date)   {len(blocks)}")
    print(f"  meters                   {len({v['meter'] for v in blocks.values()})}")
    print(f"  users                    {len({v['account'] for v in blocks.values()})}")
    print(f"  dates                    {len({v['date'] for v in blocks.values()})}")

    print()
    print("Readings of 'qualified' -- none of these is frozen yet")
    for tag, label in (("qualified_A", "A  survives the shape pool"),
                       ("qualified_B", "B  A + its meter passes ratio/capacity"),
                       ("qualified_C", "C  B + the block carries no advisory cap/ratio flag")):
        keys = [k for k, c in classified.items() if c[tag]]
        meters = {blocks[k]["meter"] for k in keys}
        users = {blocks[k]["account"] for k in keys}
        dates = {blocks[k]["date"] for k in keys}
        print(f"  {label}")
        print(f"      blocks {len(keys):<7} meters {len(meters):<5} "
              f"users {len(users):<4} dates {len(dates)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / args.csv_name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["meter_id", "user_id", "industry", "date", "flags",
                         "meter_checkable", "meter_ratio_capacity_ok",
                         "qualified_A", "qualified_B", "qualified_C",
                         "reason_if_not_C"])
        for key in sorted(blocks):
            b, c = blocks[key], classified[key]
            writer.writerow([b["meter"], b["account"], b["industry"], b["date"],
                             ";".join(sorted(b["flags"])), c["checkable"],
                             c["meter_ok"], c["qualified_A"], c["qualified_B"],
                             c["qualified_C"], c["reason"]])
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
