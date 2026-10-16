# screen_industry_blocks.py
"""Report every metered block's screening outcome, and what each unit counts.

This is evidence, not a verdict. The verdict lives in `industry_load`, and this
script calls it rather than restating it: `qualified_meters` decides the
ratio/capacity gate once per meter, `qualified_blocks` composes it with the
shape pool. Nothing here normalizes, imputes or edits a reading, and nothing
here converts to MW.

**The frozen rule is reading B.** A formal absolute-load block is one that

* survives the shape pool — it carries none of `SHAPE_DROP_FLAGS` — and
* belongs to a meter whose implied peak fits 运行容量 within
  `CAPACITY_PEAK_TOLERANCE`.

Both gates must pass. The meter gate is decided once per meter and applied to
all of its days, so no block is admitted on its own while its meter is out; a
meter whose 综合倍率 or 运行容量 cannot be read is not qualified either, because
missing information is not a pass. A block's own advisory flags
(`ratio_suspect`, `capacity_inconsistent`) do not remove it.

Readings `A` and `C` are printed alongside `B` because they bound it:
`A` is the pool alone, which is what the pipeline did before the gate existed,
and `C` adds a per-block advisory condition the rule above deliberately omits.
Reporting them keeps the cost of either loosening or tightening visible, and
neither may be substituted for `B` without a decision that says so.

Every count is printed with its unit. A *record* is one workbook row, a *block*
is one meter on one date, and the ratio/capacity verdict belongs to a *meter* —
three different things, and a figure quoted without its unit has been read as
the wrong one before.

Usage:
    PYTHONPATH=src python src/screen_industry_blocks.py --out data/processed
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import industry_load as il

#: The reading the formal absolute-load set is built from.
FROZEN_READING = "B"


def rules_in_force() -> List[Tuple[str, str]]:
    """The rules and tolerances the shipped validators apply, as they apply them."""
    return [
        ("pool drop -- identity", ", ".join(il.EXCLUDING_FLAGS)),
        ("pool drop -- repeated block", il.FLAG_DUPLICATE_ROW),
        ("pool drop -- total", ", ".join(il.SHAPE_DROP_FLAGS)),
        ("advisory, never dropping", ", ".join(il.ADVISORY_FLAGS)),
        ("capacity tolerance", f"implied peak <= {il.CAPACITY_PEAK_TOLERANCE}x "
                               f"运行容量 (peak = max|slot| * 综合倍率; a slot "
                               f"is average active power in "
                               f"{il.READING_UNIT}, so no slot rescale)"),
        ("ratio outlier factor", f"departs from the industry median by more "
                                 f"than {il.RATIO_OUTLIER_FACTOR}x (descriptive)"),
        ("reliable industry floor", f">= {il.MIN_USERS_FOR_RELIABLE} users and "
                                    f">= {il.MIN_DAYS_FOR_RELIABLE} meter-days"),
    ]


def rule_inputs() -> dict:
    """Everything the screening decision depends on, in one place.

    The unit basis belongs here as much as the tolerance does: the capacity
    check compares a peak against 运行容量, so a change in what a slot reading
    *is* changes which meters pass. Including it makes the digest move by
    itself when the unit ruling changes, which is what keeps a superseded
    result from being mistaken for a current one.
    """
    return {
        "reading": FROZEN_READING,
        "pool_drop_flags": sorted(il.SHAPE_DROP_FLAGS),
        "advisory_flags": sorted(il.ADVISORY_FLAGS),
        "capacity_peak_tolerance": il.CAPACITY_PEAK_TOLERANCE,
        "ratio_outlier_factor": il.RATIO_OUTLIER_FACTOR,
        "slot_minutes": il.SLOT_MINUTES,
        "reading_unit": il.READING_UNIT,
        "slot_reading_is_average_power": il.READING_UNIT == il.READING_UNIT_KW,
        "mw_per_kw": il.MW_PER_KW,
    }


def rule_version() -> str:
    """A digest over everything the screening decision depends on.

    Any change to a tolerance, a flag list, the slot width or the reading's unit
    moves this digest, so a manifest and a result are tied together without
    anyone remembering to bump a version by hand.
    """
    payload = json.dumps(rule_inputs(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def collect_blocks(records: Sequence[il.MeterDay]) -> Dict[Tuple[str, str], dict]:
    """One row per (meter, date) block, with the flags the block carries."""
    pool_drop = set(il.SHAPE_DROP_FLAGS)
    blocks: Dict[Tuple[str, str], dict] = {}
    for md in records:
        row = blocks.setdefault((md.meter_key, md.date), {
            "meter": md.meter_key, "account": md.account,
            "industry": md.industry, "date": md.date, "flags": set(),
            "in_pool": False,
        })
        row["flags"].update(md.flags)
        if not (set(md.flags) & pool_drop):
            row["in_pool"] = True
    return blocks


def classify(row: dict, verdicts: Dict[str, Dict[str, object]]) -> dict:
    """The three readings for one block, and the first gate that stops it."""
    flags = row["flags"]
    verdict = verdicts.get(row["meter"],
                           {"checkable": False, "qualified": False})
    a = row["in_pool"]
    b = a and bool(verdict["qualified"])
    c = b and not (flags & {"ratio_suspect", "capacity_inconsistent"})

    if not a:
        reason = "dropped from the shape pool: " + ";".join(
            sorted(flags & set(il.SHAPE_DROP_FLAGS)))
    elif not verdict["checkable"]:
        reason = ("meter not checkable: 综合倍率 or 运行容量 absent or "
                  "unparseable")
    elif not verdict["qualified"]:
        reason = ("meter implied peak exceeds "
                  f"{il.CAPACITY_PEAK_TOLERANCE}x its 运行容量")
    elif not c:
        reason = "advisory only: " + ";".join(
            sorted(flags & {"ratio_suspect", "capacity_inconsistent"}))
    else:
        reason = ""
    return {"qualified_A": a, "qualified_B": b, "qualified_C": c,
            "reason": reason, "checkable": bool(verdict["checkable"]),
            "meter_ok": verdict["qualified"] if verdict["checkable"] else None}


def _unit_counts(blocks: dict, classified: dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tag in ("qualified_A", "qualified_B", "qualified_C"):
        for unit, pick in (("blocks", lambda k: k),
                           ("meters", lambda k: blocks[k]["meter"]),
                           ("users", lambda k: blocks[k]["account"]),
                           ("dates", lambda k: blocks[k]["date"])):
            keys = [k for k, c in classified.items() if c[tag]]
            counts[f"{tag}_{unit}"] = len({pick(k) for k in keys})
    return counts


def write_manifest(path: Path, workbook: Path, records: Sequence[il.MeterDay],
                   blocks: dict, classified: dict, csv_path: Path) -> dict:
    """The record that makes the result checkable from the source file alone."""
    counts = _unit_counts(blocks, classified)
    counts.update({
        "records": len(records),
        "blocks_total": len(blocks),
        "meters_total": len({b["meter"] for b in blocks.values()}),
        "users_total": len({b["account"] for b in blocks.values()}),
        "dates_total": len({b["date"] for b in blocks.values()}),
    })
    csv_digest = il.file_sha256(csv_path)
    manifest = {
        "reading": FROZEN_READING,
        "screening_rule_version": rule_version(),
        "rule_inputs": rule_inputs(),
        "source_workbook": str(workbook),
        "source_workbook_sha256": il.file_sha256(workbook),
        "source_workbook_bytes": Path(workbook).stat().st_size,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_version": f"industry-blocks-{FROZEN_READING}/{csv_digest[:12]}",
        "block_table_sha256": csv_digest,
        "block_table": csv_path.name,
        "counts": counts,
    }
    # ASCII-only, so a reader on any locale can open the manifest. The workbook
    # path carries Chinese, and a tool that opens the file without naming an
    # encoding would otherwise fail on a Chinese Windows default.
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen every metered block and count the units separately.")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--csv-name", default="industry_block_screening.csv")
    parser.add_argument("--manifest-name",
                        default="industry_block_screening_manifest.json")
    args = parser.parse_args(argv)

    print(f"Frozen reading: {FROZEN_READING}   rule version: {rule_version()}")
    for name, text in rules_in_force():
        print(f"  {name:<28} {text}")

    workbook = il.resolve_workbook(args.workbook)
    raw = list(il.iter_meter_days(args.workbook, args.sheet))
    records = il.flag_records(raw)

    verdicts = il.qualified_meters(records)
    blocks = collect_blocks(records)
    classified = {k: classify(v, verdicts) for k, v in blocks.items()}

    checked = [v for v in verdicts.values() if v["checkable"]]
    print()
    print("Ratio and capacity gate (per meter, from qualified_meters)")
    print(f"  meters total                  {len(verdicts)}")
    print(f"  meters checked                {len(checked)}")
    print(f"  meters whose peak fits        {sum(1 for v in checked if v['qualified'])}")
    print(f"  meters whose peak contradicts {sum(1 for v in checked if not v['qualified'])}")
    print(f"  meters the check cannot reach {len(verdicts) - len(checked)}")

    print()
    print("Units, counted separately")
    print(f"  records (workbook rows)  {len(records)}")
    print(f"  blocks  (meter x date)   {len(blocks)}")
    print(f"  meters                   {len({b['meter'] for b in blocks.values()})}")
    print(f"  users                    {len({b['account'] for b in blocks.values()})}")
    print(f"  dates                    {len({b['date'] for b in blocks.values()})}")

    print()
    print("Readings of 'qualified'")
    labels = {
        "qualified_A": "A  survives the shape pool",
        "qualified_B": f"B  A + the meter passes ratio/capacity   <- FROZEN",
        "qualified_C": "C  B + the block carries no advisory cap/ratio flag",
    }
    for tag, label in labels.items():
        keys = [k for k, c in classified.items() if c[tag]]
        print(f"  {label}")
        print(f"      blocks {len(keys):<7} "
              f"meters {len({blocks[k]['meter'] for k in keys}):<5} "
              f"users {len({blocks[k]['account'] for k in keys}):<4} "
              f"dates {len({blocks[k]['date'] for k in keys})}")

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

    manifest = write_manifest(out / args.manifest_name, workbook, records,
                              blocks, classified, path)
    print(f"wrote {out / args.manifest_name}")
    print(f"  source  {manifest['source_workbook_sha256'][:16]}…  "
          f"{manifest['source_workbook_bytes']} bytes")
    print(f"  dataset {manifest['dataset_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
