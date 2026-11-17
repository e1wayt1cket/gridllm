# build_v2_cy_jn_datasets.py
"""Register the two datasets of V2-CY-JN-2025 and their independent splits.

`JN-Base-FULL96` is the committed 96-pair table. `JN-Base-NEAR` keeps the pairs
whose `price_date` is at most 14 days from the load day's own 2025 month-day.
The 14-day rule is a **distance sensitivity filter**, not a validated
scientific boundary: it exists to ask whether the far pairs drive a result, and
nothing about 14 is established by the source.

The NEAR table is a subset of FULL96, so its days also appear in FULL96. Its
split is drawn independently -- a different seed, its own stratification -- and
is not a relabelling of FULL96's. A NEAR training day can therefore sit in a
FULL96 test split; the two splits are independent *assignments*, not
independent *data*, and no result may cite them as separate evidence.

This writes tables and manifests only. It does not clear a market and does not
train.

Usage:
    PYTHONPATH=src python src/build_v2_cy_jn_datasets.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import pair_v2_cy_jn_2025 as pj

PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
SPLIT = "data/processed/v2_cy_jn_2025/split.json"
MANIFEST = "data/processed/v2_cy_jn_2025/pairing_manifest.json"
DATA_DIR = "data/processed/v2_cy_jn_2025"
NEAR_DIR = "data/processed/v2_cy_jn_2025/near"
REPORT = "reports/v2_cy_jn_2025/full96_near_sensitivity.md"
DATASETS = "data/processed/v2_cy_jn_2025/datasets.json"

FULL_ID = "JN-Base-FULL96"
NEAR_ID = "JN-Base-NEAR"
NEAR_THRESHOLD = 14
NEAR_SEED = 20261031
SPLITS = ("train", "validation", "test", "holiday_block")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gap(load_date: str, price_date: str) -> int:
    target = Date.fromisoformat(load_date).replace(year=2025)
    return abs((Date.fromisoformat(price_date) - target).days)


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pj.PAIRING_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _is_weekend(day: str) -> bool:
    return Date.fromisoformat(day).weekday() >= 5


def distance_stats(rows: Sequence[dict]) -> dict:
    gaps = [_gap(r["load_date"], r["price_date"]) for r in rows]
    return {
        "pairs": len(rows),
        "mean_days": round(float(np.mean(gaps)), 2),
        "median_days": float(np.median(gaps)),
        "max_days": int(max(gaps)),
        "over_14": sum(1 for g in gaps if g > 14),
        "over_30": sum(1 for g in gaps if g > 30),
        "same_month_day": sum(1 for r, g in zip(rows, gaps) if g == 0),
        # The recorded class, not a raw weekday-label comparison: a
        # national-day pair legitimately carries a different weekday label,
        # because inside the holiday window that label is not what is matched.
        "type_mismatch": sum(
            1 for r in rows if "mismatch" in r.get("pairing_type", "")),
    }


def group_counts(rows: Sequence[dict]) -> Dict[str, int]:
    return dict(sorted(collections.Counter(r["group"] for r in rows).items()))


def split_counts(split: dict) -> Dict[str, int]:
    return {key: len(split[key]) for key in SPLITS}


def split_overlaps(a: dict, b: dict) -> Dict[str, Dict[str, int]]:
    return {x: {y: len(set(a[x]) & set(b[y])) for y in SPLITS}
            for x in SPLITS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", default=PAIRING)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--near-dir", default=NEAR_DIR)
    parser.add_argument("--datasets", default=DATASETS)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--threshold", type=int, default=NEAR_THRESHOLD)
    parser.add_argument("--seed", type=int, default=NEAR_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    pairing_path = root / args.pairing
    rows = _read_csv(pairing_path)
    full_split = json.loads((root / args.split).read_text(encoding="utf-8"))

    # ---- FULL96 ----------------------------------------------------------
    full_stats = distance_stats(rows)
    full_groups = group_counts(rows)
    full_manifest = json.loads(
        (root / args.manifest).read_text(encoding="utf-8"))
    full_manifest.update({
        "dataset_id": FULL_ID,
        "dataset_role": "primary candidate, all 96 pairs",
        "distance_rule": None,
        "distance_stats": full_stats,
        "group_counts": full_groups,
        "split": {
            "seed": full_split["seed"],
            "counts": split_counts(full_split),
            "by_group": full_split["by_group"],
            "file": args.split,
        },
        "near_counterpart": f"{args.near_dir}/pairing_manifest.json",
        "datasets_index": args.datasets,
    })
    (root / args.manifest).write_text(
        json.dumps(full_manifest, indent=2), encoding="utf-8")

    # ---- NEAR ------------------------------------------------------------
    near = [r for r in rows
            if _gap(r["load_date"], r["price_date"]) <= args.threshold]
    near_pairs = [pj.Pair(r["load_date"], r["price_date"], r["group"],
                          r["pairing_type"], r["pair_reason"]) for r in near]
    near_split = pj.split_pairs(near_pairs, seed=args.seed)

    near_dir = root / args.near_dir
    near_csv = _write_csv(near_dir / "load_price_pairing.csv", near)
    near_split_path = near_dir / "split.json"
    near_split_path.write_text(json.dumps(near_split, indent=2),
                               encoding="utf-8")
    near_manifest = {
        "dataset_id": NEAR_ID,
        "dataset_role": "distance sensitivity subset of FULL96",
        "kind": "cross_year_controlled_simulation",
        "not_a_same_period_settlement": True,
        "price_region": "south",
        "price_unit": "CNY/MWh",
        "price_unit_confirmed": False,
        "price_unit_note": "inferred from floor/cap and magnitude; not "
                           "externally confirmed",
        "mv_value_enabled": False,
        "distance_filter": {
            "column": "month_day_gap_days",
            "rule": f"keep price_date within {args.threshold} days of the "
                    "load day's 2025 month-day",
            "threshold_days": args.threshold,
            "status": "sensitivity rule, not a validated scientific boundary "
                      "and not a property of the source",
            "source": "the committed FULL96 pairing; no re-pairing",
        },
        "group_counts": group_counts(near),
        "distance_stats": distance_stats(near),
        "split": {
            "seed": args.seed,
            "counts": split_counts(near_split),
            "by_group": near_split["by_group"],
            "independent_of": f"{FULL_ID} split (different seed)",
            "file": args.near_dir + "/split.json",
        },
        "inputs": {
            "full_pairing": {"path": args.pairing,
                             "sha256": _sha256(pairing_path)},
            "price_file": full_manifest["inputs"]["price_file"],
        },
        "outputs": {
            "pairing_csv": {"path": args.near_dir + "/load_price_pairing.csv",
                            "sha256": _sha256(near_csv)},
            "split": {"path": args.near_dir + "/split.json",
                      "sha256": _sha256(near_split_path)},
        },
    }
    near_manifest_path = near_dir / "pairing_manifest.json"
    near_manifest_path.write_text(json.dumps(near_manifest, indent=2),
                                  encoding="utf-8")

    # ---- index -----------------------------------------------------------
    datasets = {
        "experiment": "V2-CY-JN-2025",
        "datasets": {
            FULL_ID: {
                "table": args.pairing,
                "manifest": args.manifest,
                "split": args.split,
                "sha256": {"table": _sha256(pairing_path)},
                "group_counts": full_groups,
                "distance_stats": full_stats,
                "split_seed": full_split["seed"],
            },
            NEAR_ID: {
                "table": args.near_dir + "/load_price_pairing.csv",
                "manifest": args.near_dir + "/pairing_manifest.json",
                "split": args.near_dir + "/split.json",
                "sha256": {"table": _sha256(near_csv)},
                "group_counts": near_manifest["group_counts"],
                "distance_stats": near_manifest["distance_stats"],
                "split_seed": args.seed,
            },
        },
        "relationship": {
            "near_is_subset_of_full": True,
            "near_days_also_in_full": len(near),
            "cross_dataset_overlap": "every NEAR date is also a FULL96 date; "
                                     "the two splits are independent "
                                     "assignments over the same days and are "
                                     "not independent evidence",
            "split_overlap_full_vs_near": split_overlaps(full_split,
                                                         near_split),
        },
        "distance_filter": {
            "applied_to": FULL_ID,
            "to_make": NEAR_ID,
            "threshold_days": args.threshold,
            "rule": near_manifest["distance_filter"]["rule"],
            "status": near_manifest["distance_filter"]["status"],
        },
    }
    datasets_path = root / args.datasets
    datasets_path.write_text(json.dumps(datasets, indent=2), encoding="utf-8")

    # ---- report ----------------------------------------------------------
    removed = [r for r in rows
               if _gap(r["load_date"], r["price_date"]) > args.threshold]
    removed_groups = dict(sorted(collections.Counter(
        r["group"] for r in removed).items()))
    overlap = split_overlaps(full_split, near_split)
    lines = [
        "# FULL96 and NEAR sensitivity datasets (V2-CY-JN-2025)",
        "",
        "Both datasets are **cross-year controlled simulations**: the load is a "
        "2022 metered day, the price a 2025 Jiangnan day-ahead curve. Neither "
        "is a same-period settlement, and the price unit is an inference "
        "recorded in `data_audit.md`, not an external confirmation.",
        "",
        "## Why not 83 days",
        "",
        "The working name assumed only autumn pairs exceeded 14 days. That is "
        "not what the committed table contains: the filter removes **"
        f"{len(rows) - len(near)} pairs** — "
        + ", ".join(f"{g} {n}" for g, n in removed_groups.items())
        + ". The count is computed from the table, not hardcoded, "
        f"so NEAR has **{len(near)} days**, not 83.",
        "",
        "## Counts",
        "",
        "| group | FULL96 | NEAR | removed |",
        "|---|---:|---:|---:|",
    ]
    for group in pj.GROUPS:
        f = full_groups.get(group, 0)
        n = near_manifest["group_counts"].get(group, 0)
        lines.append(f"| {group} | {f} | {n} | {f - n} |")
    lines += [f"| **total** | **{len(rows)}** | **{len(near)}** | "
              f"**{len(rows) - len(near)}** |", ""]

    lines += [
        "## Seasonal shift",
        "",
        "| group | FULL share | NEAR share | change |",
        "|---|---:|---:|---:|",
    ]
    for group in pj.GROUPS:
        f = full_groups.get(group, 0) / len(rows)
        n = near_manifest["group_counts"].get(group, 0) / len(near)
        lines.append(f"| {group} | {f:.1%} | {n:.1%} | {n - f:+.1%} |")
    lines += [
        "",
        "Summer loses 3 of 21 and autumn 13 of 47, so autumn's share falls. "
        "The filter is not season-balanced; it is a distance rule and the "
        "distance distribution is season-specific.",
        "",
        "## Distance",
        "",
        "| statistic | FULL96 | NEAR |",
        "|---|---:|---:|",
        f"| pairs | {full_stats['pairs']} | "
        f"{near_manifest['distance_stats']['pairs']} |",
        f"| mean (days) | {full_stats['mean_days']:.2f} | "
        f"{near_manifest['distance_stats']['mean_days']:.2f} |",
        f"| median (days) | {full_stats['median_days']:.0f} | "
        f"{near_manifest['distance_stats']['median_days']:.0f} |",
        f"| max (days) | {full_stats['max_days']} | "
        f"{near_manifest['distance_stats']['max_days']} |",
        f"| over 30 days | {full_stats['over_30']} | "
        f"{near_manifest['distance_stats']['over_30']} |",
        f"| same month-day | {full_stats['same_month_day']} | "
        f"{near_manifest['distance_stats']['same_month_day']} |",
        f"| recorded type mismatch | {full_stats['type_mismatch']} | "
        f"{near_manifest['distance_stats']['type_mismatch']} |",
        "",
        "The single recorded type mismatch (2022-08-20 to 2025-08-20) survives "
        "the filter: it is a same-month-day pair at distance 0, and dropping it "
        "would remove an exact date match to keep a constraint the source "
        "cannot satisfy.",
        "",
        "The 14-day rule is a **sensitivity filter**, not a validated boundary: "
        "the source says nothing about 14 days. It exists so a result can be "
        "asked whether it depends on the far pairs; it does not make the NEAR "
        "set the correct one.",
        "",
        "## Removed pairs",
        "",
        "| load_date | price_date | group | gap (days) |",
        "|---|---|---|---:|",
    ]
    for r in sorted(removed, key=lambda r: -_gap(r["load_date"],
                                                 r["price_date"])):
        lines.append(f"| {r['load_date']} | {r['price_date']} | "
                     f"{r['group']} | "
                     f"{_gap(r['load_date'], r['price_date'])} |")
    lines += [
        "",
        "## Status",
        "",
        "**No market was cleared for NEAR in this phase.** The dataset and its "
        "split are prepared; `results.csv` still describes the 180 FULL96 "
        "clears (96 JN-Base and 21 each for C80/C90/C100/NoESS). A NEAR clear "
        "is a separate run and has not been performed.",
        "",
        "**For any comparison, use `common_split.json`, not these splits.** "
        "The per-dataset splits below are each dataset's own record; they are "
        "not comparable across datasets. `common_split.json` gives every "
        "common record one label in both datasets and is the only split a "
        "FULL96-vs-NEAR comparison may use; see `common_split_audit.md`.",
        "",
        "## Splits",
        "",
        "| split | FULL96 | NEAR |",
        "|---|---:|---:|",
    ]
    for key in SPLITS:
        lines.append(f"| {key} | {len(full_split[key])} | "
                     f"{len(near_split[key])} |")
    lines += [
        "",
        f"NEAR uses seed `{args.seed}`, FULL96 uses seed "
        f"`{full_split['seed']}`. Within each dataset the three training "
        "splits and the holiday block are disjoint and cover every day.",
        "",
        "## Cross-dataset overlap",
        "",
        "Every NEAR day is also a FULL96 day. The rows below count dates "
        "shared between a FULL96 split and a NEAR split:",
        "",
        "| FULL96 \\ NEAR | train | validation | test | holiday |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in SPLITS:
        lines.append(f"| {x} | " + " | ".join(
            str(overlap[x][y]) for y in SPLITS) + " |")
    lines += [
        "",
        "Non-zero off-diagonal cells mean a NEAR training day can be a FULL96 "
        "test day, and the reverse. The two splits are independent "
        "**assignments over the same days**, not independent samples.",
        "",
        f"With the committed seeds, NEAR's test split shares "
        f"{overlap['test']['test']} days with FULL96's test split and "
        f"{overlap['test']['train']} with FULL96's training split. A NEAR "
        "test score is therefore out-of-sample within NEAR but in-sample "
        "relative to FULL96's training set. **The two datasets cannot "
        "cross-validate each other**, and a FULL96 result and a NEAR result on "
        "the same day must not be reported as two confirmations.",
        "",
        "## Artifacts",
        "",
        f"- `{args.pairing}`, `{args.manifest}`, `{args.split}` "
        f"({FULL_ID})",
        f"- `{args.near_dir}/load_price_pairing.csv`, "
        f"`{args.near_dir}/pairing_manifest.json`, "
        f"`{args.near_dir}/split.json` ({NEAR_ID})",
        f"- `{args.datasets}` — both datasets, hashes, and this relationship",
    ]
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"FULL96  {len(rows)}  {full_groups}")
    print(f"NEAR    {len(near)}  {near_manifest['group_counts']}  "
          f"(seed {args.seed}, {near_manifest['split']['counts']})")
    print(f"removed {len(rows) - len(near)}: {removed_groups}")
    print(f"wrote {near_csv}")
    print(f"wrote {near_manifest_path}")
    print(f"wrote {datasets_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
