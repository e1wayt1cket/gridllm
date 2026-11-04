# build_common_split.py
"""A split both datasets can be compared on, without leaking a test day.

The per-dataset splits written earlier were drawn independently, so a NEAR test
day can sit in FULL96's training set. That makes a cross-dataset comparison
unfair in a way no amount of reporting fixes. This builds one assignment over
pair records and gives every common record the same label in both datasets:

  * a common test record is in neither dataset's training or validation split;
  * a common validation record is in neither dataset's training split;
  * FULL96-only records go to FULL96's training split and never to the test, so
    the common test set is defined only by records both datasets hold;
  * within each dataset no load date and no price date crosses a split -- which
    follows from pairing uniqueness and is checked, not assumed.

Seasonal stratification is kept and the national-day block stays whole. The
per-dataset splits are left on disk unchanged; this file is the protocol for
any FULL96-vs-NEAR comparison, and `datasets.json` says so.

Usage:
    PYTHONPATH=src python src/build_common_split.py
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
from typing import Dict, List, Sequence, Tuple

import pair_v2_cy_jn_2025 as pj

FULL = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
NEAR = "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
DATASETS = "data/processed/v2_cy_jn_2025/datasets.json"
OUT = "data/processed/v2_cy_jn_2025/common_split.json"
TEST_INDEX = "data/processed/v2_cy_jn_2025/common_test_index.csv"
REPORT = "reports/v2_cy_jn_2025/common_split_audit.md"
SEED = 20261101
SPLITS = ("train", "validation", "test", "holiday_block")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_id(row: dict) -> str:
    return f"{row['load_date']}|{row['price_date']}"


def _gap(row: dict) -> int:
    target = Date.fromisoformat(row["load_date"]).replace(year=2025)
    return abs((Date.fromisoformat(row["price_date"]) - target).days)


def _labels(load_dates: Sequence[str], split: dict,
           id_by_load: Dict[str, str]) -> Dict[str, str]:
    return {id_by_load[d]: key for key in SPLITS for d in split[key]}


def invariants(full: Dict[str, str], near: Dict[str, str],
               common_test: Sequence[str]) -> Dict[str, bool]:
    full_test = {i for i, label in full.items() if label == "test"}
    near_test = {i for i, label in near.items() if label == "test"}
    full_train = {i for i, label in full.items() if label == "train"}
    near_train = {i for i, label in near.items() if label == "train"}
    full_val = {i for i, label in full.items() if label == "validation"}
    near_val = {i for i, label in near.items() if label == "validation"}
    return {
        "common_test_equals_full_test": set(common_test) == full_test,
        "common_test_equals_near_test": set(common_test) == near_test,
        "common_test_not_in_full_train_or_val": not (
            set(common_test) & (full_train | full_val)),
        "common_test_not_in_near_train_or_val": not (
            set(common_test) & (near_train | near_val)),
        "common_val_not_in_full_train": not (full_val & full_train),
        "common_val_not_in_near_train": not (near_val & near_train),
        "full_partition_is_disjoint": (
            len(full_train) + len(full_val) + len(full_test)
            + len({i for i, l in full.items() if l == "holiday_block"})
            == len(full)),
        "near_partition_is_disjoint": (
            len(near_train) + len(near_val) + len(near_test)
            + len({i for i, l in near.items() if l == "holiday_block"})
            == len(near)),
    }


def _date_uniqueness(rows: Sequence[dict], labels: Dict[str, str]) -> bool:
    """No load or price date may appear under two non-holiday split labels."""
    by_label: Dict[str, set] = collections.defaultdict(set)
    for row in rows:
        by_label[labels[pair_id(row)]].add(row["load_date"])
        by_label[labels[pair_id(row)]].add(row["price_date"])
    keys = [k for k in SPLITS if k != "holiday_block"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if by_label[a] & by_label[b]:
                return False
    return True


def _write_test_index(path: Path, rows: Sequence[dict],
                      labels: Dict[str, str]) -> Path:
    columns = ["pair_id", "load_date", "price_date", "group", "season",
               "month_day_gap_days", "load_curve_sha256",
               "price_curve_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if labels[pair_id(row)] != "test":
                continue
            writer.writerow({
                "pair_id": pair_id(row),
                "load_date": row["load_date"],
                "price_date": row["price_date"],
                "group": row["group"],
                "season": row["season"],
                "month_day_gap_days": _gap(row),
                "load_curve_sha256": row["load_curve_sha256"],
                "price_curve_sha256": row["price_curve_sha256"],
            })
    return path


def build_report(*, path: Path, common_rows: Sequence[dict],
                 full_only: Sequence[dict], labels_full: Dict[str, str],
                 labels_near: Dict[str, str], checks: Dict[str, bool],
                 seed: int, test_index: Path, split_path: Path,
                 command: str) -> Path:
    def _counts(labels, rows):
        by = collections.Counter(labels[pair_id(r)] for r in rows)
        return {k: by.get(k, 0) for k in SPLITS}

    common_counts = _counts(labels_full,
                            [r for r in common_rows])
    full_counts = _counts(labels_full, common_rows + list(full_only))
    near_counts = _counts(labels_near, common_rows)
    by_season = collections.defaultdict(lambda: collections.Counter())
    for row in common_rows:
        by_season[row["group"]][labels_full[pair_id(row)]] += 1
    only = collections.Counter(
        (r["group"], labels_full[pair_id(r)]) for r in full_only)

    lines = [
        "# Common split audit (V2-CY-JN-2025)",
        "",
        "The protocol for any FULL96-vs-NEAR comparison. Every pair record "
        "common to the two datasets carries the **same** split label in both, "
        "so the test set is the same days for each. The per-dataset splits "
        "written earlier remain on disk but are not comparable across "
        "datasets; `datasets.json` names this file as the comparison "
        "protocol.",
        "",
        f"- Seed: `{seed}`",
        "- Pair key: `load_date|price_date`; load dates and price dates are "
        "unique within each dataset",
        f"- Common records: {len(common_rows)}; FULL96-only: "
        f"{len(full_only)}",
        "",
        "## Counts",
        "",
        "| view | train | validation | test | holiday | total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, counts, total in (
            ("common (both datasets)", common_counts, len(common_rows)),
            ("FULL96", full_counts, len(common_rows) + len(full_only)),
            ("NEAR", near_counts, len(common_rows))):
        lines.append(f"| {name} | {counts['train']} | "
                     f"{counts['validation']} | {counts['test']} | "
                     f"{counts['holiday_block']} | {total} |")
    lines += [
        "",
        f"The **common test set has {common_counts['test']} days**, and it is "
        "the same list for both datasets: FULL96's test split equals NEAR's. "
        "FULL96's validation split is also exactly the common validation "
        "split.",
        "",
        "## Seasonal composition of the common split",
        "",
        "| group | train | validation | test | holiday |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in pj.GROUPS:
        c = by_season.get(group, collections.Counter())
        lines.append(f"| {group} | {c['train']} | {c['validation']} | "
                     f"{c['test']} | {c['holiday_block']} |")
    lines += [
        "",
        "## FULL96-only records",
        "",
        "These are the long-distance pairs NEAR excludes. They are placed in "
        "FULL96's training split only, so they cannot enter the common test "
        "or validation set.",
        "",
        "| group | split | records |",
        "|---|---|---:|",
    ]
    for (group, label), count in sorted(only.items()):
        lines.append(f"| {group} | {label} | {count} |")
    lines += [
        "",
        "## Invariants",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for name, ok in checks.items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        "## Use",
        "",
        "Any FULL96-vs-NEAR performance comparison must report **per-day "
        "paired results on the common test set**, using the same days for "
        "both. The two datasets' original test splits may be shown as "
        "supplementary context only, and cannot be the main evidence for a "
        "filtering effect: they are not the same days and not independent.",
        "",
        f"- Protocol: `{split_path.as_posix()}`",
        f"- Per-day test index: `{test_index.as_posix()}`",
        f"- Command: `{command}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", default=FULL)
    parser.add_argument("--near", default=NEAR)
    parser.add_argument("--datasets", default=DATASETS)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--test-index", default=TEST_INDEX)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    full_rows = _read_csv(root / args.full)
    near_rows = _read_csv(root / args.near)

    full_ids = [pair_id(r) for r in full_rows]
    near_ids = [pair_id(r) for r in near_rows]
    if len(set(full_ids)) != len(full_ids):
        raise SystemExit("FULL96 has duplicate pair records")
    if len(set(near_ids)) != len(near_ids):
        raise SystemExit("NEAR has duplicate pair records")
    missing = set(near_ids) - set(full_ids)
    if missing:
        raise SystemExit(f"NEAR has {len(missing)} records not in FULL96")

    by_id = {pair_id(r): r for r in full_rows}
    common_rows = [by_id[i] for i in near_ids]          # NEAR order
    full_only = [r for r in full_rows if pair_id(r) not in set(near_ids)]

    # Common labels: one assignment, used by both datasets.
    common_pairs = [pj.Pair(r["load_date"], r["price_date"], r["group"],
                            r["pairing_type"], r["pair_reason"])
                    for r in common_rows]
    common_split = pj.split_pairs(common_pairs, seed=args.seed)
    id_by_load = {r["load_date"]: pair_id(r) for r in common_rows}
    common_labels = _labels([r["load_date"] for r in common_rows],
                            common_split, id_by_load)
    common_ids = {key: [id_by_load[d] for d in common_split[key]]
                  for key in SPLITS}

    # FULL96-only records go to FULL96's training split.
    full_labels = dict(common_labels)
    for row in full_only:
        full_labels[pair_id(row)] = "train"
    near_labels = dict(common_labels)

    common_test = list(common_ids["test"])
    checks = invariants(full_labels, near_labels, common_test)
    checks["full_unique_dates"] = _date_uniqueness(full_rows, full_labels)
    checks["near_unique_dates"] = _date_uniqueness(near_rows, near_labels)
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit(f"common split invariants failed: {failed}")

    def _view(labels, rows):
        return {key: sorted(i for r in rows
                            for i in [pair_id(r)]
                            if labels[i] == key)
                for key in SPLITS}

    payload = {
        "protocol": "FULL96 and NEAR comparison split",
        "purpose": "comparable train/validation/test assignment across the "
                   "two datasets",
        "kind": "cross_year_controlled_simulation",
        "seed": args.seed,
        "pair_key": "load_date|price_date",
        "common": common_ids,
        "full96_only": _view(full_labels, full_only)["train"],
        "full96": _view(full_labels, full_rows),
        "near": _view(near_labels, near_rows),
        "counts": {
            "common": {k: len(common_ids[k]) for k in SPLITS},
            "full96": {k: len(_view(full_labels, full_rows)[k])
                       for k in SPLITS},
            "near": {k: len(_view(near_labels, near_rows)[k])
                     for k in SPLITS},
        },
        "invariants": checks,
        "inputs": {
            "full_pairing": {"path": args.full,
                             "sha256": _sha256(root / args.full)},
            "near_pairing": {"path": args.near,
                             "sha256": _sha256(root / args.near)},
        },
        "note": "per-dataset splits written earlier are not comparable across "
                "datasets; use this file for any FULL96-vs-NEAR comparison",
    }
    out_path = root / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    test_index = _write_test_index(root / args.test_index, full_rows,
                                   full_labels)

    report = build_report(
        path=root / args.report, common_rows=common_rows, full_only=full_only,
        labels_full=full_labels, labels_near=near_labels, checks=checks,
        seed=args.seed, test_index=test_index.relative_to(root),
        split_path=out_path.relative_to(root), command=" ".join(sys.argv))

    datasets_path = root / args.datasets
    datasets = json.loads(datasets_path.read_text(encoding="utf-8"))
    datasets["common_split"] = {
        "path": args.out,
        "sha256": _sha256(out_path),
        "seed": args.seed,
        "counts": payload["counts"],
        "test_index": args.test_index,
        "test_index_sha256": _sha256(test_index),
        "supersedes": "the per-dataset splits for any cross-dataset "
                      "comparison; those remain the datasets' own records",
        "report": args.report,
    }
    datasets_path.write_text(json.dumps(datasets, indent=2), encoding="utf-8")

    print(f"common        {payload['counts']['common']}")
    print(f"FULL96        {payload['counts']['full96']}")
    print(f"NEAR          {payload['counts']['near']}")
    print(f"full96-only   {len(full_only)} -> train")
    print(f"invariants    all PASS")
    print(f"wrote {out_path}")
    print(f"wrote {test_index}")
    print(f"wrote {datasets_path}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
