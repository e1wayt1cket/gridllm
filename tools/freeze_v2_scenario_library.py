# freeze_v2_scenario_library.py
"""Freeze the V2 seasonal scenario library after the support screen passes.

The frozen object is six **different** fixed user sets (four seasons plus 2022
Spring Festival and National Day), their representative dates, and one network
support assumption: `line50_q2.0` (slack 1.0, uniform line R/X at 0.5x, ideal
local reactive compensation at 2x each bus's load Q). The support is a
diagnostic assumption, not a device design.

The freeze refuses unless the `line50_q2.0` screen has results for every group
and every representative day, and every one passes. It does not start training:
the real dated price gate is still unmet.

This writes new V2 artifacts only; the V1 single-day benchmark and the proposed
V2 files are left untouched.

Usage:
    PYTHONPATH=src python src/freeze_v2_scenario_library.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

SUPPORT_SETTING = "line50_q2.0"


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict],
               columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, *, manifest: dict, groups: Sequence[dict],
                  command: str, artifacts: Dict[str, str]) -> Path:
    support = manifest["network_support"]
    price = manifest["price_coverage"]
    lines = [
        "# V2 scenario library frozen",
        "",
        "## Status",
        "",
        f"- Version: **{manifest['version']}**",
        f"- Frozen at: `{manifest['frozen_at']}`",
        "- The library contains **six different fixed user sets**, not one "
        "system. A result from spring is not directly comparable to a result "
        "from winter or from a holiday subset.",
        "- V1 is preserved as the historical single-day benchmark.",
        "- Formal training is **not** started.",
        "",
        "## Groups",
        "",
        "| group | type | season | users | buses | industries | common dates | "
        "representative days | median peak MW | /3.715 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(
            f"| {group['group']} | {group['group_type']} | {group['season']} | "
            f"{group['users']} | {group['buses']} | {group['industries']} | "
            f"{group['common_dates']} | "
            f"{len(group['representative_dates'])} | "
            f"{float(group['median_peak_mw']):.3f} | "
            f"{float(group['peak_ratio_to_3_715']):.2f}x |")
    lines += [
        "",
        "## User sets",
        "",
        "The frozen user lists are in the manifest and the frozen CSV. They are "
        "deliberately different; do not read a cross-season difference as a "
        "policy difference without naming the user-set change.",
        "",
        "## Network support assumption",
        "",
        f"- Assumption name: `{support['assumption']}`.",
        f"- Slack voltage: {support['slack_vm_pu']} p.u.",
        f"- Uniform line R/X scale: {support['line_impedance_scale']} "
        "(all in-service lines).",
        f"- Ideal local reactive compensation: "
        f"{support['reactive_fraction_of_load_q']}x each bus's load Q.",
        "- This is a screening assumption; no device rating, cost, control "
        "dynamics or market clearing is modeled. It is not a cost-optimal "
        "network design.",
        f"- Representative days screened: "
        f"{support['screening']['representative_days_checked']}; all pass: "
        f"**{support['screening']['all_pass']}**.",
        "",
        "| group | worst min V | max V | voltage violations | line overloads |",
        "|---|---:|---:|---:|---:|"]
    for group, result in support["screening"]["per_group"].items():
        lines.append(
            f"| {group} | {result['worst_min_v']:.4f} | "
            f"{result['max_v']:.4f} | {result['voltage_violations']} | "
            f"{result['line_overloads']} |")
    lines += [
        "",
        "## Price coverage and training gate",
        "",
        f"- Candidate load days: {price['candidate_days']}",
        f"- Representative days: {price['representative_days']}",
        f"- Candidate days with a real dated price: "
        f"{price['priced_candidates']}",
        f"- Representative days with a real dated price: "
        f"{price['priced_representatives']}",
        f"- Formal training blocked: **{price['training_blocked']}**.",
        f"- Reason: {price['reason']}.",
        "",
        "## Artifacts", "",
        "| path | sha256 |", "|---|---|"]
    for path_text, digest in artifacts.items():
        lines.append(f"| `{path_text}` | `{digest}` |")
    lines += ["", "## Command", "", f"`{command}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the V2 seasonal scenario library.")
    parser.add_argument("--subsets", default="data/processed/v2_seasonal_subsets.csv")
    parser.add_argument("--candidates",
                        default="data/processed/v2_seasonal_candidates.csv")
    parser.add_argument("--support-screen",
                        default="data/processed/v2/v2_voltage_support_screen.csv")
    parser.add_argument("--price-pairing",
                        default="reports/v2_load_price_pairing.csv")
    parser.add_argument("--coverage", default="reports/v2_user_date_coverage.csv")
    parser.add_argument("--core", default="data/processed/v2/v2_provisional_core.json")
    parser.add_argument("--frozen-csv",
                        default="data/processed/v2/v2_seasonal_subsets_frozen.csv")
    parser.add_argument(
        "--manifest", default="data/processed/v2/v2_scenario_library_manifest.json")
    parser.add_argument("--report", default="reports/v2_scenario_library_frozen.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    paths = {name: root / value for name, value in (
        ("subsets", args.subsets), ("candidates", args.candidates),
        ("support", args.support_screen), ("price", args.price_pairing),
        ("coverage", args.coverage), ("core", args.core))}
    for path in paths.values():
        if not path.exists():
            raise SystemExit(f"{path} not found; run the earlier V2 stages")

    subsets = _read_csv(paths["subsets"])
    if len(subsets) != 6 or any(not r.get("user_ids") for r in subsets):
        raise SystemExit(
            "the proposed library must carry six groups with non-empty user "
            "sets; refusing to freeze")
    candidates = _read_csv(paths["candidates"])
    support_rows = [r for r in _read_csv(paths["support"])
                    if r["setting"] == SUPPORT_SETTING]
    if not support_rows:
        raise SystemExit(f"no {SUPPORT_SETTING} rows in the support screen")
    failed = [r for r in support_rows if not _as_bool(r["passes"])]
    if failed:
        raise SystemExit(
            f"{SUPPORT_SETTING} does not pass {len(failed)} representative "
            "days; refusing to freeze")
    groups = sorted({r["group"] for r in support_rows})
    if len(groups) != 6:
        raise SystemExit(
            f"the support screen covers {len(groups)} groups, expected 6")

    reps: Dict[str, List[str]] = collections.defaultdict(list)
    for row in candidates:
        if _as_bool(row["representative"]):
            reps[row["group"]].append(row["date"])
    for row in subsets:
        if not reps.get(row["group"]):
            raise SystemExit(
                f"{row['group']} has no representative dates; refusing to freeze")

    price_rows = _read_csv(paths["price"])
    priced_candidates = sum(1 for r in price_rows if r["price_date"])
    priced_reps = sum(1 for r in price_rows
                      if r["price_date"] and _as_bool(r["representative"]))
    core = json.loads(paths["core"].read_text(encoding="utf-8"))
    raw = core.get("input_sha256", {})
    mapping_path = root / "data/processed/industry_meter_bus_map_v2.csv"
    workbook_path = root / "docs/16类行业用户.xlsx"
    if _sha256(mapping_path) != raw.get("mapping"):
        raise SystemExit("the mapping changed since the V2 audit; refusing to freeze")
    if _sha256(workbook_path) != raw.get("load_workbook"):
        raise SystemExit("the load workbook changed since the V2 audit; refusing to freeze")

    per_group = {}
    for group in groups:
        rows = [r for r in support_rows if r["group"] == group]
        per_group[group] = {
            "dates": len(rows),
            "worst_min_v": min(float(r["min_bus_voltage_pu"]) for r in rows),
            "max_v": max(float(r["max_bus_voltage_pu"]) for r in rows),
            "voltage_violations": sum(
                int(r["periods_voltage_violation"]) for r in rows),
            "line_overloads": sum(int(r["periods_line_overload"])
                                  for r in rows),
        }
    group_payload = {}
    for row in subsets:
        group = row["group"]
        group_payload[group] = {
            "group": group,
            "group_type": row["group_type"],
            "season": row["season"],
            "user_ids": row["user_ids"].split(";"),
            "users": int(row["users"]),
            "bus_ids": [int(b) for b in row["bus_ids"].split(";") if b],
            "industry_names": [x for x in row["industry_names"].split(";") if x],
            "buses": int(row["buses"]),
            "industries": int(row["industries"]),
            "common_dates": int(row["common_dates"]),
            "representative_dates": sorted(reps[group]),
            "median_peak_mw": float(row["median_peak_mw"]),
            "median_energy_mwh": float(row["median_energy_mwh"]),
            "peak_ratio_to_3_715": float(row["peak_ratio_to_3_715"]),
            "top_bus": int(row["top_bus"]),
            "top_bus_energy_share": float(row["top_bus_energy_share"]),
        }
    frozen_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    version_payload = json.dumps({
        "groups": group_payload,
        "support": SUPPORT_SETTING,
        "support_rows": per_group,
    }, sort_keys=True)
    version = "v2-frozen-" + hashlib.sha256(
        version_payload.encode("utf-8")).hexdigest()[:12]
    manifest = {
        "dataset": "v2_seasonal_scenario_library",
        "version": version,
        "status": "frozen",
        "frozen_at": frozen_at,
        "not_a_single_system": True,
        "v1_preserved": True,
        "groups": group_payload,
        "network_support": {
            "assumption": SUPPORT_SETTING,
            "slack_vm_pu": 1.0,
            "line_impedance_scale": 0.5,
            "line_scope": "all in-service lines",
            "reactive_fraction_of_load_q": 2.0,
            "reactive_support": (
                "ideal local shunt at every load bus proportional to its own "
                "load Q; no device rating, cost or control dynamics"),
            "screening": {
                "source": str(paths["support"]),
                "representative_days_checked": len(support_rows),
                "all_pass": True,
                "per_group": per_group,
            },
        },
        "price_coverage": {
            "source": str(paths["price"]),
            "candidate_days": len(price_rows),
            "representative_days": sum(
                1 for r in price_rows if _as_bool(r["representative"])),
            "priced_candidates": priced_candidates,
            "priced_representatives": priced_reps,
            "training_blocked": True,
            "reason": (
                "no real dated day-ahead price curve covers the selected "
                "candidate or representative days; window means are not daily "
                "prices and are not used"),
        },
        "training": {
            "formal_training_started": False,
            "reason": "price coverage gate is not met",
        },
        "fingerprints": {
            "mapping_sha256": raw.get("mapping"),
            "load_workbook_sha256": raw.get("load_workbook"),
            "coverage_csv_sha256": _sha256(paths["coverage"]),
            "subsets_csv_sha256": _sha256(paths["subsets"]),
            "candidates_csv_sha256": _sha256(paths["candidates"]),
            "support_screen_csv_sha256": _sha256(paths["support"]),
            "price_pairing_csv_sha256": _sha256(paths["price"]),
        },
    }
    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                        ensure_ascii=False) + "\n",
                             encoding="utf-8")

    frozen_rows = []
    for row in subsets:
        frozen = dict(row)
        frozen["status"] = "frozen"
        frozen["network_assumption"] = SUPPORT_SETTING
        frozen["frozen_at"] = frozen_at
        frozen_rows.append(frozen)
    frozen_columns = list(subsets[0]) + [
        "network_assumption", "frozen_at"]
    frozen_csv = _write_csv(root / args.frozen_csv, frozen_rows, frozen_columns)

    artifacts = {
        str(paths["subsets"]): _sha256(paths["subsets"]),
        str(paths["candidates"]): _sha256(paths["candidates"]),
        str(paths["support"]): _sha256(paths["support"]),
        str(paths["price"]): _sha256(paths["price"]),
        str(frozen_csv): _sha256(frozen_csv),
        str(manifest_path): _sha256(manifest_path),
    }
    report_path = _write_report(
        root / args.report, manifest=manifest,
        groups=[group_payload[r["group"]] for r in subsets],
        command=" ".join(sys.argv), artifacts=artifacts)
    print(f"frozen {len(subsets)} groups, {len(support_rows)} support rows")
    print(f"version {version}")
    print(f"wrote {frozen_csv}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
