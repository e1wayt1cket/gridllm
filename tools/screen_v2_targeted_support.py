# screen_v2_targeted_support.py
"""Try a lighter, targeted voltage support on the two weak feeder paths.

The uniform reinforcement that passed the full screen scaled every line's R/X
and added ideal reactive support at 2x each bus's load Q. This script asks
whether a targeted alternative is enough: scale only the lines on the main
feeder end (to buses 1..17) and the branch to bus 32 (to buses 25..32), and add
fixed capacitors at the two far-end buses 17 and 32.

The screen uses each group's worst baseline representative day. Loads are
unchanged; this is a forced-serve power flow, not a market clearing.

Usage:
    PYTHONPATH=src python src/screen_v2_targeted_support.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import build_industry_node_loads as bnl
import scenarios
import screen_v2_physical as sp

TARGET_BUSES = tuple(range(1, 18)) + tuple(range(25, 33))

SETTINGS = (
    # name, line R/X scale on TARGET_BUSES, fixed capacitors (Mvar)
    ("path_line50", 0.50, {17: 0.0, 32: 0.0}),
    ("path_line50_cap11", 0.50, {17: 1.0, 32: 1.0}),
    ("path_line50_cap12", 0.50, {17: 1.0, 32: 2.0}),
    ("path_line70_cap11", 0.70, {17: 1.0, 32: 1.0}),
    ("path_line70_cap12", 0.70, {17: 1.0, 32: 2.0}),
    ("path_line60_cap12", 0.60, {17: 1.0, 32: 2.0}),
)

COLUMNS = (
    "setting", "path_line_scale", "caps_mvar_bus17", "caps_mvar_bus32",
    "group", "date", "peak_load_mw", "converged", "min_bus_voltage_pu",
    "max_bus_voltage_pu", "periods_voltage_violation",
    "periods_line_overload", "max_line_loading", "binding_constraint",
    "passes",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, *, rows: Sequence[dict],
                  command: str, artifacts: Dict[str, str]) -> Path:
    groups = sorted({r["group"] for r in rows})
    passing = collections.defaultdict(set)
    for row in rows:
        if sp._as_bool(row["passes"]):
            passing[row["setting"]].add(row["group"])
    full = [s for s, covered in passing.items() if covered == set(groups)]
    lines = [
        "# V2 targeted voltage-support screen",
        "",
        "## Verdict",
        "",
        "This is a **forced-serve power-flow screen**, not a market clearing. "
        "The measured loads are unchanged. The support is a targeted "
        "alternative to the uniform `line50_q2.0` setting: scale only the main "
        "feeder end (`to_bus` 1..17) and the branch to 32 (`to_bus` 25..32), "
        "plus fixed capacitors at buses 17 and 32.",
        "",
        "Each group is screened on its worst baseline representative day.",
        "",
        "| setting | path R/X | cap bus 17 | cap bus 32 | group | date | "
        "min V | max V | voltage periods | line overloads | passes |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {row['path_line_scale']} | "
            f"{row['caps_mvar_bus17']} | {row['caps_mvar_bus32']} | "
            f"{row['group']} | {row['date']} | "
            f"{row['min_bus_voltage_pu']} | {row['max_bus_voltage_pu']} | "
            f"{row['periods_voltage_violation']} | "
            f"{row['periods_line_overload']} | "
            f"{'yes' if sp._as_bool(row['passes']) else 'no'} |")
    lines += ["", "## Conclusion", ""]
    if full:
        lines.append(
            f"Targeted settings that pass every group: **{full}**. This is a "
            "support-assumption screen, not a cost-optimal design.")
    else:
        lines.append(
            "No targeted setting passes every group. Targeted line R/X "
            "scaling on the two weak paths plus fixed capacitors at buses 17 "
            "and 32 is insufficient; the remaining low voltage is driven by "
            "the branch's active-power drop and native load. The uniform "
            "`line50_q2.0` setting remains the lightest passing option "
            "screened. A voltage regulator/OLTC or a stronger reinforcement "
            "is required for a lighter design.")
    lines += [
        "",
        "## Limits",
        "",
        "- One worst representative day per group, not all 31 days.",
        "- Fixed capacitors are static and unrated; no device cost or control "
        "dynamics are modeled.",
        "- No load is scaled; no market clearing, storage, shedding or price "
        "is modeled.",
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
        description="Targeted voltage-support screen for the V2 seasonal groups.")
    parser.add_argument("--subsets", default="data/processed/v2_seasonal_subsets.csv")
    parser.add_argument("--candidates",
                        default="data/processed/v2_seasonal_candidates.csv")
    parser.add_argument("--users", default="data/processed/v2/v2_user_summary.csv")
    parser.add_argument("--curves", default="data/processed/v2/v2_user_day_curves.npz")
    parser.add_argument("--baseline",
                        default="data/processed/v2/v2_physical_screen.csv")
    parser.add_argument("--out",
                        default="data/processed/v2/v2_targeted_support_screen.csv")
    parser.add_argument("--report", default="reports/v2_targeted_support_screen.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    paths = {name: root / value for name, value in (
        ("subsets", args.subsets), ("candidates", args.candidates),
        ("users", args.users), ("curves", args.curves),
        ("baseline", args.baseline))}
    for path in paths.values():
        if not path.exists():
            raise SystemExit(f"{path} not found; run the earlier V2 stages")
    users = {r["user_id"]: int(r["bus"])
             for r in sp._read_csv(paths["users"])}
    subsets = {r["group"]: r for r in sp._read_csv(paths["subsets"])}
    reps: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in sp._read_csv(paths["candidates"]):
        if sp._as_bool(row["representative"]):
            reps[row["group"]].append(row)
    worst: Dict[str, tuple] = {}
    for row in sp._read_csv(paths["baseline"]):
        if not sp._as_bool(row["converged"]):
            continue
        group = row["group"]
        value = float(row["min_bus_voltage_pu"])
        if group not in worst or value < worst[group][0]:
            worst[group] = (value, row["date"])
    reps = {
        group: [r for r in group_rows
                if r["date"] == worst.get(group, (None, None))[1]]
        for group, group_rows in reps.items()
    }
    curves = np.load(paths["curves"])
    config = scenarios.scenario_model_config(
        "typical_day_industry_subset", T=96, opf_mode="socp")
    rated = bnl.native_bus_load()
    native_shape = bnl.native_shape()
    renewables = sp._renewable_profiles(config)

    rows = []
    for setting, scale, caps in SETTINGS:
        for group, group_rows in reps.items():
            if not group_rows:
                continue
            subset = subsets[group]
            group_users = subset["user_ids"].split(";")
            day = group_rows[0]
            loads, _replace = sp._bus_loads(
                group_users, users, day["date"], curves, rated, native_shape)
            peak = max(float(np.sum([series[t] for series in loads.values()]))
                       for t in range(96))
            result = sp.run_day(
                day["date"], loads, renewables, config,
                line_impedance_scale=scale,
                line_scale_to_buses=TARGET_BUSES,
                fixed_caps_mvar=caps)
            converged = bool(result.get("converged"))
            vviol = int(result.get("periods_voltage_violation") or 0)
            lov = int(result.get("periods_line_overload") or 0)
            rows.append({
                "setting": setting,
                "path_line_scale": scale,
                "caps_mvar_bus17": caps[17],
                "caps_mvar_bus32": caps[32],
                "group": group,
                "date": day["date"],
                "peak_load_mw": peak,
                **{k: v for k, v in result.items() if k != "date"},
                "passes": bool(converged and vviol == 0 and lov == 0),
            })
        print(f"{setting}: {len(rows)} rows", flush=True)
    out_path = _write_csv(root / args.out, rows)
    artifacts = {
        str(paths["subsets"]): _sha256(paths["subsets"]),
        str(paths["candidates"]): _sha256(paths["candidates"]),
        str(paths["users"]): _sha256(paths["users"]),
        str(paths["curves"]): _sha256(paths["curves"]),
        str(paths["baseline"]): _sha256(paths["baseline"]),
        str(out_path): _sha256(out_path),
    }
    report_path = _write_report(
        root / args.report, rows=rows, command=" ".join(sys.argv),
        artifacts=artifacts)
    full = [s for s, *_rest in SETTINGS
            if all(r["passes"] for r in rows if r["setting"] == s)]
    print(f"targeted settings that pass all groups: {full}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
