# screen_v2_voltage_support.py
"""Re-run the V2 seasonal physical screen with voltage-support assumptions.

Three support levers are screened separately and together:

* **Slack voltage** -- an OLTC / feeder-head voltage set-point.
* **Local reactive compensation** -- an ideal shunt proportional to the bus's
  own load Q, representing an SVC or inverter reactive capability. Negative
  shunt q is capacitive injection in pandapower.
* **Line impedance scale** -- a network-reinforcement sensitivity that scales
  every line's R and X. It is not a load change.

The load is never scaled, filled or edited. The screen is still a forced-serve
power flow, not a market clearing, and the support settings are diagnostic
assumptions rather than a frozen V2 network.

Usage:
    PYTHONPATH=src python src/screen_v2_voltage_support.py
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

SETTINGS = (
    # name, slack vm_pu, reactive fraction of load Q, line impedance scale
    ("baseline", 1.0, 0.0, 1.0),
    ("slack_1.05", 1.05, 0.0, 1.0),
    ("q1.0", 1.0, 1.0, 1.0),
    ("q2.5", 1.0, 2.5, 1.0),
    ("line50", 1.0, 0.0, 0.5),
    ("line50_q1.0", 1.0, 1.0, 0.5),
    ("line75_q2.0", 1.0, 2.0, 0.75),
    ("line60_q2.0", 1.0, 2.0, 0.60),
    ("line50_q1.5", 1.0, 1.5, 0.50),
    ("line60_q1.5", 1.0, 1.5, 0.60),
    ("line50_q2.0", 1.0, 2.0, 0.5),
    ("line25_q2.0", 1.0, 2.0, 0.25),
    ("slack1.05_q1.0", 1.05, 1.0, 1.0),
)

COLUMNS = (
    "setting", "slack_vm_pu", "reactive_fraction", "line_impedance_scale",
    "group", "date", "peak_load_mw", "converged", "periods",
    "min_bus_voltage_pu", "max_bus_voltage_pu",
    "periods_voltage_violation", "bus_periods_below_lo",
    "bus_periods_above_hi", "max_line_loading", "periods_line_overload",
    "binding_constraint", "passes",
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


def _write_report(path: Path, *, rows: Sequence[dict], worst_only: bool,
                  command: str, artifacts: Dict[str, str]) -> Path:
    by_setting: Dict[str, List[dict]] = collections.OrderedDict()
    for row in rows:
        by_setting.setdefault(row["setting"], []).append(row)
    lines = [
        "# V2 voltage-support re-screen",
        "",
        "## Verdict",
        "",
        "This is a **forced-serve power-flow screen with voltage-support "
        "assumptions**, not a market clearing. The measured loads are "
        "unchanged; the support is a diagnostic assumption on the network, "
        "not a frozen V2 design.",
        "",
        "The levers are: feeder-head voltage (OLTC), ideal local reactive "
        "compensation proportional to each bus's load Q (SVC/inverter), and a "
        "uniform line R/X scale (network reinforcement). The line scale is a "
        "sensitivity, not a proposal to edit the benchmark data.",
        "",
        ("Each group is screened on its **worst baseline representative day** "
         "(the day with the lowest minimum voltage in the baseline screen)."
         if worst_only else
         "Each group is screened on all of its representative days."),
        "",
        "| setting | slack | Q fraction | line R/X | group | dates | "
        "voltage violations | line overloads | worst min V | max V | passes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for setting, group_rows in by_setting.items():
        groups = collections.OrderedDict()
        for row in group_rows:
            groups.setdefault(row["group"], []).append(row)
        for group, rows_g in groups.items():
            converged = [r for r in rows_g if sp._as_bool(r["converged"])]
            if not converged:
                lines.append(
                    f"| {setting} | {group_rows[0]['slack_vm_pu']} | "
                    f"{group_rows[0]['reactive_fraction']} | "
                    f"{group_rows[0]['line_impedance_scale']} | {group} | "
                    f"{len(rows_g)} | — | — | — | — | no |")
                continue
            vviol = sum(int(r["periods_voltage_violation"])
                        for r in converged)
            lov = sum(int(r["periods_line_overload"]) for r in converged)
            passes = (len(converged) == len(rows_g) and vviol == 0
                      and lov == 0)
            lines.append(
                f"| {setting} | {group_rows[0]['slack_vm_pu']} | "
                f"{group_rows[0]['reactive_fraction']} | "
                f"{group_rows[0]['line_impedance_scale']} | {group} | "
                f"{len(rows_g)} | {vviol} | {lov} | "
                f"{min(float(r['min_bus_voltage_pu']) for r in converged):.4f} | "
                f"{max(float(r['max_bus_voltage_pu']) for r in converged):.4f} | "
                f"{'yes' if passes else 'no'} |")
        lines.append("")
    passing = collections.defaultdict(set)
    groups = sorted({r["group"] for r in rows})
    for row in rows:
        if sp._as_bool(row["passes"]):
            passing[row["setting"]].add(row["group"])
    full = [setting for setting, covered in passing.items()
            if covered == set(groups)]
    lines += ["## Conclusion", ""]
    if full:
        lines.append(
            f"The first setting in the screened ladder that makes every group "
            f"pass is **{full[0]}**. This is not a cost-optimal design: no "
            "device rating, cost or control dynamics are modeled. The network, "
            "the support rating and the V2 subset still have to be frozen "
            "deliberately.")
    else:
        lines.append(
            "No screened setting makes every group pass. Even with line R/X "
            "reduced to a quarter and reactive compensation at 2x the load Q, "
            "the far-end voltage remains the binding constraint. A stronger "
            "network reinforcement, a voltage regulator/series compensation, "
            "or a smaller load is required before these groups can be frozen.")
    lines += [
        "",
        "## Limits",
        "",
        "- The support is ideal and local; no device rating, cost, dispatch "
        "or control dynamics are modelled.",
        "- The screen does not model storage, bids, market clearing, "
        "congestion pricing or shedding.",
        "- No load is scaled; the seasonal subset user sets and measured "
        "curves are unchanged.",
        "- The price blocker remains: no real dated prices exist for these "
        "days, so no training or priced evaluation is possible.",
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
        description="Voltage-support re-screen for the V2 seasonal subsets.")
    parser.add_argument("--subsets", default="data/processed/v2_seasonal_subsets.csv")
    parser.add_argument("--candidates",
                        default="data/processed/v2_seasonal_candidates.csv")
    parser.add_argument("--users", default="data/processed/v2/v2_user_summary.csv")
    parser.add_argument("--curves", default="data/processed/v2/v2_user_day_curves.npz")
    parser.add_argument("--out",
                        default="data/processed/v2/v2_voltage_support_screen.csv")
    parser.add_argument("--report", default="reports/v2_voltage_support_screen.md")
    parser.add_argument(
        "--worst-only", action="store_true",
        help="screen only each group's worst baseline representative day")
    parser.add_argument(
        "--settings", default=None,
        help="comma-separated setting names; default is the full ladder")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    subsets_path = root / args.subsets
    candidates_path = root / args.candidates
    users_path = root / args.users
    curves_path = root / args.curves
    for path in (subsets_path, candidates_path, users_path, curves_path):
        if not path.exists():
            raise SystemExit(f"{path} not found; run the seasonal selection first")
    users = {r["user_id"]: int(r["bus"])
             for r in sp._read_csv(users_path)}
    subsets = {r["group"]: r for r in sp._read_csv(subsets_path)}
    reps: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in sp._read_csv(candidates_path):
        if sp._as_bool(row["representative"]):
            reps[row["group"]].append(row)
    baseline_path = root / "data/processed/v2/v2_physical_screen.csv"
    worst_only = bool(args.worst_only and baseline_path.exists())
    if worst_only:
        worst: Dict[str, tuple] = {}
        for row in sp._read_csv(baseline_path):
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
        print(f"worst-only: {len(worst)} groups, "
              f"{sum(len(v) for v in reps.values())} days", flush=True)
    elif args.worst_only:
        print("baseline physical screen not found; screening all reps",
              flush=True)
    curves = np.load(curves_path)
    config = scenarios.scenario_model_config(
        "typical_day_industry_subset", T=96, opf_mode="socp")
    rated = bnl.native_bus_load()
    native_shape = bnl.native_shape()
    renewables = sp._renewable_profiles(config)

    prepared = {}
    for group, subset in subsets.items():
        if not subset.get("user_ids"):
            continue
        group_users = subset["user_ids"].split(";")
        for day in reps.get(group, []):
            loads, _replace = sp._bus_loads(
                group_users, users, day["date"], curves, rated, native_shape)
            peak = max(float(np.sum([series[t] for series in loads.values()]))
                       for t in range(96))
            prepared[(group, day["date"])] = (loads, peak, day)

    settings = SETTINGS
    if args.settings:
        wanted = {s.strip() for s in args.settings.split(",") if s.strip()}
        settings = tuple(s for s in SETTINGS if s[0] in wanted)
        if not settings:
            raise SystemExit(
                f"no known settings in {sorted(wanted)}; expected "
                f"{[s[0] for s in SETTINGS]}")
    rows = []
    for setting, slack, fraction, scale in settings:
        for (group, date), (loads, peak, day) in prepared.items():
            result = sp.run_day(
                date, loads, renewables, config,
                slack_vm_pu=slack, reactive_fraction=fraction,
                line_impedance_scale=scale)
            converged = bool(result.get("converged"))
            vviol = int(result.get("periods_voltage_violation") or 0)
            lov = int(result.get("periods_line_overload") or 0)
            rows.append({
                "setting": setting,
                "slack_vm_pu": slack,
                "reactive_fraction": fraction,
                "line_impedance_scale": scale,
                "group": group,
                "date": date,
                "peak_load_mw": peak,
                **{k: v for k, v in result.items() if k != "date"},
                "passes": bool(converged and vviol == 0 and lov == 0),
            })
        print(f"{setting}: {len(prepared)} days screened", flush=True)
    out_path = _write_csv(root / args.out, rows)
    artifacts = {
        str(subsets_path): _sha256(subsets_path),
        str(candidates_path): _sha256(candidates_path),
        str(users_path): _sha256(users_path),
        str(curves_path): _sha256(curves_path),
        str(out_path): _sha256(out_path),
    }
    report_path = _write_report(
        root / args.report, rows=rows, worst_only=worst_only,
        command=" ".join(sys.argv), artifacts=artifacts)
    passing = [s for s, *_rest in settings
               if all(r["passes"] for r in rows if r["setting"] == s)]
    print(f"settings that pass all groups: {passing}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
