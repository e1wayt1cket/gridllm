# select_v2_seasonal_subsets.py
"""A per-season fixed user subset, with the differences made explicit.

The fixed-subset screen found no one subset that simultaneously kept four-season
all-meters coverage, a reasonable bus/industry breadth and a load near the
IEEE33 reference. The chosen fallback is four **different** fixed subsets, one
per season. They are reported as different systems; a result from one season is
not a result from another.

Selection rule, per season:

1. candidate users have at least `--min-user-days` all-meters complete dates in
   that season;
2. order them by median individual peak, smallest first;
3. add a user only if the remaining common complete dates stay at or above
   `--min-common-days` and the aggregate median peak stays at or below
   `--max-ratio` times 3.715 MW.

No load is scaled, no high-load period is deleted and no multiplier is edited.
The subsets are **proposed**; they are not frozen and no training is started.

Usage:
    PYTHONPATH=src python src/select_v2_seasonal_subsets.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import multiday_dataset as md
import select_v2_typical_days as day_select

HOURS_PER_SLOT = 0.25
TARGET_MW = 3.715
SEASONS = ("spring", "summer", "autumn", "winter")

SUBSET_COLUMNS = (
    "group", "group_type", "season", "status", "selection_rule", "users",
    "buses", "industries", "common_dates", "representative_dates",
    "median_peak_mw", "min_peak_mw", "max_peak_mw", "peak_ratio_to_3_715",
    "median_energy_mwh", "top_bus", "top_bus_energy_share", "user_ids",
    "bus_ids", "industry_names",
)

CANDIDATE_COLUMNS = (
    "group", "group_type", "season", "group_users", "date", "year", "month",
    "weekday", "day_type", "holiday_class", "holiday_name", "holiday_offset",
    "peak_mw", "mean_mw", "energy_mwh", "load_factor", "representative",
    "selection_reason",
)


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


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_user_data(coverage_path: Path, users_path: Path,
                   curves_path: Path) -> tuple:
    users = {
        r["user_id"]: {
            "bus": int(r["bus"]),
            "industry": r["industry"],
            "all_meters_days": int(r["all_meters_days"]),
        }
        for r in _read_csv(users_path)
    }
    curves = np.load(curves_path)
    data: Dict[tuple, dict] = collections.defaultdict(
        lambda: {"dates": set(), "curves": {}})
    for row in _read_csv(coverage_path):
        if not _as_bool(row["all_meters_complete"]):
            continue
        user = row["user_id"]
        key = f"{row['date']}|{user}"
        if key not in curves:
            continue
        curve = np.asarray(curves[key], dtype=float)
        season = row["season"]
        data[(user, season)]["dates"].add(row["date"])
        data[(user, season)]["curves"][row["date"]] = curve
        if (row["day_type"] == "holiday"
                and row["holiday_class"] in ("spring_festival",
                                             "national_day")):
            group = f"{row['holiday_class']}_{row['date'][:4]}"
            data[(user, group)]["dates"].add(row["date"])
            data[(user, group)]["curves"][row["date"]] = curve
    return users, data


def _aggregate(curves: Dict[str, np.ndarray], dates: Sequence[str],
               users: Sequence[str]) -> List[dict]:
    rows = []
    for date in sorted(dates):
        arr = np.zeros(96)
        for user in users:
            arr += curves[(user, date)]
        rows.append({
            "date": date,
            "peak_mw": float(np.max(arr)),
            "mean_mw": float(np.mean(arr)),
            "energy_mwh": float(np.sum(arr) * HOURS_PER_SLOT),
            "load_factor": (float(np.mean(arr) / np.max(arr))
                            if np.max(arr) else 0.0),
        })
    return rows


def _median_peak(data: Dict[tuple, dict], group: str,
                 users: Sequence[str], dates: Sequence[str]) -> float:
    peaks = []
    for date in dates:
        arr = np.zeros(96)
        for user in users:
            arr += data[(user, group)]["curves"][date]
        peaks.append(float(np.max(arr)))
    return statistics.median(peaks) if peaks else float("nan")


def select_group(users: Dict[str, dict], data: Dict[tuple, dict],
                 group: str, group_type: str, min_user_days: int,
                 min_common_days: int, max_ratio: float) -> dict:
    candidates = []
    for user in users:
        dates = data[(user, group)]["dates"]
        if len(dates) < min_user_days:
            continue
        peaks = [float(np.max(c)) for c in
                 data[(user, group)]["curves"].values()]
        candidates.append({
            "user": user,
            "dates": set(dates),
            "median_peak": statistics.median(peaks) if peaks else 0.0,
        })
    candidates.sort(key=lambda r: (r["median_peak"], r["user"]))
    chosen: List[str] = []
    common = None
    for item in candidates:
        new_common = item["dates"] if common is None else common & item["dates"]
        if len(new_common) < min_common_days:
            continue
        median_peak = _median_peak(
            data, group, chosen + [item["user"]], sorted(new_common))
        if median_peak > max_ratio * TARGET_MW:
            continue
        chosen.append(item["user"])
        common = new_common
    chosen = sorted(chosen)
    common = sorted(common or [])
    aggregate = _aggregate(
        {(u, d): data[(u, group)]["curves"][d]
         for u in chosen for d in common}, common, chosen)
    peaks = [r["peak_mw"] for r in aggregate]
    energies = [r["energy_mwh"] for r in aggregate]
    bus_energy = collections.defaultdict(float)
    for user in chosen:
        bus = users[user]["bus"]
        for date in common:
            bus_energy[bus] += float(
                np.sum(data[(user, group)]["curves"][date])) * HOURS_PER_SLOT
    total = sum(bus_energy.values())
    top_bus = max(bus_energy, key=bus_energy.get) if bus_energy else ""
    season = (md.calendar_label(common[0])["season"] if common else "")
    return {
        "group": group,
        "group_type": group_type,
        "season": season,
        "status": "proposed",
        "selection_rule": (
            f"users with >= {min_user_days} all-meters days in {group}; add "
            f"smallest-median-peak users while common dates >= "
            f"{min_common_days} and aggregate median peak <= "
            f"{max_ratio:g}x {TARGET_MW} MW"),
        "users": len(chosen),
        "buses": len({users[u]["bus"] for u in chosen}),
        "industries": len({users[u]["industry"] for u in chosen}),
        "common_dates": len(common),
        "representative_dates": 0,
        "median_peak_mw": (statistics.median(peaks) if peaks else ""),
        "min_peak_mw": (min(peaks) if peaks else ""),
        "max_peak_mw": (max(peaks) if peaks else ""),
        "peak_ratio_to_3_715": (statistics.median(peaks) / TARGET_MW
                                if peaks else ""),
        "median_energy_mwh": (statistics.median(energies) if energies else ""),
        "top_bus": top_bus,
        "top_bus_energy_share": (bus_energy[top_bus] / total
                                 if top_bus != "" and total else ""),
        "user_ids": ";".join(chosen),
        "bus_ids": ";".join(str(b) for b in
                            sorted({users[u]["bus"] for u in chosen})),
        "industry_names": ";".join(sorted({users[u]["industry"]
                                           for u in chosen})),
        "_chosen": chosen,
        "_dates": common,
        "_aggregate": aggregate,
    }


def _group_rows(group: str, group_type: str, chosen: Sequence[str],
                dates: Sequence[str], aggregate: Sequence[dict],
                users: Dict[str, dict]) -> List[dict]:
    out = []
    agg_by_date = {r["date"]: r for r in aggregate}
    for date in dates:
        label = md.calendar_label(date)
        agg = agg_by_date[date]
        out.append({
            "group": group,
            "group_type": group_type,
            "season": label["season"],
            "group_users": len(chosen),
            "date": date,
            "year": date[:4],
            "month": date[5:7],
            **{k: label[k] for k in (
                "weekday", "day_type", "holiday_class", "holiday_name",
                "holiday_offset")},
            "peak_mw": agg["peak_mw"],
            "mean_mw": agg["mean_mw"],
            "energy_mwh": agg["energy_mwh"],
            "load_factor": agg["load_factor"],
            "core_peak_mw": agg["peak_mw"],
            "core_energy_mwh": agg["energy_mwh"],
            "representative": False,
            "selection_reason": "",
        })
    return out


def _mark_representatives(rows: List[dict], limit: int = 3) -> None:
    for day_type in ("weekday", "weekend"):
        selected = day_select._stratum_representatives(
            [r for r in rows if r["day_type"] == day_type],
            rows[0]["season"], day_type, limit)
        chosen = {r["date"]: r["selection_reason"] for r in selected}
        for row in rows:
            if row["date"] in chosen:
                row["representative"] = True
                row["selection_reason"] = chosen[row["date"]]
    selected = day_select._holiday_representatives(
        [r for r in rows if r["day_type"] == "holiday"],
        rows[0]["season"], limit)
    chosen = {r["date"]: r["selection_reason"] for r in selected}
    for row in rows:
        if row["date"] in chosen:
            row["representative"] = True
            row["selection_reason"] = chosen[row["date"]]


def _write_report(path: Path, *, rows: Sequence[dict],
                  per_season: Dict[str, List[dict]],
                  users: Dict[str, dict], command: str,
                  artifacts: Dict[str, str]) -> Path:
    lines = [
        "# V2 seasonal user subsets",
        "",
        "## Verdict",
        "",
        "The chosen fallback is **four seasonal subsets plus two 2022 holiday "
        "subsets**. They are not one system: a result from spring is not "
        "directly comparable to a result from winter, and Spring Festival is "
        "not the winter ordinary subset. The user sets are listed below so the "
        "difference cannot be hidden.",
        "",
        "The subsets are **proposed**, not frozen. No training is started and "
        "no physical OPF feasibility is claimed.",
        "",
        "## Selection rule",
        "",
        "- Candidate users have at least the stated number of all-meters "
        "complete dates in the group.",
        "- Users are added smallest-median-peak first while the common "
        "complete dates stay at or above the minimum and the aggregate median "
        "peak stays within the stated multiple of 3.715 MW.",
        "- No load is scaled, no high-load period is deleted and no multiplier "
        "is edited.",
        "",
        "## Seasonal subsets",
        "",
        "| season | users | buses | industries | common dates | "
        "representative days | median peak MW | /3.715 | top bus share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    seasonal = [r for r in rows if r["group_type"] == "season"]
    holiday = [r for r in rows if r["group_type"] == "holiday"]
    for season in SEASONS:
        r = next((x for x in seasonal if x["group"] == season), None)
        if (r is None or int(r["users"]) == 0
                or r["median_peak_mw"] == ""):
            lines.append(f"| {season} | 0 | 0 | 0 | 0 | 0 | — | — | — |")
            continue
        lines.append(
            f"| {season} | {r['users']} | {r['buses']} | {r['industries']} | "
            f"{r['common_dates']} | {r['representative_dates']} | "
            f"{float(r['median_peak_mw']):.3f} | "
            f"{float(r['peak_ratio_to_3_715']):.2f}x | "
            f"{float(r['top_bus_energy_share']):.1%} (bus {r['top_bus']}) |")
    lines += ["", "## 2022 holiday subsets", "",
              "| group | season | users | buses | industries | common dates | "
              "representative days | median peak MW | /3.715 | top bus share |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in holiday:
        if int(r["users"]) == 0 or r["median_peak_mw"] == "":
            lines.append(f"| {r['group']} | {r['season']} | 0 | 0 | 0 | 0 | "
                         "0 | — | — | — |")
            continue
        lines.append(
            f"| {r['group']} | {r['season']} | {r['users']} | {r['buses']} | "
            f"{r['industries']} | {r['common_dates']} | "
            f"{r['representative_dates']} | "
            f"{float(r['median_peak_mw']):.3f} | "
            f"{float(r['peak_ratio_to_3_715']):.2f}x | "
            f"{float(r['top_bus_energy_share']):.1%} (bus {r['top_bus']}) |")
    lines += ["", "## User sets (explicitly different)", ""]
    for r in rows:
        chosen = r["user_ids"].split(";") if r["user_ids"] else []
        lines += [f"### {r['group']}", "",
                  "| user | bus | industry |", "|---|---:|---|"]
        for user in chosen:
            lines.append(f"| {user} | {users[user]['bus']} | "
                         f"{users[user]['industry']} |")
        lines.append("")
    lines += ["## Representative days", ""]
    for r in rows:
        rows_g = per_season.get(r["group"], [])
        reps = [x for x in rows_g if _as_bool(x["representative"])]
        lines += [f"### {r['group']}", "",
                  "| date | day type | holiday | offset | peak MW | "
                  "energy MWh | reason |",
                  "|---|---|---|---:|---:|---:|---|"]
        for x in sorted(reps, key=lambda item: item["date"]):
            offset = (x["holiday_offset"]
                      if x["holiday_offset"] not in (None, "", "None") else "")
            lines.append(
                f"| {x['date']} | {x['day_type']} | {x['holiday_name']} | "
                f"{offset} | {float(x['peak_mw']):.3f} | "
                f"{float(x['energy_mwh']):.3f} | {x['selection_reason']} |")
        lines.append("")
    lines += [
        "## Data gaps and limits",
        "",
        "- The 2022 Spring Festival and National Day are represented by "
        "separate holiday subsets, not by the seasonal ordinary subsets. The "
        "seasonal and holiday user sets are not the same system.",
        "- The 2021 Spring Festival and National Day are not covered by these "
        "groups; their all-meters common sets are too small.",
        "- Spring and winter have fewer common dates; their representative "
        "libraries are smaller than autumn's.",
        "- The price blocker is unchanged: only one real single-day price curve "
        "is present, so no priced multi-day training set is possible.",
        "- Physical feasibility is screened in "
        "`reports/v2_physical_screen.md` and "
        "`reports/v2_voltage_support_screen.md`. Without support every group "
        "violates the voltage band; with the `line50_q2.0` assumption "
        "(uniform line R/X at 0.5x plus ideal local reactive compensation at "
        "2x load Q) all 31 representative days pass. The network and support "
        "rating are not frozen.",
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
        description="Propose per-season fixed V2 user subsets.")
    parser.add_argument("--coverage", default="reports/v2_user_date_coverage.csv")
    parser.add_argument("--users", default="data/processed/v2/v2_user_summary.csv")
    parser.add_argument("--curves", default="data/processed/v2/v2_user_day_curves.npz")
    parser.add_argument("--subset-out",
                        default="data/processed/v2_seasonal_subsets.csv")
    parser.add_argument("--candidate-out",
                        default="data/processed/v2_seasonal_candidates.csv")
    parser.add_argument("--metadata",
                        default="data/processed/v2/v2_seasonal_subsets.json")
    parser.add_argument("--report", default="reports/v2_seasonal_subsets.md")
    parser.add_argument("--min-user-days", type=int, default=20)
    parser.add_argument("--min-common-days", type=int, default=20)
    parser.add_argument("--holiday-min-dates", type=int, default=2)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    coverage = root / args.coverage
    users_path = root / args.users
    curves_path = root / args.curves
    for path in (coverage, users_path, curves_path):
        if not path.exists():
            raise SystemExit(f"{path} not found; run the V2 audit first")
    users, data = load_user_data(coverage, users_path, curves_path)
    groups = [
        ("spring", "season"),
        ("summer", "season"),
        ("autumn", "season"),
        ("winter", "season"),
        ("spring_festival_2022", "holiday"),
        ("national_day_2022", "holiday"),
    ]
    rows = []
    per_season: Dict[str, List[dict]] = {}
    for group, group_type in groups:
        min_user_days = (args.min_user_days if group_type == "season"
                         else args.holiday_min_dates)
        min_common_days = (args.min_common_days if group_type == "season"
                           else args.holiday_min_dates)
        result = select_group(
            users, data, group, group_type, min_user_days,
            min_common_days, args.max_ratio)
        if not result["_chosen"]:
            result["status"] = "no_feasible_subset"
            rows.append(result)
            per_season[group] = []
            continue
        group_rows = _group_rows(
            group, group_type, result["_chosen"], result["_dates"],
            result["_aggregate"], users)
        _mark_representatives(group_rows, limit=3)
        result["representative_dates"] = sum(
            1 for r in group_rows if _as_bool(r["representative"]))
        per_season[group] = group_rows
        rows.append(result)
    subset_path = _write_csv(
        root / args.subset_out, rows, SUBSET_COLUMNS)
    candidate_rows = [r for group, _type in groups
                      for r in per_season[group]]
    candidate_path = _write_csv(
        root / args.candidate_out, candidate_rows, CANDIDATE_COLUMNS)
    metadata = {
        "status": "proposed",
        "note": ("four different fixed user sets, one per season; not one "
                 "system and not a frozen selection"),
        "rule": {
            "min_user_days": args.min_user_days,
            "min_common_days": args.min_common_days,
            "max_ratio": args.max_ratio,
            "target_mw": TARGET_MW,
        },
        "groups": {
            r["group"]: {k: v for k, v in r.items() if not k.startswith("_")}
            for r in rows
        },
    }
    metadata_path = _write_json(root / args.metadata, metadata)
    artifacts = {
        str(coverage): _sha256(coverage),
        str(users_path): _sha256(users_path),
        str(curves_path): _sha256(curves_path),
        str(subset_path): _sha256(subset_path),
        str(candidate_path): _sha256(candidate_path),
        str(metadata_path): _sha256(metadata_path),
    }
    report_path = _write_report(
        root / args.report, rows=rows, per_season=per_season, users=users,
        command=" ".join(sys.argv), artifacts=artifacts)
    for row in rows:
        peak = ("" if row["median_peak_mw"] == ""
                else f"{float(row['median_peak_mw']):.3f}")
        ratio = ("" if row["peak_ratio_to_3_715"] == ""
                 else f"{float(row['peak_ratio_to_3_715']):.2f}")
        print(f"{row['season']:<7} users={row['users']:>2} "
              f"buses={row['buses']:>2} industries={row['industries']:>2} "
              f"dates={row['common_dates']:>3} "
              f"reps={row['representative_dates']:>2} "
              f"peak={peak} ratio={ratio}")
    print(f"wrote {subset_path}")
    print(f"wrote {candidate_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
