# analyze_v2_subset_feasibility.py
"""Which fixed user subsets can cover the selected representative days?

This is a load-coverage screen, not a market or network result. For each
candidate fixed subset it asks:

* is every user all-meters complete on every selected representative day?
* how many representative days survive in each season?
* what is the subset's coincident peak, daily energy and bus concentration on
  the days it does cover?

The selected representative days were chosen from the provisional core. A
larger subset may not cover all of them; that is reported rather than repaired.
Physical OPF feasibility is deliberately not claimed here; it must be checked
after a subset is chosen.

Usage:
    PYTHONPATH=src python src/analyze_v2_subset_feasibility.py
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

HOURS_PER_SLOT = 0.25
SEASONS = ("spring", "summer", "autumn", "winter")

BUS_COLUMNS = ("bus", "users", "energy_mwh", "share")

COLUMNS = (
    "subset", "users", "buses", "industries", "v1_overlap",
    "all_meters_complete_dates", "representative_dates", "reps_complete",
    "spring_reps", "summer_reps", "autumn_reps", "winter_reps",
    "multiseason", "min_season_reps", "median_peak_mw", "min_peak_mw",
    "max_peak_mw", "peak_ratio_to_3_715", "median_energy_mwh",
    "max_bus_energy_share", "top_bus",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict],
               columns: Sequence[str] = COLUMNS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _subset_metrics(subset: str, users: Sequence[str],
                    user_meta: Dict[str, dict],
                    complete: Dict[str, set],
                    reps: Sequence[dict],
                    curves) -> dict:
    rep_dates = [r["date"] for r in reps]
    common = None
    for user in users:
        common = set(complete.get(user, set())) if common is None \
            else common & complete.get(user, set())
    common = common or set()
    covered = [r for r in reps if r["date"] in common]
    season_counts = collections.Counter(r["season"] for r in covered)
    peaks: List[float] = []
    energies: List[float] = []
    bus_energy: Dict[int, float] = collections.defaultdict(float)
    for row in covered:
        date = row["date"]
        curve = np.zeros(96)
        for user in users:
            key = f"{date}|{user}"
            if key in curves:
                curve += np.asarray(curves[key], dtype=float)
        peak = float(np.max(curve))
        energy = float(np.sum(curve) * HOURS_PER_SLOT)
        peaks.append(peak)
        energies.append(energy)
        for user in users:
            key = f"{date}|{user}"
            if key not in curves:
                continue
            amount = float(np.sum(np.asarray(curves[key], dtype=float))
                           * HOURS_PER_SLOT)
            bus_energy[int(user_meta[user]["bus"])] += amount
    total_energy = sum(bus_energy.values())
    top_bus = max(bus_energy, key=bus_energy.get) if bus_energy else ""
    top_share = (bus_energy[top_bus] / total_energy
                 if top_bus != "" and total_energy else "")
    industries = len({user_meta[u]["industry"] for u in users})
    buses = len({user_meta[u]["bus"] for u in users})
    v1_overlap = sum(1 for u in users if user_meta[u]["in_v1"])
    min_reps = min(season_counts.get(s, 0) for s in SEASONS)
    return {
        "subset": subset,
        "users": len(users),
        "buses": buses,
        "industries": industries,
        "v1_overlap": v1_overlap,
        "all_meters_complete_dates": len(common),
        "representative_dates": len(reps),
        "reps_complete": len(covered),
        "spring_reps": season_counts.get("spring", 0),
        "summer_reps": season_counts.get("summer", 0),
        "autumn_reps": season_counts.get("autumn", 0),
        "winter_reps": season_counts.get("winter", 0),
        "multiseason": min_reps > 0,
        "min_season_reps": min_reps,
        "median_peak_mw": (statistics.median(peaks) if peaks else ""),
        "min_peak_mw": (min(peaks) if peaks else ""),
        "max_peak_mw": (max(peaks) if peaks else ""),
        "peak_ratio_to_3_715": (statistics.median(peaks) / 3.715
                                if peaks else ""),
        "median_energy_mwh": (statistics.median(energies) if energies else ""),
        "max_bus_energy_share": top_share,
        "top_bus": top_bus,
    }


def _bus_distribution(users: Sequence[str], user_meta: Dict[str, dict],
                      reps: Sequence[dict], curves) -> List[dict]:
    energy: Dict[int, float] = collections.defaultdict(float)
    counts: Dict[int, int] = collections.defaultdict(int)
    for user in users:
        bus = int(user_meta[user]["bus"])
        counts[bus] += 1
        for row in reps:
            key = f"{row['date']}|{user}"
            if key in curves:
                energy[bus] += float(np.sum(np.asarray(curves[key], dtype=float))
                                     * HOURS_PER_SLOT)
    total = sum(energy.values())
    return [
        {"bus": bus, "users": counts[bus], "energy_mwh": energy[bus],
         "share": (energy[bus] / total if total else "")}
        for bus in sorted(energy)
    ]


def _write_report(path: Path, *, rows: Sequence[dict], core: dict,
                  user_meta: Dict[str, dict], reps: Sequence[dict],
                  bus_rows: Sequence[dict], csv_path: Path,
                  bus_csv_path: Path, artifacts: Dict[str, str],
                  command: str) -> Path:
    lines = [
        "# V2 fixed-subset feasibility",
        "",
        "## Verdict",
        "",
        "This is a **load-coverage screen** on the representative dates "
        "selected in Phase B. It does not freeze a subset and does not claim "
        "that a subset is physically operable; the OPF/network check belongs "
        "to the next step.",
        "",
        "The key trade-off is coverage versus breadth: a smaller fixed subset "
        "survives more seasons and dates; a larger subset adds buses and "
        "industries but loses seasons because some users have few complete "
        "all-meters days.",
        "",
        "## Subsets compared",
        "",
        "| subset | users | buses | industries | V1 overlap | all-meters dates | "
        "representative days covered | spring | summer | autumn | winter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['subset']} | {r['users']} | {r['buses']} | "
            f"{r['industries']} | {r['v1_overlap']} | "
            f"{r['all_meters_complete_dates']} | {r['reps_complete']} | "
            f"{r['spring_reps']} | {r['summer_reps']} | "
            f"{r['autumn_reps']} | {r['winter_reps']} |")
    lines += ["", "## Load scale on covered representative days", "",
              "| subset | median peak MW | median peak / 3.715 | peak range MW | "
              "median energy MWh | top bus energy share |",
              "|---|---:|---:|---|---:|---:|"]
    for r in rows:
        peak_range = (f"{float(r['min_peak_mw']):.3f}-"
                      f"{float(r['max_peak_mw']):.3f}"
                      if r["min_peak_mw"] != "" else "")
        share = (f"{float(r['max_bus_energy_share']):.1%} (bus "
                 f"{r['top_bus']})" if r["max_bus_energy_share"] != "" else "")
        median_peak = (f"{float(r['median_peak_mw']):.3f}"
                       if r["median_peak_mw"] != "" else "")
        ratio = (f"{float(r['peak_ratio_to_3_715']):.2f}x"
                 if r["peak_ratio_to_3_715"] != "" else "")
        median_energy = (f"{float(r['median_energy_mwh']):.3f}"
                         if r["median_energy_mwh"] != "" else "")
        lines.append(f"| {r['subset']} | {median_peak} | {ratio} | "
                     f"{peak_range} | {median_energy} | {share} |")
    lines += ["", "## Projection of the 3.715 MW reference", "",
              f"3.715 MW is the IEEE33 reference, not a target. The table "
              "above is the measured load of each candidate subset; a subset "
              "far above the reference may still be a real observation, but it "
              "needs a separate physical feasibility check before it is used "
              "as a market scenario.", "",
              "## Provisional core", "",
              f"- Phase A core: **{len(core['core_users'])} users**, "
              f"{len(core['core_buses'])} buses, "
              f"{len(core['core_industries'])} industries, "
              f"{core['core_dates_count']} all-meters complete dates.",
              f"- Season counts: {core['season_counts']}.",
              f"- Holiday counts: {core['holiday_counts']}.",
              f"- Representative dates selected from this core: "
              f"{len(reps)}.", "",
              "## Provisional core bus distribution", "",
              "| bus | users | energy MWh | share |",
              "|---:|---:|---:|---:|"]
    for row in bus_rows:
        lines.append(
            f"| {row['bus']} | {row['users']} | "
            f"{float(row['energy_mwh']):.3f} | "
            f"{float(row['share']):.1%} |")
    lines += ["", "## Recommendation", ""]
    multiseason = [r for r in rows if r["multiseason"]]
    within_2x = [r for r in multiseason
                 if r["peak_ratio_to_3_715"] != ""
                 and float(r["peak_ratio_to_3_715"]) <= 2.0]
    if within_2x:
        best = max(within_2x, key=lambda r: (
            int(r["users"]), int(r["buses"]), int(r["industries"])))
        lines.append(
            f"- Largest multi-season subset at or below 2x the 3.715 MW "
            f"reference: **{best['subset']}** ({best['users']} users, "
            f"{best['buses']} buses, {best['industries']} industries, "
            f"median peak {float(best['median_peak_mw']):.3f} MW, "
            f"{float(best['peak_ratio_to_3_715']):.2f}x).")
    else:
        lines.append(
            "- No multi-season subset in this screen has a median peak at or "
            "below 2x the 3.715 MW reference. A physically reasonable "
            "multi-season fixed subset is therefore **not established**; this "
            "is a data/scale gap, not a tuning problem.")
    if multiseason:
        broad = max(multiseason, key=lambda r: (
            int(r["buses"]), int(r["industries"]), int(r["users"])))
        lines.append(
            f"- Broadest multi-season subset: **{broad['subset']}** "
            f"({broad['users']} users, {broad['buses']} buses, "
            f"{broad['industries']} industries, median peak "
            f"{float(broad['median_peak_mw']):.3f} MW, "
            f"{float(broad['peak_ratio_to_3_715']):.2f}x).")
    lines += [
        "- The smaller subsets keep more complete dates but cover fewer buses "
        "and industries; the larger subsets lose seasons at the all-meters "
        "standard.",
        "- The V1 17-user subset is retained as the historical single-day "
        "benchmark. At the all-meters standard it covers none of the selected "
        "representative dates, so it is not a drop-in V2 multi-season subset.",
        "- Do not freeze a subset until the breadth/coverage/scale trade-off "
        "has been reviewed and a physical OPF/network check has been run on "
        "the chosen days.",
        "",
        "## Artifacts", "",
        f"- `{csv_path}`",
        f"- `{bus_csv_path}`"]
    lines += ["", "| path | sha256 |", "|---|---|"]
    for path_text, digest in artifacts.items():
        lines.append(f"| `{path_text}` | `{digest}` |")
    lines += ["", "## Command", "", f"`{command}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen fixed user subsets on the V2 representative days.")
    parser.add_argument("--coverage", default="reports/v2_user_date_coverage.csv")
    parser.add_argument("--users", default="data/processed/v2/v2_user_summary.csv")
    parser.add_argument("--core", default="data/processed/v2/v2_provisional_core.json")
    parser.add_argument("--candidates",
                        default="data/processed/v2_typical_day_candidates.csv")
    parser.add_argument("--curves", default="data/processed/v2/v2_user_day_curves.npz")
    parser.add_argument("--out", default="data/processed/v2/v2_subset_feasibility.csv")
    parser.add_argument("--report", default="reports/v2_subset_feasibility.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    paths = {name: root / value for name, value in (
        ("coverage", args.coverage), ("users", args.users),
        ("core", args.core), ("candidates", args.candidates),
        ("curves", args.curves))}
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"{path} not found; run the earlier V2 stages first")

    user_rows = _read_csv(paths["users"])
    user_meta = {
        r["user_id"]: {
            "bus": int(r["bus"]),
            "industry": r["industry"],
            "in_v1": _as_bool(r["in_v1_subset"]),
            "all_meters_days": int(r["all_meters_days"]),
        }
        for r in user_rows
    }
    complete: Dict[str, set] = collections.defaultdict(set)
    for row in _read_csv(paths["coverage"]):
        if _as_bool(row["all_meters_complete"]):
            complete[row["user_id"]].add(row["date"])
    candidates = _read_csv(paths["candidates"])
    reps = [r for r in candidates if _as_bool(r["representative"])]
    core = json.loads(paths["core"].read_text(encoding="utf-8"))
    curves = np.load(paths["curves"])
    ranked = sorted(user_meta, key=lambda u: (
        -user_meta[u]["all_meters_days"], u))
    v1 = [u for u in ranked if user_meta[u]["in_v1"]]
    subsets = {
        "top03_all_meters": ranked[:3],
        "top05_all_meters": ranked[:5],
        "top08_all_meters": ranked[:8],
        "top10_all_meters": ranked[:10],
        "top12_all_meters": ranked[:12],
        "top15_provisional_core": list(core["core_users"]),
        "top20_all_meters": ranked[:20],
        "v1_17": v1,
    }
    rows = [
        _subset_metrics(name, users, user_meta, complete, reps, curves)
        for name, users in subsets.items()
    ]
    out_path = _write_csv(root / args.out, rows)
    bus_rows = _bus_distribution(
        list(core["core_users"]), user_meta, reps, curves)
    bus_csv_path = _write_csv(
        root / "data/processed/v2/v2_core_bus_distribution.csv", bus_rows,
        BUS_COLUMNS)
    artifacts = {
        str(paths["coverage"]): _sha256(paths["coverage"]),
        str(paths["users"]): _sha256(paths["users"]),
        str(paths["core"]): _sha256(paths["core"]),
        str(paths["candidates"]): _sha256(paths["candidates"]),
        str(paths["curves"]): _sha256(paths["curves"]),
        str(out_path): _sha256(out_path),
        str(bus_csv_path): _sha256(bus_csv_path),
    }
    report_path = _write_report(
        root / args.report, rows=rows, core=core, user_meta=user_meta,
        reps=reps, bus_rows=bus_rows, csv_path=out_path,
        bus_csv_path=bus_csv_path, artifacts=artifacts,
        command=" ".join(sys.argv))
    for row in rows:
        print(f"{row['subset']:<24} users={row['users']:>2} "
              f"buses={row['buses']:>2} reps={row['reps_complete']:>2} "
              f"seasons={row['spring_reps']}/{row['summer_reps']}/"
              f"{row['autumn_reps']}/{row['winter_reps']}")
    print(f"wrote {out_path}")
    print(f"wrote {bus_csv_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
