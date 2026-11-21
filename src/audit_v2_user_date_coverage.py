# audit_v2_user_date_coverage.py
"""Every mapped user on every real date, without rescaling or filling.

V1 answered "which 17 users reproduce 3.715 MW on 2022-11-30". V2 asks a
different question: which real dates carry a **complete** curve for a fixed
user set across the seasons? This script builds the user x date coverage the
question needs.

Two completeness levels are recorded separately because they are not the same
claim:

* `user_complete` -- at least one of the user's mapped meters has a qualified,
  convertible 96-slot day. This is the V1 rule.
* `all_meters_complete` -- **every** mapped meter of the user does. This is the
  stricter V2 rule: a user with 11 meters and one observed meter is not a
  complete user-day, and summing the observed one would silently shrink the
  user's load.

Partial days are reported, never filled. A user with no admissible reading has
blank load fields, not zeros. The provisional fixed core is selected by
coverage alone: rank users by all-meters complete days, then take the largest
prefix that still has at least `--min-season-dates` complete dates in every
season. It is a candidate for Phase C, not a frozen subset.

Usage:
    PYTHONPATH=src python src/audit_v2_user_date_coverage.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import industry_load as il
import multiday_dataset as md

HOURS_PER_SLOT = il.HOURS_PER_SLOT
SEASONS = ("spring", "summer", "autumn", "winter")
FULL_MAPPING = Path("data/processed/industry_meter_bus_map_v2.csv")
V1_MAPPING = Path("data/processed/industry_user_subset_mapping_v2.csv")

COVERAGE_COLUMNS = (
    "date", "season", "weekday", "day_type", "holiday_class", "holiday_name",
    "holiday_offset", "user_id", "bus", "industry", "in_v1_subset",
    "n_meters", "records", "qualified_blocks", "converted_blocks",
    "missing_meters", "slots_min", "slots_max", "flags", "user_complete",
    "all_meters_complete", "partial_meters", "peak_mw", "mean_mw",
    "energy_mwh", "peak_period", "missing_reason",
)

DATE_SUMMARY_COLUMNS = (
    "date", "season", "weekday", "day_type", "holiday_class", "holiday_name",
    "holiday_offset", "user_complete_users", "all_meters_users",
    "user_complete_buses", "all_meters_buses", "max_complete_users",
    "max_all_meters_users", "all_41_complete", "all_37_complete",
    "all_35_complete",
    "all_30_complete", "core_complete", "v1_17_user_complete",
    "v1_17_all_meters_complete", "core_peak_mw", "core_mean_mw",
    "core_energy_mwh", "core_peak_period", "core_load_factor",
    "core_buses_with_data",
)

USER_SUMMARY_COLUMNS = (
    "user_id", "bus", "industry", "n_meters", "in_v1_subset",
    "user_complete_days", "all_meters_days", "partial_days",
    "qualified_days", "first_all_meters_date", "last_all_meters_date",
    "mean_peak_mw", "mean_energy_mwh",
)


def _iso_date_or_none(value) -> Optional[str]:
    try:
        return md.as_date(value).isoformat()
    except (TypeError, ValueError):
        return None


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


def load_mapping(path: Path) -> tuple:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    mapped = [r for r in rows if r["candidate_status"] == "mapped"]
    v1_rows = list(csv.DictReader(V1_MAPPING.open(encoding="utf-8")))
    v1_users = {r["user_id"] for r in v1_rows
                if r["candidate_status"] == "mapped"}
    users: Dict[str, dict] = {}
    for row in mapped:
        user = row["user_id"]
        meters = {m for m in str(row["meter_ids"]).split(";") if m}
        if not meters:
            raise SystemExit(f"{user} has no mapped meter")
        users[user] = {
            "user_id": user,
            "bus": int(row["destination_bus"]),
            "industry": row["industry"],
            "n_meters": len(meters),
            "meters": meters,
            "in_v1": user in v1_users,
        }
    if len(users) != 41:
        raise SystemExit(
            f"{path} carries {len(users)} mapped users; the full V2 pool is 41")
    if len({u["bus"] for u in users.values()}) != 17:
        raise SystemExit("the mapped pool does not cover 17 buses")
    if not v1_users <= set(users):
        raise SystemExit("a V1 user is missing from the full mapped pool")
    return users, v1_users


def _new_entry() -> dict:
    return {
        "records": 0,
        "seen_meters": set(),
        "qualified_meters": set(),
        "converted_meters": set(),
        "unconvertible_meters": set(),
        "min_slots": il.N_SLOTS,
        "max_slots": 0,
        "flags": set(),
        "curve": np.zeros(il.N_SLOTS, dtype=float),
    }


def _slots_present(values) -> int:
    return sum(1 for value in values
               if value is not None and np.isfinite(float(value)))


def collect(records: Sequence[il.MeterDay], users: Dict[str, dict]) -> tuple:
    """Per (date, user) curves and raw coverage, in one pass over the records."""
    valid_keys = {
        (meter, _iso_date_or_none(day))
        for meter, day in il.qualified_block_keys(records)
    }
    stats: Dict[tuple, dict] = {}
    issues = {
        "unmapped_user_records": 0,
        "unmapped_meter_records": 0,
        "missing_date_records": 0,
        "records": len(records),
    }
    for record in records:
        user = record.user_key
        if user not in users:
            issues["unmapped_user_records"] += 1
            continue
        date = _iso_date_or_none(record.date)
        if date is None:
            issues["missing_date_records"] += 1
            continue
        if record.meter_key not in users[user]["meters"]:
            issues["unmapped_meter_records"] += 1
            continue
        entry = stats.setdefault((date, user), _new_entry())
        entry["records"] += 1
        entry["seen_meters"].add(record.meter_key)
        slots = _slots_present(record.values)
        entry["min_slots"] = min(entry["min_slots"], slots)
        entry["max_slots"] = max(entry["max_slots"], slots)
        entry["flags"].update(record.flags)
        if (record.meter_key, date) not in valid_keys:
            continue
        entry["qualified_meters"].add(record.meter_key)
        if record.meter_key in entry["converted_meters"]:
            continue
        curve = il.reading_to_mw(record)
        if curve is None:
            entry["unconvertible_meters"].add(record.meter_key)
            continue
        entry["curve"] += curve
        entry["converted_meters"].add(record.meter_key)
    return stats, issues


def _curve_metrics(curve: Optional[np.ndarray]) -> dict:
    if curve is None:
        return {"peak_mw": "", "mean_mw": "", "energy_mwh": "",
                "peak_period": ""}
    arr = np.asarray(curve, dtype=float)
    return {
        "peak_mw": float(np.max(arr)),
        "mean_mw": float(np.mean(arr)),
        "energy_mwh": float(np.sum(arr) * HOURS_PER_SLOT),
        "peak_period": int(np.argmax(arr)),
    }


def build_rows(users: Dict[str, dict], stats: Dict[tuple, dict],
               dates: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for date in dates:
        label = md.calendar_label(date)
        for user, meta in sorted(users.items()):
            entry = stats.get((date, user))
            mapped = meta["meters"]
            converted = entry["converted_meters"] if entry else set()
            qualified = entry["qualified_meters"] if entry else set()
            user_complete = bool(converted)
            all_complete = bool(entry) and mapped <= converted
            partial = user_complete and not all_complete
            if entry is None:
                reason = "no_record"
            elif not qualified:
                reason = "not_qualified"
            elif not converted:
                reason = "incomplete_or_unconvertible"
            elif partial:
                reason = "partial_meters"
            else:
                reason = ""
            metrics = _curve_metrics(entry["curve"] if user_complete else None)
            rows.append({
                "date": date,
                **{k: label[k] for k in (
                    "season", "weekday", "day_type", "holiday_class",
                    "holiday_name", "holiday_offset")},
                "user_id": user,
                "bus": meta["bus"],
                "industry": meta["industry"],
                "in_v1_subset": meta["in_v1"],
                "n_meters": meta["n_meters"],
                "records": entry["records"] if entry else 0,
                "qualified_blocks": len(qualified),
                "converted_blocks": len(converted),
                "missing_meters": ";".join(sorted(mapped - converted)),
                "slots_min": (entry["min_slots"] if entry
                              and entry["records"] else 0),
                "slots_max": entry["max_slots"] if entry else 0,
                "flags": ";".join(sorted(entry["flags"])) if entry else "",
                "user_complete": user_complete,
                "all_meters_complete": all_complete,
                "partial_meters": partial,
                "missing_reason": reason,
                **metrics,
            })
    return rows


def _season_counts(dates) -> Dict[str, int]:
    counts = collections.Counter(md.calendar_label(d)["season"] for d in dates)
    return {s: int(counts.get(s, 0)) for s in SEASONS}


def provisional_core(users: Dict[str, dict], rows: Sequence[dict],
                     min_season_dates: int) -> tuple:
    """Largest coverage-ranked prefix with enough dates in every season."""
    by_user: Dict[str, set] = collections.defaultdict(set)
    for row in rows:
        if row["all_meters_complete"]:
            by_user[row["user_id"]].add(row["date"])
    ranked = sorted(users, key=lambda u: (-len(by_user[u]), u))
    curve = []
    best_k = 0
    best_common: set = set()
    common = None
    for k, user in enumerate(ranked, start=1):
        common = set(by_user[user]) if common is None else common & by_user[user]
        counts = _season_counts(common)
        curve.append({
            "k": k,
            "user": user,
            "dates": len(common),
            "season_counts": counts,
            "min_season_dates": min(counts.values()) if counts else 0,
        })
        if all(counts[s] >= min_season_dates for s in SEASONS):
            best_k = k
            best_common = set(common)
        elif best_k > 0:
            # The intersection only shrinks, so no larger prefix can recover
            # a season that has already fallen below the threshold.
            break
    return ranked[:best_k], best_common, ranked, curve


def build_date_summary(rows: Sequence[dict], users: Dict[str, dict],
                       core_users: Sequence[str], core_dates: set) -> List[dict]:
    by_date: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    core = set(core_users)
    v1 = {u for u, meta in users.items() if meta["in_v1"]}
    bus_of = {u: meta["bus"] for u, meta in users.items()}
    summary = []
    for date in sorted(by_date):
        day_rows = by_date[date]
        label = md.calendar_label(date)
        complete_users = [r for r in day_rows if r["user_complete"]]
        all_users = [r for r in day_rows if r["all_meters_complete"]]
        core_rows = [r for r in day_rows
                     if r["user_id"] in core and r["all_meters_complete"]]
        core_complete = len(core_rows) == len(core) and bool(core)
        summary.append({
            "date": date,
            **{k: label[k] for k in (
                "season", "weekday", "day_type", "holiday_class",
                "holiday_name", "holiday_offset")},
            "user_complete_users": len(complete_users),
            "all_meters_users": len(all_users),
            "user_complete_buses": len({r["bus"] for r in complete_users}),
            "all_meters_buses": len({r["bus"] for r in all_users}),
            "max_complete_users": len(complete_users),
            "max_all_meters_users": len(all_users),
            "all_41_complete": len(all_users) == len(users),
            "all_37_complete": len(all_users) >= 37,
            "all_35_complete": len(all_users) >= 35,
            "all_30_complete": len(all_users) >= 30,
            "core_complete": core_complete,
            "v1_17_user_complete": all(
                any(r["user_id"] == u and r["user_complete"] for r in day_rows)
                for u in v1),
            "v1_17_all_meters_complete": all(
                any(r["user_id"] == u and r["all_meters_complete"]
                    for r in day_rows) for u in v1),
            "core_buses_with_data": len({bus_of[u] for u in core
                                         if any(r["user_id"] == u
                                                and r["all_meters_complete"]
                                                for r in day_rows)}),
        })
    return summary


def write_curve_store(path: Path, rows: Sequence[dict],
                      stats: Dict[tuple, dict]) -> dict:
    """The 96-point user-day curves, for subset evaluations without re-reading."""
    arrays = {}
    for row in rows:
        if row["user_complete"]:
            key = f"{row['date']}|{row['user_id']}"
            arrays[key] = np.asarray(
                stats[(row["date"], row["user_id"])]["curve"], dtype=float)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "curves": len(arrays),
        "note": ("one key per user-day with at least one convertible qualified "
                 "meter; partial user-days are stored but flagged in the CSV"),
    }


def _write_report(path: Path, *, mapping_path: Path, mapping_sha: str,
                  workbook_path: Path, workbook_sha: str, users: Dict[str, dict],
                  v1_users: set, rows: Sequence[dict],
                  date_summary: Sequence[dict], core_users: Sequence[str],
                  core_curve: Sequence[dict], core_dates: set,
                  issues: dict, min_season_dates: int,
                  artifacts: Dict[str, str], command: str) -> Path:
    by_season = {s: [] for s in SEASONS}
    for row in date_summary:
        by_season[row["season"]].append(row)
    core_meta = [users[u] for u in core_users]
    core_buses = sorted({m["bus"] for m in core_meta})
    core_industries = sorted({m["industry"] for m in core_meta})
    core_season = _season_counts(core_dates)
    core_holidays = collections.Counter(
        md.calendar_label(d)["holiday_class"] for d in core_dates
        if md.calendar_label(d)["holiday_class"] not in
        ("ordinary", "makeup_workday"))

    lines = [
        "# V2 user-date coverage audit",
        "",
        "## Verdict",
        "",
        f"- Mapped users audited: **{len(users)}**",
        f"- Mapped buses: **{len({m['bus'] for m in users.values()})}**",
        f"- Calendar dates: **{len(date_summary)}**",
        f"- V1 subset users: **{len(v1_users)}** (kept as a historical "
        "benchmark, not used as the V2 selection rule)",
        f"- Provisional fixed core: **{len(core_users)} users** with all-meters "
        f"data on **{len(core_dates)} dates**, at least "
        f"{min_season_dates} dates in every season",
        "",
        "The V2 completeness standard is **all mapped meters complete**, not "
        "one meter per user. Partial user-days are counted and reported but "
        "are not eligible for the fixed scenario core. Nothing was scaled, "
        "interpolated, resampled, copied or filled.",
        "",
        "## Inputs",
        "",
        f"- Mapping: `{mapping_path}` (`{mapping_sha}`)",
        f"- Load workbook: `{workbook_path}` (`{workbook_sha}`)",
        f"- Records read: {issues['records']}",
        f"- Unmapped-user records ignored: {issues['unmapped_user_records']}",
        f"- Unmapped-meter records ignored: {issues['unmapped_meter_records']}",
        f"- Records with no usable date: {issues['missing_date_records']}",
        "",
        "## Seasonal coverage",
        "",
        "| season | dates | dates with ≥1 all-meters user | dates with all "
        "41 all-meters | dates with all 37 | dates with core complete | "
        "max all-meters users | users with ≥1 all-meters date | buses with "
        "≥1 all-meters date | max user-level complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for season in SEASONS:
        group = by_season[season]
        if not group:
            lines.append(f"| {season} | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            continue
        lines.append(
            f"| {season} | {len(group)} | "
            f"{sum(1 for r in group if r['all_meters_users'] > 0)} | "
            f"{sum(1 for r in group if r['all_41_complete'])} | "
            f"{sum(1 for r in group if r['all_37_complete'])} | "
            f"{sum(1 for r in group if r['core_complete'])} | "
            f"{max(r['all_meters_users'] for r in group)} | "
            f"{len({r['user_id'] for r in rows if r['season'] == season and r['all_meters_complete']})} | "
            f"{len({r['bus'] for r in rows if r['season'] == season and r['all_meters_complete']})} | "
            f"{max(r['user_complete_users'] for r in group)} |")
    lines += ["", "## Holiday coverage", "",
              "| holiday | date | all-meters users | core complete |",
              "|---|---|---:|---|"]
    holiday_rows = [r for r in date_summary
                    if r["holiday_class"] not in
                    ("ordinary", "makeup_workday")]
    for r in holiday_rows:
        lines.append(
            f"| {r['holiday_name']} (offset {r['holiday_offset']}) | "
            f"{r['date']} | {r['all_meters_users']} | "
            f"{'yes' if r['core_complete'] else 'no'} |")
    lines += ["", "## User summary", "",
              "| user | bus | industry | V1 | complete days | all-meters "
              "days | partial days |",
              "|---|---:|---|---:|---:|---:|---:|"]
    user_summary = collections.defaultdict(lambda: {
        "user_complete": 0, "all_meters": 0, "partial": 0})
    for row in rows:
        item = user_summary[row["user_id"]]
        item["user_complete"] += int(bool(row["user_complete"]))
        item["all_meters"] += int(bool(row["all_meters_complete"]))
        item["partial"] += int(bool(row["partial_meters"]))
    for user in sorted(users, key=lambda u: (
            -user_summary[u]["all_meters"], u)):
        meta = users[user]
        item = user_summary[user]
        lines.append(
            f"| {user} | {meta['bus']} | {meta['industry']} | "
            f"{'yes' if meta['in_v1'] else 'no'} | "
            f"{item['user_complete']} | {item['all_meters']} | "
            f"{item['partial']} |")
    lines += ["", "## Provisional fixed core", "",
              f"Rule: rank users by all-meters complete days, then take the "
              f"largest prefix that keeps at least {min_season_dates} dates in "
              "each season. This is a coverage-ranked candidate for Phase C, "
              "not a frozen V2 subset.", "",
              f"- Users: **{len(core_users)}**",
              f"- Buses: **{core_buses}** ({len(core_buses)} of "
              f"{len({m['bus'] for m in users.values()})})",
              f"- Industries: **{len(core_industries)}**",
              f"- Dates: **{len(core_dates)}** ({core_season})",
              f"- Holiday dates: {dict(core_holidays)}", "",
              "| user | bus | industry | V1 | all-meters days |",
              "|---|---:|---|---:|---:|"]
    for user in core_users:
        meta = users[user]
        lines.append(
            f"| {user} | {meta['bus']} | {meta['industry']} | "
            f"{'yes' if meta['in_v1'] else 'no'} | "
            f"{user_summary[user]['all_meters']} |")
    lines += ["", "## Coverage ladder", "",
              "| prefix k | dates all k complete | season counts |",
              "|---:|---:|---|"]
    for item in core_curve:
        if item["k"] > len(core_users) + 5:
            break
        lines.append(
            f"| {item['k']} | {item['dates']} | {item['season_counts']} |")
    lines += ["", "## Data gaps", "",
              "- No date carries all 41 mapped users at the all-meters "
              "standard; the maximum is 37.",
              "- A user-day with one meter missing is partial and is not "
              "admitted to the fixed core. Its load fields are computed only "
              "from the meters that converted and are marked partial.",
              "- Users with very few all-meters days cannot be part of a "
              "multi-season fixed subset; they remain in the audit but not in "
              "the provisional core.",
              "- The provisional core is a coverage choice, not a load-scale "
              "or physical-feasibility choice. Phase C must still check the "
              "network.",
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
        description="Audit all mapped users on all real dates for V2.")
    parser.add_argument("--mapping", default=str(FULL_MAPPING))
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--out", default="data/processed/v2")
    parser.add_argument("--min-season-dates", type=int, default=20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    mapping_path = root / args.mapping
    reports = root / args.reports
    out = root / args.out
    users, v1_users = load_mapping(mapping_path)
    mapping_sha = _sha256(mapping_path)
    workbook_path = il.resolve_workbook()
    workbook_sha = _sha256(workbook_path)

    print("reading workbook (this is the slow step)...", flush=True)
    records = il.flag_records(list(il.iter_meter_days()))
    print(f"  {len(records)} records", flush=True)
    stats, issues = collect(records, users)
    dates = sorted({d for d, _u in stats})
    # The store is keyed by date, user; every mapped user gets every date.
    missing_dates = sorted({_iso_date_or_none(r.date) for r in records}
                           - set(dates) - {None})
    dates = sorted(set(dates) | set(missing_dates))
    print(f"  {len(dates)} dates, {len(users)} users", flush=True)

    rows = build_rows(users, stats, dates)
    core_users, core_dates, ranked, coverage_curve = provisional_core(
        users, rows, args.min_season_dates)
    if not core_users:
        raise SystemExit(
            f"no coverage-ranked prefix keeps {args.min_season_dates} dates in "
            "every season; there is no feasible fixed core at this threshold")
    date_summary = build_date_summary(rows, users, core_users, core_dates)

    # Fill the core aggregate metrics from the stored curves.
    curve_index = {(d, u): stats[(d, u)]["curve"]
                   for (d, u) in stats}
    core_set = set(core_users)
    for row in date_summary:
        if not row["core_complete"]:
            continue
        curve = np.zeros(il.N_SLOTS)
        for user in core_set:
            curve += curve_index[(row["date"], user)]
        metrics = _curve_metrics(curve)
        row.update({
            "core_peak_mw": metrics["peak_mw"],
            "core_mean_mw": metrics["mean_mw"],
            "core_energy_mwh": metrics["energy_mwh"],
            "core_peak_period": metrics["peak_period"],
            "core_load_factor": (metrics["mean_mw"] / metrics["peak_mw"]
                                 if metrics["peak_mw"] else ""),
        })

    coverage_path = _write_csv(
        reports / "v2_user_date_coverage.csv", rows, COVERAGE_COLUMNS)
    date_summary_path = _write_csv(
        out / "v2_date_summary.csv", date_summary, DATE_SUMMARY_COLUMNS)
    user_summary_rows = []
    for user in sorted(users):
        meta = users[user]
        user_rows = [r for r in rows if r["user_id"] == user]
        all_dates = [r["date"] for r in user_rows if r["all_meters_complete"]]
        peaks = [float(r["peak_mw"]) for r in user_rows
                 if r["all_meters_complete"] and r["peak_mw"] != ""]
        energies = [float(r["energy_mwh"]) for r in user_rows
                    if r["all_meters_complete"] and r["energy_mwh"] != ""]
        user_summary_rows.append({
            "user_id": user,
            "bus": meta["bus"],
            "industry": meta["industry"],
            "n_meters": meta["n_meters"],
            "in_v1_subset": meta["in_v1"],
            "user_complete_days": sum(1 for r in user_rows
                                      if r["user_complete"]),
            "all_meters_days": len(all_dates),
            "partial_days": sum(1 for r in user_rows
                                if r["partial_meters"]),
            "qualified_days": sum(1 for r in user_rows
                                  if int(r["qualified_blocks"]) > 0),
            "first_all_meters_date": min(all_dates) if all_dates else "",
            "last_all_meters_date": max(all_dates) if all_dates else "",
            "mean_peak_mw": float(np.mean(peaks)) if peaks else "",
            "mean_energy_mwh": float(np.mean(energies)) if energies else "",
        })
    user_summary_path = _write_csv(
        out / "v2_user_summary.csv", user_summary_rows, USER_SUMMARY_COLUMNS)

    # Wide completeness matrix: 1 = all meters, 0.5 = partial, 0 = none.
    matrix_path = reports / "v2_user_date_completeness_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "bus", "industry", "in_v1_subset",
                         "all_meters_days", "user_complete_days", *dates])
        for user in sorted(users):
            meta = users[user]
            user_rows = {r["date"]: r for r in rows if r["user_id"] == user}
            values = []
            for date in dates:
                row = user_rows[date]
                values.append("1" if row["all_meters_complete"]
                              else "0.5" if row["user_complete"] else "0")
            writer.writerow([
                user, meta["bus"], meta["industry"], meta["in_v1"],
                sum(1 for r in user_rows.values() if r["all_meters_complete"]),
                sum(1 for r in user_rows.values() if r["user_complete"]),
                *values,
            ])

    curve_path = out / "v2_user_day_curves.npz"
    curve_manifest = write_curve_store(curve_path, rows, stats)
    core_payload = {
        "rule": (f"rank users by all-meters complete days, take the largest "
                 f"prefix with at least {args.min_season_dates} dates in every "
                 "season"),
        "min_season_dates": args.min_season_dates,
        "core_users": list(core_users),
        "core_buses": sorted({users[u]["bus"] for u in core_users}),
        "core_industries": sorted({users[u]["industry"] for u in core_users}),
        "core_dates": sorted(core_dates),
        "core_dates_count": len(core_dates),
        "season_counts": _season_counts(core_dates),
        "holiday_counts": dict(collections.Counter(
            md.calendar_label(d)["holiday_class"] for d in core_dates
            if md.calendar_label(d)["holiday_class"] not in
            ("ordinary", "makeup_workday"))),
        "coverage_ladder": coverage_curve,
        "input_sha256": {
            "mapping": mapping_sha,
            "load_workbook": workbook_sha,
        },
        "curve_store": curve_manifest,
    }
    core_path = _write_json(out / "v2_provisional_core.json", core_payload)

    artifacts = {
        str(coverage_path): _sha256(coverage_path),
        str(matrix_path): _sha256(matrix_path),
        str(date_summary_path): _sha256(date_summary_path),
        str(user_summary_path): _sha256(user_summary_path),
        str(core_path): _sha256(core_path),
        str(curve_path): curve_manifest["sha256"],
    }
    report_path = _write_report(
        reports / "v2_user_date_coverage.md",
        mapping_path=mapping_path, mapping_sha=mapping_sha,
        workbook_path=workbook_path, workbook_sha=workbook_sha,
        users=users, v1_users=v1_users, rows=rows,
        date_summary=date_summary, core_users=core_users,
        core_curve=coverage_curve, core_dates=core_dates,
        issues=issues, min_season_dates=args.min_season_dates,
        artifacts=artifacts, command=" ".join(sys.argv))

    print(f"mapped users            {len(users)}")
    print(f"dates                   {len(dates)}")
    print(f"provisional core        {len(core_users)} users / "
          f"{len(core_dates)} dates")
    print(f"  seasons               {_season_counts(core_dates)}")
    print(f"wrote {coverage_path}")
    print(f"wrote {matrix_path}")
    print(f"wrote {date_summary_path}")
    print(f"wrote {user_summary_path}")
    print(f"wrote {core_path}")
    print(f"wrote {curve_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
