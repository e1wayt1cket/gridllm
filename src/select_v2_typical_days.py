# select_v2_typical_days.py
"""Choose a multi-season candidate library from real, complete user-days.

The candidate universe is every date on which the provisional fixed core is
complete at the all-meters standard. Within each season and day type the
representative days are chosen by proximity to the stratum's median peak and
energy, not by taking the largest or smallest load. Holiday representatives
are chosen by the holiday's own relative position (eve, day 0, later days), so
a fixed Gregorian month-day is never used as a holiday rule.

Every selected date is a real date with complete measured data. No curve is
scaled, repeated or synthesised. The provisional core is a coverage candidate;
this script does not freeze the final V2 user subset.

Usage:
    PYTHONPATH=src python src/select_v2_typical_days.py
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

import multiday_dataset as md

CANDIDATE_COLUMNS = (
    "date", "year", "season", "month", "weekday", "day_type",
    "holiday_class", "holiday_name", "holiday_offset", "core_users",
    "core_buses", "core_peak_mw", "core_mean_mw", "core_energy_mwh",
    "core_peak_period", "core_load_factor", "v1_17_user_complete",
    "v1_17_all_meters_complete", "representative", "selection_reason",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _num(row: dict, key: str) -> float:
    return float(row[key])


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _stratum_representatives(rows: Sequence[dict], season: str,
                             day_type: str, limit: int = 3) -> List[dict]:
    """The rows closest to the stratum's median peak and energy."""
    if not rows:
        return []
    peaks = [_num(r, "core_peak_mw") for r in rows]
    energies = [_num(r, "core_energy_mwh") for r in rows]
    med_peak = _median(peaks)
    med_energy = _median(energies)
    scored = []
    for row, peak, energy in zip(rows, peaks, energies):
        score = (abs(peak - med_peak) / med_peak if med_peak else 0.0)
        score += (abs(energy - med_energy) / med_energy
                  if med_energy else 0.0)
        scored.append((score, row["date"], row))
    scored.sort(key=lambda item: (item[0], item[1]))
    chosen = []
    for _score, _date, row in scored[:limit]:
        row = dict(row)
        row["representative"] = True
        row["selection_reason"] = (
            f"closest_to_{season}_{day_type}_median")
        chosen.append(row)
    return chosen


def _holiday_representatives(rows: Sequence[dict], season: str,
                             limit: int = 3) -> List[dict]:
    """Up to three positions per real holiday instance.

    The grouping is `(holiday_class, year)`, not the month-day: the 2021 and
    2022 Spring Festival are different real holiday instances even when a
    Gregorian date looks similar.
    """
    groups: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["holiday_class"], row["date"][:4])].append(row)
    chosen: List[dict] = []
    for (holiday_class, year), group in sorted(groups.items()):
        by_offset = {}
        for row in group:
            offset = int(row["holiday_offset"]) if row["holiday_offset"] != "" \
                else None
            if offset is not None and offset not in by_offset:
                by_offset[offset] = row
        offsets = sorted(by_offset)
        if not offsets:
            continue
        preferred = []
        for target in (0, -1, 1):
            if target in by_offset and by_offset[target] not in preferred:
                preferred.append(by_offset[target])
        for offset in (offsets[0], offsets[-1]):
            if by_offset[offset] not in preferred:
                preferred.append(by_offset[offset])
        for offset in offsets:
            if len(preferred) >= limit:
                break
            if by_offset[offset] not in preferred:
                preferred.append(by_offset[offset])
        for row in preferred[:limit]:
            row = dict(row)
            row["representative"] = True
            offset = row["holiday_offset"]
            row["selection_reason"] = (
                f"holiday_{holiday_class}_{year}_offset_{offset}")
            chosen.append(row)
    return chosen


def select(candidates: Sequence[dict], limit: int = 3) -> List[dict]:
    """Mark representatives and return every candidate row, sorted by date."""
    selected = []
    by_season = collections.defaultdict(list)
    for row in candidates:
        by_season[row["season"]].append(row)
    for season in ("spring", "summer", "autumn", "winter"):
        rows = by_season.get(season, [])
        for day_type in ("weekday", "weekend"):
            selected.extend(_stratum_representatives(
                [r for r in rows if r["day_type"] == day_type],
                season, day_type, limit))
        selected.extend(_holiday_representatives(
            [r for r in rows if r["day_type"] == "holiday"], season, limit))
    chosen_dates = {r["date"] for r in selected}
    out = []
    for row in sorted(candidates, key=lambda r: r["date"]):
        row = dict(row)
        if row["date"] in chosen_dates:
            # Preserve the reason attached to the selected copy.
            selected_row = next(r for r in selected if r["date"] == row["date"])
            row["representative"] = True
            row["selection_reason"] = selected_row["selection_reason"]
        else:
            row["representative"] = False
            row["selection_reason"] = ""
        out.append(row)
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CANDIDATE_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, *, core: dict, rows: Sequence[dict],
                  command: str, csv_path: Path, hashes: Dict[str, str]) -> Path:
    reps = [r for r in rows if _as_bool(r["representative"])]
    by_season = collections.Counter(r["season"] for r in rows)
    by_season_reps = collections.Counter(r["season"] for r in reps)
    by_type = collections.Counter(r["day_type"] for r in rows)
    by_type_reps = collections.Counter(r["day_type"] for r in reps)
    lines = [
        "# V2 typical-day selection",
        "",
        "## Verdict",
        "",
        f"- Candidate universe: **{len(rows)} real dates** on which the "
        f"provisional **{len(core['core_users'])}-user** core is complete at "
        "the all-meters standard.",
        f"- Representative dates: **{len(reps)}**",
        f"- Seasons represented: **{', '.join(sorted(by_season))}**",
        "- No date is scaled, copied, interpolated or synthesised. Holiday "
        "positions are taken from the actual holiday period.",
        "",
        "## Source",
        "",
        f"- Provisional core: {len(core['core_users'])} users, "
        f"{len(core['core_buses'])} buses, {len(core['core_industries'])} "
        "industries.",
        f"- Core dates: {core['core_dates_count']} "
        f"({core['season_counts']}).",
        f"- Core holiday counts: {core['holiday_counts']}.",
        "- The core is a coverage candidate from Phase A; the final V2 subset "
        "is not frozen.",
        "",
        "## Candidate coverage",
        "",
        "| season | candidates | representative |",
        "|---|---:|---:|",
    ]
    for season in ("spring", "summer", "autumn", "winter"):
        lines.append(f"| {season} | {by_season.get(season, 0)} | "
                     f"{by_season_reps.get(season, 0)} |")
    lines += ["", "| day type | candidates | representative |",
              "|---|---:|---:|"]
    for day_type in ("weekday", "weekend", "holiday"):
        lines.append(f"| {day_type} | {by_type.get(day_type, 0)} | "
                     f"{by_type_reps.get(day_type, 0)} |")
    lines += ["", "| season | weekday | weekend | holiday |",
              "|---|---:|---:|---:|"]
    for season in ("spring", "summer", "autumn", "winter"):
        counts = collections.Counter(
            r["day_type"] for r in rows if r["season"] == season)
        lines.append(
            f"| {season} | {counts.get('weekday', 0)} | "
            f"{counts.get('weekend', 0)} | {counts.get('holiday', 0)} |")
    lines += ["", "## Representative dates", "",
              "| date | season | day type | holiday | offset | peak MW | "
              "energy MWh | load factor | reason |",
              "|---|---|---|---|---:|---:|---:|---:|---|"]
    for row in sorted(reps, key=lambda r: r["date"]):
        lines.append(
            f"| {row['date']} | {row['season']} | {row['day_type']} | "
            f"{row['holiday_name']} | {row['holiday_offset']} | "
            f"{_num(row, 'core_peak_mw'):.3f} | "
            f"{_num(row, 'core_energy_mwh'):.3f} | "
            f"{_num(row, 'core_load_factor'):.3f} | "
            f"{row['selection_reason']} |")
    lines += ["", "## Selection rule", "",
              "- Ordinary weekdays and weekends: the three candidates closest "
              "to the stratum's median coincident peak and median daily "
              "energy. This is a centrality rule, not an extreme-value rule.",
              "- Holidays: up to three real positions per holiday instance, "
              "preferring the festival day (offset 0), the eve (-1) and the "
              "next day (+1), then the earliest and latest available positions.",
              "- Holiday instances are grouped by holiday class and year, so "
              "the 2021 and 2022 Spring Festivals are not collapsed into one "
              "fixed month-day.",
              "",
              "## Data gaps",
              "",
              "- A season or day type with no row above has no complete real "
              "candidate under the provisional core; it is reported, not "
              "filled.",
              "- The candidate loads are the core's measured load. They are "
              "not required to match the 3.715 MW IEEE33 reference; that "
              "comparison belongs to Phase C.",
              "- Price coverage is not part of this file; it is audited "
              "separately and remains the blocking input for training.",
              "",
              "## Artifacts", "",
              f"- `{csv_path}`",
              "",
              "| path | sha256 |", "|---|---|"]
    for path_text, digest in hashes.items():
        lines.append(f"| `{path_text}` | `{digest}` |")
    lines += ["", "## Command", "", f"`{command}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select V2 typical load days from complete real dates.")
    parser.add_argument("--date-summary",
                        default="data/processed/v2/v2_date_summary.csv")
    parser.add_argument("--core",
                        default="data/processed/v2/v2_provisional_core.json")
    parser.add_argument("--out", default="data/processed/v2_typical_day_candidates.csv")
    parser.add_argument("--report", default="reports/v2_typical_day_selection.md")
    parser.add_argument("--limit", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    summary_path = root / args.date_summary
    core_path = root / args.core
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found; run the V2 audit first")
    if not core_path.exists():
        raise SystemExit(f"{core_path} not found; run the V2 audit first")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    candidates = []
    for row in _read_csv(summary_path):
        if not _as_bool(row["core_complete"]):
            continue
        row = dict(row)
        row["year"] = row["date"][:4]
        row["month"] = row["date"][5:7]
        candidates.append(row)
    if not candidates:
        raise SystemExit(
            "no core-complete dates in the date summary; there is no "
            "candidate library to select from")
    rows = select(candidates, args.limit)
    out_path = root / args.out
    _write_csv(out_path, rows)
    report_path = root / args.report
    hashes = {
        str(summary_path): _sha256(summary_path),
        str(core_path): _sha256(core_path),
        str(out_path): _sha256(out_path),
    }
    _write_report(report_path, core=core, rows=rows,
                  command=" ".join(sys.argv), csv_path=out_path, hashes=hashes)
    print(f"candidate universe      {len(candidates)}")
    print(f"representative dates    {sum(1 for r in rows if _as_bool(r['representative']))}")
    print(f"seasons                 "
          f"{dict(collections.Counter(r['season'] for r in rows))}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
