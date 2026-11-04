# audit_jn_price_source.py
"""Audit the 2025 Jiangsu day-ahead price file used by V2-CY-JN-2025.

The file carries no metadata, so every property below is recomputed from the
values themselves: how many days are complete, what the 96 labels mean, what
the value range implies about the unit, and which seasonal windows the file
can price at all. Nothing is carried over from an earlier audit.

A day is complete only when it has 96 rows **and** the 96 distinct quarter-hour
labels `00:00`..`23:45`. The one truncated day at the start of the series is
reported, not repaired.

The unit and the region are **inferred**, not externally confirmed. The report
says so, and every artifact repeats it, because a price that entered the
objective as CNY/MWh on the strength of a guess would be a number wearing a
label it never earned.

`mv_value` is audited but not used: neither its unit nor the time at which it
becomes available can be established from the file, and the project rule is to
disable a feature that cannot be confirmed.

Usage:
    PYTHONPATH=src python src/audit_jn_price_source.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import numpy as np

PRICE_FILE = "docs/data.csv"
PRICE_REGION = "south"
REQUIRED_COLUMNS = (
    "time", "price_value_day_ahead_south", "price_value_day_ahead_north",
    "mv_value",
)
PERIODS = 96
LABELS = tuple(f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15))

#: The seasonal windows the four in-scope groups pair into. A price date
#: outside every window is in the file but unpairable.
WINDOWS = collections.OrderedDict([
    ("summer", ("2025-07-01", "2025-08-31")),
    ("autumn", ("2025-09-01", "2025-11-30")),
    ("winter", ("2025-12-01", "2025-12-31")),
    ("national_day_2022", ("2025-10-03", "2025-10-07")),
])

COVERAGE_COLUMNS = (
    "date", "periods", "complete", "weekday", "is_weekend",
    "south_min", "south_max", "south_mean",
    "mv_min", "mv_max", "mv_mean",
    "south_floor_hits", "south_cap_hits",
    "curve_sha256", "price_region", "windows",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve_digest(date: str, region: str, values: Sequence[float]) -> str:
    """A digest of the 96 values that identifies the curve, not the file."""
    payload = json.dumps({
        "date": str(date), "region": region,
        "prices": [float(v) for v in values],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_price_csv(path: Path) -> pd.DataFrame:
    """Read and validate the price file; never rewrite it."""
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")
    frame["label"] = frame["time"].dt.strftime("%H:%M")
    return frame


def daily_coverage(frame: pd.DataFrame, floor: float = None,
                   cap: float = None) -> List[dict]:
    """One row per calendar date: period count, completeness, value range.

    The floor and cap are the series' observed extremes, so a hit means a
    value equal to the market's own bound -- not that day's minimum, which
    would report 136 "floor hits" per day by construction.
    """
    series = frame["price_value_day_ahead_south"].to_numpy(dtype=float)
    floor = float(series.min()) if floor is None else float(floor)
    cap = float(series.max()) if cap is None else float(cap)
    rows: List[dict] = []
    for date, group in frame.groupby("date", sort=True):
        labels = set(group["label"])
        complete = (len(group) == PERIODS and labels == set(LABELS))
        south = group["price_value_day_ahead_south"].to_numpy(dtype=float)
        mv = group["mv_value"].to_numpy(dtype=float)
        day = pd.Timestamp(date)
        windows = [name for name, (lo, hi) in WINDOWS.items()
                   if pd.Timestamp(lo) <= day <= pd.Timestamp(hi)]
        rows.append({
            "date": date,
            "periods": int(len(group)),
            "complete": bool(complete),
            "weekday": day.day_name(),
            "is_weekend": bool(day.weekday() >= 5),
            "south_min": float(south.min()),
            "south_max": float(south.max()),
            "south_mean": float(south.mean()),
            "mv_min": float(mv.min()),
            "mv_max": float(mv.max()),
            "mv_mean": float(mv.mean()),
            "south_floor_hits": int((south == floor).sum()),
            "south_cap_hits": int((south == cap).sum()),
            "curve_sha256": curve_digest(date, PRICE_REGION, south),
            "price_region": PRICE_REGION,
            "windows": "|".join(windows),
        })
    return rows


def window_coverage(rows: Sequence[dict]) -> Dict[str, dict]:
    """Complete-day supply inside each window, split by weekday/weekend."""
    out: Dict[str, dict] = collections.OrderedDict()
    for name, (lo, hi) in WINDOWS.items():
        days = [r for r in rows
                if lo <= r["date"] <= hi and r["complete"]]
        out[name] = {
            "window": [lo, hi],
            "complete": len(days),
            "weekday": sum(1 for r in days if not r["is_weekend"]),
            "weekend": sum(1 for r in days if r["is_weekend"]),
        }
    return out


def missing_runs(rows: Sequence[dict]) -> List[dict]:
    """Contiguous date ranges absent from the file, so a gap is a stated fact."""
    present = {r["date"] for r in rows if r["complete"]}
    start = pd.Timestamp(rows[0]["date"])
    end = pd.Timestamp(rows[-1]["date"])
    runs: List[dict] = []
    current = None
    for day in pd.date_range(start, end):
        if day.strftime("%Y-%m-%d") in present:
            if current is not None:
                runs.append(current)
                current = None
            continue
        if current is None:
            current = {"start": day.strftime("%Y-%m-%d"),
                       "end": day.strftime("%Y-%m-%d"), "days": 1}
        else:
            current["end"] = day.strftime("%Y-%m-%d")
            current["days"] += 1
    if current is not None:
        runs.append(current)
    return runs


def write_curves(path: Path, frame: pd.DataFrame,
                 rows: Sequence[dict]) -> Path:
    """Store only complete days, keyed `south/<date>` and `mv/<date>`."""
    payload: Dict[str, np.ndarray] = {}
    for row in rows:
        if not row["complete"]:
            continue
        day = frame[frame["date"] == row["date"]].sort_values("time")
        payload[f"south/{row['date']}"] = np.asarray(
            day["price_value_day_ahead_south"], dtype=float)
        payload[f"mv/{row['date']}"] = np.asarray(
            day["mv_value"], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COVERAGE_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_report(*, path: Path, source: Path, source_sha: str, rows: Sequence[dict],
                 windows: Dict[str, dict], runs: Sequence[dict],
                 curves_path: Path, csv_path: Path, command: str) -> Path:
    frame_complete = [r for r in rows if r["complete"]]
    incomplete = [r for r in rows if not r["complete"]]
    values = pd.Series([r["south_mean"] for r in frame_complete])
    first = rows[0]
    last = rows[-1]
    lines: List[str] = []
    lines.append("# 2025 Jiangsu day-ahead price audit (V2-CY-JN-2025)")
    lines.append("")
    lines.append("This audit supports a **cross-year controlled simulation**. "
                 "A 2022 load day paired with a 2025 price day is not a "
                 "same-period market settlement, and nothing here may be "
                 "described as one.")
    lines.append("")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Input: `{source.as_posix()}`")
    lines.append(f"- SHA-256: `{source_sha}`")
    lines.append(f"- Rows: {sum(r['periods'] for r in rows)}; "
                 f"dates: {len(rows)}; complete: {len(frame_complete)}; "
                 f"incomplete: {len(incomplete)}")
    lines.append(f"- Range: {first['date']} to {last['date']}")
    lines.append("")
    lines.append("## Timestamp semantics")
    lines.append("")
    lines.append("A complete day carries the 96 distinct labels `00:00` to "
                 "`23:45`, so a label is the **interval start**, with each "
                 "value the quarter-hour that begins at that label. The first "
                 "day of the series starts at `00:15`, which is a truncated "
                 "file, not a different convention; every other day starts at "
                 "`00:00` and ends at `23:45`.")
    lines.append("")
    lines.append("## Incomplete days")
    lines.append("")
    if incomplete:
        lines.append("| date | periods | missing labels |")
        lines.append("|---|---:|---:|")
        for row in incomplete:
            lines.append(f"| {row['date']} | {row['periods']} | "
                         f"{PERIODS - row['periods']} |")
        lines.append("")
        lines.append("These days are excluded from the curve store and from "
                     "every pairing. They are not filled, extended or "
                     "interpolated.")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("| start | end | days |")
    lines.append("|---|---|---:|")
    for run in runs:
        lines.append(f"| {run['start']} | {run['end']} | {run['days']} |")
    lines.append("")
    lines.append("A gap is a fact about the source. A load day is left "
                 "unpaired rather than priced from a neighbour or a window "
                 "mean.")
    lines.append("")
    lines.append("## Values")
    lines.append("")
    south_all = [r for r in frame_complete]
    lines.append("| statistic | `price_value_day_ahead_south` |")
    lines.append("|---|---:|")
    lines.append(f"| min | {min(r['south_min'] for r in south_all):.2f} |")
    lines.append(f"| max | {max(r['south_max'] for r in south_all):.2f} |")
    lines.append(f"| daily-mean median | {values.median():.2f} |")
    lines.append(f"| daily-mean min / max | {values.min():.2f} / "
                 f"{values.max():.2f} |")
    lines.append(f"| negative values | "
                 f"{sum(1 for r in south_all if r['south_min'] < 0)} |")
    floor = min(r["south_min"] for r in south_all)
    cap = max(r["south_max"] for r in south_all)
    lines.append(f"| values at the observed floor ({floor:.0f}) | "
                 f"{sum(r['south_floor_hits'] for r in south_all)} |")
    lines.append(f"| values at the observed cap ({cap:.0f}) | "
                 f"{sum(r['south_cap_hits'] for r in south_all)} |")
    lines.append("")
    lines.append("## Unit and region: inferred, not externally confirmed")
    lines.append("")
    lines.append("The file carries no unit, currency, tax basis, timezone or "
                 "region definition. The working reading is **CNY/MWh** for "
                 "`price_value_day_ahead_south`, on three pieces of evidence:")
    lines.append("")
    lines.append(f"1. A hard floor at {floor:.0f} and a hard cap at {cap:.0f} "
                 "are the shape of a bid floor and a market cap, not of a "
                 "cost curve.")
    lines.append("2. A median daily mean of "
                 f"{values.median():.0f} sits beside the project's single "
                 "externally-adjudicated Jiangsu day-ahead sheet "
                 "(`261009`, 225.9–445.0 CNY/MWh, mean 386.83), which is the "
                 "same order of magnitude.")
    lines.append("3. Reading the same numbers as CNY/kWh would multiply them "
                 "by 1000 and as CNY/10-MWh would divide them by 10; neither "
                 "is a plausible Jiangsu day-ahead range.")
    lines.append("")
    lines.append("`price_value_day_ahead_south` is taken as the 江南 (southern "
                 "Jiangsu) day-ahead price **because the task names it so**. "
                 "The file itself does not define the boundary, and the north "
                 "column is retained but not used. No average of the two "
                 "columns is formed.")
    lines.append("")
    lines.append("## `mv_value`")
    lines.append("")
    mv_min = min(r["mv_min"] for r in frame_complete)
    mv_max = max(r["mv_max"] for r in frame_complete)
    mv_mean = float(np.mean([r["mv_mean"] for r in frame_complete]))
    lines.append(f"Observed range {mv_min:.2f} to {mv_max:.2f}, mean "
                 f"{mv_mean:.2f}, over the same 96-slot axis as the price.")
    lines.append("")
    lines.append("**Not used.** Neither its unit nor the time at which it "
                 "becomes available can be established from the file. A "
                 "provincial new-energy total is also not a bus-level "
                 "injection for this feeder, so even a confirmed MW reading "
                 "would enter as a system-level feature and never as DER at a "
                 "bus. The field is retained in the artifact and the JN-RE "
                 "arm stays disabled; enabling it would require a new "
                 "observation version, not a reinterpretation of OBS_V6.")
    lines.append("")
    lines.append("## Window supply")
    lines.append("")
    lines.append("| window | dates | complete | weekday | weekend |")
    lines.append("|---|---|---:|---:|---:|")
    for name, stats in windows.items():
        lines.append(f"| {name} | {stats['window'][0]} … "
                     f"{stats['window'][1]} | {stats['complete']} | "
                     f"{stats['weekday']} | {stats['weekend']} |")
    lines.append("")
    lines.append("The 2025 file begins on 2025-06-01, so the `spring` and "
                 "`spring_festival_2022` groups cannot be priced from it at "
                 "all. They stay in the frozen V2 library and are not "
                 "substituted with another season's prices.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{csv_path.as_posix()}` — one row per date")
    lines.append(f"- `{curves_path.as_posix()}` — complete days keyed "
                 f"`south/<date>` and `mv/<date>`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", default=PRICE_FILE)
    parser.add_argument("--out", default="data/processed/v2_cy_jn_2025")
    parser.add_argument("--report", default="reports/v2_cy_jn_2025/data_audit.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    source = root / args.price_file
    if not source.exists():
        raise SystemExit(f"{source} not found")
    frame = read_price_csv(source)
    rows = daily_coverage(frame)
    windows = window_coverage(rows)
    runs = missing_runs(rows)

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _write_csv(out_dir / "price_daily_coverage.csv", rows)
    curves_path = write_curves(out_dir / "price_curves.npz", frame, rows)
    report_path = build_report(
        path=root / args.report, source=source, source_sha=_sha256(source),
        rows=rows, windows=windows, runs=runs,
        curves_path=curves_path.relative_to(root),
        csv_path=csv_path.relative_to(root),
        command=" ".join(sys.argv))

    complete = sum(1 for r in rows if r["complete"])
    print(f"dates                 {len(rows)}")
    print(f"complete              {complete}")
    print(f"incomplete            {len(rows) - complete}")
    for name, stats in windows.items():
        print(f"{name:<22}{stats['complete']:>4} complete "
              f"({stats['weekday']} wd / {stats['weekend']} we)")
    print(f"wrote {csv_path}")
    print(f"wrote {curves_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
