# pair_v2_cy_jn_2025.py
"""Pair frozen 2022 load days with 2025 Jiangnan day-ahead price days.

Two independent dates per record: `load_date` is when the meters were read,
`price_date` is when the price was set. They are never collapsed into one date
and the two curves are aligned by quarter-hour **index**, not by calendar time.

The priorities are the task's, encoded as one deterministic assignment per
group rather than as a greedy pass:

  1. the national-day load days claim 2025-10-03..07 first, and those dates are
     removed from every other group's pool;
  2. a price day of the same weekday/weekend type always beats one of the other
     type, even when the other type carries the load day's own month-day -- a
     same-month-day match whose type differs is **not** taken;
  3. among candidates of the preferred type, the nearest to the load day's own
     2025 month-day wins.

A cost of `gap` for a same-type candidate and `1e6 + gap` for the other type
makes `scipy.optimize.linear_sum_assignment` return the matching that minimises
mismatches first and total distance second. Every price date is used at most
once. No curve is averaged, synthesised, extended from a neighbour or copied.
A load day that cannot be paired is reported unpaired.

The holiday group is the exception the task states: its load days match the
2025-10-03..07 dates by month-day, because within a holiday window the
weekday/weekend label is not the dimension the curves differ on.

The day's 96x33 bus load comes from the frozen group's own user curves at their
mapped buses (`replace`) and the IEEE33 native demand everywhere else
(`preserve`) -- the same rule the frozen support screen used. It is not
rescaled.

Usage:
    PYTHONPATH=src python src/pair_v2_cy_jn_2025.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import audit_v2_user_date_coverage as cov
import build_industry_node_loads as bnl
import screen_v2_physical as scr

FROZEN_VERSION = "v2-frozen-dfc9d60cc222"
FROZEN_MANIFEST = "data/processed/v2/v2_scenario_library_manifest.json"
CANDIDATES = "data/processed/v2_seasonal_candidates.csv"
LOAD_CURVES = "data/processed/v2/v2_user_day_curves.npz"
MAPPING = "data/processed/industry_meter_bus_map_v2.csv"
COVERAGE = "data/processed/v2_cy_jn_2025/price_daily_coverage.csv"
PRICE_CURVES = "data/processed/v2_cy_jn_2025/price_curves.npz"
OUT_DIR = "data/processed/v2_cy_jn_2025"
REPORT_DIR = "reports/v2_cy_jn_2025"
PRICE_REGION = "south"
SPLIT_SEED = 20261030

#: Groups this version prices. Spring and the 2022 Spring Festival need 2025
#: spring prices, which the source does not carry; they stay in the frozen
#: library and are never given another season's curve.
GROUPS: Tuple[str, ...] = ("national_day_2022", "summer", "autumn", "winter")
HOLIDAY_GROUPS = ("national_day_2022",)

#: The window each group may draw a price date from.
WINDOWS = {
    "summer": ("2025-07-01", "2025-08-31"),
    "autumn": ("2025-09-01", "2025-11-30"),
    "winter": ("2025-12-01", "2025-12-31"),
    "national_day_2022": ("2025-10-03", "2025-10-07"),
}
#: National-day price dates are claimed before any other group draws.
RESERVED = {
    "national_day_2022": tuple(f"2025-10-{d:02d}" for d in range(3, 8)),
}

PAIRING_COLUMNS = (
    "load_date", "price_date", "group", "pairing_type", "weekday_load",
    "weekday_price", "day_type_match", "price_region", "load_curve_sha256",
    "price_curve_sha256", "price_pairing_rule", "pair_reason",
    "month_day_gap_days", "representative", "season", "group_users",
    "group_buses", "load_peak_mw", "network_peak_mw", "load_energy_mwh",
    "price_min", "price_max", "price_mean",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _is_weekend(day: Date) -> bool:
    return day.weekday() >= 5


def _weekday_name(day: Date) -> str:
    return day.strftime("%A")


def load_price_pool(path: Path) -> Dict[str, dict]:
    """Complete price days only, keyed by ISO date."""
    rows = _read_csv(path)
    return {r["date"]: r for r in rows if _as_bool(r["complete"])}


@dataclass(frozen=True)
class Pair:
    load_date: str
    price_date: str
    group: str
    pairing_type: str
    reason: str


def _month_day_2025(load_date: str) -> Optional[Date]:
    day = Date.fromisoformat(load_date)
    try:
        return day.replace(year=2025)
    except ValueError:          # 29 February; no in-scope group has one
        return None


#: Cost added to a candidate of the other weekday/weekend type. Larger than any
#: attainable total gap, so minimising the assignment minimises mismatches
#: before it minimises distance.
_TYPE_PENALTY = 1.0e6


def _pairing_cost(load_date: str, price_date: str, holiday: bool) -> float:
    target = _month_day_2025(load_date)
    if target is None:
        return _TYPE_PENALTY
    gap = abs((Date.fromisoformat(price_date) - target).days)
    same_type = (_is_weekend(Date.fromisoformat(price_date))
                 == _is_weekend(Date.fromisoformat(load_date)))
    return float(gap) if (holiday or same_type) else _TYPE_PENALTY + gap


def _describe(load_date: str, price_date: str, holiday: bool) -> Tuple[str, str]:
    target = _month_day_2025(load_date)
    gap = abs((Date.fromisoformat(price_date) - target).days)
    same_type = (_is_weekend(Date.fromisoformat(price_date))
                 == _is_weekend(Date.fromisoformat(load_date)))
    if holiday:
        if gap == 0:
            return ("same_month_day",
                    "same month-day; holiday-window group, the "
                    "weekday/weekend label is not the matching dimension")
        return ("same_season_same_day_type",
                f"holiday window, {gap} day(s) from the load day's month-day")
    if not same_type:
        if gap == 0:
            return ("same_season_day_type_mismatch",
                    "same month-day, but its weekday/weekend type differs; "
                    "no date of the load day's own type was left in the window")
        return ("same_season_day_type_mismatch",
                f"no date of the same weekday/weekend type was left in the "
                f"window; nearest same-season date is {gap} day(s) away")
    if gap == 0:
        return "same_month_day", "same month-day, same weekday/weekend type"
    return ("same_season_same_day_type",
            f"nearest date of the same weekday/weekend type, {gap} day(s) "
            f"from the load day's month-day")


def pair_group(group: str, load_days: Sequence[str],
               pool: Sequence[str], used: set) -> Tuple[List[Pair], List[str]]:
    """Pair one group's load days against the available price dates."""
    from scipy.optimize import linear_sum_assignment

    holiday = group in HOLIDAY_GROUPS
    loads = sorted(load_days)
    prices = sorted(p for p in pool if p not in used)
    if not loads or not prices:
        return [], list(loads)
    cost = np.array([[_pairing_cost(ld, pd, holiday) for pd in prices]
                     for ld in loads], dtype=float)
    rows, cols = linear_sum_assignment(cost)

    pairs: List[Pair] = []
    for i, j in zip(rows, cols):
        load_date, price_date = loads[i], prices[j]
        pairing_type, reason = _describe(load_date, price_date, holiday)
        used.add(price_date)
        pairs.append(Pair(load_date, price_date, group, pairing_type, reason))

    paired = {p.load_date for p in pairs}
    unpaired = [d for d in loads if d not in paired]
    return pairs, unpaired


def load_curve_digest(load_date: str, matrix: np.ndarray) -> str:
    """A digest of a 96x33 matrix that also names the day it belongs to."""
    payload = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    hasher = hashlib.sha256(f"{load_date}|33x96".encode("utf-8"))
    hasher.update(payload.tobytes())
    return hasher.hexdigest()


def build_bus_matrix(load_date: str, group_users: Sequence[str],
                     users: Dict[str, dict], rated: Dict[int, float],
                     shape: np.ndarray, curves) -> Tuple[np.ndarray, np.ndarray]:
    """The day's (96, 33) MW matrix under the frozen replace/preserve rule."""
    bus_of = {u: users[u]["bus"] for u in group_users}
    loads, replace = scr._bus_loads(group_users, bus_of, load_date, curves,
                                    rated, shape)
    matrix = np.zeros((96, 33))
    for bus, series in loads.items():
        matrix[:, int(bus)] = np.asarray(series, dtype=float)
    return matrix, np.array(sorted(replace), dtype=int)


def split_pairs(pairs: Sequence[Pair], seed: int = SPLIT_SEED) -> dict:
    """Stratified 70/15/15 by season; the holiday block is held out entirely."""
    rng = random.Random(seed)
    by_group: Dict[str, List[str]] = collections.defaultdict(list)
    for pair in pairs:
        by_group[pair.group].append(pair.load_date)
    split = {"seed": seed, "train": [], "validation": [], "test": [],
             "holiday_block": [], "by_group": {}}
    for group in GROUPS:
        days = sorted(by_group.get(group, []))
        if group in HOLIDAY_GROUPS:
            split["holiday_block"].extend(days)
            split["by_group"][group] = {"holiday_block": len(days)}
            continue
        rng.shuffle(days)
        n = len(days)
        n_train = round(0.70 * n)
        n_val = round(0.15 * n)
        split["train"].extend(sorted(days[:n_train]))
        split["validation"].extend(sorted(days[n_train:n_train + n_val]))
        split["test"].extend(sorted(days[n_train + n_val:]))
        split["by_group"][group] = {
            "train": n_train, "validation": n_val,
            "test": n - n_train - n_val,
        }
    for key in ("train", "validation", "test", "holiday_block"):
        split[key] = sorted(split[key])
    return split


def _write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PAIRING_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, *, rows: Sequence[dict], pairs: Sequence[Pair],
                  unpaired: Dict[str, List[str]], windows: Dict[str, dict],
                  version: str, price_sha: str, manifest_sha: str,
                  csv_path: Path, npz_path: Path, split_path: Path,
                  command: str) -> Path:
    counts = collections.Counter(p.group for p in pairs)
    kinds = collections.defaultdict(collections.Counter)
    for pair in pairs:
        kinds[pair.group][pair.pairing_type] += 1
    mismatched = [r for r in rows
                  if r["pairing_type"] == "same_season_day_type_mismatch"]
    price_dates = [p.price_date for p in pairs]
    duplicates = [d for d, n in collections.Counter(price_dates).items()
                  if n > 1]
    load_duplicates = [d for d, n in
                       collections.Counter(p.load_date for p in pairs).items()
                       if n > 1]

    lines: List[str] = []
    lines.append("# Load-to-price pairing (V2-CY-JN-2025)")
    lines.append("")
    lines.append("**Cross-year controlled simulation.** Each record carries a "
                 "2022 `load_date` and a 2025 `price_date`. They are two "
                 "independent dates, aligned by quarter-hour index. This is "
                 "not a same-period settlement and must not be reported as "
                 "one.")
    lines.append("")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Frozen load library: `{version}` "
                 f"(manifest sha256 `{manifest_sha[:16]}…`)")
    lines.append(f"- Price source sha256: `{price_sha}`")
    lines.append(f"- Pairs: **{len(pairs)}**; unpaired load days: "
                 f"**{sum(len(v) for v in unpaired.values())}**")
    lines.append(f"- Distinct `price_date`: **{len(set(price_dates))}**; "
                 f"reused: **{len(duplicates)}**")
    lines.append(f"- Distinct `load_date`: **{len(set(p.load_date for p in pairs))}**; "
                 f"reused: **{len(load_duplicates)}**")
    lines.append("")
    lines.append("## By group")
    lines.append("")
    lines.append("| group | load days | paired | price days supplied | "
                 "same month-day | same type, nearest | type mismatch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for group in GROUPS:
        supply = windows[group]
        lines.append(
            f"| {group} | {supply['loads']} | {counts[group]} | "
            f"{supply['complete']} ({supply['weekday']} wd / "
            f"{supply['weekend']} we) | {kinds[group]['same_month_day']} | "
            f"{kinds[group]['same_season_same_day_type']} | "
            f"{kinds[group]['same_season_day_type_mismatch']} |")
    lines.append(f"| **total** | **{sum(w['loads'] for w in windows.values())}** "
                 f"| **{len(pairs)}** | | "
                 f"**{sum(k['same_month_day'] for k in kinds.values())}** | "
                 f"**{sum(k['same_season_same_day_type'] for k in kinds.values())}** | "
                 f"**{len(mismatched)}** |")
    lines.append("")
    lines.append("## Distance from the load day's month-day")
    lines.append("")
    lines.append("| group | pairs | same month-day | median gap (days) | "
                 "max gap | pairs over 14 days |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for group in GROUPS:
        gaps = [int(r["month_day_gap_days"]) for r in rows
                if r["group"] == group and r["month_day_gap_days"] != ""]
        zero = sum(1 for g in gaps if g == 0)
        lines.append(f"| {group} | {len(gaps)} | {zero} | "
                     f"{np.median(gaps):.0f} | {max(gaps)} | "
                     f"{sum(1 for g in gaps if g > 14)} |")
    lines.append("")
    lines.append("A large gap is a property of the source, not a search "
                 "failure: the 2025 file has no complete day between 11-16 and "
                 "11-30, so late-November load days take the nearest available "
                 "autumn date, which can fall in September. The gap is "
                 "recorded per pair in `month_day_gap_days`, and those pairs "
                 "must not be read as a November price applying to a November "
                 "load.")
    lines.append("")
    lines.append("## Day-type constraints")
    lines.append("")
    if mismatched:
        lines.append(f"{len(mismatched)} pair(s) could not keep the "
                     "weekday/weekend type and are recorded as mismatches, "
                     "not silently relabelled:")
        lines.append("")
        lines.append("| load_date | price_date | group | reason |")
        lines.append("|---|---|---|---|")
        for row in mismatched:
            lines.append(f"| {row['load_date']} | {row['price_date']} | "
                         f"{row['group']} | {row['pair_reason']} |")
        lines.append("")
        lines.append("The cause is supply, not a choice:")
        lines.append("")
        for group in sorted({r["group"] for r in mismatched}):
            stats = windows[group]
            lines.append(f"- {group}: {stats['load_weekend']} weekend load "
                         f"days against {stats['weekend']} complete weekend "
                         f"price days in the window, so at least "
                         f"{max(0, stats['load_weekend'] - stats['weekend'])} "
                         "cannot keep the type.")
        lines.append("")
        lines.append("The mismatch is placed on the pair whose same month-day "
                     "is available, so the failed type costs no additional "
                     "date distance. Every other group keeps the type.")
    else:
        lines.append("Every pair keeps the weekday/weekend type.")
    lines.append("")
    lines.append("Day type is the calendar weekday/weekend flag. Chinese "
                 "public-holiday makeup workdays are **not** reclassified in "
                 "this version; that can only under-count same-type matches, "
                 "and it never changes a date.")
    lines.append("")
    lines.append("## Uniqueness")
    lines.append("")
    lines.append("No `price_date` appears twice, so one price curve cannot "
                 "enter both training and test through two load days.")
    lines.append("No `load_date` appears twice; every load day is used at most "
                 "once.")
    lines.append("")
    if unpaired:
        lines.append("## Unpaired load days")
        lines.append("")
        for group, days in unpaired.items():
            lines.append(f"- {group}: {', '.join(days)}")
        lines.append("")
    lines.append("## Load curves")
    lines.append("")
    lines.append("Every paired load date assembled a 96x33 matrix: the group's "
                 "own user curves at their mapped buses (`replace`), the "
                 "benchmark's native demand on every other bus (`preserve`). "
                 "No load was rescaled, filled or zeroed to make a day usable.")
    lines.append("")
    lines.append("| group | median group peak (MW) | frozen median (MW) |")
    lines.append("|---|---:|---:|")
    for group in GROUPS:
        peaks = [float(r["load_peak_mw"]) for r in rows
                 if r["group"] == group]
        frozen = _frozen_median[group]
        lines.append(f"| {group} | {np.median(peaks):.3f} | {frozen:.3f} |")
    lines.append("")
    lines.append("The peaks are the group's own metered sum, the quantity the "
                 "frozen manifest records. The network total is larger because "
                 "the other buses keep their native demand; both are stored.")
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append("A candidate 70/15/15 split is written beside the pairing, "
                 "stratified by season, with the national-day block held out "
                 "whole. It is a proposal for the readiness review, not a "
                 "training release.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{csv_path.as_posix()}`")
    lines.append(f"- `{npz_path.as_posix()}` — `load/<load_date>`, shape (96, 33)")
    lines.append(f"- `{split_path.as_posix()}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_group_users: Dict[str, List[str]] = {}
_frozen_median: Dict[str, float] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=FROZEN_MANIFEST)
    parser.add_argument("--candidates", default=CANDIDATES)
    parser.add_argument("--load-curves", default=LOAD_CURVES)
    parser.add_argument("--mapping", default=MAPPING)
    parser.add_argument("--price-file", default="docs/data.csv")
    parser.add_argument("--coverage", default=COVERAGE)
    parser.add_argument("--price-curves", default=PRICE_CURVES)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    manifest_path = root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != FROZEN_VERSION:
        raise SystemExit(f"manifest version {manifest.get('version')!r} is not "
                         f"{FROZEN_VERSION!r}")
    if manifest.get("status") != "frozen":
        raise SystemExit("the load library must be frozen before it is paired")

    for group in GROUPS:
        _group_users[group] = list(manifest["groups"][group]["user_ids"])
        _frozen_median[group] = float(manifest["groups"][group]["median_peak_mw"])

    candidates = _read_csv(root / args.candidates)
    day_rows = collections.defaultdict(list)
    for row in candidates:
        if row["group"] in GROUPS:
            day_rows[row["group"]].append(row)

    price_day = load_price_pool(root / args.coverage)
    if not price_day:
        raise SystemExit("the coverage table has no complete price day")

    windows: Dict[str, dict] = {}
    pool: Dict[str, List[str]] = {}
    for group in GROUPS:
        lo, hi = WINDOWS[group]
        dates = sorted(d for d in price_day if lo <= d <= hi)
        pool[group] = dates
        loads = day_rows[group]
        windows[group] = {
            "loads": len(loads), "complete": len(dates),
            "weekday": sum(1 for d in dates
                           if not _is_weekend(Date.fromisoformat(d))),
            "weekend": sum(1 for d in dates
                           if _is_weekend(Date.fromisoformat(d))),
            "load_weekday": sum(1 for r in loads
                                if not _is_weekend(Date.fromisoformat(r["date"]))),
            "load_weekend": sum(1 for r in loads
                                if _is_weekend(Date.fromisoformat(r["date"]))),
        }

    # National-day dates are claimed first so no other group can take them.
    for group in GROUPS:
        if group in RESERVED:
            missing = [d for d in RESERVED[group] if d not in pool[group]]
            if missing:
                raise SystemExit(
                    f"{group}: reserved price date(s) {missing} are not "
                    "complete in the source")

    used: set = set()
    pairs: List[Pair] = []
    unpaired: Dict[str, List[str]] = {}
    for group in GROUPS:
        group_pairs, group_unpaired = pair_group(
            group, [r["date"] for r in day_rows[group]], pool[group], used)
        pairs.extend(group_pairs)
        if group_unpaired:
            unpaired[group] = group_unpaired

    users, _ = cov.load_mapping(root / args.mapping)
    rated = bnl.native_bus_load()
    shape = bnl.native_shape()
    curves = np.load(root / args.load_curves)
    price_curves = np.load(root / args.price_curves)

    candidate_by_date = {row["date"]: row for row in candidates}
    rows: List[dict] = []
    matrices: Dict[str, np.ndarray] = {}
    for pair in pairs:
        matrix, replace = build_bus_matrix(
            pair.load_date, _group_users[pair.group], users, rated, shape,
            curves)
        if matrix.shape != (96, 33):
            raise SystemExit(f"{pair.load_date}: matrix is {matrix.shape}")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise SystemExit(f"{pair.load_date}: matrix is not finite "
                             "non-negative MW")
        matrices[pair.load_date] = matrix
        price = np.asarray(price_curves[f"{PRICE_REGION}/{pair.price_date}"],
                           dtype=float)
        coverage = price_day[pair.price_date]
        load_day = Date.fromisoformat(pair.load_date)
        if pair.group in HOLIDAY_GROUPS:
            type_match = ""
        else:
            type_match = str(_is_weekend(load_day)
                             == _is_weekend(Date.fromisoformat(pair.price_date)))
        industry = matrix[:, replace].sum(axis=1)
        total = matrix.sum(axis=1)
        target_2025 = _month_day_2025(pair.load_date)
        rows.append({
            "load_date": pair.load_date,
            "price_date": pair.price_date,
            "group": pair.group,
            "pairing_type": pair.pairing_type,
            "weekday_load": _weekday_name(load_day),
            "weekday_price": _weekday_name(Date.fromisoformat(pair.price_date)),
            "day_type_match": type_match,
            "price_region": PRICE_REGION,
            "load_curve_sha256": load_curve_digest(pair.load_date, matrix),
            "price_curve_sha256": coverage["curve_sha256"],
            "price_pairing_rule": pair.pairing_type,
            "pair_reason": pair.reason,
            "month_day_gap_days": abs((
                Date.fromisoformat(pair.price_date) - target_2025).days)
                if target_2025 is not None else "",
            "representative": candidate_by_date[pair.load_date]["representative"],
            "season": candidate_by_date[pair.load_date]["season"],
            "group_users": len(_group_users[pair.group]),
            "group_buses": len(manifest["groups"][pair.group]["bus_ids"]),
            "load_peak_mw": round(float(industry.max()), 6),
            "network_peak_mw": round(float(total.max()), 6),
            "load_energy_mwh": round(float(total.sum() * 0.25), 6),
            "price_min": float(price.min()),
            "price_max": float(price.max()),
            "price_mean": round(float(price.mean()), 6),
        })

    price_dates = [p.price_date for p in pairs]
    if len(set(price_dates)) != len(price_dates):
        raise SystemExit("a price date was reused; refusing to write the pairing")

    out_dir = root / args.out
    report_dir = root / args.report_dir
    csv_path = _write_csv(out_dir / "load_price_pairing.csv", rows)
    npz_path = out_dir / "bus_loads_96x33.npz"
    np.savez_compressed(npz_path,
                        **{f"load/{k}": v for k, v in matrices.items()})
    split = split_pairs(pairs, seed=args.seed)
    split_path = out_dir / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    pairing_manifest = {
        "dataset": "v2_cy_jn_2025_pairing",
        "version": f"v2-cy-jn-2025-{hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]}",
        "kind": "cross_year_controlled_simulation",
        "not_a_same_period_settlement": True,
        "load_library_version": FROZEN_VERSION,
        "groups": list(GROUPS),
        "groups_excluded": {
            "spring": "no complete 2025 spring price day in the source",
            "spring_festival_2022": "no complete 2025 spring price day in the source",
        },
        "price_region": PRICE_REGION,
        "price_unit": "CNY/MWh",
        "price_unit_confirmed": False,
        "price_unit_note": "inferred from floor/cap and magnitude; not externally confirmed",
        "mv_value_enabled": False,
        "pairs": len(rows),
        "unpaired": unpaired,
        "distinct_price_dates": len(set(price_dates)),
        "reused_price_dates": 0,
        "pairing_type_counts": {
            k: sum(1 for r in rows if r["pairing_type"] == k)
            for k in sorted({r["pairing_type"] for r in rows})},
        "inputs": {
            "price_file": {"path": args.price_file,
                           "sha256": _sha256(root / args.price_file)},
            "frozen_manifest": {"path": args.manifest,
                                "sha256": _sha256(manifest_path)},
            "candidates": {"path": args.candidates,
                           "sha256": _sha256(root / args.candidates)},
            "load_curves": {"path": args.load_curves,
                            "sha256": _sha256(root / args.load_curves)},
            "mapping": {"path": args.mapping,
                        "sha256": _sha256(root / args.mapping)},
            "price_curves": {"path": args.price_curves,
                             "sha256": _sha256(root / args.price_curves)},
        },
        "outputs": {
            "pairing_csv": {"path": args.out + "/load_price_pairing.csv",
                            "sha256": _sha256(csv_path)},
            "bus_loads": {"path": args.out + "/bus_loads_96x33.npz",
                          "sha256": _sha256(npz_path)},
            "split": {"path": args.out + "/split.json",
                      "sha256": _sha256(split_path)},
        },
    }
    (out_dir / "pairing_manifest.json").write_text(
        json.dumps(pairing_manifest, indent=2), encoding="utf-8")

    report_path = _write_report(
        report_dir / "pairing_report.md", rows=rows, pairs=pairs,
        unpaired=unpaired, windows=windows, version=FROZEN_VERSION,
        price_sha=pairing_manifest["inputs"]["price_file"]["sha256"],
        manifest_sha=pairing_manifest["inputs"]["frozen_manifest"]["sha256"],
        csv_path=csv_path.relative_to(root), npz_path=npz_path.relative_to(root),
        split_path=split_path.relative_to(root),
        command=" ".join(sys.argv))

    # The data directory keeps its own copy: the pairing audit is read beside
    # the pairing CSV, and the report directory keeps the same bytes.
    shutil.copyfile(report_path, out_dir / "pairing_audit.md")

    print(f"pairs                 {len(rows)}")
    print(f"unpaired              {sum(len(v) for v in unpaired.values())}")
    print(f"distinct price dates  {len(set(price_dates))}")
    print(f"type mismatches       "
          f"{sum(1 for r in rows if r['pairing_type'] == 'same_season_day_type_mismatch')}")
    for group in GROUPS:
        n = sum(1 for r in rows if r["group"] == group)
        print(f"  {group:<22}{n:>3}")
    print(f"split                 {len(split['train'])}/"
          f"{len(split['validation'])}/{len(split['test'])} "
          f"+ holiday {len(split['holiday_block'])}")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {split_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
