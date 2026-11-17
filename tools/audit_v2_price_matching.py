# audit_v2_price_matching.py
"""Audit the existing V2 price sources and match them to the frozen load days.

The frozen input is `data/processed/v2/v2_scenario_library_manifest.json`
version `v2-frozen-dfc9d60cc222`. This script reads it, never edits it. It
inventories every price artefact in the project, then classifies each of the
123 frozen candidate dates and its 31 representative subset as exact, partial,
missing or unverifiable. No adjacent-date or window-mean substitution is made.

Outputs:
    reports/v2_price_source_audit.md
    reports/v2_load_price_matching_audit.md
    data/processed/v2/v2_load_price_matching_audit.csv
    data/processed/v2/v2_price_source_inventory.csv
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

import numpy as np

import dianyicha_price as dp
import single_day_data as sdd
import v2_price_matching as vpm

FROZEN_VERSION = "v2-frozen-dfc9d60cc222"
FROZEN_MANIFEST = Path("data/processed/v2/v2_scenario_library_manifest.json")
CANDIDATES = Path("data/processed/v2_seasonal_candidates.csv")
COVERAGE = Path("reports/v2_user_date_coverage.csv")
PRICE_WORKBOOK = Path("docs/日前价格261009.xlsx")
PRICE_96 = Path("data/processed/single_day_diagnostic/price_96.csv")
SYNTHETIC = Path("data/wholesale/synthetic_da_v1.csv")
CALIBRATION = Path("data/wholesale/price_ref_calibration.json")

SOURCE_COLUMNS = (
    "source_id", "path", "sheet", "date_or_range", "periods_per_day",
    "interval_minutes", "label_convention", "timezone", "units", "currency",
    "tax_status", "kind", "source_kind", "traceable",
    "externally_confirmed", "spatial_statistic", "missing_periods",
    "duplicate_labels", "min_cny_mwh", "max_cny_mwh", "mean_cny_mwh",
    "notes",
)

MATCH_COLUMNS = (
    "group", "group_type", "season", "holiday_class", "date",
    "representative", "load_users", "load_curve_complete", "load_periods",
    "load_interval_minutes", "load_label_convention", "price_source_id",
    "price_sheet", "price_date", "price_periods", "price_interval_minutes",
    "price_label_convention", "price_units", "price_currency",
    "price_tax_status", "price_kind", "price_source_kind",
    "price_traceable", "price_externally_confirmed", "same_date",
    "same_granularity", "interval_alignment", "match_class", "reason",
    "required_data",
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


def _numeric_stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "min_cny_mwh": float(np.min(arr)),
        "max_cny_mwh": float(np.max(arr)),
        "mean_cny_mwh": float(np.mean(arr)),
    }


def _curve_values(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _price_96_inventory() -> dict:
    rows = _curve_values(PRICE_96)
    values = [float(r["price_cny_mwh"]) for r in rows]
    starts = [r["interval_start"] for r in rows]
    ends = [r["interval_end"] for r in rows]
    return {
        "source_id": "single_day_diagnostic:price_96",
        "path": str(PRICE_96),
        "sheet": "",
        "date_or_range": "2026-10-09",
        "periods_per_day": len(rows),
        "interval_minutes": 15,
        "label_convention": "interval_end",
        "timezone": "unknown",
        "units": "CNY/MWh",
        "currency": "CNY",
        "tax_status": "unknown",
        "kind": "single_day",
        "source_kind": "derived_from_real",
        "traceable": True,
        "externally_confirmed": False,
        "spatial_statistic": "spatial_mean",
        "missing_periods": sum(1 for v in values if not np.isfinite(v)),
        "duplicate_labels": len(set(starts)) != len(starts)
                            or len(set(ends)) != len(ends),
        **_numeric_stats(values),
        "notes": ("extracted copy of the 261009 workbook sheet; not an "
                  "independent market source"),
    }


def _synthetic_inventory() -> dict:
    rows = _curve_values(SYNTHETIC)
    values = [float(r["price_cny_mwh"]) for r in rows]
    days = {r["day_id"] for r in rows}
    slots = {(r["day_id"], r["time_index"]) for r in rows}
    return {
        "source_id": "synthetic_v1",
        "path": str(SYNTHETIC),
        "sheet": "",
        "date_or_range": f"{len(days)} day_ids; no calendar dates",
        "periods_per_day": 96,
        "interval_minutes": 15,
        "label_convention": "interval_start",
        "timezone": "not_applicable",
        "units": "CNY/MWh",
        "currency": "CNY",
        "tax_status": "not_applicable",
        "kind": "synthetic",
        "source_kind": "synthetic",
        "traceable": True,
        "externally_confirmed": False,
        "spatial_statistic": "not_applicable",
        "missing_periods": len(rows) - len(slots),
        "duplicate_labels": len(slots) != len(rows),
        **_numeric_stats(values),
        "notes": ("generated library, master seed 20260920; not measured "
                  "market data"),
    }


def _calibration_inventory() -> dict:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return {
        "source_id": "price_ref_calibration",
        "path": str(CALIBRATION),
        "sheet": "",
        "date_or_range": "none",
        "periods_per_day": 0,
        "interval_minutes": 0,
        "label_convention": "unknown",
        "timezone": "not_applicable",
        "units": "CNY/MWh",
        "currency": "CNY",
        "tax_status": "not_applicable",
        "kind": "derived_reference",
        "source_kind": "derived",
        "traceable": True,
        "externally_confirmed": False,
        "spatial_statistic": "not_applicable",
        "missing_periods": 0,
        "duplicate_labels": False,
        "min_cny_mwh": "",
        "max_cny_mwh": "",
        "mean_cny_mwh": payload.get("value_cny_per_mwh", ""),
        "notes": ("one derived scalar, not a daily price curve; excluded from "
                  "the 261009 single-day reading"),
    }


def build_source_inventory() -> List[dict]:
    curves = dp.read_curves(str(PRICE_WORKBOOK))
    inventory: List[dict] = []
    for name, curve in curves.items():
        stats = curve.stats()
        if curve.is_single_day:
            iso = sdd.iso_from_reference_period(curve.reference_period, name)
            date_or_range = f"{iso} (raw {curve.reference_period})"
        else:
            date_or_range = "undated window mean"
        inventory.append({
            "source_id": f"dianyicha:{name}",
            "path": str(PRICE_WORKBOOK),
            "sheet": name,
            "date_or_range": date_or_range,
            "periods_per_day": 96,
            "interval_minutes": 15,
            "label_convention": "interval_end",
            "timezone": "unknown",
            "units": dp.UNITS,
            "currency": "CNY",
            "tax_status": "unknown",
            "kind": curve.temporal_statistic,
            "source_kind": curve.source_kind,
            "traceable": True,
            "externally_confirmed": curve.externally_confirmed,
            "spatial_statistic": curve.spatial_statistic,
            "missing_periods": 0,
            "duplicate_labels": False,
            **stats,
            "notes": (
                "single-day reading is a project ruling; spatial mean; no "
                "timezone or tax field in the workbook"
                if curve.is_single_day else
                f"trailing {curve.window_days}-day mean; not a daily curve"),
        })
    inventory.append(_price_96_inventory())
    inventory.append(_synthetic_inventory())
    inventory.append(_calibration_inventory())
    exports = sorted(Path("exports").glob("**/da_price.csv"))
    if exports:
        sample = _curve_values(exports[0])
        values = [float(r["system_price_cny_per_mwh"]) for r in sample]
        inventory.append({
            "source_id": "exports:da_price",
            "path": f"{len(exports)} files under exports/**/da_price.csv",
            "sheet": "",
            "date_or_range": "none (scenario output)",
            "periods_per_day": len(sample),
            "interval_minutes": 15,
            "label_convention": "interval_start",
            "timezone": "not_applicable",
            "units": "CNY/MWh",
            "currency": "CNY",
            "tax_status": "not_applicable",
            "kind": "scenario_output",
            "source_kind": "simulation_output",
            "traceable": True,
            "externally_confirmed": False,
            "spatial_statistic": "not_applicable",
            "missing_periods": 0,
            "duplicate_labels": False,
            **_numeric_stats(values),
            "notes": ("legacy scenario output, not a raw market observation; "
                      "statistics are from the first one of "
                      f"{len(exports)} files; shown only to prevent it being "
                      "mistaken for a source"),
        })
    return inventory


def _price_days() -> List[vpm.PriceDay]:
    curves = dp.read_curves(str(PRICE_WORKBOOK))
    out = []
    for name, curve in curves.items():
        date = None
        if curve.is_single_day:
            date = sdd.iso_from_reference_period(curve.reference_period, name)
        out.append(vpm.PriceDay(
            source_id=f"dianyicha:{name}",
            path=str(PRICE_WORKBOOK),
            sheet=name,
            date=date,
            periods=96,
            interval_minutes=15,
            label_convention="interval_end",
            units=dp.UNITS,
            currency="CNY",
            tax_status="unknown",
            kind=curve.temporal_statistic,
            source_kind=curve.source_kind,
            traceable=True,
            externally_confirmed=curve.externally_confirmed,
            spatial_statistic=curve.spatial_statistic,
            provenance={"workbook_sha256": curve.provenance.get("workbook_sha256")},
        ))
    return out


def _load_complete(coverage: Dict[tuple, bool], group: dict,
                   date: str) -> bool:
    return all(coverage.get((user, date), False)
               for user in group["user_ids"])


def build_matching_rows(manifest: dict, candidates: Sequence[dict],
                        coverage: Sequence[dict],
                        price_days: Sequence[vpm.PriceDay]) -> List[dict]:
    complete = {(r["user_id"], r["date"]): _as_bool(r["all_meters_complete"])
                for r in coverage}
    rows = []
    for row in candidates:
        group = manifest["groups"][row["group"]]
        load_ok = _load_complete(complete, group, row["date"])
        match = vpm.match_day(row["date"], price_days)
        if not load_ok:
            match = dict(match)
            match.update({
                "match_class": vpm.UNVERIFIABLE,
                "reason": "the load curve for this group is not complete on "
                          "this date",
            })
        required = (
            "a real 15-min day-ahead price curve for this actual date, with "
            "market/region/product, CNY/MWh and tax status, interval "
            "convention, timezone and export provenance; no window mean or "
            "adjacent date"
            if match["match_class"] != vpm.EXACT else "")
        rows.append({
            "group": row["group"],
            "group_type": row["group_type"],
            "season": row["season"],
            "holiday_class": row["holiday_class"],
            "date": row["date"],
            "representative": _as_bool(row["representative"]),
            "load_users": len(group["user_ids"]),
            "load_curve_complete": load_ok,
            "load_periods": 96,
            "load_interval_minutes": 15,
            "load_label_convention": "interval_start",
            **match,
            "required_data": required,
        })
    return rows


def _class_counts(rows: Sequence[dict]) -> Dict[str, int]:
    return vpm.count_classes(rows)


def _write_source_report(path: Path, *, manifest: dict,
                         inventory: Sequence[dict], manifest_sha: str,
                         price_workbook_sha: str, command: str) -> Path:
    lines = [
        "# V2 price-source audit",
        "",
        "## Frozen input",
        "",
        f"- Manifest: `{FROZEN_MANIFEST}`",
        f"- Version: **{manifest.get('version')}**",
        f"- Status: **{manifest.get('status')}**",
        f"- Manifest sha256: `{manifest_sha}`",
        f"- Load workbook sha256: "
        f"`{manifest.get('fingerprints', {}).get('load_workbook_sha256')}`",
        f"- Price workbook sha256: `{price_workbook_sha}`",
        "",
        "The frozen manifest was read only. The six user sets, the 31 "
        "representative days, the load curves and the `line50_q2.0` support "
        "assumption were not changed.",
        "",
        "## Existing price artefacts",
        "",
        "| source | sheet | date / range | periods | interval | labels | "
        "units | kind | source kind | traceable | external confirmation | "
        "min / max / mean CNY/MWh |",
        "|---|---|---|---:|---:|---|---|---|---|---|---|---|"]
    for row in inventory:
        rng = (f"{row['min_cny_mwh']} / {row['max_cny_mwh']} / "
               f"{row['mean_cny_mwh']}" if row["min_cny_mwh"] != "" else "—")
        lines.append(
            f"| {row['source_id']} | {row['sheet']} | {row['date_or_range']} | "
            f"{row['periods_per_day']} | {row['interval_minutes']} | "
            f"{row['label_convention']} | {row['units']} | {row['kind']} | "
            f"{row['source_kind']} | {row['traceable']} | "
            f"{row['externally_confirmed']} | {rng} |")
    lines += [
        "",
        "## Source-by-source notes",
        "",
        "### `261009` (single day, 2026-10-09)",
        "",
        "- The sheet's own header reads `日前均值（CNY/MWh）`; the workbook does "
        "not state a timezone, tax status or spatial weighting.",
        "- The single-day reading is a **project ruling**, not an external "
        "confirmation. `externally_confirmed` is false.",
        "- The curve is a **spatial mean**; no bus, zone or settlement point "
        "is represented.",
        "- It is the only real single-day curve in the project. Its actual "
        "date is 2026-10-09, which does not overlap any frozen load date.",
        "",
        "### `7days`, `30days`, `halfyear`",
        "",
        "- These are trailing window means, not daily observations. They have "
        "no actual date and cannot match a load date. They are recorded here "
        "as rejected, not used.",
        "",
        "### `single_day_diagnostic/price_96.csv`",
        "",
        "- An extracted copy of the `261009` sheet. It is traceable to the same "
        "workbook and has the same 2026-10-09 date; it adds no independent "
        "coverage.",
        "",
        "### `synthetic_v1` and `price_ref_calibration.json`",
        "",
        "- `synthetic_v1` is generated, not measured market data. It has 40 "
        "day ids and no calendar dates.",
        "- `price_ref_calibration.json` is one derived scalar, not a daily "
        "curve.",
        "",
        "### `exports/**/da_price.csv`",
        "",
        "- Scenario outputs from earlier runs. They are not raw market "
        "observations and carry no calendar date.",
        "",
        "## Data exists versus legally matchable",
        "",
        "- **Data exists** for one real single day (2026-10-09) and three real "
        "window means.",
        "- **Legally matchable to a frozen load day**: **none**. The single day "
        "has the wrong actual date; the window means are not days; the copy is "
        "the same single day; the synthetic library is not real; the exports "
        "are not observations.",
        "- A same-month-day cross-year pair is not a same-date price and is "
        "not counted as a match.",
        "",
        "## External source assessment",
        "",
        "The following is a **source survey**, not a data download. No "
        "subscription was purchased and no account was created.",
        "",
        "| candidate source | authority / access | what it could provide | "
        "confirmed for 2022 15-min daily data? | gap |",
        "|---|---|---|---|---|",
        "| 江苏电力交易中心 / Jiangsu spot-market disclosure platform | "
        "official market operator; registration/authorization likely | "
        "day-ahead clearing prices, 96 x 15-min periods | **not confirmed**; "
        "no public bulk download found in the survey | request an authorized "
        "export with market/product/unit/tax/interval metadata |",
        "| 电易电查 (the existing workbook's source) | commercial data vendor; "
        "subscription/authorization | historical daily curves | **not "
        "confirmed** beyond the single 2026-10-09 sheet | request a dated "
        "2022 export and its field definitions |",
        "| Mysteel / 北极星 / 我的煤炭 and similar news aggregators | secondary; "
        "public snippets or paid reports | single-day snapshots or "
        "monthly/medium-term averages | **no**; not a 15-min daily series in "
        "the public results | do not use as a daily price source |",
        "| academic / repository copies | provenance varies | possibly a "
        "shared Jiangsu spot dataset | **not verified** | require source, "
        "license, date and interval documentation before use |",
        "",
        "Public search on 2026-10-21 found secondary descriptions of Jiangsu's "
        "spot market design (96 x 15-min day-ahead periods, 00:15–24:00, zonal "
        "pricing) and single-day or monthly price news, but **not** a "
        "confirmed public 2022 96-point daily series. The official market "
        "operator remains the authoritative route. Secondary sources are "
        "listed only to show what was checked, not as admissible data.",
        "",
        "### Survey references (secondary, not data sources)",
        "",
        "- 2022 Jiangsu electricity market summary: "
        "http://news.mcar.com.cn/kx/20230103/103658.html",
        "- Jiangsu day-ahead market snapshot (Mysteel): "
        "https://www.mysteel.com/report/2609/B021AAF51145C26D.html",
        "- Jiangsu electricity market news index (我的煤炭): "
        "https://www.mycoal.cn/news/search-htm-kw-%E6%B1%9F%E8%8B%8F%E7%94%B5%E5%8A%9B%E4%BA%A4%E6%98%93%E4%B8%AD%E5%BF%83.html",
        "- Jiangsu electricity market news index (北极星): "
        "https://www.bjx.com.cn/search?kw=%E6%B1%9F%E8%8B%8F%E7%94%B5%E5%8A%9B%E5%B8%82%E5%9C%BA&type=5",
        "",
        "These are secondary/contextual pages. None is recorded as the "
        "authoritative price series for this study.",
        "",
        "## Specific data request",
        "",
        "At minimum, provide real 15-minute day-ahead price curves for the "
        "frozen representative dates (31 days) or, preferably, all 123 "
        "candidate dates. Each curve must state:",
        "",
        "- actual trading date and market/region (Jiangsu day-ahead spot);",
        "- product and settlement/pricing scope (zonal vs nodal, if relevant);",
        "- 96 periods per day, 15-minute granularity, interval convention;",
        "- timezone and daylight-saving handling;",
        "- CNY/MWh and tax status (included/excluded/unknown);",
        "- source, export path, file sha256 and field definitions;",
        "- no window means, no synthetic values, no copied or adjacent dates.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_matching_report(path: Path, *, manifest: dict,
                           rows: Sequence[dict], manifest_sha: str,
                           command: str, csv_path: Path,
                           gitignored: bool) -> Path:
    counts = _class_counts(rows)
    reps = [r for r in rows if r["representative"]]
    rep_counts = _class_counts(reps)
    by_season = collections.defaultdict(lambda: {
        "candidates": 0, "representatives": 0, "exact": 0, "missing": 0})
    by_group = collections.defaultdict(lambda: {
        "candidates": 0, "representatives": 0, "exact": 0, "missing": 0})
    for row in rows:
        for bucket in (by_season[row["season"]], by_group[row["group"]]):
            bucket["candidates"] += 1
            bucket["representatives"] += int(row["representative"])
            bucket["exact"] += int(row["match_class"] == vpm.EXACT)
            bucket["missing"] += int(row["match_class"] == vpm.MISSING)
    lines = [
        "# V2 load-price matching audit",
        "",
        "## Frozen input",
        "",
        f"- Version: **{manifest.get('version')}**",
        f"- Manifest sha256: `{manifest_sha}`",
        "- Six user sets, 123 candidate dates, 31 representative dates and the "
        "`line50_q2.0` support assumption are unchanged.",
        "",
        "## Result",
        "",
        "| class | candidates (123) | representative (31) |",
        "|---|---:|---:|"]
    for name in vpm.MATCH_CLASSES:
        lines.append(f"| {name} | {counts[name]} | {rep_counts[name]} |")
    lines += [
        "",
        f"- Exact matches: **{counts[vpm.EXACT]} / {len(rows)}** candidates, "
        f"**{rep_counts[vpm.EXACT]} / {len(reps)}** representatives.",
        "- The existing real single-day curve is dated 2026-10-09; no frozen "
        "load date is 2026-10-09. No adjacent-date or window-mean substitution "
        "was made.",
        "",
        "## Matching rule",
        "",
        "- The price must have the **same actual date** as the load day.",
        "- It must be a traceable real single-day market curve, not a window "
        "mean, synthetic library or scenario output.",
        "- It must have 96 periods of 15 minutes and an axis that aligns with "
        "the load's 96 periods. `single_day_data.check_alignment` is the "
        "authority; no shift or resampling is applied.",
        "- Missing is reported as missing; it is never filled with a neighbour "
        "or a mean.",
        "",
        "## Coverage by season",
        "",
        "| season | candidates | representatives | exact | missing |",
        "|---|---:|---:|---:|---:|"]
    for season in ("spring", "summer", "autumn", "winter"):
        row = by_season.get(season, {})
        lines.append(
            f"| {season} | {row.get('candidates', 0)} | "
            f"{row.get('representatives', 0)} | {row.get('exact', 0)} | "
            f"{row.get('missing', 0)} |")
    lines += ["", "## Coverage by group", "",
              "| group | candidates | representatives | exact | missing |",
              "|---|---:|---:|---:|---:|"]
    for group in sorted(by_group):
        row = by_group[group]
        lines.append(
            f"| {group} | {row['candidates']} | {row['representatives']} | "
            f"{row['exact']} | {row['missing']} |")
    lines += [
        "",
        "## Main gaps",
        "",
        "- No candidate date has a same-date real daily price. The single real "
        "day is four years away from the 2022 load dates.",
        "- Spring Festival and National Day 2022 are represented as load "
        "groups, but neither has a real 2022 day-ahead price curve.",
        "- The window means cannot substitute for missing dates; doing so "
        "would erase day-to-day variation and leak the same curve across "
        "dates.",
        "",
        "## Training gate",
        "",
        "No train/validation/test split is generated. The minimum to reach the "
        "formal training gate is:",
        "",
        "1. real 15-minute day-ahead price curves for the **31 representative "
        "dates** (all six groups), or preferably all 123 candidate dates. The "
        "31 representative days meet the earlier ≥30 distinct priced-day "
        "minimum, but a 70/15/15 split on them is thin (~22/5/4); the full "
        "123-day set is preferred;",
        "2. same market/region/product as the load study (Jiangsu day-ahead "
        "spot), with zonal/nodal scope stated;",
        "3. CNY/MWh, tax status, timezone and interval convention stated;",
        "4. a traceable export (source, file sha256, field definitions) for "
        "each date;",
        "5. no window means, no synthetic values and no reused/adjacent dates.",
        "",
        f"Until those exist, the available exact matches are "
        f"**{counts[vpm.EXACT]}** and the candidate library cannot be priced.",
        "",
        "## Artifact",
        "",
        f"- `{csv_path}`",
        f"- CSV is gitignored by the repository rule (`*.csv`): "
        f"**{gitignored}**. The file exists on disk but is not versioned; the "
        "rule was not changed.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _is_gitignored(path: Path) -> bool:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=Path.cwd(), capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit V2 price sources and match them to frozen load days.")
    parser.add_argument("--manifest", default=str(FROZEN_MANIFEST))
    parser.add_argument("--candidates", default=str(CANDIDATES))
    parser.add_argument("--coverage", default=str(COVERAGE))
    parser.add_argument("--out",
                        default="data/processed/v2/v2_load_price_matching_audit.csv")
    parser.add_argument("--inventory-out",
                        default="data/processed/v2/v2_price_source_inventory.csv")
    parser.add_argument("--source-report", default="reports/v2_price_source_audit.md")
    parser.add_argument("--match-report",
                        default="reports/v2_load_price_matching_audit.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    manifest_path = root / args.manifest
    candidates_path = root / args.candidates
    coverage_path = root / args.coverage
    for path in (manifest_path, candidates_path, coverage_path):
        if not path.exists():
            raise SystemExit(f"{path} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != FROZEN_VERSION:
        raise SystemExit(
            f"the frozen manifest is version {manifest.get('version')!r}, "
            f"expected {FROZEN_VERSION!r}")
    if manifest.get("status") != "frozen":
        raise SystemExit("the scenario library is not frozen; refusing to audit")

    inventory = build_source_inventory()
    price_days = _price_days()
    candidates = _read_csv(candidates_path)
    coverage = _read_csv(coverage_path)
    rows = build_matching_rows(manifest, candidates, coverage, price_days)
    counts = _class_counts(rows)
    if len(rows) != 123:
        raise SystemExit(
            f"the matching audit covers {len(rows)} candidates, expected 123")
    if counts[vpm.MISSING] != 123:
        # The current data has no match; a future price file changes the
        # number, and the script should report that rather than fail.
        print(f"note: match classes are {counts}", flush=True)

    inventory_path = _write_csv(root / args.inventory_out, inventory,
                                SOURCE_COLUMNS)
    out_path = _write_csv(root / args.out, rows, MATCH_COLUMNS)
    workbook_sha = _sha256(root / PRICE_WORKBOOK)
    manifest_sha = _sha256(manifest_path)
    source_report = _write_source_report(
        root / args.source_report, manifest=manifest, inventory=inventory,
        manifest_sha=manifest_sha, price_workbook_sha=workbook_sha,
        command=" ".join(sys.argv))
    match_report = _write_matching_report(
        root / args.match_report, manifest=manifest, rows=rows,
        manifest_sha=manifest_sha, command=" ".join(sys.argv),
        csv_path=out_path, gitignored=_is_gitignored(out_path))
    print(f"candidates {len(rows)}  classes {counts}")
    print(f"representatives {sum(1 for r in rows if r['representative'])}  "
          f"classes {_class_counts([r for r in rows if r['representative']])}")
    print(f"wrote {inventory_path}")
    print(f"wrote {out_path}")
    print(f"wrote {source_report}")
    print(f"wrote {match_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
