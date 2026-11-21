# build_multiday_manifest.py
"""Audit the frozen industry subset across every real date it covers.

The multi-day study starts with a question that must be answered before any
training: on which real dates is the **same** 17 users / 9 buses composition
whole? This module answers it from the workbook once, writes one formal ledger
per eligible date, and records every rejected date with the reason it failed.

It does not select or rebalance the user subset. The mapping is frozen on
2022-11-30; a date either carries all 17 users or it does not. Missing data is
reported, never filled, interpolated or copied.

Price pairing is audited but not invented. The only real dated curve in the
workbook is one day; when it is the only curve, at most one load date can be
paired and the rest are recorded as a data gap. The split is therefore left
`pending_price`; this module writes the blocked manifest rather than a fake
train/validation/test split.

Usage:
    PYTHONPATH=src python src/build_multiday_manifest.py --audit-only
    PYTHONPATH=src python src/build_multiday_manifest.py --build-ledgers
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
from typing import Dict, List, Optional, Sequence

import build_industry_node_loads as bld
import config_loader as cfg
import dianyicha_price as dp
import industry_load as il
import multiday_dataset as md
import single_day_data as sdd
import validate_formal_ledger as vfl

#: The frozen subset. These are assertions, not knobs: a different mapping is a
#: different study and must be a deliberate change, not a silent re-selection.
EXPECTED_USERS = 17
EXPECTED_BUSES = {3, 16, 23, 24, 25, 26, 27, 30, 31}
EXPECTED_MAPPING_SHA256 = (
    "35e4ef933980fa4d7a3a806d11d933814b83acbea5bce5b910161c8016aab9a2")
MIN_PRICED_DAYS_FOR_SPLIT = 30

HOURS_PER_SLOT = il.HOURS_PER_SLOT

MANIFEST_COLUMNS = (
    "date", "season", "weekday", "day_type", "is_weekend", "is_workday",
    "holiday_class", "holiday_name", "holiday_offset",
    "users_required", "users_covered", "users_complete", "users_missing",
    "qualified_blocks", "buses_required", "buses_covered", "buses_mapped",
    "buses_without_data", "readings", "readings_not_convertible",
    "slots_complete", "load_eligible", "ledger_eligible", "ledger_sha256",
    "energy_mwh", "peak_mw", "peak_period", "peak_bus",
    "price_date_available", "price_date", "price_pair_rule", "price_year_gap",
    "price_source", "price_curve_sha256", "pair_quality", "pair_reason",
    "split", "dataset_status", "exclusion_reason",
    "mapping_sha256", "workbook_sha256", "scenario_fingerprint",
)

AUDIT_COLUMNS = (
    "date", "season", "weekday", "is_weekend", "is_workday",
    "holiday_class", "holiday_name", "holiday_offset",
    "users_required", "users_covered", "users_complete", "users_missing",
    "qualified_blocks", "buses_covered", "buses_mapped", "buses_without_data",
    "readings", "readings_not_convertible", "slots_complete",
    "load_complete", "load_eligible", "exclusion_reason",
)

PAIRING_COLUMNS = (
    "load_date", "load_eligible", "holiday_class", "day_type",
    "price_date", "price_pair_rule", "price_year_gap", "price_source",
    "price_curve_sha256", "pair_quality", "pair_reason",
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def load_frozen_mapping(path: Path) -> tuple:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    mapped = [r for r in rows if r["candidate_status"] == "mapped"]
    users = [r["user_id"] for r in mapped]
    buses = {int(r["destination_bus"]) for r in mapped}
    meters = [m for r in mapped for m in str(r["meter_ids"]).split(";") if m]
    if len(mapped) != EXPECTED_USERS or len(set(users)) != EXPECTED_USERS:
        raise SystemExit(
            f"{path} carries {len(set(users))} mapped users; the frozen subset "
            f"is {EXPECTED_USERS}")
    if buses != EXPECTED_BUSES:
        raise SystemExit(
            f"{path} maps to buses {sorted(buses)}; the frozen subset is "
            f"{sorted(EXPECTED_BUSES)}")
    if len(meters) != len(set(meters)):
        raise SystemExit(f"{path} repeats a meter id across users")
    digest = il.file_sha256(path)
    if digest != EXPECTED_MAPPING_SHA256:
        raise SystemExit(
            f"{path} has sha256 {digest}, not the frozen "
            f"{EXPECTED_MAPPING_SHA256}; re-selecting the subset is a study "
            "change and must be deliberate")
    user_bus = {r["user_id"]: int(r["destination_bus"]) for r in mapped}
    return user_bus, digest


def _declared_policies(scenario: str) -> Dict[int, str]:
    load_model = (cfg.get_scenario_cfg(scenario).get("load_model") or {})
    declared = load_model.get("bus_load_policy") or {}
    if not declared:
        raise SystemExit(f"{scenario} declares no load_model.bus_load_policy")
    return {int(k): str(v) for k, v in declared.items()}


def _assert_policy_matches_mapping(scenario: str, user_bus: Dict[str, int]) -> None:
    policies = _declared_policies(scenario)
    replace = {b for b, p in policies.items() if p == "replace"}
    if replace != set(user_bus.values()):
        raise SystemExit(
            f"{scenario} declares replace buses {sorted(replace)}, but the "
            f"frozen mapping uses {sorted(set(user_bus.values()))}")
    unexpected = {b: p for b, p in policies.items()
                  if b != 0 and b not in replace and p != "preserve"}
    if unexpected:
        raise SystemExit(
            f"{scenario} gives non-preserve policies to buses without a "
            f"frozen user: {unexpected}")


def _zero_audit_row(date: str, required_users: int,
                    mapped_buses: Sequence[int]) -> dict:
    return {
        "date": date,
        "meters_with_a_complete_day": 0,
        "users_required": required_users,
        "users_covered": 0,
        "users_complete": 0,
        "users_missing": ";".join(sorted(set())),
        "qualified_blocks": 0,
        "buses_covered": 0,
        "buses_mapped": len(mapped_buses),
        "buses_without_data": ";".join(str(b) for b in sorted(mapped_buses)),
        "readings": 0,
        "readings_not_convertible": 0,
        "slots_complete": False,
        "load_complete": False,
        "price_date_available": False,
        "price_note": "no dated real price curve overlaps this load date",
    }


def _load_eligible(row: dict) -> bool:
    return (_as_bool(row["slots_complete"])
            and int(row["users_complete"]) == int(row["users_required"])
            and int(row["buses_covered"]) == int(row["buses_mapped"])
            and int(row["readings_not_convertible"]) == 0)


def _exclusion_reason(row: dict) -> str:
    if int(row["qualified_blocks"]) == 0:
        return "no_qualified_block_for_subset"
    missing_users = int(row["users_required"]) - int(row["users_complete"])
    if missing_users > 0:
        return f"missing_users:{missing_users}"
    if int(row["buses_covered"]) < int(row["buses_mapped"]):
        return f"missing_buses:{row['buses_without_data']}"
    if not _as_bool(row["slots_complete"]):
        return "incomplete_slots"
    if int(row["readings_not_convertible"]) > 0:
        return "unconvertible_readings"
    return ""


def _iso_date_or_none(value) -> Optional[str]:
    try:
        return md.as_date(value).isoformat()
    except (TypeError, ValueError):
        return None


def _merged_audit(records, user_bus: Dict[str, int]) -> List[dict]:
    audit = {}
    for row in bld.date_audit(records, user_bus,
                              sorted(set(user_bus.values()))):
        key = _iso_date_or_none(row["date"])
        if key is not None:
            audit[key] = dict(row)
    all_dates = sorted({
        d for d in (_iso_date_or_none(md.date) for md in records)
        if d is not None})
    mapped_buses = sorted(set(user_bus.values()))
    rows = []
    for date in all_dates:
        row = audit.get(date) or _zero_audit_row(
            date, len(user_bus), mapped_buses)
        row["date"] = date
        row["users_required"] = len(user_bus)
        row["buses_mapped"] = len(mapped_buses)
        if int(row["users_complete"]) == 0:
            row["users_missing"] = ";".join(sorted(user_bus))
        rows.append(row)
    return rows


def _curve_descriptor(name: str, curve) -> dict:
    return {
        "sheet": name,
        "role": curve.role,
        "source_kind": curve.source_kind,
        "spatial_statistic": curve.spatial_statistic,
        "temporal_statistic": curve.temporal_statistic,
        "window_days": curve.window_days,
        "reference_period": curve.reference_period,
        "externally_confirmed": curve.externally_confirmed,
        "source": f"{Path(str(curve.provenance.get('workbook', 'price'))).name}#{name}",
        "workbook_sha256": curve.provenance.get("workbook_sha256"),
    }


def load_price_curves(workbook: Optional[str] = None) -> tuple:
    curves = dp.read_curves(workbook)
    accepted: List[md.PriceCurve] = []
    rejected: List[dict] = []
    descriptors: List[dict] = []
    for name, curve in curves.items():
        descriptor = _curve_descriptor(name, curve)
        if curve.source_kind != "real_market_data":
            descriptor.update({
                "accepted": False,
                "reason": "not_real_market_data",
            })
            rejected.append(descriptor)
        elif curve.is_single_day:
            date_text = sdd.iso_from_reference_period(
                curve.reference_period, name)
            digest = _sha256_text(json.dumps({
                "sheet": name, "date": date_text,
                "prices": [float(p) for p in curve.prices],
            }, sort_keys=True))
            descriptor.update({
                "price_date": date_text, "curve_sha256": digest,
                "accepted": True,
            })
            accepted.append(md.PriceCurve(
                price_date=md.as_date(date_text),
                curve_id=name,
                source=descriptor["source"],
                curve_sha256=digest,
                temporal_statistic=curve.temporal_statistic,
                externally_confirmed=curve.externally_confirmed,
            ))
        else:
            descriptor.update({
                "accepted": False,
                "reason": "not_a_single_day_curve",
            })
            rejected.append(descriptor)
        descriptors.append(descriptor)
    if not accepted:
        raise SystemExit(
            "the price workbook carries no single-day curve; there is nothing "
            "to pair a load date with")
    return accepted, rejected, descriptors


def _ledger_stats(path: Path) -> dict:
    """Whole-network energy and coincident peak for one ledger.

    `peak_mw` is the maximum of the per-period sum across buses, not the
    largest single-bus value: a bus peak is not the network load and comparing
    it with the 3.715 MW benchmark would be a category error.
    """
    total_mwh = 0.0
    per_period: Dict[str, float] = collections.defaultdict(float)
    per_bus: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    blank = 0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = row["final_mw"].strip()
            if not text:
                blank += 1
                continue
            value = float(text)
            total_mwh += value * HOURS_PER_SLOT
            per_period[row["time_index"]] += value
            per_bus[row["time_index"]][row["bus"]] = value
    peak_period = max(per_period, key=per_period.get) if per_period else ""
    peak_bus = ""
    if peak_period != "":
        peak_bus = max(per_bus[peak_period], key=per_bus[peak_period].get)
    return {
        "energy_mwh": total_mwh,
        "peak_mw": per_period[peak_period] if peak_period != "" else "",
        "peak_period": peak_period,
        "peak_bus": peak_bus,
        "blank_final_mw": blank,
    }


def _ledger_paths(out_root: Path, date_text: str, scenario: str) -> tuple:
    base = out_root / date_text / scenario
    return base / "bus_loads_96x33.csv", base / "manifest.json"


def _existing_ledger(ledger: Path, manifest_path: Path, date_text: str,
                     scenario: str, mapping_sha: str,
                     workbook_sha: str) -> Optional[dict]:
    if not ledger.exists() and not manifest_path.exists():
        return None
    if not ledger.exists() or not manifest_path.exists():
        raise SystemExit(
            f"{ledger.parent} is incomplete; remove it and rebuild rather "
            "than overwriting one half of a pair")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "date": manifest.get("date") == date_text,
        "scenario": manifest.get("scenario") == scenario,
        "ledger_kind": manifest.get("ledger_kind") == "formal",
        "mapping_sha256": manifest.get("mapping_sha256") == mapping_sha,
        "workbook_sha256": manifest.get("workbook_sha256") == workbook_sha,
        "ledger_sha256": manifest.get("ledger_sha256") == il.file_sha256(ledger),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            f"{ledger.parent} does not match the frozen inputs on {failed}; "
            "remove it and rebuild rather than mixing versions")
    return manifest


def _build_ledger(scenario: str, date_text: str, out_root: Path,
                  mapping: Path, records, workbook_sha: str,
                  selection: dict) -> tuple:
    result = bld.build(
        scenario, date_text, out_root / date_text, mapping, formal=True,
        selection=selection, records=records, workbook_sha256=workbook_sha)
    ledger = Path(result["path"])
    manifest_path = ledger.parent / "manifest.json"
    return ledger, manifest_path, result["manifest"]


def _validate_ledger(ledger: Path, manifest_path: Path,
                     price_available: bool) -> List[dict]:
    return vfl.validate(
        ledger, manifest_path, price_dates_available=price_available,
        case_type="cross_date_diagnostic")


def _scenario_fingerprint(scenario: str) -> str:
    from counterfactual_eval import _scenario_fingerprint as fingerprint
    return fingerprint(scenario)


def _counts(rows: Sequence[dict], key: str) -> Dict[str, int]:
    return dict(sorted(collections.Counter(str(r[key]) for r in rows).items()))


def _artifact_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(p): il.file_sha256(p) for p in paths if p.exists()}


def _write_report(path: Path, *, command: str, rows: Sequence[dict],
                  pairing_rows: Sequence[dict],
                  price_descriptors: Sequence[dict],
                  rejected_curves: Sequence[dict],
                  ledger_failures: Sequence[dict], date_issues: dict,
                  mapping_sha: str, workbook_sha: str,
                  price_workbook_sha: str, artifacts: Dict[str, str],
                  build_ledgers: bool) -> Path:
    eligible = [r for r in rows if _as_bool(r["load_eligible"])]
    ledger_ok = [r for r in rows if _as_bool(r["ledger_eligible"])]
    priced = [r for r in eligible if _as_bool(r["price_date_available"])
              and (not build_ledgers or _as_bool(r["ledger_eligible"]))]
    reasons = collections.Counter(r["exclusion_reason"] for r in rows
                                  if r["exclusion_reason"])

    lines = [
        "# Multiday load and price audit",
        "",
        "## Verdict",
        "",
        f"- Calendar dates examined: **{len(rows)}**",
        f"- Load-eligible dates (17/17 users, 9/9 buses, 96/96 slots): "
        f"**{len(eligible)}**",
        f"- Dates with a validated formal ledger: "
        f"**{len(ledger_ok) if build_ledgers else 'not built in this run'}**",
        f"- Dates with a real single-day price pair: **{len(priced)}**",
        f"- Split status: **blocked** (needs at least "
        f"{MIN_PRICED_DAYS_FOR_SPLIT} distinct priced days)",
        "",
        "No raw load or price workbook was modified. No missing value was "
        "filled, interpolated, copied or resampled.",
        "",
        "## Eligibility rule",
        "",
        "A date is load-eligible only when every one of the frozen 17 users has "
        "at least one qualified meter-day that converts to a complete 96-slot "
        "curve, all 9 mapped buses are covered, and no qualified block on the "
        "date fails conversion. The user subset and mapping are frozen on "
        "2022-11-30; they were not re-selected per date.",
        "",
        f"- Frozen mapping: `{mapping_sha}`",
        f"- Load workbook: `{workbook_sha}`",
        f"- Price workbook: `{price_workbook_sha}`",
        f"- Records with no usable date: "
        f"{date_issues['missing_date_records']} missing, "
        f"{date_issues['unparseable_date_records']} unparseable",
    ]
    missing_users = date_issues.get("missing_date_users") or []
    if missing_users:
        shown = ", ".join(missing_users[:10])
        more = "" if len(missing_users) <= 10 else \
            f" (+{len(missing_users) - 10} more)"
        lines.append(f"- Missing-date records belong to users: {shown}{more}")
    lines += [
        "",
        "## Eligible dates by season",
        "",
        "| season | dates |",
        "|---|---:|",
    ]
    for key, n in _counts(eligible, "season").items():
        lines.append(f"| {key} | {n} |")
    lines += ["", "## Eligible dates by day type", "",
              "| day type | dates |", "|---|---:|"]
    for key, n in _counts(eligible, "day_type").items():
        lines.append(f"| {key} | {n} |")
    lines += ["", "## Eligible dates by month", "",
              "| month | dates |", "|---|---:|"]
    by_month = collections.Counter(r["date"][:7] for r in eligible)
    for key, n in sorted(by_month.items()):
        lines.append(f"| {key} | {n} |")

    peaks = sorted(float(r["peak_mw"]) for r in eligible if r["peak_mw"])
    energies = sorted(float(r["energy_mwh"]) for r in eligible
                      if r["energy_mwh"])
    lines += ["", "## Load magnitude", "",
              "The frozen subset is not scale-stable across dates: it was "
              "selected to match 3.715 MW on 2022-11-30, and the same users "
              "can be far larger or smaller on another date. The study does "
              "not re-select users to make a date fit.", ""]
    if peaks and energies:
        lines += [
            "| quantity | min | median | max |",
            "|---|---:|---:|---:|",
            f"| network peak (MW) | {peaks[0]:.3f} | "
            f"{peaks[len(peaks) // 2]:.3f} | {peaks[-1]:.3f} |",
            f"| energy (MWh) | {energies[0]:.3f} | "
            f"{energies[len(energies) // 2]:.3f} | {energies[-1]:.3f} |",
            "",
            f"- Dates within 5% of the 3.715 MW benchmark: "
            f"**{sum(1 for p in peaks if abs(p - 3.715) <= 0.05 * 3.715)}**",
            f"- Dates above twice the benchmark: "
            f"**{sum(1 for p in peaks if p > 2 * 3.715)}**",
        ]
        high = [r for r in eligible
                if float(r["peak_mw"] or 0) > 2 * 3.715]
        dominant = collections.Counter(
            r.get("peak_bus", "") for r in high).most_common(1)
        if dominant and dominant[0][0]:
            lines += [f"- Among dates above twice the benchmark, bus "
                      f"{dominant[0][0]} is the dominant bus at the system "
                      f"peak on {dominant[0][1]} of {len(high)} dates."]
        top = sorted(eligible, key=lambda r: float(r["peak_mw"] or 0),
                     reverse=True)[:5]
        lines += ["", "| date | peak MW | energy MWh | dominant bus |",
                  "|---|---:|---:|---:|"]
        for r in top:
            lines.append(f"| {r['date']} | {float(r['peak_mw']):.3f} | "
                         f"{float(r['energy_mwh']):.3f} | "
                         f"{r.get('peak_bus', '')} |")
    lines += ["", "## Rejected dates", "",
              "| reason | dates |", "|---|---:|"]
    for key, n in sorted(reasons.items()):
        lines.append(f"| {key} | {n} |")
    lines += ["", "## Price curves", "",
              "| sheet | date | temporal | accepted | source |",
              "|---|---|---|---|---|"]
    for d in price_descriptors:
        lines.append(
            f"| {d['sheet']} | {d.get('price_date', '')} | "
            f"{d['temporal_statistic']} | "
            f"{'yes' if d.get('accepted') else 'no: ' + d.get('reason', '')} | "
            f"{d['source']} |")
    lines += ["", "Only single-day curves are admissible. Window means are "
              "recorded below and never relabelled as days.", ""]
    if rejected_curves:
        lines += ["| rejected sheet | reason |", "|---|---|"]
        for d in rejected_curves:
            lines.append(f"| {d['sheet']} | {d['reason']} |")
        lines.append("")
    paired = [r for r in pairing_rows if r["price_date"]]
    unpaired_eligible = sum(
        1 for r in pairing_rows
        if _as_bool(r["load_eligible"]) and not r["price_date"])
    lines += ["## Pairing", "",
              "| load date | price date | rule | quality |",
              "|---|---|---|---|"]
    for r in paired:
        lines.append(
            f"| {r['load_date']} | {r['price_date']} | "
            f"{r['price_pair_rule']} | {r['pair_quality']} |")
    lines += ["",
              f"- {unpaired_eligible} load-eligible dates are unpaired: no "
              "single-day price curve is available for them.",
              "", "## Ledger build", ""]
    if build_ledgers:
        lines.append(f"- Built and validated: **{len(ledger_ok)}** of "
                     f"{len(eligible)} eligible dates.")
        if ledger_failures:
            lines += ["", "| date | violations |", "|---|---|"]
            for failure in ledger_failures:
                codes = ", ".join(v["code"] for v in failure["violations"])
                lines.append(f"| {failure['date']} | {codes} |")
        else:
            lines += ["", "Every built ledger passed the formal-ledger "
                      "validator."]
    else:
        lines.append("Ledgers were not built in this run (`--audit-only`).")
    lines += ["", "## Data gaps", "",
              "- The frozen subset is whole only on the dates listed above; "
              "the full per-date list is in "
              "`reports/multiday_dataset_manifest.csv`. The remaining dates "
              "are rejected rather than padded.",
              "- The eligible window contains no Spring Festival, spring or "
              "summer dates. The raw load workbook ends on 2022-12-31, so no "
              "later Spring Festival can be audited without extending the "
              "raw load range.",
              "- Only one real single-day price curve exists; at most one load "
              "date can be paired, and that pair is a cross-year diagnostic, "
              "not a same-day replay. A multi-day priced dataset and a "
              "train/validation/test split are therefore blocked.",
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
        description="Audit the frozen industry subset across all real dates.")
    parser.add_argument("--scenario", default="typical_day_industry_subset")
    parser.add_argument(
        "--mapping", default="data/processed/industry_user_subset_mapping_v2.csv")
    parser.add_argument(
        "--out", default="data/processed/node_loads/multiday_industry_subset")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--price-workbook", default=None)
    parser.add_argument(
        "--build-ledgers", action="store_true",
        help="build and validate one formal ledger per load-eligible date")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="write the audit and manifest without building ledgers")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_only and args.build_ledgers:
        raise SystemExit("--audit-only and --build-ledgers are alternatives")
    build_ledgers = bool(args.build_ledgers)

    root = Path.cwd()
    mapping_path = root / args.mapping
    out_root = root / args.out
    reports = root / args.reports
    load_workbook = il.resolve_workbook()
    price_workbook = dp.resolve_workbook(args.price_workbook)
    raw_before = {
        "load": il.file_sha256(load_workbook),
        "price": il.file_sha256(price_workbook),
    }

    user_bus, mapping_sha = load_frozen_mapping(mapping_path)
    _assert_policy_matches_mapping(args.scenario, user_bus)
    scenario_fingerprint = _scenario_fingerprint(args.scenario)

    records = il.flag_records(list(il.iter_meter_days()))
    missing_dates = [r for r in records if r.date is None]
    date_issues = {
        "missing_date_records": len(missing_dates),
        "missing_date_users": sorted({r.user_key for r in missing_dates}),
        "missing_date_meters": sorted({r.meter_key for r in missing_dates}),
        "unparseable_date_records": sum(
            1 for r in records
            if r.date is not None and _iso_date_or_none(r.date) is None),
    }
    audit_rows = _merged_audit(records, user_bus)
    for row in audit_rows:
        row["load_eligible"] = _load_eligible(row)
        row["exclusion_reason"] = _exclusion_reason(row)

    accepted_curves, rejected_curves, price_descriptors = load_price_curves(
        args.price_workbook)
    eligible_dates = [r["date"] for r in audit_rows if r["load_eligible"]]
    pairing = md.pair_prices(eligible_dates, accepted_curves)
    pairs_by_date = {p["load_date"]: p for p in pairing["pairs"]}
    rejected_curves = list(rejected_curves) + list(pairing["rejected_curves"])

    ledger_results: Dict[str, dict] = {}
    ledger_failures: List[dict] = []
    if build_ledgers:
        for date_text in eligible_dates:
            ledger, manifest_path = _ledger_paths(
                out_root, date_text, args.scenario)
            existing = _existing_ledger(
                ledger, manifest_path, date_text, args.scenario, mapping_sha,
                raw_before["load"])
            if existing is None:
                selection = {
                    "rule": "frozen_subset_eligible_date",
                    "evidence": {
                        "source": "reports/multiday_subset_date_audit.csv",
                        "users_required": len(user_bus),
                        "buses_required": len(set(user_bus.values())),
                        "why": ("the date carries all 17 frozen users and all "
                                "9 mapped buses on a complete 96-slot curve"),
                    },
                }
                ledger, manifest_path, manifest = _build_ledger(
                    args.scenario, date_text, out_root, mapping_path, records,
                    raw_before["load"], selection)
            else:
                manifest = existing
            violations = _validate_ledger(
                ledger, manifest_path, bool(pairs_by_date.get(date_text)))
            stats = _ledger_stats(ledger)
            ledger_results[date_text] = {
                "ledger": ledger,
                "manifest": manifest_path,
                "manifest_data": manifest,
                "violations": violations,
                "stats": stats,
            }
            if violations:
                ledger_failures.append({
                    "date": date_text,
                    "violations": violations,
                })

    pairing_rows: List[dict] = []
    for audit in audit_rows:
        date_text = audit["date"]
        label = md.calendar_label(date_text)
        if audit["load_eligible"]:
            pair = pairs_by_date.get(date_text, {
                "load_date": date_text, "price_date": "",
                "price_pair_rule": "", "price_year_gap": "",
                "price_source": "", "price_curve_sha256": "",
                "pair_quality": "unpaired",
                "pair_reason": "no_single_day_curve",
            })
        else:
            pair = {
                "load_date": date_text, "price_date": "",
                "price_pair_rule": "", "price_year_gap": "",
                "price_source": "", "price_curve_sha256": "",
                "pair_quality": "unpaired",
                "pair_reason": "load_day_not_eligible",
            }
        pairing_rows.append({
            "load_date": date_text,
            "load_eligible": audit["load_eligible"],
            "holiday_class": label["holiday_class"],
            "day_type": label["day_type"],
            **{k: pair[k] for k in (
                "price_date", "price_pair_rule", "price_year_gap",
                "price_source", "price_curve_sha256", "pair_quality",
                "pair_reason")},
        })

    manifest_rows: List[dict] = []
    for audit in audit_rows:
        date_text = audit["date"]
        label = md.calendar_label(date_text)
        pair = next(p for p in pairing_rows if p["load_date"] == date_text)
        result = ledger_results.get(date_text)
        violations = result["violations"] if result else []
        stats = result["stats"] if result else {
            "energy_mwh": "", "peak_mw": "", "peak_period": ""}
        if build_ledgers and audit["load_eligible"]:
            ledger_eligible = not violations
        elif build_ledgers:
            ledger_eligible = False
        else:
            ledger_eligible = False
        if not audit["load_eligible"]:
            status = "ineligible"
            reason = audit["exclusion_reason"]
        elif build_ledgers and violations:
            status = "ledger_validation_failed"
            reason = "ledger_validation:" + ";".join(
                sorted({v["code"] for v in violations}))
        elif not build_ledgers:
            status = "eligible_no_ledger"
            reason = ""
        elif pair["price_date"]:
            status = ("diagnostic_pair_only"
                      if pair["pair_quality"].startswith("same_month_day_")
                      and pair["pair_quality"].endswith("mismatch")
                      else "eligible_with_price")
            reason = ""
        else:
            status = "eligible_no_price"
            reason = ""
        manifest_rows.append({
            "date": date_text,
            **{k: label[k] for k in (
                "season", "weekday", "day_type", "is_weekend", "is_workday",
                "holiday_class", "holiday_name", "holiday_offset")},
            "users_required": audit["users_required"],
            "users_covered": audit["users_covered"],
            "users_complete": audit["users_complete"],
            "users_missing": audit["users_missing"],
            "qualified_blocks": audit["qualified_blocks"],
            "buses_required": audit["buses_mapped"],
            "buses_covered": audit["buses_covered"],
            "buses_mapped": audit["buses_mapped"],
            "buses_without_data": audit["buses_without_data"],
            "readings": audit["readings"],
            "readings_not_convertible": audit["readings_not_convertible"],
            "slots_complete": audit["slots_complete"],
            "load_eligible": audit["load_eligible"],
            "ledger_eligible": ledger_eligible,
            "ledger_sha256": (result["manifest_data"].get("ledger_sha256", "")
                              if result else ""),
            "energy_mwh": (f"{stats['energy_mwh']:.6f}"
                           if stats["energy_mwh"] != "" else ""),
            "peak_mw": (f"{stats['peak_mw']:.6f}"
                        if stats["peak_mw"] != "" else ""),
            "peak_period": stats["peak_period"],
            "peak_bus": stats.get("peak_bus", ""),
            "price_date_available": bool(pair["price_date"]),
            "price_date": pair["price_date"],
            "price_pair_rule": pair["price_pair_rule"],
            "price_year_gap": pair["price_year_gap"],
            "price_source": pair["price_source"],
            "price_curve_sha256": pair["price_curve_sha256"],
            "pair_quality": pair["pair_quality"],
            "pair_reason": pair["pair_reason"],
            "split": "pending_price",
            "dataset_status": status,
            "exclusion_reason": reason,
            "mapping_sha256": mapping_sha,
            "workbook_sha256": raw_before["load"],
            "scenario_fingerprint": scenario_fingerprint,
        })

    # Audit-only runs must not replace the full build's deliverables with a
    # manifest that says no ledger was built. They get a suffixed set instead.
    suffix = "" if build_ledgers else ".audit_only"
    audit_path = _write_csv(
        reports / f"multiday_subset_date_audit{suffix}.csv", audit_rows,
        AUDIT_COLUMNS)
    manifest_path = _write_csv(
        reports / f"multiday_dataset_manifest{suffix}.csv", manifest_rows,
        MANIFEST_COLUMNS)
    pairing_path = _write_csv(
        reports / f"multiday_price_pairing_audit{suffix}.csv", pairing_rows,
        PAIRING_COLUMNS)

    priced = [
        r for r in manifest_rows
        if _as_bool(r["load_eligible"]) and r["price_date_available"]
        and (not build_ledgers or _as_bool(r["ledger_eligible"]))
    ]
    split_status = ("blocked" if len(priced) < MIN_PRICED_DAYS_FOR_SPLIT
                    else "ready_for_split")
    split_manifest = {
        "status": split_status,
        "reason": (
            f"{len(priced)} distinct real priced day(s); at least "
            f"{MIN_PRICED_DAYS_FOR_SPLIT} are required before a split is "
            "frozen. No train/validation/test dates were assigned."
            if split_status == "blocked" else
            "the priced-day minimum is met; the split is not frozen in this "
            "phase."),
        "candidate_days": sum(1 for r in manifest_rows
                              if _as_bool(r["load_eligible"])),
        "ledger_eligible_days": sum(1 for r in manifest_rows
                                    if _as_bool(r["ledger_eligible"])),
        "priced_days": len(priced),
        "required_minimum_priced_days": MIN_PRICED_DAYS_FOR_SPLIT,
        "split": {},
        "test_days_sealed": False,
        "pairs": [r for r in pairing_rows if r["price_date"]],
    }
    split_path = _write_json(
        reports / f"multiday_split_manifest{suffix}.json", split_manifest)

    raw_after = {
        "load": il.file_sha256(load_workbook),
        "price": il.file_sha256(price_workbook),
    }
    if raw_after != raw_before:
        raise SystemExit(
            "a raw workbook changed during the audit; refusing to write a "
            "dataset manifest against moved inputs")

    dataset_manifest = {
        "dataset": "multiday_industry_subset",
        "scenario": args.scenario,
        "created_at": dt.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "command": " ".join(sys.argv),
        "eligibility_rule": (
            "17/17 frozen users, 9/9 mapped buses, 96/96 slots, zero "
            "unconvertible qualified blocks"),
        "frozen_subset": {
            "users": len(user_bus),
            "buses": sorted(set(user_bus.values())),
            "mapping_sha256": mapping_sha,
        },
        "raw_inputs": {
            "load_workbook": str(load_workbook),
            "load_workbook_sha256": raw_before["load"],
            "price_workbook": str(price_workbook),
            "price_workbook_sha256": raw_before["price"],
            "raw_files_modified": False,
        },
        "counts": {
            "calendar_dates": len(manifest_rows),
            "load_eligible": split_manifest["candidate_days"],
            "ledger_eligible": split_manifest["ledger_eligible_days"],
            "ledgers_built": len(ledger_results),
            "ledger_validation_failures": len(ledger_failures),
            "priced_days": len(priced),
            "records_without_a_usable_date": date_issues,
        },
        "price_curves": list(price_descriptors),
        "rejected_price_curves": list(rejected_curves),
        "split": split_manifest,
        "scenario_fingerprint": scenario_fingerprint,
        "files": _artifact_hashes([
            audit_path, manifest_path, pairing_path, split_path]),
    }
    dataset_path = _write_json(out_root / f"dataset_manifest{suffix}.json",
                               dataset_manifest)
    raw_check_path = _write_json(out_root / "raw_input_hashes.json", raw_after)

    artifacts = _artifact_hashes([
        audit_path, manifest_path, pairing_path, split_path, dataset_path,
        raw_check_path])
    report_path = _write_report(
        reports / "multiday_dataset_audit.md",
        command=" ".join(sys.argv),
        rows=manifest_rows, pairing_rows=pairing_rows,
        price_descriptors=price_descriptors,
        rejected_curves=rejected_curves, ledger_failures=ledger_failures,
        date_issues=date_issues,
        mapping_sha=mapping_sha, workbook_sha=raw_before["load"],
        price_workbook_sha=raw_before["price"], artifacts=artifacts,
        build_ledgers=build_ledgers)

    print(f"calendar dates           {len(manifest_rows)}")
    print(f"load-eligible dates      {split_manifest['candidate_days']}")
    print(f"ledgers built/validated  {len(ledger_results)} "
          f"({len(ledger_failures)} failed)")
    print(f"priced dates             {len(priced)}")
    print(f"split                    {split_status}")
    print(f"wrote {audit_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {pairing_path}")
    print(f"wrote {split_path}")
    print(f"wrote {dataset_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
