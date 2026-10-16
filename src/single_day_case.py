# single_day_case.py
"""The single-day diagnostic case, built through the common data entry point.

**What this case is.** A day of real metered industry load, composed onto the
IEEE33 buses, paired with a real day-ahead price curve. Both are measured data.
Neither is synthesised, copied, interpolated or rescaled, and the network, the
bus mapping and the facilities are the ones the rest of the study uses.

**What it is not, and this is the part a reader must not lose.** The load day and
the price day are **different calendar days**, so this is *not* a replay of a
historical day: it is not what that load day actually met in its own market, and
no revenue, profit or arbitrage figure computed from it describes anything that
happened. The pairing is a stated modelling choice, declared as the case type
`cross_date_diagnostic` and enforced by `single_day_data.load_case`.

Everything about *pairing* -- the date rule, each series' own period axis, and
the cross-curve alignment -- lives in `single_day_data` and is applied there.
This module does not re-implement any of it; it reads the curve, records what
was checked, and writes the artifacts. That split is deliberate: an alignment
rule with two implementations is a rule that will disagree with itself.

Writes `data/processed/single_day_diagnostic/`:

    price_96.csv          the price curve, one row per period
    price_manifest.json   what the curve is, where it came from, what was checked
    case_manifest.json    both days, both sources, the alignment, the fingerprints

Read-only with respect to every input: the workbook is read, never written.

Usage:
    PYTHONPATH=src python src/single_day_case.py
    PYTHONPATH=src python src/single_day_case.py --sheet 261009 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dianyicha_price as dp  # noqa: E402
import industry_load as il  # noqa: E402
import single_day_data as sdd  # noqa: E402

#: The curve this case prices against. One of the four the workbook holds, and
#: the only one that names a date.
PRICE_SHEET = "261009"

#: The case type this case is. Stated, never inferred: the date rule follows
#: from it, and `single_day_data` refuses a case that has not said which it is.
CASE_TYPE = "cross_date_diagnostic"

CASE_DIR = Path("data/processed/single_day_diagnostic")

#: Where the load ledger the builder writes lives.
LEDGER_DIR = Path("data/processed/node_loads/typical_day_industry")
LEDGER_MANIFEST = LEDGER_DIR / "manifest.json"
LEDGER_CSV = LEDGER_DIR / "bus_loads_96x33_audit.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_price(sheet: str = PRICE_SHEET) -> dict:
    """Read one price curve and record every check it had to pass.

    The checks are not repeated here -- they are the reader's, and they raise.
    What this function adds is the *record*: a reader that does not throw is
    evidence only if somebody says which assertions ran. `dp.read_curves`
    refuses a missing sheet, a blank, non-numeric or non-finite price, a slot
    axis that is not exactly 96 interval-END labels in ascending order, and a
    repeated label.
    """
    curves = dp.read_curves(sheets=(sheet,))
    curve = curves[sheet]
    prices = list(curve.prices)
    # The same axis check the entry point applies, so the record and the
    # enforced path cannot come to describe different axes.
    axis = sdd.load_price_axis(curve)

    workbook = dp.resolve_workbook()
    return {
        "sheet": sheet,
        "date": sdd.iso_from_reference_period(curve.reference_period, sheet),
        "units": dp.UNITS,
        "semantics": ("a price per unit of energy (CNY/MWh) for one 15-minute "
                      "interval; not a power and not an energy"),
        "n_slots": axis.n_periods,
        "interval_minutes": axis.interval_minutes,
        "interval_convention": "label_is_interval_end",
        "slot_labels": "00:15 .. 24:00, ascending, at 15-minute steps",
        "timezone": axis.timezone,
        "timezone_note": ("the workbook states no timezone; the labels are "
                          "clock times only, so any offset is unstated"),
        "source_kind": curve.source_kind,
        "spatial_statistic": curve.spatial_statistic,
        "spatial_note": ("a spatial mean over an unstated set of points, so no "
                         "single bus or zone is represented by this curve"),
        "temporal_statistic": curve.temporal_statistic,
        "window_days": curve.window_days,
        "externally_confirmed": curve.externally_confirmed,
        "declared_by": curve.declared_by,
        "reference_period_raw": curve.reference_period,
        "date_is_parsed_from": "the sheet name's own YY/MM/DD label",
        "workbook": str(workbook),
        "workbook_sha256": _sha256(workbook),
        "checks": {
            "slot_count_is_96": len(prices) == dp.N_SLOTS,
            "labels_are_the_interval_end_clock_in_order":
                "enforced by dianyicha_price._check_slot_axis; no exception",
            "no_blank_or_non_numeric_price":
                "enforced by dianyicha_price._check_price; no exception",
            "no_non_finite_price":
                "enforced by dianyicha_price._check_price; no exception",
            "no_repeated_label":
                "enforced by dianyicha_price._check_slot_axis; no exception",
        },
        "min_cny_mwh": min(prices),
        "max_cny_mwh": max(prices),
        "mean_cny_mwh": sum(prices) / len(prices),
        "prices": prices,
    }


def write_price_input(verified: dict, out_dir: Path) -> Dict[str, Path]:
    """The curve as a file, plus the manifest that says what it is."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "price_96.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_index", "interval_start", "interval_end",
                    "price_cny_mwh"])
        for t, value in enumerate(verified["prices"]):
            start = t * dp.SLOT_MINUTES
            end = start + dp.SLOT_MINUTES
            w.writerow([t,
                        f"{start // 60:02d}:{start % 60:02d}",
                        f"{end // 60:02d}:{end % 60:02d}",
                        f"{value:.6f}"])

    manifest = {k: v for k, v in verified.items() if k != "prices"}
    manifest["file"] = csv_path.as_posix()
    manifest["file_sha256"] = _sha256(csv_path)
    manifest["case_version"] = "single_day_price_v1"
    (out_dir / "price_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {"csv": csv_path, "manifest": out_dir / "price_manifest.json"}


def build(out_dir: Path = CASE_DIR, sheet: str = PRICE_SHEET,
          case_type: str = CASE_TYPE) -> dict:
    """Verify both inputs through the entry point, and write the case manifest."""
    verified = verify_price(sheet)
    written = write_price_input(verified, out_dir)

    # The pairing is validated here or not at all: the case type selects the
    # date rule, each axis is checked on its own terms, and the cross-curve
    # alignment is checked last.
    case = sdd.load_case(case_type, LEDGER_CSV, LEDGER_MANIFEST, sheet)
    described = case.describe()

    manifest = {
        "case_version": "single_day_diagnostic_v1",
        "load_date": described["date_rule"]["load_date"],
        "price_date": described["date_rule"]["price_date"],
        "same_day": described["date_rule"]["same_day"],
        "case_type": case_type,
        "load_source": described["load_source"],
        "price_source": {
            **described["price_source"],
            "file": written["csv"].as_posix(),
            "file_sha256": _sha256(written["csv"]),
            "date_confidence": (
                "a parsed sheet-name label; the workbook states no date field, "
                "no timezone, and the single-day reading is a project ruling"),
        },
        "unit_status": described["unit_status"],
        "reading_unit": case.ledger_manifest["reading_unit"],
        "reading_unit_evidence": case.ledger_manifest["reading_unit_evidence"],
        "load_kind": described["load_kind"],
        "mapping_version": described["mapping_version"],
        "mapping_sha256": case.ledger_manifest["mapping_sha256"],
        "load_policies": case.ledger_manifest["policies"],
        "period_alignment": described["alignment"],
        "input_fingerprints": {
            "load_workbook_sha256": case.ledger_manifest.get("workbook_sha256"),
            "mapping_sha256": case.ledger_manifest["mapping_sha256"],
            "ledger_sha256": case.ledger_manifest["ledger_sha256"],
            "price_workbook_sha256": verified["workbook_sha256"],
            "price_file_sha256": _sha256(written["csv"]),
        },
        "integrity_checks": {
            "load": {
                "buses_covered": case.ledger_manifest.get("buses"),
                "buses_with_data_on_this_date":
                    case.ledger_manifest.get("buses_with_data_on_this_date"),
                "users_with_data": case.ledger_manifest.get("users_with_data"),
                "periods_per_bus": case.ledger_manifest.get("slots_per_day"),
                "excluded_cells": case.ledger_manifest.get("excluded_cells"),
                "compositions_excluded_no_data":
                    case.ledger_manifest.get("compositions_excluded_no_data"),
                "cross_date_aggregation": "none",
                "double_counting": ("none: a user's meters are aggregated to "
                                    "one unit before mapping"),
                "implicit_filling": "none: a period with no admissible reading "
                                    "is left empty and its reason recorded",
            },
            "price": verified["checks"],
        },
        "not_a_same_day_replay": described["not_a_same_day_replay"],
    }
    path = out_dir / "case_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["_path"] = path.as_posix()
    manifest["_price_files"] = {k: v.as_posix() for k, v in written.items()}
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sheet", default=PRICE_SHEET,
                   help="which price curve to use (default: the only one that "
                        "names a date)")
    p.add_argument("--case-type", default=CASE_TYPE,
                   choices=list(sdd.CASE_TYPES))
    p.add_argument("--out", default=str(CASE_DIR))
    args = p.parse_args(argv)

    case = build(Path(args.out), args.sheet, args.case_type)
    print(f"case       {case['case_type']}")
    print(f"load  date {case['load_date']}   "
          f"rule {case['load_source']['selection_rule']}")
    print(f"price date {case['price_date']}   sheet {args.sheet}")
    print(f"same day   {case['same_day']}")
    print(f"ledger     {case['load_kind']}   unit {case['unit_status']} "
          f"({case['reading_unit_evidence']})")
    al = case["period_alignment"]
    print(f"alignment  {al['method']}")
    print(f"           load {al['load_axis']['label_convention']}")
    print(f"           price {al['price_axis']['label_convention']}")
    print(f"wrote      {case['_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
