"""Build the June-December 2021-2022 source pool for the v3 day library.

The body is the frozen v1 extraction with its train-only date guard removed, so
the two pools are comparable curve for curve; only the date window differs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from lxml import etree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import industry_load as il

TARGET = 3.715
UPPER = TARGET * 1.05
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
WORKBOOK = ROOT / "docs" / "16类行业用户.xlsx"
SCREENING = ROOT / "data/processed/industry_block_screening.csv"
WINDOW_MONTHS = ("06", "07", "08", "09", "10", "11", "12")
WINDOW_YEARS = ("2021", "2022")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an unlabelled empty table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_dates(screening_path):
    """Every screening date in June-December of 2021 or 2022, ascending."""
    dates = set()
    with Path(screening_path).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            date = row["date"]
            if date[:4] in WINDOW_YEARS and date[5:7] in WINDOW_MONTHS:
                dates.add(date)
    return tuple(sorted(dates))


def cell_value(cell, strings):
    value = cell.find(NS + "v")
    if value is None:
        return None
    return strings[int(value.text)] if cell.get("t") == "s" else value.text


def extract(pool_dir):
    output = Path(pool_dir)
    output.mkdir(parents=True, exist_ok=False)
    dates = window_dates(SCREENING)
    allowed = set(dates)
    il.require_unit_basis()
    admissible = {}
    with SCREENING.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["date"] not in allowed:
                continue
            flags = set(row["flags"].split(";"))
            if (row["qualified_B"] == "True"
                    and row["industry"] != il.PV_INDUSTRY
                    and not flags.intersection({"negative_load", "duplicate_conflict",
                                                "capacity_inconsistent"})):
                admissible[(row["date"], row["meter_id"])] = row
    curves = {}
    metadata = {}
    rejected = collections.Counter()
    conflicts = set()
    visited = 0
    started = time.monotonic()
    with ZipFile(WORKBOOK) as archive:
        strings = ["".join(item.itertext()) for item in
                   etree.fromstring(archive.read("xl/sharedStrings.xml"))]
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in etree.iterparse(sheet, events=("end",), tag=NS + "row"):
                visited += 1
                cells = {cell.get("r").rstrip("0123456789"): cell for cell in row}
                date = cell_value(cells["L"], strings) if "L" in cells else None
                if date in allowed:
                    account = cell_value(cells["B"], strings) or ""
                    meter = cell_value(cells["I"], strings) or ""
                    user_id = il.anon_id("U", account)
                    meter_id = il.anon_id("M", f"{account}|{meter}")
                    key = (date, meter_id)
                    if key in admissible:
                        raw = [None] * 108
                        for col, cell in cells.items():
                            index = 0
                            for letter in col:
                                index = index * 26 + ord(letter) - ord("A") + 1
                            if index <= len(raw):
                                raw[index - 1] = cell_value(cell, strings)
                        md = il.flag_meter_day(il._make_meter_day(raw, int(row.get("r"))))
                        curve = il.reading_to_mw(md)
                        excluded = {"missing_slots", "all_zero", "placeholder", "other_phase",
                                    "negative_load", "capacity_inconsistent", "ratio_missing"}
                        if (curve is None or excluded.intersection(md.flags)
                                or not np.isfinite(curve).all() or (curve < 0).any()):
                            rejected["raw_quality"] += 1
                        else:
                            name = f"{date}|{meter_id}"
                            if name in curves and not np.array_equal(curves[name], curve):
                                conflicts.add(name)
                            elif name not in curves:
                                curves[name] = curve
                                metadata[name] = {
                                    "date": date, "source_id": meter_id, "user_id": user_id,
                                    "industry": md.industry, "excel_row": md.excel_row,
                                    "ratio": md.ratio, "running_capacity": md.running_capacity,
                                    "peak_mw": float(curve.max()),
                                    "energy_mwh": float(curve.sum() * 0.25),
                                    "curve_sha256": array_digest(curve),
                                    "quality_flags": ";".join(sorted(set(md.flags))),
                                    "source_kind": "individual_meter",
                                }
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]
                if visited % 20000 == 0:
                    print(f"Scanned {visited} workbook rows; retained {len(curves)} "
                          f"window meter-days", flush=True)
    for key in conflicts:
        curves.pop(key, None)
        metadata.pop(key, None)
    np.savez_compressed(output / "meter_curves.npz", **curves)
    rows = [metadata[key] for key in sorted(metadata)]
    write_csv(output / "meter_sources.csv", rows)
    inventory = []
    for date in dates:
        eligible = [r for r in rows if r["date"] == date and 0 < r["peak_mw"] <= UPPER]
        inventory.append({"date": date, "eligible_meters": len(eligible),
                          "eligible_distinct_users": len({r["user_id"] for r in eligible})})
    write_csv(output / "meter_inventory.csv", inventory)
    manifest = {
        "status": "SOURCE_POOL_ONLY", "window_dates": dates,
        "materialized_sample_dates": sorted({r["date"] for r in rows}),
        "unit": "MW", "unit_basis": il.READING_UNIT_EVIDENCE,
        "unit_status": il.unit_status(), "external_unit_verified": il.is_externally_verified(),
        "unit_source": il.READING_UNIT_SOURCE,
        "transform": "original slot reading * original ratio * 0.001",
        "zero_imputation": False, "rescaled": False,
        "meter_exclusivity": "At most one selected meter per user",
        "rows_scanned": visited, "window_meter_days": len(curves),
        "conflict_keys_removed": sorted(conflicts), "rejected": dict(rejected),
        "seconds": time.monotonic() - started,
        "inputs": {str(p.relative_to(ROOT)): digest(p)
                   for p in [WORKBOOK, SCREENING, Path(il.__file__)]},
        "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"extracted_meter_days": len(curves), "inventory": inventory},
                     ensure_ascii=True), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory for the pool")
    args = parser.parse_args(argv)
    extract(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
