"""Independently verify a written v3 library against the workbook and the rules.

This module re-derives every gating invariant from the files on disk; it
imports neither the builder nor its selection rule, so a builder defect cannot
be inherited by the check that is supposed to catch it.
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
from zipfile import ZipFile

import numpy as np
from lxml import etree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import industry_load as il

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
WORKBOOK = ROOT / "docs" / "16类行业用户.xlsx"
BAND = (3.3435, 4.0865)
REACTIVE_RATIO = 0.6190
REASONS = {"initial_in_band", "raised_in_band", "no_swap_below_band", "nothing_selectable"}
MONTHS = ("06", "07", "08", "09", "10", "11", "12")
LIBRARY_FILES = ("day_selection.csv", "source_bus_mapping.csv", "bus_active_load_mw.npz",
                 "bus_reactive_load_mvar.npz", "split.json", "manifest.json")


def _sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def _content_sha(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(arrays[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_npz(path):
    with np.load(path) as npz:
        return {key: np.asarray(npz[key], dtype=np.float64) for key in npz.files}


def _cell(cell, strings):
    value = cell.find(NS + "v")
    if value is None:
        return None
    return strings[int(value.text)] if cell.get("t") == "s" else value.text


def _workbook_curves(wanted):
    """Re-read the named meter-days straight from the workbook."""
    curves = {}
    with ZipFile(WORKBOOK) as archive:
        strings = ["".join(item.itertext()) for item in
                   etree.fromstring(archive.read("xl/sharedStrings.xml"))]
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in etree.iterparse(sheet, events=("end",), tag=NS + "row"):
                cells = {cell.get("r").rstrip("0123456789"): cell for cell in row}
                date = _cell(cells["L"], strings) if "L" in cells else None
                account = _cell(cells["B"], strings) or ""
                meter = _cell(cells["I"], strings) or ""
                key = (date, il.anon_id("M", f"{account}|{meter}"))
                if key in wanted:
                    raw = [None] * 108
                    for column, cell in cells.items():
                        index = 0
                        for letter in column:
                            index = index * 26 + ord(letter) - ord("A") + 1
                        if index <= len(raw):
                            raw[index - 1] = _cell(cell, strings)
                    md = il.flag_meter_day(il._make_meter_day(raw, int(row.get("r"))))
                    curve = il.reading_to_mw(md)
                    if curve is not None:
                        curves[key] = curve
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]
    return curves


def _group(rows, field):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return grouped


def _curve_preservation(mapping_rows, active):
    by_date = _group(mapping_rows, "date")
    wanted = {(row["date"], row["source_id"]) for row in mapping_rows}
    curves = _workbook_curves(wanted)
    ok = len(curves) == len(wanted)
    for date, rows in sorted(by_date.items()):
        matrix = active.get(date)
        if matrix is None:
            return False
        columns = [matrix[:, bus] for bus in range(matrix.shape[1])
                   if np.any(matrix[:, bus] != 0.0)]
        expected = []
        for row in rows:
            curve = curves.get((date, row["source_id"]))
            if curve is None or _sha(curve) != row["curve_sha256"]:
                return False
            expected.append(_sha(curve))
        if collections.Counter(_sha(column) for column in columns) != collections.Counter(expected):
            return False
    return ok


def _one_meter_per_bus(mapping_rows):
    for date, rows in _group(mapping_rows, "date").items():
        buses = [row["bus_zero_based"] for row in rows]
        sources = [row["source_id"] for row in rows]
        if len(set(buses)) != len(buses) or len(set(sources)) != len(sources):
            return False
    return True


def _reactive_proportionality(active, reactive):
    if set(active) != set(reactive):
        return False
    for date in active:
        p = active[date]
        q = reactive[date]
        if p.shape != q.shape:
            return False
        loaded = p > 0.0
        if np.any(np.abs(q[loaded] / p[loaded] - REACTIVE_RATIO) > 1e-9):
            return False
        if np.any(q[~loaded] != 0.0):
            return False
    return True


def _band_library_only(day_rows):
    for row in day_rows:
        peak = float(row["peak_mw"])
        if row["status"] == "IN_BAND":
            if not BAND[0] <= peak <= BAND[1]:
                return False
        elif row["reason"] not in REASONS:
            return False
    return True


def _split_integrity(split, active):
    blocks = split["blocks"]
    loads = [date for label in blocks for date in blocks[label]]
    prices = [pair.split("|")[1] for label in split["pairs"]
              for pair in split["pairs"][label]]
    if len(set(loads)) != len(loads) or len(set(prices)) != len(prices):
        return False
    if set(active) != set(loads):
        return False
    if split["library_size"] != len(loads):
        return False
    if any(split["counts"][label] != len(blocks[label]) for label in blocks):
        return False
    pattern = split["pattern"]
    dealt = split["library_size"] - split["counts"]["holiday"]
    expected = collections.Counter(pattern[i % len(pattern)] for i in range(dealt))
    expected["holiday"] = split["counts"]["holiday"]
    return dict(expected) == dict(split["counts"])


def _month_coverage(split):
    months = {date[5:7] for label in split["blocks"] for date in split["blocks"][label]}
    return months == set(MONTHS)


def _determinism(library_dir, manifest, active, reactive):
    outputs = manifest["outputs"]
    return (outputs["bus_active_load_mw.npz"]["content_sha256"] == _content_sha(active)
            and outputs["bus_reactive_load_mvar.npz"]["content_sha256"] == _content_sha(reactive))


def _concentration(mapping_rows):
    import pandapower.networks

    net = pandapower.networks.case33bw()
    p0 = np.zeros(33, dtype=np.float64)
    for row in net.load.itertuples():
        p0[int(row.bus)] += float(row.p_mw)
    native_total = float(p0.sum())
    native_max = float(p0.max()) / native_total
    per_day = {}
    for date, rows in _group(mapping_rows, "date").items():
        share = {row["source_id"]: float(row["meter_share"]) for row in rows}
        target = {int(row["bus_zero_based"]): float(row["target_share"]) for row in rows}
        assigned = collections.defaultdict(float)
        for row in rows:
            assigned[int(row["bus_zero_based"])] += float(row["meter_share"])
        d_final = sum((assigned[bus] / target[bus] - 1.0) ** 2 for bus in target)
        by_weight = sorted(target, key=lambda bus: (-target[bus], bus))
        by_size = sorted(share, key=lambda source: (-share[source], source))
        d_init = sum((share[source] / target[bus] - 1.0) ** 2
                     for bus, source in zip(by_weight, by_size))
        per_day[date] = {
            "realized_max_share": round(max(assigned.values()), 9),
            "target_max_share": round(max(target.values()), 9),
            "native_max_share": round(native_max, 9),
            "d_init": round(d_init, 9),
            "d_final": round(d_final, 9),
            "excluded_native_share": round(
                1.0 - float(sum(p0[bus] for bus in target)) / native_total, 9),
        }
    ratios = [values["realized_max_share"] / values["target_max_share"]
              for values in per_day.values()]
    median_ratio = statistics.median(ratios) if ratios else 0.0
    warning = None
    if median_ratio > 1.1:
        warning = {"code": "M_A_QUALITY_WARNING",
                   "dates": sorted(date for date, values in per_day.items()
                                   if values["realized_max_share"]
                                   > 1.1 * values["target_max_share"])}
    return {"per_day": per_day, "median_realized_over_target": round(median_ratio, 9),
            "warning": warning}


def verify(library_dir):
    library_dir = Path(library_dir)
    for name in LIBRARY_FILES:
        if not (library_dir / name).exists():
            raise FileNotFoundError(f"library is missing {name}")
    mapping_rows = _read_csv(library_dir / "source_bus_mapping.csv")
    day_rows = _read_csv(library_dir / "day_selection.csv")
    split = json.loads((library_dir / "split.json").read_text(encoding="utf-8"))
    manifest = json.loads((library_dir / "manifest.json").read_text(encoding="utf-8"))
    active = _read_npz(library_dir / "bus_active_load_mw.npz")
    reactive = _read_npz(library_dir / "bus_reactive_load_mvar.npz")

    checks = {
        "curve_preservation": _curve_preservation(mapping_rows, active),
        "one_meter_per_bus": _one_meter_per_bus(mapping_rows),
        "reactive_proportionality": _reactive_proportionality(active, reactive),
        "band_library_only": _band_library_only(day_rows),
        "split_integrity": _split_integrity(split, active),
        "month_coverage": _month_coverage(split),
        "determinism": _determinism(library_dir, manifest, active, reactive),
    }
    return {
        "checks": checks,
        "report": {"concentration": _concentration(mapping_rows),
                   "library_size": split["library_size"],
                   "counts": split["counts"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a written v3 library.")
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args.library)
    payload = {"checks": result["checks"], "report": result["report"],
               "passed": all(result["checks"].values())}
    (Path(args.library) / "verification.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    for name, value in sorted(result["checks"].items()):
        print(f"{'PASS' if value else 'FAIL'} {name}")
    print(f"median_realized_over_target {payload['report']['concentration']['median_realized_over_target']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
