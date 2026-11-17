"""Build the v3 candidate-day library from the window source pool.

The builder only orchestrates `library_rules`; every value that decides an
outcome is in a rule module, and no price, clearing or reward is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import library_rules as rules

ROOT = Path(__file__).resolve().parents[3]
WORKBOOK = ROOT / "docs" / "16类行业用户.xlsx"
SCREENING = ROOT / "data/processed/industry_block_screening.csv"
DESIGN_DOC = ROOT / "docs/superpowers/specs/2026-11-17-candidate-day-library-v3-design.md"
DEFAULT_PRICE_CURVES = ROOT / "data/processed/v2_cy_jn_2025/price_curves.npz"
DATASET_VERSION = "candidate_days_v3_2026-11-17"

DAY_COLUMNS = ["date", "month", "peak_mw", "gap_percent", "in_band", "nodes_loaded",
               "hit_bus_ceiling", "status", "reason"]
MAPPING_COLUMNS = ["date", "source_id", "user_id", "bus_zero_based", "meter_mean_mw",
                   "meter_share", "target_share", "curve_sha256"]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_content_sha256(arrays):
    """Hash the arrays, not the zip bytes, so zip encoding cannot change the digest."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(arrays[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def _label(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def load_pool(pool_dir):
    """Read the source pool as curves per date, plus the user and hash of each meter."""
    pool_dir = Path(pool_dir)
    with np.load(pool_dir / "meter_curves.npz") as npz:
        flat = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}
    curve_sha = {}
    user_of = {}
    with open(pool_dir / "meter_sources.csv", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["date"] + "|" + row["source_id"]
            user_of[key] = row["user_id"]
            curve_sha[key] = row["curve_sha256"]
    if set(flat) != set(curve_sha):
        raise ValueError("pool key sets differ: meter_curves.npz and meter_sources.csv")
    for key in sorted(flat):
        if rules._array_sha256(flat[key]) != curve_sha[key]:
            raise ValueError(f"curve hash mismatch for {key}")
    curves_by_date = {}
    for key in sorted(flat):
        date, source = key.split("|", 1)
        curves_by_date.setdefault(date, {})[source] = flat[key]
    return curves_by_date, user_of, curve_sha


def mv_price_dates(npz_path):
    """Same-month 2025 price dates under the `mv/` prefix, keyed by `MM`."""
    with np.load(npz_path) as npz:
        names = [k for k in npz.files if k.startswith("mv/")]
    by_month = {}
    for name in sorted(names):
        date = name.split("/", 1)[1]
        by_month.setdefault(date[5:7], []).append(date)
    return {month: tuple(sorted(dates)) for month, dates in sorted(by_month.items())}


def _day_row(selection):
    return [selection.date, int(selection.date[5:7]), f"{selection.peak_mw:.6f}",
            f"{(selection.peak_mw / rules.ANCHOR_MW - 1.0) * 100.0:.6f}",
            "true" if selection.status == "IN_BAND" else "false",
            len(selection.sources), "true" if selection.hit_bus_ceiling else "false",
            selection.status, selection.reason]


def run(pool_dir, out_dir, price_dates):
    pool_dir = Path(pool_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"output directory is not empty: {out_dir}")

    curves_by_date, user_of, curve_sha = load_pool(pool_dir)
    dates = sorted(curves_by_date)
    selections = {}
    for date in dates:
        curves = curves_by_date[date]
        users = {source: user_of[f"{date}|{source}"] for source in curves}
        selections[date] = rules.select_day(date, curves, users)

    in_band = [selections[date] for date in dates if selections[date].status == "IN_BAND"]
    split = rules.draw_split([s.date for s in in_band], price_dates)
    library_dates = sorted(date for label in rules.BLOCKS for date in split.blocks[label])

    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = rules.native_p0()
    active = {}
    reactive = {}
    mapping_rows = []
    for date in library_dates:
        curves = curves_by_date[date]
        mean = {source: float(curves[source].mean()) for source in selections[date].sources}
        total_mean = sum(mean.values())
        pi = {source: value / total_mean for source, value in mean.items()}
        assignment = rules.assign_buses(pi, p0)
        matrix = rules.bus_matrix(assignment.bus_of_source, curves)
        active[date] = matrix
        reactive[date] = rules.reactive_from_active(matrix)
        for source in sorted(mean):
            bus = assignment.bus_of_source[source]
            mapping_rows.append([
                date, source, user_of[f"{date}|{source}"], bus, f"{mean[source]:.9f}",
                f"{pi[source]:.9f}", f"{assignment.target_share[bus]:.9f}",
                curve_sha[f"{date}|{source}"],
            ])

    _write_csv(out_dir / "day_selection.csv", DAY_COLUMNS,
               [_day_row(selections[date]) for date in dates])
    _write_csv(out_dir / "source_bus_mapping.csv", MAPPING_COLUMNS, mapping_rows)
    np.savez(out_dir / "bus_active_load_mw.npz", **active)
    np.savez(out_dir / "bus_reactive_load_mvar.npz", **reactive)
    split_payload = {
        "seed": rules.SPLIT_SEED,
        "pattern": ["train"] * 5 + ["validation", "test"],
        "library_size": split.library_size,
        "counts": split.counts,
        "blocks": {label: list(split.blocks[label]) for label in rules.BLOCKS},
        "pairs": {label: list(split.pairs[label]) for label in rules.BLOCKS},
        "discarded": {month: list(dates_) for month, dates_ in split.discarded.items()},
        "holiday_shortfall": split.holiday_shortfall,
    }
    _write_json(out_dir / "split.json", split_payload)

    pool_manifest = pool_dir / "manifest.json"
    extraction = {}
    if pool_manifest.exists():
        recorded = json.loads(pool_manifest.read_text(encoding="utf-8"))
        extraction = {key: recorded.get(key) for key in
                      ("window_meter_days", "rejected", "rows_scanned")}
    month_counts = {}
    for date in library_dates:
        month = date[5:7]
        month_counts[month] = month_counts.get(month, 0) + 1
    covered = sorted(month_counts)
    universe = set()
    with SCREENING.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            universe.add(row["date"])
    universe.discard("")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "status": "LOAD_ONLY_PASS" if covered == list(rules.LIBRARY_MONTHS) else "PARTIAL",
        "band": {"anchor_mw": rules.ANCHOR_MW, "tolerance": rules.TOLERANCE,
                 "low_mw": rules.LOW_MW, "high_mw": rules.HIGH_MW},
        "power_factor": rules.POWER_FACTOR,
        "reactive_ratio": rules.REACTIVE_RATIO,
        "periods": rules.PERIODS,
        "bus_count": rules.BUS_COUNT,
        "max_meters": rules.MAX_METERS,
        "universe_dates": {"count": len(universe),
                           "first": min(universe), "last": max(universe)},
        "window_dates": {"count": len(dates), "first": dates[0], "last": dates[-1]},
        "extraction": extraction,
        "in_band_count": len(in_band),
        "library_size": split.library_size,
        "month_counts": month_counts,
        "hit_bus_ceiling_count": sum(1 for date in dates
                                     if selections[date].hit_bus_ceiling),
        "out_of_band_dates": [{"date": selections[date].date,
                               "peak_mw": selections[date].peak_mw,
                               "reason": selections[date].reason}
                              for date in dates if selections[date].status != "IN_BAND"],
        "split_seed": rules.SPLIT_SEED,
        "applied_counts": split.counts,
        "discarded_dates": {month: list(values)
                            for month, values in split.discarded.items()},
        "holiday_shortfall": split.holiday_shortfall,
        "inputs": {_label(path): _file_sha256(path)
                   for path in (WORKBOOK, SCREENING, DESIGN_DOC,
                                pool_dir / "meter_curves.npz",
                                pool_dir / "meter_sources.csv")
                   if path.exists()},
        "outputs": {
            "day_selection.csv": {"sha256": _file_sha256(out_dir / "day_selection.csv")},
            "source_bus_mapping.csv": {
                "sha256": _file_sha256(out_dir / "source_bus_mapping.csv")},
            "split.json": {"sha256": _file_sha256(out_dir / "split.json")},
            "bus_active_load_mw.npz": {
                "sha256": _file_sha256(out_dir / "bus_active_load_mw.npz"),
                "content_sha256": _npz_content_sha256(active)},
            "bus_reactive_load_mvar.npz": {
                "sha256": _file_sha256(out_dir / "bus_reactive_load_mvar.npz"),
                "content_sha256": _npz_content_sha256(reactive)},
        },
        "rescaled": False,
        "synthetic_background": False,
        "clearing_tested": False,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the v3 candidate-day library.")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--price-curves", type=Path, default=DEFAULT_PRICE_CURVES)
    args = parser.parse_args(argv)
    manifest = run(args.pool, args.out, mv_price_dates(args.price_curves))
    print(f"{manifest['status']}: library_size {manifest['library_size']}, "
          f"months {sorted(manifest['month_counts'])}, counts {manifest['applied_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
