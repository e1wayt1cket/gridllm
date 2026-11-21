# diagnose_profit_opportunity.py
"""Is there a profitable single-unit deviation from truthful bidding?

The question this answers is whether the market the policy is trained in has
any money in it for one storage unit that quotes away from truthful, with the
other three held at truthful. It is not a policy result and it does not train.

Three properties are load-bearing:

*The days are the train partition of the frozen protocol.* `common_split.json`
under the `near` view, the same file and view `runs/FORMAL_7834d0aa_seed11`
records. Any pair that also appears in validation, test or the holiday block
is a refusal, not a filter, so a mis-specified view cannot silently solve a
held-out day.

*The quote is the only thing that varies.* `audit_single_ess_grid.run_day` is
called unchanged: one unit moves to an explicit `(charge, discharge)` held for
all 24 blocks, the other three bid truthful, and the day is settled by the
same 24-block rolling clear with terminal SOC pinned at 0.5. Its truthful
point is re-solved every run and must reproduce the stored B1 day payoff.

*Nothing existing is written.* Results go to a fresh request directory and an
existing one is refused unless `--resume` is given, which only continues a run
whose inputs hash identically to the ones on disk.

Results are written one batch at a time. A batch is one unit on one day over
the whole declared grid; an interrupted run keeps every completed batch and
resumes at the first one that is not, so a kill costs at most one batch.

The B1 reference is `b1_terminal_neutral_results.csv`, the terminal-pinned
rolling baseline. `pilot/b1_daily_results.csv` is the older baseline that did
not pin terminal SOC and must not be used here.

Usage:
    PYTHONPATH=src python src/diagnose_profit_opportunity.py --verify-only
    PYTHONPATH=src python src/diagnose_profit_opportunity.py --request-id r1
    PYTHONPATH=src python src/diagnose_profit_opportunity.py --request-id r1 --resume
    PYTHONPATH=src python src/diagnose_profit_opportunity.py --request-id s1 --days 1

Per-block payoff is NOT recorded: `run_day` returns per-block charge,
discharge and SOC but only a day-level payoff, and it is out of bounds to
change it. Per-block energy and SOC are kept in the batch `.npz`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path.cwd()
DATA = ROOT / "data" / "processed" / "v2_cy_jn_2025"
COMMON_SPLIT = DATA / "common_split.json"
B1_MANIFEST = DATA / "b1_terminal_neutral_manifest.json"
B1_RESULTS = DATA / "b1_terminal_neutral_results.csv"
B1_BLOCKS = DATA / "b1_terminal_neutral_block_payoffs.json"
NEAR_PAIRING = DATA / "near" / "load_price_pairing.csv"
FORMAL_RUN = ROOT / "runs" / "FORMAL_7834d0aa_seed11" / "run_manifest.json"
OUT_ROOT = ROOT / "reports" / "algorithm_diagnostics" / "profit_opportunity"

UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
CHARGE_POINTS = (0.0, 40.0, 80.0, 150.0)
DISCHARGE_POINTS = (0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 250.0, 300.0)
TERMINAL_SOC = 0.5
NEAR_ZERO_CNY = 1.0

ROW_COLUMNS = (
    "unit", "pair_id", "load_date", "price_date", "charge", "discharge",
    "payoff", "b1_payoff", "dG", "fleet", "b1_fleet", "fleet_dG",
    "charge_mwh", "discharge_mwh", "zero_blocks", "soc_end", "terminal_err",
    "all_optimal", "fallbacks", "voltage_cells", "line_cells", "runtime_s",
)

# Named outcomes, assigned in this order so they are exhaustive and exclusive.
OUTCOMES = ("worsened", "true_profit", "loss_stopped",
            "improved_but_loss", "unchanged")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_revision() -> dict:
    try:
        rev = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT,
                             capture_output=True, text=True, check=True)
        status = subprocess.run(("git", "status", "--porcelain"), cwd=ROOT,
                                capture_output=True, text=True, check=True)
        return {"revision": rev.stdout.strip(),
                "dirty": bool(status.stdout.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"revision": None, "dirty": None}


def _classify(payoff: float, b1: float, dG: float) -> str:
    """One label per day. Absolute outcome first, relative second.

    `true_profit` means the unit earned money, not merely that it lost less
    than truthful bidding did.
    """
    if dG < -NEAR_ZERO_CNY:
        return "worsened"
    if payoff > NEAR_ZERO_CNY:
        return "true_profit"
    if abs(payoff) <= NEAR_ZERO_CNY and b1 < -NEAR_ZERO_CNY:
        return "loss_stopped"
    if payoff < -NEAR_ZERO_CNY and dG > NEAR_ZERO_CNY:
        return "improved_but_loss"
    return "unchanged"


# ---------------------------------------------------------------------------
# Train partition


def train_pairs() -> List[dict]:
    """The formal protocol's train pairs, and nothing else.

    The view and name are asserted against the formal run's own manifest, so a
    split file reused under a different view cannot pass unnoticed.
    """
    split = _read_json(COMMON_SPLIT)
    view = split["near"]
    pairs = [{"pair_id": p, "load_date": p.split("|")[0],
              "price_date": p.split("|")[1]} for p in view["train"]]
    forbidden = set(view["validation"]) | set(view["test"]) \
        | set(view["holiday_block"])
    leaked = [p["pair_id"] for p in pairs if p["pair_id"] in forbidden]
    if leaked:
        raise SystemExit(f"train split overlaps a held-out block: {leaked}")

    known = {r["load_date"] + "|" + r["price_date"]
             for r in csv.DictReader(NEAR_PAIRING.open(encoding="utf-8"))}
    missing = [p["pair_id"] for p in pairs if p["pair_id"] not in known]
    if missing:
        raise SystemExit(f"train pair absent from the near pairing: {missing}")

    if FORMAL_RUN.exists():
        manifest = _read_json(FORMAL_RUN)
        if manifest.get("split_name") != "common" \
                or manifest.get("split_view") != "near":
            raise SystemExit(
                "formal run did not use the common/near split; refusing")
        recorded = set(manifest.get("train_pairs") or ())
        if recorded != {p["pair_id"] for p in pairs}:
            raise SystemExit(
                "common_split near train does not match the formal run's "
                "train_pairs")
    return pairs


# ---------------------------------------------------------------------------
# B1 artifact verification


def verify_b1() -> dict:
    """Check the stored B1 against itself and against this environment.

    This certifies the artifact, not the solver. Re-solving would only show
    determinism and would overwrite the only copy of the evidence.
    """
    checks: Dict[str, dict] = {}
    manifest = _read_json(B1_MANIFEST)

    hashed = {}
    for name, rec in list(manifest["inputs"].items()) \
            + list(manifest["outputs"].items()):
        path = ROOT / rec["path"]
        actual = _sha(path) if path.exists() else None
        hashed[name] = {"path": rec["path"], "expected": rec["sha256"],
                        "actual": actual, "match": actual == rec["sha256"]}
    checks["input_output_hashes"] = {
        "pass": all(v["match"] for v in hashed.values()),
        "detail": hashed}

    rows = list(csv.DictReader(B1_RESULTS.open(encoding="utf-8")))
    ids = [r["pair_id"] for r in rows]
    checks["day_completeness"] = {
        "pass": len(rows) == manifest["days"] == 96
        and len(set(ids)) == len(ids) and not manifest["failures"],
        "detail": {"rows": len(rows), "unique": len(set(ids)),
                   "manifest_days": manifest["days"],
                   "failures": manifest["failures"]}}

    blocks = _read_json(B1_BLOCKS)
    lengths, mismatches, missing = set(), [], []
    for r in rows:
        per_unit = blocks.get(r["pair_id"])
        if per_unit is None:
            missing.append(r["pair_id"])
            continue
        for unit in UNITS:
            series = per_unit.get(unit) or []
            lengths.add(len(series))
            if len(series) != 24 or abs(sum(series) - float(r["G_" + unit])) > 1e-6:
                mismatches.append({"pair_id": r["pair_id"], "unit": unit,
                                   "blocks": len(series),
                                   "block_sum": sum(series),
                                   "daily": float(r["G_" + unit])})
    checks["block_aggregation"] = {
        "pass": not missing and not mismatches and lengths == {24}
        and len(blocks) == len(rows),
        "detail": {"block_lengths": sorted(lengths),
                   "mismatches": len(mismatches),
                   "missing": len(missing),
                   "extra_keys": len(set(blocks) - set(ids)),
                   "examples": mismatches[:3]}}

    terminal = max(abs(float(r.get("terminal_err_" + u) or 0.0))
                   for r in rows for u in UNITS)
    infeasible = [r["pair_id"] for r in rows if r["feasible"] != "True"]
    fallbacks = sum(int(float(r.get("fallbacks") or 0)) for r in rows)
    checks["terminal_and_feasibility"] = {
        "pass": terminal <= 1e-6 and not infeasible and fallbacks == 0,
        "detail": {"max_abs_terminal_err": terminal,
                   "infeasible_days": infeasible, "fallbacks": fallbacks,
                   "manifest_terminal_target": manifest["terminal_soc_target"]}}

    import dataclasses
    import screen_v2_cy_jn_2025 as screen
    config = screen.build_config(100.0)
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True,
                         default=str)
    config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    checks["config_matches_environment"] = {
        "pass": config_hash == manifest["config_hash"],
        "detail": {"environment_config_hash": config_hash,
                   "manifest_config_hash": manifest["config_hash"],
                   "cycle_cost": config.storage.cycle_cost}}

    formal = _read_json(FORMAL_RUN) if FORMAL_RUN.exists() else {}
    checks["mechanism_matches_formal_run"] = {
        "pass": bool(formal)
        and formal.get("b1_mechanism_hash") == manifest["mechanism_hash"],
        "detail": {"manifest_mechanism_hash": manifest["mechanism_hash"],
                   "formal_run_mechanism_hash":
                       formal.get("b1_mechanism_hash")}}

    checks["accounting"] = {
        "pass": B1_RESULTS.name == "b1_terminal_neutral_results.csv",
        "detail": {"note": "pilot/b1_daily_results.csv is the unpinned "
                           "baseline and is not used",
                   "soc0": manifest["soc0"],
                   "terminal_soc_target": manifest["terminal_soc_target"],
                   "mechanism": manifest["mechanism"]}}

    return {
        "artifact_verification": "PASS" if all(
            c["pass"] for c in checks.values()) else "FAIL",
        "solver_rerun_reproducibility": "NOT_TESTED",
        "why_not_rerun": "Re-solving would overwrite the only copy of the "
                         "artifact and would only demonstrate determinism.",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Batches


def _batch_key(unit: str, pair: dict) -> str:
    return f"{unit}__{pair['load_date']}__{pair['price_date']}"


def _run_one(job: tuple) -> dict:
    import audit_single_ess_grid as gridmod

    unit, pair, charge, discharge = job
    started = time.time()
    out = gridmod.run_day(pair, unit, charge, discharge)
    return {
        "unit": unit, "pair_id": pair["pair_id"],
        "load_date": pair["load_date"], "price_date": pair["price_date"],
        "charge": charge, "discharge": discharge,
        "payoff": out["payoff"], "fleet": out["fleet"],
        "charge_mwh": out["charge_mwh"],
        "discharge_mwh": out["discharge_mwh"],
        "zero_blocks": out["zero_blocks"], "soc_end": out["soc_end"],
        "all_optimal": out["all_optimal"], "fallbacks": out["fallbacks"],
        "voltage_cells": out["voltage_cells"],
        "line_cells": out["line_cells"],
        "soc_path": out["soc_path"],
        "runtime_s": time.time() - started,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _run_batch(pool, unit: str, pair: dict, charges, discharges,
               b1: float, b1_fleet: float) -> dict:
    """One unit, one day, the whole grid. Returns the index entry."""
    key = _batch_key(unit, pair)
    started = time.time()
    results = list(pool.imap_unordered(_run_one, [
        (unit, pair, c, d) for c in charges for d in discharges]))

    rows, soc = [], {}
    for r in results:
        row = {k: r[k] for k in ROW_COLUMNS
               if k not in ("b1_payoff", "dG", "b1_fleet", "fleet_dG",
                            "terminal_err")}
        row["b1_payoff"] = b1
        row["dG"] = r["payoff"] - b1
        row["b1_fleet"] = b1_fleet
        row["fleet_dG"] = r["fleet"] - b1_fleet
        row["terminal_err"] = r["soc_end"] - TERMINAL_SOC
        rows.append(row)
        soc[f'{r["charge"]}|{r["discharge"]}'] = \
            np.asarray(r["soc_path"], dtype=float)

    rows.sort(key=lambda r: (r["charge"], r["discharge"]))
    csv_path = BATCHES_DIR / f"{key}.csv"
    npz_path = BATCHES_DIR / f"{key}.npz"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, csv_path)
    npz_tmp = npz_path.with_name(npz_path.name + ".tmp")
    # Through a handle: savez appends ".npz" to a name that lacks it, which
    # would leave the temp file under a different name than the one replaced.
    with npz_tmp.open("wb") as fh:
        np.savez_compressed(fh, **soc)
    os.replace(npz_tmp, npz_path)

    return {
        "unit": unit, "pair_id": pair["pair_id"],
        "load_date": pair["load_date"], "price_date": pair["price_date"],
        "n_rows": len(rows),
        "csv": csv_path.name, "csv_sha256": _sha(csv_path),
        "npz": npz_path.name, "npz_sha256": _sha(npz_path),
        "runtime_s": round(time.time() - started, 1),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Summary


def _stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "min": float(arr.min()),
            "max": float(arr.max())}


def _summarize(rows: Sequence[dict], units: Sequence[str],
               n_days: int) -> dict:
    out: Dict[str, dict] = {}
    for unit in units:
        unit_rows = [r for r in rows if r["unit"] == unit]
        by_point: Dict[tuple, List[dict]] = {}
        for r in unit_rows:
            by_point.setdefault((r["charge"], r["discharge"]), []).append(r)

        curve = []
        for (charge, discharge), group in sorted(by_point.items()):
            curve.append({
                "charge": charge, "discharge": discharge,
                "n_days": len(group),
                "mean_dG": float(np.mean([r["dG"] for r in group])),
                "median_dG": float(np.median([r["dG"] for r in group])),
                "min_dG": float(np.min([r["dG"] for r in group])),
                "max_dG": float(np.max([r["dG"] for r in group])),
                "mean_payoff": float(np.mean([r["payoff"] for r in group])),
                "mean_fleet_dG":
                    float(np.mean([r["fleet_dG"] for r in group])),
            })
        best = max(curve, key=lambda c: c["mean_dG"]) if curve else None
        at_best = by_point.get((best["charge"], best["discharge"]), []) \
            if best else []

        counts = {name: 0 for name in OUTCOMES}
        fleet_counts = {name: 0 for name in OUTCOMES}
        for r in at_best:
            counts[_classify(r["payoff"], r["b1_payoff"], r["dG"])] += 1
            fleet_counts[_classify(r["fleet"], r["b1_fleet"],
                                   r["fleet_dG"])] += 1

        active = [r for r in at_best if abs(r["b1_payoff"]) > NEAR_ZERO_CNY]
        truthful = by_point.get((0.0, 0.0), [])
        closure = [abs(r["payoff"] - r["b1_payoff"]) for r in truthful]

        out[unit] = {
            "n_days": len({r["pair_id"] for r in unit_rows}),
            "expected_days": n_days,
            "n_points": len(curve),
            "best_point": best,
            "outcomes_at_best": counts,
            "fleet_outcomes_at_best": fleet_counts,
            "delta_stats_at_best": _stats([r["dG"] for r in at_best])
            if at_best else None,
            "fleet_delta_stats_at_best": _stats([r["fleet_dG"] for r in at_best])
            if at_best else None,
            "active_days": {
                "n": len(active),
                "mean_dG": float(np.mean([r["dG"] for r in active]))
                if active else None,
                "mean_fleet_dG": float(np.mean(
                    [r["fleet_dG"] for r in active])) if active else None,
            },
            "truthful_closure": {
                "n": len(closure),
                "max_abs_error": max(closure) if closure else None,
            },
            "physical": {
                "not_optimal": sum(1 for r in unit_rows
                                   if not r["all_optimal"]),
                "fallbacks": sum(r["fallbacks"] for r in unit_rows),
                "voltage_cells": sum(r["voltage_cells"] for r in unit_rows),
                "line_cells": sum(r["line_cells"] for r in unit_rows),
                "max_abs_terminal_err": max(
                    abs(r["soc_end"] - TERMINAL_SOC) for r in unit_rows)
                if unit_rows else None,
            },
            "curve": curve,
        }
    return out


# ---------------------------------------------------------------------------
# Entry point


def main(argv: Sequence[str] | None = None) -> int:
    global BATCHES_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--units", nargs="+", default=list(UNITS))
    parser.add_argument("--days", type=int, default=0,
                        help="use only the first N train days")
    parser.add_argument("--charge-points", nargs="+", type=float,
                        default=list(CHARGE_POINTS))
    parser.add_argument("--discharge-points", nargs="+", type=float,
                        default=list(DISCHARGE_POINTS))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    unknown = [u for u in args.units if u not in UNITS]
    if unknown:
        raise SystemExit(f"unknown unit(s): {unknown}")

    verification = verify_b1()
    print(json.dumps({"b1_verification": verification["artifact_verification"],
                      "failing": [k for k, v in verification["checks"].items()
                                  if not v["pass"]]}))
    if args.verify_only:
        return 0 if verification["artifact_verification"] == "PASS" else 2
    if verification["artifact_verification"] != "PASS":
        raise SystemExit("B1 artifact verification failed; refusing to scan")

    pairs = train_pairs()
    if args.days:
        pairs = pairs[:args.days]

    b1_rows = list(csv.DictReader(B1_RESULTS.open(encoding="utf-8")))
    b1_unit = {u: {r["pair_id"]: float(r["G_" + u]) for r in b1_rows}
               for u in UNITS}
    b1_fleet = {r["pair_id"]: sum(float(r["G_" + u]) for u in UNITS)
                for r in b1_rows}
    unknown_pairs = [p["pair_id"] for p in pairs
                     if p["pair_id"] not in b1_fleet]
    if unknown_pairs:
        raise SystemExit(f"train pairs missing from B1: {unknown_pairs}")

    wanted = dict(
        units=list(args.units),
        charge_points=list(args.charge_points),
        discharge_points=list(args.discharge_points),
        days=[p["pair_id"] for p in pairs],
        terminal_soc=TERMINAL_SOC,
        near_zero_cny=NEAR_ZERO_CNY,
    )
    n_runs = len(pairs) * len(args.units) * len(args.charge_points) \
        * len(args.discharge_points)
    n_batches = len(pairs) * len(args.units)

    request_id = args.request_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / request_id

    inputs_path = out_dir / "inputs.json"
    if out_dir.exists():
        if not args.resume:
            raise SystemExit(f"{out_dir} already exists; pick another request "
                             "id or pass --resume")
        if not inputs_path.exists():
            raise SystemExit(f"{out_dir} has no inputs.json; refusing resume")
        saved = _read_json(inputs_path)
        current_files = {str(p.relative_to(ROOT)): _sha(p) for p in (
            COMMON_SPLIT, B1_MANIFEST, B1_RESULTS, B1_BLOCKS, NEAR_PAIRING,
            FORMAL_RUN) if p.exists()}
        if saved.get("files") != current_files:
            raise SystemExit("input hashes changed since this run started")
        if saved.get("params") != wanted:
            raise SystemExit("request parameters differ from the saved run")
        if saved.get("b1_verification") != \
                verification["artifact_verification"]:
            raise SystemExit("B1 verification state changed")
        print(json.dumps({"resume": request_id,
                          "batches_total": saved["n_batches"]}))
    else:
        if args.resume:
            raise SystemExit(f"no run to resume at {out_dir}")

    if args.dry_run:
        print(json.dumps({"request_id": request_id, "days": len(pairs),
                          "units": args.units, "runs": n_runs,
                          "batches": n_batches, "workers": args.workers}))
        return 0

    BATCHES_DIR = out_dir / "batches"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        BATCHES_DIR.mkdir()
        _atomic_write_text(inputs_path, json.dumps({
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "git": _git_revision(),
            "files": {str(p.relative_to(ROOT)): _sha(p) for p in (
                COMMON_SPLIT, B1_MANIFEST, B1_RESULTS, B1_BLOCKS,
                NEAR_PAIRING, FORMAL_RUN) if p.exists()},
            "params": wanted,
            "b1_verification": verification["artifact_verification"],
            "n_days": len(pairs), "n_runs": n_runs, "n_batches": n_batches,
        }, indent=2, sort_keys=True))
        _atomic_write_text(out_dir / "b1_verification.json",
                           json.dumps(verification, indent=2, sort_keys=True))
    BATCHES_DIR.mkdir(exist_ok=True)

    index_path = out_dir / "batch_index.json"
    index = _read_json(index_path) if index_path.exists() else {"batches": {}}
    # A batch counts as done only if its files are still there and still hash
    # to what was recorded; anything else is redone rather than trusted.
    for key, entry in list(index["batches"].items()):
        if entry.get("status") != "complete":
            continue
        ok = True
        for field, hash_field in (("csv", "csv_sha256"), ("npz", "npz_sha256")):
            path = BATCHES_DIR / entry[field]
            if not path.exists() or _sha(path) != entry[hash_field]:
                ok = False
        if not ok:
            entry["status"] = "invalidated"
    done = {k for k, v in index["batches"].items() if v.get("status") == "complete"}

    started = time.time()
    completed = 0
    with mp.Pool(args.workers) as pool:
        for unit in args.units:
            for pair in pairs:
                key = _batch_key(unit, pair)
                if key in done:
                    continue
                entry = _run_batch(pool, unit, pair, args.charge_points,
                                   args.discharge_points,
                                   b1_unit[unit][pair["pair_id"]],
                                   b1_fleet[pair["pair_id"]])
                entry["status"] = "complete"
                index["batches"][key] = entry
                _atomic_write_text(index_path, json.dumps(
                    index, indent=2, sort_keys=True))
                completed += 1
                total_done = len([v for v in index["batches"].values()
                                  if v.get("status") == "complete"])
                print(f"batch {total_done}/{n_batches} {key} "
                      f"{entry['runtime_s']}s", flush=True)

    finished = [v for v in index["batches"].values()
                if v.get("status") == "complete"]
    if len(finished) != n_batches:
        print(json.dumps({"status": "INCOMPLETE",
                          "completed": len(finished), "expected": n_batches}))
        return 3

    rows = []
    for entry in index["batches"].values():
        with (BATCHES_DIR / entry["csv"]).open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        for k in ("payoff", "b1_payoff", "dG", "fleet", "b1_fleet",
                  "fleet_dG", "charge_mwh", "discharge_mwh", "soc_end",
                  "terminal_err", "charge", "discharge"):
            r[k] = float(r[k])
        for k in ("zero_blocks", "fallbacks", "voltage_cells", "line_cells"):
            r[k] = int(r[k])
        r["all_optimal"] = str(r["all_optimal"]) == "True"

    summary = {
        "request_id": request_id,
        "status": "COMPLETE",
        "b1_artifact_verification": verification["artifact_verification"],
        "b1_solver_rerun_reproducibility":
            verification["solver_rerun_reproducibility"],
        "n_days": len(pairs),
        "n_runs": len(rows),
        "n_batches": n_batches,
        "units": _summarize(rows, args.units, len(pairs)),
        "session_runtime_s": time.time() - started,
    }
    _atomic_write_text(out_dir / "summary.json", json.dumps(
        summary, indent=2, sort_keys=True, default=float))
    hashes = {p.name: _sha(p) for p in sorted(out_dir.iterdir())
              if p.is_file()}
    hashes.update({f"batches/{p.name}": _sha(p)
                   for p in sorted(BATCHES_DIR.iterdir()) if p.is_file()})
    _atomic_write_text(out_dir / "hashes.json",
                       json.dumps(hashes, indent=2, sort_keys=True))

    print(json.dumps({
        "out_dir": str(out_dir.relative_to(ROOT)),
        "status": "COMPLETE", "runs": len(rows),
        "best": {u: summary["units"][u]["best_point"] for u in args.units},
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
