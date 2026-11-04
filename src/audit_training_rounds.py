"""Round-by-round validation audit for the development training run.

Loads each round's real checkpoint through a reconstructed, loadable root,
re-evaluates the validation split with the enhanced evaluator, and compares
the result with the snapshot written at training time. It also hashes every
checkpoint tree and records episode/counter consistency.

It never evaluates the test split and never reads the H2 holdout pairs.

Usage:
    PYTHONPATH=src python src/audit_training_rounds.py \
        --run-id EXPLORATORY_NEAR_B_C2_GATE_seed11_v2 --rounds 1-5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import rl_b_c2_gate as mod
import train_b_c2_gate as tb

ROOT = Path.cwd()
RUNS = ROOT / "runs"
SNAPSHOTS = ROOT / "snapshots"
NEAR_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
FULL_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
NEAR_SPLIT = ROOT / "data/processed/v2_cy_jn_2025/near/split.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            h.update(str(item.relative_to(path)).encode("utf-8"))
            h.update(item.read_bytes())
    return h.hexdigest()


def link_tree(src: Path, dst: Path) -> None:
    for dirpath, _, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target = dst if rel == "." else dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            s = Path(dirpath) / name
            t = target / name
            try:
                os.link(s, t)
            except OSError:
                shutil.copy2(s, t)


def parse_rounds(text: str) -> List[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]


def pairs_for(load_dates: Sequence[str], load_to_pair: dict,
              full_by_pair: dict, near_by_pair: dict) -> List[dict]:
    out = []
    for load in load_dates:
        pid = load_to_pair[load]
        out.append({"load_date": load,
                    "price_date": full_by_pair[pid]["price_date"],
                    "season": near_by_pair[pid].get("season", "")})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", default="1-10")
    args = parser.parse_args(argv)
    rounds = parse_rounds(args.rounds)

    run_dir = RUNS / args.run_id
    snap_dir = SNAPSHOTS / args.run_id
    audit_root_base = RUNS / f"{args.run_id}_audit_roots"
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("h2_holdout_read"):
        raise RuntimeError("the run manifest says H2 was read")
    latest_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8"))

    near_rows = read_csv(NEAR_PAIRING)
    full_rows = read_csv(FULL_PAIRING)
    near_by_pair = {f"{r['load_date']}|{r['price_date']}": r for r in near_rows}
    full_by_pair = {f"{r['load_date']}|{r['price_date']}": r for r in full_rows}
    load_to_pair = {r["load_date"]: f"{r['load_date']}|{r['price_date']}"
                    for r in near_rows}
    split = json.loads(NEAR_SPLIT.read_text(encoding="utf-8"))
    validation = pairs_for(split["validation"], load_to_pair,
                           full_by_pair, near_by_pair)
    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(encoding="utf-8"))
    mechanism_hash = b1_manifest["mechanism_hash"]
    b1_cache = mod.load_b1_cache()

    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    template_env = tb.envmod.build_env_for_pair(
        validation[0], differential_reward=False, terminal_soc_target=0.5)
    trainer = mod.MATD3BC2Gate(template_env, experiment)

    out_path = run_dir / "validation_audit_by_round.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    index_path = run_dir / "checkpoint_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) \
        if index_path.exists() else {}

    for round_no in rounds:
        snap_path = snap_dir / f"round_{round_no:03d}.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        step = int(Path(snap["checkpoint"]).name.split("_")[-1])
        step_dir = run_dir / "checkpoints" / f"step_{step:08d}"
        if not step_dir.exists():
            raise FileNotFoundError(step_dir)
        audit_root = audit_root_base / f"round_{round_no:03d}"
        if audit_root.exists():
            shutil.rmtree(audit_root)
        (audit_root / "checkpoints").mkdir(parents=True)
        link_tree(step_dir, audit_root / "checkpoints" / step_dir.name)
        counters = json.loads(
            (step_dir / "counters.json").read_text(encoding="utf-8"))
        manifest = json.loads(json.dumps(latest_manifest))
        manifest["step"] = step
        manifest["counters"] = counters
        manifest["buffer"] = dict(manifest["buffer"])
        manifest["buffer"]["count"] = int(counters.get("buffer_len", 0))
        (audit_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        trainer.load(audit_root)
        val = tb.evaluate(trainer, validation, b1_cache,
                          mechanism_hash=mechanism_hash,
                          pair_rows=full_by_pair)
        original = snap.get("validation", {}).get("mean_fleet_delta")
        reproduced = (original is not None
                      and abs(val["mean_fleet_delta"] - float(original))
                      < 1e-6)
        record = {
            "round": round_no,
            "episode": int(snap["episode"]),
            "step": step,
            "counters_in_checkpoint": counters,
            "validation": val,
            "snapshot_mean_fleet_delta": original,
            "reproduced": bool(reproduced),
            "audit_manifest_sha256": sha(audit_root / "manifest.json"),
            "counters_sha256": sha(step_dir / "counters.json"),
            "checkpoint_tree_sha256": tree_sha(step_dir),
            "snapshot_sha256": sha(snap_path),
        }
        existing[str(round_no)] = record
        index[str(round_no)] = {
            "round": round_no,
            "episode": int(snap["episode"]),
            "step": step,
            "checkpoint_path": str(step_dir),
            "counters": counters,
            "manifest_sha256": sha(audit_root / "manifest.json"),
            "counters_sha256": record["counters_sha256"],
            "checkpoint_tree_sha256": record["checkpoint_tree_sha256"],
            "snapshot_sha256": record["snapshot_sha256"],
            "reproduced": bool(reproduced),
        }
        print(f"round {round_no}: step {step} mean_fleet_delta="
              f"{val['mean_fleet_delta']:.3f} reproduced={reproduced}",
              flush=True)

    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
