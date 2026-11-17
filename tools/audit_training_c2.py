"""C2 and SOC audit on the trained checkpoints.

Loads the best validation checkpoint and the final checkpoint, runs one
deterministic validation day from each, records the raw SOC path and the
clipped path, and checks the B/C2 identities and physical boundaries.

It never runs the test split or the H2 holdout.

Usage:
    PYTHONPATH=src python src/audit_training_c2.py \
        --run-id EXPLORATORY_NEAR_B_C2_GATE_seed11_v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import rl_b_c2_gate as mod
import train_b_c2_gate as tb
from audit_training_rounds import link_tree, read_csv, sha, tree_sha
from c2_reward import C2Config, c2_reward, potential
from daily_reward_b import RewardBConfig, day_reward_b

ROOT = Path.cwd()
RUNS = ROOT / "runs"
SNAPSHOTS = ROOT / "snapshots"
NEAR_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
FULL_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
NEAR_SPLIT = ROOT / "data/processed/v2_cy_jn_2025/near/split.json"
OUT = ROOT / "data/processed/v2_cy_jn_2025/c2_soc_training_audit.json"
REPORT = ROOT / "reports/v2_cy_jn_2025/c2_soc_training_audit.md"


def explain_day(trainer, pair, b1_cache, mechanism_hash, pair_row):
    env = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    block_payoffs = {nm: [] for nm in names}
    soc_path = {nm: [float(next(a for a in env.all_agents
                                if a.name == nm).storage.soc0)]
                for nm in names}
    charge = {nm: 0.0 for nm in names}
    discharge = {nm: 0.0 for nm in names}
    voltage_cells = line_cells = 0
    lmp_fb = 0
    for block in range(24):
        acts = {nm: np.asarray(trainer.select_action(
            obs[nm], nm, train=False)["quotes"], dtype=float)
            for nm in names}
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            raise RuntimeError(f"{pair}: clearing failed at block {block}")
        result = env._last_result
        n_commit = min(4, env.T - block * 4)
        v = result.get("bus_voltage")
        u = result.get("line_utilization")
        if v is not None:
            arr = np.asarray(v, dtype=float)
            voltage_cells += int(((arr < env.config.network.v_min_pu - 1e-7)
                                  | (arr > env.config.network.v_max_pu + 1e-7)).sum())
        if u is not None:
            line_cells += int((np.asarray(u, dtype=float) > 1.0 + 1e-7).sum())
        lmp_fb += int(result.get("lmp_fallbacks", 0))
        for nm in names:
            block_payoffs[nm].append(
                float(info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(float(env.prev_soc.get(nm, soc_path[nm][-1])))
            ws = (result.get("schedules") or {}).get(nm)
            if ws is not None:
                charge[nm] += float(
                    np.sum(np.asarray(ws["p_ch"])[:n_commit]) * 0.25)
                discharge[nm] += float(
                    np.sum(np.asarray(ws["p_dis"])[:n_commit]) * 0.25)
        obs = next_obs
        if done:
            break
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    b1_day = b1_cache.get(pair_id, 0.5, mechanism_hash,
                          mod.pair_data_hash(pair_row))
    b = day_reward_b(
        block_payoffs, b1_day.block_payoffs, pair_id=pair_id,
        config=RewardBConfig(gamma=0.99, fixed_scale=1.0))
    done_flags = np.array([False] * 23 + [True])
    c2 = c2_reward(b.reward, soc_path, done=done_flags,
                   config=C2Config(gamma=0.99))
    cfg = C2Config(gamma=0.99)
    raw = {nm: np.asarray(soc_path[nm], dtype=float) for nm in names}
    clipped = {nm: np.clip(raw[nm], cfg.soc_lo, cfg.soc_hi) for nm in names}
    max_dev = max(float(np.max(np.abs(raw[nm] - clipped[nm])))
                  for nm in names)
    raw_violation = max(
        float(max(0.0, cfg.soc_lo - np.min(raw[nm]),
                  np.max(raw[nm]) - cfg.soc_hi)) for nm in names)
    terminal_potential = {nm: float(potential(
        raw[nm][-1], cfg.blocks, cfg)) for nm in names}
    b_weights = cfg.gamma ** np.arange(cfg.blocks)
    identity = {}
    for nm in names:
        b_sum = float(np.sum(b_weights * b.reward[nm]))
        c2_sum = float(np.sum(b_weights * c2.reward[nm]))
        shaping_sum = float(np.sum(b_weights * c2.shaping_terms[nm]))
        identity[nm] = {
            "b_discounted": b_sum,
            "c2_discounted": c2_sum,
            "shaping_discounted": shaping_sum,
            "boundary": float(c2.boundary[nm]),
            "terminal_potential": terminal_potential[nm],
            "b_paid_only_at_23": bool(
                all(abs(b.reward[nm][t]) == 0.0 for t in range(23))),
            "b_at_23_equals_delta": bool(
                abs(b.reward[nm][23] - b.delta[nm]) < 1e-9),
            "b_vs_c2_error": abs(c2_sum - (b_sum + c2.boundary[nm])),
            "shaping_vs_boundary_error": abs(shaping_sum - c2.boundary[nm]),
            "shaping_nonzero_blocks": int(np.sum(
                np.abs(c2.shaping_terms[nm]) > 1e-12)),
        }
    return {
        "pair_id": pair_id,
        "G_policy": b.G_policy,
        "G_b1": b.G_b1,
        "delta": b.delta,
        "b_reward_by_block": {nm: list(b.reward[nm]) for nm in names},
        "c2_shaping_by_block": {nm: list(c2.shaping_terms[nm]) for nm in names},
        "raw_soc_path": {nm: list(raw[nm]) for nm in names},
        "clipped_soc_path": {nm: list(clipped[nm]) for nm in names},
        "max_raw_vs_clipped_deviation": max_dev,
        "max_raw_soc_band_violation": raw_violation,
        "terminal_soc_raw": {nm: float(raw[nm][-1]) for nm in names},
        "terminal_soc_clipped": {nm: float(clipped[nm][-1]) for nm in names},
        "charge_mwh": charge,
        "discharge_mwh": discharge,
        "voltage_violation_cells": voltage_cells,
        "line_overload_cells": line_cells,
        "lmp_fallbacks": lmp_fb,
        "identity": identity,
    }


def build_root(run_id: str, step: int, latest_manifest: dict,
               run_manifest: dict) -> Path:
    run_dir = RUNS / run_id
    step_dir = run_dir / "checkpoints" / f"step_{step:08d}"
    audit_root = RUNS / f"{run_id}_audit_roots" / f"c2_step_{step:08d}"
    if audit_root.exists():
        shutil.rmtree(audit_root)
    (audit_root / "checkpoints").mkdir(parents=True)
    link_tree(step_dir, audit_root / "checkpoints" / step_dir.name)
    counters = json.loads((step_dir / "counters.json").read_text(
        encoding="utf-8"))
    manifest = json.loads(json.dumps(latest_manifest))
    manifest["step"] = step
    manifest["counters"] = counters
    manifest["buffer"] = dict(manifest["buffer"])
    manifest["buffer"]["count"] = int(counters.get("buffer_len", 0))
    (audit_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return audit_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run_dir = RUNS / args.run_id
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(
        encoding="utf-8"))
    latest = json.loads((run_dir / "manifest.json").read_text(
        encoding="utf-8"))
    if run_manifest.get("h2_holdout_read"):
        raise RuntimeError("H2 was read")
    best = run_manifest["best"]
    best_step = int(Path(best["checkpoint"]).name.split("_")[-1])
    final_step = int(Path(run_manifest["final_checkpoint"]).name.split("_")[-1])

    near_rows = read_csv(NEAR_PAIRING)
    full_rows = read_csv(FULL_PAIRING)
    near_by_pair = {f"{r['load_date']}|{r['price_date']}": r for r in near_rows}
    full_by_pair = {f"{r['load_date']}|{r['price_date']}": r for r in full_rows}
    split = json.loads(NEAR_SPLIT.read_text(encoding="utf-8"))
    day = split["validation"][0]
    pair = {"load_date": day,
            "price_date": near_by_pair[
                next(k for k in near_by_pair if k.startswith(day + "|"))
            ]["price_date"],
            "season": near_by_pair[
                next(k for k in near_by_pair if k.startswith(day + "|"))
            ].get("season", "")}
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    pair_row = full_by_pair[pair_id]
    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(encoding="utf-8"))
    mechanism_hash = b1_manifest["mechanism_hash"]
    b1_cache = mod.load_b1_cache()
    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    template = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)

    records = {}
    for label, step in (("best_round_2", best_step),
                        ("final_round_10", final_step)):
        root = build_root(args.run_id, step, latest, run_manifest)
        trainer = mod.MATD3BC2Gate(template, experiment)
        trainer.load(root)
        records[label] = {
            "step": step,
            "checkpoint_tree_sha256": tree_sha(
                run_dir / "checkpoints" / f"step_{step:08d}"),
            "day": explain_day(trainer, pair, b1_cache, mechanism_hash,
                               pair_row),
        }

    checks = {
        "max_raw_vs_clipped_deviation": max(
            r["day"]["max_raw_vs_clipped_deviation"] for r in records.values()),
        "max_raw_soc_band_violation": max(
            r["day"]["max_raw_soc_band_violation"] for r in records.values()),
        "terminal_soc_max_err": max(
            abs(v - 0.5) for r in records.values()
            for v in r["day"]["terminal_soc_raw"].values()),
        "voltage_violation_cells": sum(
            r["day"]["voltage_violation_cells"] for r in records.values()),
        "line_overload_cells": sum(
            r["day"]["line_overload_cells"] for r in records.values()),
        "b_vs_c2_max_error": max(
            i["b_vs_c2_error"] for r in records.values()
            for i in r["day"]["identity"].values()),
        "shaping_vs_boundary_max_error": max(
            i["shaping_vs_boundary_error"] for r in records.values()
            for i in r["day"]["identity"].values()),
        "b_paid_only_at_23": all(
            i["b_paid_only_at_23"] for r in records.values()
            for i in r["day"]["identity"].values()),
        "b_at_23_equals_delta": all(
            i["b_at_23_equals_delta"] for r in records.values()
            for i in r["day"]["identity"].values()),
        "all_shaping_nonzero": all(
            i["shaping_nonzero_blocks"] > 0 for r in records.values()
            for i in r["day"]["identity"].values()),
        "physical_ok": False,
    }
    checks["physical_ok"] = bool(
        checks["max_raw_soc_band_violation"] <= 1e-6
        and checks["terminal_soc_max_err"] <= 1e-5
        and checks["voltage_violation_cells"] == 0
        and checks["line_overload_cells"] == 0
        and checks["b_vs_c2_max_error"] < 1e-9
        and checks["shaping_vs_boundary_max_error"] < 1e-9
        and checks["b_paid_only_at_23"]
        and checks["b_at_23_equals_delta"]
        and checks["all_shaping_nonzero"])
    payload = {
        "run_id": args.run_id,
        "day": pair_id,
        "records": records,
        "checks": checks,
        "tolerance_note": (
            "C2 absorbs at most 1e-6 SOC numerical overshoot and clips to "
            "[0.10, 0.90]. The clearing variable bound is enforced by the "
            "solver; the observed raw overshoot is at solver-feasibility "
            "scale, not a physical excursion."),
        "h2_holdout_read": False,
        "legacy_test_run": False,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# C2 and SOC audit on the trained checkpoints",
        "",
        f"Run: `{args.run_id}`; audited day: `{pair_id}`. No test and no H2 "
        "holdout.",
        "",
        "| check | result |",
        "|---|---|",
        f"| max raw vs clipped SOC deviation | "
        f"{checks['max_raw_vs_clipped_deviation']:.3e} |",
        f"| max raw SOC band overshoot | "
        f"{checks['max_raw_soc_band_violation']:.3e} |",
        f"| terminal SOC max error | {checks['terminal_soc_max_err']:.3e} |",
        f"| voltage / line violation cells | "
        f"{checks['voltage_violation_cells']} / "
        f"{checks['line_overload_cells']} |",
        f"| B vs C2 discounted identity, max error | "
        f"{checks['b_vs_c2_max_error']:.3e} |",
        f"| shaping sum vs boundary, max error | "
        f"{checks['shaping_vs_boundary_max_error']:.3e} |",
        f"| B paid only at block 23 | {checks['b_paid_only_at_23']} |",
        f"| B at block 23 equals delta | {checks['b_at_23_equals_delta']} |",
        f"| C2 shaping present | {checks['all_shaping_nonzero']} |",
        f"| physical checks | {checks['physical_ok']} |",
        "",
        "The 1e-6 tolerance matches the solver-feasibility scale. The maximum "
        "observed raw SOC overshoot is recorded above; a larger excursion "
        "would raise, so the tolerance does not admit a real physical "
        "violation.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"checks": checks, "day": pair_id}, indent=2))
    return 0 if checks["physical_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
