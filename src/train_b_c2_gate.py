"""Development training run for MATD3_B_C2_GATE.

This is an **exploratory/development** run. The current data have no joint
source-load holdout (H1 = 0), so its test-set numbers cannot support a paper
claim. It trains on the NEAR training split only and never reads the H2
holdout pairs.

Per round: one exposure to each of the 53 training days, then deterministic
validation on the 11 NEAR validation days. The selection metric is the mean
fleet ΔG from actual settlement, never a training reward. A full resume
checkpoint is saved every round; a separate best checkpoint is saved whenever
validation improves.

Usage:
    PYTHONPATH=src python src/train_b_c2_gate.py \
        --seed 11 --exposures 10 --run-id EXPLORATORY_NEAR_B_C2_GATE_seed11
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import b1_daily_baseline as b1_base
import rl_b_c2_gate as mod
import v2_cy_jn_env as envmod
from market import clear_market  # noqa: F401  (kept for import-order parity)

ROOT = Path.cwd()
NEAR_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv"
FULL_PAIRING = ROOT / "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
NEAR_SPLIT = ROOT / "data/processed/v2_cy_jn_2025/near/split.json"
BUS_LOADS = ROOT / "data/processed/v2_cy_jn_2025/bus_loads_96x33.npz"
PRICE_CURVES = ROOT / "data/processed/v2_cy_jn_2025/price_curves.npz"
COMMON_SPLIT = ROOT / "data/processed/v2_cy_jn_2025/common_split.json"
RUNS = ROOT / "runs"
SNAPSHOTS = ROOT / "snapshots"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def code_revision() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "working-tree"


def season_order(pairs: Sequence[dict], seed: int) -> List[dict]:
    """Season-stratified, deterministic order; every three episodes mix seasons."""
    by = {}
    for pair in pairs:
        by.setdefault(pair.get("season", ""), []).append(pair)
    rng = random.Random(seed)
    for values in by.values():
        rng.shuffle(values)
    order = []
    seasons = [s for s in ("summer", "autumn", "winter") if s in by]
    while any(by[s] for s in seasons):
        for s in seasons:
            if by[s]:
                order.append(by[s].pop())
    return order


def data_hashes() -> Dict[str, str]:
    return {
        "pairing": sha(NEAR_PAIRING),
        "bus_loads": sha(BUS_LOADS),
        "price_curves": sha(PRICE_CURVES),
        "common_split": sha(COMMON_SPLIT),
    }


def eval_day(trainer: mod.MATD3BC2Gate, pair: dict, b1_cache, *,
             mechanism_hash: str, pair_row: dict) -> dict:
    """Deterministic full-day evaluation; no buffer write, no exposure.

    Records settlement, per-ESS energy/cycles, raw SOC extrema and the
    deterministic gate behaviour of every quote leg. It does not call
    `train_day`, so validation cannot change the model or the exposure log.
    """
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    block_payoffs = {nm: [] for nm in names}
    soc_path = {nm: [float(next(a for a in env.all_agents
                                if a.name == nm).storage.soc0)]
                for nm in names}
    charge_mwh = {nm: 0.0 for nm in names}
    discharge_mwh = {nm: 0.0 for nm in names}
    gate_open = {nm: [0, 0] for nm in names}
    gate_prob_sum = {nm: [0.0, 0.0] for nm in names}
    gate_closed = {nm: [0, 0] for nm in names}
    quote_open = {nm: [[], []] for nm in names}
    mode_counts: Dict[int, int] = {}
    sat_upper = sat_lower = 0
    soc_min = {nm: math.inf for nm in names}
    soc_max = {nm: -math.inf for nm in names}
    decisions = {nm: 0 for nm in names}
    for block in range(24):
        acts, unit_modes = {}, []
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            unit_modes.append(int(d["mode"]))
            gates = np.asarray(d["gates"], dtype=float)
            probs = np.asarray(d["probs"], dtype=float)
            price_quotes = np.asarray(d["price_quotes"], dtype=float)
            decisions[nm] += 1
            for leg in (0, 1):
                gate_prob_sum[nm][leg] += float(probs[leg])
                if gates[leg] > 0:
                    gate_open[nm][leg] += 1
                    quote_open[nm][leg].append(float(price_quotes[leg]))
                    if price_quotes[leg] >= 0.99 * mod.LEG_HIGHS[leg]:
                        sat_upper += 1
                    if price_quotes[leg] <= 0.01 * mod.LEG_HIGHS[leg]:
                        sat_lower += 1
                else:
                    gate_closed[nm][leg] += 1
        mode = mod.fleet_mode(unit_modes)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            raise RuntimeError(f"{pair}: clearing failed at block {block}")
        result = env._last_result
        n_commit = min(4, env.T - block * 4)
        for nm in names:
            block_payoffs[nm].append(
                float(info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(float(env.prev_soc.get(nm, soc_path[nm][-1])))
            ws = (result.get("schedules") or {}).get(nm)
            if ws is None:
                continue
            charge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_ch"])[:n_commit]) * 0.25)
            discharge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_dis"])[:n_commit]) * 0.25)
            soc_arr = ws.get("soc")
            if soc_arr is not None:
                vals = np.asarray(soc_arr, dtype=float)[:n_commit + 1]
                if vals.size:
                    soc_min[nm] = min(soc_min[nm], float(np.min(vals)))
                    soc_max[nm] = max(soc_max[nm], float(np.max(vals)))
        obs = next_obs
        if done:
            break
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    b1_day = b1_cache.get(pair_id, 0.5, mechanism_hash,
                          mod.pair_data_hash(pair_row))
    G_pi = {nm: float(sum(block_payoffs[nm])) for nm in names}
    G_b1 = {nm: float(sum(b1_day.block_payoffs[nm])) for nm in names}
    delta = {nm: G_pi[nm] - G_b1[nm] for nm in names}
    e_max = {nm: float(next(a for a in env.all_agents
                            if a.name == nm).storage.e_max) for nm in names}
    cycles = {
        nm: ((charge_mwh[nm] + discharge_mwh[nm]) / (2.0 * e_max[nm])
             if e_max[nm] > 0 else 0.0)
        for nm in names}
    for nm in names:
        if not math.isfinite(soc_min[nm]):
            soc_min[nm] = min(soc_path[nm])
            soc_max[nm] = max(soc_path[nm])
    result = env._last_result
    voltage = result.get("bus_voltage")
    v = np.asarray(voltage, dtype=float) if voltage is not None else np.zeros(0)
    util_src = result.get("line_utilization")
    util = np.asarray(util_src, dtype=float) \
        if util_src is not None else np.zeros(0)
    return {
        "pair_id": pair_id,
        "season": pair.get("season", ""),
        "G_policy": G_pi,
        "G_b1": G_b1,
        "delta": delta,
        "delta_fleet": float(sum(delta.values())),
        "terminal_soc": {nm: soc_path[nm][-1] for nm in names},
        "charge_mwh": charge_mwh,
        "discharge_mwh": discharge_mwh,
        "cycles": cycles,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "clears": 24,
        "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
        "voltage_violation_cells": int(
            ((v < env.config.network.v_min_pu - 1e-7)
             | (v > env.config.network.v_max_pu + 1e-7)).sum())
        if v.size else 0,
        "line_overload_cells": int((util > 1.0 + 1e-7).sum())
        if util.size else 0,
        "gate_open_frequency": {
            nm: [gate_open[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "gate_mean_probability": {
            nm: [gate_prob_sum[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "gate_closed_fraction": {
            nm: [gate_closed[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "quote_open_values": quote_open,
        "gate_saturation_upper": sat_upper,
        "gate_saturation_lower": sat_lower,
        "mode_counts": {str(k): int(v) for k, v in sorted(mode_counts.items())},
        "distinct_modes": len(mode_counts),
    }


def _quote_stats(values: Sequence[float]):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def evaluate(trainer: mod.MATD3BC2Gate, pairs: Sequence[dict], b1_cache,
             *, mechanism_hash: str, pair_rows: Dict[str, dict]) -> dict:
    days = []
    for pair in pairs:
        pair_id = f"{pair['load_date']}|{pair['price_date']}"
        row = pair_rows[pair_id]
        days.append(eval_day(trainer, pair, b1_cache,
                             mechanism_hash=mechanism_hash, pair_row=row))
    units = trainer.agent_names
    fleet = [d["delta_fleet"] for d in days]
    per_unit = {nm: float(np.mean([d["delta"][nm] for d in days]))
                for nm in units}
    terminal_err = max(
        abs(v - 0.5) for d in days for v in d["terminal_soc"].values())
    voltage_cells = int(sum(d["voltage_violation_cells"] for d in days))
    line_cells = int(sum(d["line_overload_cells"] for d in days))
    c2 = mod.C2Config()
    raw_soc_min = min(d["soc_min"][nm] for d in days for nm in units)
    raw_soc_max = max(d["soc_max"][nm] for d in days for nm in units)
    raw_band_violation = max(0.0, c2.soc_lo - raw_soc_min,
                             raw_soc_max - c2.soc_hi)
    if terminal_err > 1e-5 or voltage_cells or line_cells \
            or raw_band_violation > 1e-6:
        raise RuntimeError(
            "validation/evaluation physical check failed: "
            f"terminal_err={terminal_err:.2e} voltage={voltage_cells} "
            f"line={line_cells} raw_soc_band={raw_band_violation:.2e}")
    gate_open = {
        f"{nm}.{leg}": float(np.mean([d["gate_open_frequency"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    gate_prob = {
        f"{nm}.{leg}": float(np.mean([d["gate_mean_probability"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    gate_closed = {
        f"{nm}.{leg}": float(np.mean([d["gate_closed_fraction"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    quote_dist = {
        f"{nm}.{leg}": _quote_stats(
            [value for d in days for value in d["quote_open_values"][nm][i]])
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    modes: Dict[str, int] = {}
    for d in days:
        for key, count in d["mode_counts"].items():
            modes[key] = modes.get(key, 0) + int(count)
    return {
        "days": days,
        "mean_fleet_delta": float(np.mean(fleet)),
        "median_fleet_delta": float(np.median(fleet)),
        "mean_delta_by_unit": per_unit,
        "positive_days": int(sum(1 for x in fleet if x > 0)),
        "mean_charge_mwh": {nm: float(np.mean(
            [d["charge_mwh"][nm] for d in days])) for nm in units},
        "mean_discharge_mwh": {nm: float(np.mean(
            [d["discharge_mwh"][nm] for d in days])) for nm in units},
        "mean_cycles": {nm: float(np.mean(
            [d["cycles"][nm] for d in days])) for nm in units},
        "mean_terminal_soc": {nm: float(np.mean(
            [d["terminal_soc"][nm] for d in days])) for nm in units},
        "soc_min": raw_soc_min,
        "soc_max": raw_soc_max,
        "raw_soc_band_violation": raw_band_violation,
        "voltage_violation_cells": voltage_cells,
        "line_overload_cells": line_cells,
        "lmp_fallbacks": int(sum(d["lmp_fallbacks"] for d in days)),
        "terminal_soc_max_err": terminal_err,
        "gate": {
            "open_frequency": gate_open,
            "mean_probability": gate_prob,
            "closed_fraction": gate_closed,
            "quote_distribution": quote_dist,
            "mode_counts": {k: int(v) for k, v in sorted(
                modes.items(), key=lambda kv: int(kv[0]))},
            "distinct_modes": int(len(modes)),
            "permanent_open": sorted(
                k for k, v in gate_open.items() if v >= 1.0 - 1e-12),
            "permanent_closed": sorted(
                k for k, v in gate_open.items() if v <= 1e-12),
            "saturation_upper": int(sum(
                d["gate_saturation_upper"] for d in days)),
            "saturation_lower": int(sum(
                d["gate_saturation_lower"] for d in days)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--exposures", type=int, default=10)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --run-id")
    parser.add_argument("--pause-after-rounds", type=int, default=0,
                        help="run this many rounds, save a resume checkpoint, "
                             "and stop without the final test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    args = parser.parse_args(argv)

    run_id = args.run_id or f"EXPLORATORY_NEAR_B_C2_GATE_seed{args.seed}"
    run_dir = RUNS / run_id
    snap_dir = SNAPSHOTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_fh = log_path.open("a", encoding="utf-8")
    if not (run_dir / "eval_history.json").exists():
        (run_dir / "eval_history.json").write_text("[]", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    near_rows = read_csv(NEAR_PAIRING)
    full_rows = read_csv(FULL_PAIRING)
    near_by_pair = {f"{r['load_date']}|{r['price_date']}": r
                    for r in near_rows}
    full_by_pair = {f"{r['load_date']}|{r['price_date']}": r
                    for r in full_rows}
    split = json.loads(NEAR_SPLIT.read_text(encoding="utf-8"))
    load_to_pair = {r["load_date"]: f"{r['load_date']}|{r['price_date']}"
                    for r in near_rows}

    def pairs_for(load_dates: Sequence[str]) -> List[dict]:
        out = []
        for load in load_dates:
            pid = load_to_pair.get(load)
            if pid is None:
                raise KeyError(f"{load}: no NEAR pairing")
            out.append({"load_date": load,
                        "price_date": full_by_pair[pid]["price_date"],
                        "season": near_by_pair[pid].get("season", "")})
        return out

    train = pairs_for(split["train"])
    validation = pairs_for(split["validation"])
    test = pairs_for(split["test"])
    order = season_order(train, args.seed)
    budget = len(order) * args.exposures
    if args.max_episodes:
        budget = min(budget, args.max_episodes)

    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(encoding="utf-8"))
    mechanism_hash = b1_manifest["mechanism_hash"]
    b1_cache = mod.load_b1_cache()

    template_env = envmod.build_env_for_pair(
        train[0], differential_reward=False, terminal_soc_target=0.5)
    experiment = mod.ExperimentConfig(
        run_id, "B_C2", batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity)
    trainer = mod.MATD3BC2Gate(template_env, experiment)
    market_hash = mod.config_hash(template_env.config)
    hashes = data_hashes()

    resume_from = None
    if args.resume:
        if not (run_dir / "manifest.json").exists():
            raise FileNotFoundError(f"{run_dir}: no checkpoint manifest to resume")
        bundle = trainer.load(run_dir)
        completed = int(trainer.episodes)
        if completed % len(order):
            # A partial round is not in the last checkpoint. Restart at the
            # next full-round boundary; the discarded partial exposure is
            # reported, not silently counted.
            completed = ((completed // len(order)) + 1) * len(order)
        resume_from = {
            "checkpoint_step": bundle["manifest"]["step"],
            "episodes_in_checkpoint": int(trainer.episodes),
            "resume_episode": completed,
            "partial_round_discarded": bool(
                int(trainer.episodes) % len(order)),
        }
        for path, key in ((run_dir / "episodes.csv", "episode"),
                          (run_dir / "gate_stats.jsonl", "episode")):
            if not path.exists():
                continue
            if path.suffix == ".csv":
                rows = read_csv(path)
                keep = [r for r in rows if int(r[key]) <= completed]
                if rows:
                    with path.open("w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(keep)
            else:
                lines = path.read_text(encoding="utf-8").splitlines()
                keep = []
                for line in lines:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(row.get("episode", 0)) <= completed:
                        keep.append(line)
                path.write_text("\n".join(keep) + ("\n" if keep else ""),
                                encoding="utf-8")
        log(f"RESUME from checkpoint step {resume_from['checkpoint_step']} "
            f"at episode {completed}; partial-round="
            f"{resume_from['partial_round_discarded']}")
    if args.resume and (run_dir / "run_manifest.json").exists():
        provenance = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        provenance.update({
            "status": "resumed",
            "resumed_at": now(),
            "resume_from": resume_from,
            "code_revision": code_revision(),
            "source_sha256": {
                "rl_b_c2_gate.py": sha(ROOT / "src/rl_b_c2_gate.py"),
                "rl_env.py": sha(ROOT / "src/rl_env.py"),
                "dispatch_socp.py": sha(ROOT / "src/dispatch_socp.py"),
                "b1_daily_baseline.py": sha(ROOT / "src/b1_daily_baseline.py"),
                "c2_reward.py": sha(ROOT / "src/c2_reward.py"),
            },
        })
    else:
        provenance = {
            "run_id": run_id,
            "status": "exploratory_development",
            "seed": args.seed,
            "experiment": experiment.as_dict(),
            "dataset": "JN-Base-NEAR train 53",
            "split": str(NEAR_SPLIT),
            "train_days": [p["load_date"] for p in train],
            "validation_days": [p["load_date"] for p in validation],
            "test_days": [p["load_date"] for p in test],
            "exposures_per_day": args.exposures,
            "budget_episodes": budget,
            "data_hashes": hashes,
            "market_config_hash": market_hash,
            "b1_mechanism_hash": mechanism_hash,
            "code_revision": code_revision(),
            "source_sha256": {
                "rl_b_c2_gate.py": sha(ROOT / "src/rl_b_c2_gate.py"),
                "rl_env.py": sha(ROOT / "src/rl_env.py"),
                "dispatch_socp.py": sha(ROOT / "src/dispatch_socp.py"),
                "b1_daily_baseline.py": sha(ROOT / "src/b1_daily_baseline.py"),
                "c2_reward.py": sha(ROOT / "src/c2_reward.py"),
            },
            "created_at": now(),
            "h2_holdout_read": False,
            "claim": ("exploratory/development only; no joint source-load "
                      "holdout exists"),
        }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    log(f"run {run_id} seed={args.seed} budget={budget} episodes; "
        f"train={len(train)} val={len(validation)} test={len(test)}"
        + (f"; resume={resume_from}" if resume_from else ""))

    episode_rows = []
    evaluations = []
    if args.resume and (run_dir / "eval_history.json").exists():
        evaluations = json.loads(
            (run_dir / "eval_history.json").read_text(encoding="utf-8"))
    best = {"value": float("-inf"), "round": -1, "checkpoint": ""}
    if evaluations:
        existing = max(evaluations,
                       key=lambda r: r.get("mean_fleet_delta", -math.inf))
        snap = snap_dir / f"round_{int(existing['round']):03d}.json"
        checkpoint = ""
        if snap.exists():
            checkpoint = json.loads(
                snap.read_text(encoding="utf-8")).get("checkpoint", "")
        best = {"value": existing["mean_fleet_delta"],
                "round": existing["round"], "checkpoint": checkpoint}
    best_root = RUNS / f"{run_id}_best"
    started = time.perf_counter()
    try:
        episode = (resume_from["resume_episode"] if resume_from else 0)
        round_idx = episode // len(order)
        chunk_end = budget
        if args.pause_after_rounds > 0:
            chunk_end = min(
                budget, episode + args.pause_after_rounds * len(order))
        while episode < chunk_end:
            round_idx += 1
            round_start = episode
            while episode < chunk_end \
                    and episode - round_start < len(order):
                pair = order[episode - round_start]
                pair_id = f"{pair['load_date']}|{pair['price_date']}"
                row = full_by_pair[pair_id]
                env = envmod.build_env_for_pair(
                    pair, differential_reward=False,
                    terminal_soc_target=0.5)
                out = trainer.train_day(
                    env, pair, b1_cache,
                    config_hash_value=mechanism_hash,
                    data_hash=mod.pair_data_hash(row))
                episode += 1
                b = out["b"]
                row_out = {
                    "episode": episode,
                    "round": round_idx,
                    "pair_id": pair_id,
                    "delta_fleet": float(sum(b["delta"].values())),
                    "G_policy_fleet": float(sum(b["G_policy"].values())),
                    "G_b1_fleet": float(sum(b["G_b1"].values())),
                    "updates": trainer.updates,
                    "total_steps": trainer.total_steps,
                    "gate_distinct_modes": out["gate"]["distinct_modes"],
                    "gate_saturated_upper": out["gate"]["saturated_upper"],
                    "gate_saturated_lower": out["gate"]["saturated_lower"],
                }
                episode_rows.append(row_out)
                with (run_dir / "episodes.csv").open(
                        "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row_out))
                    if fh.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row_out)
                with (run_dir / "gate_stats.jsonl").open(
                        "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"episode": episode,
                                         "pair_id": pair_id,
                                         "gate": out["gate"]}) + "\n")
                log(f"ep {episode}/{budget} r{round_idx} {pair_id} "
                    f"delta={row_out['delta_fleet']:.1f} "
                    f"upd={trainer.updates}")
                if episode >= budget:
                    break

            if args.skip_eval:
                continue
            val = evaluate(trainer, validation, b1_cache,
                           mechanism_hash=mechanism_hash,
                           pair_rows=full_by_pair)
            val.update({"round": round_idx, "episode": episode,
                        "timestamp": now()})
            evaluations.append(val)
            (run_dir / "eval_history.json").write_text(
                json.dumps(evaluations, indent=2), encoding="utf-8")
            ckpt = trainer.save(
                run_dir, level="resume", dataset="NEAR", seed=args.seed,
                config_hash_value=market_hash, data=hashes,
                code_revision=provenance["code_revision"],
                exposure_log=str(run_dir / "exposure.json"),
                eval_provenance=str(run_dir / "eval_history.json"))
            snap = {
                "round": round_idx,
                "episode": episode,
                "checkpoint": str(ckpt),
                "validation": val,
                "counters": {
                    "total_steps": trainer.total_steps,
                    "updates": trainer.updates,
                    "episodes": trainer.episodes,
                    "buffer_len": len(trainer.buffer),
                },
                "timestamp": now(),
            }
            (snap_dir / f"round_{round_idx:03d}.json").write_text(
                json.dumps(snap, indent=2), encoding="utf-8")
            if val["mean_fleet_delta"] > best["value"]:
                best = {"value": val["mean_fleet_delta"],
                        "round": round_idx, "checkpoint": str(ckpt)}
                best_ckpt = trainer.save(
                    best_root, level="resume", dataset="NEAR",
                    seed=args.seed, config_hash_value=market_hash,
                    data=hashes,
                    code_revision=provenance["code_revision"],
                    exposure_log=str(best_root / "exposure.json"),
                    eval_provenance=str(run_dir / "eval_history.json"))
                best["checkpoint"] = str(best_ckpt)
                log(f"new best val mean fleet delta={best['value']:.1f} "
                    f"round={round_idx}")
            log(f"val r{round_idx}: mean fleet delta={val['mean_fleet_delta']:.1f} "
                f"positive={val['positive_days']}/{len(val['days'])} "
                f"v_cells={val['voltage_violation_cells']} "
                f"line_cells={val['line_overload_cells']}")

        if episode < budget:
            ckpt = trainer.save(
                run_dir, level="resume", dataset="NEAR", seed=args.seed,
                config_hash_value=market_hash, data=hashes,
                code_revision=provenance["code_revision"],
                exposure_log=str(run_dir / "exposure.json"),
                eval_provenance=str(run_dir / "eval_history.json"))
            provenance.update({
                "status": "paused",
                "paused_at": now(),
                "paused_episode": episode,
                "next_round": episode // len(order) + 1,
                "episodes_run": trainer.episodes,
                "total_steps": trainer.total_steps,
                "updates": trainer.updates,
                "buffer_len": len(trainer.buffer),
                "best": best,
                "paused_checkpoint": str(ckpt),
            })
            (run_dir / "run_manifest.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8")
            (snap_dir / "run_manifest.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8")
            log(f"paused at episode {episode}; next round "
                f"{episode // len(order) + 1}; checkpoint {ckpt}")
            return 0

        final = trainer.save(
            run_dir, level="bundle", dataset="NEAR", seed=args.seed,
            config_hash_value=market_hash, data=hashes,
            code_revision=provenance["code_revision"],
            exposure_log=str(run_dir / "exposure.json"),
            eval_provenance=str(run_dir / "eval_history.json"))
        final_counters = {
            "episodes": trainer.episodes,
            "total_steps": trainer.total_steps,
            "updates": trainer.updates,
            "buffer_len": len(trainer.buffer),
        }
        test_result = None
        if not args.skip_test:
            # The selected model is the best validation checkpoint, not the
            # last update. The test is a viewed legacy split and is reported
            # as development only.
            if best["round"] >= 0 and (best_root / "manifest.json").exists():
                trainer.load(best_root)
            test_result = evaluate(trainer, test, b1_cache,
                                   mechanism_hash=mechanism_hash,
                                   pair_rows=full_by_pair)
            test_result.update({
                "label": "LEGACY_SEEN_TEST_DEVELOPMENT_ONLY",
                "policy": "best_validation_checkpoint"
                if best["round"] >= 0 else "last_checkpoint",
                "timestamp": now()})
            (run_dir / "test_eval.json").write_text(
                json.dumps(test_result, indent=2), encoding="utf-8")
            log(f"test (development only, best validation policy): "
                f"mean fleet delta={test_result['mean_fleet_delta']:.1f}")
        provenance.update({
            "status": "completed",
            "finished_at": now(),
            "runtime_s": round(time.perf_counter() - started, 1),
            "episodes_run": final_counters["episodes"],
            "total_steps": final_counters["total_steps"],
            "updates": final_counters["updates"],
            "buffer_len": final_counters["buffer_len"],
            "best": best,
            "final_checkpoint": str(final),
            "evaluations": str(run_dir / "eval_history.json"),
            "test_evaluation": str(run_dir / "test_eval.json")
            if test_result is not None else None,
        })
        (run_dir / "run_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        (snap_dir / "run_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        log(f"completed: best round {best['round']} "
            f"val={best['value']:.1f}; final={final}")
        return 0
    except Exception:
        err = traceback.format_exc()
        log("FAILED:\n" + err)
        try:
            trainer.save(
                run_dir, level="resume", dataset="NEAR", seed=args.seed,
                config_hash_value=market_hash, data=hashes,
                code_revision=provenance["code_revision"],
                exposure_log=str(run_dir / "exposure.json"),
                eval_provenance=str(run_dir / "eval_history.json"))
        except Exception:
            log("could not save diagnostic checkpoint")
        provenance.update({
            "status": "failed", "error": err, "finished_at": now(),
            "episodes_run": trainer.episodes,
            "total_steps": trainer.total_steps,
            "updates": trainer.updates,
            "buffer_len": len(trainer.buffer),
        })
        (run_dir / "run_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        raise
    finally:
        log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
