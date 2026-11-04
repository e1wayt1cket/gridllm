"""Development training run for MATD3_DA_GATE_v2.

This is an **exploratory/development** run. The current data have no joint
source-load holdout (H1 = 0), so its test-set numbers cannot support a paper
claim. It trains on the selected NEAR training split only and never reads the
H2 holdout pairs.

`--split near` (default) preserves the original near/split.json reproduction.
`--split common` selects the common_split.json NEAR view used for the formal
run. In both cases the test and holiday blocks are recorded but never run.

Per round: one exposure to each selected training day, then deterministic
validation on the split's validation days. The selection metric is the mean
fleet delta G from actual settlement, never a training reward. A full resume
checkpoint is saved every round; a separate best checkpoint is saved whenever
validation improves. `--train-days` restricts training to a fixed subset for a
controlled smoke run; `--trace-blocks` writes one decision row per block/ESS.

Usage:
    PYTHONPATH=src python src/train_da_gate.py --split common \
        --seed 11 --exposures 10 --run-id FORMAL_NEAR_COMMON_seed11
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
import rl_da_gate as mod
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


SOURCE_FILES = (
    "rl_da_gate.py", "da_quote_map.py", "da_replay.py", "rl_b_c2_gate.py",
    "rl_spec.py", "rl_env.py", "dispatch_socp.py", "b1_daily_baseline.py",
    "c2_reward.py", "daily_reward_b.py", "v2_cy_jn_env.py",
    "train_da_gate.py",
)


def source_hashes() -> Dict[str, str]:
    return {name: sha(ROOT / "src" / name) for name in SOURCE_FILES}


def artifact_hashes(run_dir: Path) -> Dict[str, str]:
    out = {}
    for name in ("episodes.csv", "gate_stats.jsonl", "sampler_stats.jsonl",
                 "block_decisions.jsonl", "eval_history.json", "train.log"):
        path = run_dir / name
        if path.exists():
            out[name] = sha(path)
    return out


def no_improvement_streak(evaluations: Sequence[dict]) -> int:
    """Consecutive evaluations after the best validation value."""
    if not evaluations:
        return 0
    best_idx = max(range(len(evaluations)),
                   key=lambda i: evaluations[i].get("mean_fleet_delta",
                                                    -math.inf))
    return len(evaluations) - 1 - best_idx


def episode_budget(order_size: int, exposures: int,
                   max_episodes: int = 0) -> int:
    """Episodes in one full run: every training day exposed `exposures` times."""
    budget = order_size * exposures
    if max_episodes:
        budget = min(budget, max_episodes)
    return budget


def completed_rounds(episodes: int, order_size: int) -> int:
    """Rounds whose validation is complete; a partial round counts as next."""
    return (int(episodes) + order_size - 1) // order_size


def should_early_stop(round_idx: int, evaluations: Sequence[dict],
                      patience: int, min_rounds: int) -> bool:
    """Stop only after `min_rounds` completed rounds, then apply patience.

    `round_idx` is the number of rounds whose validation has been recorded, so
    the floor cannot be satisfied by a partial round.
    """
    if patience <= 0 or round_idx < max(int(min_rounds), 0):
        return False
    return no_improvement_streak(evaluations) >= patience


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


def eval_day(trainer: mod.DAGateTrainer, pair: dict, b1_cache, *,
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


def evaluate(trainer: mod.DAGateTrainer, pairs: Sequence[dict], b1_cache,
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
    parser.add_argument("--exposures", type=int, default=3)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--split", choices=("near", "common"), default="near",
                        help="near: near/split.json (legacy reproduction); "
                             "common: common_split.json near view (formal)")
    parser.add_argument("--train-days", default="",
                        help="comma-separated load dates; must be a subset of "
                             "the selected split's train days and is used in "
                             "the given order")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="stop after this many validation evaluations "
                             "without a new best; 0 disables")
    parser.add_argument("--min-rounds", type=int, default=0,
                        help="completed validation rounds required before "
                             "early stopping is allowed; default 0")
    parser.add_argument("--trace-blocks", action="store_true",
                        help="write block_decisions.jsonl (24 blocks x 4 ESS "
                             "per episode)")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest v3 checkpoint in --run-id")
    parser.add_argument("--pause-after-rounds", type=int, default=0,
                        help="run this many rounds, save a resume checkpoint, "
                             "and stop")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--uniform-fraction", type=float, default=0.5)
    parser.add_argument("--sampling-beta", type=float, default=0.5)
    parser.add_argument("--sampling-max-upweight", type=float, default=5.0)
    parser.add_argument("--exploration-start", type=float, default=0.20)
    parser.add_argument("--exploration-end", type=float, default=0.05)
    parser.add_argument("--exploration-decay-episodes", type=int, default=159)
    parser.add_argument("--gate-temperature-start", type=float, default=4.0)
    parser.add_argument("--gate-temperature-end", type=float, default=1.0)
    parser.add_argument("--gate-warmup-episodes", type=int, default=106)
    args = parser.parse_args(argv)

    run_id = args.run_id or f"DEV_DA_GATE_seed{args.seed}_v2"
    run_dir = RUNS / run_id
    snap_dir = SNAPSHOTS / run_id
    best_root = RUNS / f"{run_id}_best"
    if not args.resume:
        for path in (run_dir, snap_dir, best_root):
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(
                    f"{path} already exists and is not empty; use --resume or "
                    "a new --run-id")
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
                        "season": near_by_pair[pid].get("season", ""),
                        "pair_id": pid})
        return out

    def pairs_from_ids(pair_ids: Sequence[str]) -> List[dict]:
        out = []
        for pid in pair_ids:
            row = near_by_pair.get(pid)
            if row is None:
                raise KeyError(f"{pid}: no NEAR pairing")
            out.append({"load_date": row["load_date"],
                        "price_date": full_by_pair[pid]["price_date"],
                        "season": row.get("season", ""),
                        "pair_id": pid})
        return out

    if args.split == "common":
        split_path = COMMON_SPLIT
        split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        split_view = split_doc["near"]
        train = pairs_from_ids(split_view["train"])
        validation = pairs_from_ids(split_view["validation"])
        test = pairs_from_ids(split_view["test"])
        holiday = pairs_from_ids(split_view["holiday_block"])
    else:
        split_path = NEAR_SPLIT
        split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        split_view = split_doc
        train = pairs_for(split_view["train"])
        validation = pairs_for(split_view["validation"])
        test = pairs_for(split_view["test"])
        holiday = pairs_for(split_view.get("holiday_block", []))

    sets = {
        "train": {p["pair_id"] for p in train},
        "validation": {p["pair_id"] for p in validation},
        "test": {p["pair_id"] for p in test},
        "holiday": {p["pair_id"] for p in holiday},
    }
    for left, right in (("train", "validation"), ("train", "test"),
                        ("train", "holiday"), ("validation", "test"),
                        ("validation", "holiday"), ("test", "holiday")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise RuntimeError(
                f"split {args.split}: {left}/{right} overlap {sorted(overlap)}")

    train_full = train
    if args.train_days:
        requested = [s.strip() for s in args.train_days.split(",")
                     if s.strip()]
        if len(set(requested)) != len(requested):
            raise ValueError("--train-days contains duplicate dates")
        by_load = {p["load_date"]: p for p in train_full}
        missing = [d for d in requested if d not in by_load]
        if missing:
            raise ValueError(
                f"--train-days not in {args.split} train split: {missing}")
        train = [by_load[d] for d in requested]
        order = list(train)
    else:
        order = season_order(train, args.seed)
    budget = episode_budget(len(order), args.exposures, args.max_episodes)

    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(encoding="utf-8"))
    mechanism_hash = b1_manifest["mechanism_hash"]
    b1_cache = mod.load_b1_cache()

    template_env = envmod.build_env_for_pair(
        train[0], differential_reward=False, terminal_soc_target=0.5)
    experiment = mod.DAExperimentConfig(
        run_id, "B_C2", batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        uniform_fraction=args.uniform_fraction,
        sampling_beta=args.sampling_beta,
        sampling_max_upweight=args.sampling_max_upweight,
        exploration_epsilon_start=args.exploration_start,
        exploration_epsilon_end=args.exploration_end,
        exploration_decay_episodes=args.exploration_decay_episodes,
        gate_temperature_start=args.gate_temperature_start,
        gate_temperature_end=args.gate_temperature_end,
        gate_warmup_episodes=args.gate_warmup_episodes)
    trainer = mod.DAGateTrainer(template_env, experiment)
    market_hash = mod.config_hash(template_env.config)
    hashes = data_hashes()

    resume_from = None
    if args.resume:
        if not (run_dir / "manifest.json").exists():
            raise FileNotFoundError(f"{run_dir}: no checkpoint manifest to resume")
        if not (run_dir / "run_manifest.json").exists():
            raise FileNotFoundError(f"{run_dir}: no run manifest to resume")
        saved = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        checks = {
            "seed": saved.get("seed") == args.seed,
            "data_hashes": saved.get("data_hashes") == hashes,
            "market_config_hash":
                saved.get("market_config_hash") == market_hash,
            "b1_mechanism_hash":
                saved.get("b1_mechanism_hash") == mechanism_hash,
            "split": saved.get("split") == str(split_path),
            "exposures_per_day":
                saved.get("exposures_per_day") == args.exposures,
            "budget_episodes": saved.get("budget_episodes") == budget,
            "train_subset": saved.get("train_subset")
                == [p["pair_id"] for p in train],
            "action_schema":
                saved.get("action_schema") == "storage_quote_v3",
            "quote_map": saved.get("quote_map") == "linear_core_v1",
            "source_sha256": saved.get("source_sha256") == source_hashes(),
        }
        bad = [key for key, ok in checks.items() if not ok]
        if bad:
            raise RuntimeError(
                f"resume refused: run manifest mismatch in {bad}")
        bundle = trainer.load(run_dir)
        da_meta = (bundle.get("manifest") or {}).get("da_gate") or {}
        if da_meta.get("actor_loss_map") != "linear_core_v1":
            raise RuntimeError(
                "resume refused: checkpoint predates the actor-loss v3 map "
                "fix and cannot be continued under the corrected trainer")
        # A partial round is not in the last checkpoint. Restart at the next
        # full-round boundary; the discarded partial exposure is reported, not
        # silently counted.
        completed = completed_rounds(int(trainer.episodes), len(order)) * len(order)
        resume_from = {
            "checkpoint_step": bundle["manifest"]["step"],
            "episodes_in_checkpoint": int(trainer.episodes),
            "resume_episode": completed,
            "partial_round_discarded": bool(
                int(trainer.episodes) % len(order)),
        }
        for path, key in ((run_dir / "episodes.csv", "episode"),
                          (run_dir / "gate_stats.jsonl", "episode"),
                          (run_dir / "sampler_stats.jsonl", "episode"),
                          (run_dir / "block_decisions.jsonl", "episode")):
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
            "source_sha256": source_hashes(),
        })
    else:
        provenance = {
            "run_id": run_id,
            "status": "exploratory_development",
            "seed": args.seed,
            "experiment": experiment.as_dict(),
            "dataset": "JN-Base-NEAR train 53",
            "split": str(split_path),
            "split_name": args.split,
            "split_view": "near" if args.split == "common" else "root",
            "train_days": [p["load_date"] for p in train_full],
            "train_pairs": [p["pair_id"] for p in train_full],
            "train_subset": [p["pair_id"] for p in train],
            "validation_days": [p["load_date"] for p in validation],
            "validation_pairs": [p["pair_id"] for p in validation],
            "test_days": [p["load_date"] for p in test],
            "test_pairs": [p["pair_id"] for p in test],
            "holiday_pairs": [p["pair_id"] for p in holiday],
            "legacy_seen_test_pairs": sorted(
                sets["test"] | sets["holiday"]),
            "exposures_per_day": args.exposures,
            "budget_episodes": budget,
            "early_stop_patience": int(args.early_stop_patience),
            "min_rounds": int(args.min_rounds),
            "data_hashes": hashes,
            "market_config_hash": market_hash,
            "b1_mechanism_hash": mechanism_hash,
            "action_schema": "storage_quote_v3",
            "quote_map": "linear_core_v1",
            "sampler": {
                "uniform_fraction": args.uniform_fraction,
                "beta": args.sampling_beta,
                "max_upweight": args.sampling_max_upweight,
                "cells": 432,
            },
            "exploration": {
                "epsilon_start": args.exploration_start,
                "epsilon_end": args.exploration_end,
                "decay_episodes": args.exploration_decay_episodes,
                "charge_scale": experiment.exploration_charge_scale,
                "high_fraction": experiment.exploration_high_fraction,
                "bounded": [0.0, 1.0],
            },
            "gate_schedule": {
                "start": args.gate_temperature_start,
                "end": args.gate_temperature_end,
                "warmup_episodes": args.gate_warmup_episodes,
                "evaluation_temperature": 1.0,
            },
            "test_evaluation": None,
            "legacy_seen_test_run": False,
            "da_config": experiment.as_dict(),
            "code_revision": code_revision(),
            "source_sha256": source_hashes(),
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
    started = time.perf_counter()
    early_stop = {
        "patience": int(args.early_stop_patience),
        "min_rounds": int(args.min_rounds),
        "triggered": False,
        "round": None,
        "best_round": best["round"],
        "best_value": best["value"],
    }
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
                    data_hash=mod.pair_data_hash(row),
                    trace=args.trace_blocks)
                episode += 1
                b = out["b"]
                explored = np.asarray(
                    trainer.buffer.extras["explored"])[-24:]
                sampler = out.get("sampler", {})
                row_out = {
                    "episode": episode,
                    "round": round_idx,
                    "pair_id": pair_id,
                    "delta_fleet": float(sum(b["delta"].values())),
                    "G_policy_fleet": float(sum(b["G_policy"].values())),
                    "G_b1_fleet": float(sum(b["G_b1"].values())),
                    "updates": trainer.updates,
                    "total_steps": trainer.total_steps,
                    "gate_temperature": out["gate_temperature"],
                    "exploration_epsilon": out["exploration_epsilon"],
                    "explored_legs": int(explored.sum()),
                    "explored_discharge": int(explored[:, :, 1].sum()),
                    "sampler_cells": sampler.get("cells_nonempty", ""),
                    "sampler_ess_fraction": sampler.get("ess_fraction", ""),
                    "sampler_prob_max": sampler.get("sample_prob_max", ""),
                    "gate_distinct_modes": out["gate"]["distinct_modes"],
                    "gate_saturated_upper": out["gate"]["saturated_upper"],
                    "gate_saturated_lower": out["gate"]["saturated_lower"],
                }
                for nm in trainer.agent_names:
                    row_out[f"delta_{nm}"] = float(b["delta"][nm])
                    row_out[f"G_policy_{nm}"] = float(
                        b["G_policy"][nm])
                    row_out[f"G_b1_{nm}"] = float(b["G_b1"][nm])
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
                                         "gate": out["gate"],
                                         "gate_temperature":
                                             out["gate_temperature"],
                                         "exploration_epsilon":
                                             out["exploration_epsilon"]})
                             + "\n")
                with (run_dir / "sampler_stats.jsonl").open(
                        "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"episode": episode,
                                         "pair_id": pair_id,
                                         "sampler": out.get("sampler", {})})
                             + "\n")
                if args.trace_blocks:
                    with (run_dir / "block_decisions.jsonl").open(
                            "a", encoding="utf-8") as fh:
                        for decision in out.get("block_decisions", []):
                            fh.write(json.dumps({
                                "episode": episode,
                                "round": round_idx,
                                **decision}) + "\n")
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
            patience = int(args.early_stop_patience)
            if should_early_stop(round_idx, evaluations, patience,
                                 int(args.min_rounds)):
                early_stop.update({
                    "triggered": True,
                    "round": round_idx,
                    "best_round": best["round"],
                    "best_value": best["value"],
                })
                log(f"early stop at round {round_idx}: {patience} "
                    "evaluations without a new best")
                break

        if episode < budget and not early_stop["triggered"]:
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
                "early_stop": early_stop,
                "paused_checkpoint": str(ckpt),
            })
            log_fh.flush()
            provenance["artifacts_sha256"] = artifact_hashes(run_dir)
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
        # The legacy test split is deliberately not evaluated in this phase.
        test_result = None
        log("legacy test not evaluated; H2 unread")
        provenance.update({
            "status": ("completed_early_stop" if early_stop["triggered"]
                       else "completed"),
            "finished_at": now(),
            "runtime_s": round(time.perf_counter() - started, 1),
            "episodes_run": final_counters["episodes"],
            "total_steps": final_counters["total_steps"],
            "updates": final_counters["updates"],
            "buffer_len": final_counters["buffer_len"],
            "best": best,
            "early_stop": early_stop,
            "final_checkpoint": str(final),
            "evaluations": str(run_dir / "eval_history.json"),
            "test_evaluation": None,
            "legacy_seen_test_run": False,
            "h2_holdout_read": False,
        })
        log_fh.flush()
        provenance["artifacts_sha256"] = artifact_hashes(run_dir)
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
            "early_stop": early_stop,
        })
        log_fh.flush()
        provenance["artifacts_sha256"] = artifact_hashes(run_dir)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        raise
    finally:
        log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
