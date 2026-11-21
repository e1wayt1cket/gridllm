"""Quote-map saturation audit and offline candidate comparison.

Read-only with respect to every original artifact. Nothing here trains, steps
an optimizer, remembers a transition, saves a checkpoint, or writes outside
`reports/algorithm_diagnostics/quote_mapping_saturation_2026-11-12/`.

Phases (each writes its own files, so a kill keeps the finished ones):

  0  evidence freeze: input hashes, route checks, protocol identity
  1  read-only observations: history replay, buffer forward, rollout probes
  2  real training-distribution statistics, cluster-bootstrapped by day pair
  3  candidate maps on identical logits, plus the math/numerical test battery
  4  fixed-sample actor-gradient probe against the frozen critic
  5  admission decision and report

The single registered map stays `linear_core_v1`; `quote_map_candidates` is
imported only by this script.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import da_quote_map as qmap
import quote_map_candidates as cands
import rl_b_c2_gate as base
import rl_da_gate as mod
import da_replay as replay_mod
import diagnose_actor_update as diag
import diagnose_c0_mapping_v4 as d3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics" / "quote_map_saturation_2026-11-12"
MANIFEST_DIR = ROOT / "reports" / "v2_cy_jn_2025" / "formal_training_v4"

SEEDS = (11, 23)
RUN = {s: ROOT / "runs" / f"FORMAL_7834d0aa_seed{s}" for s in SEEDS}
BEST = {s: ROOT / "runs" / f"FORMAL_7834d0aa_seed{s}_best" for s in SEEDS}
FINAL_STEP = 12720
BEST_STEP = 1272

NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")
HIGHS = np.asarray(qmap.LEG_HIGHS, dtype=float)
MAP_VERSION = "linear_core_v1"
ACTION_SCHEMA = "storage_quote_v4"
PROBE_PAIRS = (
    "2022-07-23|2025-07-13",
    "2022-08-14|2025-08-17",
    "2022-08-17|2025-08-19",
    "2022-10-13|2025-10-13",
    "2022-10-15|2025-10-12",
    "2022-10-17|2025-10-17",
    "2022-12-09|2025-12-09",
    "2022-12-10|2025-12-07",
)
BOOTSTRAP_RESAMPLES = 1000
GRAD_BATCH = 128
GRAD_SEED = 20261112
SOURCES = (0, 1, 2, 3)
SHORTFALL_COVERAGE = 0.90
SHORTFALL_ESS_LEG = 0.80
EDGE_FRACTION = 0.05
PREREG_MIN_LEG_SAMPLE = 100


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
def write_json(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(diag.jsonable(payload), indent=2),
                    encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return diag.sha256_file(Path(path))


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"revision": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD")}


def quantiles(values, qs=(10, 50, 90, 95)) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def edge_band(leg: int) -> tuple:
    return EDGE_FRACTION * HIGHS[leg], (1.0 - EDGE_FRACTION) * HIGHS[leg]


def edge_fraction(quote_values, leg: int) -> float:
    arr = np.asarray(quote_values, dtype=float)
    if arr.size == 0:
        return float("nan")
    lo, hi = edge_band(leg)
    return float(np.mean((arr <= lo) | (arr >= hi)))


def cluster_bootstrap(values, clusters, stat, resamples=BOOTSTRAP_RESAMPLES,
                      seed=0) -> dict:
    """Percentile CI over resampled day pairs, never over blocks of one day."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.size == 0:
        return {"lo": None, "hi": None}
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(resamples):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        if idx.size:
            stats.append(stat(values[idx]))
    if not stats:
        return {"lo": None, "hi": None}
    return {"lo": float(np.percentile(stats, 2.5)),
            "hi": float(np.percentile(stats, 97.5))}


# ----------------------------------------------------------------------
# Phase 0 - evidence freeze
# ----------------------------------------------------------------------
def evidence_paths() -> dict:
    paths = {}
    for name in ("da_quote_map.py", "rl_da_gate.py", "rl_b_c2_gate.py",
                 "train_da_gate.py", "rl_spec.py", "da_replay.py",
                 "diagnose_c0_mapping_v4.py", "quote_map_candidates.py",
                 "diagnose_quote_map_saturation.py"):
        paths[f"src/{name}"] = ROOT / "src" / name
    for name in ("test_da_quote_mapping.py", "test_v4_mechanics.py"):
        paths[f"tests/{name}"] = ROOT / "tests" / name
    for name in ("c0_d3_mapping.json", "c0_d3_mapping.md",
                 "P1_v4_evidence.json", "thresholds.json"):
        paths[f"reports/algorithm_diagnostics/{name}"] = (
            ROOT / "reports" / "algorithm_diagnostics" / name)
    for name in ("formal_training_manifest.json",
                 "formal_training_code_manifest.json",
                 "formal_training_config.yaml",
                 "formal_training_data_manifest.json",
                 "phase25_v4_freeze.md"):
        paths[f"formal_training_v4/{name}"] = MANIFEST_DIR / name
    for seed in SEEDS:
        for tag, root in (("", RUN[seed]), ("_best", BEST[seed])):
            key = f"runs/FORMAL_7834d0aa_seed{seed}{tag}"
            paths[f"{key}/manifest.json"] = root / "manifest.json"
            paths[f"{key}/exposure.json"] = root / "exposure.json"
            if root.joinpath("run_manifest.json").exists():
                paths[f"{key}/run_manifest.json"] = root / "run_manifest.json"
        paths[f"runs/FORMAL_7834d0aa_seed{seed}/block_decisions.jsonl"] = (
            RUN[seed] / "block_decisions.jsonl")
        paths[f"runs/FORMAL_7834d0aa_seed{seed}/eval_history.json"] = (
            RUN[seed] / "eval_history.json")
        steps = sorted(int(p.name.split("_")[1])
                       for p in (RUN[seed] / "checkpoints").iterdir())
        for step in steps:
            ck = RUN[seed] / "checkpoints" / f"step_{step:08d}"
            paths[f"runs/FORMAL_7834d0aa_seed{seed}/checkpoints/"
                  f"step_{step:08d}/buffer/replay.npz"] = ck / "buffer/replay.npz"
        best_ck = BEST[seed] / "checkpoints" / f"step_{BEST_STEP:08d}"
        paths[f"runs/FORMAL_7834d0aa_seed{seed}_best/checkpoints/"
              f"step_{BEST_STEP:08d}/buffer/replay.npz"] = (
            best_ck / "buffer/replay.npz")
        for nm in NAMES:
            paths[f"{rel(RUN[seed])}/checkpoints/step_{FINAL_STEP:08d}/"
                  f"actors/{nm}.pt"] = (
                RUN[seed] / "checkpoints" / f"step_{FINAL_STEP:08d}" /
                "actors" / f"{nm}.pt")
            paths[f"{rel(BEST[seed])}/checkpoints/step_{BEST_STEP:08d}/"
                  f"actors/{nm}.pt"] = (best_ck / "actors" / f"{nm}.pt")
    return paths


def preregistration() -> dict:
    """Fixed before the first measurement in this directory.

    Coverage differs enough between the three readings to change which
    candidate is admitted, so the primary is named here and every table in the
    report is labelled with the reading it uses.
    """
    return {
        "written_before_measurement": True,
        "primary_coverage_reading": (
            "checkpoint actor forward on the stored replay observations of the "
            "same round, legs whose recorded Bernoulli gate was open"),
        "secondary_reading": (
            "recorded block_decisions price_logits (post-noise training "
            "distribution), gate open and exploration_source == 0"),
        "deployment_reading": (
            "seeded eval-mode rollout on the 8 pre-registered train pairs, "
            "deterministic gate"),
        "coverage_definition": "share of leg-decisions with |z| < Z_CORE",
        "minimum_leg_sample": 100,
        "criteria": {
            "coverage_overall": SHORTFALL_COVERAGE,
            "coverage_each_ess_leg": SHORTFALL_ESS_LEG,
            "abs_dq_dz_median_vs_v1": 10.0,
            "abs_dq_dz_p10_vs_v1": 3.0,
            "edge_fraction_drop_pp": 20.0,
        },
        "per_seed_independent": True,
        "selection_rule": "smallest Z passing every criterion on both seeds",
    }


def phase0() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = evidence_paths()
    hashes, missing = {}, []
    for key, path in paths.items():
        if path.exists():
            hashes[key] = sha256_file(path)
        else:
            missing.append(key)

    manifest = read_json(MANIFEST_DIR / "formal_training_manifest.json")
    code_manifest = read_json(
        MANIFEST_DIR / "formal_training_code_manifest.json")
    pins = code_manifest.get("sources", code_manifest.get("files", {}))
    pin_rows = {}
    for name, expected in pins.items():
        path = ROOT / name
        actual = sha256_file(path) if path.exists() else None
        pin_rows[name] = {
            "expected": expected if isinstance(expected, str) else
            expected.get("sha256"),
            "actual": actual,
            "match": actual is not None
            and actual == (expected if isinstance(expected, str)
                           else expected.get("sha256")),
        }

    routes = diag.route_source_checks()
    run_manifests = {s: read_json(RUN[s] / "run_manifest.json") for s in SEEDS}
    run_pins = {}
    for seed, rm in run_manifests.items():
        for name, expected in rm.get("source_sha256", {}).items():
            path = ROOT / "src" / name
            actual = sha256_file(path) if path.exists() else None
            run_pins[f"seed{seed}:{name}"] = {
                "expected": expected, "actual": actual,
                "match": actual == expected}
    split_checks = {}
    for seed, rm in run_manifests.items():
        train = set(rm["train_pairs"])
        probe = set(PROBE_PAIRS)
        others = set(rm["validation_pairs"]) | set(rm["test_pairs"]) \
            | set(rm["holiday_pairs"])
        split_checks[str(seed)] = {
            "probe_pairs_subset_of_train": probe <= train,
            "probe_pairs_disjoint_from_holdouts": not (probe & others),
            "probe_pairs_present": len(probe & train),
        }

    identity = {
        "map_version": MAP_VERSION,
        "action_schema": ACTION_SCHEMA,
        "protocol_version": manifest.get("protocol_version"),
        "config_hash": manifest.get("config_hash"),
        "quote_map_in_manifest": json.loads(
            (RUN[11] / "manifest.json").read_text(encoding="utf-8")
        )["specs"]["action_spec"].get("quote_map"),
        "base_b_c2_uses_v2_sigmoid_map": True,
        "base_b_c2_audited_here": False,
        "phase19_20_routing_issues": "already fixed; not re-reported as current",
    }
    payload = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "git": git_state(),
        "environment": diag.env_versions(),
        "inputs": hashes,
        "missing_inputs": missing,
        "code_manifest_pins": pin_rows,
        "code_manifest_all_match": all(r["match"] for r in pin_rows.values()),
        "routes": routes,
        "routes_all_pass": all(routes.values()),
        "run_source_pins": run_pins,
        "run_source_pins_all_match": all(
            r["match"] for r in run_pins.values()),
        "probe_pair_split": split_checks,
        "identity": identity,
        "preregistration": preregistration(),
    }
    write_json("input_hashes.json", payload)
    print(f"phase0: {len(hashes)} inputs hashed, "
          f"{len(missing)} missing, pins match="
          f"{payload['code_manifest_all_match']}, "
          f"routes={payload['routes_all_pass']}", flush=True)
    return payload


# ----------------------------------------------------------------------
# Phase 1 - read-only observations
# ----------------------------------------------------------------------
def copy_checkpoint(source: Path, dest: Path) -> Path:
    """Immutable working copy: real copies, never hardlinks.

    Only the networks and counters the probe needs are copied, so the audit
    directory stays small while every byte read by the probe is a copy whose
    hash is checked against the original.
    """
    import shutil
    for group in ("actors", "critics", "buffer"):
        (dest / group).mkdir(parents=True, exist_ok=True)
        for path in sorted((source / group).iterdir()):
            if path.is_file():
                shutil.copy2(path, dest / group / path.name)
    shutil.copy2(source / "counters.json", dest / "counters.json")
    return dest


def load_actors(ckpt: Path) -> dict:
    """Rebuild the architecture from the frozen classes and load the weights.

    `load_state_dict` is strict, so a shape or key drift between the saved
    checkpoint and the current classes fails here rather than silently.
    """
    actors = {}
    for nm in NAMES:
        module = base.GateActor(12)
        state = torch.load(ckpt / "actors" / f"{nm}.pt", map_location="cpu",
                           weights_only=False)
        module.load_state_dict(state)
        module.eval()
        actors[nm] = module
    return actors


def load_critics(ckpt: Path) -> dict:
    critics = {}
    for nm in NAMES:
        module = base.CentralizedCritic(len(NAMES), 12, 2)
        state = torch.load(ckpt / "critics" / f"{nm}.pt", map_location="cpu",
                           weights_only=False)
        module.load_state_dict(state)
        module.eval()
        critics[nm] = module
    return critics


def raw_logits(actor, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    with torch.no_grad():
        out = actor(torch.as_tensor(obs))
    return out.numpy()


def verify_history_rows(seed: int) -> dict:
    """Row-level replay of the mapping layer on the recorded trace."""
    remap_err, closed_violations, open_mismatch = 0.0, 0, 0
    open_rows = 0
    for row in iter_block_decisions(seed):
        z = np.asarray(row["price_logits"], dtype=np.float32)
        quotes = np.asarray(row["quotes"], dtype=np.float32)
        gates = np.asarray(row["gates"], dtype=np.float32)
        source = np.asarray(row["exploration_source"], dtype=np.float32)
        high = HIGHS.astype(np.float32)
        mapped = high * qmap.map_fraction(torch.as_tensor(z)).numpy()
        err = float(np.max(np.abs(mapped - np.asarray(
            row["price_quotes"], dtype=np.float32)))) \
            if "price_quotes" in row else None
        if err is not None:
            remap_err = max(remap_err, err)
        for leg in range(2):
            if gates[leg] == 0.0 and quotes[leg] != 0.0:
                closed_violations += 1
            if gates[leg] > 0.0 and source[leg] == 0.0:
                open_rows += 1
                if abs(float(quotes[leg]) - float(mapped[leg])) > 1e-5:
                    open_mismatch += 1
    return {"seed": seed, "rows": open_rows,
            "closed_gate_nonzero": closed_violations,
            "open_no_override_mismatch": open_mismatch,
            "recorded_price_quotes_field": "absent in block_decisions.jsonl",
            "layer1_raw_logits": "NOT_TESTED for historical rows; "
                                 "recovered from checkpoints instead"}


def iter_block_decisions(seed: int):
    path = RUN[seed] / "block_decisions.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            validate_decision_row(row, seed, line_no)
            yield row


def validate_decision_row(row: dict, seed: int, line_no: int) -> None:
    for key in ("episode", "round", "pair_id", "block", "agent",
                "price_logits", "gates", "quotes", "exploration_source"):
        if key not in row:
            raise ValueError(f"seed{seed} line {line_no}: missing {key}")
    if row["agent"] not in NAMES:
        raise ValueError(f"seed{seed} line {line_no}: agent {row['agent']}")
    for key in ("price_logits", "gates", "quotes", "exploration_source"):
        arr = np.asarray(row[key], dtype=float)
        if arr.shape != (2,):
            raise ValueError(f"seed{seed} line {line_no}: {key} shape")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"seed{seed} line {line_no}: {key} not finite")
    for g in row["gates"]:
        if float(g) not in (0.0, 1.0):
            raise ValueError(f"seed{seed} line {line_no}: gate {g}")
    for s in row["exploration_source"]:
        if float(s) not in SOURCES:
            raise ValueError(f"seed{seed} line {line_no}: source {s}")


def buffer_forward(seed: int) -> tuple[dict, dict]:
    """Raw actor logits for every stored observation, per round checkpoint.

    Also re-derives the stored `price_quotes` and the executed action from the
    stored price logits, so the mapping layer is verified on the buffer the
    same way it is verified on the trace.
    """
    rows = {}
    checks = {"remap_max_abs_err": 0.0, "executed_mismatch": 0,
              "closed_gate_nonzero": 0, "rows": 0}
    for step_dir in sorted((RUN[seed] / "checkpoints").iterdir()):
        round_no = int(step_dir.name.split("_")[1]) // 1272
        actors = load_actors(step_dir)
        data = np.load(step_dir / "buffer" / "replay.npz")
        obs, gates = data["obs"], data["gates"]
        stored_logits = data["price_logits"].astype(np.float32)
        stored_quotes = data["price_quotes"].astype(np.float32)
        executed = data["actions"].astype(np.float32)
        explored = data["explored"].astype(np.float32)
        mapped = HIGHS.astype(np.float32) * qmap.map_fraction(
            torch.as_tensor(stored_logits)).numpy()
        checks["remap_max_abs_err"] = max(
            checks["remap_max_abs_err"],
            float(np.max(np.abs(mapped - stored_quotes))))
        checks["executed_mismatch"] += int(np.sum(
            (np.abs(executed - gates * stored_quotes) > 1e-5) & (explored == 0)))
        checks["closed_gate_nonzero"] += int(np.sum(
            (gates == 0) & (executed != 0.0)))
        checks["rows"] += int(obs.shape[0])
        for i, nm in enumerate(NAMES):
            rows[f"{seed}|{round_no}|{nm}"] = raw_logits(
                actors[nm], obs[:, i, :]).astype(np.float32)
        rows[f"{seed}|{round_no}|_gates"] = gates.astype(np.float32)
    return rows, checks


def save_buffer_raw(seed: int, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"buffer_raw_logits_seed{seed}.npz"
    np.savez_compressed(path, **payload)
    return path


def snapshot_rng() -> dict:
    import random
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def restore_rng(state: dict) -> None:
    import random
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def route_probe(trainer) -> dict:
    """Which map each path actually reaches, measured rather than read.

    Patching only `rl_da_gate.forward_policy_v3` would miss the target action:
    `_target_quotes` calls `target_quotes_v3`, which resolves
    `forward_policy_v3` in `da_quote_map` itself. Both are patched here and
    both are restored in `finally`. No optimizer step is involved.
    """
    counts = {"da_v3": 0, "map_v3": 0, "v2_sigmoid": 0}
    original = (mod.forward_policy_v3, qmap.forward_policy_v3,
                base.forward_policy)

    def wrap(key, fn):
        def inner(*args, **kwargs):
            counts[key] += 1
            return fn(*args, **kwargs)
        return inner

    mod.forward_policy_v3 = wrap("da_v3", original[0])
    qmap.forward_policy_v3 = wrap("map_v3", original[1])
    base.forward_policy = wrap("v2_sigmoid", original[2])
    actor = trainer.actors[NAMES[0]]
    obs = np.zeros(12, dtype=np.float32)
    state = snapshot_rng()
    out = {}
    try:
        for name, call in (
                ("execution", lambda: trainer._policy_action(actor, obs,
                                                             train=False)),
                ("actor_loss", lambda: trainer._actor_loss_quotes(
                    actor(torch.zeros(2, 12)))),
                ("target", lambda: trainer._target_quotes(
                    actor, torch.zeros(2, 12)))):
            for key in counts:
                counts[key] = 0
            call()
            out[name] = dict(counts)
    finally:
        (mod.forward_policy_v3, qmap.forward_policy_v3,
         base.forward_policy) = original
        restore_rng(state)
    out["v2_sigmoid_used_in_da_paths"] = any(
        v["v2_sigmoid"] for v in out.values())
    return out


def build_probe_trainer():
    """One trainer object for the counterfactual probe; no step is ever run."""
    pairs = diag.train_pairs()["pairs"]
    import train_da_gate as T
    import v2_cy_jn_env as envmod
    first = pairs[PROBE_PAIRS[0]]
    env = envmod.build_env_for_pair(first, differential_reward=False,
                                    terminal_soc_target=0.5)
    trainer = mod.DAGateTrainer(env, mod.DAExperimentConfig("SATURATION_AUDIT"))
    return trainer


def rollout_probe(trainer, seed: int, kind: str, ckpt: Path,
                  pairs: dict, trace_path: Path) -> dict:
    import v2_cy_jn_env as envmod
    episode = int(read_json(ckpt / "counters.json")["episode"])
    actors = load_actors(ckpt)
    for nm in NAMES:
        trainer.actors[nm].load_state_dict(actors[nm].state_dict())
        trainer.actors[nm].eval()
    trainer._current_episode = episode
    tau = trainer.current_gate_temperature()
    eps = trainer.experiment.exploration_epsilon_at(episode)

    rows, raw_all = [], []
    for pid in PROBE_PAIRS:
        pair = pairs[pid]
        env = envmod.build_env_for_pair(pair, differential_reward=False,
                                        terminal_soc_target=0.5)
        obs = env.reset()
        for block in range(24):
            acts = {}
            for nm in NAMES:
                actor = trainer.actors[nm]
                raw = raw_logits(actor, obs[nm])[0]
                policy_out = torch.as_tensor(raw, dtype=torch.float32)
                with torch.no_grad():
                    out = qmap.forward_policy_v3(policy_out, train=False)
                quotes = out["quotes"].numpy()
                gates = out["gates"].numpy()
                acts[nm] = quotes.astype(float)
                for leg in range(2):
                    if gates[leg] > 0:
                        raw_all.append(float(raw[2 + leg]))
                # Counterfactual mechanism probe on the same observation:
                # the noise, exploration and calibration layers of the
                # training path, never rolled into the environment.
                for calibrating in (False, True):
                    # `_policy_action(train=True)` consumes torch and numpy
                    # RNG (price noise, Bernoulli gate, exploration); the
                    # probe restores both so it cannot perturb a later phase.
                    state = snapshot_rng()
                    np.random.seed(GRAD_SEED + block)
                    torch.manual_seed(GRAD_SEED + block)
                    trainer._calibrating = calibrating
                    d = trainer._policy_action(actor, obs[nm], train=True)
                    restore_rng(state)
                    noisy = np.asarray(d["price_logits"], dtype=float)
                    rows.append({
                        "seed": seed, "checkpoint": kind, "episode": episode,
                        "pair_id": pid, "block": block, "agent": nm,
                        "gate_temperature": tau, "epsilon": eps,
                        "calibrating": int(calibrating),
                        "raw_charge": float(raw[2]),
                        "raw_discharge": float(raw[3]),
                        "noise_charge": float(noisy[0] - raw[2]),
                        "noise_discharge": float(noisy[1] - raw[3]),
                        "gate_charge": float(d["gates"][0]),
                        "gate_discharge": float(d["gates"][1]),
                        "mapped_charge": float(d["price_quotes"][0]),
                        "mapped_discharge": float(d["price_quotes"][1]),
                        "executed_charge": float(d["quotes"][0]),
                        "executed_discharge": float(d["quotes"][1]),
                        "source_charge": float(d["exploration_source"][0]),
                        "source_discharge": float(d["exploration_source"][1]),
                    })
                trainer._calibrating = False
            obs, _, done, _ = env.step(acts)
            if env._last_result is None:
                raise RuntimeError(f"{pid}: clearing failed at block {block}")
            if done:
                break
    with trace_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if trace_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
    return {"seed": seed, "checkpoint": kind, "episode": episode,
            "gate_temperature": tau, "epsilon": eps,
            "rows": len(rows), "raw_logits": len(raw_all)}


def phase1() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"history": [verify_history_rows(s) for s in SEEDS]}

    buffer_rows = {}
    for seed in SEEDS:
        payload, checks = buffer_forward(seed)
        path = save_buffer_raw(seed, payload)
        checks["path"] = rel(path)
        checks["sha256"] = sha256_file(path)
        buffer_rows[str(seed)] = checks
    summary["buffer"] = buffer_rows

    trainer = build_probe_trainer()
    summary["routes_measured"] = route_probe(trainer)
    pairs = diag.train_pairs()["pairs"]
    missing = [p for p in PROBE_PAIRS if p not in pairs]
    if missing:
        raise RuntimeError(f"probe pairs not in the pairing table: {missing}")
    trace_path = OUT / "probe_trace.csv"
    trace_path.write_text("", encoding="utf-8")
    rollouts = []
    for seed in SEEDS:
        for kind, root, step in (("final", RUN[seed], FINAL_STEP),
                                 ("best", BEST[seed], BEST_STEP)):
            ckpt = root / "checkpoints" / f"step_{step:08d}"
            rollouts.append(rollout_probe(trainer, seed, kind, ckpt, pairs,
                                          trace_path))
    summary["rollout"] = rollouts
    summary["probe_trace_sha256"] = sha256_file(trace_path)
    write_json("phase1_observations.json", summary)
    print(json.dumps(summary["history"]), flush=True)
    print(json.dumps(rollouts), flush=True)
    return summary


# ----------------------------------------------------------------------
# Phase 2 - real training distribution
# ----------------------------------------------------------------------
def leg_stats(z: np.ndarray, quotes: np.ndarray, gates: np.ndarray,
              leg: int) -> dict:
    z = np.asarray(z, dtype=float)
    deriv = qmap.map_derivative_np(z) / qmap.SLOPE
    effective = gates * HIGHS[leg] * qmap.map_derivative_np(z)
    q_open = np.asarray(quotes, dtype=float)[gates > 0.5]
    return {
        "n": int(z.size),
        "frac_beyond_core": float(np.mean(np.abs(z) > qmap.Z_CORE))
        if z.size else None,
        "abs_logit_p50": quantiles(np.abs(z), (50,))["p50"],
        "abs_logit_p90": quantiles(np.abs(z), (90,))["p90"],
        "abs_logit_p95": quantiles(np.abs(z), (95,))["p95"],
        "grad_retention_p10": quantiles(deriv, (10,))["p10"],
        "grad_retention_p50": quantiles(deriv, (50,))["p50"],
        "grad_retention_p90": quantiles(deriv, (90,))["p90"],
        "effective_dq_dz_p10": quantiles(effective, (10,))["p10"],
        "effective_dq_dz_p50": quantiles(effective, (50,))["p50"],
        "effective_dq_dz_p90": quantiles(effective, (90,))["p90"],
        "edge_quote_fraction": edge_fraction(q_open, leg)
        if q_open.size else None,
        "gate_open_fraction": float(np.mean(gates > 0.5)) if gates.size else None,
    }


def collect_history(seed: int) -> list:
    return list(iter_block_decisions(seed))


def group_history(rows: list, seed: int) -> list:
    """One record per (round, agent, leg, gate, source)."""
    groups = {}
    for row in rows:
        z = np.asarray(row["price_logits"], dtype=float)
        q = np.asarray(row["quotes"], dtype=float)
        g = np.asarray(row["gates"], dtype=float)
        src = np.asarray(row["exploration_source"], dtype=float)
        for leg in range(2):
            rec = groups.setdefault(
                (int(row["round"]), row["agent"], leg,
                 int(g[leg] > 0.5), int(src[leg])),
                {"z": [], "q": [], "g": [], "pair_id": []})
            rec["z"].append(z[leg])
            rec["q"].append(q[leg])
            rec["g"].append(g[leg])
            rec["pair_id"].append(row["pair_id"])
    out = []
    for (rnd, agent, leg, gate, source), rec in sorted(groups.items()):
        z = np.asarray(rec["z"], dtype=float)
        q = np.asarray(rec["q"], dtype=float)
        g = np.asarray(rec["g"], dtype=float)
        stats = leg_stats(z, q, g, leg)
        out.append({"seed": seed, "round": rnd, "agent": agent,
                    "leg": LEGS[leg], "gate": gate, "source": source,
                    "pairs": sorted(set(rec["pair_id"])), **stats})
    return out


def history_rows_csv(groups: list) -> list:
    columns = ["seed", "round", "agent", "leg", "gate", "source", "n",
               "frac_beyond_core", "abs_logit_p50", "abs_logit_p90",
               "abs_logit_p95", "grad_retention_p10", "grad_retention_p50",
               "grad_retention_p90", "effective_dq_dz_p10",
               "effective_dq_dz_p50", "effective_dq_dz_p90",
               "edge_quote_fraction", "gate_open_fraction"]
    return [{k: row[k] for k in columns} for row in groups]


def seed_round_summary(rows: list, seed: int) -> list:
    """Per round, gate-open and no-override only: the main training series."""
    by_round = {}
    for row in rows:
        z = np.asarray(row["price_logits"], dtype=float)
        q = np.asarray(row["quotes"], dtype=float)
        g = np.asarray(row["gates"], dtype=float)
        src = np.asarray(row["exploration_source"], dtype=float)
        for leg in range(2):
            if g[leg] <= 0.5 or src[leg] != 0.0:
                continue
            rec = by_round.setdefault(int(row["round"]),
                                      {"z": [], "q": [], "leg_mix": 0})
            rec["z"].append(z[leg])
            rec["q"].append(q[leg])
    out = []
    for rnd in sorted(by_round):
        z = np.asarray(by_round[rnd]["z"], dtype=float)
        q = np.asarray(by_round[rnd]["q"], dtype=float)
        deriv = qmap.map_derivative_np(z) / qmap.SLOPE
        out.append({
            "seed": seed, "round": rnd, "n": int(z.size),
            "coverage_v1": float(np.mean(np.abs(z) <= qmap.Z_CORE)),
            "frac_beyond_core": float(np.mean(np.abs(z) > qmap.Z_CORE)),
            "abs_logit_p50": quantiles(np.abs(z), (50,))["p50"],
            "abs_logit_p95": quantiles(np.abs(z), (95,))["p95"],
            "grad_retention_p10": quantiles(deriv, (10,))["p10"],
            "grad_retention_p50": quantiles(deriv, (50,))["p50"],
        })
    return out


def ess_leg_summary(rows: list, seed: int) -> list:
    """Gate-open, no-override logits per ESS and leg, bootstrapped by day."""
    out = []
    for agent in NAMES:
        for leg_idx, leg in enumerate(LEGS):
            z, pairs = [], []
            for row in rows:
                if row["agent"] != agent:
                    continue
                if float(row["gates"][leg_idx]) <= 0.5:
                    continue
                if float(row["exploration_source"][leg_idx]) != 0.0:
                    continue
                z.append(float(row["price_logits"][leg_idx]))
                pairs.append(row["pair_id"])
            if not z:
                continue
            z = np.asarray(z, dtype=float)
            ci = cluster_bootstrap(np.abs(z) <= qmap.Z_CORE,
                                   np.asarray(pairs), np.mean)
            out.append({
                "seed": seed, "agent": agent, "leg": leg, "n": int(z.size),
                "coverage_v1": float(np.mean(np.abs(z) <= qmap.Z_CORE)),
                "coverage_v1_ci_lo": ci["lo"], "coverage_v1_ci_hi": ci["hi"],
                "abs_logit_p50": quantiles(np.abs(z), (50,))["p50"],
                "abs_logit_p90": quantiles(np.abs(z), (90,))["p90"],
            })
    return out


def write_csv(name: str, rows: list) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def phase2() -> dict:
    all_groups, seed_rows, ess_rows = [], [], []
    for seed in SEEDS:
        rows = collect_history(seed)
        groups = group_history(rows, seed)
        all_groups.extend(groups)
        seed_rows.extend(seed_round_summary(rows, seed))
        ess_rows.extend(ess_leg_summary(rows, seed))
        print(f"phase2 seed{seed}: {len(rows)} decisions, "
              f"{len(groups)} groups", flush=True)
    write_csv("execution_mapping_stats.csv", history_rows_csv(all_groups))
    write_csv("seed_summary.csv", seed_rows)
    write_csv("ess_leg_summary.csv", ess_rows)
    payload = {"groups": all_groups, "seed_round": seed_rows,
               "ess_leg": ess_rows}
    write_json("phase2_distribution.json", payload)
    return payload


# ----------------------------------------------------------------------
# Phase 3 - candidate maps on identical logits
# ----------------------------------------------------------------------
def load_buffer_raw() -> dict:
    """`{seed}|{round}|{ESS}` raw logits and `{seed}|{round}|_gates` masks."""
    samples = {}
    for seed in SEEDS:
        path = OUT / f"buffer_raw_logits_seed{seed}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        for key in data.files:
            samples[key] = data[key].astype(float)
    return samples


def probe_logits_from_trace() -> dict:
    path = OUT / "probe_trace.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = {}
        for row in reader:
            if row["calibrating"] != "0":
                continue
            key = f"{row['seed']}|probe|{row['checkpoint']}"
            rec = out.setdefault(key, {"charge": [], "discharge": [],
                                       "gate_charge": [], "gate_discharge": []})
            rec["charge"].append(float(row["raw_charge"]))
            rec["discharge"].append(float(row["raw_discharge"]))
            rec["gate_charge"].append(float(row["gate_charge"]))
            rec["gate_discharge"].append(float(row["gate_discharge"]))
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()}
            for k, v in out.items()}


def candidate_report(z: np.ndarray, gates: np.ndarray, leg: int,
                     z_core: float) -> dict:
    z = np.asarray(z, dtype=float)
    g = cands.fraction_np(z, z_core)
    quotes = HIGHS[leg] * g
    deriv = cands.derivative_np(z, z_core)
    dq = HIGHS[leg] * deriv
    effective = np.asarray(gates, dtype=float) * dq
    open_mask = np.asarray(gates, dtype=float) > 0.5
    inside = np.abs(z) <= z_core
    return {
        "n": int(z.size),
        "n_open": int(open_mask.sum()),
        "coverage": float(np.mean(inside)) if z.size else None,
        "coverage_open": float(np.mean(inside[open_mask]))
        if open_mask.any() else None,
        "frac_beyond_core": float(np.mean(~inside)) if z.size else None,
        "quote_mean": float(np.mean(quotes)) if quotes.size else None,
        "quote_p05": quantiles(quotes, (5,))["p5"] if quotes.size else None,
        "quote_p95": quantiles(quotes, (95,))["p95"] if quotes.size else None,
        "edge_quote_fraction": edge_fraction(quotes[open_mask], leg)
        if open_mask.any() else None,
        "grad_retention_p10": quantiles(deriv / (qmap.SLOPE), (10,))["p10"]
        if deriv.size else None,
        "grad_retention_p50": quantiles(deriv / (qmap.SLOPE), (50,))["p50"]
        if deriv.size else None,
        "abs_dq_dz_p10": quantiles(dq, (10,))["p10"] if dq.size else None,
        "abs_dq_dz_p50": quantiles(dq, (50,))["p50"] if dq.size else None,
        "effective_dq_dz_p10": quantiles(effective, (10,))["p10"]
        if effective.size else None,
        "effective_dq_dz_p50": quantiles(effective, (50,))["p50"]
        if effective.size else None,
    }


def deployment_samples() -> dict:
    """Seeded eval-mode rollout logits, keyed `{seed}|{checkpoint}`.

    `final` and `best` are different policies and are never pooled.
    """
    out = {}
    for key, rec in probe_logits_from_trace().items():
        seed, _, kind = key.split("|")
        out.setdefault(f"{seed}|{kind}", []).append(rec)
    return out


def phase3() -> dict:
    tests = {str(z): cands.run_tests(z) for z in cands.CANDIDATE_Z}
    write_json("mapping_unit_test_results.json", tests)
    write_json("mapping_candidates.json", {
        "note": "candidates are diagnostic variants; no version is registered",
        "base_map": MAP_VERSION,
        "proposed_registry_if_admitted": {
            "quote_map": "linear_core_v2",
            "action_schema": "ACTION_STORAGE_QUOTE_V5",
        },
        "formula": {
            "fraction_core": "0.5 + (0.49/Z) * z, |z| <= Z",
            "fraction_upper": "1 - 0.01 / (1 + B*(z - Z))^2",
            "fraction_lower": "0.01 / (1 + B*(-z - Z))^2",
            "B": "0.49/Z/0.02",
            "endpoints": [0.01, 0.99],
        },
        "candidates": [
            {"id": cands.candidate_id(z), "z_core": z,
             "params": cands.params(z),
             "math_all_pass": tests[str(z)]["all_pass"]}
            for z in cands.CANDIDATE_Z],
    })

    samples = load_buffer_raw()
    comparisons = {}
    for z_core in cands.CANDIDATE_Z:
        per_seed = {}
        for seed in SEEDS:
            charge, discharge = [], []
            gate_charge, gate_discharge = [], []
            for key, arr in samples.items():
                if not key.startswith(f"{seed}|") or key.endswith("|_gates"):
                    continue
                gate_key = f"{seed}|{key.split('|')[1]}|_gates"
                gates = samples.get(gate_key)
                if gates is None:
                    continue
                agent = key.split("|")[2]
                idx = NAMES.index(agent)
                # Columns 0-1 are gate logits; the price legs are 2-3.
                charge.append(arr[:, 2])
                discharge.append(arr[:, 3])
                gate_charge.append(gates[:, idx, 0])
                gate_discharge.append(gates[:, idx, 1])
            if not charge:
                continue
            c = np.concatenate(charge)
            d = np.concatenate(discharge)
            gc = np.concatenate(gate_charge)
            gd = np.concatenate(gate_discharge)
            opened = np.concatenate([gc, gd]) > 0.5
            per_seed[str(seed)] = {
                "overall": {
                    "coverage": float(np.mean(
                        np.abs(np.concatenate([c, d])) <= z_core)),
                    "coverage_open": float(np.mean(
                        np.abs(np.concatenate([c, d]))[opened] <= z_core))
                    if opened.any() else None,
                    "n": int(c.size + d.size),
                    "n_open": int(opened.sum()),
                },
                "charge": candidate_report(c, gc, 0, z_core),
                "discharge": candidate_report(d, gd, 1, z_core),
                "quote_shift_vs_v1": {
                    LEGS[leg]: quantiles(
                        HIGHS[leg] * (cands.fraction_np(arr, z_core)
                                      - qmap.map_fraction_np(arr)),
                        (5, 50, 95))
                    for leg, arr in enumerate((c, d))},
                "pooled_abs_dq_dz_p50": quantiles(
                    np.concatenate([HIGHS[0] * cands.derivative_np(c, z_core),
                                    HIGHS[1] * cands.derivative_np(d, z_core)]),
                    (50,))["p50"],
                "pooled_abs_dq_dz_p10": quantiles(
                    np.concatenate([HIGHS[0] * cands.derivative_np(c, z_core),
                                    HIGHS[1] * cands.derivative_np(d, z_core)]),
                    (10,))["p10"],
            }
        comparisons[str(z_core)] = {
            "candidate": cands.candidate_id(z_core),
            "by_seed": per_seed,
        }
    ess_rows = []
    legs_arrays = buffer_leg_arrays()
    for z_core in cands.CANDIDATE_Z:
        for (seed, nm, leg), (z, gates) in sorted(legs_arrays.items()):
            rep = candidate_report(z, gates, LEGS.index(leg), z_core)
            ess_rows.append({"candidate": cands.candidate_id(z_core),
                             "z_core": z_core, "seed": seed, "ess": nm,
                             "leg": leg, **rep})
    write_csv("candidate_ess_leg.csv", ess_rows)

    deployment = {}
    for tag, recs in deployment_samples().items():
        charge = np.concatenate([r["charge"] for r in recs])
        discharge = np.concatenate([r["discharge"] for r in recs])
        z_all = np.concatenate([charge, discharge])
        opened = np.concatenate([
            np.concatenate([r["gate_charge"] for r in recs]) > 0.5,
            np.concatenate([r["gate_discharge"] for r in recs]) > 0.5])
        per_candidate = {}
        for z_core in cands.CANDIDATE_Z:
            per_candidate[str(z_core)] = {
                "coverage_open": float(np.mean(np.abs(z_all[opened]) <= z_core))
                if opened.any() else None,
                "n": int(opened.sum()),
                "quote_shift_vs_v1_charge": quantiles(
                    HIGHS[0] * (cands.fraction_np(charge, z_core)
                                - qmap.map_fraction_np(charge)), (50,)),
            }
        deployment[tag] = per_candidate
    payload = {"unit_tests": tests, "comparison": comparisons,
               "deployment_reading": deployment,
               "buffer_by_round": buffer_round_coverage(),
               "samples": {k: int(v.shape[0])
                           for k, v in samples.items()}}
    write_json("phase3_candidates.json", payload)
    print(json.dumps({z: {s: round(v["overall"]["coverage"], 4)
                          for s, v in c["by_seed"].items()}
                      for z, c in comparisons.items()}), flush=True)
    return payload


# ----------------------------------------------------------------------
# Phase 4 - fixed-sample actor gradient probe
# ----------------------------------------------------------------------
def actor_loss_terms(experiment, actors, critics, batch, agent_idx,
                     z_core: float, tau: float, seed: int) -> dict:
    """Reproduce the actor branch of `MATD3BC2Gate.update` without stepping.

    Only the price logits carry gradient. The gate draw is seeded identically
    for every candidate, so the only thing that differs is the price map.
    """
    nm = NAMES[agent_idx]
    actor = actors[nm]
    critic = critics[nm]
    obs, act = batch["obs"], batch["actions"]
    B = obs.shape[0]
    obs_flat = obs.reshape(B, -1)
    policy_out = actor(obs[:, agent_idx, :])

    generator = torch.Generator().manual_seed(seed)
    with cands.patched(z_core):
        out = qmap.forward_policy_v3(policy_out, train=True,
                                     generator=generator,
                                     gate_temperature=tau)
    quotes = out["quotes"]
    gate_hard = out["gates"].detach()
    derivative = cands.derivative_torch(out["price_logits"].detach(), z_core)

    replaced = act.clone()
    replaced[:, agent_idx, :] = quotes
    q1, q2 = critic(obs_flat, replaced.reshape(B, -1))
    q_policy = torch.min(q1, q2)
    with torch.no_grad():
        truthful = act.clone()
        truthful[:, agent_idx, :] = 0.0
        tq1, tq2 = critic(obs_flat, truthful.reshape(B, -1))
        q_truth = torch.min(tq1, tq2)
        twin_gap = (q1.detach() - q2.detach()).abs()
        delta_eff = torch.clamp(
            experiment.hinge_twin_gap_weight * twin_gap,
            min=experiment.advantage_hinge_delta)
    advantage = q_policy - q_truth
    hinge = -torch.relu(advantage - delta_eff).mean()
    highs = torch.as_tensor(HIGHS, dtype=quotes.dtype, device=quotes.device)
    anchor = ((quotes / highs) ** 2).mean()
    loss = hinge + experiment.anchor_weight * anchor

    d_hinge = torch.autograd.grad(hinge, quotes, retain_graph=True,
                                  allow_unused=True)[0]
    d_anchor = torch.autograd.grad(anchor, quotes, retain_graph=True,
                                   allow_unused=True)[0]
    d_quotes = torch.autograd.grad(loss, quotes, retain_graph=True)[0]
    d_logits = torch.autograd.grad(loss, out["price_logits"])[0]
    return {
        "loss": float(loss.detach()),
        "hinge": float(hinge.detach()),
        "anchor": float(anchor.detach()),
        "hinge_active": (advantage.detach() > delta_eff).float(),
        "gate_hard": gate_hard,
        "quotes": quotes.detach(),
        "derivative": derivative,
        "dq_dz": gate_hard * highs[None, :] * derivative,
        "d_hinge_dquote": d_hinge.detach() if d_hinge is not None else None,
        "d_anchor_dquote": d_anchor.detach() if d_anchor is not None else None,
        "d_loss_dquote": d_quotes.detach(),
        "d_loss_dz": d_logits.detach(),
    }


def fixed_batch(ckpt: Path, seed: int) -> dict:
    data = np.load(ckpt / "buffer" / "replay.npz")
    rng = np.random.default_rng(seed)
    n = data["obs"].shape[0]
    idx = rng.choice(n, size=min(GRAD_BATCH, n), replace=False)
    return {k: torch.as_tensor(data[k][idx], dtype=torch.float32)
            for k in ("obs", "actions", "rewards", "next_obs", "done")}


def critic_contract(critics) -> dict:
    nm = NAMES[0]
    critic = critics[nm]
    obs = torch.zeros(2, len(NAMES), 12)
    act = torch.zeros(2, len(NAMES), 2)
    with torch.no_grad():
        q1, q2 = critic(obs.reshape(2, -1), act.reshape(2, -1))
    return {
        "obs_flat_dim": int(obs.reshape(2, -1).shape[1]),
        "action_dim": int(act.reshape(2, -1).shape[1]),
        "agent_order": list(NAMES),
        "legs_per_unit": 2,
        "quote_units": "CNY/MWh",
        "quote_box": {"charge": [0.0, float(HIGHS[0])],
                      "discharge": [0.0, float(HIGHS[1])]},
        "q_outputs": int(q1.shape[0]),
        "ok": int(obs.reshape(2, -1).shape[1]) == 48
        and int(act.reshape(2, -1).shape[1]) == 8
        and list(q1.shape) == [2, 1],
    }


def phase4() -> dict:
    rows = []
    contract = None
    fingerprints = {}
    copies = {}
    for seed in SEEDS:
        for kind, root, step in (("final", RUN[seed], FINAL_STEP),
                                 ("best", BEST[seed], BEST_STEP)):
            source = root / "checkpoints" / f"step_{step:08d}"
            ckpt = copy_checkpoint(source, OUT / "checkpoints" /
                                   f"seed{seed}_{kind}")
            copies[f"{seed}|{kind}"] = {
                "source": rel(source), "copy": rel(ckpt),
                "copied_sha256": {p.relative_to(ckpt).as_posix(): sha256_file(p)
                                  for p in sorted(ckpt.rglob("*"))
                                  if p.is_file()},
                "source_sha256": {p.relative_to(source).as_posix():
                                  sha256_file(p)
                                  for p in sorted(source.rglob("*"))
                                  if p.is_file()
                                  and p.relative_to(source).as_posix()
                                  in {q.relative_to(ckpt).as_posix()
                                      for q in ckpt.rglob("*") if q.is_file()}},
            }
            if copies[f"{seed}|{kind}"]["copied_sha256"] != \
                    copies[f"{seed}|{kind}"]["source_sha256"]:
                raise RuntimeError(f"checkpoint copy differs: {source}")
            actors = load_actors(ckpt)
            critics = load_critics(ckpt)
            if contract is None:
                contract = critic_contract(critics)
                if not contract["ok"]:
                    write_json("phase4_gradients.json",
                               {"status": "BLOCKED",
                                "reason": "critic action contract mismatch",
                                "contract": contract})
                    return {"status": "BLOCKED", "contract": contract}
            counters = read_json(ckpt / "counters.json")
            episode = int(counters["episode"])
            experiment = mod.DAExperimentConfig("SATURATION_AUDIT")
            tau = experiment.gate_temperature_at(episode)
            batch = fixed_batch(ckpt, GRAD_SEED)
            before = {nm: {k: v.detach().clone()
                           for k, v in m.state_dict().items()}
                      for nm, m in {**actors, **critics}.items()}
            for z_core in cands.CANDIDATE_Z:
                for agent_idx, nm in enumerate(NAMES):
                    out = actor_loss_terms(experiment, actors, critics, batch,
                                           agent_idx, z_core, tau, GRAD_SEED)
                    rows.append(summarise_gradient(
                        seed, kind, episode, tau, z_core, nm, out))
            after = {nm: {k: v.detach().clone()
                          for k, v in m.state_dict().items()}
                     for nm, m in {**actors, **critics}.items()}
            fingerprints[f"{seed}|{kind}"] = all(
                torch.equal(before[nm][k], after[nm][k])
                for nm in before for k in before[nm])
    write_csv("actor_update_gradient_stats.csv", rows)
    payload = {"status": "PASS" if all(fingerprints.values()) else "FAIL",
               "contract": contract, "parameters_unchanged": fingerprints,
               "checkpoint_copies": copies, "rows": rows}
    write_json("phase4_gradients.json", payload)
    print(json.dumps({"status": payload["status"],
                      "parameters_unchanged": fingerprints}), flush=True)
    return payload


def summarise_gradient(seed, kind, episode, tau, z_core, nm, out) -> dict:
    gate = out["gate_hard"].numpy()
    open_mask = gate > 0.5
    dq = out["dq_dz"].numpy()
    d_quote = out["d_loss_dquote"].numpy()
    d_z = out["d_loss_dz"].numpy()
    # The hinge is per sample (one Q per agent), not per leg.
    hinge_active = out["hinge_active"].numpy().reshape(-1)
    row = {"seed": seed, "checkpoint": kind, "episode": episode,
           "gate_temperature": tau, "candidate": cands.candidate_id(z_core),
           "z_core": z_core, "agent": nm,
           "n": int(gate.size), "open_legs": int(open_mask.sum())}
    for leg, name in enumerate(LEGS):
        m = open_mask[:, leg]
        row[f"{name}_dq_dz_p50"] = quantiles(np.abs(dq[m, leg]), (50,))["p50"] \
            if m.any() else None
        row[f"{name}_dloss_dquote_p50"] = quantiles(
            np.abs(d_quote[m, leg]), (50,))["p50"] if m.any() else None
        row[f"{name}_dloss_dz_p50"] = quantiles(
            np.abs(d_z[m, leg]), (50,))["p50"] if m.any() else None
        row[f"{name}_hinge_active_fraction"] = float(
            hinge_active[m].mean()) if m.any() else None
        row[f"{name}_closed_dz_max_abs"] = quantiles(
            np.abs(d_z[~m, leg]), (100,))["p100"] if (~m).any() else None
    return row


# ----------------------------------------------------------------------
# Phase 5 - admission
# ----------------------------------------------------------------------
def criterion_coverage(comparison: dict, z_core: float) -> dict:
    per_seed = {}
    for s, v in comparison[str(z_core)]["by_seed"].items():
        exact = v["overall"]["coverage"]
        opened = v["overall"]["coverage_open"]
        # The primary reading excludes closed gates; the all-legs share is
        # carried beside it so the choice is visible in the verdict record.
        per_seed[s] = {"coverage_open": opened, "coverage_all": exact,
                       "n_open": v["overall"]["n_open"]}
    passed = len(per_seed) == len(SEEDS) and all(
        v["coverage_open"] is not None
        and v["coverage_open"] >= SHORTFALL_COVERAGE
        for v in per_seed.values())
    return {"criterion": "raw_logit_core_coverage_overall_open_legs",
            "threshold": SHORTFALL_COVERAGE, "per_seed": per_seed,
            "verdict": "PASS" if passed else "FAIL"}


def buffer_leg_arrays() -> dict:
    """Raw logits and recorded gates per seed, ESS and leg, all rounds pooled.

    The buffer does not record pair or block, so the pooled reading cannot be
    clustered by day; `phase2` carries the per-day view from the decision log.
    """
    samples = load_buffer_raw()
    out = {}
    for seed in SEEDS:
        for nm in NAMES:
            for leg_idx, leg in enumerate(LEGS):
                parts, gates = [], []
                for key, arr in samples.items():
                    if (not key.startswith(f"{seed}|")
                        or not key.endswith(f"|{nm}")):
                        continue
                    # Columns 0-1 are gate logits; the price legs are 2-3.
                    parts.append(arr[:, 2 + leg_idx])
                    gates.append(samples[f"{seed}|{key.split('|')[1]}|_gates"][
                        :, NAMES.index(nm), leg_idx])
                if parts:
                    out[(seed, nm, leg)] = (np.concatenate(parts),
                                            np.concatenate(gates))
    return out


def buffer_round_coverage() -> dict:
    """Open-leg coverage per seed, round and candidate.

    The deployment rollout reads one checkpoint, so it is compared against the
    buffer slice written by the same round, not against the pooled reading.
    """
    samples = load_buffer_raw()
    out = {}
    for seed in SEEDS:
        rounds = sorted({k.split("|")[1] for k in samples
                         if k.startswith(f"{seed}|") and not k.endswith("|_gates")},
                        key=int)
        for r in rounds:
            zs, gs = [], []
            for i, nm in enumerate(NAMES):
                zs.append(samples[f"{seed}|{r}|{nm}"][:, 2:])
                gs.append(samples[f"{seed}|{r}|_gates"][:, i, :])
            z = np.concatenate(zs)
            opened = np.concatenate(gs) > 0.5
            out[f"{seed}|{r}"] = {
                str(zc): (float(np.mean(np.abs(z[opened]) <= zc))
                          if opened.any() else None)
                for zc in cands.CANDIDATE_Z}
    return out


def buffer_leg_coverage(legs_arrays: dict, z_core: float) -> dict:
    """Coverage per seed, ESS and leg on the primary reading."""
    per_seed: dict = {}
    for (seed, nm, leg), (z, g) in legs_arrays.items():
        opened = g > 0.5
        per_seed.setdefault(str(seed), {})[f"{nm}.{leg}"] = {
            "n": int(z.size), "n_open": int(opened.sum()),
            "coverage": float(np.mean(np.abs(z) <= z_core)),
            "coverage_open": float(np.mean(np.abs(z[opened]) <= z_core))
            if opened.any() else None,
        }
    return per_seed


def criterion_ess_leg(legs_arrays: dict, z_core: float) -> dict:
    per_seed = buffer_leg_coverage(legs_arrays, z_core)
    failures, insufficient = [], []
    for seed, legs in per_seed.items():
        for key, row in legs.items():
            if row["n_open"] < PREREG_MIN_LEG_SAMPLE:
                insufficient.append(f"seed{seed}:{key}:n={row['n_open']}")
                continue
            if row["coverage_open"] < SHORTFALL_ESS_LEG:
                failures.append(f"seed{seed}:{key}="
                                f"{row['coverage_open']:.3f}")
    verdict = "PASS"
    if failures:
        verdict = "FAIL"
    elif insufficient:
        verdict = "BLOCKED"
    return {"criterion": "raw_logit_coverage_each_ess_leg",
            "threshold": SHORTFALL_ESS_LEG,
            "minimum_open_leg_sample": PREREG_MIN_LEG_SAMPLE,
            "per_seed": per_seed, "failures": failures,
            "insufficient_n": insufficient, "verdict": verdict}


def criterion_gradient_ratio(comparison: dict, z_core: float, factor: float,
                             quantile_key: str) -> dict:
    """Sample-quantile ratio against v1 on the same logits.

    v1's sample median is the quantity being compared: on a saturated sample
    its derivative is near zero, so this measures how much of the actor
    gradient a candidate restores, not its core slope. The v1 row has no
    ratio to report and is not a candidate.
    """
    baseline = {s: v[quantile_key]
                for s, v in comparison["2.5"]["by_seed"].items()}
    entry = comparison[str(z_core)]
    ratios = {}
    for s, v in entry["by_seed"].items():
        old, new = baseline.get(s), v[quantile_key]
        if old is None or new is None:
            ratios[s] = None
        elif old <= 0.0:
            ratios[s] = float("inf") if new > 0.0 else None
        else:
            ratios[s] = new / old
    if float(z_core) == 2.5:
        verdict = "NOT_TESTED"
        note = "baseline row; ratio is 1.0 against itself by construction"
    elif ratios and all(r is not None and r >= factor
                        for r in ratios.values()):
        verdict, note = "PASS", ""
    else:
        verdict, note = "FAIL", "one or more seeds below the factor"
    return {"criterion": f"{quantile_key}_{factor}x", "threshold": factor,
            "ratios_by_seed": ratios, "verdict": verdict, "note": note}


def phase5() -> dict:
    phase3_payload = read_json(OUT / "phase3_candidates.json")
    comparison = phase3_payload["comparison"]
    tests = phase3_payload["unit_tests"]
    grad_path = OUT / "actor_update_gradient_stats.csv"
    grad_rows = []
    if grad_path.exists() and grad_path.stat().st_size:
        with grad_path.open(encoding="utf-8") as fh:
            grad_rows = list(csv.DictReader(fh))

    legs_arrays = buffer_leg_arrays()
    decision = {}
    for z_core in cands.CANDIDATE_Z:
        crit = {
            "math_tests": {
                "criterion": "math_and_numerical_tests",
                "verdict": "PASS" if tests[str(z_core)]["all_pass"]
                else "FAIL",
                "checks": tests[str(z_core)]["checks"]},
            "coverage_overall": criterion_coverage(comparison, z_core),
            "coverage_ess_leg": criterion_ess_leg(legs_arrays, z_core),
            "gradient_median": criterion_gradient_ratio(
                comparison, z_core, 10.0, "pooled_abs_dq_dz_p50"),
            "gradient_p10": criterion_gradient_ratio(
                comparison, z_core, 3.0, "pooled_abs_dq_dz_p10"),
            "edge_fraction": criterion_edge(comparison, z_core),
            "closed_gate_zero": criterion_closed_gate(z_core),
            "physical_errors": criterion_physical(z_core, grad_rows),
        }
        cross = deployment_cross_check(phase3_payload, z_core)
        crit["coverage_overall"]["deployment_cross_check"] = cross
        if cross["disagreement"]:
            crit["coverage_overall"]["verdict"] = "PARTIAL"
        verdicts = [v["verdict"] for v in crit.values()]
        decision[str(z_core)] = {
            "candidate": cands.candidate_id(z_core),
            "criteria": crit,
            "all_pass": all(v == "PASS" for v in verdicts),
            "partial": any(v == "PARTIAL" for v in verdicts),
        }
    passing = [float(z) for z, d in decision.items() if d["all_pass"]]
    admitted = min(passing) if passing else None
    payload = {
        "decision": decision,
        "admitted_z": admitted,
        "status": (f"ADMITTED_Z{admitted}" if admitted is not None
                   else "INCONCLUSIVE"),
        "rule": "smallest Z passing every criterion, per-seed, both seeds",
    }
    write_json("candidate_comparison.json", payload)
    write_decision_report(payload, phase3_payload, grad_rows)
    write_audit_manifest(payload)
    write_output_hashes()
    print(json.dumps({"status": payload["status"],
                      "by_candidate": {z: d["all_pass"]
                                       for z, d in decision.items()}}),
          flush=True)
    return payload


#: The rollout probe reads one checkpoint, so it is matched to the buffer
#: slice that checkpoint wrote: `best` is step 1272 (round 1), `final` is
#: step 12720 (round 10).
ROUND_FOR_CHECKPOINT = {"best": "1", "final": "10"}


def deployment_cross_check(phase3_payload: dict, z_core: float) -> dict:
    """A candidate that passes on the buffer and fails in deployment, or the
    other way round, is `PARTIAL`, never `PASS`."""
    rounds = phase3_payload.get("buffer_by_round", {})
    rows, disagreement = {}, False
    for tag, per in phase3_payload.get("deployment_reading", {}).items():
        seed, kind = tag.split("|")
        round_key = f"{seed}|{ROUND_FOR_CHECKPOINT[kind]}"
        dep = per.get(str(z_core), {}).get("coverage_open")
        buf = rounds.get(round_key, {}).get(str(z_core))
        rows[tag] = {"matched_round": round_key, "buffer_coverage": buf,
                     "deployment_coverage": dep,
                     "n_deployment": per.get(str(z_core), {}).get("n")}
        if dep is None or buf is None:
            continue
        if (buf >= SHORTFALL_COVERAGE) != (dep >= SHORTFALL_COVERAGE):
            disagreement = True
    return {"per_checkpoint": rows, "disagreement": disagreement}


def write_decision_report(payload: dict, phase3_payload: dict,
                          grad_rows: list) -> Path:
    dep = phase3_payload.get("deployment_reading", {})
    lines = [
        "# Quote-map saturation audit and offline candidate comparison", "",
        "label: NON_FORMAL_DIAGNOSTIC; "
        "map under audit: `linear_core_v1`; no version registered", "",
        f"admission status: **{payload['status']}**", "",
        "## Admission criteria",
        "",
        "| Z | math | coverage | per ESS-leg | median dq/dz | p10 dq/dz | "
        "edge -pp | closed gate | physical | all pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for z, entry in payload["decision"].items():
        c = entry["criteria"]
        lines.append(
            f"| {z} | {c['math_tests']['verdict']} | "
            f"{c['coverage_overall']['verdict']} | "
            f"{c['coverage_ess_leg']['verdict']} | "
            f"{c['gradient_median']['verdict']} | "
            f"{c['gradient_p10']['verdict']} | "
            f"{c['edge_fraction']['verdict']} | "
            f"{c['closed_gate_zero']['verdict']} | "
            f"{c['physical_errors']['verdict']} | "
            f"**{entry['all_pass']}** |")
    lines += ["", "## Readings of the coverage criterion", "",
              "The pooled column is the admission reading. The rollout reads "
              "one checkpoint, so it is cross-checked against the buffer slice "
              "that checkpoint wrote (`best` = round 1, `final` = round 10).",
              "",
              "| Z | checkpoint | buffer, matched round | buffer, pooled | "
              "deployment rollout |",
              "|---|---|---|---|---|"]
    for z in payload["decision"]:
        c = payload["decision"][z]["criteria"]["coverage_overall"]
        for tag, row in sorted(
                c["deployment_cross_check"]["per_checkpoint"].items()):
            pooled = c["per_seed"].get(tag.split("|")[0], {}).get(
                "coverage_open")
            lines.append(
                f"| {z} | {tag} | {fmt(row['buffer_coverage'])} | "
                f"{fmt(pooled)} | {fmt(row['deployment_coverage'])} |")
    lines += ["", "## Criterion values", "",
              "| Z | seed | min ESS-leg cov | med dq/dz ratio | "
              "p10 dq/dz ratio | edge -pp charge | edge -pp discharge |",
              "|---|---|---|---|---|---|---|"]
    for z, entry in payload["decision"].items():
        c = entry["criteria"]
        for seed in SEEDS:
            legs = c["coverage_ess_leg"]["per_seed"].get(str(seed), {})
            usable = [v["coverage_open"] for v in legs.values()
                      if v["coverage_open"] is not None
                      and v["n_open"] >= PREREG_MIN_LEG_SAMPLE]
            drop = c["edge_fraction"].get("drop_pp", {}).get(str(seed), {})
            lines.append(
                f"| {z} | {seed} | "
                f"{fmt(min(usable) if usable else None)} | "
                f"{fmt_ratio(c['gradient_median']['ratios_by_seed'].get(str(seed)))} | "
                f"{fmt_ratio(c['gradient_p10']['ratios_by_seed'].get(str(seed)))} | "
                f"{fmt(drop.get('charge'))} | {fmt(drop.get('discharge'))} |")
    lines += [
        "",
        "## Verdict vocabulary",
        "",
        "- `PASS` — the named criterion's own evidence clears its threshold.",
        "- `FAIL` — the evidence contradicts the threshold.",
        "- `PARTIAL` — the buffer reading and the deployment rollout disagree "
        "on the same criterion; it is never reported as `PASS`.",
        "- `NOT_TESTED` — the baseline row (v1 against itself) and the "
        "historical raw-logit layer, which was never recorded.",
        "- `INCONCLUSIVE` — no candidate clears every criterion; no new "
        "candidate is added and no training is started.",
        "",
        "## Not tested or blocked",
        "",
        "- Historical per-row raw logits: `NOT_TESTED`. "
        "`block_decisions.jsonl` carries the post-noise price logits and the "
        "executed quotes, not the actor output; the buffer forward is a "
        "different reading and is labelled as such.",
        "- Buffer samples cannot be clustered by day or episode: `NOT_TESTED` "
        "for that stratification. `ess_leg_summary.csv` clusters the "
        "historical noisy layer by training pair instead.",
        "- Payoff, clearing outcome and settlement under any candidate: "
        "`NOT_TESTED`. Nothing here runs the market.",
        "- The base B+C2 stack (`rl_b_c2_gate`, v2 sigmoid map): "
        "`NOT_TESTED` by scope, not by failure.",
        "",
        "## What the evidence supports",
        "",
        "- The mapping layer is reproduced exactly by the candidate module at "
        "`Z=2.5`; the remap error over every recorded decision row and every "
        "buffer sample is in `phase1_observations.json`.",
        "- Per-seed core coverage, gradient retention and edge-quote fractions "
        "for each candidate on identical logits, in `phase3_candidates.json` "
        "and `candidate_ess_leg.csv`.",
        "- Per-seed, per-ESS-leg coverage on the primary reading, in "
        "`candidate_comparison.json`.",
        f"- Admission: **{payload['status']}**.",
        "",
        "## What this does not show",
        "",
        "- Offline quote statistics are not payoffs. A wider core lowers every "
        "core quote (`g(2.5)` falls from 0.99 to 0.65 at Z=8 and 0.62 at Z=10), "
        "so the candidates are a quote-level intervention, not a tail-only "
        "repair; the clearing and the settlement are not simulated here.",
        "- The historical actor-loss inputs were never recorded and cannot be "
        "recovered: historical rows are `NOT_TESTED` for raw logits, and the "
        "buffer forward is the checkpoint policy on stored states, not the "
        "policy that filled the buffer.",
        "- `best` is round 1 (episode 53), `final` is round 10 (episode 530). "
        "They are reported separately and never averaged.",
        "- The base B+C2 stack still uses the v2 sigmoid map and is outside "
        "this audit.",
        "",
        "## Files",
        "",
        "`input_hashes.json`, `phase1_observations.json`, "
        "`execution_mapping_stats.csv`, `seed_summary.csv`, "
        "`ess_leg_summary.csv`, `mapping_candidates.json`, "
        "`mapping_unit_test_results.json`, `phase3_candidates.json`, "
        "`actor_update_gradient_stats.csv`, `candidate_comparison.json`, "
        "`probe_trace.csv`, `buffer_raw_logits_seed*.npz`, "
        "`output_hashes.json`.",
    ]
    path = OUT / "decision_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def fmt_ratio(value) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf (v1 median is 0)"
    return f"{value:.1f}x"


def write_audit_manifest(payload: dict) -> Path:
    files = {p.name: sha256_file(p) for p in sorted(OUT.glob("*"))
             if p.is_file()}
    manifest = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": "NON_FORMAL_DIAGNOSTIC",
        "map_under_audit": MAP_VERSION,
        "action_schema": ACTION_SCHEMA,
        "audited_trainer": "rl_da_gate.DAGateTrainer",
        "out_of_scope": "rl_b_c2_gate v2 sigmoid map (base B+C2 stack)",
        "git": git_state(),
        "environment": diag.env_versions(),
        "status": payload["status"],
        "phases": ["0 evidence", "1 observations", "2 distribution",
                   "3 candidates", "4 gradients", "5 decision"],
        "artifacts": files,
    }
    return write_json("audit_manifest.json", manifest)


def write_output_hashes() -> Path:
    files = {p.name: sha256_file(p) for p in sorted(OUT.glob("*"))
             if p.is_file() and p.name != "output_hashes.json"}
    copies = {p.relative_to(OUT).as_posix(): sha256_file(p)
              for p in sorted((OUT / "checkpoints").rglob("*"))
              if p.is_file()} if (OUT / "checkpoints").exists() else {}
    return write_json("output_hashes.json",
                      {"files": files, "checkpoint_copies": copies})


def criterion_edge(comparison: dict, z_core: float) -> dict:
    old = comparison["2.5"]["by_seed"]
    new = comparison[str(z_core)]["by_seed"]
    drops, verdicts = {}, []
    for seed in SEEDS:
        o = old[str(seed)]
        n = new[str(seed)]
        d = {}
        for leg in ("charge", "discharge"):
            if o[leg]["edge_quote_fraction"] is None \
                    or n[leg]["edge_quote_fraction"] is None:
                d[leg] = None
                continue
            d[leg] = 100.0 * (o[leg]["edge_quote_fraction"]
                              - n[leg]["edge_quote_fraction"])
        drops[str(seed)] = d
        if z_core == 2.5:
            continue
        verdicts.extend(v is not None and v >= 20.0 for v in d.values())
    if z_core == 2.5:
        return {"criterion": "edge_fraction_drop_pp", "threshold": 20.0,
                "drop_pp": drops, "verdict": "NOT_TESTED",
                "note": "baseline candidate is the reference, not a pass"}
    return {"criterion": "edge_fraction_drop_pp", "threshold": 20.0,
            "drop_pp": drops,
            "verdict": "PASS" if all(verdicts) and verdicts else "FAIL"}


def criterion_closed_gate(z_core: float) -> dict:
    checks = cands.run_tests(z_core)["checks"]
    return {"criterion": "closed_gate_price_gradient_zero",
            "verdict": "PASS" if checks["closed_gate_zero"] else "FAIL",
            "checks": {"closed_gate_zero": checks["closed_gate_zero"]}}


def criterion_physical(z_core: float, grad_rows: list) -> dict:
    """A shut gate must carry no price gradient, and no recorded gradient may
    be infinite or NaN. Quote bounds `[0, high]` follow from `g in [0, 1]`."""
    relevant = [r for r in grad_rows
                if abs(float(r["z_core"]) - z_core) < 1e-9]
    closed, nonfinite = [], []
    for row in relevant:
        for key, value in row.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if key.endswith("_closed_dz_max_abs"):
                closed.append(abs(v))
            elif not np.isfinite(v):
                nonfinite.append(key)
    ok = not nonfinite and all(v == 0.0 for v in closed)
    return {"criterion": "no_physical_or_numerical_errors",
            "closed_leg_dz_max_abs": max(closed) if closed else None,
            "nonfinite_fields": sorted(set(nonfinite)),
            "verdict": "PASS" if ok else "FAIL"}


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
PHASES = {"0": phase0, "1": phase1, "2": phase2, "3": phase3,
          "4": phase4, "5": phase5}


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    chosen = []
    for arg in args:
        if arg.startswith("--phase"):
            continue
        chosen.append(arg)
    if not chosen or chosen == ["all"]:
        order = ["0", "1", "2", "3", "4", "5"]
    else:
        order = [a for a in chosen if a in PHASES]
        unknown = [a for a in chosen if a not in PHASES]
        if unknown:
            raise SystemExit(f"unknown phase(s): {unknown}")
    started = time.time()
    for phase in order:
        t0 = time.time()
        print(f"[phase {phase}] start", flush=True)
        PHASES[phase]()
        print(f"[phase {phase}] done in {time.time() - t0:.1f}s", flush=True)
    print(f"all phases done in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
