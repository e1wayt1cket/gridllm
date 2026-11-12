"""Reproduce the seed-53 clearing failure on one day, read-only.

The failure checkpoint was written by the failure handler, so its RNG state is
already past the failing block. This script does not claim to replay the
original stochastic actions. It runs the same weights on the same day in
deterministic and truthful modes, stresses the failing block with its captured
inputs, and reports a classification with that limit stated.

The run directory is only read. Solver dumps are disabled so an infeasible
model never writes `socp_infeasible.ilp` into the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import traceback
from contextlib import contextmanager

import numpy as np
import torch

import gurobipy as gp

# An infeasible model would otherwise be dumped beside the working directory.
gp.Model.write = lambda *args, **kwargs: None

import market  # noqa: E402
import rl_da_gate as mod  # noqa: E402
import v2_cy_jn_env as envmod  # noqa: E402
from rl_spec import ACTION_STORAGE_QUOTE_V2  # noqa: E402

ROOT = pathlib.Path.cwd()
N_BLOCKS = 24
PAIR = {"load_date": "2022-10-20", "price_date": "2025-10-20"}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(manifest: dict) -> dict:
    expected = manifest.get("source_sha256") or {}
    actual, mismatched = {}, []
    for name, digest in expected.items():
        path = ROOT / "src" / name
        actual[name] = sha256(path) if path.exists() else None
        if actual[name] != digest:
            mismatched.append(name)
    return {"checked": len(expected), "mismatched": mismatched}


def build_trainer(run_dir: pathlib.Path):
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    experiment = mod.DAExperimentConfig(**manifest["experiment"])
    env = envmod.build_env_for_pair(
        PAIR, differential_reward=False, terminal_soc_target=0.5)
    trainer = mod.DAGateTrainer(env, experiment)
    bundle = trainer.load(run_dir)
    counters = bundle["counters"]
    return manifest, env, trainer, {
        "total_steps": int(counters["total_steps"]),
        "episode": int(counters["episode"]),
        "updates": int(counters["updates"]),
        "buffer_len": len(trainer.buffer.buffer),
        "rng_state": sorted(
            key for key in bundle.get("rng", {})),
    }


@contextmanager
def record_clearing():
    """Capture every clearing call and the exception the env swallows."""
    original = market.clear_market
    state = {"calls": [], "last_args": None, "last_kwargs": None}

    def wrapped(*args, **kwargs):
        try:
            result = original(*args, **kwargs)
            solver = (result or {}).get("solver", {}).get("final", {})
            state["calls"].append({"ok": True, "solver": solver})
            return result
        except Exception as exc:  # noqa: BLE001 - the point is to expose it
            state["last_args"] = args
            state["last_kwargs"] = kwargs
            state["calls"].append({
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            })
            raise

    market.clear_market = wrapped
    try:
        yield state
    finally:
        market.clear_market = original


def run_day(env, trainer, *, train: bool) -> list[dict]:
    obs = env.reset()
    log = []
    for block in range(N_BLOCKS):
        actions, diags = trainer._fleet_action(obs, train=train)
        prev_soc = {k: float(v) for k, v in env.prev_soc.items()}
        obs, _, _, _ = env.step(actions)
        log.append({
            "block": block,
            "ok": env._last_result is not None,
            "prev_soc": prev_soc,
            "quotes": {
                nm: np.asarray(actions[nm], dtype=float).tolist()
                for nm in actions},
            "gates": {
                nm: np.asarray(diags[nm]["gates"], dtype=float).tolist()
                for nm in diags},
        })
        if not log[-1]["ok"]:
            break
    return log


def run_truthful_day(env) -> list[dict]:
    truthful = ACTION_STORAGE_QUOTE_V2.truthful_array()
    obs = env.reset()
    log = []
    for block in range(N_BLOCKS):
        actions = {agent.name: truthful for agent in env.rl_agents}
        obs, _, _, _ = env.step(actions)
        log.append({"block": block, "ok": env._last_result is not None})
        if not log[-1]["ok"]:
            break
    return log


def terminal_reachability(env, soc_by_agent: dict, periods: int = 4) -> dict:
    out = {}
    for agent in env.all_agents:
        spec = agent.storage
        if spec is None:
            continue
        soc = float(soc_by_agent.get(agent.name, spec.soc0))
        high = soc
        for _ in range(periods):
            power = spec.max_charge_feasible(high)
            high = min(spec.soc_max,
                       high + power * 0.25 * spec.eta_ch / spec.e_max)
        low = soc
        for _ in range(periods):
            power = spec.max_discharge_feasible(low)
            low = max(spec.soc_min,
                      low - power * 0.25 / spec.eta_dis / spec.e_max)
        out[agent.name] = {
            "soc": soc,
            "reachable_min": low,
            "reachable_max": high,
            "terminal_reachable": low - 1e-9 <= 0.5 <= high + 1e-9,
        }
    return out


def stress_block(original, args, kwargs, repeats: int = 3) -> list[dict]:
    out = []
    for _ in range(repeats):
        try:
            original(*args, **kwargs)
            out.append({"ok": True})
        except Exception as exc:  # noqa: BLE001
            out.append({"ok": False, "error": type(exc).__name__,
                        "message": str(exc)})
    return out


def rng_sweep(env, trainer, seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        log = run_day(env, trainer, train=True)
        failed = [row for row in log if not row["ok"]]
        out.append({
            "seed": seed,
            "blocks_run": len(log),
            "failed_block": failed[0]["block"] if failed else None,
            "failed_row": failed[0] if failed else None,
        })
    return out


def classify(det, truthful, stress, sweep) -> str:
    det_failed = next((r for r in det if not r["ok"]), None)
    truth_failed = next((r for r in truthful if not r["ok"]), None)
    if det_failed is not None and truth_failed is not None:
        return "DATA_INHERENT"
    if det_failed is not None:
        if stress and not all(row["ok"] for row in stress):
            return "ACTION_POLICY_INHERENT"
        return "SOLVER_TRANSIENT"
    if any(row["failed_block"] is not None for row in sweep):
        return "ACTION_POLICY_INHERENT"
    return "NOT_REPRODUCED"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--sweep-count", type=int, default=5)
    args = parser.parse_args(argv)
    run_dir = ROOT / args.run_dir

    manifest, env, trainer, restored = build_trainer(run_dir)
    report = {
        "run_dir": run_dir.as_posix(),
        "pair": PAIR,
        "sources": verify_sources(manifest),
        "restored_state": restored,
        "limitation": (
            "the checkpoint was saved by the failure handler; its RNG state is "
            "after the failing block, so the original stochastic actions "
            "cannot be replayed"),
        "pair_in_train_subset": (
            f"{PAIR['load_date']}|{PAIR['price_date']}"
            in manifest.get("train_subset", [])),
    }

    with record_clearing() as rec:
        det = run_day(env, trainer, train=False)
    report["deterministic"] = det
    report["deterministic_clearing"] = rec["calls"]
    det_failed = next((r for r in det if not r["ok"]), None)
    if det_failed is not None:
        report["reachability_at_failure"] = terminal_reachability(
            env, det_failed["prev_soc"])
        if rec["last_args"] is not None:
            # Outside the recorder, `market.clear_market` is the original.
            report["block_stress"] = stress_block(
                market.clear_market, rec["last_args"], rec["last_kwargs"])
    else:
        report["reachability_at_block_23"] = terminal_reachability(
            env, det[-1]["prev_soc"])

    with record_clearing() as truth_rec:
        truthful = run_truthful_day(env)
    report["truthful"] = truthful
    report["truthful_clearing"] = truth_rec["calls"]

    sweep = []
    if not args.no_sweep:
        with record_clearing() as sweep_rec:
            sweep = rng_sweep(
                env, trainer, list(range(1, args.sweep_count + 1)))
        report["rng_sweep"] = sweep
        report["rng_sweep_clearing"] = [
            call for call in sweep_rec["calls"] if not call["ok"]]

    report["classification"] = classify(
        det, truthful, report.get("block_stress"), sweep)
    print(json.dumps({k: report[k] for k in (
        "sources", "restored_state", "pair_in_train_subset",
        "classification")}, indent=2))
    for key in ("deterministic", "truthful", "rng_sweep"):
        rows = report.get(key) or []
        failed = [r for r in rows if not r.get("ok", True)]
        print(key, "blocks=", len(rows), "failed=", failed[:1])

    if args.write:
        out = ROOT / "reports/cc02/seed53_replay.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
