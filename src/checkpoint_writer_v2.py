# checkpoint_writer_v2.py
"""Write a schema-v2 MATD3 checkpoint. Pure I/O; no training code is touched."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from validate_checkpoint_schema import validate

SCHEMA_VERSION = "matd3-checkpoint-v2"


def _state(modules: Dict[str, torch.nn.Module]) -> Dict[str, dict]:
    return {name: mod.state_dict() for name, mod in modules.items()}


def _buffer_arrays(buffer) -> Dict[str, np.ndarray]:
    rows = list(getattr(buffer, "buffer", []))
    if not rows:
        arrays = {"obs": np.zeros((0,)), "actions": np.zeros((0,)),
                  "rewards": np.zeros((0,)), "next_obs": np.zeros((0,)),
                  "done": np.zeros((0,))}
    else:
        obs, act, rew, next_obs, done = zip(*rows)
        arrays = {"obs": np.asarray(obs), "actions": np.asarray(act),
                  "rewards": np.asarray(rew),
                  "next_obs": np.asarray(next_obs),
                  "done": np.asarray(done)}
    # A gate buffer carries the quantities its estimator needs beyond the
    # five-tuple; they are saved under their own names so loading can rebuild
    # the same buffer rather than only the reward-relevant slice.
    extras = getattr(buffer, "extra_arrays", None)
    if extras:
        for key, values in extras.items():
            arrays[str(key)] = np.asarray(values)
    return arrays


def save_checkpoint(root: str | Path, *, level: str, run_id: str,
                    dataset: str, seed: int, code_revision: str,
                    config_hash: str, specs: dict, actors: dict,
                    actor_targets: dict, critics: dict,
                    critic_targets: dict, optimizers: dict,
                    counters: dict, reward: dict, data: Optional[dict] = None,
                    hyperparameters: Optional[dict] = None,
                    exposure_log: Optional[str] = None,
                    eval_provenance: Optional[str] = None,
                    buffer=None, buffer_reason: Optional[str] = None,
                    step: Optional[int] = None,
                    schema_version: str = SCHEMA_VERSION,
                    extra_manifest: Optional[dict] = None) -> Path:
    """Save one checkpoint directory and its manifest.

    `optimizers` maps a name to an optimizer object; their state dicts are
    saved. `buffer` may be an object with `.buffer` and `.capacity`, or None.
    """
    if level not in ("inference", "resume", "bundle"):
        raise ValueError(f"unknown level {level!r}")
    root = Path(root)
    step = int(counters.get("total_steps", 0) if step is None else step)
    ckpt = root / "checkpoints" / f"step_{step:08d}"
    for name, mods in (("actors", actors), ("actor_targets", actor_targets),
                       ("critics", critics),
                       ("critic_targets", critic_targets)):
        for agent, mod in mods.items():
            path = ckpt / name / f"{agent}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mod.state_dict(), path)

    if level in ("resume", "bundle"):
        opt = ckpt / "optimizers.pt"
        opt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.state_dict() for k, v in optimizers.items()}, opt)
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
        }
        torch.save(rng, ckpt / "rng.pt")
        (ckpt / "counters.json").write_text(
            json.dumps(counters, indent=2), encoding="utf-8")

        if buffer is not None:
            arrays = _buffer_arrays(buffer)
            ckpt.joinpath("buffer").mkdir(parents=True, exist_ok=True)
            np.savez(ckpt / "buffer" / "replay.npz", **arrays)
            buffer_manifest = {
                "saved": True,
                "path": "buffer/replay.npz",
                "capacity": int(getattr(buffer, "capacity", len(arrays["obs"]))),
                "index": int(getattr(buffer, "index", len(arrays["obs"]))),
                "count": int(len(arrays["obs"])),
            }
        else:
            buffer_manifest = {"saved": False,
                               "reason": buffer_reason
                               or "the caller declared no buffer"}
    else:
        buffer_manifest = {"saved": False,
                           "reason": "inference checkpoints carry no buffer"}

    manifest = {
        "schema_version": schema_version,
        "level": level,
        "step": step,
        "run_id": run_id,
        "dataset": dataset,
        "seed": int(seed),
        "code_revision": code_revision,
        "config_hash": config_hash,
        "specs": specs,
        "networks": {
            "actors": list(actors),
            "actor_targets": list(actor_targets),
            "critics": list(critics),
            "critic_targets": list(critic_targets),
        },
        "optimizers": list(optimizers) if optimizers else None,
        "counters": dict(counters),
        "rng": {"python": True, "numpy": True, "torch_cpu": True,
                "torch_cuda": torch.cuda.is_available(), "env": None},
        "buffer": buffer_manifest,
        "reward": reward,
        "hyperparameters": hyperparameters or {},
        "data": data or {},
        "exposure_log": exposure_log,
        "eval_provenance": eval_provenance,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    errors = validate(manifest)
    if errors:
        raise ValueError("checkpoint manifest failed schema validation: "
                         + "; ".join(errors))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    return ckpt
