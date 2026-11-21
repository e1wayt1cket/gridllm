# checkpoint_loader_v2.py
"""Load and restore a schema-v2 MATD3 checkpoint. Pure I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from validate_checkpoint_schema import validate


def _load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file missing: {path}")
    # Local, trusted artifacts: the RNG state carries numpy objects, so the
    # weights-only restriction would reject it.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(root: str | Path) -> dict:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate(manifest)
    if errors:
        raise ValueError("checkpoint manifest failed validation: "
                         + "; ".join(errors))
    ckpt = root / "checkpoints" / f"step_{int(manifest['step']):08d}"
    result = {"manifest": manifest, "networks": {}}
    for group in ("actors", "actor_targets", "critics", "critic_targets"):
        result["networks"][group] = {
            agent: _load_state(ckpt / group / f"{agent}.pt")
            for agent in manifest["networks"][group]}
    if manifest["level"] in ("resume", "bundle"):
        result["optimizers"] = _load_state(ckpt / "optimizers.pt")
        result["counters"] = json.loads(
            (ckpt / "counters.json").read_text(encoding="utf-8"))
        result["rng"] = _load_state(ckpt / "rng.pt")
        if manifest["buffer"].get("saved"):
            path = ckpt / manifest["buffer"]["path"]
            if not path.exists():
                raise FileNotFoundError(f"buffer missing: {path}")
            with np.load(path) as data:
                result["buffer"] = {k: data[k] for k in data.files}
    return result


def restore_checkpoint(bundle: dict, *, actors: Dict[str, torch.nn.Module],
                       actor_targets: Dict[str, torch.nn.Module],
                       critics: Dict[str, torch.nn.Module],
                       critic_targets: Dict[str, torch.nn.Module],
                       optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None
                       ) -> None:
    """Load state dicts and restore the RNG. Raises on a missing agent."""
    groups = (("actors", actors), ("actor_targets", actor_targets),
              ("critics", critics), ("critic_targets", critic_targets))
    for name, modules in groups:
        for agent, module in modules.items():
            state = bundle["networks"][name].get(agent)
            if state is None:
                raise KeyError(f"{name}.{agent} missing from the checkpoint")
            module.load_state_dict(state)
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            state = bundle["optimizers"].get(name)
            if state is None:
                raise KeyError(f"optimizer {name} missing")
            optimizer.load_state_dict(state)
    rng = bundle.get("rng")
    if rng is not None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])


def verify_files(bundle: dict, root: str | Path) -> list:
    """Check every file the manifest names exists."""
    root = Path(root)
    manifest = bundle["manifest"]
    ckpt = root / "checkpoints" / f"step_{int(manifest['step']):08d}"
    missing = []
    for group in ("actors", "actor_targets", "critics", "critic_targets"):
        for agent in manifest["networks"][group]:
            if not (ckpt / group / f"{agent}.pt").exists():
                missing.append(f"{group}/{agent}.pt")
    if manifest["level"] in ("resume", "bundle"):
        for name in ("optimizers.pt", "rng.pt", "counters.json"):
            if not (ckpt / name).exists():
                missing.append(name)
        if manifest["buffer"].get("saved"):
            if not (ckpt / manifest["buffer"]["path"]).exists():
                missing.append(manifest["buffer"]["path"])
    return missing
