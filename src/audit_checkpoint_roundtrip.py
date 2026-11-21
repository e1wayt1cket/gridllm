# audit_checkpoint_roundtrip.py
"""Save, destroy, load and compare a small checkpoint. No training, no env."""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from checkpoint_loader_v2 import load_checkpoint, restore_checkpoint, verify_files
from checkpoint_writer_v2 import save_checkpoint

OUT = "data/processed/v2_cy_jn_2025/pilot/checkpoint_roundtrip_results.json"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


class _Buffer:
    def __init__(self, n=8):
        self.capacity = 100
        self.buffer = [
            (np.full((4,), i, dtype=np.float32),
             np.array([i, i + 1], dtype=np.float32),
             np.array([i * 0.5], dtype=np.float32),
             np.full((4,), i + 1, dtype=np.float32),
             np.array([0.0], dtype=np.float32))
            for i in range(n)]


def _opt_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_opt_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    return a == b


def _modules(seed=0):
    torch.manual_seed(seed)
    actors = {u: nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
              for u in UNITS}
    actor_targets = {u: nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                                      nn.Linear(8, 2)) for u in UNITS}
    critics = {u: nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
               for u in UNITS}
    critic_targets = {u: nn.Sequential(nn.Linear(6, 16), nn.ReLU(),
                                       nn.Linear(16, 1)) for u in UNITS}
    for group in (actor_targets, critic_targets):
        for u in group:
            group[u].load_state_dict(
                (actors if group is actor_targets else critics)[u]
                .state_dict())
    return actors, actor_targets, critics, critic_targets


def run(root: Path) -> dict:
    actors, actor_targets, critics, critic_targets = _modules()
    opt_a = torch.optim.Adam([p for m in actors.values()
                              for p in m.parameters()], lr=1e-3)
    opt_c = torch.optim.Adam([p for m in critics.values()
                              for p in m.parameters()], lr=1e-3)
    # Advance optimizers so their state is non-empty.
    loss = sum(m(torch.zeros(1, 4)).sum() for m in actors.values()) \
        + sum(m(torch.zeros(1, 6)).sum() for m in critics.values())
    loss.backward()
    opt_a.step()
    opt_c.step()

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    reward = {"definition": "C2_potential_soc_time", "gamma": 0.99,
              "scale": 1000.0, "potential": {"form": "soc_time"},
              "normalization": "fixed", "terminal_rule": "none",
              "b1_source": "b1_daily_manifest.json"}
    config_hash = "b" * 64
    counters = {"total_steps": 48, "episode": 2, "updates": 0,
                "buffer_len": 8}
    save_checkpoint(
        root, level="resume", run_id="roundtrip", dataset="NEAR", seed=11,
        code_revision="rev", config_hash=config_hash,
        specs={"obs_spec": {"name": "v6_12d", "version": 6},
               "action_spec": {"name": "storage_quote_v2", "version": 2}},
        actors=actors, actor_targets=actor_targets, critics=critics,
        critic_targets=critic_targets,
        optimizers={"actor": opt_a, "critic": opt_c},
        counters=counters, reward=reward, buffer=_Buffer(),
        data={}, hyperparameters={"lr": 1e-3})
    expected_states = {
        "actor": {u: actors[u].state_dict() for u in UNITS},
        "critic": {u: critics[u].state_dict() for u in UNITS},
        "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict()}
    probe = torch.tensor([[0.3, -0.2, 1.0, 2.0]])
    expected_action = actors["ESS9"](probe).detach().clone()
    expected_rng = (torch.rand(3).clone(), float(np.random.rand()),
                    random.random())
    del actors, actor_targets, critics, critic_targets, opt_a, opt_c

    bundle = load_checkpoint(root)
    a2, at2, c2, ct2 = _modules(seed=99)
    oa2 = torch.optim.Adam([p for m in a2.values() for p in m.parameters()],
                           lr=1e-3)
    oc2 = torch.optim.Adam([p for m in c2.values() for p in m.parameters()],
                           lr=1e-3)
    restore_checkpoint(bundle, actors=a2, actor_targets=at2, critics=c2,
                       critic_targets=ct2,
                       optimizers={"actor": oa2, "critic": oc2})
    checks = {
        "actor_states_equal": all(torch.equal(
            a2[u].state_dict()[k], expected_states["actor"][u][k])
            for u in UNITS for k in expected_states["actor"][u]),
        "critic_states_equal": all(torch.equal(
            c2[u].state_dict()[k], expected_states["critic"][u][k])
            for u in UNITS for k in expected_states["critic"][u]),
        "optimizer_actor_equal": _opt_equal(
            oa2.state_dict(), expected_states["opt_a"]),
        "optimizer_critic_equal": _opt_equal(
            oc2.state_dict(), expected_states["opt_c"]),
        "deterministic_action": torch.equal(a2["ESS9"](probe).detach(),
                                            expected_action),
        "rng_restored": (torch.equal(torch.rand(3), expected_rng[0])
                         and abs(float(np.random.rand()) - expected_rng[1])
                         < 1e-15
                         and random.random() == expected_rng[2]),
        "buffer_restored": bundle["buffer"]["obs"].shape == (8, 4),
        "counters_restored": bundle["counters"] == counters,
        "config_hash_restored": bundle["manifest"]["config_hash"]
        == config_hash,
        "files_present": verify_files(bundle, root) == [],
    }
    # Incomplete checkpoint must be refused.
    broken_root = root.parent / "broken"
    if broken_root.exists():
        shutil.rmtree(broken_root)
    shutil.copytree(root, broken_root)
    (broken_root / "checkpoints" / "step_00000048" / "critics"
     / "ESS9.pt").unlink()
    try:
        load_checkpoint(broken_root)
        checks["incomplete_rejected"] = False
    except FileNotFoundError:
        checks["incomplete_rejected"] = True
    shutil.rmtree(broken_root)
    return {"checks": checks, "all_pass": all(checks.values())}


def main(argv=None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        payload = run(Path(tmp) / "run")
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
