"""P22: save/destroy/restore/continue round trip for the accepted candidate.

Read-only over the candidate smoke run; writes only a new temporary checkpoint
and a JSON result. No historical checkpoint is modified.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import rl_da_gate as mod

RUN = Path("runs/P21_candidate_001_smoke_seed11")
OUT = Path("reports/v2_cy_jn_2025/formal_training_ready/"
           "phase22_checkpoint_roundtrip.json")
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def fake_env():
    return SimpleNamespace(
        rl_agents=[SimpleNamespace(name=n) for n in NAMES],
        get_state_dim=lambda: 12,
    )


def build(experiment):
    return mod.DAGateTrainer(
        fake_env(), experiment, agent_names=list(NAMES))


def actions(trainer):
    obs = {nm: np.linspace(0.0, 1.0, 12, dtype=np.float32) for nm in NAMES}
    return {nm: np.asarray(trainer.select_action(
        obs[nm], nm, train=False)["quotes"], dtype=float) for nm in NAMES}


def seed_all(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    experiment = mod.DAExperimentConfig(**manifest["experiment"])
    source = build(experiment)
    source.load(RUN)
    before_actions = actions(source)
    before_counters = {
        "total_steps": source.total_steps,
        "episodes": source.episodes,
        "updates": source.updates,
        "buffer_len": len(source.buffer),
    }
    with tempfile.TemporaryDirectory(prefix="makerb_ckpt_") as tmp:
        path = source.save(
            Path(tmp), level="resume", dataset="NEAR",
            seed=int(manifest["seed"]),
            config_hash_value=manifest["market_config_hash"],
            data=manifest["data_hashes"],
            code_revision="roundtrip-audit")
        restored = build(experiment)
        restored.load(Path(tmp))
        after_actions = actions(restored)
        after_counters = {
            "total_steps": restored.total_steps,
            "episodes": restored.episodes,
            "updates": restored.updates,
            "buffer_len": len(restored.buffer),
        }
        seed_all()
        source_out = source.update()
        seed_all()
        restored_out = restored.update()
        loss_keys = ("critic_loss", "actor_loss", "q_gap")
        losses = {
            key: {"source": source_out[key], "restored": restored_out[key]}
            for key in loss_keys
        }
        checkpoint_manifest = json.loads(
            (Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "actions_equal": all(
            np.allclose(before_actions[nm], after_actions[nm], atol=0.0)
            for nm in NAMES),
        "counters_equal": before_counters == after_counters,
        "buffer_equal": len(source.buffer) == len(restored.buffer),
        "losses_equal": all(
            abs(losses[key]["source"] - losses[key]["restored"]) < 1e-8
            for key in loss_keys),
        "schema_v3": checkpoint_manifest["schema_version"]
        == "matd3-checkpoint-v3",
        "actor_loss_map": checkpoint_manifest["da_gate"]["actor_loss_map"]
        == "linear_core_v1",
    }
    payload = {
        "boundary": ("temporary checkpoint only; the source run is read-only"),
        "source_run": str(RUN),
        "checks": checks,
        "all_pass": all(checks.values()),
        "before_counters": before_counters,
        "after_counters": after_counters,
        "losses": losses,
        "actions": {nm: np.asarray(values).tolist()
                    for nm, values in before_actions.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": payload["all_pass"], "checks": checks},
                     indent=2))
    print(f"wrote {OUT}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
