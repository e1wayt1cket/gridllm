"""Compare the critic loss and value under uniform, mixture and stratified-only
sampling on the saved v3 buffers.

This is a post-hoc offline measurement, not a training run. It holds the
critic, target critic and target actors fixed and changes only the sampling
distribution, so it reports the loss weighting the mixture declares rather
than a new learning result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import rl_da_gate as mod
import train_da_gate as tb
from diagnose_critic_long_horizon import WINTER, pairs

DATA = Path("data/processed/v2_cy_jn_2025")
RUNS = Path("runs")
RUN_ID = "DEV_DA_GATE_seed11_v2"


def load(root):
    experiment = mod.DAExperimentConfig(**json.loads(
        (RUNS / RUN_ID / "run_manifest.json").read_text(
            encoding="utf-8"))["experiment"])
    pair_map, _ = pairs()
    env = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainer = mod.DAGateTrainer(env, experiment)
    trainer.load(root)
    return trainer


def losses(trainer, uniform_fraction, batch=768):
    b = trainer.buffer.sample(batch, uniform_fraction=uniform_fraction)
    obs = b["obs"]
    act = b["actions"]
    rew = b["rewards"]
    nobs = b["next_obs"]
    done = b["done"]
    bs = obs.shape[0]
    obs_flat = obs.reshape(bs, -1)
    out = {}
    for i, nm in enumerate(trainer.agent_names):
        with torch.no_grad():
            target_actions = torch.stack([
                mod.target_quotes_v3(trainer.actor_targets[nm2](nobs[:, j, :]))
                for j, nm2 in enumerate(trainer.agent_names)], dim=1)
            q1t, q2t = trainer.critic_targets[nm](
                obs_flat, target_actions.reshape(bs, -1))
            target = rew[:, i:i + 1] + trainer.experiment.gamma \
                * (1.0 - done[:, None]) * torch.min(q1t, q2t)
            q1, q2 = trainer.critics[nm](obs_flat, act.reshape(bs, -1))
            loss = float((torch.nn.functional.mse_loss(q1, target)
                          + torch.nn.functional.mse_loss(q2, target)) / 2)
        out[nm] = {
            "loss": loss,
            "q1_mean": float(q1.mean()),
            "q2_mean": float(q2.mean()),
            "twin_gap": float((q1 - q2).abs().mean()),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
        }
    out["sampler"] = b["sampler_stats"]
    return out


def main() -> int:
    result = {}
    for label, root in (("best", RUNS / f"{RUN_ID}_best"),
                        ("final", RUNS / RUN_ID)):
        trainer = load(root)
        result[label] = {
            "uniform": losses(trainer, 1.0),
            "mixture": losses(trainer, 0.5),
            "stratified_only": losses(trainer, 0.0),
            "buffer": len(trainer.buffer),
        }
    (DATA / "da_gate_sampler_critic_loss.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    for label, r in result.items():
        print(label, "buffer", r["buffer"])
        for mode in ("uniform", "mixture", "stratified_only"):
            losses_mean = np.mean([r[mode][u]["loss"]
                                   for u in r[mode] if u != "sampler"])
            ess = r[mode]["sampler"]["ess_fraction"]
            print(f"  {mode}: mean critic loss {losses_mean:.5f} "
                  f"ess_fraction {ess:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
