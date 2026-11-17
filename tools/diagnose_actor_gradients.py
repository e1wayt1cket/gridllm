# diagnose_actor_gradients.py
"""Penalty vs critic gradient at the saved actors, plus tanh saturation.

The trained critics were **not checkpointed** with the actors, so the trained
critic's gradient cannot be reconstructed. This script reports the penalty
gradient exactly at each saved actor and a critic-gradient scale reference from
a freshly initialised critic. The reference is labelled as such; it is not
evidence about the trained critic.

Usage:
    PYTHONPATH=src python src/diagnose_actor_gradients.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

import diagnose_pilot_behaviour as behaviour
import run_matd3_pilot as pilot
import v2_cy_jn_env as envmod
from rl_spec import OBS_V6
from rl_td3 import deviation_penalty

REPORT = "reports/v2_cy_jn_2025/actor_gradient_penalty_diagnostic.md"
CSV_OUT = "data/processed/v2_cy_jn_2025/pilot/actor_gradients.csv"


def collect_obs(dataset: str, seed: int, days: Sequence[dict]):
    import torch
    env, trainer = behaviour._load_actors(dataset, seed)
    names = trainer.agent_names
    samples = []
    for row in days:
        agents, price = pilot.day_agents(row, env.config)
        envmod.configure_price(row["price_date"])
        env.set_agents(agents, day_id=row["price_date"])
        obs = env.reset()
        samples.append({nm: torch.as_tensor(obs[nm], dtype=torch.float32)
                        for nm in names})
    return env, trainer, samples


def gradients(dataset: str, seed: int, days: Sequence[dict]) -> List[dict]:
    import torch
    env, trainer, samples = collect_obs(dataset, seed, days)
    names = trainer.agent_names
    for a in trainer.actors.values():
        a.eval()
    for c in trainer.critics.values():
        c.eval()
    rows = []
    for nm in names:
        actor = trainer.actors[nm]
        low, high = trainer.bounds_by_agent[nm]
        truthful = trainer.truthful_action[nm]
        pen_norms, pen_dim, sat = [], [], []
        for sample in samples:
            obs = sample[nm]
            logits, action = actor.forward_logits(obs.unsqueeze(0))
            penalty = deviation_penalty(logits, truthful, (low, high),
                                        pilot.PILOT["dev_penalty"])
            grads = torch.autograd.grad(penalty, list(actor.parameters()),
                                        allow_unused=True,
                                        retain_graph=True)
            pen_norms.append(float(torch.sqrt(sum(
                (g ** 2).sum() for g in grads if g is not None))))
            dim_grad = torch.autograd.grad(penalty, logits)[0]
            pen_dim.append(float(dim_grad.abs().max()))
            tanh_abs = torch.abs(torch.tanh(logits))
            sat.append(float((tanh_abs > 0.99).float().mean()))
        rows.append({
            "dataset": dataset, "seed": seed, "unit": nm,
            "penalty_grad_norm": float(np.mean(pen_norms)),
            "penalty_grad_max_dim": float(np.mean(pen_dim)),
            "saturated_share": float(np.mean(sat)),
        })
    critic_ref = _critic_reference(trainer, samples)
    for row in rows:
        row["critic_grad_norm_at_init"] = critic_ref.get(row["unit"],
                                                         float("nan"))
    return rows


def _critic_reference(trainer, samples) -> dict:
    """Gradient of -min(Q1,Q2) through each actor at a *fresh* critic."""
    import torch
    names = trainer.agent_names
    out = {}
    for nm in names:
        critic = trainer.critics[nm]
        norms = []
        for sample in samples:
            global_obs = torch.cat([sample[n] for n in names]).unsqueeze(0)
            joint_act = torch.cat(
                [trainer.actors[n](sample[n]).unsqueeze(0) for n in names],
                dim=-1)
            q1, q2 = critic(global_obs, joint_act)
            loss = -torch.min(q1, q2).mean()
            grads = torch.autograd.grad(
                loss, list(trainer.actors[nm].parameters()), allow_unused=True)
            norms.append(float(torch.sqrt(sum(
                (g ** 2).sum() for g in grads if g is not None))))
        out[nm] = float(np.mean(norms))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="NEAR,FULL96")
    parser.add_argument("--seeds", default="11,22")
    args = parser.parse_args(argv)
    root = Path.cwd()
    days = behaviour._sample_days()[:4]          # four training/validation days
    all_rows = []
    for dataset in args.datasets.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            all_rows.extend(gradients(dataset, seed, days))
            print(f"{dataset} seed{seed} gradients done", flush=True)
    with Path(root / CSV_OUT).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    groups = {}
    for row in all_rows:
        groups.setdefault(row["unit"], []).append(row)
    lines = [
        "# Actor gradient vs deviation penalty (V2-CY-JN-2025)",
        "",
        "## A gap first",
        "",
        "**The trained critics were not saved.** `save_policy` writes actors "
        "only, so the critic term's gradient at the trained weights cannot be "
        "reconstructed without re-running training, which this phase forbids. "
        "The `critic gradient` column below is measured with a **freshly "
        "initialised** critic at the saved actor's observations; it is an "
        "initialisation-scale reference, not a statement about the trained "
        "critic. This is a reproducibility gap to close in any future run by "
        "checkpointing the critics.",
        "",
        "## Penalty gradient (exact at the saved actors)",
        "",
        "| unit | penalty grad norm | max per-action logit grad | "
        "saturated share | critic grad norm (fresh, reference) |",
        "|---|---:|---:|---:|---:|",
    ]
    for unit, rows in sorted(groups.items()):
        lines.append(
            f"| {unit} | {np.mean([r['penalty_grad_norm'] for r in rows]):.3e} | "
            f"{np.mean([r['penalty_grad_max_dim'] for r in rows]):.3e} | "
            f"{np.mean([r['saturated_share'] for r in rows]):.2f} | "
            f"{np.mean([r['critic_grad_norm_at_init'] for r in rows]):.3e} |")
    lines += [
        "",
        "## What the numbers mean",
        "",
        "For a storage unit the truthful point is the lower corner of each "
        "box, so with `d = (1 + tanh(x)) / 2` the penalty gradient with respect "
        "to one logit is at most `16/27 * weight = 0.0593` at weight 0.10. The "
        "actors sit near `tanh = 0`, where the derivative is `0.5 * weight = "
        "0.05` per action, not zero. The penalty is therefore measurable — "
        "but small beside the critic term's gradient at initialisation "
        "(compare the last two numeric columns).",
        "",
        "It is **not** evidence that the penalty is too weak on its own. The "
        "actors did travel to saturated corners, but often the wrong corner "
        "(a charge quote near 150 refuses to charge); the block reward does "
        "not price the SOC that refusal leaves unused. The frozen weight is "
        "not retuned here.",
        "",
        "## Saturation",
        "",
        "Saturation is partial: some action dimensions reach |tanh| > 0.99 "
        "(the table's share is over the sampled observations), so on those "
        "dimensions the penalty gradient does vanish. The action diagnostic "
        "reports the per-block share over twelve days; the actor is not "
        "uniformly parked at the centre, but neither has it moved the whole "
        "quote pair to the truthful corner.",
    ]
    path = root / REPORT
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / CSV_OUT}\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
