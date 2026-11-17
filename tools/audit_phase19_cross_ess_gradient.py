"""P19-C: cross-ESS gradient conflict on the actual network structure.

The actors are per agent: `trainer.actors[name]` are separate `GateActor`
instances with disjoint parameters, so no ESS can cancel another ESS's actor
update through a shared actor trunk. The audit states that directly and then
measures the two conflicts that do exist:

* each centralized critic's credit to each unit's action (own-action share);
* within one actor, the charge and discharge legs' gradients through the
  shared trunk (`net.0`, `net.2`).

It also reports each agent's own actor-gradient norm on the same state, so a
"shared parameter" claim can be checked rather than assumed.

Read-only: no training, no checkpoint write, no H2 and no legacy test.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import da_quote_map as v3
import train_b_c2_gate as tb
from audit_daily_counterfactual import load_checkpoints
from diagnose_critic_long_horizon import collect_state, pairs, WINTER

DATA = Path("data/processed/v2_cy_jn_2025")
STATES = (("2022-12-24", 8, "ESS14"), ("2022-12-24", 8, "ESS27"),
          ("2022-12-29", 14, "ESS14"), ("2022-12-29", 14, "ESS27"))
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


def resolve(pair_map: dict, load_date: str) -> str:
    hits = [k for k in pair_map if k.startswith(load_date + "|")]
    if len(hits) != 1:
        raise KeyError(f"expected one pair for {load_date}, got {hits}")
    return hits[0]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def critic_credit(trainer, obs, actions, critic_unit: str) -> dict:
    names = list(obs)
    obs_flat = torch.tensor(np.stack([obs[n] for n in names]),
                            dtype=torch.float32).reshape(1, -1)
    act = torch.tensor(np.stack([actions[n] for n in names]),
                       dtype=torch.float32).unsqueeze(0)
    leaf = act.clone().requires_grad_(True)
    q1, q2 = trainer.critics[critic_unit](
        obs_flat, leaf.reshape(1, -1))
    qmin = torch.min(q1.mean(), q2.mean())
    grad = torch.autograd.grad(qmin, leaf)[0][0]
    norms = {nm: float(grad[i].norm()) for i, nm in enumerate(names)}
    total = sum(v * v for v in norms.values())
    share = {nm: (v * v / total if total > 0 else 0.0)
             for nm, v in norms.items()}
    return {"grad_norms": norms, "own_action_share": share[critic_unit],
            "credit": share,
            "grad_vectors": {nm: grad[i].numpy().tolist()
                             for i, nm in enumerate(names)}}


def actor_trunk_grad(actor, obs_t, output_grad: np.ndarray,
                     mask: np.ndarray) -> torch.Tensor:
    out = actor(obs_t)
    shared = [actor.net[0].weight, actor.net[0].bias,
              actor.net[2].weight, actor.net[2].bias]
    g = torch.tensor(output_grad * mask, dtype=torch.float32).unsqueeze(0)
    grads = torch.autograd.grad(out, shared, grad_outputs=g)
    return torch.cat([x.reshape(-1) for x in grads])


def actor_conflict(trainer, obs, baseline, unit: str) -> dict:
    actor = trainer.actors[unit]
    names = list(obs)
    idx = names.index(unit)
    obs_t = torch.tensor(obs[unit], dtype=torch.float32).unsqueeze(0)
    obs_flat = torch.tensor(np.stack([obs[n] for n in names]),
                            dtype=torch.float32).reshape(1, -1)
    act = torch.tensor(np.stack([baseline[n] for n in names]),
                       dtype=torch.float32).unsqueeze(0)
    out = actor(obs_t)
    out.retain_grad()
    torch.manual_seed(0)
    shaped = v3.forward_policy_v3(
        out, train=True, gate_temperature=trainer.experiment.gate_temperature)
    replaced = act.clone()
    replaced[:, idx, :] = shaped["quotes"]
    q1, q2 = trainer.critics[unit](obs_flat, replaced.reshape(1, -1))
    (-torch.min(q1, q2).mean()).backward()
    grad = out.grad.detach().numpy()[0].copy()
    actor.zero_grad()
    masks = {
        "charge_leg": np.array([grad[0], 0.0, grad[2], 0.0]),
        "discharge_leg": np.array([0.0, grad[1], 0.0, grad[3]]),
        "gate_head": np.array([grad[0], grad[1], 0.0, 0.0]),
        "price_head": np.array([0.0, 0.0, grad[2], grad[3]]),
    }
    vecs = {name: actor_trunk_grad(actor, obs_t, grad, mask)
            for name, mask in masks.items()}
    return {
        "output_grad": grad.tolist(),
        "trunk_grad_norm": {k: float(v.norm()) for k, v in vecs.items()},
        "trunk_cosine_charge_vs_discharge": cosine(
            vecs["charge_leg"], vecs["discharge_leg"]),
        "trunk_cosine_gate_vs_price": cosine(
            vecs["gate_head"], vecs["price_head"]),
    }


def parameter_disjoint(trainer) -> dict:
    refs = {}
    for unit in trainer.agent_names:
        refs[unit] = {id(p) for p in trainer.actors[unit].parameters()}
    overlap = {}
    for i, a in enumerate(trainer.agent_names):
        for b in trainer.agent_names[i + 1:]:
            overlap[f"{a}|{b}"] = len(refs[a] & refs[b])
    return {"parameter_tensors_shared": sum(overlap.values()),
            "pairwise_shared": overlap,
            "actors_count": len(trainer.agent_names)}


def main() -> int:
    pair_map, _ = pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainers = load_checkpoints(template, pair_map)
    rows = []
    for label in ("da_best", "da_final"):
        trainer = trainers[label]
        structure = parameter_disjoint(trainer)
        for load_date, block, unit in STATES:
            pid = resolve(pair_map, load_date)
            env, obs, baseline, past, soc, gates, probs, plog, price = \
                collect_state(trainer, pair_map[pid], block)
            credit = {u: critic_credit(trainer, obs, baseline, u)
                      for u in UNITS}
            conflict = actor_conflict(trainer, obs, baseline, unit)
            for critic_unit, cr in credit.items():
                for action_unit in UNITS:
                    rows.append({
                        "checkpoint": label, "pair_id": pid,
                        "block": block, "state_unit": unit,
                        "critic": critic_unit, "action_unit": action_unit,
                        "credit_norm": cr["grad_norms"][action_unit],
                        "credit_share": cr["credit"][action_unit],
                        "critic_own_action_share": cr["own_action_share"],
                        "state_unit_own_share":
                            credit[unit]["own_action_share"],
                        "trunk_cosine_charge_vs_discharge":
                            conflict["trunk_cosine_charge_vs_discharge"],
                        "trunk_cosine_gate_vs_price":
                            conflict["trunk_cosine_gate_vs_price"],
                        "trunk_grad_norm_charge":
                            conflict["trunk_grad_norm"]["charge_leg"],
                        "trunk_grad_norm_discharge":
                            conflict["trunk_grad_norm"]["discharge_leg"],
                        "structure_shared_actor_tensors":
                            structure["parameter_tensors_shared"],
                    })
            print(f"{label} {pid} b{block} {unit}: "
                  f"own={credit[unit]['own_action_share']:.3f} "
                  f"leg_cos={conflict['trunk_cosine_charge_vs_discharge']:.3f} "
                  f"gate_price_cos="
                  f"{conflict['trunk_cosine_gate_vs_price']:.3f}",
                  flush=True)
    out = DATA / "phase19_cross_ess_gradient.json"
    out.write_text(json.dumps({
        "structure": parameter_disjoint(trainers["da_best"]),
        "rows": rows}, indent=2), encoding="utf-8")
    csv_path = DATA / "phase19_cross_ess_gradient.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(parameter_disjoint(trainers["da_best"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
