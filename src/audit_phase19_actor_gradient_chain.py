"""P19-B: actor gradient chain at named states.

Two states are the phase-18 "critic right, actor wrong" cases; both are NEAR
validation days and are used for offline diagnosis only. Two NEAR training
states are added so the defect can be separated from the state choice.

The audit replays the trainer's own actor loss for a single state:

    actor(obs) -> quote map -> critic -> -min(Q1,Q2) -> logit grad -> Adam

It records the logits, the gate forward and surrogate backward, the v3 quote
map and its derivative, the critic gradient with respect to the action, the
actor gradient and its clip, the Adam state, and a virtual one-step update on
a deep copy. The code path uses `forward_policy` (v2 sigmoid map); the
intended v3 path is computed beside it with the same seed.

Read-only: no training, no checkpoint write, no H2 and no legacy test.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import da_quote_map as v3
import train_b_c2_gate as tb
import truthful_gate_actor_adapter as v2
from audit_daily_counterfactual import load_checkpoints
from diagnose_critic_long_horizon import collect_state, pairs, WINTER

DATA = Path("data/processed/v2_cy_jn_2025")
NAMED = (("2022-12-30|2025-12-31", 10, "ESS14"),
         ("2022-12-13|2025-12-11", 12, "ESS27"))
TRAIN = (("2022-12-24", 8, "ESS14"), ("2022-12-29", 14, "ESS27"))


def resolve_pair(pair_map: dict, pid: str) -> str:
    if pid in pair_map:
        return pid
    hits = [k for k in pair_map if k.startswith(pid)]
    if len(hits) != 1:
        raise KeyError(f"expected one pair for {pid}, got {hits}")
    return hits[0]


def critic_action_grad(trainer, obs, action, unit: str) -> dict:
    names = list(obs)
    obs_flat = torch.tensor(np.stack([obs[n] for n in names]),
                            dtype=torch.float32).reshape(1, -1)
    act = torch.tensor(np.stack([action[n] for n in names]),
                       dtype=torch.float32).unsqueeze(0)
    target_act = act[0, names.index(unit)].clone().requires_grad_(True)
    full = act.clone()
    full[0, names.index(unit)] = target_act
    q1, q2 = trainer.critics[unit](obs_flat, full.reshape(1, -1))
    q1 = q1.mean()
    q2 = q2.mean()
    qmin = torch.min(q1, q2)
    g1 = torch.autograd.grad(q1, target_act, retain_graph=True,
                             allow_unused=True)[0]
    g2 = torch.autograd.grad(q2, target_act, retain_graph=True,
                             allow_unused=True)[0]
    gm = torch.autograd.grad(qmin, target_act, allow_unused=True)[0]
    return {"dq1_dquote": g1.numpy().tolist(),
            "dq2_dquote": g2.numpy().tolist(),
            "dqmin_dquote": gm.numpy().tolist(),
            "q1": float(q1.detach()), "q2": float(q2.detach())}


def adam_state(opt) -> dict:
    out = {}
    with torch.no_grad():
        for i, p in enumerate(opt.param_groups[0]["params"]):
            st = opt.state.get(p)
            if not st:
                continue
            out[str(i)] = {
                "step": int(st["step"]) if "step" in st else None,
                "exp_avg_norm": float(st["exp_avg"].norm()),
                "exp_avg_sq_sqrt_mean": float(st["exp_avg_sq"].sqrt().mean()),
            }
    return out


def actor_head_state(actor, opt) -> dict:
    """Final-layer rows for the two gate logits and the two price logits."""
    final_w = actor.net[4].weight
    final_idx = 4
    st = opt.state.get(final_w, {})
    rows = {}
    for name, idx in (("gate_charge", 0), ("gate_discharge", 1),
                      ("price_charge", 2), ("price_discharge", 3)):
        rows[name] = {
            "weight_row_norm": float(final_w.data[idx].norm()),
            "adam_exp_avg_norm": (float(st["exp_avg"][idx].norm())
                                  if "exp_avg" in st else None),
            "adam_exp_avg_sq_sqrt_mean": (
                float(st["exp_avg_sq"][idx].sqrt().mean())
                if "exp_avg_sq" in st else None),
        }
    rows["final_index"] = final_idx
    return rows


def run_state(trainer, label, pid, block, unit, pair_map) -> dict:
    pair = pair_map[pid]
    env, obs, baseline, past, soc, gates, probs, plog, price = \
        collect_state(trainer, pair, block)
    names = list(obs)
    actor = trainer.actors[unit]
    opt = trainer.actor_opts[unit]
    obs_t = torch.tensor(obs[unit], dtype=torch.float32).unsqueeze(0)
    obs_flat = torch.tensor(np.stack([obs[n] for n in names]),
                            dtype=torch.float32).reshape(1, -1)
    act = torch.tensor(np.stack([baseline[n] for n in names]),
                       dtype=torch.float32).unsqueeze(0)
    idx = names.index(unit)

    with torch.no_grad():
        logits = actor(obs_t).numpy()[0]

    def chain(mapfn, kw):
        out = actor(obs_t)
        out.retain_grad()
        torch.manual_seed(0)
        shaped = mapfn(out, train=True, **kw)
        replaced = act.clone()
        replaced[:, idx, :] = shaped["quotes"]
        q1, q2 = trainer.critics[unit](obs_flat, replaced.reshape(1, -1))
        loss = -torch.min(q1, q2).mean()
        actor.zero_grad()
        loss.backward()
        grad = out.grad.detach().numpy()[0].copy()
        layers = {n: (None if p.grad is None else float(p.grad.norm()))
                  for n, p in actor.named_parameters()}
        actor.zero_grad()
        return loss, grad, layers, shaped

    loss_code, grad_code, layer_code, shaped_code = chain(
        v2.forward_policy, {"temperature": trainer.experiment.gate_temperature})
    loss_v3, grad_v3, layer_v3, shaped_v3 = chain(
        v3.forward_policy_v3,
        {"gate_temperature": trainer.experiment.gate_temperature})

    z = torch.as_tensor(logits[2:], dtype=torch.float32)
    with torch.no_grad():
        out_eval = actor(obs_t)
        v3_quotes = v3.forward_policy_v3(
            out_eval, train=False)["quotes"].numpy()[0]
        v2_quotes = v2.forward_policy(
            out_eval, train=False)["quotes"].numpy()[0]
    v3_deriv = [float(v3.map_derivative(z[i]) * v3.LEG_HIGHS[i])
                for i in range(2)]
    v2_deriv = [float(v2.LEG_HIGHS[i] * torch.sigmoid(z[i])
                      * (1 - torch.sigmoid(z[i]))) for i in range(2)]

    cg = critic_action_grad(trainer, obs,
                            {n: baseline[n].copy() for n in names}, unit)

    norm_code = float(np.linalg.norm(grad_code))
    norm_v3 = float(np.linalg.norm(grad_v3))
    cos = float(grad_code @ grad_v3
                / (norm_code * norm_v3 + 1e-12))

    # Virtual one-step update on a deep copy, both paths, same seed.
    virtual = {}
    for path, mapfn, kw in (
            ("code_v2", v2.forward_policy, {"temperature": 1.0}),
            ("intended_v3", v3.forward_policy_v3,
             {"gate_temperature": 1.0})):
        a2, o2 = copy.deepcopy((actor, opt))
        a2.zero_grad()
        torch.manual_seed(0)
        out2 = a2(obs_t)
        shaped2 = mapfn(out2, train=True, **kw)
        rep2 = act.clone()
        rep2[:, idx, :] = shaped2["quotes"]
        qa, qb = trainer.critics[unit](obs_flat, rep2.reshape(1, -1))
        l2 = -torch.min(qa, qb).mean()
        l2.backward()
        before = {n: p.detach().clone() for n, p in a2.named_parameters()}
        total = torch.nn.utils.clip_grad_norm_(a2.parameters(), 1.0)
        o2.step()
        deltas = {n: float((p - before[n]).norm())
                  for n, p in a2.named_parameters()}
        with torch.no_grad():
            new_quotes = v3.forward_policy_v3(
                a2(obs_t), train=False)["quotes"].numpy()[0]
            new_logits = a2(obs_t).numpy()[0]
        virtual[path] = {
            "loss": float(l2.detach()),
            "param_delta_norm": float(np.sqrt(sum(v * v
                                                  for v in deltas.values()))),
            "param_deltas": deltas,
            "logits_before": logits.tolist(),
            "logits_after": new_logits.tolist(),
            "v3_quotes_before": v3_quotes.tolist(),
            "v3_quotes_after": new_quotes.tolist(),
            "clip_total_norm_before": float(total),
        }

    return {
        "checkpoint": label, "pair_id": pid, "block": block, "unit": unit,
        "soc_start": soc[unit],
        "gate_eval_deterministic": probs[unit],
        "gate_probability_sigmoid": gates[unit],
        "gate_hard_sampled_train": shaped_code["hard_gates"]
            .detach().numpy()[0].tolist(),
        "gate_surrogate_factor": [
            float(shaped_code["probs"][0, 0]
                  * (1 - shaped_code["probs"][0, 0])),
            float(shaped_code["probs"][0, 1]
                  * (1 - shaped_code["probs"][0, 1]))],
        "logits": logits.tolist(),
        "v2_quotes_eval": v2_quotes.tolist(),
        "v3_quotes_eval": v3_quotes.tolist(),
        "v3_derivative": v3_deriv,
        "v2_derivative": v2_deriv,
        "critic_action_grad": cg,
        "loss_code_v2": float(loss_code.detach()),
        "loss_intended_v3": float(loss_v3.detach()),
        "logit_grad_code_v2": grad_code.tolist(),
        "logit_grad_intended_v3": grad_v3.tolist(),
        "grad_cosine_v2_v3": cos,
        "grad_norm_ratio_v2_v3": float(norm_code / (norm_v3 + 1e-12)),
        "layer_grad_norm_code_v2": layer_code,
        "layer_grad_norm_intended_v3": layer_v3,
        "clip_threshold": 1.0,
        "clip_coef_code_v2": float(min(1.0, 1.0 / (norm_code + 1e-12))),
        "clip_coef_intended_v3": float(min(1.0, 1.0 / (norm_v3 + 1e-12))),
        "adam_state": adam_state(opt),
        "adam_head_state": actor_head_state(actor, opt),
        "virtual_update": virtual,
        "policy_delay": trainer.experiment.policy_delay,
        "updates_at_checkpoint": trainer.updates,
    }


def main() -> int:
    pair_map, _ = pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainers = load_checkpoints(template, pair_map)
    rows = []
    for label in ("da_best", "da_final"):
        trainer = trainers[label]
        states = list(NAMED) + [
            (resolve_pair(pair_map, pid), b, u) for pid, b, u in TRAIN]
        for pid, block, unit in states:
            rows.append(run_state(trainer, label, pid, block, unit, pair_map))
            print(f"{label} {pid} b{block} {unit}: "
                  f"cos={rows[-1]['grad_cosine_v2_v3']:.3f} "
                  f"ratio={rows[-1]['grad_norm_ratio_v2_v3']:.3f}",
                  flush=True)
    out = DATA / "phase19_actor_gradient_chain.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat = []
    for r in rows:
        flat.append({
            "checkpoint": r["checkpoint"], "pair_id": r["pair_id"],
            "block": r["block"], "unit": r["unit"],
            "loss_code_v2": r["loss_code_v2"],
            "loss_intended_v3": r["loss_intended_v3"],
            "grad_cosine_v2_v3": r["grad_cosine_v2_v3"],
            "grad_norm_ratio_v2_v3": r["grad_norm_ratio_v2_v3"],
            "grad_norm_code_v2": float(np.linalg.norm(
                r["logit_grad_code_v2"])),
            "grad_norm_intended_v3": float(np.linalg.norm(
                r["logit_grad_intended_v3"])),
            "quote_diff_charge": (r["v3_quotes_eval"][0]
                                  - r["v2_quotes_eval"][0]),
            "quote_diff_discharge": (r["v3_quotes_eval"][1]
                                     - r["v2_quotes_eval"][1]),
            "actor_charge_quote": r["v3_quotes_eval"][0],
            "actor_discharge_quote": r["v3_quotes_eval"][1],
            "dqmin_dcharge": r["critic_action_grad"]["dqmin_dquote"][0],
            "dqmin_ddischarge": r["critic_action_grad"]["dqmin_dquote"][1],
            "virtual_code_quote_delta_charge": (
                r["virtual_update"]["code_v2"]["v3_quotes_after"][0]
                - r["virtual_update"]["code_v2"]["v3_quotes_before"][0]),
            "virtual_code_quote_delta_discharge": (
                r["virtual_update"]["code_v2"]["v3_quotes_after"][1]
                - r["virtual_update"]["code_v2"]["v3_quotes_before"][1]),
            "virtual_v3_quote_delta_charge": (
                r["virtual_update"]["intended_v3"]["v3_quotes_after"][0]
                - r["virtual_update"]["intended_v3"]["v3_quotes_before"][0]),
            "virtual_v3_quote_delta_discharge": (
                r["virtual_update"]["intended_v3"]["v3_quotes_after"][1]
                - r["virtual_update"]["intended_v3"]["v3_quotes_before"][1]),
        })
    csv_path = DATA / "phase19_actor_gradient_chain.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(flat, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
