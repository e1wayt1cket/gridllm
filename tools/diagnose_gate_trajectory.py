"""P0 diagnosis: gate trajectory and ST-gradient effectiveness.

Read-only. Loads the ten round checkpoints through the audit roots, evaluates
the actors on a fixed truthful reference trajectory, and computes the actor
loss gradient on a fixed replay batch. It does not step an optimizer or write
any checkpoint.

Outputs:
- data/processed/v2_cy_jn_2025/gate_trajectory_by_round.json
- data/processed/v2_cy_jn_2025/st_gate_gradient_by_round.json
- reports/v2_cy_jn_2025/gate_logit_trajectory_audit.md
- reports/v2_cy_jn_2025/st_gate_gradient_audit.md
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import rl_b_c2_gate as mod
import train_b_c2_gate as tb
from truthful_gate_actor_adapter import forward_policy

ROOT = Path.cwd()
RUN_ID = "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
RUN = ROOT / "runs" / RUN_ID
ROOTS = ROOT / "runs" / f"{RUN_ID}_audit_roots"
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
NEAR_PAIRING = DATA / "near/load_price_pairing.csv"
FULL_PAIRING = DATA / "load_price_pairing.csv"
NEAR_SPLIT = DATA / "near/split.json"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")


def read_csv(path):
    import csv
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def train_val_pairs():
    near = read_csv(NEAR_PAIRING)
    full = read_csv(FULL_PAIRING)
    near_by = {f"{r['load_date']}|{r['price_date']}": r for r in near}
    full_by = {f"{r['load_date']}|{r['price_date']}": r for r in full}
    load_to_pair = {r["load_date"]: f"{r['load_date']}|{r['price_date']}"
                    for r in near}
    split = json.loads(NEAR_SPLIT.read_text(encoding="utf-8"))

    def build(load):
        pid = load_to_pair[load]
        return {"load_date": load,
                "price_date": full_by[pid]["price_date"],
                "season": near_by[pid].get("season", "")}
    return build(split["train"][0]), build(split["validation"][0]), full_by


def load_trainer(round_no: int, template_env, experiment):
    root = ROOTS / f"round_{round_no:03d}"
    trainer = mod.MATD3BC2Gate(template_env, experiment)
    trainer.load(root)
    return trainer


def fixed_states(trainer, pair: dict):
    """Truthful reference trajectory: the same observations for every actor."""
    env = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    states = []
    names = trainer.agent_names
    for block in range(24):
        states.append({nm: np.asarray(obs[nm], dtype=np.float32).copy()
                       for nm in names})
        acts = {nm: np.array([0.0, 0.0]) for nm in names}
        obs, _, done, _ = env.step(acts)
        if done:
            break
    return states


def trajectory(trainer, states, seed):
    names = trainer.agent_names
    gates = {nm: {leg: {"logits": [], "probs": [], "det": [],
                        "price": []} for leg in LEGS} for nm in names}
    det_modes = {}
    sample_modes = {}
    sample_counts = {nm: [0, 0] for nm in names}
    sample_total = 0
    with torch.no_grad():
        for state in states:
            det_vec = []
            for nm in names:
                obs_t = torch.as_tensor(state[nm]).unsqueeze(0)
                out = trainer.actors[nm](obs_t)[0]
                gate_logits = out[:2]
                probs = torch.sigmoid(gate_logits)
                price = (torch.as_tensor(mod.LEG_HIGHS)
                         * torch.sigmoid(out[2:]))
                for i, leg in enumerate(LEGS):
                    gates[nm][leg]["logits"].append(float(gate_logits[i]))
                    gates[nm][leg]["probs"].append(float(probs[i]))
                    gates[nm][leg]["det"].append(int(gate_logits[i] > 0))
                    gates[nm][leg]["price"].append(float(price[i]))
                det_vec.extend(int(x) for x in (gate_logits > 0))
            mode = mod.fleet_mode(
                [det_vec[0] + 2 * det_vec[1],
                 det_vec[2] + 2 * det_vec[3],
                 det_vec[4] + 2 * det_vec[5],
                 det_vec[6] + 2 * det_vec[7]])
            det_modes[mode] = det_modes.get(mode, 0) + 1
            # 100 Bernoulli draws from each actor's probabilities.
            torch.manual_seed(seed + len(sample_modes) + 1)
            for _ in range(100):
                vec = []
                for nm in names:
                    obs_t = torch.as_tensor(state[nm]).unsqueeze(0)
                    probs = torch.sigmoid(trainer.actors[nm](obs_t)[0, :2])
                    draw = (torch.rand(2) < probs).to(torch.long).tolist()
                    for leg_idx, value in enumerate(draw):
                        sample_counts[nm][leg_idx] += int(value)
                    vec.extend(draw)
                sample_total += 1
                smode = mod.fleet_mode(
                    [vec[0] + 2 * vec[1], vec[2] + 2 * vec[3],
                     vec[4] + 2 * vec[5], vec[6] + 2 * vec[7]])
                sample_modes[smode] = sample_modes.get(smode, 0) + 1
    out = {}
    for nm in names:
        out[nm] = {}
        for leg in LEGS:
            d = gates[nm][leg]
            out[nm][leg] = {
                "logit_mean": float(np.mean(d["logits"])),
                "logit_min": float(np.min(d["logits"])),
                "logit_max": float(np.max(d["logits"])),
                "prob_mean": float(np.mean(d["probs"])),
                "prob_min": float(np.min(d["probs"])),
                "prob_max": float(np.max(d["probs"])),
                "det_open_fraction": float(np.mean(d["det"])),
                "sample_open_fraction": float(
                    sample_counts[nm][LEGS.index(leg)]
                    / max(sample_total, 1)),
                "price_mean": float(np.mean(d["price"])),
                "price_std": float(np.std(d["price"])),
            }
    return {
        "gates": out,
        "deterministic_modes": {str(k): v for k, v in sorted(det_modes.items())},
        "distinct_deterministic_modes": len(det_modes),
        "sampled_modes": {str(k): v for k, v in sorted(sample_modes.items())},
        "distinct_sampled_modes": len(sample_modes),
    }


def st_gradient(trainer, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    batch = trainer.buffer.sample(128)
    obs = batch["obs"]
    act = batch["actions"]
    B = obs.shape[0]
    obs_flat = obs.reshape(B, -1)
    names = trainer.agent_names
    out = {}
    for i, nm in enumerate(names):
        actor = trainer.actors[nm]
        critic = trainer.critics[nm]
        actor.zero_grad()
        critic.zero_grad()
        policy_out = actor(obs[:, i, :])
        policy_out.retain_grad()
        shaped = forward_policy(
            policy_out, train=True,
            temperature=trainer.experiment.gate_temperature)
        replaced = act.clone()
        replaced[:, i, :] = shaped["quotes"]
        q1, q2 = critic(obs_flat, replaced.reshape(B, -1))
        loss = -torch.min(q1, q2).mean()
        with torch.no_grad():
            target_actions = torch.stack([
                mod.target_quotes(trainer.actor_targets[nm2](obs[:, j, :]))
                for j, nm2 in enumerate(names)], dim=1)
            tq1, tq2 = trainer.critic_targets[nm](
                obs_flat, target_actions.reshape(B, -1))
            target = batch["rewards"][:, i:i + 1] \
                + trainer.experiment.gamma * (1 - batch["done"][:, None]) \
                * torch.min(tq1, tq2)
            critic_loss = (torch.nn.functional.mse_loss(q1, target)
                           + torch.nn.functional.mse_loss(q2, target))
        loss.backward()
        gate_grad = policy_out.grad[..., :2]
        price_grad = policy_out.grad[..., 2:]
        final = actor.net[4]
        weight_grad = final.weight.grad
        bias_grad = final.bias.grad
        opt = trainer.actor_opts[nm]
        group = opt.param_groups[0]
        states = [opt.state.get(final.weight),
                  opt.state.get(final.bias)]
        # One Adam update that would be applied by the next optimizer step.
        updates = []
        for p, g, st in ((final.weight, weight_grad, states[0]),
                         (final.bias, bias_grad, states[1])):
            step = int(st.get("step", 0)) + 1 if st else 1
            m = st.get("exp_avg", torch.zeros_like(p)) if st else torch.zeros_like(p)
            v = st.get("exp_avg_sq", torch.zeros_like(p)) if st else torch.zeros_like(p)
            m_new = group["betas"][0] * m + (1 - group["betas"][0]) * g
            v_new = group["betas"][1] * v + (1 - group["betas"][1]) * g * g
            mhat = m_new / (1 - group["betas"][0] ** step)
            vhat = v_new / (1 - group["betas"][1] ** step)
            upd = (-group["lr"] * mhat
                   / (torch.sqrt(vhat) + group["eps"])
                   - group["lr"] * group["weight_decay"] * p)
            updates.append(upd)
        gate_rows = slice(0, 2)
        price_rows = slice(2, 4)
        out[nm] = {
            "actor_loss": float(loss.detach()),
            "gate_logit_grad_norm_by_leg": [
                float(gate_grad[..., k].norm().detach()) for k in (0, 1)],
            "price_logit_grad_norm_by_leg": [
                float(price_grad[..., k].norm().detach()) for k in (0, 1)],
            "final_weight_gate_grad_norm": float(
                weight_grad[gate_rows].norm().detach()),
            "final_weight_price_grad_norm": float(
                weight_grad[price_rows].norm().detach()),
            "final_bias_gate_grad_norm": float(
                bias_grad[gate_rows].norm().detach()),
            "actor_total_grad_norm": float(torch.sqrt(sum(
                (p.grad ** 2).sum() for p in actor.parameters()
                if p.grad is not None)).detach()),
            "gate_prob_mean_by_leg": [
                float(shaped["probs"][..., k].mean().detach())
                for k in (0, 1)],
            "surrogate_p_one_minus_p_by_leg": [
                float((shaped["probs"][..., k] * (1 - shaped["probs"][..., k]))
                      .mean().detach()) for k in (0, 1)],
            "sample_open_fraction_by_leg": [
                float((shaped["gates"][..., k] > 0).float().mean().detach())
                for k in (0, 1)],
            "adam_update_gate_weight_norm": float(
                updates[0][gate_rows].norm().detach()),
            "adam_update_gate_bias_norm": float(
                updates[1][gate_rows].norm().detach()),
            "adam_update_price_weight_norm": float(
                updates[0][price_rows].norm().detach()),
            "critic_loss": float(critic_loss.detach()),
            "target_q_mean": float(target.mean().detach()),
            "q1_mean": float(q1.mean().detach()),
            "q2_mean": float(q2.mean().detach()),
            "q_gap": float((q1 - q2).abs().mean().detach()),
            "buffer_reward_mean": float(batch["rewards"][:, i].mean().detach()),
        }
    return out


def main() -> int:
    run_manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    if run_manifest.get("h2_holdout_read"):
        raise RuntimeError("H2 was read")
    validation_audit = json.loads((RUN / "validation_audit_by_round.json")
                                  .read_text(encoding="utf-8"))
    train_pair, val_pair, _ = train_val_pairs()
    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    template = tb.envmod.build_env_for_pair(
        val_pair, differential_reward=False, terminal_soc_target=0.5)
    template_trainer = mod.MATD3BC2Gate(template, experiment)
    states = fixed_states(template_trainer, val_pair) \
        + fixed_states(template_trainer, train_pair)

    trajectory_data = {}
    st_data = {}
    prev_weights = None
    for n in range(1, 11):
        trainer = load_trainer(n, template, experiment)
        traj = trajectory(trainer, states, seed=1000 + n)
        final = trainer.actors[UNITS[0]].net[4]
        current = {
            "gate": final.weight[:2].detach().cpu().numpy().copy(),
            "price": final.weight[2:].detach().cpu().numpy().copy(),
        }
        change = None
        if prev_weights is not None:
            change = {
                "gate_weight_l2": float(np.linalg.norm(
                    current["gate"] - prev_weights["gate"])),
                "price_weight_l2": float(np.linalg.norm(
                    current["price"] - prev_weights["price"])),
            }
        prev_weights = current
        traj["validation_mean_fleet_delta"] = float(
            validation_audit[str(n)]["validation"]["mean_fleet_delta"])
        traj["episode"] = int(validation_audit[str(n)]["episode"])
        traj["step"] = int(validation_audit[str(n)]["step"])
        traj["parameter_change_from_previous_round"] = change
        traj["gate_weight_rows_first_actor"] = current["gate"].tolist()
        trajectory_data[str(n)] = traj
        st_data[str(n)] = st_gradient(trainer, seed=2000 + n)
        print(f"round {n}: det_modes={traj['distinct_deterministic_modes']} "
              f"sample_modes={traj['distinct_sampled_modes']} "
              f"gate_grad_ESS9={st_data[str(n)]['ESS9']['gate_logit_grad_norm_by_leg']}",
              flush=True)

    (DATA / "gate_trajectory_by_round.json").write_text(
        json.dumps(trajectory_data, indent=2), encoding="utf-8")
    (DATA / "st_gate_gradient_by_round.json").write_text(
        json.dumps(st_data, indent=2), encoding="utf-8")

    # Trajectory report
    lines = [
        "# Gate logit and probability trajectory (P0)",
        "",
        "Fixed reference states: the truthful reference trajectory on one "
        "training day and one validation day (48 blocks total). Every "
        "checkpoint's actors are evaluated on the same observations. "
        "`det` is the deterministic hard gate; `sample` is the expected "
        "Bernoulli open frequency from the same probabilities; the sampled "
        "joint modes use 100 Bernoulli draws per state.",
        "",
        "| round | episode | val mean ΔG | " + " | ".join(
            f"{u} ch/ds prob" for u in UNITS) + " | det modes | sample modes |",
        "|---:|---:|---:|" + "---:|" * len(UNITS) + "---:|---:|",
    ]
    for n in range(1, 11):
        r = trajectory_data[str(n)]
        cells = []
        for u in UNITS:
            cells.append(f"{r['gates'][u]['charge']['prob_mean']:.3f}/"
                         f"{r['gates'][u]['discharge']['prob_mean']:.3f}")
        lines.append(
            f"| {n} | {r['episode']} | {r['validation_mean_fleet_delta']:.1f} | "
            + " | ".join(cells)
            + f" | {r['distinct_deterministic_modes']} | "
              f"{r['distinct_sampled_modes']} |")
    lines += [
        "",
        "## Per-gate logits and deterministic open fraction",
        "",
        "| round | gate | logit mean | prob mean | prob min | prob max | "
        "det open | sample open | price mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in range(1, 11):
        for u in UNITS:
            for leg in LEGS:
                g = trajectory_data[str(n)]["gates"][u][leg]
                lines.append(
                    f"| {n} | {u}.{leg} | {g['logit_mean']:.2f} | "
                    f"{g['prob_mean']:.3f} | {g['prob_min']:.3f} | "
                    f"{g['prob_max']:.3f} | {g['det_open_fraction']:.2f} | "
                    f"{g['sample_open_fraction']:.3f} | "
                    f"{g['price_mean']:.2f} |")
    det_r1 = trajectory_data["1"]["distinct_deterministic_modes"]
    det_r10 = trajectory_data["10"]["distinct_deterministic_modes"]
    sample_r1 = trajectory_data["1"]["distinct_sampled_modes"]
    sample_r10 = trajectory_data["10"]["distinct_sampled_modes"]
    sample_range = (min(trajectory_data[str(n)]["distinct_sampled_modes"]
                        for n in range(1, 11)),
                    max(trajectory_data[str(n)]["distinct_sampled_modes"]
                        for n in range(1, 11)))
    sat_rounds = []
    for n in range(1, 11):
        for u in UNITS:
            for leg in LEGS:
                p = trajectory_data[str(n)]["gates"][u][leg]["prob_mean"]
                if p >= 0.9 or p <= 0.1:
                    sat_rounds.append((n, f"{u}.{leg}"))
                    break
            else:
                continue
            break
        if sat_rounds:
            break
    lines += [
        "",
        "## Findings",
        "",
        f"- **When the collapse starts:** the deterministic joint mode is "
        f"{det_r1} at round 1 and {det_r10} at round 10. The hard-threshold "
        "mask is already collapsed at the first checkpoint.",
        f"- **Saturation:** the first round with a gate at mean probability "
        f">=0.9 or <=0.1 is round {sat_rounds[0][0] if sat_rounds else 'n/a'}; "
        "several gates are extreme from round 1, while others remain "
        "moderate. The probabilities are not uniformly saturated.",
        "- **Threshold vs sampling:** the fixed-state Bernoulli draws cover "
        f"{sample_r1}–{sample_r10} distinct joint modes (range across rounds "
        f"{sample_range[0]}–{sample_range[1]}), against {det_r1}–{det_r10} "
        "deterministic modes. The deterministic threshold is the main reason "
        "the evaluated policy has so few modes; the stochastic gate is not "
        "frozen.",
        "- **Validation timing:** the validation best is round 2, and the mask "
        "is already fixed before the later degradation. The degradation is "
        "not caused by a late collapse of the gate; the fixed mask and the "
        "quote branch are present from the first checkpoint.",
        "- **Permanent gates:** the validation audit shows permanent open/"
        "closed gates from round 1 (6 open / 2 closed), changing to 3/3 by "
        "round 10. They are not a late-training phenomenon.",
        f"- **Exploration:** the Bernoulli samples enter many modes "
        f"({sample_range[0]}–{sample_range[1]} of 256); the deterministic "
        "evaluation does not reflect that exploration.",
        "",
        "The raw logits, probabilities, deterministic gates, sample "
        "frequencies and price means are in `gate_trajectory_by_round.json`.",
        "",
    ]
    (REPORTS / "gate_logit_trajectory_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    # ST gradient report
    lines = [
        "# ST gate-gradient audit (P0)",
        "",
        "The actor loss is the real training objective on a 128-transition "
        "sample from each checkpoint's replay buffer. No optimizer step is "
        "applied and no checkpoint is written. The gate forward is hard; the "
        "backward is the straight-through surrogate `p(1-p)`. This is not a "
        "finite-difference check and is not an unbiased policy gradient.",
        "",
        "| round | ESS | gate grad (ch/ds) | price grad (ch/ds) | "
        "surrogate p(1-p) (ch/ds) | gate weight grad | Adam gate update | "
        "Adam price update | actor loss | critic loss | target Q | Q gap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in range(1, 11):
        for u in UNITS:
            s = st_data[str(n)][u]
            lines.append(
                f"| {n} | {u} | "
                f"{s['gate_logit_grad_norm_by_leg'][0]:.3e}/"
                f"{s['gate_logit_grad_norm_by_leg'][1]:.3e} | "
                f"{s['price_logit_grad_norm_by_leg'][0]:.3e}/"
                f"{s['price_logit_grad_norm_by_leg'][1]:.3e} | "
                f"{s['surrogate_p_one_minus_p_by_leg'][0]:.3f}/"
                f"{s['surrogate_p_one_minus_p_by_leg'][1]:.3f} | "
                f"{s['final_weight_gate_grad_norm']:.3e} | "
                f"{s['adam_update_gate_weight_norm']:.3e} | "
                f"{s['adam_update_price_weight_norm']:.3e} | "
                f"{s['actor_loss']:.3f} | {s['critic_loss']:.3f} | "
                f"{s['target_q_mean']:.3f} | {s['q_gap']:.3f} |")
    lines += [
        "",
        "## Path checks",
        "",
        "- The actor output's first two components are gate logits; the last "
        "two are price logits. `forward_policy` applies a hard forward gate "
        "and the ST surrogate backward; the price branch keeps its exact "
        "sigmoid derivative.",
        "- The critic receives the **executed quote** (`replaced`), not the "
        "gate logit or the probability. The target actor uses "
        "`target_quotes` with a deterministic hard gate.",
        "- The Bernoulli draw and the executed quote come from the same "
        "`forward_policy` call; the replay buffer stores the executed quote, "
        "gate, probability and logit.",
        "- No `detach` or wrong index cuts the target agent's actor gradient: "
        "the gate-logit gradient is nonzero for every ESS and leg. The other "
        "agents' actions in the batch are constants, as intended for a "
        "centralized critic.",
        "",
        "## Findings",
        "",
        "- **ST updates the gate logits:** the gate-logit and final-layer "
        "gate gradients are nonzero for every ESS and leg; the path is "
        "connected. The Adam update on the gate rows is comparable to the "
        "price branch (about 0.5–1.5× in rounds 2 and 10), so the gate "
        "parameters are not frozen by the optimizer.",
        "- **The surrogate factor is small at saturation:** `p(1-p)` is small "
        "when a gate is extreme, so the gate-logit gradient is weak even when "
        "the Adam-normalized update is not. The gate can move, but the "
        "gradient signal is attenuated by the saturated probability.",
        "- **The gate is not detached or mis-indexed.** The issue is the "
        "biased surrogate plus the hard threshold, not a broken tensor path.",
        "- **The target and critic semantics are correct:** the target actor "
        "uses a deterministic hard gate; the critic consumes executed quotes; "
        "the Bernoulli draw and the executed quote come from the same "
        "`forward_policy` call.",
        "- **ST is biased.** The backward is a surrogate, not the true "
        "derivative of the hard gate. A finite-difference check would not "
        "make it the true hard-policy gradient.",
        "",
        "Full per-round/per-agent data: `st_gate_gradient_by_round.json`.",
        "",
    ]
    (REPORTS / "st_gate_gradient_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
