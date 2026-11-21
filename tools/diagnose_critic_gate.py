"""P0/P1 diagnosis: critic values on a fixed quote grid and real clearing.

Read-only. For the best and final checkpoints on one training day and one
validation day, this varies one ESS's quote/gate at a fixed state, records the
critic Q1/Q2/min and the critic gradients, and runs the actual window clearing
for the flagged ESS17/ESS14 cases. No optimizer step and no checkpoint write.
"""

from __future__ import annotations

import copy
import csv
import json
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
HIGHS = {"charge": 150.0, "discharge": 300.0}
FOCUS = (("ESS17", "discharge"), ("ESS14", "charge"), ("ESS14", "discharge"))


def read_csv(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pairs():
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


def load_trainer(round_no, template, experiment):
    trainer = mod.MATD3BC2Gate(template, experiment)
    trainer.load(ROOTS / f"round_{round_no:03d}")
    return trainer


def state_at_block(trainer, pair, block=10):
    env = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    for b in range(block):
        acts = {nm: np.asarray(trainer.select_action(
            obs[nm], nm, train=False)["quotes"], dtype=float)
            for nm in names}
        obs, _, done, _ = env.step(acts)
        if done:
            raise RuntimeError("episode ended before the audit block")
    baseline = {nm: np.asarray(trainer.select_action(
        obs[nm], nm, train=False)["quotes"], dtype=float)
        for nm in names}
    return env, obs, baseline


def clear_grid(env, obs, baseline, target, leg, quote):
    names = list(baseline)
    actions = {nm: baseline[nm].copy() for nm in names}
    target_action = np.zeros(2)
    target_action[LEGS.index(leg)] = quote
    actions[target] = target_action
    clone = copy.deepcopy(env)
    _, _, _, info = clone.step(actions)
    result = clone._last_result
    if result is None:
        return None
    profit = info.get("raw_profit", {})
    sched = (result.get("schedules") or {}).get(target, {})
    n = 4
    p_ch = float(np.sum(np.asarray(sched.get("p_ch", []))[:n]))
    p_dis = float(np.sum(np.asarray(sched.get("p_dis", []))[:n]))
    return {
        "target_payoff": float(profit.get(target, 0.0)),
        "fleet_payoff": float(sum(float(v) for v in profit.values())),
        "target_p_ch_mwh": p_ch * 0.25,
        "target_p_dis_mwh": p_dis * 0.25,
        "accepted": bool((p_ch + p_dis) > 1e-9),
        "g_policy_total": float(sum(float(v) for v in profit.values())),
    }


def critic_grid(trainer, obs, baseline, target, leg, quote):
    names = list(baseline)
    actions = {nm: baseline[nm].copy() for nm in names}
    target_action = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        target_action[LEGS.index(leg)] = quote
    # The no-grad assignment above would break the graph; rebuild as a leaf.
    target_action = torch.tensor(
        [quote if i == LEGS.index(leg) else 0.0 for i in range(2)],
        dtype=torch.float32, requires_grad=True)
    action_t = torch.tensor(
        np.stack([actions[nm] for nm in names]), dtype=torch.float32)
    full = action_t.clone()
    full[names.index(target)] = target_action
    obs_t = torch.tensor(np.stack([obs[nm] for nm in names]),
                         dtype=torch.float32).reshape(1, -1)
    critic = trainer.critics[target]
    critic.zero_grad()
    q1, q2 = critic(obs_t, full.reshape(1, -1))
    qmin = torch.min(q1, q2).mean()
    qmin.backward()
    grad = float(target_action.grad[LEGS.index(leg)].detach())
    # Gate-logit proxy: the same critic, the target actor's gate logit, and a
    # price logit that produces the requested quote. This is the ST surrogate,
    # not the true hard-gate derivative.
    actor = trainer.actors[target]
    with torch.no_grad():
        out = actor(torch.tensor(obs[target], dtype=torch.float32).unsqueeze(0))
    policy_out = out.clone().detach()
    high = HIGHS[leg]
    if quote <= 0:
        price_logit = -20.0
    elif quote >= high:
        price_logit = 20.0
    else:
        p = quote / high
        price_logit = float(np.log(p / (1.0 - p)))
    policy_out[0, 2 + LEGS.index(leg)] = price_logit
    policy_out.requires_grad_(True)
    torch.manual_seed(7)
    shaped = forward_policy(
        policy_out, train=True,
        temperature=trainer.experiment.gate_temperature)
    full2 = full.detach().clone()
    full2[names.index(target)] = shaped["quotes"][0]
    q1b, q2b = critic(obs_t, full2.reshape(1, -1))
    torch.min(q1b, q2b).mean().backward()
    gate_proxy = policy_out.grad[0, LEGS.index(leg)]
    return {
        "q1": float(q1.mean().detach()),
        "q2": float(q2.mean().detach()),
        "q_min": float(qmin.detach()),
        "twin_gap": float((q1 - q2).abs().mean().detach()),
        "dq_dquote": grad,
        "gate_logit_proxy_grad": float(gate_proxy.detach()),
        "gate_prob": float(shaped["probs"][0, LEGS.index(leg)].detach()),
        "gate_sample": int(shaped["gates"][0, LEGS.index(leg)].item()
                           > 0),
    }


def main() -> int:
    manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    if manifest.get("h2_holdout_read"):
        raise RuntimeError("H2 was read")
    experiment = mod.ExperimentConfig(**manifest["experiment"])
    train_pair, val_pair, _ = pairs()
    template = tb.envmod.build_env_for_pair(
        val_pair, differential_reward=False, terminal_soc_target=0.5)
    rows = []
    for label, round_no in (("best", 2), ("final", 10)):
        trainer = load_trainer(round_no, template, experiment)
        for day_label, pair in (("train", train_pair), ("val", val_pair)):
            env, obs, baseline = state_at_block(trainer, pair)
            for target in UNITS:
                for leg in LEGS:
                    high = HIGHS[leg]
                    for gate_label, quote in (
                            ("closed", 0.0),
                            ("open_0", 0.0),
                            ("open_25", 0.25 * high),
                            ("open_50", 0.5 * high),
                            ("open_75", 0.75 * high),
                            ("open_100", 1.0 * high)):
                        q = critic_grid(trainer, obs, baseline,
                                        target, leg, quote)
                        row = {
                            "checkpoint": label, "round": round_no,
                            "day": day_label, "pair_id":
                                f"{pair['load_date']}|{pair['price_date']}",
                            "target": target, "leg": leg,
                            "gate": gate_label, "quote": quote, **q}
                        if (target, leg) in FOCUS:
                            c = clear_grid(env, obs, baseline,
                                           target, leg, quote)
                            if c is not None:
                                row.update(c)
                        rows.append(row)
    out = DATA / "critic_gate_action_grid.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Compact report for the flagged agents.
    lines = [
        "# Critic gate/action value audit (P0)",
        "",
        "Fixed state: block 10 of one training and one validation day. Other "
        "agents keep the checkpoint's deterministic quotes. The quote grid "
        "varies only the target leg; the other target leg is 0. The critic "
        "values are offline; the settlement columns are actual window "
        "clearings. They are not a full policy counterfactual.",
        "",
        "| ckpt | day | target | leg | gate | quote | Q1 | Q2 | min Q | "
        "twin gap | dQ/dquote | gate proxy grad | target payoff | fleet payoff | "
        "accepted |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if (r["target"], r["leg"]) not in FOCUS:
            continue
        lines.append(
            f"| {r['checkpoint']} | {r['day']} | {r['target']} | {r['leg']} | "
            f"{r['gate']} | {r['quote']:.1f} | {r['q1']:.3f} | "
            f"{r['q2']:.3f} | {r['q_min']:.3f} | {r['twin_gap']:.3f} | "
            f"{r['dq_dquote']:.3e} | {r['gate_logit_proxy_grad']:.3e} | "
            + (f"{r.get('target_payoff', float('nan')):.2f} | "
               f"{r.get('fleet_payoff', float('nan')):.2f} | "
               f"{r.get('accepted', '')} |" if "target_payoff" in r
               else "nan | nan | |"))
    lines += [
        "",
        "## How to read it",
        "",
        "- A rising `min Q` with quote while the actual target/fleet payoff "
        "falls is evidence that the critic prefers a quote the market does not "
        "reward.",
        "- A near-zero `dQ/dquote` means the critic is locally flat in the "
        "quote; a large positive/negative value means it actively ranks the "
        "quote.",
        "- The gate proxy gradient is the straight-through surrogate at the "
        "target actor's gate logit. It is not a true hard-gate derivative.",
        "- The settlement is a one-block rolling-window clearing at the fixed "
        "state; it is the correct economic comparison for this state, not a "
        "claim about a full-day policy.",
        "",
        "Full grid for all four ESS: `critic_gate_action_grid.csv`.",
        "",
    ]
    (REPORTS / "critic_gate_action_value_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote {out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
