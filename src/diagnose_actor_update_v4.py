"""P1: v3 quote-map regression and in-memory actor single-step diagnostic.

Read-only with respect to every original artifact. Checkpoints are opened
through `checkpoint_loader_v2` and restored into an in-memory trainer; the
actor is copied again from a snapshot between arms. The critic, the critic
target, the replay buffer and the market configuration are never written.

The actor step reproduces the actor branch of `MATD3BC2Gate.update()` and
nothing else: no critic step, no target soft update, no `remember()`, no
`save()`. A full-day rollout is evaluated with the same deterministic path as
`train_da_gate.eval_day`; the rollout is checked against `eval_day` on one day
before any arm is run.

The empirical gate draw is part of the training-mode map, so each arm's batch
is sampled once and reused for its 1, 5 or 10 updates. Five replicates vary
only the sampled batch.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import audit_phase20_action_routing as audit
import da_quote_map as qmap
import rl_b_c2_gate as base
import rl_da_gate as mod
import train_da_gate as T
import v2_cy_jn_env as envmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
THRESHOLDS = OUT / "thresholds.json"
THRESHOLDS_SHA256 = (
    "12549ac67a879f1bec4de4e49c082839ab0abd6b02e43ff254029f569ab54592")
CSV_PATH = OUT / "P1_v4_actor_update_results.csv"
EVIDENCE = OUT / "P1_v4_evidence.json"

CHECKPOINTS = {
    11: ROOT / "runs" / "V4D_SHORT_seed11_best",
    23: ROOT / "runs" / "V4D_SHORT_seed23_best",
}
ARMS = (0, 1, 5, 10)
REPLICATES = 5
BATCH = 128
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")

CSV_COLUMNS = (
    "seed", "arm", "replicate", "pair_id", "season",
    "G_policy", "delta_vs_arm0", "delta_vs_b1",
    *[f"G_{nm}" for nm in NAMES], *[f"delta_vs_arm0_{nm}" for nm in NAMES],
    "terminal_soc_max_abs_error", "charge_mwh", "discharge_mwh", "cycles",
    "gate_mean_probability", "quote_mean_charge", "quote_mean_discharge",
    "gate_saturation_upper", "gate_saturation_lower",
    "soc_min", "soc_max",
    *[f"soc_path_{nm}" for nm in NAMES],
    "q_eval_before", "q_eval_after", "q_train_mean_step",
    "grad_norm_gate", "grad_norm_price", "grad_cos_gate_price",
    "grad_norm_actor", "updates_applied",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(f"{path.relative_to(root).as_posix()}:"
                           f"{sha256_file(path)}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def env_versions() -> dict:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gurobi": list(__import__("gurobipy").gurobi.version()),
        "cwd": os.getcwd(),
    }


def jsonable(value):
    """Plain JSON types; the sampler metadata carries numpy scalars."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach())
        return value.detach().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump_evidence(evidence: dict) -> None:
    """Write after every completed arm, so a kill keeps the finished ones."""
    EVIDENCE.write_text(json.dumps(jsonable(evidence), indent=2),
                        encoding="utf-8")


# ----------------------------------------------------------------------
# Train pairs, fixed before measurement
# ----------------------------------------------------------------------
def train_pairs() -> dict:
    full = {f"{r['load_date']}|{r['price_date']}": r
            for r in T.read_csv(T.FULL_PAIRING)}
    near = {f"{r['load_date']}|{r['price_date']}": r
            for r in T.read_csv(T.NEAR_PAIRING)}
    doc = json.loads((ROOT / T.COMMON_SPLIT).read_text(encoding="utf-8"))
    split = doc["near"]
    pairs = {}
    for pid in split["train"]:
        row = near[pid]
        pairs[pid] = {"load_date": row["load_date"],
                      "price_date": full[pid]["price_date"],
                      "season": row.get("season", ""),
                      "pair_id": pid}
    return {"pairs": pairs, "full": full, "mechanism_hash":
            json.loads(Path(mod.B1_MANIFEST).read_text(encoding="utf-8"))
            ["mechanism_hash"], "train_ids": split["train"],
            "validation_ids": split["validation"],
            "test_ids": split["test"]}


def select_pairs(all_pairs: dict) -> list:
    """Season-stratified, pre-registered: 3 summer, 3 autumn, 2 winter."""
    chosen = []
    for season, count in (("summer", 3), ("autumn", 3), ("winter", 2)):
        ids = sorted(pid for pid, p in all_pairs.items()
                     if p["season"] == season)
        if len(ids) < count:
            raise RuntimeError(f"not enough {season} pairs")
        chosen.extend(ids[:count])
    return chosen


# ----------------------------------------------------------------------
# P1-A: mapping regression
# ----------------------------------------------------------------------
def route_source_checks() -> dict:
    src = inspect.getsource(mod.DAGateTrainer._policy_action)
    target = inspect.getsource(mod.DAGateTrainer._target_quotes)
    actor_loss = inspect.getsource(mod.DAGateTrainer._actor_loss_quotes)
    update = inspect.getsource(base.MATD3BC2Gate.update)
    return {
        "execution_uses_forward_policy_v3": "forward_policy_v3(" in src,
        "actor_update_uses_forward_policy_v3":
            "forward_policy_v3(" in actor_loss,
        "target_actor_uses_target_quotes_v3":
            "target_quotes_v3(" in target,
        "critic_actor_action_uses_actor_loss_map":
            "_actor_loss_quotes(policy_out)" in update,
        "critic_target_action_uses_target_quotes":
            "_target_quotes(self.actor_targets" in update,
        "v2_map_absent_from_da_module":
            "forward_policy(" not in inspect.getsource(mod),
    }


def map_tests() -> dict:
    z = torch.tensor([-10.0, -2.5, -1.0, 0.0, 1.0, 2.5, 10.0],
                     dtype=torch.float64)
    g = qmap.map_fraction(z)
    dg = qmap.map_derivative(z)
    zz = z.clone().requires_grad_(True)
    qmap.map_fraction(zz).sum().backward()
    autograd_ok = bool(torch.allclose(zz.grad, dg, atol=1e-10))

    fd = {}
    for point in z.tolist():
        h = 1e-6
        num = float((qmap.map_fraction(torch.tensor([point + h],
                                                    dtype=torch.float64))
                     - qmap.map_fraction(torch.tensor([point - h],
                                                      dtype=torch.float64)))
                    / (2 * h))
        ana = float(qmap.map_derivative(torch.tensor([point],
                                                     dtype=torch.float64)))
        fd[point] = {"finite_difference": num, "map_derivative": ana,
                     "abs_error": abs(num - ana)}

    # Closed gate: the quote is exactly zero and the closed leg's price logit
    # receives no gradient because the gate multiplies it. In train mode the
    # straight-through term gives every gate logit p(1-p)/tau * price_quote.
    logits = torch.tensor([[-2.0, 3.0, 1.0, 1.0]], dtype=torch.float64,
                          requires_grad=True)
    out = qmap.forward_policy_v3(logits, train=False)
    closed_quote = float(out["quotes"][0, 0])
    out["quotes"].sum().backward()
    grad = logits.grad[0].tolist()
    open_price_grad = grad[3]

    train_logits = torch.tensor([[0.4, -0.3, 1.5, -1.5]],
                                dtype=torch.float64, requires_grad=True)
    torch.manual_seed(11)
    train_out = qmap.forward_policy_v3(train_logits, train=True,
                                       gate_temperature=2.0)
    train_out["quotes"].sum().backward()
    # The straight-through term is `probs - probs.detach()` on a logit
    # scaled by the gate temperature, so the surrogate carries 1/tau. In v4
    # the gate logits first pass the soft bound B*tanh(raw/B), so the
    # surrogate gradient also carries the bound's derivative
    # 1 - tanh(raw/B)^2 on each gate logit.
    bound_deriv = 1.0 - torch.tanh(
        train_logits[:, :2] / qmap.GATE_LOGIT_BOUND) ** 2
    expected = ((train_out["probs"] * (1.0 - train_out["probs"])
                 / 2.0 * train_out["price_quotes"]) * bound_deriv)
    gate_grad_tol = float((train_logits.grad[:, :2] - expected).abs().max())
    gate_price_quote = train_out["price_quotes"][0, 0]

    # Straight-through gate: same logits and the same torch seed give the
    # same draw, hence the same quotes.
    policy = torch.tensor([[0.4, -0.3, 1.5, -1.5],
                           [-0.6, 0.9, 0.2, 2.9]], dtype=torch.float64)
    torch.manual_seed(11)
    a = qmap.forward_policy_v3(policy, train=True, gate_temperature=2.0)
    torch.manual_seed(11)
    b = qmap.forward_policy_v3(policy, train=True, gate_temperature=2.0)
    same_gate = bool(torch.equal(a["gates"], b["gates"]))
    same_quotes = bool(torch.equal(a["quotes"], b["quotes"]))

    # Eval map: forward_policy_v3(train=False), target_quotes_v3 and
    # critic_action_v3 are the same object.
    eval_quotes = qmap.forward_policy_v3(policy, train=False)["quotes"]
    target_quotes = qmap.target_quotes_v3(policy)
    critic_quotes = qmap.critic_action_v3(policy)

    trainer = mod.DAGateTrainer(audit.fake_env(), mod.DAExperimentConfig(),
                                agent_names=list(NAMES))
    obs = np.zeros(12, dtype=np.float32)
    with torch.no_grad():
        direct = qmap.forward_policy_v3(
            trainer.actors["ESS9"](torch.zeros(1, 12)), train=False)
    execution = trainer.select_action(obs, "ESS9", train=False)
    torch.manual_seed(11)
    loss_map = trainer._actor_loss_quotes(policy)
    torch.manual_seed(11)
    direct_loss = qmap.forward_policy_v3(
        policy, train=True,
        gate_temperature=trainer.current_gate_temperature())["quotes"]

    return {
        "finite_difference": fd,
        "finite_difference_ok": all(v["abs_error"] < 1e-5
                                    for v in fd.values()),
        "autograd_matches_map_derivative": autograd_ok,
        "closed_gate_quote_exactly_zero": closed_quote == 0.0,
        "closed_gate_price_logit_grad_zero": grad[2] == 0.0,
        "closed_gate_open_leg_price_grad": open_price_grad,
        "train_gate_grad_matches_st": gate_grad_tol < 1e-10,
        "train_gate_grad_max_abs_error": gate_grad_tol,
        "train_price_quote_leg0": float(gate_price_quote),
        "same_seed_same_gate": same_gate,
        "same_seed_same_quotes": same_quotes,
        "target_equals_eval_map": bool(torch.equal(eval_quotes,
                                                   target_quotes)),
        "critic_equals_eval_map": bool(torch.equal(eval_quotes,
                                                   critic_quotes)),
        "execution_equals_direct_map": bool(np.allclose(
            execution["quotes"], direct["quotes"].squeeze(0).numpy())),
        "actor_loss_map_equals_direct_v3": bool(torch.equal(loss_map,
                                                            direct_loss)),
        "actor_loss_map_gate_temperature":
            trainer.current_gate_temperature(),
    }


def mapping_regression() -> dict:
    routing = audit.routing_checks()
    update = audit.update_call_audit()
    routes = route_source_checks()
    tests = map_tests()
    checks = dict(routes)
    checks.update({
        "routing_ess_order": (
            routing["ess_order_expected"] == routing["ess_order_trainer"]
            == routing["ess_order_real_env"]),
        "routing_actor_output_4": routing["actor_output_dim"] == 4,
        "routing_quote_dim_2": routing["quote_dim"] == 2,
        "routing_critic_input_56": routing["critic_input_dim"] == 56,
        "routing_closed_gate_zero": routing["closed_charge_quote"] == 0.0,
        "routing_target_v3": routing["target_route_equals_v3"],
        "routing_finite_difference": all(
            v <= 5e-2 for v in routing["finite_difference_abs_error"].values()),
        "v2_not_called_in_update": update["v2_calls_in_update"] == 0,
        "v3_called_in_update": update["v3_calls_in_update"] > 0,
        "actor_loss_finite": update["actor_loss_finite"],
        "map_finite_difference": tests["finite_difference_ok"],
        "map_autograd": tests["autograd_matches_map_derivative"],
        "closed_gate_zero": tests["closed_gate_quote_exactly_zero"],
        "closed_gate_price_grad_zero":
            tests["closed_gate_price_logit_grad_zero"],
        "closed_gate_st_grad": tests["train_gate_grad_matches_st"],
        "same_seed_same_gate": tests["same_seed_same_gate"],
        "same_seed_same_quotes": tests["same_seed_same_quotes"],
        "target_equals_eval_map": tests["target_equals_eval_map"],
        "critic_equals_eval_map": tests["critic_equals_eval_map"],
        "execution_equals_direct_map": tests["execution_equals_direct_map"],
        "actor_loss_map_equals_direct": tests["actor_loss_map_equals_direct_v3"],
    })
    return {"checks": {k: bool(v) for k, v in checks.items()},
            "all_pass": all(checks.values()),
            "routing": routing, "update_audit": update, "map_tests": tests}


# ----------------------------------------------------------------------
# Rollout: eval_day plus the SOC path
# ----------------------------------------------------------------------
def rollout_day(trainer, pair, b1_cache, *, mechanism_hash, pair_row) -> dict:
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    block_payoffs = {nm: [] for nm in names}
    soc_path = {nm: [float(next(a for a in env.all_agents
                                if a.name == nm).storage.soc0)]
                for nm in names}
    charge_mwh = {nm: 0.0 for nm in names}
    discharge_mwh = {nm: 0.0 for nm in names}
    gate_open = {nm: [0, 0] for nm in names}
    gate_prob_sum = {nm: [0.0, 0.0] for nm in names}
    quote_open = {nm: [[], []] for nm in names}
    sat_upper = sat_lower = 0
    soc_min = {nm: math.inf for nm in names}
    soc_max = {nm: -math.inf for nm in names}
    decisions = {nm: 0 for nm in names}
    for block in range(24):
        acts = {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            gates = np.asarray(d["gates"], dtype=float)
            probs = np.asarray(d["probs"], dtype=float)
            price_quotes = np.asarray(d["price_quotes"], dtype=float)
            decisions[nm] += 1
            for leg in (0, 1):
                gate_prob_sum[nm][leg] += float(probs[leg])
                if gates[leg] > 0:
                    gate_open[nm][leg] += 1
                    quote_open[nm][leg].append(float(price_quotes[leg]))
                    if price_quotes[leg] >= 0.99 * mod.LEG_HIGHS[leg]:
                        sat_upper += 1
                    if price_quotes[leg] <= 0.01 * mod.LEG_HIGHS[leg]:
                        sat_lower += 1
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            raise RuntimeError(f"{pair['pair_id']}: clearing failed at "
                               f"block {block}")
        result = env._last_result
        n_commit = min(4, env.T - block * 4)
        for nm in names:
            block_payoffs[nm].append(
                float(info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(
                float(env.prev_soc.get(nm, soc_path[nm][-1])))
            ws = (result.get("schedules") or {}).get(nm)
            if ws is None:
                continue
            charge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_ch"])[:n_commit]) * 0.25)
            discharge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_dis"])[:n_commit]) * 0.25)
            soc_arr = ws.get("soc")
            if soc_arr is not None:
                vals = np.asarray(soc_arr, dtype=float)[:n_commit + 1]
                if vals.size:
                    soc_min[nm] = min(soc_min[nm], float(np.min(vals)))
                    soc_max[nm] = max(soc_max[nm], float(np.max(vals)))
        obs = next_obs
        if done:
            break
    pair_id = pair["pair_id"]
    b1_day = b1_cache.get(pair_id, 0.5, mechanism_hash,
                          mod.pair_data_hash(pair_row))
    G_pi = {nm: float(sum(block_payoffs[nm])) for nm in names}
    G_b1 = {nm: float(sum(b1_day.block_payoffs[nm])) for nm in names}
    terminal = {nm: soc_path[nm][-1] for nm in names}
    e_max = {nm: float(next(a for a in env.all_agents
                            if a.name == nm).storage.e_max) for nm in names}
    result = env._last_result
    voltage = np.asarray(result.get("bus_voltage", np.zeros(0)), dtype=float)
    util = np.asarray(result.get("line_utilization", np.zeros(0)),
                      dtype=float)
    return {
        "pair_id": pair_id,
        "season": pair.get("season", ""),
        "G_policy": G_pi,
        "G_b1": G_b1,
        "delta": {nm: G_pi[nm] - G_b1[nm] for nm in names},
        "delta_fleet": float(sum(G_pi[nm] - G_b1[nm] for nm in names)),
        "terminal_soc": terminal,
        "soc_path": soc_path,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "charge_mwh": charge_mwh,
        "discharge_mwh": discharge_mwh,
        "cycles": {nm: (charge_mwh[nm] + discharge_mwh[nm])
                   / (2.0 * e_max[nm]) for nm in names},
        "gate_open_frequency": {
            nm: [gate_open[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "gate_mean_probability": {
            nm: [gate_prob_sum[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "quote_mean": {nm: [float(np.mean(quote_open[nm][leg]))
                            if quote_open[nm][leg] else 0.0
                            for leg in (0, 1)] for nm in names},
        "gate_saturation_upper": sat_upper,
        "gate_saturation_lower": sat_lower,
        "voltage_violation_cells": int(
            ((voltage < env.config.network.v_min_pu - 1e-7)
             | (voltage > env.config.network.v_max_pu + 1e-7)).sum())
        if voltage.size else 0,
        "line_overload_cells": int((util > 1.0 + 1e-7).sum())
        if util.size else 0,
    }


def rollout_check(trainer, triples, mechanism_hash, b1_cache) -> dict:
    """The rollout must reproduce eval_day on the same day."""
    pid, pair, row = triples[0]
    a = rollout_day(trainer, pair, b1_cache,
                    mechanism_hash=mechanism_hash, pair_row=row)
    b = T.eval_day(trainer, pair, b1_cache,
                   mechanism_hash=mechanism_hash, pair_row=row)
    diffs = {}
    for field in ("G_policy", "G_b1", "delta", "charge_mwh", "discharge_mwh",
                  "cycles", "soc_min", "soc_max", "gate_mean_probability",
                  "gate_open_frequency"):
        values = []
        for key, right in b[field].items():
            left = a[field][key]
            if isinstance(right, list):
                values.append(float(np.max(np.abs(
                    np.asarray(left) - np.asarray(right)))))
            elif isinstance(right, dict):
                continue
            else:
                values.append(abs(float(left) - float(right)))
        diffs[field] = max(values) if values else 0.0
    diffs["gate_saturation_upper"] = abs(
        a["gate_saturation_upper"] - b["gate_saturation_upper"])
    diffs["gate_saturation_lower"] = abs(
        a["gate_saturation_lower"] - b["gate_saturation_lower"])
    return {"pair_id": pid, "max_abs_diff": diffs,
            "agrees_to_1e_6": all(v <= 1e-6 for v in diffs.values())}


# ----------------------------------------------------------------------
# Actor-step machinery
# ----------------------------------------------------------------------
def clone_state(trainer) -> dict:
    import copy
    return {
        "actors": {nm: {k: v.detach().clone()
                        for k, v in trainer.actors[nm].state_dict().items()}
                   for nm in trainer.agent_names},
        # state_dict() shares the Adam moment tensors; a step would mutate
        # the snapshot in place without the deep copy.
        "actor_opts": {nm: copy.deepcopy(
            trainer.actor_opts[nm].state_dict())
            for nm in trainer.agent_names},
    }


def restore_state(trainer, snap: dict) -> None:
    import copy
    for nm in trainer.agent_names:
        trainer.actors[nm].load_state_dict(snap["actors"][nm])
        # Optimizer.load_state_dict keeps references to the incoming tensors,
        # so a step would mutate the snapshot's Adam moments in place; the
        # copy makes every restore start from the recorded moments.
        trainer.actor_opts[nm].load_state_dict(
            copy.deepcopy(snap["actor_opts"][nm]))


def param_fingerprint(modules) -> dict:
    return {name: {k: v.detach().clone()
                   for k, v in module.state_dict().items()}
            for name, module in modules.items()}


def fingerprints_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for name in a:
        if set(a[name]) != set(b[name]):
            return False
        for key in a[name]:
            if not torch.equal(a[name][key], b[name][key]):
                return False
    return True


def eval_q(trainer, agent: str, obs, act, obs_flat) -> float:
    i = trainer.agent_names.index(agent)
    actor = trainer.actors[agent]
    with torch.no_grad():
        policy_out = actor(obs[:, i, :])
        replaced = act.clone()
        replaced[:, i, :] = qmap.target_quotes_v3(policy_out)
        q1, q2 = trainer.critics[agent](
            obs_flat, replaced.reshape(obs.shape[0], -1))
        return float(torch.min(q1, q2).mean())


def actor_update(trainer, agent: str, obs, act, obs_flat) -> dict:
    i = trainer.agent_names.index(agent)
    actor = trainer.actors[agent]
    critic = trainer.critics[agent]
    policy_out = actor(obs[:, i, :])
    replaced = act.clone()
    replaced[:, i, :] = trainer._actor_loss_quotes(policy_out)
    q1, q2 = critic(obs_flat, replaced.reshape(obs.shape[0], -1))
    loss = -torch.min(q1, q2).mean()
    g_out, = torch.autograd.grad(loss, policy_out, retain_graph=True)
    gate_grad = g_out[:, :2].flatten()
    price_grad = g_out[:, 2:].flatten()
    cos = float(torch.nn.functional.cosine_similarity(
        gate_grad.unsqueeze(0), price_grad.unsqueeze(0)).item())
    trainer.actor_opts[agent].zero_grad()
    loss.backward()
    grad_norm = float(torch.sqrt(sum(
        p.grad.pow(2).sum() for p in actor.parameters()
        if p.grad is not None)))
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    trainer.actor_opts[agent].step()
    with torch.no_grad():
        policy_after = actor(obs[:, i, :])
    return {
        "loss": float(loss.detach()),
        "q_train": float(torch.min(q1, q2).mean().detach()),
        "grad_norm_gate": float(gate_grad.norm()),
        "grad_norm_price": float(price_grad.norm()),
        "grad_cos_gate_price": cos,
        "grad_norm_actor": grad_norm,
        "price_logit_abs_mean": float(policy_after[:, 2:].abs().mean()),
    }


def run_arm(trainer, snap, obs, act, obs_flat, steps: int,
            record: list) -> dict:
    restore_state(trainer, snap)
    q_before = {nm: eval_q(trainer, nm, obs, act, obs_flat)
                for nm in trainer.agent_names}
    steps_log = []
    for step in range(steps):
        for nm in trainer.agent_names:
            row = actor_update(trainer, nm, obs, act, obs_flat)
            row.update({"step": step + 1, "agent": nm})
            steps_log.append(row)
            record.append(row)
    q_after = {nm: eval_q(trainer, nm, obs, act, obs_flat)
               for nm in trainer.agent_names}
    return {"steps_applied": steps,
            "q_eval_before": q_before, "q_eval_after": q_after,
            "step_log": steps_log}


def paired_delta(days_a: dict, days_b: dict) -> float:
    return float(sum(days_a[k]["G_policy"] - days_b[k]["G_policy"]
                     for k in days_a))


def bootstrap_ci(values, rng, resamples=10000, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return [float(arr.mean()), float(arr.mean())]
    draws = rng.choice(arr, size=(resamples, arr.size), replace=True)
    means = draws.mean(axis=1)
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


def classify(per_arm: dict) -> str:
    """User-directed vocabulary; effect size and CI from thresholds.json."""
    def mean(arm):
        return per_arm[arm]["mean"]

    def ci(arm):
        return per_arm[arm]["ci"]

    def sig_neg(arm):
        return mean(arm) <= -1.0 and ci(arm)[1] < 0.0

    def sig_pos(arm):
        return mean(arm) >= 1.0 and ci(arm)[0] > 0.0

    if sig_pos(1) and sig_neg(10):
        return "SHORT_TERM_IMPROVEMENT_LONG_TERM_DEGRADATION"
    if mean(10) <= mean(5) <= mean(1) and sig_neg(10):
        return "CUMULATIVE_DEGRADATION"
    if sig_neg(1):
        return "IMMEDIATE_DEGRADATION"
    if sig_pos(1) and sig_pos(10):
        return "CONSISTENT_IMPROVEMENT"
    if all(ci(arm)[0] <= 0.0 <= ci(arm)[1] for arm in (1, 5, 10)):
        return "NO_SIGNIFICANT_CHANGE"
    return "INCONCLUSIVE"


def day_row(seed, arm, rep, pid, day, base_day, stats, applied) -> dict:
    d = day["G_policy"]
    d0 = base_day["G_policy"]
    step_log = (stats or {}).get("step_log") or []
    mean = (lambda key: float(np.mean([s[key] for s in step_log])))
    return {
        "seed": seed, "arm": arm, "replicate": rep,
        "pair_id": pid, "season": day["season"],
        "G_policy": float(sum(d.values())),
        "delta_vs_arm0": float(sum(d[nm] - d0[nm] for nm in NAMES)),
        "delta_vs_b1": day["delta_fleet"],
        **{f"G_{nm}": d[nm] for nm in NAMES},
        **{f"delta_vs_arm0_{nm}": float(d[nm] - d0[nm]) for nm in NAMES},
        "terminal_soc_max_abs_error": max(
            abs(v - 0.5) for v in day["terminal_soc"].values()),
        "charge_mwh": float(sum(day["charge_mwh"].values())),
        "discharge_mwh": float(sum(day["discharge_mwh"].values())),
        "cycles": float(sum(day["cycles"].values())),
        "gate_mean_probability": float(np.mean(
            [v for nm in NAMES for v in day["gate_mean_probability"][nm]])),
        "quote_mean_charge": float(np.mean(
            [day["quote_mean"][nm][0] for nm in NAMES])),
        "quote_mean_discharge": float(np.mean(
            [day["quote_mean"][nm][1] for nm in NAMES])),
        "gate_saturation_upper": day["gate_saturation_upper"],
        "gate_saturation_lower": day["gate_saturation_lower"],
        "soc_min": float(min(day["soc_min"].values())),
        "soc_max": float(max(day["soc_max"].values())),
        **{f"soc_path_{nm}": ";".join(
            f"{v:.4f}" for v in day["soc_path"][nm]) for nm in NAMES},
        "q_eval_before": (float(np.mean(list(
            stats["q_eval_before"].values()))) if stats else None),
        "q_eval_after": (float(np.mean(list(
            stats["q_eval_after"].values()))) if stats else None),
        "q_train_mean_step": mean("q_train") if step_log else None,
        "grad_norm_gate": mean("grad_norm_gate") if step_log else None,
        "grad_norm_price": mean("grad_norm_price") if step_log else None,
        "grad_cos_gate_price": mean("grad_cos_gate_price")
        if step_log else None,
        "grad_norm_actor": mean("grad_norm_actor") if step_log else None,
        "updates_applied": applied,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256_file(THRESHOLDS) != THRESHOLDS_SHA256:
        print("thresholds.json changed; refusing to run", flush=True)
        return 2
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    if tuple(thresholds["p1"]["step_arms"]) != (1, 5, 10):
        print("thresholds arms differ from the pre-registration", flush=True)
        return 2

    data = train_pairs()
    ids = select_pairs(data["pairs"])
    if sorted(ids) != sorted(thresholds["p1"]["train_pairs"]):
        print(f"pair selection differs from the pre-registration: {ids}",
              flush=True)
        return 2
    triples = [(pid, data["pairs"][pid], data["full"][pid]) for pid in ids]

    evidence = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": env_versions(),
        "thresholds": {"path": str(THRESHOLDS.relative_to(ROOT)),
                       "sha256": sha256_file(THRESHOLDS),
                       "arms": [0, 1, 5, 10],
                       "replicates": REPLICATES, "pairs": ids,
                       "deviation": ("thresholds.json pre-registers arms "
                                     "1/5/10; step 0 is the measured "
                                     "baseline and the CONSISTENT_IMPROVEMENT "
                                     "class is the user-directed addition"),
                       },
        "mapping": mapping_regression(),
    }
    if "--mapping-only" in sys.argv:
        print(json.dumps({"mapping_all_pass":
                          evidence["mapping"]["all_pass"],
                          "checks": evidence["mapping"]["checks"]},
                         indent=2))
        return 0 if evidence["mapping"]["all_pass"] else 1
    if not evidence["mapping"]["all_pass"]:
        evidence["verdict"] = "NO_GO_P1_MAPPING"
        dump_evidence(evidence)
        print(json.dumps({"verdict": evidence["verdict"]}))
        return 1

    b1_cache = mod.load_b1_cache()
    mechanism_hash = data["mechanism_hash"]
    csv_fh = CSV_PATH.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_fh.flush()

    def emit_row(row):
        writer.writerow(row)
        csv_fh.flush()

    evidence["verdict"] = "PARTIAL_IN_PROGRESS"
    guard_trees = {str(p.relative_to(ROOT)): tree_hash(p)
                   for p in CHECKPOINTS.values()}

    for seed, ckpt_root in CHECKPOINTS.items():
        template_pair = triples[0][1]
        template_env = envmod.build_env_for_pair(
            template_pair, differential_reward=False,
            terminal_soc_target=0.5)
        trainer = mod.DAGateTrainer(template_env,
                                    mod.DAExperimentConfig("diagnostic"))
        bundle = trainer.load(ckpt_root)
        manifest = bundle["manifest"]
        provenance = {
            "checkpoint": str(ckpt_root.relative_to(ROOT)),
            "step": manifest["step"],
            "run_id": manifest["run_id"],
            "actor_loss_map": manifest.get("da_gate", {}).get(
                "actor_loss_map"),
            "quote_map": manifest.get("specs", {}).get(
                "action_spec", {}).get("quote_map"),
            "obs_spec": manifest.get("specs", {}).get("obs_spec", {}).get(
                "name"),
            "action_spec": manifest.get("specs", {}).get(
                "action_spec", {}).get("name"),
            "episode": manifest["counters"]["episode"],
            "buffer_len": len(trainer.buffer),
            "gate_temperature": trainer.current_gate_temperature(),
        }
        print(f"seed {seed}: loaded {provenance['checkpoint']} "
              f"step {provenance['step']} episode {provenance['episode']} "
              f"tau {provenance['gate_temperature']}", flush=True)
        check = rollout_check(trainer, triples, mechanism_hash, b1_cache)
        provenance["rollout_matches_eval_day"] = check["agrees_to_1e_6"]
        if not check["agrees_to_1e_6"]:
            evidence[f"seed{seed}"] = {"provenance": provenance,
                                       "rollout_check": check}
            evidence["verdict"] = "NO_GO_P1_REPRODUCIBILITY"
            dump_evidence(evidence)
            print(json.dumps({"verdict": evidence["verdict"]}))
            return 1
        print(f"seed {seed}: rollout matches eval_day (max abs diff "
              f"{max(check['max_abs_diff'].values()):.3g})", flush=True)

        snap = clone_state(trainer)
        critic_snap = param_fingerprint({
            **{f"critic_{nm}": m for nm, m in trainer.critics.items()},
            **{f"target_{nm}": m for nm, m in trainer.critic_targets.items()},
        })
        target_actor_snap = param_fingerprint({
            f"actor_target_{nm}": m
            for nm, m in trainer.actor_targets.items()})
        buffer_len_before = len(trainer.buffer)

        baseline = {pid: rollout_day(trainer, pair, b1_cache,
                                     mechanism_hash=mechanism_hash,
                                     pair_row=row)
                    for pid, pair, row in triples}
        raw = {0: {}, 1: {}, 5: {}, 10: {}}
        raw_unit = {arm: {nm: {} for nm in NAMES} for arm in (0, 1, 5, 10)}
        for pid, _, _ in triples:
            raw[0][pid] = [0.0]
            for nm in NAMES:
                raw_unit[0][nm][pid] = [0.0]
            emit_row(day_row(seed, 0, 0, pid, baseline[pid],
                             baseline[pid], None, 0))
        print(f"seed {seed}: arm 0 baseline done on {len(triples)} days "
              f"(fleet {sum(sum(d['G_policy'].values()) for d in baseline.values()):.1f} "
              f"CNY)", flush=True)
        arm_steps = {}
        evidence[f"seed{seed}"] = {"provenance": provenance,
                                   "rollout_check": check,
                                   "buffer_len_before": buffer_len_before,
                                   "arm_steps": arm_steps}

        for arm in (1, 5, 10):
            for pid, _, _ in triples:
                raw[arm][pid] = []
                for nm in NAMES:
                    raw_unit[arm][nm][pid] = []
            for rep in range(REPLICATES):
                random.seed(10000 * seed + 100 * arm + rep)
                np.random.seed(10000 * seed + 100 * arm + rep)
                torch.manual_seed(10000 * seed + 100 * arm + rep)
                batch = trainer.buffer.sample(BATCH)
                obs, act = batch["obs"], batch["actions"]
                obs_flat = obs.reshape(obs.shape[0], -1)
                steps = []
                stats = run_arm(trainer, snap, obs, act, obs_flat, arm,
                                steps)
                arm_steps[f"{arm}_{rep}"] = {
                    "q_eval_before": stats["q_eval_before"],
                    "q_eval_after": stats["q_eval_after"],
                    "grad_norm_actor_mean": float(np.mean(
                        [s["grad_norm_actor"] for s in stats["step_log"]])),
                    "grad_norm_gate_mean": float(np.mean(
                        [s["grad_norm_gate"] for s in stats["step_log"]])),
                    "grad_norm_price_mean": float(np.mean(
                        [s["grad_norm_price"] for s in stats["step_log"]])),
                    "grad_cos_mean": float(np.mean(
                        [s["grad_cos_gate_price"]
                         for s in stats["step_log"]])),
                    "q_train_mean": float(np.mean(
                        [s["q_train"] for s in stats["step_log"]])),
                    "sampler_stats": batch.get("sampler_stats"),
                    "step_log": stats["step_log"],
                }
                for pid, pair, row in triples:
                    day = rollout_day(trainer, pair, b1_cache,
                                      mechanism_hash=mechanism_hash,
                                      pair_row=row)
                    d = day["G_policy"]
                    d0 = baseline[pid]["G_policy"]
                    raw[arm][pid].append(float(sum(
                        d[nm] - d0[nm] for nm in NAMES)))
                    for nm in NAMES:
                        raw_unit[arm][nm][pid].append(float(d[nm] - d0[nm]))
                    emit_row(day_row(seed, arm, rep, pid, day,
                                     baseline[pid], stats, arm))
                restore_state(trainer, snap)
                rep_mean = float(np.mean([v for pid in raw[arm]
                                          for v in raw[arm][pid]]))
                print(f"seed {seed} arm {arm} rep {rep}: fleet delta so far "
                      f"{rep_mean:+.2f} CNY/day", flush=True)
                dump_evidence(evidence)

        guard_trees[str(ckpt_root.relative_to(ROOT)) + "_after"] = tree_hash(
            ckpt_root)
        evidence[f"seed{seed}"] = {
            "provenance": provenance,
            "rollout_check": check,
            "buffer_len_before": buffer_len_before,
            "buffer_len_after": len(trainer.buffer),
            "critic_params_unchanged": fingerprints_equal(
                critic_snap, param_fingerprint({
                    **{f"critic_{nm}": m
                       for nm, m in trainer.critics.items()},
                    **{f"target_{nm}": m
                       for nm, m in trainer.critic_targets.items()}})),
            "actor_target_params_unchanged": fingerprints_equal(
                target_actor_snap, param_fingerprint({
                    f"actor_target_{nm}": m
                    for nm, m in trainer.actor_targets.items()})),
            "arm_steps": arm_steps,
            "summary": {},
        }
        rng = np.random.default_rng(20261108)
        for arm in (1, 5, 10):
            values = [v for pid in raw[arm] for v in raw[arm][pid]]
            fleet_mean = float(np.mean(values))
            ci = bootstrap_ci(values, rng)
            unit = {}
            for nm in NAMES:
                unit_values = [v for pid in raw_unit[arm][nm]
                               for v in raw_unit[arm][nm][pid]]
                unit[nm] = {"mean": float(np.mean(unit_values)),
                            "ci": bootstrap_ci(unit_values, rng)}
            evidence[f"seed{seed}"]["summary"][str(arm)] = {
                "fleet_mean": fleet_mean, "fleet_ci": ci,
                "per_unit_mean": {nm: unit[nm]["mean"] for nm in NAMES},
                "per_unit_ci": {nm: unit[nm]["ci"] for nm in NAMES},
                "day_means": {pid: float(np.mean(raw[arm][pid]))
                              for pid in raw[arm]},
                "days_better": int(sum(1 for pid in raw[arm]
                                       if np.mean(raw[arm][pid]) > 0)),
                "n_values": len(values),
            }
        per_arm_summary = {
            arm: {"mean": evidence[f"seed{seed}"]["summary"][str(arm)]
                  ["fleet_mean"],
                  "ci": evidence[f"seed{seed}"]["summary"][str(arm)][
                      "fleet_ci"]}
            for arm in (1, 5, 10)}
        classification = classify(per_arm_summary)
        evidence[f"seed{seed}"]["classification"] = classification
        means = {arm: round(s["mean"], 2) for arm, s in per_arm_summary.items()}
        print(f"seed {seed}: {classification} fleet means {means}",
              flush=True)
        dump_evidence(evidence)

    verdicts = [evidence[f"seed{s}"]["classification"] for s in CHECKPOINTS]
    evidence["overall_classification"] = (
        verdicts[0] if len(set(verdicts)) == 1 else "INCONCLUSIVE")
    evidence["verdict"] = "PASS_P1_ACTOR_DIAGNOSTIC"
    if not all(evidence[f"seed{s}"]["critic_params_unchanged"]
               and evidence[f"seed{s}"]["actor_target_params_unchanged"]
               and evidence[f"seed{s}"]["buffer_len_before"]
               == evidence[f"seed{s}"]["buffer_len_after"]
               for s in CHECKPOINTS):
        evidence["verdict"] = "NO_GO_P1_REPRODUCIBILITY"
    for path, original in guard_trees.items():
        if not path.endswith("_after"):
            if guard_trees.get(path + "_after") != original:
                evidence["verdict"] = "NO_GO_P1_REPRODUCIBILITY"
    evidence["checkpoint_trees_unchanged"] = all(
        guard_trees.get(path + "_after") == original
        for path, original in list(guard_trees.items())
        if not path.endswith("_after"))
    evidence["elapsed_seconds"] = round(time.time() - t_all, 1)
    csv_fh.close()
    dump_evidence(evidence)
    print(json.dumps({
        "verdict": evidence["verdict"],
        "overall_classification": evidence["overall_classification"],
        "per_seed": {str(s): evidence[f"seed{s}"]["classification"]
                     for s in CHECKPOINTS},
        "checkpoint_trees_unchanged":
            evidence["checkpoint_trees_unchanged"],
        "elapsed_seconds": evidence["elapsed_seconds"],
    }, indent=2))
    return 0 if evidence["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
