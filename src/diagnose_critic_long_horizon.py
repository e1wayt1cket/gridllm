"""P0 critic long-horizon calibration (Phase 15).

Read-only. For a fixed mid-day state, one ESS/one quote leg is changed, then
the frozen deterministic policy continues to the day end. The future SOC is
not forced to the original path. It records the critic Q, the actual
discounted training return, the actual settlement and the B1 comparison.

No training, no test, no H2, no checkpoint write.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import c2_reward as c2
import rl_b_c2_gate as mod
import train_b_c2_gate as tb

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
WINTER = ("2022-12-13|2025-12-11", "2022-12-30|2025-12-31",
          "2022-12-31|2025-12-28")
CONTROL = ("2022-07-23|2025-07-13", "2022-08-14|2025-08-17")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pairs():
    near = read_csv(NEAR_PAIRING)
    full = read_csv(FULL_PAIRING)
    near_by = {f"{r['load_date']}|{r['price_date']}": r for r in near}
    full_by = {f"{r['load_date']}|{r['price_date']}": r for r in full}
    out = {}
    for pid in set(near_by) | set(full_by):
        row = full_by.get(pid) or near_by.get(pid)
        out[pid] = {"load_date": row["load_date"],
                    "price_date": row["price_date"],
                    "season": row.get("season", "")}
    return out, full_by


def load_trainer(cp, template, experiment):
    trainer = mod.MATD3BC2Gate(template, experiment)
    trainer.load(ROOTS / f"round_{cp:03d}")
    return trainer


def collect_state(trainer, pair, block):
    env = tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    past = {nm: [] for nm in names}
    for b in range(block):
        acts = {nm: np.asarray(trainer.select_action(
            obs[nm], nm, train=False)["quotes"], dtype=float)
            for nm in names}
        next_obs, _, done, info = env.step(acts)
        for nm in names:
            past[nm].append(float(info.get("raw_profit", {}).get(nm, 0.0)))
        obs = next_obs
        if done:
            raise RuntimeError("episode ended before audit block")
    baseline = {nm: np.asarray(trainer.select_action(
        obs[nm], nm, train=False)["quotes"], dtype=float)
        for nm in names}
    soc = {nm: float(env.prev_soc.get(
        nm, next(a for a in env.all_agents if a.name == nm).storage.soc0))
        for nm in names}
    gates, probs, price_logits = {}, {}, {}
    for nm in names:
        with torch.no_grad():
            out = trainer.actors[nm](
                torch.tensor(obs[nm], dtype=torch.float32).unsqueeze(0))[0]
        gates[nm] = [float(x) for x in torch.sigmoid(out[:2])]
        probs[nm] = [int(x) for x in (out[:2] > 0)]
        price_logits[nm] = [float(x) for x in out[2:]]
    price_slice = np.asarray(env.wholesale[block * 4:block * 4 + 16],
                             dtype=float)
    return env, obs, baseline, past, soc, gates, probs, price_logits, price_slice


def critic_q(trainer, obs, baseline, target, leg, quote):
    names = list(baseline)
    actions = {nm: baseline[nm].copy() for nm in names}
    actions[target] = baseline[target].copy()
    actions[target][LEGS.index(leg)] = quote
    obs_t = torch.tensor(np.stack([obs[nm] for nm in names]),
                         dtype=torch.float32).reshape(1, -1)
    action_t = torch.tensor(np.stack([actions[nm] for nm in names]),
                            dtype=torch.float32)
    target_quote = torch.tensor(
        [quote if i == LEGS.index(leg) else float(baseline[target][i])
         for i in range(2)],
        dtype=torch.float32, requires_grad=True)
    full = action_t.clone()
    full[names.index(target)] = target_quote
    critic = trainer.critics[target]
    critic.zero_grad()
    q1, q2 = critic(obs_t, full.reshape(1, -1))
    qmin = torch.min(q1, q2).mean()
    qmin.backward()
    return {
        "q1": float(q1.mean().detach()),
        "q2": float(q2.mean().detach()),
        "q_min": float(qmin.detach()),
        "twin_gap": float((q1 - q2).abs().mean().detach()),
        "dq_dquote": float(target_quote.grad[LEGS.index(leg)].detach()),
    }


def rollout(trainer, env, obs, baseline, target, leg, quote, block,
            past, soc, b1_blocks, b1_G, scale):
    names = list(baseline)
    env2 = copy.deepcopy(env)
    actions = {nm: baseline[nm].copy() for nm in names}
    actions[target] = baseline[target].copy()
    actions[target][LEGS.index(leg)] = quote
    obs2, _, done, info = env2.step(actions)
    future = {nm: [float(info.get("raw_profit", {}).get(nm, 0.0))]
              for nm in names}
    soc_path = {nm: [soc[nm], float(env2.prev_soc.get(nm, soc[nm]))]
                for nm in names}
    charge = {nm: 0.0 for nm in names}
    discharge = {nm: 0.0 for nm in names}
    losses = 0.0
    voltage_cells = line_cells = 0
    accepted = False
    result = env2._last_result
    ws = (result.get("schedules") or {}).get(target, {})
    p_ch = float(np.sum(np.asarray(ws.get("p_ch", []))[:4]))
    p_dis = float(np.sum(np.asarray(ws.get("p_dis", []))[:4]))
    accepted = (p_ch + p_dis) > 1e-9
    for b in range(block + 1, 24):
        acts = {nm: np.asarray(trainer.select_action(
            obs2[nm], nm, train=False)["quotes"], dtype=float)
            for nm in names}
        obs2, _, done, info = env2.step(acts)
        result = env2._last_result
        for nm in names:
            future[nm].append(float(
                info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(float(env2.prev_soc.get(nm, soc_path[nm][-1])))
        if result is not None:
            loss = result.get("line_loss_mw")
            if loss is not None:
                losses += float(np.sum(np.asarray(loss)) * 0.25)
            v = result.get("bus_voltage")
            if v is not None:
                arr = np.asarray(v, dtype=float)
                voltage_cells += int(((arr < env2.config.network.v_min_pu - 1e-7)
                                      | (arr > env2.config.network.v_max_pu + 1e-7)).sum())
            u = result.get("line_utilization")
            if u is not None:
                line_cells += int((np.asarray(u, dtype=float) > 1.0 + 1e-7).sum())
        if done:
            break
    full_G = past[target] + future[target]
    full_delta = float(sum(full_G)) - float(sum(b1_blocks))
    future_payoff = float(sum(future[target]))
    b1_future = float(sum(b1_blocks[block:]))
    cfg = c2.C2Config(gamma=trainer.experiment.gamma)
    shaping = []
    for t in range(block, 24):
        shaping.append(float(c2.shaping(soc_path[target][t - block],
                                        soc_path[target][t - block + 1],
                                        t, cfg)))
    gamma = trainer.experiment.gamma
    disc_shaping = sum((gamma ** (t - block)) * shaping[t - block]
                       for t in range(block, 24))
    training_return_scaled = (
        (gamma ** (23 - block)) * full_delta
        + disc_shaping) / scale
    training_return_cny = training_return_scaled * scale
    discounted_settlement = sum(
        (gamma ** (t - block)) * future[target][t - block]
        for t in range(block, 24))
    terminal_soc = soc_path[target][-1]
    for nm in names:
        if nm == target:
            pass
    return {
        "future_payoff": future_payoff,
        "full_G_policy": float(sum(full_G)),
        "full_delta_cny": full_delta,
        "b1_future_payoff": b1_future,
        "future_minus_b1": future_payoff - b1_future,
        "training_return_scaled": training_return_scaled,
        "training_return_cny": training_return_cny,
        "discounted_settlement_cny": discounted_settlement,
        "terminal_soc": terminal_soc,
        "soc_path": [soc_path[target][i] for i in range(len(soc_path[target]))],
        "c2_shaping": shaping,
        "c2_shaping_sum": float(sum(shaping)),
        "accepted_at_block": accepted,
        "p_ch_mwh_at_block": p_ch * 0.25,
        "p_dis_mwh_at_block": p_dis * 0.25,
        "future_payoffs": future[target],
        "losses_mwh": losses,
        "voltage_violation_cells": voltage_cells,
        "line_overload_cells": line_cells,
    }


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, required=True,
                        choices=(2, 10))
    parser.add_argument("--group", required=True, choices=("winter", "control"))
    parser.add_argument("--days", default="",
                        help="comma-separated pair ids; default all in group")
    parser.add_argument("--blocks", default="")
    parser.add_argument("--targets", default="")
    args = parser.parse_args(argv)
    days = [x for x in args.days.split(",") if x] or \
        list(WINTER if args.group == "winter" else CONTROL)
    blocks = [int(x) for x in args.blocks.split(",") if x] or \
        ([2, 11, 20] if args.group == "winter" else [11])
    targets = [x for x in args.targets.split(",") if x] or \
        (["ESS14", "ESS27"] if args.group == "winter" else ["ESS9", "ESS17"])
    run_manifest = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    if run_manifest.get("h2_holdout_read"):
        raise RuntimeError("H2 was read")
    experiment = mod.ExperimentConfig(**run_manifest["experiment"])
    pair_map, full_by = pairs()
    b1_manifest = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))
    mech = b1_manifest["mechanism_hash"]
    b1_cache = mod.load_b1_cache()
    template = tb.envmod.build_env_for_pair(
        pair_map[days[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainer = load_trainer(args.checkpoint, template, experiment)
    days_tag = "_".join(pid.split("|")[0] for pid in days)
    out_csv = DATA / (f"critic_long_horizon_cp{args.checkpoint}_"
                      f"{args.group}_{days_tag}.csv")
    state_json = DATA / (f"critic_long_horizon_states_cp{args.checkpoint}_"
                         f"{args.group}_{days_tag}.json")
    rows = []
    states = {}
    for pid in days:
        pair = pair_map[pid]
        row = full_by[pid]
        b1_day = b1_cache.get(pid, 0.5, mech, mod.pair_data_hash(row))
        for block in blocks:
            env, obs, baseline, past, soc, gates, probs, plog, price = \
                collect_state(trainer, pair, block)
            state_id = f"{pid}|b{block}|cp{args.checkpoint}"
            states[state_id] = {
                "pair_id": pid, "block": block, "checkpoint": args.checkpoint,
                "soc": soc,
                "gate_probability": gates,
                "gate_deterministic": probs,
                "price_logits": plog,
                "current_quotes": {nm: baseline[nm].tolist() for nm in baseline},
                "wholesale_window": price.tolist(),
                "past_payoffs": {nm: past[nm] for nm in past},
                "data_hashes": run_manifest["data_hashes"],
                "checkpoint_step": int(json.loads(
                    (ROOTS / f"round_{args.checkpoint:03d}/manifest.json")
                    .read_text(encoding="utf-8"))["step"]),
            }
            for target in targets:
                for leg in LEGS:
                    high = HIGHS[leg]
                    candidates = [0.0, 0.25 * high, 0.5 * high,
                                  0.75 * high, high, float(baseline[target][LEGS.index(leg)])]
                    seen = set()
                    for quote in candidates:
                        key = round(float(quote), 9)
                        if key in seen:
                            continue
                        seen.add(key)
                        q = critic_q(trainer, obs, baseline, target, leg, quote)
                        roll = rollout(
                            trainer, env, obs, baseline, target, leg, quote,
                            block, past, soc, b1_day.block_payoffs[target],
                            sum(b1_day.block_payoffs[target]),
                            experiment.fixed_scale)
                        rows.append({
                            "state_id": state_id, "checkpoint": args.checkpoint,
                            "pair_id": pid, "block": block, "target": target,
                            "leg": leg, "quote": float(quote),
                            "actor_quote": float(baseline[target][LEGS.index(leg)]),
                            "q1": q["q1"], "q2": q["q2"], "q_min": q["q_min"],
                            "twin_gap": q["twin_gap"], "dq_dquote": q["dq_dquote"],
                            **{k: roll[k] for k in (
                                "future_payoff", "full_delta_cny",
                                "b1_future_payoff", "future_minus_b1",
                                "training_return_scaled", "training_return_cny",
                                "discounted_settlement_cny", "terminal_soc",
                                "c2_shaping_sum", "accepted_at_block",
                                "p_ch_mwh_at_block", "p_dis_mwh_at_block",
                                "losses_mwh", "voltage_violation_cells",
                                "line_overload_cells")},
                            "future_payoffs": json.dumps(roll["future_payoffs"]),
                            "soc_path": json.dumps(roll["soc_path"]),
                            "c2_shaping": json.dumps(roll["c2_shaping"]),
                        })
            print(f"{state_id}: {len(rows)} rows", flush=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    state_json.write_text(json.dumps(states, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} ({len(rows)} rows) and {state_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
