"""Offline and one-day end-to-end validation for MATD3_DA_GATE_v2.

Read-only with respect to the existing run: it writes only a temporary
checkpoint and the validation JSON. It checks the D1 map, the A1 gate schedule
and ST path, the D2 sampler probabilities/fields and a complete 24-block day,
then a v3 save/restore and the v2 refusal.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import da_quote_map as qm
import rl_da_gate as mod
import v2_cy_jn_env as envmod
from diagnose_critic_long_horizon import pairs
from rl_b_c2_gate import pair_data_hash

DATA = Path("data/processed/v2_cy_jn_2025")
ROOT = Path.cwd()
TRAIN_PID = "2022-08-14|2025-08-17"
UNIT = ("ESS9", "ESS14", "ESS17", "ESS27")


def check(condition, message, out):
    out.append({"ok": bool(condition), "check": message})
    if not condition:
        raise AssertionError(message)


def mapping_checks(out):
    z = np.linspace(-100.0, 100.0, 400001)
    g = qm.map_fraction_np(z)
    d = qm.map_derivative_np(z)
    check(np.all(np.diff(g) >= -1e-15), "map monotone", out)
    check(np.all(g >= 0.0) and np.all(g <= 1.0), "map bounded", out)
    check(np.all(np.isfinite(g)) and np.all(np.isfinite(d)),
          "map and derivative finite", out)
    check(abs(float(qm.map_fraction_np(np.array([0.0]))[0]) - 0.5) < 1e-15,
          "map(0)=0.5", out)
    for point in (-10.0, -3.1, -2.5, -1.0, 0.0, 1.0, 2.5, 3.1, 10.0):
        h = 1e-6
        num = (float(qm.map_fraction_np(np.array([point + h]))[0])
               - float(qm.map_fraction_np(np.array([point - h]))[0])) / (2 * h)
        ana = float(qm.map_derivative_np(np.array([point]))[0])
        tol = 1e-5 if abs(abs(point) - qm.Z_CORE) < 1e-12 else 1e-8
        check(abs(num - ana) < tol, f"numeric derivative at {point}", out)
    # Gate-closed exact zero and gate-open legal quote, both legs.
    closed = qm.forward_policy_v3(
        torch.tensor([[-10.0, -10.0, 100.0, 100.0]]), train=False)
    check(float(closed["quotes"].sum()) == 0.0,
          "gate-closed quote exactly zero", out)
    opened = qm.forward_policy_v3(
        torch.tensor([[10.0, 10.0, 100.0, 100.0]]), train=False)
    check(float(opened["quotes"][0, 0]) <= 150.0
          and float(opened["quotes"][0, 1]) <= 300.0,
          "gate-open quote within bounds", out)
    # High-region gradient versus the old sigmoid.
    def sig_grad(frac):
        x = math.log(frac / (1 - frac))
        return 300.0 * frac * (1 - frac)
    region = np.linspace(0.86, 0.997, 200)
    logits = np.array([_inverse(region[i]) for i in range(len(region))])
    grad_c = qm.map_derivative_np(logits) * 300.0
    grad_a = np.array([sig_grad(f) for f in region])
    check(grad_c.min() > grad_a.min(),
          "high-region minimum gradient exceeds sigmoid", out)
    grid = []
    for frac in (0.5, 258.0 / 300.0, 299.0 / 300.0, 299.9 / 300.0,
                 0.01, 0.99):
        zz = _inverse(frac)
        grid.append({"fraction": frac, "quote_cny": frac * 300.0,
                     "logit": zz,
                     "dq_dlogit_cny": float(qm.map_derivative_np(
                         np.array([zz]))[0] * 300.0),
                     "sigmoid_dq_dlogit_cny": sig_grad(frac)})
    return {
        "grid": grid,
        "high_region_min_gradient_cny": float(grad_c.min()),
        "sigmoid_high_region_min_gradient_cny": float(grad_a.min()),
        "high_region_ratio_min": float(grad_c.min() / grad_a.min()),
        "logit_for_258": _inverse(258.0 / 300.0),
        "logit_for_299": _inverse(299.0 / 300.0),
    }


def _inverse(frac, lo=-20.0, hi=20.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(qm.map_fraction_np(np.array([mid]))[0]) < frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gate_checks(out):
    cfg = mod.DAExperimentConfig("CHECK", "B_C2", gate_warmup_episodes=106)
    seq = {str(ep): cfg.gate_temperature_at(ep)
           for ep in (0, 26, 53, 79, 106, 159)}
    check(abs(seq["0"] - 4.0) < 1e-12, "gate temperature starts at 4", out)
    check(abs(seq["53"] - 2.5) < 1e-9, "gate temperature midpoint 2.5", out)
    check(abs(seq["106"] - 1.0) < 1e-12, "gate temperature ends at 1", out)
    check(abs(seq["159"] - 1.0) < 1e-12, "gate temperature stays 1", out)
    # ST gradient at a saturated and a moderate logit, τ=1 vs τ=4.
    grads = {}
    for logit in (-4.0, 0.0, 4.0, 8.0):
        for tau in (1.0, 4.0):
            policy = torch.tensor([[logit, 0.0, 1.0, 1.0]],
                                  requires_grad=True)
            fp = qm.forward_policy_v3(policy, train=True,
                                      gate_temperature=tau)
            fp["quotes"].sum().backward()
            grads[f"{logit}_{tau}"] = float(policy.grad[0, 0])
    check(all(math.isfinite(v) for v in grads.values()),
          "ST gate gradients finite", out)
    check(abs(grads["4.0_4.0"]) > abs(grads["4.0_1.0"]),
          "temperature raises the ST gradient at a saturated logit", out)
    # Deterministic evaluation ignores the temperature and uses the sign.
    off = qm.forward_policy_v3(torch.tensor([[0.5, -0.5, 2.0, 2.0]]),
                               train=False)["gates"]
    check(float(off[0, 0]) == 1.0 and float(off[0, 1]) == 0.0,
          "deterministic gate is logit>0", out)
    # Bernoulli frequency.
    policy = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    torch.manual_seed(0)
    opens = 0
    for _ in range(4000):
        opens += int(qm.forward_policy_v3(
            policy, train=True, gate_temperature=1.0)["gates"].sum())
    freq = opens / 8000.0
    check(abs(freq - 0.5) < 0.03, "Bernoulli open frequency near 0.5", out)
    return {"temperature_sequence": seq, "st_gradients": grads,
            "bernoulli_open_frequency": freq}


def sampler_checks(trainer, out):
    buffer = trainer.buffer
    batch = buffer.sample(min(128, len(buffer)))
    n = len(buffer)
    stats = batch["sampler_stats"]
    check(len(batch["indices"]) == min(128, n), "sample batch size", out)
    check(bool(stats), "sampler stats present", out)
    # Field alignment: every returned row equals the indexed stored row.
    for j, idx in enumerate(batch["indices"]):
        obs, act, rew, nobs, done = buffer.buffer[int(idx)]
        check(np.allclose(batch["obs"][j].numpy(), obs),
              "sampled obs matches buffer index", out)
        check(np.allclose(batch["actions"][j].numpy(), act),
              "sampled action matches buffer index", out)
        check(np.allclose(batch["rewards"][j].numpy(), rew),
              "sampled reward matches buffer index", out)
        break
    # Recompute the sampling probability from the buffer state.
    keys = np.rint(np.asarray(buffer.extras["stratum"], dtype=float)
                   ).reshape(-1).astype(int)
    from da_replay import N_CELLS
    counts = np.bincount(keys, minlength=N_CELLS)
    nonempty = np.flatnonzero(counts > 0)
    w = counts[nonempty].astype(float) ** buffer.config["beta"]
    w /= w.sum()
    p_cell = dict(zip(nonempty.tolist(), w.tolist()))
    p_strat = np.zeros(n)
    for cell in nonempty:
        idx = np.flatnonzero(keys == cell)
        p_strat[idx] = p_cell[int(cell)] / len(idx)
    p_strat = np.minimum(p_strat, buffer.config["max_upweight"] / n)
    p_strat /= p_strat.sum()
    p = ((1 - buffer.config["uniform_fraction"]) / n
         + buffer.config["uniform_fraction"] * p_strat)
    p /= p.sum()
    check(abs(float(p.sum()) - 1.0) < 1e-12,
          "sampling probabilities sum to one", out)
    cap = ((1 - buffer.config["uniform_fraction"]) / n
           + buffer.config["uniform_fraction"]
           * buffer.config["max_upweight"] / n)
    check(float(np.max(p)) <= cap + 1e-12,
          "per-transition probability cap is traceable", out)
    check(stats["ess"] > 0 and stats["ess"] <= n,
          "effective sample size in range", out)
    check(stats["cells_nonempty"] > 0,
          "non-empty joint cells recorded", out)
    check(np.all(batch["sample_prob"] > 0),
          "sampled transition probabilities positive", out)
    check(np.all(batch["uniform_ratio"] > 0.0),
          "uniform-ratio positive", out)
    return {"stats": stats, "n": n,
            "prob_min": float(p.min()), "prob_max": float(p.max()),
            "uniform_prob": 1.0 / n,
            "cells": stats["cells_nonempty"]}


def end_to_end(out):
    pair_map, _ = pairs()
    pair = pair_map[TRAIN_PID]
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    cfg = mod.DAExperimentConfig(
        "OFFLINE_DA", "B_C2", buffer_capacity=5000, batch_size=32,
        exploration_epsilon_start=1.0, exploration_epsilon_end=1.0,
        exploration_decay_episodes=1, gate_warmup_episodes=2)
    trainer = mod.DAGateTrainer(env, cfg)
    b1 = mod.load_b1_cache()
    mech = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]
    prow = next(r for r in csv.DictReader(
        (DATA / "load_price_pairing.csv").open(encoding="utf-8"))
                if f"{r['load_date']}|{r['price_date']}" == TRAIN_PID)
    day = trainer.train_day(env, pair, b1, config_hash_value=mech,
                            data_hash=pair_data_hash(prow))
    checks = day["checks"]
    check(checks["clears"] == 24, "24/24 blocks cleared", out)
    check(checks["all_optimal"], "all clears OPTIMAL", out)
    check(checks["voltage_violation_cells"] == 0
          and checks["line_overload_cells"] == 0,
          "no voltage/line violation", out)
    terminal = max(abs(v - 0.5) for v in checks["terminal_soc"].values())
    check(terminal < 1e-6, "terminal SOC 0.5", out)
    delta_sum = sum(day["b"]["delta"].values())
    g_diff = (sum(day["b"]["G_policy"].values())
              - sum(day["b"]["G_b1"].values()))
    check(abs(delta_sum - g_diff) < 1e-9, "B delta identity", out)
    for nm in UNIT:
        b_path = np.asarray(day["b_reward_by_block"][nm])
        check(np.all(np.abs(b_path[:23]) < 1e-12),
              f"{nm}: B paid only at block 23", out)
        shaped = np.asarray(day["b_reward_by_block"][nm]) + np.asarray(
            day["c2_shaping_by_block"][nm])
        disc = float(np.sum(cfg.gamma ** np.arange(24) * shaped))
        check(abs(disc - day["c2"]["episode_return"][nm]) < 1e-9,
              f"{nm}: C2 discounted identity", out)
        check(abs(day["c2"]["potential_start"][nm]) < 1e-12,
              f"{nm}: C2 potential starts at zero", out)
    # Legal actions and stored fields.
    actions = np.asarray(trainer.buffer.buffer[0][1])
    check(np.all(actions[:, 0] <= 150.0 + 1e-9)
          and np.all(actions[:, 1] <= 300.0 + 1e-9)
          and np.all(actions >= 0.0), "executed quotes legal", out)
    check(np.all((np.asarray(trainer.buffer.extras["gates"]) == 0)
                  | (np.asarray(trainer.buffer.extras["gates"]) == 1)),
          "stored gates binary", out)
    explored = np.asarray(trainer.buffer.extras["explored"])
    check(explored.sum() > 0, "forced exploration visited at least one leg", out)
    check(len(trainer.buffer) == 24, "one day fills 24 transitions", out)
    # The sampler and the exploration source are stored.
    check("exploration_source" in trainer.buffer.extras,
          "exploration source stored", out)
    # Save / restore v3 and compare an open-gate state.
    tmp = Path(tempfile.mkdtemp(prefix="da_v3_"))
    try:
        trainer.save(
            tmp, level="bundle", dataset="NEAR", seed=11,
            config_hash_value="a" * 64,
            data={k: "a" * 64 for k in
                  ("pairing", "bus_loads", "price_curves", "common_split")},
            exposure_log=str(tmp / "exposure.json"),
            eval_provenance="offline")
        manifest = json.loads((tmp / "manifest.json").read_text(
            encoding="utf-8"))
        check(manifest["schema_version"] == "matd3-checkpoint-v3",
              "checkpoint schema v3", out)
        check(manifest["specs"]["action_spec"]["name"] == "storage_quote_v3",
              "checkpoint action schema v3", out)
        check("da_gate" in manifest and "sampler" in manifest,
              "DA metadata and sampler state saved", out)
        check(manifest["buffer"]["saved"], "replay saved in bundle", out)
        fresh = mod.DAGateTrainer(
            envmod.build_env_for_pair(
                pair, differential_reward=False, terminal_soc_target=0.5),
            cfg)
        fresh.load(tmp)
        stored = np.asarray(trainer.buffer.extras["gates"])
        open_idx = int(np.argmax(stored.reshape(len(stored), -1).sum(axis=1)))
        obs_stack = np.asarray(trainer.buffer.buffer[open_idx][0])
        gate_row = stored[open_idx]
        unit_idx, leg_idx = np.unravel_index(
            int(np.argmax(gate_row)), gate_row.shape)
        obs = obs_stack[unit_idx]
        a1 = trainer.select_action(obs, UNIT[unit_idx],
                                   train=False)["quotes"]
        a2 = fresh.select_action(obs, UNIT[unit_idx],
                                 train=False)["quotes"]
        check(np.allclose(a1, a2), "open-gate action round-trips", out)
        check(len(fresh.buffer) == len(trainer.buffer),
              "replay restored after load", out)
    finally:
        shutil.rmtree(tmp)
    # A v2 checkpoint must be refused.
    try:
        fresh.load(ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_audit_roots"
                         "/round_002")
        refused = False
    except Exception:
        refused = True
    check(refused, "v2 checkpoint refused by the v3 trainer", out)
    return {
        "day": {"clears": checks["clears"],
                "all_optimal": checks["all_optimal"],
                "terminal_soc_max_err": terminal,
                "voltage_violation_cells": checks["voltage_violation_cells"],
                "line_overload_cells": checks["line_overload_cells"],
                "delta_fleet": float(delta_sum),
                "explored_legs": int(explored.sum()),
                "buffer_len": len(trainer.buffer)},
        "gate_temperature": day["gate_temperature"],
        "exploration_epsilon": day["exploration_epsilon"],
        "sampler": day["sampler"],
    }


def main(argv=None) -> int:
    out = []
    result = {"checks": out}
    result["mapping"] = mapping_checks(out)
    result["gate"] = gate_checks(out)
    pair_map, _ = pairs()
    env = envmod.build_env_for_pair(
        pair_map[TRAIN_PID], differential_reward=False,
        terminal_soc_target=0.5)
    cfg = mod.DAExperimentConfig("SAMPLER_CHECK", "B_C2",
                                 buffer_capacity=5000, batch_size=64)
    trainer = mod.DAGateTrainer(env, cfg)
    b1 = mod.load_b1_cache()
    mech = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]
    prow = next(r for r in csv.DictReader(
        (DATA / "load_price_pairing.csv").open(encoding="utf-8"))
                if f"{r['load_date']}|{r['price_date']}" == TRAIN_PID)
    for _ in range(2):
        trainer.train_day(env, pair_map[TRAIN_PID], b1,
                          config_hash_value=mech,
                          data_hash=pair_data_hash(prow))
    result["sampler"] = sampler_checks(trainer, out)
    result["end_to_end"] = end_to_end(out)
    result["all_ok"] = all(item["ok"] for item in out)
    (DATA / "da_gate_offline_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": result["all_ok"],
                      "checks": len(out),
                      "failed": [x for x in out if not x["ok"]]}, indent=2))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
