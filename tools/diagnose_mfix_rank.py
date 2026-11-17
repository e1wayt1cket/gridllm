"""MFIX step-2 read-only diagnostic: WINTER-RANK-style critic ranking check.

NON_FORMAL. Approved 2026-11-10 (MFIX task book §3.1). For each formal
frozen run's best (r1) checkpoint, on Train days only: at selected (day,
block) states, perturb one ESS's discharge quote across a 7-point grid while
all other quotes follow the frozen eval policy; for each grid point record
(a) the twin critics' Q at the perturbed joint action, (b) the target ESS's
true remaining-day raw profit from a full re-clear (perturbation applied at
that block only). Reports per-cell Spearman(Q, payoff), top-action
agreement and twin gap, aggregated per seed against the pre-registered
thresholds (Spearman >= 0.8 and top rate >= 0.7 trustworthy; < 0.5 triggers
the C3 pause rule). No training, no parameter updates, no writes outside
reports/algorithm_diagnostics/.

Charge-leg extension is a documented follow-up; the discharge leg carries
the dominant economics (formal profit concentrated near the 300 edge).
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

import rl_da_gate as mod
import v2_cy_jn_env as envmod
import train_da_gate as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "mfix_critic_rank.json"
MD_PATH = OUT / "mfix_critic_rank.md"
SEEDS = (11, 23)
BLOCKS = (8, 18)
GRID_POINTS = 7
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LEG = 1  # discharge
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"
SPEARMAN_OK, SPEARMAN_BAD = 0.8, 0.5
TOP_OK, TOP_BAD = 0.7, 0.5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def build_trainer(seed: int, train: list):
    run = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}_best"
    saved = json.loads((ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
                        / "run_manifest.json").read_text(encoding="utf-8"))
    da_cfg = dict(saved["da_config"])
    da_cfg.pop("experiment_id", None)
    da_cfg.pop("reward_mode", None)
    template_env = envmod.build_env_for_pair(
        train[0], differential_reward=False, terminal_soc_target=0.5)
    experiment = mod.DAExperimentConfig("DIAG_RANK", "B_C2", **da_cfg)
    trainer = mod.DAGateTrainer(template_env, experiment)
    bundle = trainer.load(run)
    return trainer, int(bundle["manifest"]["step"])


def pick_state_days(train: list) -> list:
    """Up to 3 evenly spaced Train days per season (deterministic), so the
    state count (days x BLOCKS) lands in the pre-registered 10-20 range."""
    by_season = {}
    for p in train:
        by_season.setdefault(p.get("season", ""), []).append(p)
    out = []
    for season in sorted(by_season):
        days = by_season[season]
        idx = np.linspace(0, len(days) - 1, min(3, len(days))).astype(int)
        out.extend(days[i] for i in sorted(set(idx)))
    return out


def policy_quotes(trainer, obs) -> dict:
    return {nm: np.asarray(
        trainer.select_action(obs[nm], nm, train=False)["quotes"],
        dtype=float) for nm in trainer.agent_names}


def replay_payoff(trainer, pair, block_b, target, leg, grid_val,
                  prefix_env=None, prefix_obs=None):
    """True remaining-day raw profit of `target` with its leg quote at
    `block_b` overridden to grid_val; policy elsewhere. Reuses the deep-copied
    prefix env when given (state at start of block_b)."""
    if prefix_env is not None:
        env = copy.deepcopy(prefix_env)
        obs = {k: v.copy() for k, v in prefix_obs.items()}
        start = block_b
    else:
        env = envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        obs = env.reset()
        start = 0
    payoff, cleared = 0.0, True
    for block in range(start, 24):
        acts = policy_quotes(trainer, obs)
        if block == block_b:
            acts[target][leg] = grid_val
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            cleared = False
            break
        if block >= block_b:
            payoff += float(info.get("raw_profit", {}).get(target, 0.0))
        obs = next_obs
        if done:
            break
    return payoff, cleared


def main() -> int:
    t0 = time.time()
    import torch
    doc = json.loads((ROOT / "data/processed/v2_cy_jn_2025"
                      / "common_split.json").read_text(encoding="utf-8"))
    view = doc["near"]
    near_rows = T.read_csv(T.NEAR_PAIRING)
    full_rows = T.read_csv(T.FULL_PAIRING)
    nbp = {f"{r['load_date']}|{r['price_date']}": r for r in near_rows}
    fbp = {f"{r['load_date']}|{r['price_date']}": r for r in full_rows}
    train = [{"load_date": nbp[pid]["load_date"],
              "price_date": fbp[pid]["price_date"],
              "season": nbp[pid].get("season", ""), "pair_id": pid}
             for pid in view["train"]]
    state_days = pick_state_days(train)
    print("state days:", [p["pair_id"] for p in state_days], flush=True)

    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "method": {"blocks": list(BLOCKS), "grid_points": GRID_POINTS,
                         "leg": "discharge", "checkpoint": "best (r1)",
                         "thresholds": {"spearman_ok": SPEARMAN_OK,
                                        "spearman_bad": SPEARMAN_BAD,
                                        "top_ok": TOP_OK, "top_bad": TOP_BAD}},
              "per_seed": {}}
    for seed in SEEDS:
        trainer, step = build_trainer(seed, train)
        print(f"seed{seed}: loaded best checkpoint step={step}", flush=True)
        cells = []
        for pair in state_days:
            for block_b in BLOCKS:
                # One shared prefix per (day, block): replay blocks 0..b-1,
                # then deepcopy the env so each grid point only re-clears the
                # suffix. Fallback: prefix_env=None re-replays from reset.
                env = envmod.build_env_for_pair(
                    pair, differential_reward=False, terminal_soc_target=0.5)
                obs = env.reset()
                for block in range(block_b):
                    acts = policy_quotes(trainer, obs)
                    obs, _, done, _ = env.step(acts)
                    if env._last_result is None or done:
                        break
                obs_b = {k: v.copy() for k, v in obs.items()}
                try:
                    prefix = copy.deepcopy(env)
                except Exception:
                    prefix = None
                base_quotes = policy_quotes(trainer, obs_b)
                obs_flat = torch.as_tensor(np.concatenate(
                    [obs_b[nm] for nm in trainer.agent_names])[None, :],
                    dtype=torch.float32)
                hi = float(mod.LEG_HIGHS[LEG])
                grid = np.linspace(0.0, hi, GRID_POINTS)
                for target in trainer.agent_names:
                    q1s, q2s, pays = [], [], []
                    n_clear_fail = 0
                    for g in grid:
                        act = np.stack([base_quotes[nm]
                                        for nm in trainer.agent_names])
                        act[trainer.agent_names.index(target), LEG] = g
                        with torch.no_grad():
                            q1, q2 = trainer.critics[target](
                                obs_flat, torch.as_tensor(
                                    act.reshape(1, -1), dtype=torch.float32))
                        pay, ok = replay_payoff(
                            trainer, pair, block_b, target, LEG, float(g),
                            prefix_env=prefix, prefix_obs=obs_b)
                        q1s.append(float(q1)); q2s.append(float(q2))
                        pays.append(pay if ok else np.nan)
                        n_clear_fail += 0 if ok else 1
                    pays_arr = np.asarray(pays, dtype=float)
                    valid = ~np.isnan(pays_arr)
                    qmin = np.minimum(np.asarray(q1s), np.asarray(q2s))
                    if valid.sum() >= 3:
                        sp = spearman(qmin[valid], pays_arr[valid])
                        top_q = int(np.nanargmax(
                            np.where(valid, qmin, -np.inf)))
                        top_p = int(np.nanargmax(pays_arr))
                        top_agree = bool(top_q == top_p)
                    else:
                        sp, top_agree = float("nan"), False
                    cells.append({
                        "pair_id": pair["pair_id"], "block": block_b,
                        "target": target, "spearman": sp,
                        "top_agree": top_agree,
                        "twin_gap_mean": float(np.mean(
                            np.abs(np.asarray(q1s) - np.asarray(q2s)))),
                        "clear_failures": n_clear_fail,
                        "grid": [float(g) for g in grid],
                        "q_min": [float(v) for v in qmin],
                        "payoff": [None if np.isnan(p) else float(p)
                                   for p in pays_arr]})
                    print(f"seed{seed} {pair['pair_id']} b{block_b} "
                          f"{target}: spearman={sp:.3f} "
                          f"top={'Y' if top_agree else 'n'} "
                          f"clearfail={n_clear_fail}", flush=True)
        sps = [c["spearman"] for c in cells if not np.isnan(c["spearman"])]
        top_rate = float(np.mean([c["top_agree"] for c in cells]))
        med_sp = float(np.median(sps)) if sps else float("nan")
        verdict = ("TRUSTWORTHY" if med_sp >= SPEARMAN_OK
                   and top_rate >= TOP_OK
                   else "C3_EARLY" if (med_sp < SPEARMAN_BAD
                                       or top_rate < TOP_BAD)
                   else "PARTIAL")
        report["per_seed"][str(seed)] = {
            "checkpoint_step": step, "n_cells": len(cells),
            "median_spearman": med_sp, "top_agree_rate": top_rate,
            "mean_twin_gap": float(np.mean(
                [c["twin_gap_mean"] for c in cells])),
            "total_clear_failures": sum(c["clear_failures"] for c in cells),
            "verdict": verdict, "cells": cells}
        print(f"seed{seed}: median_spearman={med_sp:.3f} "
              f"top_rate={top_rate:.2f} verdict={verdict}", flush=True)

    verdicts = {v["verdict"] for v in report["per_seed"].values()}
    report["merged_verdict"] = (verdicts.pop() if len(verdicts) == 1
                                else "MIXED:" + "/".join(sorted(verdicts)))
    report["runtime_s"] = round(time.time() - t0, 1)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# MFIX Critic 排序诊断(WINTER-RANK 式,只读)", "",
             f"label: {LABEL}; merged: **{report['merged_verdict']}**", "",
             "| seed | cells | 中位 Spearman | top 一致率 | 平均 twin gap "
             "| 清算失败 | verdict |", "|---|---|---|---|---|---|---|"]
    for seed, r in report["per_seed"].items():
        lines.append(
            f"| {seed} | {r['n_cells']} | {r['median_spearman']:.3f} "
            f"| {r['top_agree_rate']:.2f} | {r['mean_twin_gap']:.4f} "
            f"| {r['total_clear_failures']} | {r['verdict']} |")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"merged": report["merged_verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
