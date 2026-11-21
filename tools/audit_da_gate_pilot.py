"""Post-pilot diagnostics for the 159-episode MATD3_DA_GATE_v2 run.

Read-only: it loads the selected checkpoint and the validation days, runs the
same-state counterfactual rollouts used in phase 15, and runs the winter
15-minute quote grid. It also reports the buffer's joint-cell and high-quote
coverage. It never reads H2 or the legacy test.

Data are written to `data/processed/v2_cy_jn_2025/da_gate_*`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import rl_da_gate as mod
import train_da_gate as tb
from audit_winter_15min_quote_acceptance import b1_day, policy_block
from da_replay import N_CELLS, stratum_key
from diagnose_critic_long_horizon import (CONTROL, WINTER, collect_state,
                                          critic_q, pairs, rollout, spearman)
from rl_b_c2_gate import pair_data_hash

DATA = Path("data/processed/v2_cy_jn_2025")
RUNS = Path("runs")
RUN_ID = "DEV_DA_GATE_seed11_v2"
UNITS = ("ESS14", "ESS27")
LEGS = ("charge", "discharge")
HIGHS = {"charge": 150.0, "discharge": 300.0}
# The phase-15 material blocks plus the winter-loss blocks the phase-16 audit
# added. All are validation days.
MATERIAL = [
    ("2022-12-13|2025-12-11", 2), ("2022-12-13|2025-12-11", 3),
    ("2022-12-13|2025-12-11", 11), ("2022-12-13|2025-12-11", 12),
    ("2022-12-13|2025-12-11", 17), ("2022-12-13|2025-12-11", 20),
    ("2022-12-30|2025-12-31", 10), ("2022-12-30|2025-12-31", 11),
    ("2022-12-30|2025-12-31", 14), ("2022-12-30|2025-12-31", 20),
    ("2022-12-31|2025-12-28", 0), ("2022-12-31|2025-12-28", 17),
    ("2022-12-31|2025-12-28", 23),
]


def load_run(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text(
        encoding="utf-8"))
    experiment = mod.DAExperimentConfig(**json.loads(
        (RUNS / RUN_ID / "run_manifest.json").read_text(
            encoding="utf-8"))["experiment"])
    pair_map, _ = pairs()
    env = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainer = mod.DAGateTrainer(env, experiment)
    trainer.load(root)
    return trainer, experiment, pair_map


def material_rollouts(trainer, experiment, pair_map, b1_cache, mech,
                      full_by):
    rows = []
    for pid, block in MATERIAL:
        pair = pair_map[pid]
        row = full_by[pid]
        b1_day_obj = b1_cache.get(pid, 0.5, mech, pair_data_hash(row))
        env, obs, baseline, past, soc, gates, probs, plog, price = \
            collect_state(trainer, pair, block)
        for target in UNITS:
            for leg in LEGS:
                high = HIGHS[leg]
                candidates = [0.0, 0.25 * high, 0.5 * high, 0.75 * high,
                              high, float(baseline[target][LEGS.index(leg)])]
                seen = set()
                for quote in candidates:
                    key = round(float(quote), 9)
                    if key in seen:
                        continue
                    seen.add(key)
                    q = critic_q(trainer, obs, baseline, target, leg, quote)
                    roll = rollout(
                        trainer, env, obs, baseline, target, leg, quote,
                        block, past, soc, b1_day_obj.block_payoffs[target],
                        sum(b1_day_obj.block_payoffs[target]),
                        experiment.fixed_scale)
                    rows.append({
                        "checkpoint": "pilot_best", "pair_id": pid,
                        "block": block, "target": target, "leg": leg,
                        "quote": float(quote),
                        "actor_quote": float(
                            baseline[target][LEGS.index(leg)]),
                        "q1": q["q1"], "q2": q["q2"], "q_min": q["q_min"],
                        "twin_gap": q["twin_gap"],
                        "dq_dquote": q["dq_dquote"],
                        "future_payoff": roll["future_payoff"],
                        "full_delta_cny": roll["full_delta_cny"],
                        "b1_future_payoff": roll["b1_future_payoff"],
                        "future_minus_b1": roll["future_minus_b1"],
                        "training_return_scaled":
                            roll["training_return_scaled"],
                        "training_return_cny": roll["training_return_cny"],
                        "c2_shaping_sum": roll["c2_shaping_sum"],
                        "accepted_at_block": roll["accepted_at_block"],
                        "p_ch_mwh_at_block": roll["p_ch_mwh_at_block"],
                        "p_dis_mwh_at_block": roll["p_dis_mwh_at_block"],
                        "terminal_soc": roll["terminal_soc"],
                        "future_payoffs": json.dumps(roll["future_payoffs"]),
                    })
        print(f"material {pid} b{block}: {len(rows)} rows", flush=True)
    return rows


def material_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['pair_id']}|b{r['block']}|{r['target']}|{r['leg']}"].append(r)
    out = []
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["quote"])
        spread = max(r["full_delta_cny"] for r in rs) - \
            min(r["full_delta_cny"] for r in rs)
        out.append({
            "group": key, "pair_id": rs[0]["pair_id"],
            "block": rs[0]["block"], "target": rs[0]["target"],
            "leg": rs[0]["leg"], "actor_quote": rs[0]["actor_quote"],
            "best_quote": max(rs, key=lambda r: r["full_delta_cny"])["quote"],
            "worst_quote": min(rs, key=lambda r: r["full_delta_cny"])["quote"],
            "delta_spread": spread,
            "spearman_q_delta": spearman(
                [r["q_min"] for r in rs],
                [r["full_delta_cny"] for r in rs]),
            "spearman_q_training": spearman(
                [r["q_min"] for r in rs],
                [r["training_return_scaled"] for r in rs]),
            "material": spread > 10.0,
            "flat": spread <= 1.0,
        })
    material = [g for g in out if g["material"]]
    return out, {
        "groups": len(out), "material_groups": len(material),
        "flat_groups": sum(g["flat"] for g in out),
        "mean_spearman_q_delta": float(np.nanmean(
            [g["spearman_q_delta"] for g in out])),
        "median_spearman_q_delta": float(np.nanmedian(
            [g["spearman_q_delta"] for g in out])),
        "material_mean_spearman": float(np.nanmean(
            [g["spearman_q_delta"] for g in material])) if material else None,
        "material_negative": int(sum(
            (g["spearman_q_delta"] or 0) < 0 for g in material)),
        "material_rows": sum(1 for r in rows
                             if any(g["group"] ==
                                    f"{r['pair_id']}|b{r['block']}|"
                                    f"{r['target']}|{r['leg']}" and
                                    g["material"] for g in out)),
    }


def scan_worst(trainer, pair_map, b1_cache, mech, full_by):
    worst = []
    for pid in WINTER:
        pair = pair_map[pid]
        b1 = b1_cache.get(pid, 0.5, mech, pair_data_hash(full_by[pid]))
        env = tb.envmod.build_env_for_pair(
            pair, differential_reward=False, terminal_soc_target=0.5)
        obs = env.reset()
        blocks = {u: [] for u in UNITS}
        for block in range(24):
            acts = {nm: np.asarray(
                trainer.select_action(obs[nm], nm, train=False)["quotes"])
                for nm in trainer.agent_names}
            obs, _, done, info = env.step(acts)
            for u in UNITS:
                blocks[u].append(float(info.get("raw_profit", {}).get(u, 0.0)))
            if done:
                break
        for u in UNITS:
            diffs = [blocks[u][t] - b1.block_payoffs[u][t]
                     for t in range(24)]
            t = int(np.argmin(diffs))
            worst.append({"pair_id": pid, "unit": u, "block": t,
                          "diff": float(diffs[t])})
    return worst


def winter_audit(trainer, pair_map, b1_cache, mech, full_by, worst):
    rows = []
    summary = {"states": {}}
    for rec in worst:
        pid, unit, block = rec["pair_id"], rec["unit"], rec["block"]
        pair = pair_map[pid]
        env, obs, baseline, past, soc, gates, probs, plog, price = \
            collect_state(trainer, pair, block)
        b1 = b1_day(pair, pid, {}, {})
        actor = float(baseline[unit][1])
        quotes = [0.0, 100.0, 150.0, 258.0, 300.0, actor]
        seen, candidates = set(), []
        for q in quotes:
            if round(q, 6) not in seen:
                seen.add(round(q, 6))
                candidates.append(q)
        key = f"{pid}|b{block}|{unit}"
        summary["states"][key] = {"actor_quote": actor,
                                  "gate": probs[unit], "scenarios": {}}
        for q in candidates:
            out = policy_block(trainer, pair, block, unit, q, baseline, env)
            block_rows = []
            for t in range(len(out["lmp"])):
                period = block * 4 + t
                b1rec = b1[unit][period]
                row = {
                    "pair_id": pid, "block": block, "unit": unit,
                    "quote": q, "actor_quote": actor, "period": period,
                    "lmp": float(out["lmp"][t]),
                    "effective_offer": q + out["cycle"],
                    "price_taking": bool(q + out["cycle"]
                                         <= out["lmp"][t] + 1e-9),
                    "accepted_mwh": float(out["p_dis"][t]) * 0.25,
                    "soc_end": float(out["soc"][t]),
                    "payoff": float(out["payoff"][t]),
                    "b1_payoff": b1rec["payoff"],
                    "diff": float(out["payoff"][t] - b1rec["payoff"]),
                }
                rows.append(row)
                block_rows.append(row)
            summary["states"][key]["scenarios"][str(q)] = {
                "block_payoff": out["block_profit"],
                "block_accepted_mwh": float(np.sum(out["p_dis"]) * 0.25),
                "b1_block_payoff": float(np.sum(
                    [b1[unit][block * 4 + t]["payoff"]
                     for t in range(len(out["lmp"]))])),
                "accepted_periods": int(sum(
                    r["accepted_mwh"] > 1e-6 for r in block_rows)),
                "mean_lmp": float(np.mean(out["lmp"])),
            }
    return rows, summary


def buffer_summary(trainer):
    keys = np.rint(np.asarray(trainer.buffer.extras["stratum"],
                              dtype=float).reshape(-1)).astype(int)
    counts = np.bincount(keys, minlength=N_CELLS)
    actions = np.stack([trainer.buffer.buffer[i][1]
                        for i in range(len(trainer.buffer))])
    explored = np.asarray(trainer.buffer.extras["explored"])
    high = {}
    for j, u in enumerate(trainer.agent_names):
        hi = actions[:, j, 1] >= 0.86 * 300.0
        high[u] = int(hi.sum())
    return {
        "transitions": int(len(trainer.buffer)),
        "cells_nonempty": int((counts > 0).sum()),
        "cell_count_min": int(counts[counts > 0].min()) if counts.any() else 0,
        "cell_count_max": int(counts.max()),
        "high_discharge_quote": high,
        "explored_legs": int(explored.sum()),
        "explored_discharge": int(explored[:, :, 1].sum()),
        "mean_discharge_quote": {
            u: float(actions[:, j, 1].mean())
            for j, u in enumerate(trainer.agent_names)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(RUNS / f"{RUN_ID}_best"))
    parser.add_argument("--buffer-root", default=str(RUNS / RUN_ID))
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not (root / "manifest.json").exists():
        root = RUNS / RUN_ID
    trainer, experiment, pair_map = load_run(root)
    buffer_trainer = trainer
    if Path(args.buffer_root).joinpath("manifest.json").exists():
        buffer_trainer, _, _ = load_run(Path(args.buffer_root))
    pair_map_full, full_by = pairs()
    b1_cache = mod.load_b1_cache()
    mech = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]
    rows = material_rollouts(trainer, experiment, pair_map, b1_cache, mech,
                             full_by)
    groups, msummary = material_summary(rows)
    worst = scan_worst(trainer, pair_map, b1_cache, mech, full_by)
    winter_rows, winter_summary = winter_audit(
        trainer, pair_map, b1_cache, mech, full_by, worst)
    summary = {
        "root": str(root),
        "material": msummary,
        "material_groups": groups,
        "winter_worst_blocks": worst,
        "winter": winter_summary,
        "buffer": buffer_summary(buffer_trainer),
        "buffer_root": str(args.buffer_root),
    }
    out_csv = DATA / "da_gate_material_calibration.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / "da_gate_material_calibration_groups.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    (DATA / "da_gate_material_calibration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (DATA / "da_gate_winter_15min.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(winter_rows[0]))
        writer.writeheader()
        writer.writerows(winter_rows)
    (DATA / "da_gate_winter_15min.json").write_text(
        json.dumps(winter_summary, indent=2), encoding="utf-8")
    print(json.dumps({"root": str(root), "material": msummary,
                      "buffer": summary["buffer"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
