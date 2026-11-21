"""P0-2: single-ESS all-day fixed quote grid with the other three truthful.

For one target ESS on 2022-12-30|2025-12-31, enumerate 10 charge quotes and 10
discharge quotes held for all 24 blocks. ESS9, ESS17 and ESS27 are truthful
all day. Each combination is a complete 24-block rolling SOCP with terminal
SOC 0.5; no block-average or single-block payoff is used.

This is a conditional optimum inside the fixed-quote family with three
truthful competitors. It is not the four-agent dynamic-bidding global optimum.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import rl_b_c2_gate as oldmod
import rl_da_gate as newmod
import train_da_gate as tb
from diagnose_critic_long_horizon import ROOTS, WINTER, load_trainer, pairs

DATA = Path("data/processed/v2_cy_jn_2025")
RUNS = Path("runs")
OLD_RUN = RUNS / "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
NEW_RUN = RUNS / "DEV_DA_GATE_seed11_v2"
NEW_BEST = RUNS / "DEV_DA_GATE_seed11_v2_best"
PAIR_ID = "2022-12-30|2025-12-31"
UNITS = ("ESS14", "ESS27")
CHARGE_POINTS = np.linspace(0.0, 150.0, 10)
DISCHARGE_POINTS = np.linspace(0.0, 300.0, 10)


def build(pair):
    return tb.envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)


def run_day(pair, target, charge, discharge, truthful_others=True):
    env = build(pair)
    obs = env.reset()
    names = [a.name for a in env.rl_agents]
    actions = {nm: np.zeros(2) for nm in names}
    if truthful_others:
        for nm in names:
            if nm == target:
                actions[nm] = np.array([charge, discharge], dtype=float)
    else:
        actions = None
    p_ch, p_dis, soc = [], [], []
    payoff = 0.0
    fleet = 0.0
    optimal = True
    fallbacks = 0
    voltage_cells = line_cells = 0
    for block in range(24):
        if actions is None:
            acts = {}
        else:
            acts = actions
        next_obs, _, done, info = env.step(acts)
        result = env._last_result
        if result is None:
            raise RuntimeError(f"{PAIR_ID} block {block}: clearing failed")
        optimal &= bool(result.get("solver", {}).get("final", {}).get("optimal"))
        fallbacks += int(result.get("fell_back", 0))
        ws = result["schedules"][target]
        n = min(4, env.T - block * 4)
        p_ch.append(float(np.sum(np.asarray(ws["p_ch"])[:n]) * 0.25))
        p_dis.append(float(np.sum(np.asarray(ws["p_dis"])[:n]) * 0.25))
        payoff += float(info.get("raw_profit", {}).get(target, 0.0))
        fleet += float(sum(info.get("raw_profit", {}).values()))
        soc.append(float(env.prev_soc.get(target, 0.5)))
        v = np.asarray(result.get("bus_voltage", []), dtype=float)[:n]
        if v.size:
            voltage_cells += int(((v < env.config.network.v_min_pu - 1e-7)
                                  | (v > env.config.network.v_max_pu
                                     + 1e-7)).sum())
        u = np.asarray(result.get("line_utilization", []), dtype=float)[:n]
        if u.size:
            line_cells += int((u > 1.0 + 1e-7).sum())
        obs = next_obs
        if done:
            break
    return {
        "charge": float(charge), "discharge": float(discharge),
        "payoff": payoff, "fleet": fleet,
        "charge_mwh": float(sum(p_ch)), "discharge_mwh": float(sum(p_dis)),
        "zero_blocks": int(sum(
            1 for c, d in zip(p_ch, p_dis) if c + d < 1e-6)),
        "soc_path": soc, "soc_end": soc[-1],
        "all_optimal": bool(optimal), "fallbacks": fallbacks,
        "voltage_cells": voltage_cells, "line_cells": line_cells,
        "p_ch_blocks": p_ch, "p_dis_blocks": p_dis,
    }


def actor_day(pair, trainer):
    env = build(pair)
    obs = env.reset()
    names = trainer.agent_names
    payoff = {nm: 0.0 for nm in names}
    quotes = {nm: [] for nm in names}
    for block in range(24):
        acts = {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            quotes[nm].append(acts[nm])
        obs, _, done, info = env.step(acts)
        for nm in names:
            payoff[nm] += float(info.get("raw_profit", {}).get(nm, 0.0))
        if done:
            break
    return {
        "payoff": payoff,
        "mean_charge": {nm: float(np.mean([q[0] for q in quotes[nm]]))
                        for nm in names},
        "mean_discharge": {nm: float(np.mean([q[1] for q in quotes[nm]]))
                           for nm in names},
    }


def nearest_index(value, points):
    return int(np.argmin(np.abs(np.asarray(points) - value)))


def analyze(results, b1, unit, actor_best, actor_r10):
    matrix = np.full((len(CHARGE_POINTS), len(DISCHARGE_POINTS)), np.nan)
    dg = np.full_like(matrix, np.nan)
    zero = np.zeros_like(matrix, dtype=int)
    fleet = np.full_like(matrix, np.nan)
    for r in results:
        i = nearest_index(r["charge"], CHARGE_POINTS)
        j = nearest_index(r["discharge"], DISCHARGE_POINTS)
        matrix[i, j] = r["payoff"]
        dg[i, j] = r["payoff"] - b1["payoff"]
        zero[i, j] = r["zero_blocks"]
        fleet[i, j] = r["fleet"]
    best = np.unravel_index(int(np.nanargmax(matrix)), matrix.shape)
    best_quote = (float(CHARGE_POINTS[best[0]]),
                  float(DISCHARGE_POINTS[best[1]]))
    best_val = float(matrix[best])
    honest_i = nearest_index(0.0, CHARGE_POINTS)
    honest_j = nearest_index(0.0, DISCHARGE_POINTS)
    local_max = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            nb = matrix[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if matrix[i, j] >= np.nanmax(nb) - 1e-9:
                local_max.append((i, j, float(matrix[i, j])))
    high_zero = int((zero > 0).sum())
    flat = int((matrix >= best_val - 1.0).sum())
    actors = {}
    for label, rec in (("da_best", actor_best), ("old_R10", actor_r10)):
        if rec is None:
            continue
        c = rec["mean_charge"][unit]
        d = rec["mean_discharge"][unit]
        i = nearest_index(c, CHARGE_POINTS)
        j = nearest_index(d, DISCHARGE_POINTS)
        actors[label] = {
            "mean_charge": c, "mean_discharge": d,
            "nearest_grid_charge": float(CHARGE_POINTS[i]),
            "nearest_grid_discharge": float(DISCHARGE_POINTS[j]),
            "actual_actor_payoff": rec["payoff"][unit],
            "grid_payoff_at_nearest": float(matrix[i, j]),
            "distance_to_best_cny": float(best_val - matrix[i, j]),
        }
    return {
        "unit": unit,
        "charge_points": CHARGE_POINTS.tolist(),
        "discharge_points": DISCHARGE_POINTS.tolist(),
        "matrix": matrix.tolist(),
        "dG_vs_b1": dg.tolist(),
        "zero_blocks": zero.tolist(),
        "fleet": fleet.tolist(),
        "b1_payoff": b1["payoff"],
        "best": {"charge": best_quote[0], "discharge": best_quote[1],
                 "payoff": best_val, "dG": best_val - b1["payoff"],
                 "boundary": bool(best[0] in (0, 9) or best[1] in (0, 9))},
        "honest": {"charge": 0.0, "discharge": 0.0,
                   "payoff": float(matrix[honest_i, honest_j]),
                   "dG": float(dg[honest_i, honest_j])},
        "flat_cells_within_1cny": flat,
        "cells": int(np.isfinite(matrix).sum()),
        "high_quote_zero_cells": high_zero,
        "local_maxima": local_max,
        "actors": actors,
    }


def refine(pair, target, best, b1):
    """5x5 local grid around the best, clipped to the legal box."""
    ci = [x for x in np.linspace(max(0.0, best["charge"] - 25.0),
                                 min(150.0, best["charge"] + 25.0), 5)]
    dj = [x for x in np.linspace(max(0.0, best["discharge"] - 60.0),
                                 min(300.0, best["discharge"] + 60.0), 5)]
    out = []
    for c in ci:
        for d in dj:
            r = run_day(pair, target, float(c), float(d))
            r["dG_vs_b1"] = r["payoff"] - b1["payoff"]
            out.append(r)
    best_r = max(out, key=lambda r: r["payoff"])
    return {"charge": ci, "discharge": dj, "results": out,
            "best": best_r}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default="ESS14")
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args(argv)
    unit = args.unit
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    pair_map, _ = pairs()
    pair = pair_map[PAIR_ID]
    # Truthful all-four B1 arm.
    b1 = run_day(pair, unit, 0.0, 0.0, truthful_others=False)
    # The truthful-others path in run_day sets the target to (0,0) for the
    # target; all four truthful is the same action vector.
    results = []
    for c in CHARGE_POINTS:
        for d in DISCHARGE_POINTS:
            results.append(run_day(pair, unit, float(c), float(d)))
            print(f"{unit} charge={c:.1f} discharge={d:.1f} "
                  f"payoff={results[-1]['payoff']:.2f}", flush=True)
    # Actors on their own deterministic policy.
    old_manifest = json.loads((OLD_RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    old_exp = oldmod.ExperimentConfig(**old_manifest["experiment"])
    r10 = load_trainer(10, build(pair_map[WINTER[0]]), old_exp)
    new_manifest = json.loads((NEW_RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    new_exp = newmod.DAExperimentConfig(**new_manifest["experiment"])
    best = newmod.DAGateTrainer(build(pair_map[WINTER[0]]), new_exp)
    best.load(NEW_BEST)
    actor_best = actor_day(pair, best)
    actor_r10 = actor_day(pair, r10)
    analysis = analyze(results, b1, unit, actor_best, actor_r10)
    if not args.no_refine:
        analysis["refinement"] = refine(
            pair, unit, analysis["best"], b1)
    rows = []
    for r in results:
        row = {k: r[k] for k in (
            "charge", "discharge", "payoff", "fleet", "charge_mwh",
            "discharge_mwh", "zero_blocks", "soc_end", "all_optimal",
            "fallbacks", "voltage_cells", "line_cells")}
        row["unit"] = unit
        row["dG_vs_b1"] = r["payoff"] - b1["payoff"]
        row["p_ch_blocks"] = json.dumps(r["p_ch_blocks"])
        row["p_dis_blocks"] = json.dumps(r["p_dis_blocks"])
        row["soc_path"] = json.dumps(r["soc_path"])
        rows.append(row)
    out_csv = DATA / f"single_ess_quote_surface_{unit}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / f"single_ess_quote_surface_{unit}.json").open(
            "w", encoding="utf-8") as fh:
        json.dump({"unit": unit, "b1": b1, "analysis": analysis,
                   "results": results,
                   "actor_best": actor_best, "actor_r10": actor_r10},
                  fh, indent=2)
    print(json.dumps({"unit": unit, "best": analysis["best"],
                      "honest": analysis["honest"],
                      "actors": analysis["actors"],
                      "high_zero_cells": analysis["high_quote_zero_cells"],
                      "flat_cells": analysis["flat_cells_within_1cny"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
