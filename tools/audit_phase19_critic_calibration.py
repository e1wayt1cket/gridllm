"""P19-A: critic calibration on training-day states.

The phase-18 calibration states are all on NEAR validation days, so they can be
reported but never used as training labels. This audit builds matched
state-action rollout labels on NEAR training days only:

* hold the joint state fixed at (day, block);
* change only the target unit's charge or discharge quote;
* continue the frozen deterministic actor to the day end;
* compare Q with the actual discounted B+C2 training return.

It also measures replay coverage of the same states and whether a calibrated
quote lies outside the buffer's executed quote range.

Read-only: no training, no checkpoint write, no validation-state label reuse,
no H2 and no legacy test.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import rl_b_c2_gate as mod
import train_b_c2_gate as tb
from audit_daily_counterfactual import load_checkpoints
from diagnose_critic_long_horizon import (collect_state, critic_q, pairs,
                                          rollout, spearman)

DATA = Path("data/processed/v2_cy_jn_2025")
TRAINS = ("2022-12-17", "2022-12-24", "2022-12-29")
BLOCKS = (8, 14)
UNITS = ("ESS14", "ESS27")
LEGS = ("charge", "discharge")
HIGHS = {"charge": 150.0, "discharge": 300.0}
GRID_FRACTIONS = (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)
# The five phase-18 cases; recorded only to state their split membership.
CASE_STATES = (
    ("2022-12-13|2025-12-11", 3, "ESS14"),
    ("2022-12-13|2025-12-11", 12, "ESS27"),
    ("2022-12-30|2025-12-31", 10, "ESS14"),
    ("2022-12-30|2025-12-31", 11, "ESS14"),
    ("2022-12-31|2025-12-28", 17, "ESS14"),
)


def band(quote: float, high: float) -> str:
    frac = quote / high
    if frac < 1 / 3:
        return "low"
    if frac < 2 / 3:
        return "mid"
    return "high"


def split_membership() -> dict:
    near = json.loads((DATA / "near/split.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    states = []
    for day, block, unit in CASE_STATES:
        entry = {"pair_id": day, "block": block, "unit": unit}
        for name, table in (("near", near), ("full96", full)):
            hits = [k for k in ("train", "validation", "test", "holiday_block")
                    if day.split("|")[0] in table.get(k, [])]
            entry[name] = hits[0] if hits else "absent"
        states.append(entry)
    return {
        "states": states,
        "train_days_used": [f"{d}|" for d in TRAINS],
        "note": ("all five phase-18 states are NEAR validation days; they are "
                 "offline cases only and contribute no training label"),
    }


def pair_id_by_load(pair_map: dict, load_date: str) -> str:
    hits = [k for k in pair_map if k.startswith(load_date + "|")]
    if len(hits) != 1:
        raise KeyError(f"expected one NEAR pair for {load_date}, got {hits}")
    return hits[0]


def quote_grid(high: float, actor_quote: float) -> list:
    values = {round(float(f * high), 6) for f in GRID_FRACTIONS}
    values.add(round(float(actor_quote), 6))
    return sorted(values)


def replay_stats(trainer, unit: str, obs: np.ndarray, block: int,
                 soc: float, grid: list, leg: int) -> dict:
    """Buffer coverage of one state's unit quote and joint observation."""
    if len(trainer.buffer) == 0:
        return {"buffer_rows": 0}
    buffer = trainer.buffer.buffer
    n = len(buffer)
    unit_idx = trainer.agent_names.index(unit)
    quotes = np.array([buffer[i][1][unit_idx][leg] for i in range(n)])
    blocks = np.array([buffer[i][0][unit_idx][4] * 23.0 for i in range(n)])
    socs = np.array([buffer[i][0][unit_idx][2] for i in range(n)])
    obs_matrix = np.array([buffer[i][0][unit_idx] for i in range(n)],
                          dtype=float)
    scale = obs_matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    dist = np.linalg.norm((obs_matrix - np.asarray(obs)[None, :]) / scale,
                          axis=1)
    near = int(np.argmin(dist))
    near_quote = float(quotes[near])
    in_state = (np.abs(blocks - block) <= 3) & (np.abs(socs - soc) <= 0.10)
    quote_in_band = (np.abs(quotes[:, None] - np.asarray(grid)[None, :])
                     <= 0.10 * HIGHS[LEGS[leg]])
    conditional = (quote_in_band & in_state[:, None]).any(axis=0)
    q_lo, q_hi = np.percentile(quotes, [5, 95])
    return {
        "buffer_rows": n,
        "executed_quote_p05": float(q_lo),
        "executed_quote_p95": float(q_hi),
        "executed_quote_median": float(np.median(quotes)),
        "nearest_state_distance": float(dist[near]),
        "nearest_state_quote": near_quote,
        "grid_covered_rows": int(conditional.sum()),
        "grid_extrapolated_rows": int(len(grid) - conditional.sum()),
        "grid_quotes": json.dumps(list(map(float, grid))),
        "grid_quote_covered": json.dumps(
            [bool(x) for x in conditional]),
    }


def main() -> int:
    pair_map, full_by = pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[pair_id_by_load(pair_map, TRAINS[0])],
        differential_reward=False, terminal_soc_target=0.5)
    trainers = load_checkpoints(template, pair_map)
    b1_cache = mod.load_b1_cache()
    mech = json.loads(Path(mod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]

    membership = split_membership()
    rows, states, coverage = [], [], []
    for label in ("da_best", "da_final"):
        trainer = trainers[label]
        exp = trainer.experiment
        for load_date in TRAINS:
            pid = pair_id_by_load(pair_map, load_date)
            pair = pair_map[pid]
            row = full_by[pid]
            b1 = b1_cache.get(pid, 0.5, mech, mod.pair_data_hash(row))
            for block in BLOCKS:
                env, obs, baseline, past, soc, gates, probs, plog, price = \
                    collect_state(trainer, pair, block)
                for unit in UNITS:
                    for leg_idx, leg in enumerate(LEGS):
                        high = HIGHS[leg]
                        actor_q = float(baseline[unit][leg_idx])
                        grid = quote_grid(high, actor_q)
                        cov = replay_stats(trainer, unit, obs[unit], block,
                                           soc[unit], grid, leg_idx)
                        coverage.append({"checkpoint": label, "pair_id": pid,
                                         "block": block, "unit": unit,
                                         "leg": leg, **cov})
                        for quote in grid:
                            q = critic_q(
                                trainer, obs, baseline, unit, leg, quote)
                            roll = rollout(
                                trainer, env, obs, baseline, unit, leg, quote,
                                block, past, soc,
                                b1.block_payoffs[unit],
                                sum(b1.block_payoffs[unit]),
                                exp.fixed_scale)
                            gamma = exp.gamma
                            b_return = (gamma ** (23 - block)
                                        * roll["full_delta_cny"]
                                        / exp.fixed_scale)
                            rows.append({
                                "checkpoint": label, "pair_id": pid,
                                "load_date": load_date, "block": block,
                                "unit": unit, "leg": leg,
                                "quote": float(quote),
                                "quote_band": band(float(quote), high),
                                "actor_quote": actor_q,
                                "actor_is_quote": abs(float(quote) - actor_q)
                                < 1e-5,
                                "q1": q["q1"], "q2": q["q2"],
                                "q_min": q["q_min"],
                                "twin_gap": q["twin_gap"],
                                "dq_dquote": q["dq_dquote"],
                                "b_return_scaled": b_return,
                                "training_return_scaled":
                                    roll["training_return_scaled"],
                                "full_delta_cny": roll["full_delta_cny"],
                                "accepted_at_block": roll[
                                    "accepted_at_block"],
                                "soc_start": soc[unit],
                                "gate_deterministic": probs[unit],
                            })
                        print(f"{label} {pid} b{block} {unit} {leg}: "
                              f"{len(grid)} quotes", flush=True)

    summary = {"checkpoints": {}, "groups": {}, "states": []}
    for label in ("da_best", "da_final"):
        cols = [r for r in rows if r["checkpoint"] == label]
        q = np.array([r["q_min"] for r in cols])
        ret = np.array([r["training_return_scaled"] for r in cols])
        dg = np.array([r["full_delta_cny"] for r in cols])
        summary["checkpoints"][label] = {
            "rows": len(cols),
            "q_vs_training_pearson": float(np.corrcoef(q, ret)[0, 1]),
            "q_vs_training_spearman": spearman(q, ret),
            "q_vs_delta_pearson": float(np.corrcoef(q, dg)[0, 1]),
            "q_vs_delta_spearman": spearman(q, dg),
            "mae_training_cny": float(np.mean(np.abs(q * 1000 - ret * 1000))),
            "rmse_training_cny": float(np.sqrt(np.mean(
                (q * 1000 - ret * 1000) ** 2))),
        }
        for dim_name, key in (("unit", "unit"), ("leg", "leg"),
                              ("quote_band", "quote_band")):
            groups = defaultdict(list)
            for r in cols:
                groups[r[key]].append(r)
            summary["groups"][f"{label}|{dim_name}"] = {
                g: {
                    "rows": len(v),
                    "pearson": float(np.corrcoef(
                        [x["q_min"] for x in v],
                        [x["training_return_scaled"] for x in v])[0, 1])
                    if len(v) > 2 and np.std(
                        [x["q_min"] for x in v]) > 0 else None,
                    "spearman": spearman([x["q_min"] for x in v],
                                         [x["training_return_scaled"]
                                          for x in v]),
                    "mean_q_cny": float(np.mean([x["q_min"] * 1000
                                                 for x in v])),
                    "mean_return_cny": float(np.mean(
                        [x["training_return_scaled"] * 1000 for x in v])),
                } for g, v in groups.items()}

        by_state = defaultdict(list)
        for r in cols:
            by_state[(r["pair_id"], r["block"], r["unit"], r["leg"])].append(r)
        for (pid, block, unit, leg), v in by_state.items():
            v = sorted(v, key=lambda r: r["quote"])
            q_top = max(v, key=lambda r: r["q_min"])
            ret_top = max(v, key=lambda r: r["training_return_scaled"])
            actor = next((r for r in v if r["actor_is_quote"]), None)
            summary["states"].append({
                "checkpoint": label, "pair_id": pid, "block": block,
                "unit": unit, "leg": leg,
                "rows": len(v),
                "spearman_q_return": spearman(
                    [r["q_min"] for r in v],
                    [r["training_return_scaled"] for r in v]),
                "q_top_quote": q_top["quote"],
                "q_top_return_cny": q_top["training_return_scaled"] * 1000,
                "return_top_quote": ret_top["quote"],
                "return_top_cny": ret_top["training_return_scaled"] * 1000,
                "q_top_matches_return_top":
                    abs(q_top["quote"] - ret_top["quote"]) < 1e-9,
                "actor_quote": actor["quote"] if actor else None,
                "actor_return_cny": (actor["training_return_scaled"] * 1000
                                     if actor else None),
                "best_gain_vs_actor_cny": (
                    (ret_top["training_return_scaled"]
                     - actor["training_return_scaled"]) * 1000
                    if actor else None),
                "return_spread_cny": (
                    (max(r["training_return_scaled"] for r in v)
                     - min(r["training_return_scaled"] for r in v)) * 1000),
            })

    out = DATA / "phase19_critic_training_calibration_rows.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cov_out = DATA / "phase19_replay_coverage.csv"
    with cov_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    (DATA / "phase19_critic_training_calibration.json").write_text(
        json.dumps({"split_membership": membership, "summary": summary},
                   indent=2), encoding="utf-8")
    print(json.dumps(summary["checkpoints"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
