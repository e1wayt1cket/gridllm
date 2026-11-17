"""Write the phase-17 pilot data artifacts from the saved v3 run.

Read-only: it loads the final and best checkpoints, reads the run logs and
writes the mapping grid, sampler probabilities, exploration log, buffer
coverage and a pilot summary. It does not train or modify a checkpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import da_quote_map as qm
import rl_da_gate as mod
import train_da_gate as tb
from da_replay import N_CELLS, decode_key
from diagnose_critic_long_horizon import WINTER, pairs

DATA = Path("data/processed/v2_cy_jn_2025")
RUNS = Path("runs")
RUN_ID = "DEV_DA_GATE_seed11_v2"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


def load_trainer(root):
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
    return trainer


def mapping_grid():
    rows = []
    for z in np.linspace(-5.0, 5.0, 2001):
        frac = float(qm.map_fraction_np(np.array([z]))[0])
        deriv = float(qm.map_derivative_np(np.array([z]))[0])
        sig = 1.0 / (1.0 + np.exp(-z))
        rows.append({
            "logit": float(z),
            "fraction": frac,
            "charge_quote": 150.0 * frac,
            "discharge_quote": 300.0 * frac,
            "d_fraction_dlogit": deriv,
            "d_charge_dlogit": 150.0 * deriv,
            "d_discharge_dlogit": 300.0 * deriv,
            "sigmoid_fraction": float(sig),
            "sigmoid_d_discharge_dlogit": float(300.0 * sig * (1 - sig)),
        })
    return rows


def buffer_artifacts(trainer):
    n = len(trainer.buffer)
    keys = np.rint(np.asarray(trainer.buffer.extras["stratum"],
                              dtype=float).reshape(-1)).astype(int)
    counts = np.bincount(keys, minlength=N_CELLS)
    nonempty = np.flatnonzero(counts > 0)
    w = counts[nonempty].astype(float) ** trainer.buffer.config["beta"]
    w /= w.sum()
    p_cell = dict(zip(nonempty.tolist(), w.tolist()))
    p_strat = np.zeros(n)
    for cell in nonempty:
        idx = np.flatnonzero(keys == cell)
        p_strat[idx] = p_cell[int(cell)] / len(idx)
    p_strat = np.minimum(
        p_strat, trainer.buffer.config["max_upweight"] / n)
    p_strat /= p_strat.sum()
    uf = trainer.buffer.config["uniform_fraction"]
    p = (1 - uf) / n + uf * p_strat
    p /= p.sum()
    rows = []
    episode_ids = list(csv.DictReader(
        (RUNS / RUN_ID / "episodes.csv").open(encoding="utf-8")))
    for i in range(n):
        obs, action, reward, _, _ = trainer.buffer.buffer[i]
        gates = np.asarray(trainer.buffer.extras["gates"][i])
        price_quotes = np.asarray(trainer.buffer.extras["price_quotes"][i])
        explored = np.asarray(trainer.buffer.extras["explored"][i])
        source = np.asarray(trainer.buffer.extras["exploration_source"][i])
        actor = gates * price_quotes
        for j, u in enumerate(trainer.agent_names):
            for leg, name in enumerate(("charge", "discharge")):
                rows.append({
                    "index": i,
                    "episode": i // 24 + 1,
                    "pair_id": episode_ids[i // 24]["pair_id"]
                    if i // 24 < len(episode_ids) else "",
                    "unit": u, "leg": name,
                    "stratum": int(keys[i]),
                    "cell": json.dumps(decode_key(int(keys[i]))),
                    "cell_count": int(counts[keys[i]]),
                    "p_uniform": 1.0 / n,
                    "p_stratified": float(p_strat[i]),
                    "p_mixture": float(p[i]),
                    "uniform_ratio": float((1.0 / n) / p[i]),
                    "actor_quote": float(actor[j, leg]),
                    "executed_quote": float(action[j, leg]),
                    "gate": float(gates[j, leg]),
                    "explored": int(explored[j, leg] > 0),
                    "exploration_source": int(source[j, leg]),
                    "reward": float(reward[j]),
                })
    transition_rows = [{k: r[k] for k in
                        ("index", "episode", "pair_id", "stratum", "cell",
                         "cell_count", "p_uniform", "p_stratified",
                         "p_mixture", "uniform_ratio")}
                        for r in rows[::8]]
    coverage = {
        "transitions": n,
        "cells_nonempty": int(len(nonempty)),
        "cell_count_min": int(counts[nonempty].min()),
        "cell_count_median": float(np.median(counts[nonempty])),
        "cell_count_max": int(counts[nonempty].max()),
        "prob_min": float(p.min()),
        "prob_max": float(p.max()),
        "uniform_ratio_min": float((1.0 / n) / p.max()),
        "uniform_ratio_max": float((1.0 / n) / p.min()),
        "ess": float(1.0 / np.sum(p ** 2)),
        "ess_fraction": float((1.0 / np.sum(p ** 2)) / n),
        "high_discharge_quote": {},
        "explored_legs": int(np.asarray(
            trainer.buffer.extras["explored"]).sum()),
        "explored_discharge": int(np.asarray(
            trainer.buffer.extras["explored"])[:, :, 1].sum()),
    }
    for j, u in enumerate(trainer.agent_names):
        coverage["high_discharge_quote"][u] = {
            "ge_0_86h": int(np.sum(
                np.asarray(trainer.buffer.buffer[i][1])[j, 1]
                >= 0.86 * 300.0 for i in range(n))),
            "ge_0_99h": int(np.sum(
                np.asarray(trainer.buffer.buffer[i][1])[j, 1]
                >= 0.99 * 300.0 for i in range(n))),
        }
    return rows, transition_rows, coverage


def main() -> int:
    final = load_trainer(RUNS / RUN_ID)
    best = load_trainer(RUNS / f"{RUN_ID}_best")
    with (DATA / "da_quote_map_grid.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        rows = mapping_grid()
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rows, transition_rows, coverage = buffer_artifacts(final)
    with (DATA / "da_gate_exploration_log.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DATA / "da_gate_sampling_probabilities.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)
    (DATA / "da_gate_buffer_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8")
    eval_history = json.loads(
        (RUNS / RUN_ID / "eval_history.json").read_text(encoding="utf-8"))
    episodes = list(csv.DictReader(
        (RUNS / RUN_ID / "episodes.csv").open(encoding="utf-8")))
    summary = {
        "run_id": RUN_ID,
        "final_root": str(RUNS / RUN_ID),
        "best_root": str(RUNS / f"{RUN_ID}_best"),
        "eval_history": eval_history,
        "training": {
            "episodes": len(episodes),
            "delta_fleet_mean": float(np.mean(
                [float(r["delta_fleet"]) for r in episodes])),
            "delta_fleet_median": float(np.median(
                [float(r["delta_fleet"]) for r in episodes])),
            "gate_temperature": {
                "start": float(episodes[0]["gate_temperature"]),
                "mid": float(episodes[len(episodes) // 2]
                             ["gate_temperature"]),
                "end": float(episodes[-1]["gate_temperature"]),
            },
            "exploration_epsilon": {
                "start": float(episodes[0]["exploration_epsilon"]),
                "end": float(episodes[-1]["exploration_epsilon"]),
            },
            "explored_legs": int(sum(
                int(r["explored_legs"]) for r in episodes)),
            "explored_discharge": int(sum(
                int(r["explored_discharge"]) for r in episodes)),
        },
        "buffer": coverage,
        "best_counters": {
            "episodes": best.episodes,
            "updates": best.updates,
            "buffer_len": len(best.buffer),
        },
        "final_counters": {
            "episodes": final.episodes,
            "updates": final.updates,
            "buffer_len": len(final.buffer),
        },
    }
    (DATA / "da_gate_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"training": summary["training"],
                      "buffer": coverage,
                      "eval": [(e["round"], e["mean_fleet_delta"])
                               for e in eval_history]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
