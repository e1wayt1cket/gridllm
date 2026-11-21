"""P0-1: full-day counterfactual quote interventions on winter states.

For each checkpoint and state, hold the joint state fixed, change only the
target ESS's current-block discharge quote, and continue the frozen
deterministic actor to the day end. The counterfactual re-initialises from the
state's SOC; it does not transplant the original trajectory's future SOC.

Checkpoints: D+A best step 1272, D+A final step 3816, old R2 step 2544 and
old R10 step 12720. No checkpoint is written. H2 and the legacy test are not
read.

Output: `daily_counterfactual_action_validation.csv/json`.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import rl_b_c2_gate as oldmod
import rl_da_gate as newmod
import train_da_gate as tb
from diagnose_critic_long_horizon import (ROOTS, WINTER, collect_state,
                                          critic_q, load_trainer, pairs,
                                          rollout, spearman)
from rl_b_c2_gate import pair_data_hash

DATA = Path("data/processed/v2_cy_jn_2025")
RUNS = Path("runs")
OLD_RUN = RUNS / "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
NEW_RUN = RUNS / "DEV_DA_GATE_seed11_v2"
NEW_BEST = RUNS / "DEV_DA_GATE_seed11_v2_best"
LEGS = ("charge", "discharge")
HIGHS = {"charge": 150.0, "discharge": 300.0}
UNITS = ("ESS14", "ESS27")
# Eight states around the ESS14/ESS27 zero-acceptance blocks.
STATES = (
    ("2022-12-13|2025-12-11", 3, "ESS14"),
    ("2022-12-13|2025-12-11", 12, "ESS27"),
    ("2022-12-30|2025-12-31", 10, "ESS14"),
    ("2022-12-30|2025-12-31", 10, "ESS27"),
    ("2022-12-30|2025-12-31", 11, "ESS14"),
    ("2022-12-30|2025-12-31", 11, "ESS27"),
    ("2022-12-31|2025-12-28", 17, "ESS14"),
    ("2022-12-31|2025-12-28", 17, "ESS27"),
)


def load_checkpoints(template, pair_map):
    out = {}
    old_manifest = json.loads((OLD_RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    old_exp = oldmod.ExperimentConfig(**old_manifest["experiment"])
    for label, cp in (("old_R2", 2), ("old_R10", 10)):
        out[label] = load_trainer(cp, template, old_exp)
    new_manifest = json.loads((NEW_RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    new_exp = newmod.DAExperimentConfig(**new_manifest["experiment"])
    for label, root in (("da_best", NEW_BEST), ("da_final", NEW_RUN)):
        trainer = newmod.DAGateTrainer(
            tb.envmod.build_env_for_pair(
                pair_map[WINTER[0]], differential_reward=False,
                terminal_soc_target=0.5), new_exp)
        trainer.load(root)
        out[label] = trainer
    return out


def critic_top(trainer, obs, baseline, target, leg):
    high = HIGHS[leg]
    best_q, best_quote = -np.inf, 0.0
    for quote in np.linspace(0.0, high, 61):
        q = critic_q(trainer, obs, baseline, target, leg, float(quote))
        if q["q_min"] > best_q:
            best_q, best_quote = q["q_min"], float(quote)
    return best_quote, best_q


def one_state(trainer, experiment, pair_map, b1_cache, mech, full_by,
              pid, block, unit, label):
    pair = pair_map[pid]
    row = full_by[pid]
    b1 = b1_cache.get(pid, 0.5, mech, pair_data_hash(row))
    env, obs, baseline, past, soc, gates, probs, plog, price = \
        collect_state(trainer, pair, block)
    leg = "discharge"
    high = HIGHS[leg]
    actor_q = float(baseline[unit][1])
    cq, cq_val = critic_top(trainer, obs, baseline, unit, leg)
    candidates = [actor_q, cq, 0.0, 0.25 * high, 0.5 * high, high]
    seen, rows = set(), []
    for quote in candidates:
        key = round(float(quote), 6)
        if key in seen:
            continue
        seen.add(key)
        q = critic_q(trainer, obs, baseline, unit, leg, quote)
        roll = rollout(
            trainer, env, obs, baseline, unit, leg, quote, block, past, soc,
            b1.block_payoffs[unit], sum(b1.block_payoffs[unit]),
            experiment.fixed_scale)
        gamma = experiment.gamma
        b_return = (gamma ** (23 - block) * roll["full_delta_cny"]
                    / experiment.fixed_scale)
        rows.append({
            "checkpoint": label, "pair_id": pid, "block": block,
            "unit": unit, "leg": leg, "quote": float(quote),
            "actor_quote": actor_q, "critic_recommended_quote": cq,
            "q1": q["q1"], "q2": q["q2"], "q_min": q["q_min"],
            "twin_gap": q["twin_gap"], "dq_dquote": q["dq_dquote"],
            "b_return_scaled": b_return,
            "training_return_scaled": roll["training_return_scaled"],
            "c2_component_scaled":
                roll["training_return_scaled"] - b_return,
            "full_delta_cny": roll["full_delta_cny"],
            "future_payoff_cny": roll["future_payoff"],
            "future_minus_b1_cny": roll["future_minus_b1"],
            "dG_vs_actor_cny": None,  # filled below
            "terminal_soc": roll["terminal_soc"],
            "accepted_at_block": roll["accepted_at_block"],
            "p_ch_mwh": roll["p_ch_mwh_at_block"],
            "p_dis_mwh": roll["p_dis_mwh_at_block"],
            "soc_start": soc[unit],
            "soc_path": json.dumps(roll["soc_path"]),
            "gate_probability": gates[unit],
            "gate_deterministic": probs[unit],
        })
    actor_row = next(r for r in rows if abs(r["quote"] - actor_q) < 1e-9)
    for r in rows:
        r["dG_vs_actor_cny"] = (r["full_delta_cny"]
                                - actor_row["full_delta_cny"])
    best_return = max(rows, key=lambda r: r["full_delta_cny"])
    best_q = max(rows, key=lambda r: r["q_min"])
    rho = spearman([r["q_min"] for r in rows],
                   [r["full_delta_cny"] for r in rows])
    return rows, {
        "checkpoint": label, "pair_id": pid, "block": block, "unit": unit,
        "actor_quote": actor_q, "actor_delta": actor_row["full_delta_cny"],
        "critic_recommended_quote": cq,
        "critic_recommended_q": cq_val,
        "best_return_quote": best_return["quote"],
        "best_return_delta": best_return["full_delta_cny"],
        "best_return_gain_vs_actor":
            best_return["full_delta_cny"] - actor_row["full_delta_cny"],
        "best_q_quote": best_q["quote"],
        "best_q_return": best_q["full_delta_cny"],
        "q_top_matches_return_top": bool(
            abs(best_q["quote"] - best_return["quote"]) < 1e-9),
        "spearman_q_delta": rho,
        "actor_gate": probs[unit], "actor_gate_probability": gates[unit],
        "soc_start": soc[unit],
    }


def main(argv=None) -> int:
    pair_map, full_by = pairs()
    template = tb.envmod.build_env_for_pair(
        pair_map[WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    trainers = load_checkpoints(template, pair_map)
    b1_cache = oldmod.load_b1_cache()
    mech = json.loads(Path(oldmod.B1_MANIFEST).read_text(
        encoding="utf-8"))["mechanism_hash"]
    all_rows, summary = [], []
    for label, trainer in trainers.items():
        exp = trainer.experiment
        for pid, block, unit in STATES:
            rows, s = one_state(trainer, exp, pair_map, b1_cache, mech,
                                full_by, pid, block, unit, label)
            all_rows.extend(rows)
            summary.append(s)
            print(f"{label} {pid} b{block} {unit}: "
                  f"actor {s['actor_delta']:.1f}, best "
                  f"{s['best_return_delta']:.1f} @ {s['best_return_quote']:.1f}",
                  flush=True)
    out_csv = DATA / "daily_counterfactual_action_validation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    by_label = defaultdict(list)
    for s in summary:
        by_label[s["checkpoint"]].append(s)
    agg = {
        label: {
            "states": len(v),
            "q_top_matches_return_top": int(sum(
                x["q_top_matches_return_top"] for x in v)),
            "mean_spearman_q_delta": float(np.nanmean(
                [x["spearman_q_delta"] for x in v])),
            "mean_best_gain_vs_actor": float(np.mean(
                [x["best_return_gain_vs_actor"] for x in v])),
        } for label, v in by_label.items()}
    (DATA / "daily_counterfactual_action_validation.json").write_text(
        json.dumps({"rows": summary, "aggregate": agg}, indent=2),
        encoding="utf-8")
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
