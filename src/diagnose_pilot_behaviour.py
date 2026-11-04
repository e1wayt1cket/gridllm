# diagnose_pilot_behaviour.py
"""Why the pilot's policies quote where they do, and what the local reward means.

Runs the saved actors deterministically on a fixed set of **training and
validation** days (never the test split), records every block's settled
profit, local truthful counterfactual, SOC path and quotes, then settles each
day with the policy's own 96-period quotes and compares that with B1.

It changes no model, reward or checkpoint. It exists to separate a computation
error from a definition mismatch from a real cross-period opportunity cost.

Usage:
    PYTHONPATH=src python src/diagnose_pilot_behaviour.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import run_matd3_pilot as pilot
import screen_v2_cy_jn_2025 as screen
import v2_cy_jn_env as envmod
from rl_spec import OBS_V6

PILOT = "data/processed/v2_cy_jn_2025/pilot"
BLOCKS_CSV = f"{PILOT}/reward_horizon_blocks.csv"
ACTIONS_CSV = f"{PILOT}/action_samples.csv"
HORIZON_REPORT = "reports/v2_cy_jn_2025/difference_reward_horizon_alignment.md"
ACTION_REPORT = "reports/v2_cy_jn_2025/matd3_action_mapping_diagnostic.md"
DAY_REPORT = "data/processed/v2_cy_jn_2025/pilot/reward_horizon_days.csv"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
TRAIN_SAMPLE = 6
VAL_SAMPLE = 6


def _read(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample_days() -> List[dict]:
    split = json.loads((Path(pilot.COMMON)).read_text(encoding="utf-8"))
    pairs = {r["load_date"]: r for r in _read(Path(pilot.PAIRING))}
    out = []
    for key, label, n in (("train", "train", TRAIN_SAMPLE),
                          ("validation", "validation", VAL_SAMPLE)):
        ids = [i.split("|")[0] for i in split["common"][key]]
        step = max(1, len(ids) // n)
        for day in ids[::step][:n]:
            row = dict(pairs[day])
            row["split"] = label
            out.append(row)
    return out


def _load_actors(dataset: str, seed: int):
    ckpt = Path(pilot.CKPT_DIR) / f"{dataset}_seed{seed}"
    env = envmod.build_env_for_pair(pilot.pools()[dataset][0])
    trainer = pilot._trainer(env)
    from rl_td3 import load_policy
    for nm in trainer.agent_names:
        trainer.actors[nm] = load_policy(
            str(ckpt / f"{nm}.pt"), env.get_state_dim(),
            trainer.bounds_by_agent[nm], obs_spec=OBS_V6,
            action_spec=trainer.spec_by_agent[nm])
    return env, trainer


def run_days(dataset: str, seed: int,
             days: Sequence[dict]) -> tuple:
    import torch
    env, trainer = _load_actors(dataset, seed)
    names = trainer.agent_names
    blocks: List[dict] = []
    day_rows: List[dict] = []
    for row in days:
        agents, price = pilot.day_agents(row, env.config)
        envmod.configure_price(row["price_date"])
        env.set_agents(agents, day_id=row["price_date"])
        obs = env.reset()
        quotes = {nm: {"charge_price": np.zeros(env.T),
                       "discharge_price": np.zeros(env.T)} for nm in names}
        soc_path = {nm: [] for nm in names}
        base_soc_path = {nm: [] for nm in names}
        block_profit = {nm: [] for nm in names}
        block_base = {nm: [] for nm in names}
        raw_total = base_total = 0.0
        for block in range(pilot.N_BLOCKS):
            t_start = block * 4
            n = min(4, env.T - t_start)
            soc_before = {a.name: float(env.prev_soc.get(
                a.name, a.storage.soc0)) for a in env.rl_agents}
            acts = {}
            logits = {}
            for nm in names:
                obs_t = torch.as_tensor(obs[nm], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    raw_logits, action = trainer.actors[nm].forward_logits(obs_t)
                acts[nm] = action.squeeze(0).numpy()
                logits[nm] = raw_logits.squeeze(0).numpy()
                quotes[nm]["charge_price"][t_start:t_start + n] = acts[nm][0]
                quotes[nm]["discharge_price"][t_start:t_start + n] = acts[nm][1]
            next_obs, rewards, done, info = env.step(acts)
            real = env._last_result
            base = env._last_base_result
            for nm in names:
                raw = float(info.get("raw_profit", {}).get(nm, 0.0))
                baseline = float(info.get("local_baseline_profit", {}).get(nm, 0.0))
                raw_total += raw
                base_total += baseline
                block_profit[nm].append(raw)
                block_base[nm].append(baseline)
                soc_after = float(env.prev_soc.get(nm, soc_before[nm]))
                base_soc = (float(base["schedules"][nm]["soc_boundary"][n])
                            if base is not None and nm in base["schedules"]
                            else float("nan"))
                soc_path[nm].append(soc_after)
                base_soc_path[nm].append(base_soc)
                charge, discharge = acts[nm]
                low, high = trainer.bounds_by_agent[nm]
                low, high = low.numpy(), high.numpy()
                truthful = np.array([trainer.spec_by_agent[nm].truthful[k]
                                     for k in trainer.spec_by_agent[nm].action_names])
                d = float(np.sum(np.abs(acts[nm] - truthful) / (high - low)))
                blocks.append({
                    "dataset": dataset, "seed": seed,
                    "load_date": row["load_date"], "split": row["split"],
                    "unit": nm, "block": block,
                    "raw_profit": raw, "local_baseline": baseline,
                    "reward": rewards.get(nm, 0.0),
                    "soc_before": soc_before[nm], "soc_after": soc_after,
                    "base_soc_after": base_soc,
                    "charge_quote": float(charge),
                    "discharge_quote": float(discharge),
                    "norm_distance": d,
                    "logit_charge": float(logits[nm][0]),
                    "logit_discharge": float(logits[nm][1]),
                    "tanh_charge": float(np.tanh(logits[nm][0])),
                    "tanh_discharge": float(np.tanh(logits[nm][1])),
                    "last_lmp": (float(env.hist_lmp[-1]) if env.hist_lmp
                                 else float("nan")),
                })
            if done:
                break
            obs = next_obs
        from market import clear_market
        result = clear_market(agents, screen.T, "DA", quotes, env.config,
                              wholesale=price, horizon_type="full_day")
        net = screen.grid.build_base_network(env.config)
        checks = (screen._checks(result, agents, env.config, price, net)
                  if result is not None else {})
        settled = {"optimal": result is not None,
                   "settled_total": sum(checks.get("payoffs", {}).values()),
                   "payoffs": checks.get("payoffs", {})}
        b1 = {r["load_date"]: r for r in _read(Path(pilot.B1_RESULTS))
              if r.get("arm") == "JN-Base"}.get(row["load_date"], {})
        per_unit_charge = {}
        per_unit_discharge = {}
        if result is not None:
            for nm in UNITS:
                sched = result["schedules"][nm]
                per_unit_charge[nm] = float(np.sum(sched["p_ch"])) * 0.25
                per_unit_discharge[nm] = float(np.sum(sched["p_dis"])) * 0.25
        total_reward = sum(sum(v) for v in block_profit.values())
        total_local = sum(sum(v) for v in block_base.values())
        day_rows.append({
            "dataset": dataset, "seed": seed, "split": row["split"],
            "load_date": row["load_date"],
            "sum_raw": total_reward, "sum_local_baseline": total_local,
            "sum_local_diff": total_reward - total_local,
            "settled_total": settled.get("settled_total", float("nan")),
            "b1_total": float(b1.get("ess_total_payoff_cny", "nan")),
            "day_diff_vs_b1": (settled.get("settled_total", float("nan"))
                               - float(b1.get("ess_total_payoff_cny", "nan"))),
            "state_path_term": total_local - float(
                b1.get("ess_total_payoff_cny", "nan")),
            "mechanism_term": (settled.get("settled_total", float("nan"))
                               - total_reward),
            "final_soc_gap": max(abs(float(env.prev_soc.get(nm, 0.5)) - 0.5)
                                 for nm in UNITS),
            "max_soc_divergence": max(
                (abs(a - b) for nm in UNITS
                 for a, b in zip(soc_path[nm], base_soc_path[nm])
                 if np.isfinite(b)), default=float("nan")),
            **{f"settled_{nm}": settled.get("payoffs", {}).get(nm, float("nan"))
               for nm in UNITS},
            **{f"charge_{nm}": per_unit_charge.get(nm, float("nan"))
               for nm in UNITS},
            **{f"discharge_{nm}": per_unit_discharge.get(nm, float("nan"))
               for nm in UNITS},
        })
    return blocks, day_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="NEAR,FULL96")
    parser.add_argument("--seeds", default="11,22")
    args = parser.parse_args(argv)
    for path in (BLOCKS_CSV, ACTIONS_CSV):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).exists():
            Path(path).unlink()
    days = _sample_days()
    all_days = []
    all_blocks = []
    for dataset in args.datasets.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            blocks, day_rows = run_days(dataset, seed, days)
            all_days.extend(day_rows)
            all_blocks.extend(blocks)
            print(f"{dataset} seed{seed}: {len(day_rows)} days, "
                  f"{len(blocks)} block-unit rows", flush=True)
    for path, rows in ((DAY_REPORT, all_days), (BLOCKS_CSV, all_blocks),
                       (ACTIONS_CSV, all_blocks)):
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {BLOCKS_CSV}\nwrote {ACTIONS_CSV}\nwrote {DAY_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
