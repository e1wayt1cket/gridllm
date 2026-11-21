# run_matd3_pilot.py
"""Controlled MATD3 pilot on the V2-CY-JN-2025 common split.

Two runs per seed differ only in the training-day pool: `NEAR` uses the 53
common training days, `FULL96` uses those 53 plus the 16 FULL96-only long
pairs. Everything else is frozen in `PILOT`: OBS_V6, the actor/critic layout,
the reward, the optimizer, the episode budget and the seeds. The full-day
settlement used for evaluation clears once with the policy's own per-block
quotes and pins terminal SOC, so it is directly comparable with B1 and B2A.

No test day enters training or tuning. `common_split.json` is the only source
of days.

Usage:
    PYTHONPATH=src python src/run_matd3_pilot.py --mode smoke
    PYTHONPATH=src python src/run_matd3_pilot.py --mode all
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import screen_v2_cy_jn_2025 as screen
import v2_cy_jn_env as envmod
from market import clear_market
from rl_bidding import MATD3, agent_action_spaces
from rl_env import BiddingEnv, N_BLOCKS, BLOCK_SIZE
from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V6
from rl_td3 import CALIBRATED_DEV_PENALTY, load_policy, save_policy

PAIRING = screen.PAIRING
BUS_LOADS = screen.BUS_LOADS
PRICE_CURVES = screen.PRICE_CURVES
COMMON = "data/processed/v2_cy_jn_2025/common_split.json"
BENCH_RAW = "data/processed/v2_cy_jn_2025/benchmark/benchmark_raw.csv"
B1_RESULTS = "reports/v2_cy_jn_2025/results.csv"
OUT = "data/processed/v2_cy_jn_2025/pilot"
CONFIG_PATH = f"{OUT}/pilot_config.json"
TRAIN_DIR = f"{OUT}/train"
CKPT_DIR = f"{OUT}/checkpoints"
EVAL_CSV = f"{OUT}/eval.csv"
LOG = f"{OUT}/run.log"

#: Frozen before any training. Every number here is declared, not chosen after
#: looking at a test result.
PILOT = {
    "version": "V2-CY-JN-2025-MATD3-pilot-1",
    "seeds": [11, 22],
    "episodes": 20,
    "start_steps": 2 * N_BLOCKS,
    "batch_size": 64,
    "buffer_capacity": 20000,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_delay": 2,
    "noise_std": 0.2,
    "noise_clip": 0.5,
    "final_noise_std": 0.05,
    "noise_anneal_steps": 2000,
    "dev_penalty": CALIBRATED_DEV_PENALTY,
    "obs_spec": OBS_V6.name,
    "action_spec": ACTION_STORAGE_QUOTE_V2.name,
    "cycle_cost": 100.0,
    "updates_per_block": 1,
    "datasets": {"NEAR": "common train only",
                 "FULL96": "common train + FULL96-only 16"},
    "evaluation_days": "common test 11 + national-day 5",
    "not_evaluated_on": "training days are not used for the final table",
}

TRAIN_COLUMNS = (
    "dataset", "seed", "episode", "load_date", "price_date", "total_steps",
    "updates", "ep_reward", "ep_raw_profit", "ep_local_baseline",
    "critic_loss", "actor_loss", "q1_mean", "q2_mean", "q_gap",
    "dev_penalty_mean", "action_charge_mean", "action_discharge_mean",
    "soc_min", "soc_max", "clears", "fallbacks", "lmp_fallbacks",
    "voltage_violation_cells", "line_overload_cells", "end_soc_gap",
    "runtime_s",
)

EVAL_COLUMNS = (
    "dataset", "seed", "split", "load_date", "price_date", "group",
    "settled_total", "settled_ESS9", "settled_ESS14", "settled_ESS17",
    "settled_ESS27", "b1_total", "b1_ESS9", "b1_ESS14", "b1_ESS17",
    "b1_ESS27", "b2a_total", "diff_vs_b1", "diff_vs_b2a",
    "charge_mwh", "discharge_mwh", "equivalent_cycles",
    "ep_reward", "ep_raw_profit", "ep_local_baseline", "dev_penalty_mean",
    "terminal_soc_max_err", "voltage_violation_cells", "line_overload_cells",
    "lmp_fallbacks", "optimal",
)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(path: Path, row: dict, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _log(message: str, path: Path = Path(LOG)) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _pairs() -> List[dict]:
    return _read_csv(Path(PAIRING))


def pools() -> Dict[str, List[dict]]:
    split = json.loads(Path(COMMON).read_text(encoding="utf-8"))
    by_load = {r["load_date"]: r for r in _pairs()}

    def days(ids):
        return [by_load[i.split("|")[0]] for i in ids]

    common_train = days(split["common"]["train"])
    extra = days(split["full96_only"])
    return {"NEAR": common_train, "FULL96": common_train + extra}


def day_agents(row: dict, config):
    cycles = np.load(PRICE_CURVES)
    loads = np.load(BUS_LOADS)
    price = np.asarray(cycles[f"south/{row['price_date']}"], dtype=float)
    load = np.asarray(loads[f"load/{row['load_date']}"], dtype=float)
    _net, agents = screen.build_agents(config, load, price, screen.ESS_BUSES)
    return agents, price


def _dev_penalty(env, obs, names, trainer) -> float:
    """The actor regularizer's value at the current observation."""
    total = 0.0
    for nm in names:
        action = trainer.select_action(obs[nm], nm, add_noise=False)
        low, high = trainer.bounds_by_agent[nm]
        low, high = low.numpy(), high.numpy()
        truthful = np.array([trainer.spec_by_agent[nm].truthful[k]
                             for k in trainer.spec_by_agent[nm].action_names])
        d = np.abs(action - truthful) / (high - low)
        total += float(np.sum(d ** 2))
    return PILOT["dev_penalty"] * total / max(len(names), 1)


def run_episode(env, trainer, row: dict, seed: int, rng: np.random.Generator,
                train: bool = True) -> dict:
    agents, price = day_agents(row, env.config)
    envmod.configure_price(row["price_date"])
    env.set_agents(agents, day_id=row["price_date"])
    obs = env.reset()
    names = trainer.agent_names
    rec = {
        "ep_reward": 0.0, "ep_raw_profit": 0.0, "ep_local_baseline": 0.0,
        "clears": 0, "fallbacks": 0, "lmp_fallbacks": 0,
        "voltage_violation_cells": 0, "line_overload_cells": 0,
        "dev_penalty_mean": 0.0, "action_charge_mean": 0.0,
        "action_discharge_mean": 0.0, "soc_min": 1.0, "soc_max": 0.0,
        "critic_loss": float("nan"), "actor_loss": float("nan"),
        "_cl": 0.0, "_al": 0.0,
        "q1_mean": float("nan"), "q2_mean": float("nan"),
        "q_gap": float("nan"), "updates": 0,
    }
    soc_lo = [a.storage.soc_min for a in env.rl_agents]
    soc_hi = [a.storage.soc_max for a in env.rl_agents]
    for block in range(N_BLOCKS):
        acts = {}
        for nm in names:
            if train and trainer.total_steps < trainer.start_steps:
                low, high = trainer.bounds_by_agent[nm]
                acts[nm] = rng.uniform(low.numpy(), high.numpy())
            else:
                acts[nm] = trainer.select_action(obs[nm], nm,
                                                 add_noise=train)
        next_obs, rewards, done, info = env.step(acts)
        result = env._last_result
        if result is None:
            rec["fallbacks"] += 1
        else:
            rec["clears"] += 1
            rec["lmp_fallbacks"] += int(result.get("lmp_fallbacks", 0))
            v = np.asarray(result["bus_voltage"], dtype=float)
            rec["voltage_violation_cells"] += int(
                ((v < 0.93 - 1e-7) | (v > 1.05 + 1e-7)).sum())
            util = np.asarray(result.get("line_utilization", np.zeros((1, 1))))
            rec["line_overload_cells"] += int((util > 1.0 + 1e-7).sum())
        for nm in names:
            rec["ep_reward"] += rewards.get(nm, 0.0)
            rec["ep_raw_profit"] += info.get("raw_profit", {}).get(nm, 0.0)
            rec["ep_local_baseline"] += info.get(
                "local_baseline_profit", {}).get(nm, 0.0)
        rec["dev_penalty_mean"] += _dev_penalty(env, obs, names, trainer)
        rec["action_charge_mean"] += float(
            np.mean([acts[nm][0] for nm in names]))
        rec["action_discharge_mean"] += float(
            np.mean([acts[nm][1] for nm in names]))
        for a in env.rl_agents:
            if a.name in env.prev_soc:
                rec["soc_min"] = min(rec["soc_min"], env.prev_soc[a.name])
                rec["soc_max"] = max(rec["soc_max"], env.prev_soc[a.name])
        if train:
            obs_arr = np.stack([obs[nm] for nm in names])
            act_arr = np.stack([acts[nm] for nm in names])
            rew_arr = np.array([rewards.get(nm, 0.0) for nm in names])
            next_arr = np.stack([next_obs[nm] for nm in names])
            done_arr = np.full(len(names), float(done), dtype=np.float32)
            trainer.remember(obs_arr, act_arr, rew_arr, next_arr, done_arr)
            trainer.total_steps += 1
            if trainer.total_steps >= trainer.start_steps:
                for _ in range(PILOT["updates_per_block"]):
                    stats = trainer.update()
                    if stats.get("critic_loss") is None:
                        break
                    rec["updates"] += 1
                    rec["_cl"] += stats["critic_loss"]
                    if stats.get("actor_loss") is not None:
                        rec["_al"] += stats["actor_loss"]
                        rec["_aln"] = rec.get("_aln", 0) + 1
                    for key in ("q1_mean", "q2_mean", "q_gap"):
                        value = stats.get(key)
                        if value is not None:
                            rec[key] = float(value)
        if done:
            break
        obs = next_obs
    if rec["updates"]:
        rec["critic_loss"] = rec["_cl"] / rec["updates"]
    if rec.get("_aln"):
        rec["actor_loss"] = rec["_al"] / rec["_aln"]
    rec["dev_penalty_mean"] /= max(N_BLOCKS, 1)
    rec["action_charge_mean"] /= max(N_BLOCKS, 1)
    rec["action_discharge_mean"] /= max(N_BLOCKS, 1)
    rec["end_soc_gap"] = max(
        (abs(env.prev_soc.get(a.name, a.storage.soc0) - a.storage.soc0)
         for a in env.rl_agents), default=0.0)
    return rec


def _trainer(env) -> MATD3:
    return MATD3(env, lr=PILOT["lr"], gamma=PILOT["gamma"],
                 tau=PILOT["tau"], policy_delay=PILOT["policy_delay"],
                 noise_std=PILOT["noise_std"],
                 noise_clip=PILOT["noise_clip"],
                 batch_size=PILOT["batch_size"],
                 buffer_capacity=PILOT["buffer_capacity"],
                 start_steps=PILOT["start_steps"],
                 dev_penalty=PILOT["dev_penalty"],
                 final_noise_std=PILOT["final_noise_std"],
                 noise_anneal_steps=PILOT["noise_anneal_steps"])


def train_run(dataset: str, seed: int, days: List[dict], log_path: Path) -> dict:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    order = [days[i] for i in rng.permutation(len(days))]
    env = envmod.build_env_for_pair(order[0])
    trainer = _trainer(env)
    started = time.perf_counter()
    log = Path(TRAIN_DIR) / f"{dataset}_seed{seed}.csv"
    if log.exists():
        log.unlink()
    for ep in range(PILOT["episodes"]):
        row = order[ep % len(order)]
        t0 = time.perf_counter()
        rec = run_episode(env, trainer, row, seed, rng, train=True)
        out = {"dataset": dataset, "seed": seed, "episode": ep,
               "load_date": row["load_date"],
               "price_date": row["price_date"],
               "total_steps": trainer.total_steps, **rec,
               "runtime_s": round(time.perf_counter() - t0, 2)}
        _append(log, out, TRAIN_COLUMNS)
        _log(f"{dataset} seed{seed} ep{ep:02d} {row['load_date']} "
             f"reward={rec['ep_reward']:.1f} raw={rec['ep_raw_profit']:.1f} "
             f"base={rec['ep_local_baseline']:.1f} "
             f"upd={rec['updates']} {out['runtime_s']}s", log_path)
    ckpt_dir = Path(CKPT_DIR) / f"{dataset}_seed{seed}"
    spec_by_agent = trainer.spec_by_agent
    for nm, actor in trainer.actors.items():
        save_policy(actor, str(ckpt_dir / f"{nm}.pt"),
                    obs_spec=OBS_V6, action_spec=spec_by_agent[nm])
    summary = {"dataset": dataset, "seed": seed,
               "episodes": PILOT["episodes"],
               "total_steps": trainer.total_steps,
               "runtime_s": round(time.perf_counter() - started, 1),
               "exposure": {d["load_date"]: sum(
                   1 for e in range(PILOT["episodes"])
                   if order[e % len(order)]["load_date"] == d["load_date"])
                   for d in days},
               "checkpoints": str(ckpt_dir)}
    (ckpt_dir / "run.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
    return summary


def _actions_for_day(env, trainer, row: dict) -> Dict[str, dict]:
    """Deterministic rollout; returns the policy's 96-period quotes."""
    agents, price = day_agents(row, env.config)
    envmod.configure_price(row["price_date"])
    env.set_agents(agents, day_id=row["price_date"])
    obs = env.reset()
    names = trainer.agent_names
    quotes = {nm: {"charge_price": np.zeros(env.T),
                   "discharge_price": np.zeros(env.T)} for nm in names}
    rewards = {nm: 0.0 for nm in names}
    raw = {nm: 0.0 for nm in names}
    base = {nm: 0.0 for nm in names}
    dev = 0.0
    clears = fallbacks = 0
    for block in range(N_BLOCKS):
        acts = {nm: trainer.select_action(obs[nm], nm, add_noise=False)
                for nm in names}
        t0 = block * BLOCK_SIZE
        n = min(BLOCK_SIZE, env.T - t0)
        for nm in names:
            quotes[nm]["charge_price"][t0:t0 + n] = acts[nm][0]
            quotes[nm]["discharge_price"][t0:t0 + n] = acts[nm][1]
        next_obs, rew, done, info = env.step(acts)
        clears += 1 if env._last_result is not None else 0
        fallbacks += 1 if env._last_result is None else 0
        for nm in names:
            rewards[nm] += rew.get(nm, 0.0)
            raw[nm] += info.get("raw_profit", {}).get(nm, 0.0)
            base[nm] += info.get("local_baseline_profit", {}).get(nm, 0.0)
        dev += _dev_penalty(env, obs, names, trainer)
        if done:
            break
        obs = next_obs
    return {"quotes": quotes, "agents": agents, "price": price,
            "ep_reward": sum(rewards.values()),
            "ep_raw_profit": sum(raw.values()),
            "ep_local_baseline": sum(base.values()),
            "dev_penalty_mean": dev / N_BLOCKS, "clears": clears,
            "fallbacks": fallbacks}


def settle_day(row: dict, rollout: dict, config) -> dict:
    """One full-day clear with the policy's own quotes; terminal SOC pinned."""
    agents = rollout["agents"]
    price = rollout["price"]
    result = clear_market(agents, screen.T, "DA", rollout["quotes"], config,
                          wholesale=price, horizon_type="full_day")
    if result is None:
        return {"optimal": False, "settled_total": float("nan"),
                "error": "clearing returned None"}
    net = screen.grid.build_base_network(config)
    checks = screen._checks(result, agents, config, price, net)
    solver = result.get("solver", {}).get("final", {})
    schedules = [result["schedules"][a.name] for a in agents
                 if a.name in result["schedules"]]
    return {
        "optimal": bool(solver.get("optimal")),
        "settled_total": sum(checks["payoffs"].values()),
        "payoffs": checks["payoffs"],
        "charge_mwh": checks["ess_charge_mwh"],
        "discharge_mwh": checks["ess_discharge_mwh"],
        "equivalent_cycles": (checks["ess_charge_mwh"]
                              + checks["ess_discharge_mwh"]) / (2.0 * 1.6),
        "terminal_soc_max_err": checks["terminal_soc_max_err"],
        "voltage_violation_cells": checks["voltage_violation_cells"],
        "line_overload_cells": checks["line_overload_cells"],
        "lmp_fallbacks": result.get("lmp_fallbacks", -1),
    }


def evaluate_run(dataset: str, seed: int, env, trainer,
                 days: List[dict]) -> List[dict]:
    b1 = {r["load_date"]: r for r in _read_csv(Path(B1_RESULTS))
          if r.get("arm") == "JN-Base"}
    b2a = {r["load_date"]: r for r in _read_csv(Path(BENCH_RAW))
           if r.get("arm") == "B2A" and r.get("split") in ("test",
                                                          "holiday_block")}
    rows = []
    for row in days:
        rollout = _actions_for_day(env, trainer, row)
        settled = settle_day(row, rollout, env.config)
        b1row = b1.get(row["load_date"], {})
        b2row = b2a.get(row["load_date"], {})
        payoffs = settled.get("payoffs", {})
        rows.append({
            "dataset": dataset, "seed": seed,
            "split": row.get("split", ""),
            "load_date": row["load_date"],
            "price_date": row["price_date"], "group": row["group"],
            "settled_total": settled.get("settled_total", float("nan")),
            **{f"settled_{u}": payoffs.get(u, float("nan"))
               for u in ("ESS9", "ESS14", "ESS17", "ESS27")},
            "b1_total": float(b1row.get("ess_total_payoff_cny", "nan")),
            **{f"b1_{u}": float(b1row.get(f"{u}_payoff_cny", "nan"))
               for u in ("ESS9", "ESS14", "ESS17", "ESS27")},
            "b2a_total": float(b2row.get("ess_total_payoff_cny", "nan")),
            "diff_vs_b1": (settled.get("settled_total", float("nan"))
                           - float(b1row.get("ess_total_payoff_cny", "nan"))),
            "diff_vs_b2a": (settled.get("settled_total", float("nan"))
                            - float(b2row.get("ess_total_payoff_cny", "nan"))),
            "charge_mwh": settled.get("charge_mwh", float("nan")),
            "discharge_mwh": settled.get("discharge_mwh", float("nan")),
            "equivalent_cycles": settled.get("equivalent_cycles",
                                             float("nan")),
            "ep_reward": rollout["ep_reward"],
            "ep_raw_profit": rollout["ep_raw_profit"],
            "ep_local_baseline": rollout["ep_local_baseline"],
            "dev_penalty_mean": rollout["dev_penalty_mean"],
            "terminal_soc_max_err": settled.get("terminal_soc_max_err",
                                                float("nan")),
            "voltage_violation_cells": settled.get("voltage_violation_cells",
                                                   float("nan")),
            "line_overload_cells": settled.get("line_overload_cells",
                                               float("nan")),
            "lmp_fallbacks": settled.get("lmp_fallbacks", -1),
            "optimal": settled.get("optimal", False),
        })
    return rows


def smoke(log_path: Path) -> int:
    split = json.loads(Path(COMMON).read_text(encoding="utf-8"))
    by_load = {r["load_date"]: r for r in _pairs()}
    row = by_load[split["common"]["train"][0].split("|")[0]]
    env = envmod.build_env_for_pair(row)
    trainer = _trainer(env)
    envmod.configure_price(row["price_date"])
    obs = env.reset()
    checks = {
        "obs_dim": env.get_state_dim() == OBS_V6.total_dim,
        "n_agents": len(trainer.agent_names) == 4,
        "obs_finite": all(np.all(np.isfinite(obs[n])) for n in trainer.agent_names),
    }
    # Machinery check only: a smaller batch and warmup so one episode actually
    # reaches a gradient update. These overrides are not the pilot budget.
    trainer.batch_size = 8
    trainer.start_steps = BLOCK_SIZE
    rec = run_episode(env, trainer, row, PILOT["seeds"][0],
                      np.random.default_rng(0), train=True)
    checks["episode_completed"] = rec["clears"] == N_BLOCKS
    checks["rewards_finite"] = np.isfinite(rec["ep_reward"])
    checks["buffer_filled"] = len(trainer.buffer) >= N_BLOCKS
    checks["updates_ran"] = rec["updates"] > 0
    ckpt = Path(OUT) / "smoke" / "ESS9.pt"
    save_policy(trainer.actors["ESS9"], str(ckpt),
                obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)
    loaded = load_policy(str(ckpt), env.get_state_dim(),
                         trainer.bounds_by_agent["ESS9"],
                         obs_spec=OBS_V6,
                         action_spec=ACTION_STORAGE_QUOTE_V2)
    probe = torch_probe(loaded, obs["ESS9"])
    checks["checkpoint_roundtrip"] = np.allclose(
        probe, trainer.select_action(obs["ESS9"], "ESS9"))
    _log(f"smoke: {checks}", log_path)
    _log(f"smoke episode: {rec}", log_path)
    return 0 if all(checks.values()) else 1


def torch_probe(actor, obs) -> np.ndarray:
    import torch
    with torch.no_grad():
        return actor(torch.as_tensor(obs, dtype=torch.float32)
                     .unsqueeze(0)).squeeze(0).numpy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "eval", "all"),
                        default="smoke")
    parser.add_argument("--datasets", default="NEAR,FULL96")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path.cwd()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    if args.episodes:
        PILOT["episodes"] = args.episodes
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else PILOT["seeds"])
    log_path = root / LOG
    pool = pools()
    test_days = [r for r in _pairs()
                 if any(r["load_date"] == i.split("|")[0]
                        for i in json.loads(
                            Path(COMMON).read_text(encoding="utf-8")
                        )["common"]["test"])]
    holiday_days = [r for r in _pairs()
                    if any(r["load_date"] == i.split("|")[0]
                           for i in json.loads(
                               Path(COMMON).read_text(encoding="utf-8")
                           )["common"]["holiday_block"])]
    for row in test_days:
        row["split"] = "test"
    for row in holiday_days:
        row["split"] = "holiday_block"

    config_payload = {
        "pilot": PILOT, "seeds": seeds,
        "pools": {k: [d["load_date"] for d in v] for k, v in pool.items()},
        "evaluation": {"test": [d["load_date"] for d in test_days],
                       "holiday_block": [d["load_date"] for d in holiday_days]},
        "hashes": {
            "pairing": _sha256(root / PAIRING),
            "bus_loads": _sha256(root / BUS_LOADS),
            "price_curves": _sha256(root / PRICE_CURVES),
            "common_split": _sha256(root / COMMON),
        },
    }
    (root / CONFIG_PATH).write_text(json.dumps(config_payload, indent=2),
                                    encoding="utf-8")

    if args.mode in ("smoke", "all"):
        rc = smoke(log_path)
        if rc:
            return rc
    summaries = []
    if args.mode in ("train", "all"):
        for dataset in args.datasets.split(","):
            for seed in seeds:
                summaries.append(train_run(dataset, seed, pool[dataset],
                                           log_path))
    if args.mode in ("eval", "all"):
        eval_path = root / EVAL_CSV
        if eval_path.exists():
            eval_path.unlink()
        for dataset in args.datasets.split(","):
            for seed in seeds:
                ckpt_dir = Path(CKPT_DIR) / f"{dataset}_seed{seed}"
                env = envmod.build_env_for_pair(pool[dataset][0])
                trainer = _trainer(env)
                for nm in trainer.agent_names:
                    trainer.actors[nm] = load_policy(
                        str(ckpt_dir / f"{nm}.pt"), env.get_state_dim(),
                        trainer.bounds_by_agent[nm], obs_spec=OBS_V6,
                        action_spec=trainer.spec_by_agent[nm])
                for r in evaluate_run(dataset, seed, env, trainer,
                                      test_days + holiday_days):
                    _append(eval_path, r, EVAL_COLUMNS)
        print(f"wrote {eval_path}")
    (root / OUT / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
