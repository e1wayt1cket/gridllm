# smoke_matd3.py
"""Environment and policy instrumentation over the calibrated B3 day.

Not a training run and not the experiment. It walks the facilities through the
environment for a fixed number of episodes and reports these directly rather
than inferring them from a reward curve:

  no NaN                  in rewards, losses, actions or the clearing
  valid actions           every declared quote inside its own box
  valid SOC               every state of charge inside [soc_min, soc_max]
  stable clearing         every block clears, with no fallback
  terminal SOC equality   the settled day starts and ends at the same SOC
  non-degenerate actions  the fleet does not declare one identical quote
  no corner collapse      the policy does not sit on its truthful corner

It never calls `trainer.remember()`, so the replay buffer stays empty,
`MATD3.update()` returns before its first gradient step, and the quotes recorded
here come from the seeded actor plus exploration noise. The readings therefore
describe the environment and the action box -- that the box is reachable, that
the clearing is stable, that the SOC constraints hold -- and not a learned
policy. `audit_protocol_smoke.check_policy_behaviour` is where the checks that
are about a trained policy live; it replays a finished run's saved policies.

A reward curve can show none of the seven. A run that has collapsed to the
truthful corner still produces a flat, plausible-looking reward.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

import money
from baseline_ladder import BASE_SCENARIO, PRIMARY_BAND, build_rung_agents
from rl_bidding import MATD3, agent_action_spaces
from rl_td3 import CALIBRATED_DEV_PENALTY
from rl_env import BiddingEnv, N_BLOCKS, BLOCK_SIZE
from rl_spec import ACTION_STORAGE_QUOTE_V2

#: An action within this fraction of the box width from truthful counts as
#: parked on the corner.
CORNER_TOLERANCE = 0.02

#: Below this spread across the fleet's quotes, the policies have collapsed
#: onto one another and there is no bidding left to measure.
MIN_QUOTE_SPREAD = 1.0


def _episode(env, trainer, warmup: bool) -> dict:
    """One training episode, with everything the checks need recorded."""
    obs = env.reset()
    names = trainer.agent_names
    rec = {"actions": [], "soc": [], "clears": 0, "fallbacks": 0,
           "nan": False, "out_of_box": 0, "soc_out_of_range": 0}
    for _ in range(N_BLOCKS):
        acts = {}
        for nm in names:
            if warmup and trainer.total_steps < trainer.start_steps:
                acts[nm] = np.random.uniform(
                    trainer.action_low.numpy(), trainer.action_high.numpy()
                ).astype(np.float32)
            else:
                acts[nm] = trainer.select_action(obs[nm], nm, add_noise=True)
        next_obs, rewards, done, info = env.step(acts)
        trainer.total_steps += 1

        if any(not np.all(np.isfinite(v)) for v in rewards.values()):
            rec["nan"] = True
        for nm in names:
            a = np.asarray(acts[nm], dtype=float)
            if not np.all(np.isfinite(a)):
                rec["nan"] = True
            rec["actions"].append(a)
            lo = trainer.bounds_by_agent[nm][0].numpy()
            hi = trainer.bounds_by_agent[nm][1].numpy()
            if np.any(a < lo - 1e-6) or np.any(a > hi + 1e-6):
                rec["out_of_box"] += 1

        result = env._last_result
        if result is None:
            rec["fallbacks"] += 1
        else:
            rec["clears"] += 1
            for nm in names:
                sched = (result.get("schedules") or {}).get(nm)
                if sched is None:
                    continue
                soc = np.asarray(sched["soc"], dtype=float)
                rec["soc"].append(soc)
                storage = next(a.storage for a in env.all_agents
                               if a.name == nm)
                if (np.nanmin(soc) < storage.soc_min - 1e-6
                        or np.nanmax(soc) > storage.soc_max + 1e-6):
                    rec["soc_out_of_range"] += 1
        if done:
            break
        obs = next_obs
    return rec


def _terminal_soc_check(config, agents, names, declared) -> dict:
    """The settled day must start and end at the same state of charge.

    This is the property the paired comparison rests on: without it an arm can
    win by simply ending the day with a fuller battery.
    """
    from counterfactual_eval import settle_day

    _wholesale_agents, wholesale, _cfg = build_rung_agents(
        "B3", PRIMARY_BAND, 123)
    profits, clean, result = settle_day(
        agents, config, names, declared, wholesale, money.PERIODS_PER_DAY)
    if result is None:
        return {"settled_clean": False}
    worst = 0.0
    for nm in names:
        sched = result["schedules"].get(nm)
        if sched is None:
            continue
        soc = np.asarray(sched["soc"], dtype=float)
        agent = next(a for a in agents if a.name == nm)
        worst = max(worst, abs(float(soc[-1]) - float(agent.storage.soc0)))
    return {"settled_clean": bool(clean), "max_terminal_soc_gap": worst}


def run_seed(seed: int, episodes: int, dev_penalty: float) -> dict:
    agents, _wholesale, config = build_rung_agents("B3", PRIMARY_BAND, seed)
    names = [a.name for a in agents
             if getattr(a, "participant_type", "") == "storage"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BiddingEnv(agents, config, rl_agent_names=names)
    trainer = MATD3(env, dev_penalty=dev_penalty, batch_size=64)

    specs, bounds, truthful, _ = agent_action_spaces(env)
    rec = {"actions": [], "soc": [], "clears": 0, "fallbacks": 0, "nan": False,
           "out_of_box": 0, "soc_out_of_range": 0}
    losses: List[float] = []
    rewards: List[float] = []
    for ep in range(episodes):
        trainer.start_steps = max(2 * N_BLOCKS, trainer.batch_size)
        part = _episode(env, trainer, warmup=True)
        for key in ("clears", "fallbacks", "out_of_box", "soc_out_of_range"):
            rec[key] += part[key]
        rec["nan"] = rec["nan"] or part["nan"]
        rec["actions"].extend(part["actions"])
        rec["soc"].extend(part["soc"])
        for _ in range(N_BLOCKS):
            info = trainer.update()
            rewards.append(info.get("reward_scale"))
        if info.get("critic_loss") is not None:
            losses.append(info["critic_loss"])

    actions = np.asarray(rec["actions"], dtype=float)
    width = np.array([ACTION_STORAGE_QUOTE_V2.bounds[n][1]
                      - ACTION_STORAGE_QUOTE_V2.bounds[n][0]
                      for n in ACTION_STORAGE_QUOTE_V2.action_names])
    deviation = np.abs(actions - np.zeros(2)) / width
    # A window clear returns only its own committed periods, so the SOC traces
    # differ in length from block to block and are flattened rather than
    # stacked.
    socs = (np.concatenate([np.ravel(s) for s in rec["soc"]])
            if rec["soc"] else np.zeros(1))

    declared = {nm: {k: np.asarray(v, dtype=float)
                     for k, v in env.current_actions[nm].items()}
                for nm in names}
    terminal = _terminal_soc_check(config, agents, names, declared)

    return {
        "seed": seed,
        "episodes": episodes,
        "dev_penalty": dev_penalty,
        "nan_seen": bool(rec["nan"])
        or any(not np.isfinite(x) for x in losses),
        "actions_out_of_box": rec["out_of_box"],
        "soc_out_of_range": rec["soc_out_of_range"],
        "soc_min_seen": float(socs.min()),
        "soc_max_seen": float(socs.max()),
        "blocks_cleared": rec["clears"],
        "clearing_fallbacks": rec["fallbacks"],
        "clearing_stable": rec["fallbacks"] == 0,
        "mean_deviation": float(deviation.mean()),
        "corner_fraction": float((deviation < CORNER_TOLERANCE).mean()),
        "quote_spread_mean": float(actions.std(axis=0).mean()),
        "quote_min": float(actions.min()),
        "quote_max": float(actions.max()),
        "final_critic_loss": float(losses[-1]) if losses else None,
        "terminal_soc": terminal,
    }


def evaluate_checks(rows: List[dict]) -> dict:
    """Every check, with its verdict, over all seeds."""
    checks = {
        "no_nan": all(not r["nan_seen"] for r in rows),
        "valid_actions": all(r["actions_out_of_box"] == 0 for r in rows),
        "valid_soc": all(r["soc_out_of_range"] == 0 for r in rows),
        "stable_clearing": all(r["clearing_stable"] for r in rows),
        "terminal_soc_equality": all(
            r["terminal_soc"].get("settled_clean")
            and r["terminal_soc"].get("max_terminal_soc_gap", 1.0) < 1e-6
            for r in rows),
        "non_degenerate_actions": all(
            r["quote_spread_mean"] > MIN_QUOTE_SPREAD for r in rows),
        "no_corner_collapse": all(
            r["corner_fraction"] < 0.5 for r in rows),
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--dev-penalty", type=float,
                        default=CALIBRATED_DEV_PENALTY)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = [run_seed(s, args.episodes, args.dev_penalty) for s in seeds]
    doc = {"rows": rows, "episodes": args.episodes,
           "dev_penalty": args.dev_penalty,
           "scenario": BASE_SCENARIO, "rung": "B3",
           "band": list(PRIMARY_BAND)}
    doc.update(evaluate_checks(rows))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")

    for r in rows:
        print(f"seed {r['seed']}: nan={r['nan_seen']} "
              f"out_of_box={r['actions_out_of_box']} "
              f"soc_oob={r['soc_out_of_range']} "
              f"clears={r['blocks_cleared']} fallbacks={r['clearing_fallbacks']} "
              f"dev={r['mean_deviation']:.3f} corner={r['corner_fraction']:.1%} "
              f"spread={r['quote_spread_mean']:.2f} "
              f"soc=[{r['soc_min_seen']:.3f},{r['soc_max_seen']:.3f}]")
    print()
    for name, ok in doc["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if doc["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
