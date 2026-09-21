# audit_protocol_smoke.py
"""Check one training run's artifacts against the frozen protocol.

A run that rotated through its days, validated on held-out days and recorded
what it earned is indistinguishable, from the outside, from one that trained on
a single day and validated on itself. Both produce a metrics.csv, an eval.csv
and a plausible curve. This reads the artifacts and answers the questions the
protocol is only worth having if they are answered:

  day rotation      each cycle covered its twenty training days exactly once,
                    in the order the declared schedule gives for this seed
  held-out days     validation rows name only validation days; no test day
                    appears anywhere in the run
  rng isolation     the recorded day order is the one `day_rotation` derives
                    from the run seed alone
  counterfactual    every validation row satisfies delta_pi == pi_ai -
                    pi_baseline, and the cycle means are their mean
  kpi completeness  the fields the research question is answered with exist
  checkpoint        the saved policies reload and reproduce the run's own
                    last validation pass

Exit status is 0 when every check passes and 1 otherwise, so a caller can gate
a longer run on it.

Usage:
    PYTHONPATH=src python src/audit_protocol_smoke.py outputs/rl/<run_id>
    PYTHONPATH=src python src/audit_protocol_smoke.py <run_id> --checkpoint-check
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import day_rotation
from rl_env import N_BLOCKS

#: A recorded delta must equal the difference of the two arms it was built
#: from, to within money's own printing precision.
DELTA_TOLERANCE = 1e-6


def _load_frame(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _day_rows(evals):
    """The per-day validation rows, separated from the summary rows."""
    if evals is None or "day_id" not in evals.columns:
        return evals.iloc[0:0] if evals is not None else None
    return evals[evals["day_id"].notna()]


def check_day_rotation(metrics, run_seed: int) -> dict:
    """Every cycle ran its twenty training days exactly once, in order."""
    problems: List[str] = []
    cycles: Dict[int, List[int]] = {}
    if metrics is None or "day_id" not in metrics.columns:
        return {"ok": False, "problems": ["metrics.csv has no day_id column"],
                "cycles": {}}
    if metrics["day_id"].isna().any():
        problems.append("some episodes recorded no day_id")
    train = day_rotation.split_of("train")
    for cycle, group in metrics.dropna(subset=["day_id"]).groupby("cycle"):
        recorded = [int(d) for d in group.sort_values("episode")["day_id"]]
        cycles[int(cycle)] = recorded
        if sorted(recorded) != sorted(train):
            problems.append(
                f"cycle {int(cycle)} covered {sorted(recorded)}, not the "
                f"twenty training days")
        expected = day_rotation.cycle_days(run_seed, int(cycle))
        if recorded != expected:
            problems.append(
                f"cycle {int(cycle)} ran {recorded}, but the schedule for "
                f"seed {run_seed} gives {expected}")
    return {"ok": not problems, "problems": problems, "cycles": cycles}


def check_held_out_isolation(metrics, evals) -> dict:
    """No held-out day is ever trained on or validated out of its split."""
    problems: List[str] = []
    train = set(day_rotation.split_of("train"))
    validation = set(day_rotation.validation_days())
    test = set(day_rotation.test_days())

    if metrics is not None and "day_id" in metrics.columns:
        leaked = set(metrics["day_id"].dropna().astype(int)) - train
        if leaked:
            problems.append(f"trained on non-training days: {sorted(leaked)}")

    days = _day_rows(evals)
    if days is not None and len(days):
        named = set(days["day_id"].astype(int))
        if named - validation:
            problems.append(
                f"validated on days outside the validation set: "
                f"{sorted(named - validation)}")
        if named & test:
            problems.append(f"test days reached validation: "
                            f"{sorted(named & test)}")
    return {"ok": not problems, "problems": problems}


def check_counterfactual(evals, kpi) -> dict:
    """The recorded pairs are consistent with their own definition."""
    problems: List[str] = []
    days = _day_rows(evals)
    if days is None or not len(days):
        return {"ok": False, "problems": ["no per-day validation rows"],
                "n_days": 0, "stats": {}}

    worst = float(np.max(np.abs(
        days["delta_pi"] - (days["pi_ai"] - days["pi_baseline"]))))
    if worst > DELTA_TOLERANCE:
        problems.append(f"delta_pi differs from pi_ai - pi_baseline by {worst}")

    # The rate divides by the magnitude of the baseline, so a loss-making
    # baseline still reads the right way round.
    expected_rate = days["delta_pi"] / (days["pi_baseline"].abs() + 1e-6)
    rate_gap = float(np.max(np.abs(days["delta_pi_percent"] - expected_rate)))
    if rate_gap > 1e-6:
        problems.append(f"delta_pi_percent off by {rate_gap}")

    # Each cycle's recorded mean must be the mean of its own day rows.
    for entry in (kpi or {}).get("cycles", []):
        cycle = entry["cycle"]
        sub = days[days["cycle"] == cycle]
        if not len(sub):
            problems.append(f"cycle {cycle} has no day rows")
            continue
        for column, field in (("delta_pi", "mean_delta_pi"),
                              ("pi_ai", "mean_pi_ai"),
                              ("pi_baseline", "mean_pi_baseline")):
            if entry.get(field) is None:
                continue
            if abs(float(np.mean(sub[column])) - float(entry[field])) > 1e-6:
                problems.append(
                    f"cycle {cycle} {field}={entry[field]} is not the mean of "
                    f"its {len(sub)} day rows")

    stats = {
        "n_days": int(len(days)),
        "mean_pi_ai": float(days["pi_ai"].mean()),
        "mean_pi_baseline": float(days["pi_baseline"].mean()),
        "mean_delta_pi": float(days["delta_pi"].mean()),
        "median_delta_pi": float(days["delta_pi"].median()),
        "p_delta_pi_positive": float((days["delta_pi"] > 0).mean()),
    }
    return {"ok": not problems, "problems": problems,
            "n_days": int(len(days)), "stats": stats}


def check_kpi_completeness(kpi) -> dict:
    """The fields the research question is answered with are all present."""
    required = ("mean_pi_ai", "mean_pi_baseline", "mean_delta_pi",
                "p_delta_pi_positive")
    problems = []
    cycles = (kpi or {}).get("cycles", [])
    if not cycles:
        problems.append("kpi.json carries no per-cycle validation block")
    for entry in cycles:
        missing = [f for f in required if entry.get(f) is None]
        if missing:
            problems.append(f"cycle {entry.get('cycle')} missing {missing}")
    return {"ok": not problems, "problems": problems, "n_cycles": len(cycles)}


#: Columns the trainer only produces once a gradient step has run. Before the
#: warmup ends these are absent, and absence is not divergence -- a CSV cannot
#: tell a written NaN from an empty cell, so the distinction is made by where
#: the gaps fall rather than by their value.
_LEARNING_ONLY_COLUMNS = ("critic_loss", "actor_loss", "q1_mean", "q2_mean",
                          "q_gap")


def check_training_health(metrics, cli_args: dict = None) -> dict:
    """Nothing in the run went non-finite, and no block fell back.

    An infinite value anywhere is a failure. An *absent* loss column is not:
    before `start_steps` random steps have elapsed the trainer takes no
    gradient step, so it has no loss to write. Those gaps are required to be
    confined to the warmup episodes and to be all-or-nothing per column; a gap
    after the warmup would be a dropped update, which is a different fault.
    """
    problems: List[str] = []
    if metrics is None or not len(metrics):
        return {"ok": False, "problems": ["metrics.csv is empty"]}

    cli_args = cli_args or {}
    start_steps = int(cli_args.get("start_steps", 0) or 0)
    warmup = start_steps // N_BLOCKS

    infinite = {}
    absent = {}
    numeric = metrics.select_dtypes(include=[np.number])
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        written = values[~np.isnan(values)]
        bad = int(np.count_nonzero(np.isinf(written)))
        if bad:
            infinite[column] = bad
        missing = int(np.count_nonzero(np.isnan(values)))
        if missing:
            absent[column] = missing
    if infinite:
        problems.append(f"infinite values in {infinite}")

    if warmup:
        early = metrics[metrics["episode"] <= warmup]
        late = metrics[metrics["episode"] > warmup]
        for column in _LEARNING_ONLY_COLUMNS:
            if column not in metrics.columns:
                continue
            if early[column].notna().any():
                problems.append(
                    f"{column} was written during the warmup, before any "
                    f"gradient step")
            if late[column].isna().any():
                dropped = int(late[column].isna().sum())
                problems.append(
                    f"{column} is absent on {dropped} episodes after the "
                    f"warmup, which is a dropped update rather than a "
                    f"warmup gap")

    for column in ("mean_reward", "clearing_objective", "re_rate"):
        if column in metrics.columns and metrics[column].isna().any():
            problems.append(f"{column} is absent on some episode")

    if "capture_fellbacks" in metrics.columns:
        fell_back = int((metrics["capture_fellbacks"].fillna(0) > 0).sum())
        if fell_back:
            problems.append(f"{fell_back} episodes had a clearing fallback")
    else:
        fell_back = None
    return {"ok": not problems, "problems": problems,
            "fallback_episodes": fell_back, "warmup_episodes": warmup,
            "absent_cells": absent, "n_episodes": int(len(metrics))}


def check_checkpoint(run_dir: str, manifest: dict, evals, kpi,
                     run_seed: int) -> dict:
    """The saved policies reload and reproduce the run's last validation.

    Reloading and re-running the held-out days is the only check that says the
    file on disk is the policy the run ended with rather than merely a file
    with the right name. The evaluation is deterministic, so the reproduced
    KPIs must match the recorded ones exactly.
    """
    from counterfactual_eval import validation_kpis, reset_baseline_cache
    from eval_agents import default_eval_config, load_policies_from_dir
    from rl_env import bounds_tensor
    from rl_spec import get_action_spec, get_obs_spec

    final = manifest.get("policy_paths", {}).get("final") or []
    missing = [p for p in final if not os.path.exists(p)]
    if missing:
        return {"ok": False, "problems": [f"missing policy files: {missing}"],
                "paths": final, "matched": False}

    days = _day_rows(evals)
    if days is None or not len(days):
        return {"ok": False, "problems": ["no validation rows to reproduce"],
                "paths": final, "matched": False}
    last_cycle = int(days["cycle"].max())

    # The specs come from the manifest by name, so a policy is reloaded into
    # the shape and box it was trained in rather than into today's defaults.
    obs_spec = get_obs_spec(manifest["obs_spec"]["name"])
    action_spec = get_action_spec(manifest["action_spec"]["name"])
    policy_dir = os.path.dirname(os.path.abspath(final[0]))
    actors = load_policies_from_dir(
        policy_dir, obs_spec.total_dim, bounds_tensor(action_spec),
        obs_spec=obs_spec, action_spec=action_spec)
    if not actors:
        return {"ok": False, "problems": [f"no policies loaded from "
                                          f"{policy_dir!r}"],
                "paths": final, "matched": False}

    reset_baseline_cache()
    reproduced = validation_kpis(
        actors, default_eval_config(),
        manifest["cli_args"].get("train_scenario") or "typical_day",
        sorted(actors),
        sorted(int(d) for d in days[days["cycle"] == last_cycle]["day_id"]),
        obs_spec=obs_spec, action_spec=action_spec)

    entry = next((c for c in (kpi or {}).get("cycles", [])
                  if c["cycle"] == last_cycle), None)
    problems = []
    matched = False
    if entry is None:
        problems.append(f"cycle {last_cycle} has no recorded kpi block")
    else:
        gap = abs(reproduced["mean_delta_pi"] - float(entry["mean_delta_pi"]))
        matched = gap <= 1e-6
        if not matched:
            problems.append(
                f"reloaded policies give mean_delta_pi="
                f"{reproduced['mean_delta_pi']}, recorded "
                f"{entry['mean_delta_pi']} (gap {gap})")
    return {"ok": not problems, "problems": problems, "paths": final,
            "matched": matched, "cycle": last_cycle,
            "reproduced_mean_delta_pi": reproduced["mean_delta_pi"]}


#: An action or a power within this of a bound counts as sitting on it.
BOUND_TOLERANCE = 1e-6


def check_policy_behaviour(run_dir: str, manifest: dict, evals) -> dict:
    """Replay the saved policies over the held-out days and watch what they do.

    The trainer records no per-block action or SOC, so the only way to check
    the trained policy against the action box and the storage bounds is to run
    it again and look. This replays the *saved* policy -- no gradient step is
    taken -- over one day per held-out day id.

    Checks, per block: every declared quote inside its own agent's box, every
    schedule free of simultaneous charge and discharge, every state of charge
    inside `[soc_min, soc_max]`, and no non-finite value anywhere.
    """
    from eval_agents import actor_predict, default_eval_config, \
        load_policies_from_dir
    from rl_env import BiddingEnv, N_BLOCKS, bounds_tensor
    from rl_spec import get_action_spec, get_obs_spec
    from scenarios import get_scenario

    final = manifest.get("policy_paths", {}).get("final") or []
    if not final:
        return {"ok": False, "problems": ["no saved policies to replay"],
                "blocks": 0}
    obs_spec = get_obs_spec(manifest["obs_spec"]["name"])
    action_spec = get_action_spec(manifest["action_spec"]["name"])
    policies = load_policies_from_dir(
        os.path.dirname(os.path.abspath(final[0])), obs_spec.total_dim,
        bounds_tensor(action_spec), obs_spec=obs_spec,
        action_spec=action_spec)
    names = sorted(policies)
    if not names:
        return {"ok": False, "problems": ["no policies loaded"], "blocks": 0}

    scenario = manifest["cli_args"].get("train_scenario") or "typical_day"
    days = _day_rows(evals)
    day_ids = sorted(set(int(d) for d in days["day_id"])) if days is not None \
        and len(days) else day_rotation.validation_days()

    problems: List[str] = []
    out_of_box = 0
    simultaneity = 0
    soc_violations = 0
    non_finite = 0
    blocks = 0
    soc_min = float("inf")
    soc_max = float("-inf")

    for day_id in day_ids:
        config = default_eval_config()
        agents, _wholesale = get_scenario(scenario, T=96, config=config,
                                          day_id=day_id)
        env = BiddingEnv(agents, config, rl_agent_names=names,
                         obs_spec=obs_spec, action_spec=action_spec)
        # Each agent's own box, so a quote is judged against the space it was
        # trained in rather than the environment's representative one.
        bounds = {a.name: env.action_bounds_for(a).numpy()
                  for a in env.rl_agents}
        obs = env.reset()
        for _ in range(N_BLOCKS):
            acts = {nm: actor_predict(policies[nm], obs[nm]) for nm in names}
            for nm in names:
                lo, hi = bounds[nm][0], bounds[nm][1]
                a = np.asarray(acts[nm], dtype=float)
                if not np.all(np.isfinite(a)):
                    non_finite += 1
                if np.any(a < lo - BOUND_TOLERANCE) or \
                        np.any(a > hi + BOUND_TOLERANCE):
                    out_of_box += 1
            obs, _rewards, done, info = env.step(acts)
            blocks += 1
            result = env._last_result or {}
            for nm in names:
                sched = (result.get("schedules") or {}).get(nm)
                if sched is None:
                    continue
                soc = np.asarray(sched["soc"], dtype=float)
                p_ch = np.asarray(sched.get("p_ch", []), dtype=float)
                p_dis = np.asarray(sched.get("p_dis", []), dtype=float)
                if not (np.all(np.isfinite(soc)) and np.all(np.isfinite(p_ch))
                        and np.all(np.isfinite(p_dis))):
                    non_finite += 1
                    continue
                storage = next(a.storage for a in env.all_agents
                               if a.name == nm)
                soc_min = min(soc_min, float(soc.min()))
                soc_max = max(soc_max, float(soc.max()))
                if (soc.min() < storage.soc_min - BOUND_TOLERANCE
                        or soc.max() > storage.soc_max + BOUND_TOLERANCE):
                    soc_violations += 1
                if p_ch.size and p_dis.size and \
                        float(np.minimum(p_ch, p_dis).max()) > BOUND_TOLERANCE:
                    simultaneity += 1
            if done:
                break

    if out_of_box:
        problems.append(f"{out_of_box} declared quotes fell outside their box")
    if soc_violations:
        problems.append(f"{soc_violations} schedules left the SOC bounds")
    if simultaneity:
        problems.append(f"{simultaneity} schedules charged and discharged at "
                        f"once")
    if non_finite:
        problems.append(f"{non_finite} non-finite actions or schedules")
    return {"ok": not problems, "problems": problems, "days": len(day_ids),
            "blocks": blocks, "actions_out_of_box": out_of_box,
            "simultaneous_charge_discharge": simultaneity,
            "soc_violations": soc_violations, "non_finite": non_finite,
            "soc_min": soc_min, "soc_max": soc_max}


def audit(run_dir: str, run_seed: Optional[int] = None,
          checkpoint_check: bool = False,
          policy_check: bool = False) -> dict:
    root = run_dir
    if not os.path.isdir(root):
        candidate = os.path.join("outputs", "rl", run_dir)
        if os.path.isdir(candidate):
            root = candidate
        else:
            raise SystemExit(f"no run artifacts at {run_dir!r}")

    with open(os.path.join(root, "manifest.json")) as fh:
        manifest = json.load(fh)
    seed = int(run_seed if run_seed is not None
               else manifest["cli_args"].get("seed", 1))
    metrics = _load_frame(os.path.join(root, "metrics.csv"))
    evals = _load_frame(os.path.join(root, "eval.csv"))
    kpi_path = os.path.join(root, "kpi.json")
    kpi = {}
    if os.path.exists(kpi_path):
        with open(kpi_path) as fh:
            kpi = json.load(fh)

    report = {
        "run_id": manifest.get("run_id"),
        "seed": seed,
        "git_sha": manifest.get("git_sha"),
        "day_rotation": check_day_rotation(metrics, seed),
        "held_out_isolation": check_held_out_isolation(metrics, evals),
        "training_health": check_training_health(metrics,
                                                 manifest.get("cli_args")),
        "counterfactual": check_counterfactual(evals, kpi),
        "kpi_completeness": check_kpi_completeness(kpi),
    }
    if policy_check:
        report["policy_behaviour"] = check_policy_behaviour(root, manifest,
                                                            evals)
    if checkpoint_check:
        report["checkpoint"] = check_checkpoint(root, manifest, evals, kpi, seed)
    report["ok"] = all(v["ok"] for v in report.values()
                       if isinstance(v, dict) and "ok" in v)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run id or path to outputs/rl/<run_id>")
    parser.add_argument("--seed", type=int, default=None,
                        help="run seed; default: the one in the manifest")
    parser.add_argument("--checkpoint-check", action="store_true",
                        help="reload the saved policies and reproduce the "
                             "last validation pass (costs a full evaluation)")
    parser.add_argument("--policy-check", action="store_true",
                        help="replay the saved policies over the held-out days "
                             "and check the action box, SOC bounds and "
                             "charge/discharge exclusivity per block")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = audit(args.run, run_seed=args.seed,
                   checkpoint_check=args.checkpoint_check,
                   policy_check=args.policy_check)

    for name, section in report.items():
        if not isinstance(section, dict) or "ok" not in section:
            continue
        mark = "PASS" if section["ok"] else "FAIL"
        print(f"{mark}  {name}")
        for problem in section.get("problems", []):
            print(f"        {problem}")
    stats = report["counterfactual"].get("stats") or {}
    if stats:
        print("\nvalidation statistics over "
              f"{stats['n_days']} held-out day-pairs:")
        for key in ("mean_pi_ai", "mean_pi_baseline", "mean_delta_pi",
                    "median_delta_pi", "p_delta_pi_positive"):
            print(f"  {key:<22s} {stats[key]:>12.4f}")
    health = report["training_health"]
    if health.get("fallback_episodes") is not None:
        print(f"\nclearing fallbacks: {health['fallback_episodes']}")
    behaviour = report.get("policy_behaviour")
    if behaviour:
        print(f"policy replay over {behaviour.get('days')} held-out days "
              f"({behaviour.get('blocks')} blocks):")
        for key in ("actions_out_of_box", "simultaneous_charge_discharge",
                    "soc_violations", "non_finite"):
            print(f"  {key:<30s} {behaviour.get(key)}")
        print(f"  SOC range: {behaviour.get('soc_min'):.4f} … "
              f"{behaviour.get('soc_max'):.4f}")
    print(f"\nOVERALL: {'PASS' if report['ok'] else 'FAIL'}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
