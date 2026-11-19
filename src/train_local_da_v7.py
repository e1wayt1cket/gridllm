"""Formal v7 training entry: one seed over the local day-ahead market.

Each round exposes every train day once, then validates on the validation
block by mean fleet delta G from settlement -- never a training reward. The
test and holiday blocks are recorded but never cleared. Every bound on the run
(min rounds, max rounds, exposures, max episodes, patience) is supplied by the
launcher from the frozen config; a resume refuses a directory whose stop rule
differs from the one it was started with.

The scheduling helpers are the frozen ones from `train_da_gate`; this module
imports nothing from the v6 data stack, so a v7 run cannot read a v6 input.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

import grid  # noqa: F401  ortools must load before pandas on this platform
from local_da_audit import SCREEN
from local_da_days import LIBRARY, load_pair, train_pairs
from rl_da_env import N_BLOCKS, EpisodeClearingError, LocalDABiddingEnv
from rl_da_gate_v7 import DAGateTrainerV7, DAV7ExperimentConfig

ROOT = Path.cwd()
RUNS = ROOT / "runs"
SNAPSHOTS = ROOT / "snapshots"
ENV_NAME = "local_da_v7"


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "working-tree"


# ----------------------------------------------------------------------
# Stop rule. The same functions `train_da_gate` freezes; a partial round
# never satisfies the round floor, and the cap permits `max_rounds` rounds.
# ----------------------------------------------------------------------
def no_improvement_streak(evaluations: Sequence[dict]) -> int:
    if not evaluations:
        return 0
    best_idx = max(range(len(evaluations)),
                   key=lambda i: evaluations[i].get("mean_fleet_delta",
                                                    -math.inf))
    return len(evaluations) - 1 - best_idx


def episode_budget(order_size: int, exposures: int,
                   max_episodes: int = 0) -> int:
    budget = order_size * exposures
    if max_episodes:
        budget = min(budget, max_episodes)
    return budget


def completed_rounds(episodes: int, order_size: int) -> int:
    return int(episodes) // order_size


def resume_position(episodes: int, order_size: int) -> dict:
    episodes = int(episodes)
    rounds = completed_rounds(episodes, order_size)
    return {"resume_episode": episodes, "rounds_completed": rounds,
            "round_start": rounds * order_size,
            "partial_round": bool(episodes % order_size)}


def should_early_stop(round_idx: int, evaluations: Sequence[dict],
                      patience: int, min_rounds: int) -> bool:
    if patience <= 0 or round_idx < max(int(min_rounds), 0):
        return False
    return no_improvement_streak(evaluations) >= patience


def round_cap_reached(round_idx: int, max_rounds: int) -> bool:
    return max_rounds > 0 and round_idx >= max_rounds


def resume_rule_checks(saved: dict, patience: int, min_rounds: int,
                       max_rounds: int) -> dict:
    return {
        "early_stop_patience": saved.get("early_stop_patience") == patience,
        "min_rounds": saved.get("min_rounds") == min_rounds,
        "max_rounds": saved.get("max_rounds") == max_rounds,
    }


# ----------------------------------------------------------------------
# Days and evaluation
# ----------------------------------------------------------------------
def library_blocks() -> dict:
    """The v3 library's own block lists; test and holiday are never cleared."""
    split = json.loads((LIBRARY / "split.json").read_text(encoding="utf-8"))
    return {block: [entry.split("|")[0] for entry in rows]
            for block, rows in split["pairs"].items()}


def train_days() -> List[dict]:
    pairs, _ = train_pairs(SCREEN)
    return [load_pair(p["load_date"]) for p in pairs]


def block_days(block: str) -> List[dict]:
    return [load_pair(date) for date in library_blocks()[block]]


def eval_day(trainer, env, pair) -> dict:
    """One day cleared with the policy at evaluation temperature.

    A window the market cannot clear ends the day without a value; the day is
    recorded as aborted rather than scored, so a policy that quotes off the
    cleared set shows up in the validation record instead of being averaged in.
    """
    obs = env.reset(pair)
    policy = {nm: 0.0 for nm in trainer.agent_names}
    baseline = {nm: 0.0 for nm in trainer.agent_names}
    soc_min = {nm: float("inf") for nm in trainer.agent_names}
    soc_max = {nm: -float("inf") for nm in trainer.agent_names}
    failure = None
    for _ in range(N_BLOCKS):
        actions, _ = trainer._fleet_action(obs, train=False)
        try:
            obs, _, done, info = env.step(actions)
        except EpisodeClearingError as error:
            failure = str(error)
            break
        for nm in trainer.agent_names:
            policy[nm] += float(info["payoff"][nm])
            baseline[nm] += float(info["baseline_payoff"][nm])
            soc_min[nm] = min(soc_min[nm], float(env.prev_soc[nm]))
            soc_max[nm] = max(soc_max[nm], float(env.prev_soc[nm]))
        if done:
            break
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    if failure is not None:
        return {"pair_id": pair_id, "status": "aborted", "failure": failure,
                "G_policy": None, "G_b1": None, "delta": None,
                "delta_fleet": None, "soc_min": None, "soc_max": None,
                "gap_solved_blocks": env.gap_solved_blocks}
    delta = {nm: policy[nm] - baseline[nm] for nm in trainer.agent_names}
    return {"pair_id": pair_id, "status": "cleared", "failure": "",
            "G_policy": policy, "G_b1": baseline, "delta": delta,
            "delta_fleet": float(sum(delta.values())),
            "soc_min": soc_min, "soc_max": soc_max,
            "gap_solved_blocks": env.gap_solved_blocks}


def evaluate(trainer, env, pairs: Sequence[dict]) -> dict:
    """Mean fleet delta over the days that cleared; aborted days are counted."""
    days = [eval_day(trainer, env, pair) for pair in pairs]
    units = trainer.agent_names
    cleared = [day for day in days if day["status"] == "cleared"]
    fleet = [day["delta_fleet"] for day in cleared]
    return {
        "days": days,
        "aborted_days": len(days) - len(cleared),
        "mean_fleet_delta": float(np.mean(fleet)) if fleet else -math.inf,
        "median_fleet_delta": float(np.median(fleet)) if fleet else -math.inf,
        "mean_delta_by_unit": {nm: (float(np.mean([d["delta"][nm]
                                                   for d in cleared]))
                                    if cleared else None)
                               for nm in units},
        "positive_days": int(sum(1 for value in fleet if value > 0)),
        "soc_min": (min(d["soc_min"][nm] for d in cleared for nm in units)
                    if cleared else None),
        "soc_max": (max(d["soc_max"][nm] for d in cleared for nm in units)
                    if cleared else None),
    }


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=ENV_NAME,
                        help="recorded dataset label; the v3 library is the only source")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exposures", type=int, required=True,
                        help="exposures per training day, from the frozen config")
    parser.add_argument("--min-rounds", type=int, required=True)
    parser.add_argument("--max-rounds", type=int, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-days", type=int, default=0,
                        help="restrict to the first N train days; a smoke run, "
                             "not a formal launch")
    parser.add_argument("--validation-days", type=int, default=0,
                        help="restrict validation to the first N days; a smoke "
                             "run, not a formal launch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seconds", type=float, default=300.0,
                        help="per-window solver limit, from the frozen config")
    args = parser.parse_args(argv)

    run_dir = RUNS / args.run_id
    snap_dir = SNAPSHOTS / args.run_id
    populated = run_dir.exists() and any(run_dir.iterdir())
    if populated and not args.resume:
        print(f"refused: {run_dir} already exists", file=sys.stderr)
        return 3

    saved_manifest = {}
    if populated:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            print(f"refused: {run_dir} has no run_manifest to resume from",
                  file=sys.stderr)
            return 3
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, ok in resume_rule_checks(
            saved_manifest, args.early_stop_patience, args.min_rounds,
            args.max_rounds).items() if not ok]
        if mismatches:
            print("refused: the stop rule differs from the started run: "
                  + ", ".join(mismatches), file=sys.stderr)
            return 3

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = train_days()
    if args.train_days:
        train = train[:args.train_days]
    validation = block_days("validation")
    blocks = library_blocks()
    if args.validation_days:
        validation = validation[:args.validation_days]
    order_size = len(train)
    budget = episode_budget(order_size, args.exposures, args.max_episodes)

    env = LocalDABiddingEnv(seconds=args.seconds)
    trainer = DAGateTrainerV7(env, DAV7ExperimentConfig())
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(
        json.dumps(trainer.experiment.as_dict(), sort_keys=True,
                   default=str).encode("utf-8")).hexdigest()

    evaluations: List[dict] = []
    resume_from = None
    if args.resume and (run_dir / "eval_history.json").exists():
        evaluations = json.loads(
            (run_dir / "eval_history.json").read_text(encoding="utf-8"))
    if args.resume and (run_dir / "checkpoints").is_dir():
        trainer.load(run_dir)
        resume_from = resume_position(trainer.episodes, order_size)
        evaluations = [row for row in evaluations
                       if int(row.get("round", 0))
                       <= resume_from["rounds_completed"]]

    manifest_provenance = {
        "run_id": args.run_id,
        "protocol": "local_da_v7",
        "split": args.split,
        "seed": args.seed,
        "exposures": args.exposures,
        "min_rounds": args.min_rounds,
        "max_rounds": args.max_rounds,
        "max_episodes": args.max_episodes,
        "early_stop_patience": args.early_stop_patience,
        "train_days": order_size,
        "formal_launch": not (args.train_days or args.validation_days),
        "validation_days": len(validation),
        "validation_days_total": len(blocks["validation"]),
        "validation_block": "never in replay",
        "test_days": len(blocks["test"]),
        "holiday_days": len(blocks["holiday"]),
        "test_evaluated": False,
        "holiday_evaluated": False,
        "screen": SCREEN,
        "config_sha256": config_hash,
        "experiment": trainer.experiment.as_dict(),
        "env": {"name": ENV_NAME, "blocks": N_BLOCKS},
        "code_revision": code_revision(),
        "started": saved_manifest.get("started") or now(),
        "stopped": None,
        "best": saved_manifest.get("best") or {"value": -math.inf,
                                               "round": -1,
                                               "checkpoint": ""},
        "resume": {"resumed": bool(resume_from),
                   "from_episode": (resume_from or {}).get("resume_episode")},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest_provenance, indent=2, default=str),
        encoding="utf-8")

    best = dict(manifest_provenance["best"])
    episode = trainer.episodes
    rounds = completed_rounds(episode, order_size)
    stop_reason = None
    aborted_days = 0
    consecutive_clearing_failures = 0
    started = time.monotonic()
    episode_fields = ["episode", "round", "status", "failure", "pair_id",
                      "delta_fleet", "updates", "total_steps",
                      "anchor_weight", "gap_solved_blocks"]
    for nm in trainer.agent_names:
        episode_fields += [f"reward_{nm}", f"soc_end_{nm}"]

    with (run_dir / "train.log").open("a", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            log_file.write(f"{now()} {message}\n")
            log_file.flush()

        def write_episode(row: dict) -> None:
            with (run_dir / "episodes.csv").open(
                    "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=episode_fields)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)

        log(f"start seed={args.seed} episode={episode}/{budget} "
            f"rounds={rounds}/{args.max_rounds}")
        while True:
            order = list(train)
            random.Random(args.seed * 1000 + rounds + 1).shuffle(order)
            # A resumed partial round keeps its own position in the order.
            start = (resume_from["resume_episode"] % order_size
                     if resume_from else 0)
            for pair in order[start:]:
                if episode >= budget:
                    break
                try:
                    out = trainer.train_day(env, pair)
                except EpisodeClearingError as error:
                    # The episode is atomic: a window the market cannot clear
                    # leaves the buffer untouched and the day is recorded, not
                    # retried. Three in a row means the data or the mechanism
                    # is broken, not the day.
                    consecutive_clearing_failures += 1
                    aborted_days += 1
                    episode += 1
                    write_episode({
                        "episode": episode, "round": rounds,
                        "status": "aborted", "failure": str(error),
                        "pair_id": f"{pair['load_date']}|{pair['price_date']}",
                        "delta_fleet": "", "updates": trainer.updates,
                        "total_steps": trainer.total_steps,
                        "anchor_weight": "",
                        "gap_solved_blocks": env.gap_solved_blocks})
                    log(f"ep {episode}/{budget} r{rounds} aborted: {error} "
                        f"({consecutive_clearing_failures} consecutive)")
                    if consecutive_clearing_failures >= 3:
                        raise RuntimeError(
                            "three consecutive clearing failures; training "
                            "stops") from error
                    continue
                consecutive_clearing_failures = 0
                episode += 1
                row = {"episode": episode, "round": rounds, "status": "cleared",
                       "failure": "", "pair_id": out["pair_id"],
                       "delta_fleet": float(sum(
                           float(sum(values)) for values in
                           out["per_block"].values())),
                       "updates": trainer.updates,
                       "total_steps": trainer.total_steps,
                       "anchor_weight": out["anchor_weight"],
                       "gap_solved_blocks": env.gap_solved_blocks}
                for nm, values in out["per_block"].items():
                    row[f"reward_{nm}"] = float(np.sum(values))
                    row[f"soc_end_{nm}"] = out["soc_end"][nm]
                write_episode(row)
                log(f"ep {episode}/{budget} r{rounds} {out['pair_id']}")
            resume_from = None
            round_idx = completed_rounds(episode, order_size)
            if round_idx == rounds:
                # The pass did not finish, so there is no round to measure.
                stop_reason = "budget" if episode >= budget else "partial_round"
                break
            validation_rows = evaluate(trainer, env, validation)
            validation_rows.update({"round": round_idx, "episode": episode,
                                    "timestamp": now()})
            evaluations.append(validation_rows)
            (run_dir / "eval_history.json").write_text(
                json.dumps(evaluations, indent=2, default=str),
                encoding="utf-8")
            checkpoint = trainer.save(
                run_dir, level="resume", dataset=ENV_NAME, seed=args.seed,
                config_hash_value=config_hash, code_revision=code_revision(),
                exposure_log=str(run_dir / "exposure.json"),
                eval_provenance=str(run_dir / "eval_history.json"))
            (snap_dir / f"round_{round_idx:03d}.json").write_text(
                json.dumps({"round": round_idx, "episode": episode,
                            "checkpoint": str(checkpoint),
                            "validation": validation_rows,
                            "timestamp": now()}, indent=2, default=str),
                encoding="utf-8")
            log(f"val r{round_idx}: mean fleet delta="
                f"{validation_rows['mean_fleet_delta']:.1f}")
            if validation_rows["mean_fleet_delta"] > best["value"]:
                best = {"value": validation_rows["mean_fleet_delta"],
                        "round": round_idx,
                        "checkpoint": str(trainer.save(
                            run_dir / "best", level="resume",
                            dataset=ENV_NAME, seed=args.seed,
                            config_hash_value=config_hash,
                            code_revision=code_revision(),
                            exposure_log=str(run_dir / "best/exposure.json"),
                            eval_provenance=str(
                                run_dir / "eval_history.json")))}
                log(f"new best {best['value']:.1f} round={round_idx}")
            rounds = round_idx
            if should_early_stop(round_idx, evaluations,
                                 args.early_stop_patience, args.min_rounds):
                stop_reason = "early_stop"
                break
            if round_cap_reached(round_idx, args.max_rounds):
                stop_reason = "max_rounds"
                break

        final = trainer.save(
            run_dir / "final", level="bundle", dataset=ENV_NAME,
            seed=args.seed, config_hash_value=config_hash,
            data={"library_split": sha(LIBRARY / "split.json")},
            code_revision=code_revision(),
            exposure_log=str(run_dir / "final/exposure.json"),
            eval_provenance=str(run_dir / "eval_history.json"))
        manifest_provenance.update({
            "stopped": now(),
            "stop_reason": stop_reason,
            "episodes": episode,
            "aborted_days": aborted_days,
            "rounds_completed": completed_rounds(episode, order_size),
            "elapsed_seconds": time.monotonic() - started,
            "best": best,
            "final_checkpoint": str(final),
        })
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest_provenance, indent=2, default=str),
            encoding="utf-8")
        log(f"stop {stop_reason} episodes={episode} aborted={aborted_days} "
            f"best={best['value']:.1f}")
    print(json.dumps({"run_dir": str(run_dir), "episodes": episode,
                      "stop_reason": stop_reason, "best": best},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
