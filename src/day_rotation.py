# day_rotation.py
"""Which day each training episode runs on, given the days that exist.

The module is a scheduler and nothing else. It does not know how many days there
are, does not hold a split, and does not name a dataset: the **caller** supplies
the days that actually exist in the real curves, and the scheduler decides the
order they are visited in. A dataset length baked in here would be a length the
real data has to be made to fit.

Three properties make the schedule falsifiable, and each is a line rather than
an intention:

**A cycle is a permutation, never a sample.** Every day the caller supplies
appears exactly once per cycle. Sampling with replacement would let a cycle
over-represent some days, which makes a per-cycle comparison a comparison of
different day mixes rather than of the policy.

**The order is a function of the run seed; the data is not.** `run_seed` decides
the permutation within each cycle and nothing else. Two runs with different
seeds see the same days in a different order, so a difference between them is a
difference in the learning rather than in the data.

**The order is derived from its own stream.** `cycle_seed(run_seed, cycle)` is
stateless, so cycle *k* does not depend on how many cycles ran before it and
drawing one more exploration sample cannot move a single day.

Splitting the days into train / validation / test is **not** done here. It is a
protocol decision that belongs with the real date set and its coverage, not a
constant in a scheduler.

Usage:
    PYTHONPATH=src python src/day_rotation.py --days 2021-01-01,2021-01-02 --seed 1 --plan
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

import numpy as np

#: Tags the rotation's random stream. Changing it changes every permutation,
#: which is a protocol change and not a seed change.
STREAM_TAG = 0x5EED


def assert_days(days: Sequence) -> List:
    """The days a caller supplied, checked, or a refusal explaining why not.

    A scheduler that accepts an empty list runs zero episodes and looks like a
    run that finished; one that accepts duplicates visits a day twice and calls
    it a cycle. Both are cheap to catch and neither is visible later.
    """
    out = list(days)
    if not out:
        raise ValueError(
            "no days were supplied. The schedule is built from the days that "
            "exist in the real curves; it does not come with a default set.")
    if any(d is None for d in out):
        raise ValueError(f"the day list contains None: {out[:10]}")
    if len(set(out)) != len(out):
        seen, dupes = set(), []
        for day in out:
            if day in seen:
                dupes.append(day)
            seen.add(day)
        raise ValueError(
            f"the day list repeats {sorted(set(dupes))[:10]}; a cycle would "
            "visit those days twice, so it would not be a permutation")
    return out


def cycle_seed(run_seed: int, cycle: int) -> int:
    """The seed for one cycle's permutation, derived from the run seed.

    Stateless on purpose: the order of cycle *k* does not depend on how many
    cycles ran before it, so a run resumed or inspected out of order gives the
    same schedule. The tag keeps this stream from colliding with the training
    seed, so drawing one more exploration sample cannot move a single day.

    Mixed rather than multiplied: a plain `run_seed * A + cycle * B` overflows
    numpy's 32-bit seeding range for a large run seed, and taking it modulo a
    power of two leaves neighbouring seeds producing near-identical streams.
    The finalizer below avalanches, so seed 1 and seed 2 give unrelated orders.
    """
    h = ((int(run_seed) & 0xFFFFFFFF) * 0x9E3779B1
         + (int(cycle) & 0xFFFFFFFF) * 0x85EBCA77
         + STREAM_TAG) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def cycle_days(days: Sequence, run_seed: int, cycle: int) -> List:
    """One cycle's days, in this run's order for that cycle.

    A permutation of exactly what was supplied, never a sample: every day
    appears once, so two cycles cover the same days and a per-cycle comparison
    compares policies rather than day mixes.
    """
    ordered = assert_days(days)
    if cycle < 0:
        raise ValueError(f"cycle must be non-negative; got {cycle}")
    order = np.random.RandomState(cycle_seed(run_seed, cycle)).permutation(
        len(ordered))
    return [ordered[i] for i in order]


def episode_day(days: Sequence, run_seed: int, episode: int) -> dict:
    """Which day one training episode runs on, and where it sits in the run.

    The trainer asks the schedule rather than dividing by a cycle length it
    keeps itself, so the length lives in one place and a change to it cannot
    leave the trainer and the plan disagreeing about what cycle an episode is
    in.
    """
    ordered = assert_days(days)
    if episode < 0:
        raise ValueError(f"episode must be non-negative; got {episode}")
    per_cycle = len(ordered)
    cycle, position = divmod(int(episode), per_cycle)
    return {
        "run_seed": int(run_seed),
        "cycle": cycle,
        "position": position,
        "episode": int(episode),
        "day": cycle_days(ordered, run_seed, cycle)[position],
        "days_per_cycle": per_cycle,
    }


def is_cycle_end(days: Sequence, episode: int) -> bool:
    """Whether this episode completes a cycle, and so triggers validation."""
    return (int(episode) + 1) % len(assert_days(days)) == 0


def episode_plan(days: Sequence, run_seed: int, max_cycles: int = 1) -> List[dict]:
    """The whole schedule: one entry per episode, in the order they run."""
    ordered = assert_days(days)
    plan = []
    for cycle in range(int(max_cycles)):
        for position, day in enumerate(cycle_days(ordered, run_seed, cycle)):
            plan.append({
                "run_seed": int(run_seed),
                "cycle": cycle,
                "position": position,
                "episode": cycle * len(ordered) + position,
                "day": day,
            })
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Order the days a training run visits.")
    parser.add_argument("--days", required=True,
                        help="comma-separated days that actually exist in the "
                             "real curves; there is no default set")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    days = [d.strip() for d in args.days.split(",") if d.strip()]
    ordered = assert_days(days)
    print(f"{len(ordered)} days; a cycle is a permutation of them")
    for cycle in range(min(3, args.max_cycles)):
        print(f"  cycle {cycle:>2}: {cycle_days(ordered, args.seed, cycle)}")
    if args.plan:
        plan = episode_plan(ordered, args.seed, args.max_cycles)
        print(f"run seed {args.seed}: {len(plan)} episodes over "
              f"{args.max_cycles} cycles")
    if args.json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"days": ordered,
                       "plan": episode_plan(ordered, args.seed, args.max_cycles)},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
