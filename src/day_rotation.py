# day_rotation.py
"""Which day each training episode is priced on, and in what order.

A training episode is one whole day. The days come from the frozen 40-day
wholesale library, split once into train 20 / validation 10 / test 10, and a
cycle is one complete traversal of the 20 training days.

Three separations make the protocol falsifiable, and each is a line rather than
an intention:

**The split is fixed and declared.** It comes from `market_opportunity`'s
regime labelling, which partitions on the days' *physics* before any economic
outcome exists. It is written out here so a run does not depend on re-deriving
it, and a test re-derives it and fails if the two disagree.

**The order is a function of the run seed; the data is not.** `run_seed` decides
the permutation within each cycle, and nothing else about which days exist or
what they contain. Two runs with different seeds see the same twenty days in
different orders -- so a difference between them is a difference in the
learning, not in the data. Drawing the order from a stream of its own, tagged
per `(run_seed, cycle)`, keeps it independent of how many random draws the
training loop happened to make first.

**A cycle is a permutation, never a sample.** Every training day appears exactly
once per cycle. Sampling with replacement would let a cycle over-represent some
days, which makes the per-cycle validation comparison a comparison of different
day mixes rather than of the policy.

`day_id` is a library index and is not derived from the run seed. The same
`day_id` must price the same load, PV, wind and wholesale for every run on
earth; the library's own manifest is what guarantees that, not this module.

Usage:
    PYTHONPATH=src python src/day_rotation.py --plan --seed 1
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

import numpy as np

#: Days in the frozen library, and the sizes the split must have.
LIBRARY_DAYS = 40
SPLIT_SIZES = {"train": 20, "validation": 10, "test": 10}

#: The split, as library day ids. Derived from `market_opportunity`'s regime
#: labelling over `ensemble_specs(40, seed=MASTER_SEED)`, where the ensemble
#: index and the library day id are the same number. Repeated here rather than
#: recomputed so a training run does not depend on that derivation succeeding,
#: and pinned by a test that regenerates it.
SPLIT: Dict[str, List[int]] = {
    "train": [0, 1, 2, 4, 6, 8, 11, 15, 16, 19, 20, 22, 28, 29, 30, 33, 35,
              36, 37, 38],
    "validation": [3, 5, 9, 10, 13, 14, 24, 26, 27, 39],
    "test": [7, 12, 17, 18, 21, 23, 25, 31, 32, 34],
}

#: Tags the rotation's random stream. Changing it changes every permutation,
#: which is a protocol change and not a seed change.
STREAM_TAG = 0x5EED


def split_of(partition: str) -> List[int]:
    return list(SPLIT[partition])


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


def cycle_days(run_seed: int, cycle: int) -> List[int]:
    """The 20 training days of one cycle, in this run's order for that cycle.

    A permutation, never a sample: the returned list is always exactly the
    training set, so every cycle covers the same days and a per-cycle
    comparison compares policies rather than day mixes.
    """
    days = split_of("train")
    if cycle < 0:
        raise ValueError(f"cycle must be non-negative; got {cycle}")
    order = np.random.RandomState(cycle_seed(run_seed, cycle)).permutation(
        len(days))
    return [days[i] for i in order]


def validation_days() -> List[int]:
    """The days a cycle is evaluated on. Never used for a gradient step."""
    return split_of("validation")


def test_days() -> List[int]:
    """The days reserved for after the model is frozen and no choice remains."""
    return split_of("test")


def episode_plan(run_seed: int, max_cycles: int = 40) -> List[dict]:
    """The whole schedule: one entry per episode, in the order they run."""
    plan = []
    for cycle in range(int(max_cycles)):
        for position, day in enumerate(cycle_days(run_seed, cycle)):
            plan.append({
                "run_seed": int(run_seed),
                "cycle": cycle,
                "position": position,
                "episode": cycle * len(SPLIT["train"]) + position,
                "day_id": day,
                "split": "train",
            })
    return plan


def episode_day(run_seed: int, episode: int,
                days_per_cycle: Optional[int] = None) -> dict:
    """Which day one training episode runs on, and where it sits in the run.

    The trainer asks the schedule rather than dividing by twenty itself, so the
    cycle length lives in one place and a change to it cannot leave the trainer
    and the plan disagreeing about what cycle an episode is in.
    """
    per_cycle = int(days_per_cycle or len(SPLIT["train"]))
    if episode < 0:
        raise ValueError(f"episode must be non-negative; got {episode}")
    cycle, position = divmod(int(episode), per_cycle)
    days = cycle_days(run_seed, cycle)
    return {
        "run_seed": int(run_seed),
        "cycle": cycle,
        "position": position,
        "episode": int(episode),
        "day_id": days[position],
        "split": "train",
        "days_per_cycle": per_cycle,
    }


def is_cycle_end(episode: int, days_per_cycle: Optional[int] = None) -> bool:
    """Whether this episode completes a cycle, and so triggers validation."""
    per_cycle = int(days_per_cycle or len(SPLIT["train"]))
    return (int(episode) + 1) % per_cycle == 0


def assert_protocol() -> dict:
    """Check the split's shape and disjointness, and report what it checked.

    Cheap enough to run at the top of a training run: a split that overlaps, or
    that is not the declared size, would silently train on validation days and
    the validation curve would stop meaning anything.
    """
    problems = []
    for name, want in SPLIT_SIZES.items():
        got = len(SPLIT[name])
        if got != want:
            problems.append(f"{name} has {got} days, expected {want}")
    seen: Dict[int, str] = {}
    for name in SPLIT:
        for day in SPLIT[name]:
            if day in seen:
                problems.append(f"day {day} is in both {seen[day]} and {name}")
            seen[day] = name
            if not 0 <= day < LIBRARY_DAYS:
                problems.append(f"day {day} is outside the library")
    missing = sorted(set(range(LIBRARY_DAYS)) - set(seen))
    if missing:
        problems.append(f"days missing from the split: {missing}")
    if problems:
        raise ValueError("the split is not usable: " + "; ".join(problems))
    return {
        "sizes": {k: len(v) for k, v in SPLIT.items()},
        "disjoint": True,
        "covers_library": True,
        "library_days": LIBRARY_DAYS,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=40)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    protocol = assert_protocol()
    if args.json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"protocol": protocol, "split": SPLIT,
                       "plan": episode_plan(args.seed, args.max_cycles)},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")

    print(f"split {protocol['sizes']}  disjoint {protocol['disjoint']}  "
          f"covers library {protocol['covers_library']}")
    print(f"train      {SPLIT['train']}")
    print(f"validation {SPLIT['validation']}")
    print(f"test       {SPLIT['test']}")
    if args.plan:
        plan = episode_plan(args.seed, args.max_cycles)
        print()
        print(f"run seed {args.seed}: {len(plan)} episodes over "
              f"{args.max_cycles} cycles")
        for cycle in range(min(3, args.max_cycles)):
            print(f"  cycle {cycle:>2}: {cycle_days(args.seed, cycle)}")
        print("  ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
