"""Smoke three training days on the local day-ahead environment.

This walks real library days through `LocalDABiddingEnv` and measures what one
block costs, so the formal-run decision rests on a number rather than on a
guess. It is not a training run: no policy is built, every quote is the
truthful declaration, and the reward reported is the environment's settled
payoff difference.

`training_contract.py` carries no wall-clock criterion -- its budget is rounds,
exposures, episodes and block steps -- so there is no latency gate to compare
against. The measured seconds are projected onto the frozen block-step bounds
instead, at both the declared 20-period window and the 12-period fallback.

A second, bounded arm walks one day with a shaded quote. A failed probe is
recorded as evidence, never retried or skipped: the environment's own probe
already reports a non-smooth point on 2021-06-04. A block the clearing engine
raises on is recorded with the exception's own text and ends that arm's day;
an aborted day is a valid result, not a reason to retry at a larger budget.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from da_quote_map_v5 import ACTION_BOX_HIGH, TRUTHFUL
from local_da_clearing import ClearingModel, choose_directions
from local_da_days import STORAGE_BUSES, load_pair, train_pairs, window_day
from rl_da_env import (BLOCK_SIZE, EpisodeClearingError, LocalDABiddingEnv,
                       N_BLOCKS, ROLL_HORIZON, SOC0, clear_window)

#: The per-day feasibility screen the v7 day library was validated against.
from local_da_audit import SCREEN  # noqa: E402  one screen definition
#: The storage spec's own band, as `rl_da_env.storage_agents` declares it.
SOC_MIN, SOC_MAX = 0.1, 0.9
#: The brief's fallback window: 12 periods, 8 of them lookahead.
FALLBACK_HORIZON = 12
#: The frozen v6 budget bounds per seed, from `training_contract.py`.
BLOCK_STEPS_PER_SEED = (12_720, 19_080)
#: A near-truthful premium declaration: both prices above zero, full quantity.
SHADED = (10.0, 0.4, 20.0, 0.4)


def declared_point_deviation(quote) -> float:
    """Box-normalized squared distance from truthful, as the trainer anchors it."""
    return float(sum(((float(q) - float(t)) / float(h)) ** 2
                     for q, t, h in zip(quote, TRUTHFUL, ACTION_BOX_HIGH)))


def summarize(record: dict) -> dict:
    """The four acceptance flags plus the declared deviation, nothing else."""
    soc_min, soc_max = record.get("soc_min"), record.get("soc_max")
    soc_in_band = (soc_min is not None and soc_max is not None
                   and SOC_MIN <= soc_min and soc_max <= SOC_MAX)
    return {
        "rewards_finite": bool(record["rewards_finite"]),
        "no_none_clears": record["none_clears"] == 0,
        "soc_in_band": soc_in_band,
        "one_inventory_term": record["inventory_terms"] == 1,
        "declared_point_deviation": record["declared_point_deviation"],
    }


def _timed_block(pair, *, seconds: float) -> dict:
    """Time one exact solve, its priced re-solve and a baseline re-clear."""
    soc = {f"ESS{bus}": SOC0 for bus in STORAGE_BUSES}
    t_end = min(ROLL_HORIZON, 96)
    day = window_day(pair, 0, t_end, soc=soc, quotes={})

    exact_model = ClearingModel(day, exact=True, seconds=seconds)
    try:
        start = time.perf_counter()
        dispatch = exact_model.solve()
        exact = time.perf_counter() - start
    finally:
        exact_model.close()
    directions, _ = choose_directions(dispatch)

    priced_model = ClearingModel(day, directions=directions, exact=False,
                                 seconds=seconds)
    try:
        start = time.perf_counter()
        priced_model.solve(prices=True)
        price = time.perf_counter() - start
    finally:
        priced_model.close()

    start = time.perf_counter()
    clear_window(window_day(pair, 0, t_end, soc=soc, quotes={}),
                 seconds=seconds)
    baseline = time.perf_counter() - start
    return {"exact": exact, "price": price, "baseline": baseline}


def _walk(pair, *, seconds: float, quote, roll_horizon: int = ROLL_HORIZON):
    """Walk one day's blocks at a fixed quote and record what each block did."""
    env = LocalDABiddingEnv(seconds=seconds, roll_horizon=roll_horizon)
    actions = {name: np.array(quote, dtype=float) for name in env.agent_names}
    record = {"date": pair["load_date"], "price_date": pair["price_date"],
              "rewards_finite": True, "clears": 0, "none_clears": 0,
              "soc_min": None, "soc_max": None,
              "declared_point_deviation": declared_point_deviation(quote),
              "inventory_terms": 0, "block_seconds": [], "blocks": [],
              "abort": False}
    start = time.perf_counter()
    env.reset(pair)
    for block in range(N_BLOCKS):
        begun = time.perf_counter()
        try:
            _, rewards, done, info = env.step(actions)
        except (EpisodeClearingError, ValueError) as error:
            record["none_clears"] += 1
            record["abort"] = True
            record["blocks"].append({
                "block": block, "status": "raised",
                "failure": getattr(error, "failure_code", None)
                           or f"{type(error).__name__}: {error}"})
            break
        record["block_seconds"].append(time.perf_counter() - begun)
        record["clears"] += 1
        record["blocks"].append({"block": block, "status": "cleared",
                                 "failure": None})
        record["rewards_finite"] = record["rewards_finite"] and all(
            math.isfinite(float(value)) for value in rewards.values())
        # The raw solve state, not the clamped handoff: a solve outside the
        # band must fail the acceptance flag rather than read the band edge.
        for soc in info["soc_raw"].values():
            record["soc_min"] = (soc if record["soc_min"] is None
                                 else min(record["soc_min"], soc))
            record["soc_max"] = (soc if record["soc_max"] is None
                                 else max(record["soc_max"], soc))
        if (block + 1) * BLOCK_SIZE >= env.T:
            record["inventory_terms"] += 1
        if done:
            break
    record["episode_seconds"] = time.perf_counter() - start

    # The recorded outcome is the walk's own block-11 probe, the same record
    # the environment keeps. The timed call repeats that shape only to price a
    # probe; its result is not the evidence.
    record["probe"] = env.probe_records[0] if env.probe_records else None
    if record["abort"]:
        record["probe_seconds"] = None
    else:
        begun = time.perf_counter()
        env.probe_block()
        record["probe_seconds"] = time.perf_counter() - begun
    return record


def _mean(values):
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def projected_hours(seconds_per_block: float) -> dict:
    """Per-seed hours at the frozen block-step bounds."""
    return {bound: seconds_per_block * steps / 3600.
            for bound, steps in zip(("min", "max"), BLOCK_STEPS_PER_SEED)}


def _report(episodes, fallback, shaded) -> dict:
    """Assemble one report from whichever arms ran."""
    latency = {
        "exact": _mean(record["exact"] for record in episodes),
        "price": _mean(record["price"] for record in episodes),
        "baseline": _mean(record["baseline"] for record in episodes),
        "probe_day": _mean(record["probe_seconds"] for record in episodes),
        "episode": _mean(record["episode_seconds"] for record in episodes),
    }
    block_seconds = _mean(step for record in episodes
                          for step in record["block_seconds"])
    fallback_seconds = fallback["block_seconds"] if fallback else None
    return {
        "days": [record["date"] for record in episodes],
        "episodes": episodes,
        "latency": latency,
        "block_seconds": block_seconds,
        "roll_horizon": ROLL_HORIZON,
        "fallback_horizon": FALLBACK_HORIZON,
        "fallback": fallback,
        "projected_hours": {
            f"horizon_{ROLL_HORIZON}": (projected_hours(block_seconds)
                                        if block_seconds else None),
            f"horizon_{FALLBACK_HORIZON}": (
                projected_hours(fallback_seconds) if fallback_seconds else None),
        },
        "shaded": shaded,
    }


def smoke(days: Sequence[str], *, seconds: float = 120.0,
          side_arms: bool = True) -> dict:
    """Walk each day truthfully, time its solves, and add the two side arms."""
    pairs = [load_pair(day) for day in days]
    episodes = []
    for pair in pairs:
        record = _walk(pair, seconds=seconds, quote=TRUTHFUL)
        # A timing solve that raises costs the latency entry, not the day's
        # record: the walk already happened and is the evidence that matters.
        try:
            record.update(_timed_block(pair, seconds=seconds))
        except Exception as error:
            record.update({"exact": None, "price": None, "baseline": None,
                           "latency_failure": f"{type(error).__name__}: {error}"})
        episodes.append(record)
    fallback = shaded = None
    if side_arms:
        walked = _walk(pairs[0], seconds=seconds, quote=TRUTHFUL,
                       roll_horizon=FALLBACK_HORIZON)
        fallback = {"episode_seconds": walked["episode_seconds"],
                    "block_seconds": _mean(walked["block_seconds"])}
        shaded = _walk(pairs[0], seconds=seconds, quote=SHADED)
    return _report(episodes, fallback, shaded)


def merge_reports(reports) -> dict:
    """Combine per-day reports into the single report the brief names."""
    episodes = [record for report in reports for record in report["episodes"]]
    fallback = next((report["fallback"] for report in reports
                     if report["fallback"]), None)
    shaded = next((report["shaded"] for report in reports
                   if report["shaded"]), None)
    return _report(episodes, fallback, shaded)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default=None,
                        help="comma-separated load dates; default is the first "
                             "three train pairs in manifest order")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--out", default="reports/local_da_v7_2026-11-18")
    parser.add_argument("--no-side-arms", action="store_true",
                        help="skip the fallback and shaded walks")
    parser.add_argument("--merge", nargs="+", default=None,
                        help="per-day smoke.json paths to combine")
    args = parser.parse_args(argv)

    if args.merge:
        report = merge_reports([json.loads(Path(path).read_text("utf-8"))
                                for path in args.merge])
    else:
        if args.days:
            days = [day.strip() for day in args.days.split(",") if day.strip()]
        else:
            pairs, _ = train_pairs(SCREEN)
            days = [pair["load_date"] for pair in pairs[:3]]
        report = smoke(days, seconds=args.seconds,
                       side_arms=not args.no_side_arms)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n",
                                    encoding="utf-8")
    shaded = report["shaded"]
    print(json.dumps({
        "days": report["days"],
        "acceptance": [summarize(record) for record in report["episodes"]],
        "latency": report["latency"],
        "block_seconds": report["block_seconds"],
        "episode_seconds": [record["episode_seconds"]
                            for record in report["episodes"]],
        "probes": [record["probe"] for record in report["episodes"]],
        "fallback": report["fallback"],
        "projected_hours": report["projected_hours"],
        "shaded": None if shaded is None else {
            "date": shaded["date"], "abort": shaded["abort"],
            "clears": shaded["clears"], "none_clears": shaded["none_clears"],
            "episode_seconds": shaded["episode_seconds"],
            "blocks": shaded["blocks"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
