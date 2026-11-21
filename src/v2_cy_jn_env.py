# v2_cy_jn_env.py
"""Bridge a V2-CY-JN-2025 pair into the MATD3 environment and smoke it.

The environment is unchanged. It prices each rolling window through
`grid.day_ahead_price_china`, which reads `wholesale_market.source` and
`day_id`; setting those to `jn2025` and the pair's price date is the whole
integration. No observation feature reads the price curve -- the price reaches
the policy only through LMPs it has already observed -- so OBS_V6 stays as it
is and no checkpoint changes meaning.

`mv_value` is not available: neither its unit nor the time it becomes
available can be confirmed from the source. Adding it would therefore mean a
new observation version, not a reinterpretation of V6.

This runs a deterministic truthful policy for three pairs. It is an interface
and environment smoke test, not a performance result, and no gradient step is
taken.

Usage:
    PYTHONPATH=src python src/v2_cy_jn_env.py --smoke 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

import baselines
import config_loader
import wholesale_price
from rl_env import BiddingEnv, N_BLOCKS
from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V6
from screen_v2_cy_jn_2025 import (PAIRING, PRICE_CURVES, REPORT_DIR,
                                  build_agents, build_config)

RL_NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def configure_price(price_date: str) -> None:
    """Point the global price source at the pair's recorded day.

    `day_ahead_price_china` resolves the source from the defaults cache, not
    from the config it is handed, so the source and day id are written there.
    A stale provider would keep serving whatever file it opened first.
    """
    config_loader.set_default("wholesale_market.source", "jn2025")
    config_loader.set_default("wholesale_market.day_id", price_date)
    wholesale_price.reset_cache()


def build_env_for_pair(pair: dict, cycle_cost: float = 100.0,
                       differential_reward: bool = True,
                       terminal_soc_target: float | None = None) -> BiddingEnv:
    cycles = np.load(PRICE_CURVES)
    loads = np.load("data/processed/v2_cy_jn_2025/bus_loads_96x33.npz")
    price = np.asarray(cycles[f"south/{pair['price_date']}"], dtype=float)
    load = np.asarray(loads[f"load/{pair['load_date']}"], dtype=float)
    config = build_config(cycle_cost)
    _net, agents = build_agents(config, load, price, [9, 14, 17, 27])
    configure_price(pair["price_date"])
    env = BiddingEnv(agents, config, rl_agent_names=list(RL_NAMES),
                     use_differential_reward=differential_reward,
                     obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2,
                     terminal_soc_target=terminal_soc_target)
    env.set_agents(agents, day_id=pair["price_date"])
    return env


def run_episode(env: BiddingEnv, price: np.ndarray) -> dict:
    """One deterministic truthful episode, recording what the smoke checks."""
    policy = baselines.truthful_policy(ACTION_STORAGE_QUOTE_V2)
    obs = env.reset()
    rec = {"obs_nan": False, "clears": 0, "fallbacks": 0,
           "lmp_fallbacks": 0, "soc_out_of_range": 0, "out_of_box": 0,
           "reward_nan": False, "price_alignment_max_dev": 0.0,
           "rewards": []}
    for block in range(N_BLOCKS):
        acts = {name: policy(obs[name]) for name in RL_NAMES}
        next_obs, rewards, done, info = env.step(acts)
        rec["clears"] += 1 if env._last_result is not None else 0
        rec["fallbacks"] += 1 if env._last_result is None else 0
        if env._last_result is None:
            break
        result = env._last_result
        rec["lmp_fallbacks"] += int(result.get("lmp_fallbacks", 0))
        for name in RL_NAMES:
            if not np.all(np.isfinite(obs[name])):
                rec["obs_nan"] = True
        for value in rewards.values():
            rec["rewards"].append(float(value))
            if not np.isfinite(value):
                rec["reward_nan"] = True
        for name in RL_NAMES:
            action = np.asarray(acts[name], dtype=float)
            spec = ACTION_STORAGE_QUOTE_V2
            for value, key in zip(action, spec.action_names):
                low, high = spec.bounds[key]
                if value < low - 1e-6 or value > high + 1e-6:
                    rec["out_of_box"] += 1
            sched = (result.get("schedules") or {}).get(name)
            if sched is not None:
                soc = np.asarray(sched["soc_boundary"], dtype=float)
                agent = next(a for a in env.all_agents if a.name == name)
                if (np.nanmin(soc) < agent.storage.soc_min - 1e-6
                        or np.nanmax(soc) > agent.storage.soc_max + 1e-6):
                    rec["soc_out_of_range"] += 1
        t_start = block * 4
        window = price[t_start:t_start + len(env._last_wholesale)]
        rec["price_alignment_max_dev"] = max(
            rec["price_alignment_max_dev"],
            float(np.max(np.abs(np.asarray(env._last_wholesale) - window))))
        if done:
            break
        obs = next_obs
    return rec


def _write_audit(path: Path, rows: Sequence[dict], repeats_identical: bool) -> Path:
    lines = [
        "# MATD3 interface audit (V2-CY-JN-2025)",
        "",
        "No training was started. This records what the environment actually "
        "consumes, what the price injection touches, and the result of a "
        "deterministic smoke run. It is a cross-year controlled simulation, "
        "not a same-period market result.",
        "",
        "## Observation: OBS_V6, unchanged",
        "",
        "| # | feature | computed from | periods read at decision block b |",
        "|---:|---|---|---|",
        "| 1 | `load_feat` | block mean of the agent's load forecast / its "
        "own forecast peak | the current block, `4b..4b+3` |",
        "| 2 | `re_feat` | block mean of PV+wind forecast / forecast peak | "
        "the current block |",
        "| 3 | `soc` | state at the start of the block | none (a state) |",
        "| 4 | `last_lmp_norm` | most recent committed cross-node mean LMP / "
        "`PRICE_REF_CNY_MWH` | only periods already cleared |",
        "| 5 | `block_pos` | `b / 23` | none |",
        "| 6 | `avg_lmp_hist_norm` | mean of committed LMPs / reference | "
        "only periods already cleared |",
        "| 7 | `slr_norm` | system load/RE ratio of the current block's "
        "forecast | the current block |",
        "| 8 | `avg_other_dev` | signed deviation of the other agents' "
        "committed quotes | the previous block only |",
        "| 9 | `all_dev_std` | spread of the same signed deviations | the "
        "previous block only |",
        "| 10 | `ema_dev` | observed LMP vs its EMA | only periods already "
        "cleared |",
        "| 11 | `price_trend` | slope of the last up-to-4 observed LMPs | "
        "only periods already cleared |",
        "| 12 | `pred_lmp_norm` | EMA forecast of the next LMP / reference | "
        "a forecast of the future, not a realized value |",
        "",
        "No feature reads the wholesale curve, this episode's or any other. "
        "The Jiangnan price reaches the policy only through LMPs it has "
        "already observed, so the causality the V4 replacement established "
        "still holds.",
        "",
        "## Decision blocks",
        "",
        "24 blocks of 4 quarter-hour periods (1 h). Each block clears a "
        "rolling window (`roll_horizon = 16` periods) and commits its first 4. "
        "The wholesale curve is sliced per window with `t_start`, and the "
        "smoke run checks every block's slice equals the pair's own curve.",
        "",
        "## Action space",
        "",
        "| participant | action | box | truthful |",
        "|---|---|---|---|",
        "| ESS9/14/17/27 | `charge_price`, `discharge_price` (CNY/MWh) | "
        "0–150, 0–300 | (0, 0) |",
        "| loads and prosumers | `bid_mult`, `offer_adder` | 0.3–1.8, 0–50 | "
        "(1, 0) |",
        "",
        "The smoke run controls the four storage units only; every other "
        "participant declares its truthful quote.",
        "",
        "## Reward and settlement",
        "",
        "Per block, reward is the storage unit's own `participant_payoff` for "
        "the committed periods minus the same window re-cleared under truthful "
        "bidding. Cycle cost is the same `config.storage.cycle_cost` the "
        "clearing applies, so C80/C90/C100 move reward and clearing together. "
        "The environment's rolling windows do not pin terminal SOC; the "
        "whole-day settlement does, and `results.csv` shows "
        "`terminal_soc_max_err = 0` for every paired day.",
        "",
        "## Smoke result",
        "",
        "| pair | blocks cleared | fallbacks | LMP fallback cells | "
        "obs NaN | reward NaN | out-of-box actions | SOC out of range | "
        "max price-slice deviation (CNY/MWh) |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['load_date']} → {row['price_date']} | {row['clears']} | "
            f"{row['fallbacks']} | {row['lmp_fallbacks']} | "
            f"{row['obs_nan']} | {row['reward_nan']} | {row['out_of_box']} | "
            f"{row['soc_out_of_range']} | "
            f"{row['price_alignment_max_dev']:.2e} |")
    lines.append("")
    lines.append(f"Repeating the same three pairs gave identical rewards: "
                 f"**{repeats_identical}**.")
    lines.append("")
    lines.append("## `mv_value` and JN-RE")
    lines.append("")
    lines.append("`mv_value` is not in any observation. Its unit, its 15-minute "
                 "status and the time it becomes available cannot be confirmed "
                 "from the source, and a provincial new-energy total is not a "
                 "bus-level injection. Enabling JN-RE would require a new "
                 "observation version (`OBS_V7`), a fresh reference-price and "
                 "normalisation decision, and would invalidate every existing "
                 "checkpoint. `load_policy` enforces the name, so a mismatched "
                 "actor is refused rather than silently reused.")
    lines.append("")
    lines.append("## What is not established")
    lines.append("")
    lines.append("- No policy was trained. The smoke policy is truthful and "
                 "fixed.")
    lines.append("- The reference price `PRICE_REF_CNY_MWH` is the frozen "
                 "synthetic-family constant. The Jiangnan daily mean is close "
                 "to it, and replacing it would be a training-scale decision "
                 "this version does not make.")
    lines.append("- Nothing here is a performance claim about MATD3.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _representatives(pairs: Sequence[dict], limit: int) -> List[dict]:
    """One pair per non-holiday season, plus the holiday block if asked."""
    chosen: List[dict] = []
    seen = set()
    for pair in pairs:
        group = pair["group"]
        if group in seen or group == "national_day_2022":
            continue
        seen.add(group)
        chosen.append(pair)
        if len(chosen) >= limit:
            break
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=3,
                        help="number of pairs to run (one per season)")
    parser.add_argument("--pairing", default=PAIRING)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    pairs = _read_csv(root / args.pairing)
    chosen = _representatives(pairs, args.smoke)
    cycles = np.load(root / PRICE_CURVES)

    rows = []
    for pair in chosen:
        env = build_env_for_pair(pair)
        price = np.asarray(cycles[f"south/{pair['price_date']}"], dtype=float)
        rec = run_episode(env, price)
        rec["load_date"] = pair["load_date"]
        rec["price_date"] = pair["price_date"]
        rows.append(rec)
        print(f"{pair['load_date']} -> {pair['price_date']}: "
              f"{rec['clears']}/{N_BLOCKS} blocks, "
              f"fallbacks={rec['fallbacks']} lmp_fb={rec['lmp_fallbacks']} "
              f"price_dev={rec['price_alignment_max_dev']:.2e}")

    repeats = []
    for pair in chosen:
        env = build_env_for_pair(pair)
        price = np.asarray(cycles[f"south/{pair['price_date']}"], dtype=float)
        repeats.append(run_episode(env, price)["rewards"])
    identical = all(
        np.allclose(first, second)
        for first, second in zip([r["rewards"] for r in rows], repeats))

    path = _write_audit(root / args.report_dir / "matd3_interface_audit.md",
                        rows, identical)
    config_loader.reload_defaults()
    wholesale_price.reset_cache()
    print(f"repeat identical: {identical}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
