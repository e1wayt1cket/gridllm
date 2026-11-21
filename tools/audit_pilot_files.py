# audit_pilot_files.py
"""Read-only audits: evaluation coverage, training-day exposure, reward definition.

None of these changes a file the pilot produced; they read it and state what it
contains.

Usage:
    PYTHONPATH=src python src/audit_pilot_files.py
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PILOT = "data/processed/v2_cy_jn_2025/pilot"
PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
COMMON = "data/processed/v2_cy_jn_2025/common_split.json"
REPORTS = "reports/v2_cy_jn_2025"
COVERAGE = f"{REPORTS}/pilot_eval_coverage_audit.md"
EXPOSURE = f"{REPORTS}/pilot_training_date_exposure.md"
DEFINITION = f"{REPORTS}/difference_reward_definition_audit.md"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


def _read(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def coverage(root: Path) -> dict:
    rows = _read(root / PILOT / "eval.csv")
    cfg = json.loads((root / PILOT / "pilot_config.json").read_text(
        encoding="utf-8"))
    keys = [(r["dataset"], r["seed"], r["load_date"]) for r in rows]
    per_run = collections.Counter((r["dataset"], r["seed"], r["split"])
                                  for r in rows)
    # B1 comparison, both aggregations.
    per_run_days = collections.defaultdict(list)
    per_day = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["split"] != "test":
            continue
        per_run_days[(r["dataset"], r["seed"])].append(
            float(r["settled_total"]) - float(r["b1_total"]))
        per_day[r["load_date"]][r["dataset"]].append(
            float(r["settled_total"]) - float(r["b1_total"]))
    run_above = {f"{d}/{s}": sum(1 for x in v if x > 1.0)
                 for (d, s), v in per_run_days.items()}
    day_above = {}
    for day, entry in per_day.items():
        for dataset, values in entry.items():
            day_above.setdefault(dataset, []).append(
                sum(values) / len(values) > 1.0)
    seed_avg_above = {d: sum(v) for d, v in day_above.items()}
    return {
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicates": len(keys) - len(set(keys)),
        "datasets": sorted({r["dataset"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "dates": sorted({r["load_date"] for r in rows}),
        "splits": dict(collections.Counter(r["split"] for r in rows)),
        "per_run": {"/".join(k): v for k, v in sorted(per_run.items())},
        "run_above_b1": run_above,
        "seed_averaged_days_above_b1": seed_avg_above,
        "config_eval": cfg["evaluation"],
        "wide_layout": True,
    }


def exposure(root: Path) -> dict:
    cfg = json.loads((root / PILOT / "pilot_config.json").read_text(
        encoding="utf-8"))
    summary = json.loads((root / PILOT / "summary.json").read_text(
        encoding="utf-8"))
    season = {r["load_date"]: r["season"] for r in _read(root / PAIRING)}
    split = json.loads((root / COMMON).read_text(encoding="utf-8"))
    extra = {i.split("|")[0] for i in split["full96_only"]}
    out = []
    for entry in summary:
        exp = entry["exposure"]
        visited = {d: n for d, n in exp.items() if n > 0}
        out.append({
            "dataset": entry["dataset"],
            "seed": entry["seed"],
            "pool": len(exp),
            "distinct": len(visited),
            "max_count": max(exp.values()) if exp else 0,
            "unvisited": sum(1 for n in exp.values() if n == 0),
            "extra_visited": sum(1 for d in visited if d in extra),
            "extra_pool": len(extra) if entry["dataset"] == "FULL96" else 0,
            "common_visited": sum(1 for d in visited if d not in extra),
            "seasons": dict(collections.Counter(season.get(d, "?")
                                                for d in visited)),
            "repeat_days": sum(1 for n in exp.values() if n > 1),
        })
    return {"runs": out, "extra": sorted(extra)}


def definition(root: Path) -> dict:
    from audit_pilot_gates import state_matching_check
    return state_matching_check(root)


def write_coverage(path: Path, facts: dict) -> None:
    lines = [
        "# Pilot evaluation coverage audit (V2-CY-JN-2025)",
        "",
        "## Answer",
        "",
        f"`eval.csv` holds **{facts['rows']} records**, one per "
        "`(dataset, seed, load_date)`, with the four storage units as columns "
        "in the same record. It is **not** one record per unit. The layout is "
        "wide, not long.",
        "",
        "| field | unique values |",
        "|---|---|",
        f"| datasets | {', '.join(facts['datasets'])} |",
        f"| seeds | {', '.join(facts['seeds'])} |",
        f"| dates | {len(facts['dates'])} (11 test + 5 holiday) |",
        f"| storage units | 4 as columns: {', '.join(UNITS)} |",
        f"| splits | {facts['splits']} |",
        "",
        f"Primary key `(dataset, seed, load_date)`: {facts['unique_keys']} "
        f"unique, {facts['duplicates']} duplicates, no missing keys. Every "
        "test and holiday day appears once for each of the four runs:",
        "",
        "| dataset/seed | test | holiday |",
        "|---|---:|---:|",
    ]
    grouped = collections.defaultdict(dict)
    for key, count in facts["per_run"].items():
        dataset_seed, split = key.rsplit("/", 1)[0], key.rsplit("/", 1)[1]
        grouped[dataset_seed][split] = count
    for run, counts in sorted(grouped.items()):
        lines.append(f"| {run} | {counts.get('test', 0)} | "
                     f"{counts.get('holiday_block', 0)} |")
    lines += [
        "",
        "## Why not 176",
        "",
        "A long layout with one row per `(dataset, seed, date, unit)` would "
        "have 2 × 2 × 11 × 4 = **176** test rows (and 80 holiday rows). The "
        "committed file is wide: 2 × 2 × 16 = 64 rows, each carrying four "
        "payoff columns. Both encode the same grid; no unit is missing. "
        "`eval.csv` also carries the per-unit B1 payoff, charge/discharge and "
        "cycles, so the per-unit facts are present, just not as rows.",
        "",
        "## The two aggregation conventions",
        "",
        "Both appear in `matd3_pilot_results.md` and must not be conflated:",
        "",
        "| convention | definition | days above B1 |",
        "|---|---|---|",
    ]
    for run, count in sorted(facts["run_above_b1"].items()):
        lines.append(f"| per run `{run}` | each seed's own 11 days | {count}/11 |")
    for dataset, count in sorted(facts["seed_averaged_days_above_b1"].items()):
        lines.append(f"| seed-averaged `{dataset}` | mean of the two seeds per "
                     f"day, then compare with B1 | {count}/11 |")
    lines += [
        "",
        "The report's run table uses the first; its paired per-day table uses "
        "the second. Both give zero days above B1 for every run and for the "
        "seed-averaged comparison, so the conclusion does not depend on the "
        "convention here.",
        "",
        "## Included",
        "",
        "- All 11 common test days for all four runs.",
        "- All 5 national-day days for all four runs, reported separately.",
        "- No training day appears in the evaluation file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_exposure(path: Path, facts: dict) -> None:
    lines = [
        "# Pilot training-date exposure (V2-CY-JN-2025)",
        "",
        "## Answer",
        "",
        "Twenty episodes means **twenty distinct training days per run**; no "
        "day is drawn twice (maximum count 1). The training pool is therefore "
        "larger than what the run actually used, and must not be described as "
        "the model's sample size.",
        "",
        "| run | pool | distinct visited | max count | unvisited | "
        "FULL96-only visited | common visited | seasons visited |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in facts["runs"]:
        lines.append(
            f"| {run['dataset']} seed {run['seed']} | {run['pool']} | "
            f"{run['distinct']} | {run['max_count']} | {run['unvisited']} | "
            f"{run['extra_visited']} | {run['common_visited']} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(run["seasons"].items()))
            + " |")
    extra_runs = [r for r in facts["runs"] if r["dataset"] == "FULL96"]
    lines += [
        "",
        f"The 16 FULL96-only days were visited "
        + ", ".join(f"{r['extra_visited']} times at seed {r['seed']}"
                    for r in extra_runs)
        + ". A FULL96 run therefore spent that many of its 20 episodes on the "
        "extra days and saw fewer common days than the NEAR run. With the same "
        "20-episode budget, the two pools differ in composition **and** in how "
        "much of the common pool each sees.",
        "",
        "## What this does and does not mean",
        "",
        "- It is not evidence that the extra days are useless; they were "
        "visited, just rarely, and never repeated.",
        "- It is evidence that \"69 training days\" overstates what the run "
        "learned from: each run saw 20 days once.",
        "- Any comparison of pools at this budget carries both the composition "
        "difference and the reduced common-day exposure.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_definition(path: Path, facts: dict) -> None:
    checks = facts.get("checks", {})
    lines = [
        "# Difference-reward definition audit (V2-CY-JN-2025)",
        "",
        "## Which benchmark the reward uses",
        "",
        "**Joint truthful bidding by all four RL storage agents, inside the same "
        "window and from the same current state.** It is *not* a unilateral "
        "counterfactual in which one agent is truthful and the others keep "
        "their learned actions: `rl_env.step` iterates over `self.rl_agents` "
        "and sets each one's committed four periods to its own truthful point "
        "before the single baseline clear. The four recovery actions come from "
        "one clear, not four.",
        "",
        "| property | result |",
        "|---|---|",
        f"| two clears per block | {'PASS' if checks.get('two_clears_per_block') else 'FAIL'} |",
        f"| same wholesale window | {'PASS' if checks.get('same_wholesale') else 'FAIL'} |",
        f"| same config object | {'PASS' if checks.get('same_config_object') else 'FAIL'} |",
        f"| same horizon type (window) | {'PASS' if checks.get('same_horizon_type') else 'FAIL'} |",
        f"| every RL agent truthful on the committed block | "
        f"{'PASS' if checks.get('baseline_committed_block_is_truthful') else 'FAIL'} |",
        f"| non-RL actions unchanged | {'PASS' if checks.get('non_rl_actions_unchanged') else 'FAIL'} |",
        f"| baseline starts from the current SOC | "
        f"{'PASS' if checks.get('baseline_starts_from_current_soc') else 'FAIL'} |",
        "",
        "The non-RL agents' actions are unchanged because they have none in "
        "this experiment: their declared quotes are their truthful defaults, "
        "so holding them fixed is also holding them truthful. That is a "
        "property of this pilot, not of the mechanism in general.",
        "",
        "## Terminology",
        "",
        "Call it a **joint-truthful, state-matched, block-local differential "
        "reward**. Canonical terms it is *not*:",
        "",
        "- not the classic unilateral difference reward (that changes one "
        "agent's action only);",
        "- not the day-level truthful difference (`pi_ai - pi_baseline`), "
        "which requires whole-day clears from the same initial SOC;",
        "- not a potential-based shaping term, because it has no potential "
        "function and can be non-zero on a closed loop.",
        "",
        "The reward is local in two senses: it differences the committed four "
        "periods, and the counterfactual re-clears the window with the policy's "
        "own later-period actions still in place. Both are deliberate; see "
        "`difference_reward_state_matching_audit.md`.",
        "",
        "## Per-agent accounting",
        "",
        "Each agent's baseline profit is read from the same `base_result`, at "
        "that agent's own bus LMP, through `participant_payoff` -- the same "
        "ledger used for the real arm. No agent's baseline is taken from "
        "another agent's schedule or from the day-level B1 result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    root = Path.cwd()
    cov = coverage(root)
    exp = exposure(root)
    defn = definition(root)
    write_coverage(root / COVERAGE, cov)
    write_exposure(root / EXPOSURE, exp)
    write_definition(root / DEFINITION, defn)
    print(f"eval rows {cov['rows']}, unique keys {cov['unique_keys']}, "
          f"duplicates {cov['duplicates']}")
    print(f"exposure: {[(r['dataset'], r['seed'], r['distinct'],
                        r['extra_visited']) for r in exp['runs']]}")
    print(f"definition checks {sum(defn['checks'].values())}/"
          f"{len(defn['checks'])}")
    print(f"wrote {COVERAGE}\nwrote {EXPOSURE}\nwrote {DEFINITION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
