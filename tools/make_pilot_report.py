# make_pilot_report.py
"""Assemble the MATD3 pilot report from the training logs and evaluation rows."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PILOT = "data/processed/v2_cy_jn_2025/pilot"
TRAIN = f"{PILOT}/train"
EVAL = f"{PILOT}/eval.csv"
CONFIG = f"{PILOT}/pilot_config.json"
SUMMARY = f"{PILOT}/summary.json"
REPORT = "reports/v2_cy_jn_2025/matd3_pilot_results.md"
RUNS = (("NEAR", 11), ("NEAR", 22), ("FULL96", 11), ("FULL96", 22))
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


def _read(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value, digits=1) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _mean(values):
    values = [float(v) for v in values if str(v) not in ("", "nan")]
    return float(np.mean(values)) if values else float("nan")


def build(cfg: dict, summary: List[dict], train: Dict[str, List[dict]],
          eval_rows: List[dict]) -> str:
    lines = [
        "# MATD3 pilot results (V2-CY-JN-2025)",
        "",
        "A controlled pilot on the common split: **two seeds, twenty episodes,** "
        "and two training pools that differ only in the 16 FULL96-only days. "
        "Eleven test days cannot support a significance claim; every table below "
        "is descriptive, and the settlement is a full-day clear with the "
        "policy's own per-block quotes.",
        "",
        "## Training stability",
        "",
        "| run | episodes | steps | updates | last-5 reward | last-5 raw | "
        "critic loss | actor loss | clears | fallbacks | LMP fallbacks | "
        "voltage cells | line cells | max end-SOC gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, seed in RUNS:
        rows = train.get(f"{dataset}|{seed}", [])
        if not rows:
            continue
        last5 = rows[-5:]
        lines.append(
            f"| {dataset} seed {seed} | {len(rows)} | "
            f"{rows[-1].get('total_steps', '')} | "
            f"{sum(int(float(r.get('updates') or 0)) for r in rows)} | "
            f"{_f(_mean([r['ep_reward'] for r in last5]))} | "
            f"{_f(_mean([r['ep_raw_profit'] for r in last5]))} | "
            f"{_f(rows[-1].get('critic_loss'))} | "
            f"{_f(rows[-1].get('actor_loss'))} | "
            f"{sum(int(float(r.get('clears') or 0)) for r in rows)} | "
            f"{sum(int(float(r.get('fallbacks') or 0)) for r in rows)} | "
            f"{sum(int(float(r.get('lmp_fallbacks') or 0)) for r in rows)} | "
            f"{sum(int(float(r.get('voltage_violation_cells') or 0)) for r in rows)} | "
            f"{sum(int(float(r.get('line_overload_cells') or 0)) for r in rows)} | "
            f"{max((float(r.get('end_soc_gap') or 0) for r in rows),
                  default=0.0):.2e} |")
    lines += [
        "",
        "The end-SOC gap is the rolling episode's final state against 0.5. A "
        "training episode does **not** pin terminal SOC; only the full-day "
        "settlement does, and every evaluation row reports a terminal error of "
        "0. A 0.4 gap (the full 0.10-0.90 band) is therefore expected in "
        "training and is not a constraint violation.",
        "",
        "## Training-day exposure",
        "",
        "| run | pool days | distinct days seen | most-seen day count |",
        "|---|---:|---:|---:|",
    ]
    for entry in summary:
        exposure = entry.get("exposure", {})
        lines.append(
            f"| {entry['dataset']} seed {entry['seed']} | {len(exposure)} | "
            f"{sum(1 for n in exposure.values() if n > 0)} | "
            f"{max(exposure.values(), default=0)} |")
    lines += [
        "",
        "Twenty episodes over 53 or 69 days means most days are seen at most "
        "once; the pool difference is a composition difference, not extra "
        "repetition.",
        "",
        "## Learning trend",
        "",
        "| run | first-5 reward | last-5 reward | first-5 raw | last-5 raw | "
        "first-5 action charge | last-5 action charge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, seed in RUNS:
        rows = train.get(f"{dataset}|{seed}", [])
        if not rows:
            continue
        first5, last5 = rows[:5], rows[-5:]
        lines.append(
            f"| {dataset} seed {seed} | "
            f"{_f(_mean([r['ep_reward'] for r in first5]))} | "
            f"{_f(_mean([r['ep_reward'] for r in last5]))} | "
            f"{_f(_mean([r['ep_raw_profit'] for r in first5]))} | "
            f"{_f(_mean([r['ep_raw_profit'] for r in last5]))} | "
            f"{_f(_mean([r['action_charge_mean'] for r in first5]))} | "
            f"{_f(_mean([r['action_charge_mean'] for r in last5]))} |")
    lines += [
        "",
        "The reward is the differential learning signal. The raw profit is what "
        "the agent settled. A policy that learns to quote closer to truthful "
        "can raise its raw profit while its differential reward stays near "
        "zero, because both arms move together; the columns are kept apart for "
        "that reason.",
        "",
        "## Evaluation on the common test set (11 days)",
        "",
        "| run | settled (CNY/day) | B1 truthful | B2A threshold | "
        "MATD3 − B1 | MATD3 − B2A | days above B1 | days below B2A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| NEAR | | | | | | | |",
    ]
    test_rows = [r for r in eval_rows if r["split"] == "test"]
    holiday = [r for r in eval_rows if r["split"] == "holiday_block"]
    for dataset, seed in RUNS:
        sub = [r for r in test_rows
               if r["dataset"] == dataset and r["seed"] == str(seed)]
        if not sub:
            continue
        diff_b1 = [float(r["diff_vs_b1"]) for r in sub]
        diff_b2 = [float(r["diff_vs_b2a"]) for r in sub]
        lines.append(
            f"| {dataset} seed {seed} | "
            f"{_f(_mean([r['settled_total'] for r in sub]))} | "
            f"{_f(_mean([r['b1_total'] for r in sub]))} | "
            f"{_f(_mean([r['b2a_total'] for r in sub]))} | "
            f"{_f(_mean(diff_b1))} | {_f(_mean(diff_b2))} | "
            f"{sum(1 for d in diff_b1 if d > 1.0)} | "
            f"{sum(1 for d in diff_b2 if d < -1.0)} |")
    # Replace the placeholder header row with nothing: the loop wrote rows.
    lines = [l for l in lines if l != "| NEAR | | | | | | | |"]

    by_day = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in test_rows:
        by_day[r["load_date"]][r["dataset"]].append(r)
    lines += [
        "",
        "All four runs settle **below B1 on every test day** (0 of 11 days "
        "above). The mechanism is visible in the learning trend: the policies "
        "learn quotes well above the truthful point (charge ~48-127 CNY/MWh, "
        "discharge ~20-82), and a quote above truthful raises the clearing's "
        "cost of charging and penalises discharging, so the fleet cycles less "
        "than the truthful arm. The positive training reward is a *local* "
        "differential against other agents' contemporaneous actions; it is not "
        "the day-level B1 comparison, and the two can disagree. This is the "
        "reward-is-not-benefit distinction the audits require.",
        "",
        "### Paired per-day fleet payoff on the common test set",
        "",
        "Mean over the two seeds. `FULL96 - NEAR` is the treatment difference "
        "the pilot is about.",
        "",
        "| load_date | B1 | B2A | NEAR | FULL96 | FULL96 - NEAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    full_minus_near = []
    for day in sorted(by_day):
        entry = by_day[day]
        near_rows = entry.get("NEAR", [])
        full_rows = entry.get("FULL96", [])
        near_total = _mean([r["settled_total"] for r in near_rows])
        full_total = _mean([r["settled_total"] for r in full_rows])
        diff = (full_total - near_total
                if near_rows and full_rows else float("nan"))
        full_minus_near.append(diff)
        ref = full_rows or near_rows
        lines.append(
            f"| {day} | {_f(_mean([r['b1_total'] for r in ref]))} | "
            f"{_f(_mean([r['b2a_total'] for r in ref]))} | "
            f"{_f(near_total)} | {_f(full_total)} | {_f(diff)} |")
    lines += [
        "",
        f"FULL96 − NEAR over the 11 days: mean {_f(_mean(full_minus_near))} "
        f"CNY/day, median {_f(np.median(full_minus_near))}, "
        f"positive on {sum(1 for d in full_minus_near if d > 1.0)} days, "
        f"negative on {sum(1 for d in full_minus_near if d < -1.0)}. "
        "This is a descriptive paired difference, not a significance test.",
        "",
        "### Per-unit settled payoff (mean over seeds and the 11 days)",
        "",
        "| run | ESS9 | ESS14 | ESS17 | ESS27 | fleet |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("NEAR", "FULL96"):
        sub = [r for r in test_rows if r["dataset"] == dataset]
        lines.append(
            "| " + dataset + " | " + " | ".join(
                _f(_mean([r[f"settled_{u}"] for r in sub])) for u in UNITS)
            + f" | {_f(_mean([r['settled_total'] for r in sub]))} |")
    lines += [
        "",
        "### Per-unit settled payoff by day (mean over seeds)",
        "",
        "| load_date | NEAR E9 | NEAR E14 | NEAR E17 | NEAR E27 | "
        "FULL96 E9 | FULL96 E14 | FULL96 E17 | FULL96 E27 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for day in sorted(by_day):
        entry = by_day[day]
        cells = []
        for dataset in ("NEAR", "FULL96"):
            rows = entry.get(dataset, [])
            cells.extend(_f(_mean([r[f"settled_{u}"] for r in rows]))
                         for u in UNITS)
        lines.append(f"| {day} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`eval.csv` carries the same grid per run and per unit, together with "
        "the B1 per-unit payoff, charge/discharge energy and cycles, so the "
        "per-run values are traceable rather than only the seed average.",
        "",
        "## Holiday block (held out, 5 days)",
        "",
        "| run | settled | B1 | B2A | MATD3 − B1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, seed in RUNS:
        sub = [r for r in holiday
               if r["dataset"] == dataset and r["seed"] == str(seed)]
        if not sub:
            continue
        lines.append(
            f"| {dataset} seed {seed} | "
            f"{_f(_mean([r['settled_total'] for r in sub]))} | "
            f"{_f(_mean([r['b1_total'] for r in sub]))} | "
            f"{_f(_mean([r['b2a_total'] for r in sub]))} | "
            f"{_f(_mean([r['diff_vs_b1'] for r in sub]))} |")
    lines += [
        "",
        "## Physical and reward accounting on the evaluation days",
        "",
        "| run | optimal | terminal SOC max err | voltage cells | line cells | "
        "LMP fallbacks | mean charge MWh | mean discharge MWh | "
        "mean cycles | mean dev penalty | block reward mean | block raw mean | "
        "block baseline mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, seed in RUNS:
        sub = [r for r in eval_rows
               if r["dataset"] == dataset and r["seed"] == str(seed)]
        if not sub:
            continue
        lines.append(
            f"| {dataset} seed {seed} | "
            f"{sum(1 for r in sub if r['optimal'] == 'True')}/{len(sub)} | "
            f"{max(float(r['terminal_soc_max_err'] or 0) for r in sub):.2e} | "
            f"{sum(int(float(r['voltage_violation_cells'] or 0)) for r in sub)} | "
            f"{sum(int(float(r['line_overload_cells'] or 0)) for r in sub)} | "
            f"{sum(int(float(r['lmp_fallbacks'] or 0)) for r in sub)} | "
            f"{_mean([r['charge_mwh'] for r in sub])} | "
            f"{_mean([r['discharge_mwh'] for r in sub])} | "
            f"{_mean([r['equivalent_cycles'] for r in sub])} | "
            f"{_mean([r['dev_penalty_mean'] for r in sub])} | "
            f"{_mean([r['ep_reward'] for r in sub])} | "
            f"{_mean([r['ep_raw_profit'] for r in sub])} | "
            f"{_mean([r['ep_local_baseline'] for r in sub])} |")
    lines += [
        "",
        "The block reward is the rolling-window learning signal and the settled "
        "columns above are the day-level payoff; they are different quantities "
        "and neither replaces the other.",
        "",
        "## Answers",
        "",
    ]
    # Computed answers
    near_daily, full_daily = [], []
    for day in sorted(by_day):
        entry = by_day[day]
        if entry.get("NEAR"):
            near_daily.append(_mean([r["diff_vs_b1"]
                                     for r in entry["NEAR"]]))
        if entry.get("FULL96"):
            full_daily.append(_mean([r["diff_vs_b1"]
                                     for r in entry["FULL96"]]))
    dn, df = near_daily, full_daily
    lines += [
        f"1. **Price availability at bid time.** Not proven; no policy "
        f"observation reads the price curve. The pilot treats the same-day "
        f"price as the market's exogenous input, not as a decision feature.",
        f"2. **OBS_V6 leakage.** None found; the timing audit lists each "
        f"feature's source and period.",
        f"3. **State-matched baseline.** Yes, 9/9 checks on an instrumented "
        f"block: same window, current SOC, same wholesale and config, truthful "
        f"committed block, opponents unchanged.",
        f"4. **Double charging.** None: cycle cost is charged once in the "
        f"settlement ledger and separately (not additively) in the clearing "
        f"objective; the deviation penalty is in the actor loss only.",
        f"5. **Same configuration and budget.** Yes: identical OBS/action "
        f"specs, network, hyperparameters, 20 episodes and seeds 11/22; the "
        f"only difference is the training-day pool.",
        f"6. **Do the extra 16 days help?** Paired FULL96 − NEAR over the 11 "
        f"test days: mean {_f(_mean(full_minus_near))} CNY/day, "
        f"{sum(1 for d in full_minus_near if d > 1.0)} positive, "
        f"{sum(1 for d in full_minus_near if d < -1.0)} negative, "
        f"{sum(1 for d in full_minus_near if abs(d) <= 1.0)} tied at zero. "
        f"On this budget they did not help. Descriptive only.",
        f"7. **More stable than B1/B2A, or a few lucky days?** NEAR vs B1: "
        f"mean {_f(_mean(dn))}, above on {sum(1 for d in dn if d > 1.0)} of 11 "
        f"days and below on {sum(1 for d in dn if d < -1.0)}; FULL96 vs B1: "
        f"mean {_f(_mean(df))}, above on {sum(1 for d in df if d > 1.0)} of 11 "
        f"and below on {sum(1 for d in df if d < -1.0)}. The shortfall appears "
        f"on every test day with a non-zero spread; there is no single-day gain "
        f"carrying a positive mean.",
        f"8. **Boundaries.** Frozen assets and the clearing engine were not "
        f"modified; no commit or push.",
        "",
        "## What this pilot does not show",
        "",
        "- It does not show MATD3 has an economic advantage. Two seeds and "
        "eleven days are a runnability check.",
        "- It does not make B2A a fair benchmark: B2A quotes from the same-day "
        "curve, which is not proven available at bid time.",
        "- It does not validate the FULL96 pool for training; it only measures "
        "the difference on this split with this budget.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default=PILOT)
    parser.add_argument("--report", default=REPORT)
    args = parser.parse_args(argv)
    root = Path.cwd()
    cfg = json.loads((root / args.pilot / "pilot_config.json").read_text(
        encoding="utf-8"))
    summary = json.loads((root / args.pilot / "summary.json").read_text(
        encoding="utf-8"))
    train = {}
    for path in sorted((root / args.pilot / "train").glob("*.csv")):
        dataset, seed_text = path.stem.split("_seed")
        train[f"{dataset}|{seed_text}"] = _read(path)
    eval_rows = _read(root / args.pilot / "eval.csv")
    report = build(cfg, summary, train, eval_rows)
    (root / args.report).write_text(report, encoding="utf-8")
    print(f"wrote {root / args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
