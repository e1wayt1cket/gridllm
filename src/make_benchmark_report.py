# make_benchmark_report.py
"""Assemble the benchmark report and the long-pair analysis from the clears.

Inputs are the raw benchmark rows (B0, B2A, B2B), the JN-Base rows (B1), the
B3 price-taker bound and the common split. The report keeps the mechanisms
apart: B0/B1/B2 are cleared markets, B3 is not.

Usage:
    PYTHONPATH=src python src/make_benchmark_report.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

RAW = "data/processed/v2_cy_jn_2025/benchmark/benchmark_raw.csv"
B3 = "data/processed/v2_cy_jn_2025/benchmark/b3_lp_bound.csv"
PARAMS = "data/processed/v2_cy_jn_2025/benchmark/rule_params.json"
RESULTS = "reports/v2_cy_jn_2025/results.csv"
PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"
COMMON = "data/processed/v2_cy_jn_2025/common_split.json"
RESULTS_REPORT = "reports/v2_cy_jn_2025/benchmark_results.md"
LONG_REPORT = "reports/v2_cy_jn_2025/long_pair_16day_analysis.md"
SPLITS = ("train", "validation", "test", "holiday_block", "full96_only")
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
ARM_LABEL = {"B0": "B0 NoESS", "B1": "B1 Truthful (= JN-Base)",
             "B2A": "B2A price threshold", "B2B": "B2B threshold + SOC band"}


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _split_map(common: dict) -> Dict[str, str]:
    out = {}
    for key, ids in common["common"].items():
        for pid in ids:
            out[pid.split("|")[0]] = key
    for pid in common.get("full96_only", []):
        out[pid.split("|")[0]] = "full96_only"
    return out


def _row(arm, source, split) -> dict:
    return {
        "arm": arm,
        "candidate": source.get("candidate", ""),
        "load_date": source["load_date"],
        "price_date": source["price_date"],
        "group": source["group"],
        "split": split,
        "payoff": float(source.get("ess_total_payoff_cny", 0.0) or 0.0),
        "units": {u: float(source.get(f"{u}_payoff_cny", 0.0) or 0.0)
                  for u in UNITS},
        "served": float(source.get("served_mwh", 0.0) or 0.0),
        "unserved": float(source.get("unserved_mwh", 0.0) or 0.0),
        "curtailment": float(source.get("curtailment_mwh", 0.0) or 0.0),
        "vmin": float(source.get("min_bus_voltage_pu", 0.0) or 0.0),
        "line": float(source.get("max_line_utilization", 0.0) or 0.0),
        "charge": float(source.get("ess_charge_mwh", 0.0) or 0.0),
        "discharge": float(source.get("ess_discharge_mwh", 0.0) or 0.0),
        "optimal": str(source.get("optimal", "")).lower() in ("true", "1"),
    }


def load_rows() -> Dict[str, List[dict]]:
    common = json.loads(Path(COMMON).read_text(encoding="utf-8"))
    split = _split_map(common)
    raw = _read_csv(Path(RAW))
    arms = collections.defaultdict(list)
    for row in raw:
        arms[row["arm"]].append(_row(row["arm"], row, row["split"]))
    for row in _read_csv(Path(RESULTS)):
        if row.get("arm") != "JN-Base":
            continue
        arms["B1"].append(_row("B1", row,
                               split.get(row["load_date"], "")))
    return arms


def _summary(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    payoff = [r["payoff"] for r in rows]
    return {
        "days": len(rows),
        "mean": float(np.mean(payoff)),
        "median": float(np.median(payoff)),
        "std": float(np.std(payoff)),
        "positive": sum(1 for p in payoff if p > 1.0),
        "served": float(np.mean([r["served"] for r in rows])),
        "curtail": float(np.mean([r["curtailment"] for r in rows])),
        "charge": float(np.mean([r["charge"] for r in rows])),
        "discharge": float(np.mean([r["discharge"] for r in rows])),
        "vmin": min(r["vmin"] for r in rows),
        "line": max(r["line"] for r in rows),
        "optimal": sum(1 for r in rows if r["optimal"]),
    }


def _fmt(value: float, digits: int = 1) -> str:
    # A settled zero reads as 0.0, not -0.0.
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def build_results_report(arms: Dict[str, List[dict]], b3: List[dict],
                         params: dict) -> str:
    lines = [
        "# Benchmark results (V2-CY-JN-2025)",
        "",
        "All figures are cross-year controlled simulations: a 2022 metered "
        "load day priced by a 2025 Jiangnan day-ahead curve. The price unit is "
        "an inference, not an externally confirmed data dictionary. Nothing "
        "here is a same-period settlement.",
        "",
        "## Arms and mechanisms",
        "",
        "| arm | mechanism | decision information | storage |",
        "|---|---|---|---|",
        "| B0 NoESS | day-ahead SOCP clearing, units removed | the published "
        "DA price | none |",
        "| B1 Truthful | day-ahead SOCP clearing, all quotes truthful | the "
        "published DA price | 4 units, `(0,0)` |",
        "| B2A threshold | day-ahead SOCP clearing, rule quotes | DA price "
        "quantiles of the same day | 4 units |",
        "| B2B threshold+SOC | day-ahead SOCP clearing, rule quotes | DA "
        "price quantiles plus the declared SOC band | 4 units |",
        "| B3 LP bound | **not a clearing**: one unit, perfect foresight of "
        "the external price | the whole external DA curve | price taker |",
        "",
        "B3 is a reference value, not an outcome. It ignores nodal prices and "
        "any effect the units have on the clearing. The fleet figure is four "
        "times the single-unit bound, so it is an upper bound that no "
        "simultaneous plan is required to reach.",
        "",
        "## Tuning",
        "",
        f"- Selected B2A: `{params['selected'].get('B2A')}`",
        f"- Selected B2B: `{params['selected'].get('B2B')}`",
        "- Search: declared grid on the 53 common training days; the top two "
        "per family were re-scored on the 11 common validation days and one "
        "selected. The test and holiday days never entered the search.",
        "",
        "### Candidate search (common training days)",
        "",
        "| family | candidate | train days | mean | median | std | "
        "days > 1 CNY | validation mean | selected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    selected = params.get("selected", {})
    for family in ("B2A", "B2B"):
        train_rows = [r for r in arms.get(family, []) if r["split"] == "train"]
        val_rows = [r for r in arms.get(family, [])
                    if r["split"] == "validation"]
        val_means = collections.defaultdict(list)
        for r in val_rows:
            val_means[r["candidate"]].append(r["payoff"])
        by_candidate = collections.defaultdict(list)
        for r in train_rows:
            by_candidate[r["candidate"]].append(r["payoff"])
        for cand in sorted(by_candidate):
            values = by_candidate[cand]
            val = (f"{np.mean(val_means[cand]):.1f}"
                   if cand in val_means else "—")
            lines.append(
                f"| {family} | `{cand}` | {len(values)} | "
                f"{np.mean(values):.1f} | {np.median(values):.1f} | "
                f"{np.std(values):.1f} | "
                f"{sum(1 for v in values if v > 1.0)} | {val} | "
                f"{'yes' if cand == selected.get(family) else ''} |")
    lines += [
        "",
        "Validation did not discriminate the candidates: the two top "
        "candidates' validation means differed by less than 0.01% for B2A and "
        "were identical for B2B, so the declared tie-break (candidate id, "
        "descending) chose. The selection is a recorded rule, not evidence "
        "that one threshold is better.",
        "",
        "## Summary by split",
        "",
    ]
    selected = params.get("selected", {})

    def _selected_rows(arm, split):
        rows = [r for r in arms.get(arm, []) if r["split"] == split]
        if arm in ("B2A", "B2B"):
            rows = [r for r in rows
                    if r["candidate"] == selected.get(arm)]
        return rows

    for split in SPLITS:
        rows = [r for arm in arms for r in arms[arm]
                if r["split"] == split and arm != "B3"]
        if not rows:
            continue
        lines.append(f"### {split} ({len({r['load_date'] for r in rows})} days)")
        lines.append("")
        lines.append("| arm | days | mean payoff (CNY/day) | median | std | "
                     "days > 1 CNY | served MWh | curtail MWh | "
                     "charge MWh | discharge MWh | min V | max line | optimal |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                     "---:|---:|---:|")
        for arm in ("B0", "B1", "B2A", "B2B"):
            stats = _summary(_selected_rows(arm, split))
            if not stats:
                continue
            lines.append(
                f"| {ARM_LABEL[arm]} | {stats['days']} | "
                f"{_fmt(stats['mean'])} | {_fmt(stats['median'])} | "
                f"{_fmt(stats['std'])} | {stats['positive']} | "
                f"{_fmt(stats['served'])} | {_fmt(stats['curtail'], 2)} | "
                f"{_fmt(stats['charge'], 2)} | {_fmt(stats['discharge'], 2)} | "
                f"{stats['vmin']:.4f} | {stats['line']:.4f} | "
                f"{stats['optimal']}/{stats['days']} |")
        lines.append("")

    # Paired per-day table on the common test set.
    by_key = {}
    for arm in ("B0", "B1", "B2A", "B2B"):
        for row in arms.get(arm, []):
            by_key[(arm, row["load_date"])] = row
    test_days = sorted({r["load_date"] for r in arms.get("B1", [])
                        if r["split"] == "test"})
    lines += [
        "## Common test set: per-day paired results",
        "",
        f"{len(test_days)} days, the same days for both datasets. Storage "
        "payoff in CNY/day; B0 has no storage, so its fleet payoff is 0 by "
        "definition.",
        "",
        "| load_date | price_date | group | B0 | B1 | B2A | B2B | "
        "B1-B0 | B2A-B1 | B2B-B1 | ESS9 B1 | ESS14 B1 | ESS17 B1 | ESS27 B1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for day in test_days:
        b0 = by_key.get(("B0", day), {})
        b1 = by_key.get(("B1", day), {})
        a = by_key.get(("B2A", day), {})
        b = by_key.get(("B2B", day), {})
        units = b1.get("units", {})
        lines.append(
            f"| {day} | {b1.get('price_date', '')} | {b1.get('group', '')} | "
            f"{_fmt(b0.get('payoff', 0.0))} | {_fmt(b1.get('payoff', 0.0))} | "
            f"{_fmt(a.get('payoff', 0.0))} | {_fmt(b.get('payoff', 0.0))} | "
            f"{_fmt(b1.get('payoff', 0.0) - b0.get('payoff', 0.0))} | "
            f"{_fmt(a.get('payoff', 0.0) - b1.get('payoff', 0.0))} | "
            f"{_fmt(b.get('payoff', 0.0) - b1.get('payoff', 0.0))} | "
            + " | ".join(f"{_fmt(units.get(u, 0.0))}" for u in UNITS)
            + " |")
    lines += [
        "",
        "### Per-unit paired differences on the common test set",
        "",
        "B0 has no storage, so its per-unit column is 0 by definition and the "
        "B1-B0 column is the value of adding that unit.",
        "",
        "| load_date | unit | B0 | B1 | B2A | B2B | B1-B0 | B2A-B1 | B2B-B1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for day in test_days:
        b1 = by_key.get(("B1", day), {})
        a = by_key.get(("B2A", day), {})
        b = by_key.get(("B2B", day), {})
        for unit in UNITS:
            p1 = b1.get("units", {}).get(unit, 0.0)
            pa = a.get("units", {}).get(unit, 0.0)
            pb = b.get("units", {}).get(unit, 0.0)
            lines.append(
                f"| {day} | {unit} | 0.0 | {_fmt(p1)} | {_fmt(pa)} | "
                f"{_fmt(pb)} | {_fmt(p1)} | {_fmt(pa - p1)} | {_fmt(pb - p1)} |")
    lines += [
        "",
        "### Paired differences on the common test set",
        "",
        "| comparison | days | mean | median | std | days positive | "
        "days zero | days negative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, left, right in (("B1 - B0", "B1", "B0"),
                               ("B2A - B1", "B2A", "B1"),
                               ("B2B - B1", "B2B", "B1")):
        diffs = [by_key[(left, d)]["payoff"] - by_key[(right, d)]["payoff"]
                 for d in test_days
                 if (left, d) in by_key and (right, d) in by_key]
        if not diffs:
            continue
        lines.append(
            f"| {label} | {len(diffs)} | {_fmt(np.mean(diffs))} | "
            f"{_fmt(np.median(diffs))} | {_fmt(np.std(diffs))} | "
            f"{sum(1 for d in diffs if d > 1.0)} | "
            f"{sum(1 for d in diffs if abs(d) <= 1.0)} | "
            f"{sum(1 for d in diffs if d < -1.0)} |")
    lines += [
        "",
        "Eleven days is too few for an inference claim. The table is the "
        "required per-day pairing, not a significance test.",
        "",
        "## Holiday block (held out from tuning)",
        "",
        "| load_date | price_date | B0 | B1 | B2A | B2B |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for day in sorted({r["load_date"] for r in arms.get("B1", [])
                       if r["split"] == "holiday_block"}):
        b1 = by_key.get(("B1", day), {})
        lines.append(
            f"| {day} | {b1.get('price_date', '')} | "
            f"{_fmt(by_key.get(('B0', day), {}).get('payoff', 0.0))} | "
            f"{_fmt(b1.get('payoff', 0.0))} | "
            f"{_fmt(by_key.get(('B2A', day), {}).get('payoff', 0.0))} | "
            f"{_fmt(by_key.get(('B2B', day), {}).get('payoff', 0.0))} |")
    lines += [
        "",
        "## B3 price-taker reference",
        "",
        "| split | days | mean single-unit bound | mean fleet (4x) bound |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        rows = [r for r in b3 if r["split"] == split]
        if not rows:
            continue
        lines.append(
            f"| {split} | {len(rows)} | "
            f"{np.mean([float(r['unit_margin_cny']) for r in rows]):.1f} | "
            f"{np.mean([float(r['fleet4_upper_bound_cny']) for r in rows]):.1f} |")
    lines += [
        "",
        "The bound is computed against the **external** price series. It is "
        "not comparable with B1/B2 payoffs as if they were the same quantity: "
        "the bound knows the whole curve, faces no market impact and settles "
        "at the external price, while B1/B2 settle at nodal LMPs they helped "
        "set.",
        "",
        "## What these tables do not support",
        "",
        "- No MATD3 policy was trained; nothing here is an RL result.",
        "- B1 is not an independent arm from JN-Base; it is the same clearing.",
        "- The 21-day C80/C90/C100/NoESS probe is not the 96-day sample.",
        "- FULL96-only long pairs are training-only and are not in the common "
        "test set.",
        "- A rule earning more than truthful is a **participant payoff** "
        "result, not a welfare result. The clearing objective values storage "
        "at its own quotes and is explicitly not social welfare; the two are "
        "different quantities in this project.",
    ]
    return "\n".join(lines) + "\n"


def build_long_report(arms: Dict[str, List[dict]]) -> str:
    pairing = _read_csv(Path(PAIRING))
    long_rows = [r for r in pairing
                 if int(r["month_day_gap_days"] or 0) > 14]
    common_days = {r["load_date"] for arm in ("B1",) for r in arms.get(arm, [])
                   if r["split"] != "full96_only"}
    long_days = {r["load_date"] for r in long_rows}
    b1 = {r["load_date"]: r for r in arms.get("B1", [])}
    b0 = {r["load_date"]: r for r in arms.get("B0", [])}

    def _median(values):
        return float(np.median(values)) if values else float("nan")

    common_pairs = [r for r in pairing if r["load_date"] in common_days]
    lines = [
        "# The 16 FULL96-only long pairs (V2-CY-JN-2025)",
        "",
        "These are the pairs NEAR excludes: `month_day_gap_days > 14`. They "
        "are **training-only** in FULL96 and must not enter the common test "
        "set or any NEAR result.",
        "",
        "## Coverage",
        "",
        f"- {len(long_rows)} pairs: "
        + ", ".join(f"{g} {n}" for g, n in
                    sorted(collections.Counter(r["group"]
                                               for r in long_rows).items()))
        + ".",
        f"- Months covered by their price dates: "
        + ", ".join(sorted({r["price_date"][:7] for r in long_rows})) + ".",
        "",
        "| load_date | price_date | group | gap (days) | price mean | "
        "price range | network peak MW | B1 fleet payoff | B0 served MWh |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(long_rows, key=lambda r: -int(r["month_day_gap_days"])):
        day = row["load_date"]
        lines.append(
            f"| {day} | {row['price_date']} | {row['group']} | "
            f"{row['month_day_gap_days']} | {float(row['price_mean']):.1f} | "
            f"{float(row['price_min']):.0f}-{float(row['price_max']):.0f} | "
            f"{float(row['network_peak_mw']):.2f} | "
            f"{b1.get(day, {}).get('payoff', float('nan')):.1f} | "
            f"{b0.get(day, {}).get('served', float('nan')):.1f} |")
    lines += [
        "",
        "## Distribution shift against the common 80",
        "",
        "| statistic | common 80 | FULL96-only 16 |",
        "|---|---:|---:|",
        f"| price daily mean, median | "
        f"{_median([float(r['price_mean']) for r in common_pairs]):.1f} | "
        f"{_median([float(r['price_mean']) for r in long_rows]):.1f} |",
        f"| price daily max, median | "
        f"{_median([float(r['price_max']) for r in common_pairs]):.1f} | "
        f"{_median([float(r['price_max']) for r in long_rows]):.1f} |",
        f"| network peak MW, median | "
        f"{_median([float(r['network_peak_mw']) for r in common_pairs]):.2f} | "
        f"{_median([float(r['network_peak_mw']) for r in long_rows]):.2f} |",
        f"| B1 fleet payoff, median | "
        f"{_median([b1[d]['payoff'] for d in common_days if d in b1]):.1f} | "
        f"{_median([b1[d]['payoff'] for d in long_days if d in b1]):.1f} |",
        f"| B1 days with payoff > 50 CNY | "
        f"{sum(1 for d in common_days if b1.get(d, {}).get('payoff', 0) > 50)}"
        f" / {len(common_days)} | "
        f"{sum(1 for d in long_days if b1.get(d, {}).get('payoff', 0) > 50)}"
        f" / {len(long_days)} |",
        "",
        "## Are they worth keeping as training samples?",
        "",
        "They are not interchangeable with the common days. Their loads are "
        "late-November and August days, and their prices come from July, "
        "September and October, so they add load-level and price-level "
        "coverage the common 80 lack. Their price level is also higher: the "
        "median daily mean is 329.7 against 237.8. What they do not add is "
        "arbitrage signal. Their curves are flatter, so the truthful fleet "
        "earns a median 4.4 CNY/day against 119.9 on the common days, and only "
        "6 of 16 days clear 50 CNY against 45 of 80. One day (2022-11-25 to "
        "2025-10-01) carries a 30-1500 range and 3433 CNY and pulls the mean "
        "up; it is one day, not a pattern.",
        "",
        "**Recommendation:** keep them in FULL96 training only, where they may "
        "teach a policy what a higher-price, lower-spread day looks like, and "
        "never let them define a test. Whether they help is a question for "
        "the FULL96-vs-NEAR comparison on the common test set; their own "
        "payoffs are not evidence for or against the filtering rule, and "
        "because they are far more often idle than the common days they "
        "cannot be used to argue that more training days are better.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=RAW)
    parser.add_argument("--b3", default=B3)
    parser.add_argument("--params", default=PARAMS)
    parser.add_argument("--results-report", default=RESULTS_REPORT)
    parser.add_argument("--long-report", default=LONG_REPORT)
    args = parser.parse_args(argv)
    root = Path.cwd()
    arms = load_rows()
    b3 = _read_csv(root / args.b3)
    params = json.loads((root / args.params).read_text(encoding="utf-8"))
    report = build_results_report(arms, b3, params)
    (root / args.results_report).write_text(report, encoding="utf-8")
    long_report = build_long_report(arms)
    (root / args.long_report).write_text(long_report, encoding="utf-8")
    for arm, rows in sorted(arms.items()):
        print(f"{arm}: {len(rows)} rows")
    print(f"wrote {root / args.results_report}")
    print(f"wrote {root / args.long_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
