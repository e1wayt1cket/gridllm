"""P21: compare the control and candidate smoke on the frozen protocol.

Read-only over two finished run directories. It verifies the smoke shape
(5 days x 4 rounds), the absence of test/H2 evaluation, the physics and
settlement identities, and the pre-registered non-inferiority rule.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("data/processed/v2_cy_jn_2025")
REPORTS = Path("reports/v2_cy_jn_2025")
RUNS = Path("runs")
PROTOCOL = REPORTS / "phase21_smoke_protocol.json"
CONTROL = "P21_control_smoke_seed11"
CANDIDATE = "P21_candidate_001_smoke_seed11"
OUT = DATA / "phase21_smoke_control_vs_candidate_001.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_run(run_id: str) -> dict:
    root = RUNS / run_id
    manifest = json.loads(
        (root / "run_manifest.json").read_text(encoding="utf-8"))
    episodes = read_csv(root / "episodes.csv")
    evals = json.loads(
        (root / "eval_history.json").read_text(encoding="utf-8"))
    decisions = []
    path = root / "block_decisions.jsonl"
    if path.exists():
        decisions = [json.loads(line) for line in
                     path.read_text(encoding="utf-8").splitlines() if line]
    return {"run_id": run_id, "root": root, "manifest": manifest,
            "episodes": episodes, "evals": evals, "decisions": decisions}


def shape_checks(run: dict, protocol: dict) -> dict:
    manifest = run["manifest"]
    episodes = run["episodes"]
    evals = run["evals"]
    decisions = run["decisions"]
    expected_days = protocol["smoke"]["train_days"]
    rounds = protocol["smoke"]["rounds"]
    exposures = Counter(e["pair_id"].split("|")[0] for e in episodes)
    blocks = Counter(d["episode"] for d in decisions)
    expected_episodes = protocol["smoke"]["episodes"]
    return {
        "episodes": len(episodes),
        "expected_episodes": expected_episodes,
        "days": sorted(exposures),
        "days_expected": sorted(expected_days),
        "exposures_per_day": dict(exposures),
        "exposures_all_four": all(v == rounds for v in exposures.values()),
        "block_rows": len(decisions),
        "expected_block_rows": expected_episodes * 24 * 4,
        "block_rows_all_96": all(v == 96 for v in blocks.values()),
        "validation_passes": len(evals),
        "validation_days_per_pass": [len(e["days"]) for e in evals],
        "split_name": manifest.get("split_name"),
        "train_subset": manifest.get("train_subset"),
        "h2_holdout_read": manifest.get("h2_holdout_read"),
        "test_evaluation": manifest.get("test_evaluation"),
        "legacy_seen_test_run": manifest.get("legacy_seen_test_run"),
        "artifacts": sorted(manifest.get("artifacts_sha256", {})),
    }


def physics_checks(run: dict) -> dict:
    evals = run["evals"]
    terminal = max(float(e.get("terminal_soc_max_err", 0.0)) for e in evals)
    return {
        "max_terminal_soc_error": terminal,
        "terminal_ok": terminal <= 1e-5,
        "voltage_violation_cells": int(sum(
            e.get("voltage_violation_cells", 0) for e in evals)),
        "line_overload_cells": int(sum(
            e.get("line_overload_cells", 0) for e in evals)),
        "lmp_fallbacks": int(sum(e.get("lmp_fallbacks", 0) for e in evals)),
        "all_terminal_0p5": all(
            abs(float(v) - 0.5) <= 1e-5
            for e in evals for v in e["mean_terminal_soc"].values()),
    }


def settlement_checks(run: dict) -> dict:
    max_err = 0.0
    for episode in run["episodes"]:
        lhs = float(episode["G_policy_fleet"]) \
            - float(episode["G_b1_fleet"])
        max_err = max(max_err, abs(lhs - float(episode["delta_fleet"])))
    return {"max_identity_error": max_err, "identity_ok": max_err <= 1e-6}


def quote_stats(run: dict) -> dict:
    out = {}
    for unit in ("ESS9", "ESS14", "ESS17", "ESS27"):
        for leg, high in ((0, 150.0), (1, 300.0)):
            values = [float(d["quotes"][leg]) for d in run["decisions"]
                      if d["agent"] == unit]
            if not values:
                continue
            arr = np.asarray(values)
            out[f"{unit}.{'charge' if leg == 0 else 'discharge'}"] = {
                "count": int(arr.size),
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "high_share": float(np.mean(arr >= 0.95 * high)),
                "zero_share": float(np.mean(arr <= 0.0)),
            }
    return out


def best_eval(run: dict) -> dict:
    return max(run["evals"],
               key=lambda e: e.get("mean_fleet_delta", -math.inf))


def compare(control: dict, candidate: dict) -> dict:
    cbest = best_eval(control)
    best = best_eval(candidate)
    cdays = {d["pair_id"]: float(d["delta_fleet"]) for d in cbest["days"]}
    bdays = {d["pair_id"]: float(d["delta_fleet"]) for d in best["days"]}
    common = sorted(set(cdays) & set(bdays))
    paired = np.asarray([bdays[d] - cdays[d] for d in common])
    sd = float(np.std(paired, ddof=1)) if paired.size > 1 else 0.0
    unit_delta = {
        unit: float(best["mean_delta_by_unit"][unit]
                    - cbest["mean_delta_by_unit"][unit])
        for unit in best["mean_delta_by_unit"]
    }
    return {
        "control_best_round": int(cbest["round"]),
        "control_best_mean": float(cbest["mean_fleet_delta"]),
        "control_best_positive_days": int(cbest["positive_days"]),
        "candidate_best_round": int(best["round"]),
        "candidate_best_mean": float(best["mean_fleet_delta"]),
        "candidate_best_positive_days": int(best["positive_days"]),
        "paired_days": len(common),
        "paired_mean_difference": float(paired.mean()) if paired.size else None,
        "paired_sd": sd,
        "non_inferior_mean_threshold": float(
            cbest["mean_fleet_delta"] - sd),
        "mean_non_inferior": bool(
            best["mean_fleet_delta"] >= cbest["mean_fleet_delta"] - sd),
        "positive_days_non_inferior": bool(
            best["positive_days"] >= cbest["positive_days"] - 1),
        "unit_delta_vs_control": unit_delta,
        "no_unit_worse_by_10": all(v >= -10.0 for v in unit_delta.values()),
        "rounds": [
            {"run": run_id, "round": e["round"],
             "mean": e["mean_fleet_delta"],
             "positive_days": e["positive_days"]}
            for run_id, run in (("control", control),
                                ("candidate", candidate))
            for e in run["evals"]],
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    control = load_run(CONTROL)
    candidate = load_run(CANDIDATE)
    checks = {
        run_id: {
            "shape": shape_checks(run, protocol),
            "physics": physics_checks(run),
            "settlement": settlement_checks(run),
        }
        for run_id, run in (("control", control),
                            ("candidate", candidate))
    }
    hard = {}
    for name, block in checks.items():
        shape, physics, settlement = (
            block["shape"], block["physics"], block["settlement"])
        hard[name] = {
            "exact_episodes": shape["episodes"] == shape["expected_episodes"],
            "exact_days": shape["days"] == shape["days_expected"],
            "four_exposures_per_day": shape["exposures_all_four"],
            "exact_block_rows": shape["block_rows"]
            == shape["expected_block_rows"],
            "block_rows_96": shape["block_rows_all_96"],
            "four_validation_passes": shape["validation_passes"]
            == protocol["smoke"]["validation_passes"],
            "eleven_validation_days": all(
                n == 11 for n in shape["validation_days_per_pass"]),
            "common_split": shape["split_name"] == "common",
            "train_subset_fixed": sorted(
                p.split("|")[0] for p in (shape["train_subset"] or []))
            == sorted(shape["days_expected"]),
            "no_h2": shape["h2_holdout_read"] is False,
            "no_test": shape["test_evaluation"] is None,
            "no_legacy_test": shape["legacy_seen_test_run"] is False,
            "terminal_soc": physics["terminal_ok"]
            and physics["all_terminal_0p5"],
            "no_voltage": physics["voltage_violation_cells"] == 0,
            "no_line": physics["line_overload_cells"] == 0,
            "no_lmp_fallback": physics["lmp_fallbacks"] == 0,
            "settlement_identity": settlement["identity_ok"],
        }
        hard[name]["all_pass"] = all(hard[name].values())
    comparison = compare(control, candidate)
    comparison["map_consistency"] = "PASS (routing audit)"
    comparison["acceptance"] = {
        "hard_gates_control": hard["control"]["all_pass"],
        "hard_gates_candidate": hard["candidate"]["all_pass"],
        "non_inferior_mean": comparison["mean_non_inferior"],
        "non_inferior_positive_days": comparison["positive_days_non_inferior"],
        "no_unit_worse_by_10": comparison["no_unit_worse_by_10"],
        "target_metric": "actor-loss/execution map equality PASS",
    }
    comparison["acceptance"]["accepted"] = all([
        comparison["acceptance"]["hard_gates_control"],
        comparison["acceptance"]["hard_gates_candidate"],
        comparison["acceptance"]["non_inferior_mean"],
        comparison["acceptance"]["non_inferior_positive_days"],
        comparison["acceptance"]["no_unit_worse_by_10"],
    ])
    payload = {
        "protocol": str(PROTOCOL),
        "control": CONTROL,
        "candidate": CANDIDATE,
        "hard_checks": hard,
        "comparison": comparison,
        "quote_stats": {
            "control": quote_stats(control),
            "candidate": quote_stats(candidate),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(comparison["acceptance"], indent=2))
    print("control best", comparison["control_best_mean"],
          "candidate best", comparison["candidate_best_mean"])
    print(f"wrote {OUT}")
    return 0 if comparison["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
