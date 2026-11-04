"""Bind the phase-18 counterfactual, grid and reports to their hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
NEW = ROOT / "runs/DEV_DA_GATE_seed11_v2"
NEW_BEST = ROOT / "runs/DEV_DA_GATE_seed11_v2_best"
OLD = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
OLD_ROOTS = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_audit_roots"

SCRIPTS = ("src/audit_daily_counterfactual.py",
           "src/audit_single_ess_grid.py",
           "src/audit_phase18_analysis.py",
           "src/make_phase18_manifest.py")
REPORTS_FILES = ("daily_counterfactual_action_validation.md",
                 "single_ess_daily_quote_grid.md",
                 "critic_vs_rollout_ground_truth.md",
                 "phase18_diagnostic_decision.md",
                 "training_readiness.md")
DATA_FILES = ("daily_counterfactual_action_validation.csv",
              "daily_counterfactual_action_validation.json",
              "single_ess_quote_surface.csv",
              "single_ess_quote_surface.json",
              "single_ess_quote_surface_ESS14.csv",
              "single_ess_quote_surface_ESS14.json",
              "single_ess_quote_surface_ESS27.csv",
              "single_ess_quote_surface_ESS27.json",
              "phase18_analysis.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    old_index = json.loads(
        (DATA / "training_checkpoint_index.json").read_text(encoding="utf-8"))
    old = {}
    for cp, name in ((2, "step_00002544"), (10, "step_00012720")):
        old[str(cp)] = {
            "manifest": sha(OLD_ROOTS / f"round_{cp:03d}" / "manifest.json"),
            "manifest_matches": (
                sha(OLD_ROOTS / f"round_{cp:03d}" / "manifest.json")
                == old_index[str(cp)]["manifest_sha256"]),
            "counters": sha(OLD / "checkpoints" / name / "counters.json"),
            "counters_matches": (
                sha(OLD / "checkpoints" / name / "counters.json")
                == old_index[str(cp)]["counters_sha256"]),
        }
    new_run = json.loads((NEW / "run_manifest.json").read_text(
        encoding="utf-8"))
    new_best = json.loads((NEW_BEST / "manifest.json").read_text(
        encoding="utf-8"))
    new_final = json.loads((NEW / "manifest.json").read_text(
        encoding="utf-8"))
    out = {
        "phase": 18,
        "checkpoints": {
            "da_best": {"root": str(NEW_BEST),
                        "step": int(new_best["step"]),
                        "schema": new_best["schema_version"]},
            "da_final": {"root": str(NEW),
                         "step": int(new_final["step"]),
                         "schema": new_final["schema_version"]},
            "old_R2": {"root": str(OLD_ROOTS / "round_002"),
                       "step": int(old_index["2"]["step"])},
            "old_R10": {"root": str(OLD_ROOTS / "round_010"),
                        "step": int(old_index["10"]["step"])},
        },
        "old_checkpoints": old,
        "run": {
            "run_id": new_run["run_id"],
            "seed": new_run["seed"],
            "train_days": new_run["train_days"],
            "validation_days": new_run["validation_days"],
            "test_days": new_run["test_days"],
            "data_hashes": new_run["data_hashes"],
            "market_config_hash": new_run["market_config_hash"],
            "b1_mechanism_hash": new_run["b1_mechanism_hash"],
            "h2_holdout_read": new_run.get("h2_holdout_read"),
            "legacy_seen_test_run": new_run.get("legacy_seen_test_run"),
            "test_evaluation": new_run.get("test_evaluation"),
        },
        "boundary": {
            "training_started": False,
            "checkpoint_modified": False,
            "h2_holdout_read": False,
            "legacy_seen_test_run": False,
            "old_checkpoints_overwritten": not all(
                v["manifest_matches"] and v["counters_matches"]
                for v in old.values()),
            "commit_or_push": False,
        },
        "scripts": {p: sha(ROOT / p) for p in SCRIPTS},
        "reports": {p: sha(REPORTS / p) for p in REPORTS_FILES},
        "data": {p: sha(DATA / p) for p in DATA_FILES},
    }
    (DATA / "phase18_artifact_hashes.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "checkpoints": out["checkpoints"],
        "old": {k: {"manifest": v["manifest_matches"],
                    "counters": v["counters_matches"]}
                for k, v in old.items()},
        "boundary": out["boundary"],
        "files": {"scripts": len(SCRIPTS), "reports": len(REPORTS_FILES),
                  "data": len(DATA_FILES)},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
