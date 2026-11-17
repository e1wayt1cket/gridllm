"""Bind the phase-16 artifacts to the run, data and checkpoint hashes.

Read-only over the existing run. It verifies that the two checkpoints used by
the audits still match their recorded manifest and counters hashes, then
records the sha256 of every new report, data file and audit script. It does not
write to any run directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
RUN = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
ROOTS = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_audit_roots"
CKPT_DIRS = {2: "step_00002544", 10: "step_00012720"}

SCRIPTS = (
    "src/audit_critic_joint_coverage.py",
    "src/audit_critic_return_calibration.py",
    "src/audit_winter_15min_quote_acceptance.py",
    "src/audit_continuous_quote_mapping.py",
    "src/audit_gate_initialization.py",
    "src/make_phase16_manifest.py",
)
REPORTS_FILES = (
    "critic_joint_state_action_coverage.md",
    "critic_return_calibration_recheck.md",
    "winter_15min_quote_acceptance_audit.md",
    "continuous_quote_redesign_comparison.md",
    "critic_calibration_improvement_design.md",
    "gate_initialization_exploration_design.md",
    "d_a_minimal_repair_spec.md",
    "next_single_seed_pilot_protocol.md",
    "training_readiness.md",
)
DATA_FILES = (
    "critic_joint_coverage_targets.csv",
    "critic_joint_coverage_states.json",
    "critic_joint_coverage_summary.json",
    "critic_return_calibration_rows.csv",
    "critic_return_calibration_groups.csv",
    "critic_return_calibration_summary.json",
    "winter_15min_quote_acceptance.csv",
    "winter_15min_quote_acceptance.json",
    "continuous_quote_mapping_grid.csv",
    "continuous_quote_mapping_summary.json",
    "gate_initialization_exploration.csv",
    "gate_initialization_exploration.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    index = json.loads((DATA / "training_checkpoint_index.json").read_text(
        encoding="utf-8"))
    run = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    verified = {}
    for cp, dirname in CKPT_DIRS.items():
        manifest_path = ROOTS / f"round_{cp:03d}" / "manifest.json"
        counters_path = RUN / "checkpoints" / dirname / "counters.json"
        verified[str(cp)] = {
            "checkpoint_path": str(RUN / "checkpoints" / dirname),
            "step": int(index[str(cp)]["step"]),
            "manifest_sha256": sha(manifest_path),
            "manifest_matches_index": (
                sha(manifest_path) == index[str(cp)]["manifest_sha256"]),
            "counters_sha256": sha(counters_path),
            "counters_matches_index": (
                sha(counters_path) == index[str(cp)]["counters_sha256"]),
            "checkpoint_tree_sha256": index[str(cp)]["checkpoint_tree_sha256"],
            "buffer": str(RUN / "checkpoints" / dirname / "buffer/replay.npz"),
        }
    out = {
        "phase": 16,
        "training_run": {
            "run_id": run["run_id"],
            "seed": run["seed"],
            "experiment": run["experiment"],
            "config_hash": run["market_config_hash"],
            "b1_mechanism_hash": run["b1_mechanism_hash"],
            "data_hashes": run["data_hashes"],
            "code_revision": run["code_revision"],
            "train_days": run["train_days"],
            "validation_days": run["validation_days"],
            "test_days": run["test_days"],
            "budget_episodes": run["budget_episodes"],
            "exposures_per_day": run["exposures_per_day"],
            "h2_holdout_read": run["h2_holdout_read"],
            "test_evaluation": run.get("test_evaluation"),
        },
        "checkpoints": verified,
        "boundary": {
            "training_started": False,
            "h2_holdout_read": False,
            "legacy_seen_test_run": False,
            "checkpoints_overwritten": not all(
                v["manifest_matches_index"] and v["counters_matches_index"]
                for v in verified.values()),
            "note": "The audits read the run, its checkpoints and the "
                    "training/validation data only. The H2 and legacy-test "
                    "artifacts were not read or executed.",
        },
        "scripts": {p: sha(ROOT / p) for p in SCRIPTS},
        "reports": {p: sha(REPORTS / p) for p in REPORTS_FILES},
        "data": {p: sha(DATA / p) for p in DATA_FILES},
    }
    (DATA / "phase16_artifact_hashes.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "checkpoints": {k: {"manifest": v["manifest_matches_index"],
                            "counters": v["counters_matches_index"]}
                        for k, v in verified.items()},
        "boundary": out["boundary"],
        "files": {"scripts": len(SCRIPTS), "reports": len(REPORTS_FILES),
                  "data": len(DATA_FILES)},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
