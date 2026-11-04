"""Bind the phase-17 implementation, pilot and reports to their hashes.

Read-only over the new and old runs. It verifies that the old R2/R10
manifest/counters still match the phase-15 index and that the new checkpoints
carry the v3 schema, then writes the artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
RUN = ROOT / "runs/DEV_DA_GATE_seed11_v2"
BEST = ROOT / "runs/DEV_DA_GATE_seed11_v2_best"
OLD = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
OLD_ROOTS = ROOT / "runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_audit_roots"

SCRIPTS = (
    "src/da_quote_map.py", "src/da_replay.py", "src/rl_da_gate.py",
    "src/train_da_gate.py", "src/audit_da_gate_offline.py",
    "src/audit_da_gate_pilot.py", "src/audit_da_sampler_loss.py",
    "src/make_da_gate_artifacts.py", "src/make_phase17_manifest.py",
    "src/rl_spec.py", "src/checkpoint_writer_v2.py",
)
REPORTS_FILES = (
    "continuous_quote_mapping_v2_audit.md",
    "critic_stratified_replay_v2.md",
    "controlled_quote_exploration_v2.md",
    "gate_temperature_warmup_v2.md",
    "da_gate_trainer_integration_audit.md",
    "da_gate_offline_validation.md",
    "da_gate_single_seed_pilot_results.md",
    "training_readiness.md",
)
DATA_FILES = (
    "da_quote_map_grid.csv",
    "da_gate_exploration_log.csv",
    "da_gate_sampling_probabilities.csv",
    "da_gate_buffer_coverage.json",
    "da_gate_pilot_summary.json",
    "da_gate_material_calibration.csv",
    "da_gate_material_calibration_groups.csv",
    "da_gate_material_calibration.json",
    "da_gate_winter_15min.csv",
    "da_gate_winter_15min.json",
    "da_gate_sampler_critic_loss.json",
    "da_gate_offline_validation.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    old_index = json.loads(
        (DATA / "training_checkpoint_index.json").read_text(encoding="utf-8"))
    old_verified = {}
    for cp, name in ((2, "step_00002544"), (10, "step_00012720")):
        old_verified[str(cp)] = {
            "manifest_sha256": sha(OLD_ROOTS / f"round_{cp:03d}" /
                                   "manifest.json"),
            "manifest_matches": (
                sha(OLD_ROOTS / f"round_{cp:03d}" / "manifest.json")
                == old_index[str(cp)]["manifest_sha256"]),
            "counters_sha256": sha(OLD / "checkpoints" / name /
                                   "counters.json"),
            "counters_matches": (
                sha(OLD / "checkpoints" / name / "counters.json")
                == old_index[str(cp)]["counters_sha256"]),
        }
    new_run = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    new_best = json.loads((BEST / "manifest.json").read_text(
        encoding="utf-8"))
    new_final = json.loads((RUN / "manifest.json").read_text(
        encoding="utf-8"))
    new_checkpoints = {
        "best": {
            "root": str(BEST),
            "step": int(new_best["step"]),
            "schema": new_best["schema_version"],
            "action_schema": new_best["specs"]["action_spec"],
            "manifest_sha256": sha(BEST / "manifest.json"),
            "counters_sha256": sha(
                BEST / f"checkpoints/step_{int(new_best['step']):08d}"
                / "counters.json"),
        },
        "final": {
            "root": str(RUN),
            "step": int(new_final["step"]),
            "schema": new_final["schema_version"],
            "action_schema": new_final["specs"]["action_spec"],
            "manifest_sha256": sha(RUN / "manifest.json"),
            "counters_sha256": sha(
                RUN / f"checkpoints/step_{int(new_final['step']):08d}"
                / "counters.json"),
        },
    }
    out = {
        "phase": 17,
        "run": {
            "run_id": new_run["run_id"],
            "seed": new_run["seed"],
            "status": new_run["status"],
            "budget_episodes": new_run["budget_episodes"],
            "episodes_run": new_run["episodes_run"],
            "best": new_run["best"],
            "experiment": new_run["experiment"],
            "action_schema": new_run.get("action_schema"),
            "quote_map": new_run.get("quote_map"),
            "sampler": new_run.get("sampler"),
            "exploration": new_run.get("exploration"),
            "gate_schedule": new_run.get("gate_schedule"),
            "train_days": new_run["train_days"],
            "validation_days": new_run["validation_days"],
            "test_days": new_run["test_days"],
            "test_evaluation": new_run.get("test_evaluation"),
            "h2_holdout_read": new_run.get("h2_holdout_read"),
            "legacy_seen_test_run": new_run.get("legacy_seen_test_run"),
            "data_hashes": new_run["data_hashes"],
            "market_config_hash": new_run["market_config_hash"],
            "b1_mechanism_hash": new_run["b1_mechanism_hash"],
            "source_sha256": new_run["source_sha256"],
        },
        "new_checkpoints": new_checkpoints,
        "old_checkpoints": old_verified,
        "boundary": {
            "pilot_training_started": True,
            "formal_five_seed_training_started": False,
            "h2_holdout_read": False,
            "legacy_seen_test_run": False,
            "old_checkpoints_overwritten": not all(
                v["manifest_matches"] and v["counters_matches"]
                for v in old_verified.values()),
            "commit_or_push": False,
        },
        "scripts": {p: sha(ROOT / p) for p in SCRIPTS},
        "reports": {p: sha(REPORTS / p) for p in REPORTS_FILES},
        "data": {p: sha(DATA / p) for p in DATA_FILES},
        "run_logs": {
            "episodes.csv": sha(RUN / "episodes.csv"),
            "eval_history.json": sha(RUN / "eval_history.json"),
            "gate_stats.jsonl": sha(RUN / "gate_stats.jsonl"),
            "sampler_stats.jsonl": sha(RUN / "sampler_stats.jsonl"),
            "run_manifest.json": sha(RUN / "run_manifest.json"),
            "exposure.json": sha(RUN / "exposure.json"),
        },
    }
    (DATA / "phase17_artifact_hashes.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "new_checkpoints": {k: {"step": v["step"], "schema": v["schema"]}
                            for k, v in new_checkpoints.items()},
        "old_checkpoints": {k: {"manifest": v["manifest_matches"],
                                "counters": v["counters_matches"]}
                            for k, v in old_verified.items()},
        "boundary": out["boundary"],
        "files": {"scripts": len(SCRIPTS), "reports": len(REPORTS_FILES),
                  "data": len(DATA_FILES)},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
