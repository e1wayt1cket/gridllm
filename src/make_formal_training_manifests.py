"""Build and check the formal-training data, config and code manifests.

The common split's NEAR view is the formal data definition. Its test and
holiday blocks are the LEGACY_SEEN_TEST set. The H2 price-date holdout is not
opened by this script; only the status file's legacy label is read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(".")
DATA = ROOT / "data/processed/v2_cy_jn_2025"
OUT = ROOT / "reports/v2_cy_jn_2025/formal_training_ready"
CANDIDATE = "P21_candidate_001"
INPUTS = (
    "near/load_price_pairing.csv",
    "bus_loads_96x33.npz",
    "price_curves.npz",
    "common_split.json",
    "holdout_status.json",
    "b1_terminal_neutral_manifest.json",
    "b1_terminal_neutral_results.csv",
    "b1_terminal_neutral_block_payoffs.json",
    "terminal_soc_config.json",
    "trainer_configs.json",
)
SOURCE_FILES = (
    "rl_da_gate.py", "rl_b_c2_gate.py", "train_da_gate.py", "da_quote_map.py",
    "da_replay.py", "truthful_gate_actor_adapter.py", "rl_spec.py",
    "rl_env.py", "c2_reward.py", "daily_reward_b.py",
    "b1_daily_baseline.py", "dispatch_socp.py", "v2_cy_jn_env.py",
    "checkpoint_writer_v2.py", "checkpoint_loader_v2.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()
        return {"revision": rev, "dirty": bool(dirty),
                "dirty_paths": [line[3:] for line in dirty.splitlines()]}
    except Exception:
        return {"revision": "working-tree", "dirty": True,
                "dirty_paths": []}


def data_payload() -> dict:
    split_doc = json.loads((DATA / "common_split.json").read_text(
        encoding="utf-8"))
    near = split_doc["near"]
    sets = {name: set(near[name]) for name in
            ("train", "validation", "test", "holiday_block")}
    overlaps = {}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f"{left}/{right}"] = sorted(sets[left] & sets[right])
    legacy = sets["test"] | sets["holiday_block"]
    status = json.loads((DATA / "holdout_status.json").read_text(
        encoding="utf-8"))
    declared = set(status["legacy_seen_test"]["pairs"])
    return {
        "split_file": str(DATA / "common_split.json"),
        "split_view": "near",
        "split_seed": split_doc["seed"],
        "pair_key": split_doc.get("pair_key", "load_date|price_date"),
        "counts": {name: len(sets[name]) for name in names},
        "train_pairs": sorted(sets["train"]),
        "validation_pairs": sorted(sets["validation"]),
        "test_pairs": sorted(sets["test"]),
        "holiday_pairs": sorted(sets["holiday_block"]),
        "legacy_seen_test_pairs": sorted(legacy),
        "legacy_label": status["legacy_seen_test"]["label"],
        "legacy_label_matches_declared": sorted(legacy) == sorted(declared),
        "overlaps": overlaps,
        "all_disjoint": all(not v for v in overlaps.values()),
        "h2_opened": False,
        "input_sha256": {
            name: sha(DATA / name) for name in INPUTS
        },
        "test_label": "LEGACY_SEEN_TEST",
        "independent_generalisation_claim": False,
    }


def config_payload(data: dict, code: dict) -> dict:
    return {
        "formal_training": {
            "experiment_id": "FORMAL_NEAR_COMMON_B_C2_GATE",
            "protocol_version": 2,
            "accepted_candidate": CANDIDATE,
            "dataset": "JN-Base-NEAR / common_split near view",
            "split": data["split_file"],
            "split_view": data["split_view"],
            "split_seed": data["split_seed"],
            "reward_mode": "B_C2",
            "gamma": 0.99,
            "fixed_scale": 1000.0,
            "terminal_soc": {"initial": 0.5, "final": 0.5,
                             "day_end_window_only": True},
            "cycle_cost_cny_per_mwh": 100.0,
            "clearing_engine": "socp",
            "network_support": "line50_q2.0",
            "b1_source": "b1_terminal_neutral",
            "settlement": "two_settlement_ledger",
            "observation": {"name": "v6_12d", "version": 6,
                            "dimensions": 12},
            "action": {"name": "storage_quote_v3", "version": 3,
                       "quote_map": "linear_core_v1",
                       "dimensions_per_unit": 2,
                       "ess_order": ["ESS9", "ESS14", "ESS17", "ESS27"]},
            "gate": {"per_unit": 2, "total": 8, "estimator":
                     "straight_through_bernoulli",
                     "temperature_start": 4.0, "temperature_end": 1.0,
                     "warmup_episodes": 106,
                     "evaluation_temperature": 1.0},
            "networks": {"actor": "256-128-4",
                         "critic": "centralized twin 256-256"},
            "optimizer": {"name": "Adam", "lr": 3e-4, "tau": 0.005,
                          "policy_delay": 2, "batch_size": 128,
                          "buffer_capacity": 100000, "start_steps": 0},
            "sampler": {"uniform_fraction": 0.5, "beta": 0.5,
                        "max_upweight": 5.0},
            "exploration": {"epsilon_start": 0.20, "epsilon_end": 0.05,
                            "decay_episodes": 159,
                            "charge_scale": 0.5,
                            "high_fraction": 0.5},
            "target": {"policy_noise": 0.2, "noise_clip": 0.5},
            "episode": {"blocks": 24, "commit_periods": 4,
                        "lookahead_periods": 16},
            "budget": {"seeds": [11, 23, 37, 53, 71],
                       "minimum_rounds": 10,
                       "exposures_per_day_min": 10,
                       "exposures_per_day_max": 15,
                       "episodes_per_seed_min": 530,
                       "episodes_per_seed_max": 795,
                       "block_steps_per_seed_min": 12720,
                       "block_steps_per_seed_max": 19080,
                       "critic_updates_per_block": 1,
                       "actor_updates_every_blocks": 2},
            "validation": {
                "passes": "after each full pass over the 53 training days",
                "metric": "mean fleet delta G over the 11 validation days "
                          "(settlement, not reward)",
                "best_checkpoint": "highest validation mean fleet delta G",
                "resume_checkpoint": "every round",
                "final_checkpoint": "end of budget or early stop",
                "early_stop_patience": 3,
            },
            "boundaries": {
                "train_only_in_replay": True,
                "validation_not_in_replay": True,
                "test_evaluated": False,
                "holiday_evaluated": False,
                "h2_opened": False,
                "test_label": "LEGACY_SEEN_TEST",
                "independent_generalisation_claim": False,
            },
            "claims": {
                "allowed": "formal training on the NEAR80 controlled "
                           "simulation under the frozen protocol",
                "not_allowed": "independent generalisation or blind-test "
                               "result",
                "old_g10_status": "NO-GO remains for an independent "
                                  "generalisation claim",
            },
            "code": {
                "candidate": CANDIDATE,
                "source_sha256": code["source_sha256"],
                "git": code["git"],
            },
            "data": {
                "counts": data["counts"],
                "legacy_seen_test_pairs": data["legacy_seen_test_pairs"],
                "input_sha256": data["input_sha256"],
            },
        }
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify existing manifests instead of writing")
    parser.add_argument("--finalize", action="store_true",
                        help="write the GO/NO-GO status from "
                             "pretraining_validation_results.json")
    parser.add_argument("--out", default=str(OUT),
                        help="manifest directory (defaults to the original)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    data = data_payload()
    code = {
        "candidate": CANDIDATE,
        "source_sha256": {
            name: sha(ROOT / "src" / name) for name in SOURCE_FILES
        },
        "git": git_state(),
        "tests": {
            "test_da_quote_mapping.py": sha(
                ROOT / "tests/test_da_quote_mapping.py"),
            "test_train_da_gate_guards.py": sha(
                ROOT / "tests/test_train_da_gate_guards.py"),
            "audit_phase20_action_routing.py": sha(
                ROOT / "src/audit_phase20_action_routing.py"),
        },
    }
    config = config_payload(data, code)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "status": "PENDING_PRETRAINING_VALIDATION",
        "training_readiness_status": None,
        "scope": ("formal training on NEAR80/common_split; Test is "
                  "LEGACY_SEEN_TEST; no independent-generalisation claim"),
        "accepted_candidate": CANDIDATE,
        "data_manifest": "formal_training_data_manifest.json",
        "code_manifest": "formal_training_code_manifest.json",
        "config_file": "formal_training_config.yaml",
        "config_sha256": config_hash,
        "data_manifest_sha256": hashlib.sha256(
            json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest(),
        "code_manifest_sha256": hashlib.sha256(
            json.dumps(code, sort_keys=True).encode("utf-8")).hexdigest(),
        "checks": {
            "common_split_counts": data["counts"]
            == {"train": 53, "validation": 11, "test": 11,
                "holiday_block": 5},
            "all_disjoint": data["all_disjoint"],
            "legacy_label_matches": data["legacy_label_matches_declared"],
            "h2_sealed": not data["h2_opened"],
        },
    }
    manifest["checks"]["all_pass"] = all(manifest["checks"].values())
    out.mkdir(parents=True, exist_ok=True)
    if args.check:
        problems = []
        for name, expected in (
                ("formal_training_data_manifest.json", data),
                ("formal_training_code_manifest.json", code),
                ("formal_training_manifest.json", manifest)):
            path = out / name
            if not path.exists():
                problems.append(f"missing {path}")
                continue
            existing = json.loads(path.read_text(encoding="utf-8"))
            if name == "formal_training_manifest.json":
                for key in ("config_sha256", "data_manifest_sha256",
                            "code_manifest_sha256"):
                    if existing.get(key) != expected.get(key):
                        problems.append(f"{name}: {key} mismatch")
            elif existing != expected:
                problems.append(f"{name}: content mismatch")
        print(json.dumps({
            "ok": not problems, "problems": problems,
            "checks": manifest["checks"]}, indent=2))
        return 0 if not problems else 1
    if args.finalize:
        results_path = out / "pretraining_validation_results.json"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        results = json.loads(results_path.read_text(encoding="utf-8"))
        manifest_path = out / "formal_training_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = ("GO_FORMAL_TRAINING" if results.get("all_pass")
                  else "NO_GO")
        existing.update({
            "status": "finalized",
            "training_readiness_status": status,
            "pretraining_validation_results": str(results_path),
            "training_readiness": str(
                out / "training_readiness.md"),
            "finalized_on": dt.date.today().isoformat(),
        })
        manifest_path.write_text(
            json.dumps(existing, indent=2), encoding="utf-8")
        print(json.dumps({
            "training_readiness_status": status,
            "all_pass": results.get("all_pass"),
            "manifest": str(manifest_path),
        }, indent=2))
        return 0 if results.get("all_pass") else 1
    (out / "formal_training_data_manifest.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")
    (out / "formal_training_code_manifest.json").write_text(
        json.dumps(code, indent=2), encoding="utf-8")
    (out / "formal_training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "formal_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))
    print(f"wrote {out}")
    return 0 if manifest["checks"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
