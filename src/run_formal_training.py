"""Formal-training entry: verify the frozen manifests, then run one seed.

`--check` only verifies the data, code and config manifests and reports the
readiness status. `--seed` refuses to start unless the status is
`GO_FORMAL_TRAINING`; `--all` runs the frozen seeds sequentially. The
script never commits, never overwrites a run directory and never evaluates the
test or holiday blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import make_formal_training_manifests as mfm
import training_contract as tc

ROOT = Path.cwd()
# Protocol v6 is the live frozen set; v5, v4, v3, v2 and
# `formal_training_ready` are historical and must be requested explicitly with
# --config-dir.
OUT = ROOT / "reports/v2_cy_jn_2025/formal_training_v6"
RUNS = ROOT / "runs"
SEEDS = (11, 23)


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def verify(out: Path = OUT) -> dict:
    config = out / "formal_training_config.yaml"
    data_manifest = out / "formal_training_data_manifest.json"
    code_manifest = out / "formal_training_code_manifest.json"
    manifest_path = out / "formal_training_manifest.json"
    for path in (config, data_manifest, code_manifest, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    saved_data = json.loads(data_manifest.read_text(encoding="utf-8"))
    saved_code = json.loads(code_manifest.read_text(encoding="utf-8"))
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_data = mfm.data_payload()
    # The same payload the generator freezes, so the launcher cannot drift
    # from the manifest it is gating.
    current_code = mfm.code_payload()
    saved_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    contract_violations = tc.validate_contract(saved_config, saved_data)
    checks = {
        "data_payload_matches": saved_data == current_data,
        "code_sources_match": saved_code.get("source_sha256")
        == current_code["source_sha256"],
        "code_tests_match": saved_code.get("tests") == current_code["tests"],
        "config_hash_matches": saved_manifest.get("config_sha256")
        == config_hash(saved_config),
        "data_manifest_hash_matches": saved_manifest.get(
            "data_manifest_sha256")
        == hashlib.sha256(json.dumps(
            saved_data, sort_keys=True).encode("utf-8")).hexdigest(),
        "code_manifest_hash_matches": saved_manifest.get(
            "code_manifest_sha256")
        == hashlib.sha256(json.dumps(
            saved_code, sort_keys=True).encode("utf-8")).hexdigest(),
        "common_split_counts": saved_data.get("counts")
        == {"train": 53, "validation": 11, "test": 11, "holiday_block": 5},
        "all_disjoint": saved_data.get("all_disjoint") is True,
        "legacy_label_matches": saved_data.get(
            "legacy_label_matches_declared") is True,
        "h2_sealed": saved_data.get("h2_opened") is False,
        "budget_contract_ok": not contract_violations,
    }
    checks["all_pass"] = all(checks.values())
    declared = saved_manifest.get("training_readiness_status")
    return {
        "checks": checks,
        # The manifest records the readiness declared at freeze time; a
        # directory that no longer verifies must not echo it back as GO.
        "status": declared if checks["all_pass"] else "NO_GO",
        "declared_status": declared,
        "accepted_candidate": saved_manifest.get("accepted_candidate"),
        "config_sha256": saved_manifest.get("config_sha256"),
        "contract_violations": contract_violations,
    }


def run_id(seed: int, out: Path = OUT) -> str:
    manifest = json.loads(
        (out / "formal_training_manifest.json").read_text(encoding="utf-8"))
    return f"FORMAL_{manifest['config_sha256'][:8]}_seed{seed}"


def command(seed: int, config: dict, out: Path = OUT) -> list[str]:
    formal = config["formal_training"]
    budget = formal["budget"]
    return [
        sys.executable, "src/train_da_gate.py",
        "--split", "common", "--seed", str(seed),
        "--exposures", str(budget["exposures_per_day_max"]),
        "--early-stop-patience", str(formal["validation"]["early_stop_patience"]),
        "--min-rounds", str(budget["minimum_rounds"]),
        "--max-rounds", str(budget["exposures_per_day_max"]),
        "--max-episodes", str(budget["episodes_per_seed_max"]),
        "--run-id", run_id(seed, out), "--trace-blocks",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-dir", default=str(OUT))
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing run directory instead of "
                             "refusing it (TD-11)")
    args = parser.parse_args(argv)
    out = Path(args.config_dir)
    state = verify(out)
    if args.check or (not args.seed and not args.all):
        print(json.dumps(state, indent=2))
        return 0 if state["checks"]["all_pass"] else 1
    if not state["checks"]["all_pass"]:
        print(json.dumps(state, indent=2))
        return 1
    if state["status"] != "GO_FORMAL_TRAINING":
        print(json.dumps({
            "refused": "formal training is not authorised",
            "status": state["status"],
            "checks": state["checks"],
        }, indent=2))
        return 2
    seeds = SEEDS if args.all else (args.seed,)
    config = yaml.safe_load(
        (out / "formal_training_config.yaml").read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    for seed in seeds:
        run = RUNS / run_id(seed, out)
        existed = run.exists() and any(run.iterdir())
        if existed and not args.resume:
            print(f"refused: {run} already exists", file=sys.stderr)
            return 3
        cmd = command(seed, config, out)
        if existed:
            cmd = cmd + ["--resume"]
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"seed {seed} failed with {result.returncode}",
                  file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
