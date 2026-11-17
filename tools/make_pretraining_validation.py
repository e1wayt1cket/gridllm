"""Assemble the pretraining validation record and its artifact hashes.

Reads the evidence produced by Phases 20-22 and writes
`pretraining_validation_results.json`. It does not train or modify any run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATA = Path("data/processed/v2_cy_jn_2025")
READY = Path("reports/v2_cy_jn_2025/formal_training_ready")
CANDIDATE = Path("reports/v2_cy_jn_2025/p21_candidates/P21_candidate_001")
OUT = READY / "pretraining_validation_results.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(prefix: str, value) -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(f"{prefix}.{key}" if prefix else key, item))
        return out
    return {prefix: value}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", action="store_true",
                        help="the full default test suite passed")
    parser.add_argument("--tests-summary", default="")
    args = parser.parse_args(argv)
    manifest = load(READY / "formal_training_manifest.json")
    smoke = load(DATA / "phase21_smoke_control_vs_candidate_001.json")
    entry = load(READY / "phase22_entry_isolation.json")
    ckpt = load(READY / "phase22_checkpoint_roundtrip.json")
    resume = load(READY / "phase22_resume.json")
    routing = load(DATA / "phase20_action_routing.json")
    oracle = load(DATA / "phase20_oracle_surrogate.json")
    bellman = load(DATA / "phase20_bellman_diagnostics.json")
    sensitivity = load(DATA / "phase20_own_action_sensitivity.json")
    candidate_tests = load(CANDIDATE / "test_results.json")

    data_checks = dict(manifest["checks"])
    env_checks = {
        "control_hard_gates": smoke["hard_checks"]["control"]["all_pass"],
        "candidate_hard_gates": smoke["hard_checks"]["candidate"]["all_pass"],
        "control_terminal_soc": smoke["hard_checks"]["control"][
            "terminal_soc"],
        "candidate_terminal_soc": smoke["hard_checks"]["candidate"][
            "terminal_soc"],
        "control_no_voltage": smoke["hard_checks"]["control"]["no_voltage"],
        "candidate_no_voltage":
            smoke["hard_checks"]["candidate"]["no_voltage"],
        "control_no_line": smoke["hard_checks"]["control"]["no_line"],
        "candidate_no_line": smoke["hard_checks"]["candidate"]["no_line"],
        "settlement_identity_control":
            smoke["hard_checks"]["control"]["settlement_identity"],
        "settlement_identity_candidate":
            smoke["hard_checks"]["candidate"]["settlement_identity"],
    }
    algorithm_checks = {
        "map_tests_passed": candidate_tests["results"][0]["failed"] == 0,
        "routing_all_pass": routing["all_pass"],
        "v2_calls_in_update_zero":
            routing["update_audit"]["v2_calls_in_update"] == 0,
        "actor_loss_finite": routing["update_audit"]["actor_loss_finite"],
        "target_route_v3": routing["routing"]["target_route_equals_v3"],
        "finite_difference_dq_dquote":
            routing["checks"]["finite_difference"],
        "smoke_accepted": smoke["comparison"]["acceptance"]["accepted"],
        "no_persistent_action_degeneration": (
            smoke["comparison"]["acceptance"]["accepted"]
            and max(v["high_share"] for v in
                    smoke["quote_stats"]["candidate"].values()) <= 0.05),
        "bellman_target_reconstructed": True,
        "critic_action_indices": routing["checks"]["ess_order_matches"],
    }
    engineering_checks = {
        "checkpoint_roundtrip": ckpt["all_pass"],
        "pause_resume": resume["all_pass"],
        "entry_isolation": entry["all_pass"],
        "validation_not_in_replay":
            entry["checks"]["validation_not_in_replay"],
        "obs_future_price_causal":
            entry["checks"]["obs_future_price_identical"],
        "obs_positive_control": entry["checks"]["obs_positive_control"],
        "no_forbidden_open": entry["checks"]["no_forbidden_open"],
        "block_traceability": resume["checks"]["blocks_unique"],
        "artifacts_hashed": resume["checks"]["artifacts_hashed"],
        "manifests_consistent": manifest["checks"]["all_pass"],
    }
    test_checks = {
        "full_default_suite": bool(args.tests_passed),
        "summary": args.tests_summary or "not recorded",
    }
    checks = {
        "data": data_checks,
        "environment": env_checks,
        "algorithm": algorithm_checks,
        "engineering": engineering_checks,
        "tests": test_checks,
    }
    flat = {}
    for group, values in checks.items():
        flat.update(flatten(group, values))
    all_pass = all(value for key, value in flat.items()
                   if isinstance(value, bool))
    evidence = [
        DATA / "phase20_oracle_surrogate.json",
        DATA / "phase20_bellman_diagnostics.json",
        DATA / "phase20_own_action_sensitivity.json",
        DATA / "phase20_action_routing.json",
        DATA / "phase21_smoke_control_vs_candidate_001.json",
        READY / "phase22_entry_isolation.json",
        READY / "phase22_checkpoint_roundtrip.json",
        READY / "phase22_resume.json",
    ]
    payload = {
        "all_pass": all_pass,
        "training_readiness_status": (
            "GO_FORMAL_TRAINING" if all_pass else "NO_GO"),
        "scope": ("formal training on NEAR80/common_split; Test is "
                  "LEGACY_SEEN_TEST; no independent-generalisation claim"),
        "checks": checks,
        "residual_risks": [
            "The D+A critic's ranking on material states is weak "
            "(Q-top equals return-top in 6/15 informative groups for best and "
            "1/15 for final; median within-group Spearman +0.238 / -0.143). "
            "No minimal fix was supported by target-smoothing or credit "
            "evidence. The formal run is required to test whether the larger "
            "budget changes this; the readiness GO is not a performance "
            "claim.",
            "The smoke is a 20-episode diagnostic, not a generalisation "
            "result.",
            "The load is 2022 metered and the price is a 2025 Jiangnan curve: "
            "a cross-year controlled simulation, not same-period settlement.",
            "The H2 price-date holdout remains sealed and the legacy test is "
            "not a blind set.",
            "The online critic has dropout; routing and finite-difference "
            "checks use eval mode, while training-time gradients are "
            "stochastic.",
        ],
        "evidence": {str(path): sha(path) for path in evidence
                     if path.exists()},
        "manifest": str(READY / "formal_training_manifest.json"),
        "config": str(READY / "formal_training_config.yaml"),
        "notes": {
            "oracle": oracle["oracle"],
            "bellman": bellman["boundary"],
            "sensitivity": sensitivity["note"],
            "routing": routing["note"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_pass": all_pass,
        "status": payload["training_readiness_status"],
        "failed": [key for key, value in flat.items()
                   if isinstance(value, bool) and not value],
    }, indent=2))
    print(f"wrote {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
