"""Build the Phase 6 input manifest and the read-only checkpoint copies.

This is a one-shot provenance builder. It reads the frozen audit directory and
the formal run manifests; it never writes outside the Phase 6 request directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "phase6_frozen_policy_2026-11-12"
AUDIT = ROOT / "reports" / "algorithm_diagnostics" / "quote_map_saturation_2026-11-12"
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")

PROBE_IDS = [
    "2022-07-23|2025-07-13",
    "2022-08-14|2025-08-17",
    "2022-08-17|2025-08-19",
    "2022-10-13|2025-10-13",
    "2022-10-15|2025-10-12",
    "2022-10-17|2025-10-17",
    "2022-12-09|2025-12-09",
    "2022-12-10|2025-12-07",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def season(pair_id: str) -> str:
    month = int(pair_id[5:7])
    return {6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
            12: "winter"}.get(month, "other")


def pair_dict(pair_id: str) -> dict:
    load, price = pair_id.split("|")
    return {"pair_id": pair_id, "load_date": load,
            "price_date": price, "season": season(pair_id)}


def main() -> int:
    m11 = json.loads((ROOT / "runs/FORMAL_7834d0aa_seed11/run_manifest.json")
                     .read_text(encoding="utf-8"))
    m23 = json.loads((ROOT / "runs/FORMAL_7834d0aa_seed23/run_manifest.json")
                     .read_text(encoding="utf-8"))
    if m11["train_pairs"] != m23["train_pairs"]:
        raise SystemExit("train pairs differ between seeds")
    if m11["validation_pairs"] != m23["validation_pairs"]:
        raise SystemExit("validation pairs differ between seeds")
    train_set = set(m11["train_pairs"])
    for pid in PROBE_IDS:
        if pid not in train_set:
            raise SystemExit(f"probe pair not in Train: {pid}")

    ckpts = {
        "seed11_final": {
            "seed": 11, "round": 10, "kind": "final", "role": "primary",
            "original_dir": "runs/FORMAL_7834d0aa_seed11/checkpoints/step_00012720",
            "audit_copy_dir": "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/checkpoints/seed11_final",
            "step": 12720,
        },
        "seed11_best": {
            "seed": 11, "round": 1, "kind": "best",
            "role": "auxiliary_regression",
            "original_dir": "runs/FORMAL_7834d0aa_seed11_best/checkpoints/step_00001272",
            "audit_copy_dir": "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/checkpoints/seed11_best",
            "step": 1272,
        },
        "seed23_final": {
            "seed": 23, "round": 10, "kind": "final", "role": "primary",
            "original_dir": "runs/FORMAL_7834d0aa_seed23/checkpoints/step_00012720",
            "audit_copy_dir": "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/checkpoints/seed23_final",
            "step": 12720,
        },
        "seed23_best": {
            "seed": 23, "round": 1, "kind": "best",
            "role": "auxiliary_regression",
            "original_dir": "runs/FORMAL_7834d0aa_seed23_best/checkpoints/step_00001272",
            "audit_copy_dir": "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/checkpoints/seed23_best",
            "step": 1272,
        },
    }
    for key, meta in ckpts.items():
        dest = OUT / "checkpoints" / key / "actors"
        dest.mkdir(parents=True, exist_ok=True)
        src = ROOT / meta["audit_copy_dir"] / "actors"
        hashes = {}
        for name in NAMES:
            source = src / f"{name}.pt"
            target = dest / f"{name}.pt"
            shutil.copy2(source, target)
            hashes[name] = sha(target)
            if hashes[name] != sha(source):
                raise SystemExit(f"copy hash mismatch: {target}")
        original = {
            name: sha(ROOT / meta["original_dir"] / "actors" / f"{name}.pt")
            for name in NAMES
        }
        if original != hashes:
            raise SystemExit(f"original actor hash mismatch: {key}")
        counters = ROOT / meta["audit_copy_dir"] / "counters.json"
        if counters.exists():
            shutil.copy2(counters, OUT / "checkpoints" / key / "counters.json")
        meta["actor_sha256"] = hashes
        meta["original_actor_sha256"] = original
        meta["phase6_copy_dir"] = f"checkpoints/{key}"

    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    config = {
        "label": "NON_FORMAL_DIAGNOSTIC",
        "experiment": "phase6_frozen_policy_paired_market_clearing",
        "created_at": now,
        "purpose": (
            "Compare the admitted quote-map candidate Z_CORE=10.0 against the "
            "control Z_CORE=2.5 under frozen actor weights, identical market "
            "inputs, identical rolling 24-block dynamics, and the frozen "
            "terminal-SOC mechanism. No training, no optimizer step, no buffer "
            "write."),
        "scope": {
            "frozen_policy_counterfactual": True,
            "post_training_new_map_policy_performance": False,
            "statement": (
                "Actors were trained under linear_core_v1. This experiment "
                "changes only the inference map. It does not establish how a "
                "policy trained under Z_CORE=10.0 would perform."),
        },
        "control": {
            "z_core": 2.5,
            "candidate_id": "candidate_linear_core_z2p5",
            "identity": "reproduces linear_core_v1 exactly; verified in-process",
        },
        "candidate": {
            "z_core": 10.0,
            "candidate_id": "candidate_linear_core_z10p0",
            "admission_source": "reports/algorithm_diagnostics/quote_map_saturation_2026-11-12/decision_report.md",
            "admission_status": "ADMITTED_Z10.0",
        },
        "invariants": {
            "actor_weights_identical_across_maps": True,
            "market_dates_and_inputs_identical": True,
            "initial_soc": 0.5,
            "terminal_soc_target": 0.5,
            "terminal_soc_mechanism": (
                "day_end window only; endpoints pinned; no terminal "
                "inventory-value term in participant_payoff"),
            "terminal_inventory_value_cny": 0.0,
            "environment_and_clearing_config_identical": True,
            "physical_constraints_identical": True,
            "exploration": "disabled",
            "calibration_exposures": "disabled",
            "training_noise": "disabled",
            "optimizer_steps": 0,
            "buffer_writes": 0,
        },
        "rolling_day": {
            "blocks": 24,
            "commit_periods": 4,
            "lookahead_periods": 16,
            "action_regenerated_each_block": True,
            "soc_updated_from_committed_boundary": True,
            "fixed_whole_day_quote_sweep": False,
        },
        "checkpoints": ckpts,
        "phase6A": {
            "name": "smoke",
            "pairs": [pair_dict(p) for p in PROBE_IDS],
            "checkpoints": ["seed11_final", "seed23_final",
                            "seed11_best", "seed23_best"],
            "pair_source": (
                "PROBE_PAIRS frozen in src/diagnose_quote_map_saturation.py; "
                "recorded in audit input_hashes.json"),
            "deviation_from_request": (
                "The request named 6 Train representative days from Phase 2. "
                "No 6-day artifact exists on disk. The user approved the 8 "
                "frozen pre-registered probe pairs on 2026-11-12 "
                "(3 summer, 3 autumn, 2 winter)."),
            "gate_to_phase6B": (
                "All 64 day-runs must clear, be OPTIMAL, pass voltage/line/"
                "SOC/terminal checks, and have no fallback or infeasible result."),
        },
        "phase6B": {
            "name": "train",
            "pairs": [pair_dict(p) for p in m11["train_pairs"]],
            "checkpoints": ["seed11_final", "seed23_final"],
            "note": (
                "Round 10 finals are the primary Train experiment. Round 1 is "
                "not used for the Train decision."),
        },
        "phase6C": {
            "name": "validation",
            "pairs": [pair_dict(p) for p in m11["validation_pairs"]],
            "checkpoints": ["seed11_final", "seed23_final"],
            "note": "Round 10 finals are the primary Validation experiment.",
        },
        "excluded_splits": {
            "test": "never read",
            "holiday": "never read",
            "h2_holdout": "never read; h2_opened remains false",
        },
        "market_mechanism": {
            "network": "line50_q2.0",
            "clearing_engine": "socp",
            "cycle_cost_cny_per_mwh": 100.0,
            "observation": "v6_12d (12 dims)",
            "action": "storage_quote_v4 (2 dims per unit)",
            "ess_order": list(NAMES),
            "gate": "8 total, straight-through Bernoulli; evaluation hard threshold at 0",
            "gate_temperature_eval": 1.0,
            "quote_map_formal": "linear_core_v1",
            "market_config_hash": m11["market_config_hash"],
            "b1_mechanism_hash": m11["b1_mechanism_hash"],
        },
        "kpis": {
            "primary": (
                "fleet net settlement income per day (sum of the four ESS "
                "participant_payoff totals)"),
            "secondary": (
                "per-ESS net settlement income, components, energy, SOC, "
                "quotes, LMP, PCC, solver and physical checks"),
            "paired_difference": (
                "Z10.0 minus Z2.5 on the same checkpoint and the same date"),
            "uncertainty": (
                "date-clustered bootstrap 95% CI, 1000 resamples, seed 20261112"),
            "single_unit_rule": (
                "A per-ESS improvement is never interpreted as a fleet "
                "improvement; the fleet is the decision unit."),
        },
        "decision_rule": {
            "prerequisite": (
                "Phase 6A clearing/physical/solver/terminal checks must all "
                "PASS; otherwise no economic verdict is reported."),
            "PASS": (
                "For both Round 10 final checkpoints, on both Train and "
                "Validation, the date-clustered bootstrap 95% CI lower bound "
                "of the fleet payoff delta is > 0 and the per-seed median "
                "delta is > 0."),
            "PARTIAL": (
                "The point estimate is positive but at least one primary CI "
                "lower bound is <= 0, or Train and Validation disagree in "
                "sign; or the Round 1 auxiliary check reverses the sign."),
            "FAIL": (
                "The point estimate is <= 0 for both Round 10 final "
                "checkpoints on either split and the corresponding CI upper "
                "bound is < 0."),
            "NOT_TESTED": "Any split, checkpoint, or metric not run.",
            "BLOCKED": (
                "Inputs, provenance, or terminal settlement accounting cannot "
                "be confirmed."),
        },
        "deliverables": [
            "phase6_config.json", "source_hashes.json",
            "provenance_addendum.json", "phase6A_smoke_results.csv",
            "phase6B_train_results.csv", "phase6C_validation_results.csv",
            "daily_paired_deltas.csv", "per_ess_payoff.csv",
            "fleet_payoff.csv", "physical_validation.csv",
            "solver_validation.csv", "bootstrap_summary.json",
            "phase6_decision_report.md", "output_hashes.json",
        ],
    }
    (OUT / "phase6_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    addendum = {
        "written_at": now,
        "label": "NON_FORMAL_DIAGNOSTIC",
        "subject": "Provenance addendum for the Phase 0-5 quote-map saturation audit",
        "historical_files_modified": False,
        "script_hash_discrepancy": {
            "path": "src/diagnose_quote_map_saturation.py",
            "recorded_in_input_hashes": "187a46ceabf6adc1b09ed919759d736cc1e023b0b75f694f955073681a45c148",
            "actual_sha256": "12ae6f3259999a8de2130c2ea4b954a9df24d52178d56978b250b6230ccb743e",
            "input_hashes_written_at": "2026-11-12T01:00:09+0800",
            "script_mtime": datetime.fromtimestamp(
                (ROOT / "src/diagnose_quote_map_saturation.py").stat().st_mtime,
                timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "output_hashes_written_at": "2026-11-12T01:11:25+0800",
            "evidence": (
                "All content files listed in output_hashes.json hash-match the "
                "files on disk. The script mtime lies after input_hashes.json "
                "and before output_hashes.json."),
            "inference": (
                "The current script produced the final Phase 5 artifacts; the "
                "recorded input hash is stale, not the outputs. The candidate "
                "math is independently re-verified by quote_map_candidates."
                "run_tests in Phase 6."),
            "impact_on_phase6": (
                "None on Phase 6 inputs: Phase 6 reads the frozen probe-pair "
                "list and re-reads all market inputs directly. It does not "
                "execute the diagnostic script."),
        },
        "nested_self_hash_note": {
            "audit_manifest_json_actual_sha256": "53e716ec5850990c1f5af514fd9247b1d39b53c582e6d2abaef8668caff85882",
            "audit_manifest_records_itself_as": "2ec5251ce51be915b6b24b4b402ceb4c79ee33577dae6fbbf72ed8f12dbdb3b5",
            "output_hashes_json_actual_sha256": "3431ac83959945e556a374794ca48aa9cb0d7f5589862f3d7ea53083e80decf1",
            "audit_manifest_records_output_hashes_as": "24101f31e5423ca9803fe0b2a2a64386269de8cccafef3400128a049312d053a",
            "input_hashes_json_actual_sha256": "695edce9c4c6a6cd11302cb0f20672af130b03f33381f9976533db0a7ae2b6f9",
            "note": (
                "A manifest cannot contain its own final hash. These two "
                "entries are pre-final-write values; output_hashes.json "
                "entries for the content files are accurate."),
        },
        "checkpoint_provenance": {
            "result": (
                "All 16 actor files in the four audit copies hash-match their "
                "original run directories and output_hashes.json."),
            "actor_sha256": {k: v["actor_sha256"] for k, v in ckpts.items()},
        },
        "terminal_accounting": {
            "terminal_soc_target": 0.5,
            "initial_soc": 0.5,
            "terminal_inventory_value_term": False,
            "source": (
                "src/participant_payoff.py PayoffBreakdown docstring; "
                "data/processed/v2_cy_jn_2025/terminal_soc_config.json"),
            "status": "CONFIRMED",
        },
    }
    (OUT / "provenance_addendum.json").write_text(
        json.dumps(addendum, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {OUT / 'phase6_config.json'}")
    print(f"wrote {OUT / 'provenance_addendum.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
