"""Assemble the v7 pretraining validation record and its evidence hashes.

Reads the artifacts the v7 acceptance evidence already names and writes
`pretraining_validation_results.json` beside the v7 manifests, where
`make_formal_training_manifests.py --finalize` then reads its `all_pass`. It
does not train, re-clear a day or modify a run.

`all_pass` is computed from the booleans below. The one substantive judgement
it encodes is the declared exception: a day whose price label qualifies but
whose settlement audit does not is a failure unless it is named here and
carries the measurement that explains it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

V7 = Path("reports/v2_cy_jn_2025/formal_training_v7")
REPORT = Path("reports/local_da_v7_2026-11-18")
QUALIFICATION = Path("reports/local_da_v7_2026-11-20/qualification_cone_tol")
SMOKE_RUN = Path("runs/SMOKE_local_da_v7_bundle")
SMOKE_ROUND = Path("snapshots/SMOKE_local_da_v7_bundle/round_001.json")
QUALIFIED_LABELS = ("SMOOTH_PASS", "SUBGRADIENT_PASS")
DECLARED_EXCEPTIONS = {
    "2022-10-30": "exact-model cone residual 1.81e-5 against a 1e-5 bound; its "
                  "dispatch audit reads 4.29e-8, every other item on both "
                  "models 1e-10, ac_replay 3.3e-11 pu. In the training block; "
                  "rl_da_env.block_payoffs settles the window anyway and does "
                  "not gate reward on qualification.",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualification() -> dict:
    rows = {}
    for path in sorted(QUALIFICATION.glob("library_*.json")):
        for row in load(path)["sample"]:
            rows[row["date"]] = row
    labels = Counter(row["price_qualification"] for row in rows.values())
    labelled = {date for date, row in rows.items()
                if row["price_qualification"] in QUALIFIED_LABELS}
    settled = {date for date, row in rows.items()
               if row["qualified_for_settlement"]}
    exceptions = labelled - settled
    return {
        "days_measured": len(rows),
        "all_days_measured": len(rows) == 136,
        "smooth_pass": labels["SMOOTH_PASS"],
        "subgradient_pass": labels["SUBGRADIENT_PASS"],
        "nonsmooth_or_failed": labels["NONSMOOTH_OR_FAILED"],
        "label_qualified": len(labelled),
        "settle_qualified": len(settled),
        "settle_qualified_without_label": sorted(settled - labelled),
        "declared_exceptions": sorted(exceptions & DECLARED_EXCEPTIONS.keys()),
        "no_undeclared_exception": not (exceptions - DECLARED_EXCEPTIONS.keys()),
    }


def acceptance(summary: dict, manifest: dict) -> dict:
    counts = summary["acceptance_8"]["counts"]
    return {
        "block_vs_day_price_verdict": summary["acceptance_4"]["verdict"],
        "block_vs_day_price_pass":
            summary["acceptance_4"]["verdict"] == "PASS",
        "block_vs_day_price_criterion": summary["acceptance_4"]["criterion"],
        "deviation_weight_verdict": summary["acceptance_6"]["verdict"],
        "deviation_weight_pass": summary["acceptance_6"]["verdict"] == "PASS",
        "deviation_weight_gradient_share":
            summary["acceptance_6"]["gradient_share"],
        "deviation_weight_criterion": summary["acceptance_6"]["criterion"],
        "smoke_flags_verdict": summary["acceptance_7"]["verdict"],
        "smoke_flags_pass": summary["acceptance_7"]["verdict"] == "PASS",
        "block_vs_day_gap": "recorded, no criterion",
        "feasibility_counts": counts,
        "feasibility_matches_manifest":
            {key: counts[key] == manifest["counts"][key]
             for key in ("train", "validation", "test", "holiday")},
        "feasibility_no_day_dropped": not summary["acceptance_8"]["dropped"],
    }


def launcher(bundle: dict) -> dict:
    manifest = bundle.get("manifest") or {}
    best = (SMOKE_RUN / "best" / "manifest.json")
    checks = {
        "eval_history_written": (SMOKE_RUN / "eval_history.json").exists(),
        "round_snapshot_written": SMOKE_ROUND.exists(),
        "best_checkpoint_written": best.exists(),
        "final_bundle_written": (SMOKE_RUN / "final" / "manifest.json").exists(),
        "bundle_reloaded": bool((manifest.get("data") or {}).get("pairing")),
        "bundle_level": manifest.get("level") == "bundle",
    }
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", action="store_true",
                        help="the full default test suite passed apart from "
                             "the named pre-existing failure")
    parser.add_argument("--tests-summary", default="")
    args = parser.parse_args(argv)

    from run_formal_training import verify
    from checkpoint_loader_v2 import load_checkpoint

    manifest = load(V7 / "formal_training_manifest.json")
    data_manifest = load(V7 / "formal_training_data_manifest.json")
    summary = load(REPORT / "audit" / "summary.json")
    smoke = load(REPORT / "smoke.json")
    bundle = load_checkpoint(SMOKE_RUN / "final")

    gate = verify(V7)["checks"]
    checks = {
        # The gate already carries the data checks: payload and manifest
        # hashes, split counts, disjointness, the legacy label and the H2
        # declaration. Re-deriving them here would be a second opinion on the
        # same file, not a second check.
        "gate": dict(gate),
        "qualification": qualification(),
        "acceptance": acceptance(summary, data_manifest),
        "launcher": launcher(bundle),
        "tests": {
            "full_default_suite": bool(args.tests_passed),
            "summary": args.tests_summary or "not recorded",
        },
    }
    booleans = [value for group in checks.values() for value in group.values()
                if isinstance(value, bool)]
    booleans += [value for group in checks.values() for value in group.values()
                 if isinstance(value, dict)
                 for value in value.values() if isinstance(value, bool)]
    all_pass = bool(booleans) and all(booleans)

    evidence = [V7 / "formal_training_manifest.json",
                V7 / "formal_training_data_manifest.json",
                REPORT / "smoke.json",
                REPORT / "audit" / "summary.json",
                REPORT / "RESULTS.md",
                *sorted(QUALIFICATION.glob("library_*.json")),
                SMOKE_ROUND]
    payload = {
        "all_pass": all_pass,
        "training_readiness_status": (
            "GO_FORMAL_TRAINING" if all_pass else "NO_GO"),
        "scope": (
            "formal training under protocol_version 7 on the 90-day training "
            "block of the frozen library; the Test and holiday blocks are "
            "legacy-seen and never enter training or model selection; the H2 "
            "price-date holdout stays sealed; no independent-generalisation "
            "claim"),
        "protocol_version": 7,
        "candidate": "V7_candidate_004",
        "config_sha256": manifest.get("config_sha256"),
        "checks": checks,
        "declared_exceptions": DECLARED_EXCEPTIONS,
        "residual_risks": [
            "The load is 2022 metered and the price is a 2025 Jiangnan curve: "
            "a cross-year controlled simulation, not same-period settlement.",
            "The cone bound is 1e-5, looser than the 1e-6 it replaced. It is "
            "justified by the nonlinear replay passing to 3e-11 pu on every "
            "sampled day, and it is a measured limit rather than a proof that "
            "no claimed clearance is a slack relaxation.",
            "The launcher's reduced-round path is exercised on two training "
            "days; `run_formal_training.py --seed` itself has not started a "
            "formal run, because the status gate is what this record sets.",
            "The full suite has one pre-existing failure that touches no file "
            "this protocol pins, and one failure that is the interpreter's "
            "matplotlib rather than the tree.",
            "`wall_clock` is a projection from one measured day, not a "
            "bounded cost.",
        ],
        "evidence": {str(path): sha(path) for path in evidence
                     if path.exists()},
        "manifest": str(V7 / "formal_training_manifest.json"),
        "config": str(V7 / "formal_training_config.yaml"),
        "notes": {
            "split_counts": data_manifest["counts"],
            "h2_opened": data_manifest.get("h2_opened"),
            "h2_use": data_manifest.get("h2_use"),
            "smoke_days": smoke["days"],
            "dispatch_status": summary["dispatch_status"]},
    }
    out = V7 / "pretraining_validation_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [key for group, values in checks.items() for key, value in
              values.items() if value is False]
    print(json.dumps({"all_pass": all_pass,
                      "status": payload["training_readiness_status"],
                      "failed": failed}, indent=2))
    print(f"wrote {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
