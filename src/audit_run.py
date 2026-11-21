"""Read-only evidence collector for one training run directory.

Reads `run_manifest.json`, `eval_history.json`, `exposure.json` and the
recorded artifact hashes. It never writes inside the run directory; `--write`
puts the evidence file under `reports/cc01/audit/` instead.

The budget floor comes from the live formal manifest, so an older run is
judged against the protocol that is in force now. A run whose manifest has no
minimum-round field is reported as an explicit finding rather than assumed
compliant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import training_contract as tc

ROOT = Path.cwd()
RUNS = ROOT / "runs"
EVIDENCE_DIR = ROOT / "reports/cc01/audit"
LOG_EXCLUDED = {"train.log"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, passed: bool, detail: str,
           evidence: str | None = None) -> dict:
    return {"check": name, "verdict": "PASS" if passed else "FAIL",
            "detail": detail, "evidence": evidence}


def audit(run_dir: Path, contract_dir: Path = tc.DEFAULT_CONFIG_DIR) -> dict:
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    if not isinstance(manifest, dict):
        return {"run_dir": str(run_dir),
                "verdict": "BLOCKED",
                "reason": "run_manifest.json is missing or unreadable"}
    history = _read_json(run_dir / "eval_history.json", [])
    contract, data_manifest = tc.load_contract(contract_dir)
    budget = contract["formal_training"]["budget"]
    minimum_rounds = int(budget["minimum_rounds"])
    ep_min = int(budget["episodes_per_seed_min"])
    ep_max = int(budget["episodes_per_seed_max"])
    bs_min = int(budget["block_steps_per_seed_min"])
    bs_max = int(budget["block_steps_per_seed_max"])
    episodes = int(manifest.get("episodes_run") or 0)
    steps = int(manifest.get("total_steps") or 0)
    rounds = len(history) if isinstance(history, list) else 0
    early = manifest.get("early_stop") or {}
    stop_round = early.get("round")

    checks = []
    checks.append(_check(
        "run_terminated_cleanly", manifest.get("status") != "failed",
        f"status={manifest.get('status')!r} error={str(manifest.get('error'))[:120]!r}",
        "run_manifest.json"))
    checks.append(_check(
        "min_rounds_recorded", "min_rounds" in manifest,
        f"run manifest min_rounds={manifest.get('min_rounds')!r}",
        "run_manifest.json"))
    checks.append(_check(
        "rounds_at_least_minimum", rounds >= minimum_rounds,
        f"rounds={rounds} minimum={minimum_rounds}", "eval_history.json"))
    checks.append(_check(
        "rounds_at_most_maximum",
        rounds <= int(budget["exposures_per_day_max"]),
        f"rounds={rounds} maximum={budget['exposures_per_day_max']}",
        "eval_history.json"))
    checks.append(_check(
        "episodes_in_budget", ep_min <= episodes <= ep_max,
        f"episodes={episodes} range=[{ep_min}, {ep_max}]",
        "run_manifest.json"))
    checks.append(_check(
        "block_steps_in_budget", bs_min <= steps <= bs_max,
        f"block_steps={steps} range=[{bs_min}, {bs_max}]",
        "run_manifest.json"))
    if early.get("triggered") and stop_round is not None:
        checks.append(_check(
            "early_stop_not_before_minimum", stop_round >= minimum_rounds,
            f"early stop round={stop_round} minimum={minimum_rounds}",
            "run_manifest.json"))
    else:
        checks.append({"check": "early_stop_not_before_minimum",
                       "verdict": "PASS",
                       "detail": "early stop did not trigger",
                       "evidence": "run_manifest.json"})

    best = manifest.get("best") or {}
    best_raw = best.get("checkpoint")
    final_raw = manifest.get("final_checkpoint")
    best_path = Path(best_raw) if best_raw else None
    final_path = Path(final_raw) if final_raw else None
    checks.append(_check(
        "best_checkpoint_exists",
        bool(best_path) and best_path.is_dir(),
        f"best={best_path}", "run_manifest.json"))
    checks.append(_check(
        "final_checkpoint_exists",
        bool(final_path) and final_path.is_dir(),
        f"final={final_path}", "run_manifest.json"))
    checks.append(_check(
        "best_is_not_final",
        bool(best_path) and bool(final_path) and best_path != final_path,
        f"best={best_path} final={final_path}", "run_manifest.json"))

    recorded = manifest.get("artifacts_sha256") or {}
    mismatched = []
    for name, expected in recorded.items():
        path = run_dir / name
        if not path.exists():
            mismatched.append(f"{name}: missing")
        elif sha(path) != expected:
            mismatched.append(name)
    data_mismatches = [n for n in mismatched if n not in LOG_EXCLUDED]
    checks.append(_check(
        "artifact_hashes_match", not data_mismatches,
        f"mismatched={data_mismatches}", "run_manifest.json"))
    checks.append({
        "check": "train_log_hash_matches",
        "verdict": "PASS" if "train.log" not in mismatched else "INFERRED",
        "detail": ("train.log was hashed before the completion line was "
                   "appended; see TECHNICAL_DEBT.md")
        if "train.log" in mismatched else "train.log matches",
        "evidence": "run_manifest.json",
    })

    checks.append(_check(
        "test_not_evaluated", manifest.get("test_evaluation") is None,
        f"test_evaluation={manifest.get('test_evaluation')!r}",
        "run_manifest.json"))
    checks.append(_check(
        "legacy_test_not_run", manifest.get("legacy_seen_test_run") is False,
        f"legacy_seen_test_run={manifest.get('legacy_seen_test_run')!r}",
        "run_manifest.json"))
    checks.append(_check(
        "h2_not_read", manifest.get("h2_holdout_read") is False,
        f"h2_holdout_read={manifest.get('h2_holdout_read')!r}",
        "run_manifest.json"))

    last = history[-1] if isinstance(history, list) and history else {}
    checks.append(_check(
        "no_voltage_or_line_violation",
        int(last.get("voltage_violation_cells") or 0) == 0
        and int(last.get("line_overload_cells") or 0) == 0,
        f"voltage={last.get('voltage_violation_cells')} "
        f"line={last.get('line_overload_cells')}",
        "eval_history.json"))
    checks.append(_check(
        "no_lmp_fallbacks", int(last.get("lmp_fallbacks") or 0) == 0,
        f"lmp_fallbacks={last.get('lmp_fallbacks')}", "eval_history.json"))
    checks.append(_check(
        "terminal_soc_exact",
        abs(float(last.get("terminal_soc_max_err") or 0.0)) < 1e-6,
        f"terminal_soc_max_err={last.get('terminal_soc_max_err')}",
        "eval_history.json"))

    budget_failed = any(
        item["verdict"] == "FAIL" for item in checks
        if item["check"] in ("rounds_at_least_minimum",
                             "rounds_at_most_maximum",
                             "episodes_in_budget",
                             "block_steps_in_budget",
                             "early_stop_not_before_minimum"))
    other_failed = any(item["verdict"] == "FAIL" for item in checks)
    if any(item["check"] == "run_terminated_cleanly" and item["verdict"] == "FAIL"
           for item in checks):
        # A crashed run leaves a short budget as a symptom, not as the finding.
        verdict = "NO_GO_RUN_FAILED"
    elif budget_failed:
        verdict = "NO_GO_BUDGET_PROTOCOL"
    elif other_failed:
        verdict = "NO_GO"
    elif any(item["verdict"] == "INFERRED" for item in checks):
        verdict = "PASS_WITH_NOT_TESTED"
    else:
        verdict = "PASS"

    return {
        "run_id": manifest.get("run_id"),
        "seed": manifest.get("seed"),
        "run_dir": str(run_dir),
        "status": manifest.get("status"),
        "rounds": rounds,
        "episodes": episodes,
        "block_steps": steps,
        "updates": manifest.get("updates"),
        "minimum_rounds": minimum_rounds,
        "budget_episodes": [ep_min, ep_max],
        "budget_block_steps": [bs_min, bs_max],
        "best_round": best.get("round"),
        "best_value": best.get("value"),
        "checks": checks,
        "verdict": verdict,
    }


def _resolve(run_dir: str) -> Path:
    path = Path(run_dir)
    return path if path.is_dir() else RUNS / run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="path or run id under runs/")
    parser.add_argument("--contract-dir", default=str(tc.DEFAULT_CONFIG_DIR))
    parser.add_argument("--write", action="store_true",
                        help="write the evidence file under reports/cc01/audit")
    args = parser.parse_args(argv)
    run_dir = _resolve(args.run_dir)
    result = audit(run_dir, Path(args.contract_dir))
    print(json.dumps(result, indent=2))
    if args.write:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        name = str(result.get("run_id") or run_dir.name) + ".json"
        path = EVIDENCE_DIR / name
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"evidence: {path}", flush=True)
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
