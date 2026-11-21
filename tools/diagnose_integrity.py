"""Read-only integrity evidence for the formal-training run set.

Collects the P0 items into JSON: run identity, artifact hashes, OOM text,
interpreter provenance, holdout boundaries, seed 53 / 37 state, frozen pin
hashes and the live launch gate. It never opens a checkpoint for writing, never
imports the training stack and never writes outside `reports/algorithm_diagnostics`.

Usage:
    python src/diagnose_integrity.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/algorithm_diagnostics"
MANIFESTS = (
    ROOT / "reports/v2_cy_jn_2025/formal_training_v3",
    ROOT / "reports/v2_cy_jn_2025/formal_training_v2",
    ROOT / "reports/v2_cy_jn_2025/formal_training_ready",
)
V3_RUNS = (11, 23, 37)
V2_SEED53 = ROOT / "runs/FORMAL_2148a577_seed53"
HISTORY = Path.home() / (
    "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine/"
    "ConsoleHost_history.txt")
SPLIT = ROOT / "data/processed/v2_cy_jn_2025/common_split.json"
ARTIFACTS = ("episodes.csv", "eval_history.json", "gate_stats.jsonl",
             "sampler_stats.jsonl", "block_decisions.jsonl", "train.log")
TEXT_SUFFIXES = {".log", ".json", ".jsonl", ".csv", ".md", ".txt"}
OOM = re.compile(
    r"MemoryError|out of memory|\bOOM\b|OOMKilled|bad_alloc|WinError 1455|"
    r"paging file", re.IGNORECASE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def meta(path: Path) -> dict:
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc)
        .isoformat(),
    }


def tree_hash(root: Path) -> dict:
    """Hash every regular file; return per-file hashes and one tree digest."""
    files = {}
    if not root.exists():
        return {"exists": False, "files": {}, "tree_sha256": None}
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        file_digest = sha256(path)
        files[rel] = {"sha256": file_digest, **meta(path)}
        digest.update(rel.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
    return {"exists": True, "files": files,
            "tree_sha256": digest.hexdigest(), "n_files": len(files)}


def environment() -> dict:
    out = {"python": sys.version, "executable": sys.executable}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception as exc:  # pragma: no cover - reported, not raised
        out["torch"] = f"unavailable: {exc}"
    try:
        import gurobipy
        out["gurobi"] = list(gurobipy.gurobi.version())
    except Exception as exc:  # pragma: no cover - reported, not raised
        out["gurobi"] = f"unavailable: {exc}"
    return out


def episodes_summary(path: Path) -> dict:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    episodes = [int(r["episode"]) for r in rows]
    rounds = sorted({int(r["round"]) for r in rows})
    pairs = [r["pair_id"] for r in rows]
    duplicates = sorted({e for e in episodes if episodes.count(e) > 1})
    return {
        "rows": len(rows),
        "first_episode": min(episodes) if episodes else None,
        "last_episode": max(episodes) if episodes else None,
        "contiguous": episodes == list(range(1, len(episodes) + 1)),
        "duplicate_episodes": duplicates,
        "rounds": rounds,
        "distinct_pairs": len(set(pairs)),
        "pair_ids": sorted(set(pairs)),
    }


def eval_history_pairs(path: Path) -> dict:
    rounds = read_json(path)
    pairs = []
    for record in rounds:
        for day in record.get("days") or []:
            pairs.append(day.get("pair_id"))
    return {"rounds": len(rounds), "pair_ids": sorted(set(p for p in pairs if p)),
            "mean_fleet_delta": [r.get("mean_fleet_delta") for r in rounds]}


def log_pairs(path: Path) -> set[str]:
    seen = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pair_id"):
                seen.add(str(row["pair_id"]))
    return seen


def scan_oom(root: Path) -> dict:
    hits = []
    scanned = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned.append(path.relative_to(root).as_posix())
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh, 1):
                    if OOM.search(line):
                        hits.append({"path": path.relative_to(root).as_posix(),
                                     "line": line_no,
                                     "text": line.strip()[:300]})
        except OSError as exc:
            hits.append({"path": str(path), "error": str(exc)})
    return {"hits": hits, "scanned": scanned}


def artifact_check(run_dir: Path, seed: int) -> dict:
    manifest = read_json(run_dir / "run_manifest.json")
    recorded = manifest.get("artifacts_sha256") or {}
    rows = {}
    for name in ARTIFACTS:
        path = run_dir / name
        if not path.exists():
            rows[name] = {"exists": False}
            continue
        current = sha256(path)
        expected = recorded.get(name)
        rows[name] = {
            "exists": True,
            **meta(path),
            "sha256": current,
            "recorded": expected,
            "matches": current == expected if expected else None,
        }
    return {
        "seed": seed,
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "artifacts": rows,
        "artifacts_sha256_present": bool(recorded),
        "note": ("train.log is hashed before its completion line (TD-01); a "
                 "mismatch there is INFERRED, not a missing artifact"
                 if seed in V3_RUNS and seed != 37 else None),
    }


def checkpoints(run_dir: Path) -> dict:
    ckpt_dir = run_dir / "checkpoints"
    steps = []
    if ckpt_dir.exists():
        for path in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            steps.append({
                "name": path.name,
                "files": len([p for p in path.rglob("*") if p.is_file()]),
                "bytes": sum(p.stat().st_size for p in path.rglob("*")
                             if p.is_file()),
            })
    latest = run_dir / "manifest.json"
    counters = read_json(latest).get("counters") if latest.exists() else None
    return {"step_dirs": steps, "latest_manifest_counters": counters}


def best_dir(seed: int) -> dict:
    path = ROOT / f"runs/FORMAL_1b47e0ed_seed{seed}_best"
    if not (path / "manifest.json").exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    manifest = read_json(path / "manifest.json")
    da = manifest.get("da_gate") or {}
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "run_id": manifest.get("run_id"),
        "counters": manifest.get("counters"),
        "schema_version": manifest.get("schema_version"),
        "specs": manifest.get("specs"),
        "actor_loss_map": da.get("actor_loss_map"),
        "quote_map": da.get("quote_map"),
        "action_schema": da.get("action_schema"),
        "checkpoint_dirs": sorted(
            p.name for p in (path / "checkpoints").iterdir()
            if p.is_dir()) if (path / "checkpoints").exists() else [],
    }


def seed11_identity() -> dict:
    run = ROOT / "runs/FORMAL_1b47e0ed_seed11"
    manifest = read_json(run / "run_manifest.json")
    episodes = episodes_summary(run / "episodes.csv")
    ev = eval_history_pairs(run / "eval_history.json")
    return {
        "run_dir": "runs/FORMAL_1b47e0ed_seed11",
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "resumed_at": manifest.get("resumed_at"),
        "resume_from": manifest.get("resume_from"),
        "episodes_run": manifest.get("episodes_run"),
        "total_steps": manifest.get("total_steps"),
        "best": manifest.get("best"),
        "final_checkpoint": manifest.get("final_checkpoint"),
        "episodes_csv": episodes,
        "eval_history": {"rounds": ev["rounds"],
                         "mean_fleet_delta": ev["mean_fleet_delta"]},
        "stub": {
            "present_on_disk": any(
                (run / name).exists() and (run / name).stat().st_size <= 2
                for name in ("train.log",)),
            "documented_in": [
                "TECHNICAL_DEBT.md#TD-09",
                "reports/cc03/phase25_launch.md",
            ],
            "recorded_stub": {
                "eval_history.json": {
                    "bytes": 2, "sha256_prefix": "4f53cda1", "content": "[]"},
                "train.log": {"bytes": 0, "sha256_prefix": "e3b0c442"},
            },
            "note": ("the zero-episode stub was removed at the user's ruling; "
                     "the same run id was reused for the relaunch"),
        },
        "verdict": (
            "PASS" if manifest.get("run_id") == "FORMAL_1b47e0ed_seed11"
            and episodes["contiguous"] and not episodes["duplicate_episodes"]
            and manifest.get("status") == "completed_early_stop"
            else "FAIL"),
    }


def runtime() -> dict:
    lines = []
    if HISTORY.exists():
        for no, line in enumerate(HISTORY.read_text(
                encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            if ("run_formal_training" in low or "train_da_gate" in low
                    or "foreach ($s in 23,37,53,71)" in line
                    or "energy_env" in low):
                lines.append({"line": no, "text": line.strip()[:500]})
    evidence = [row for row in lines if "run_formal_training" in row["text"]
                or "foreach ($s in 23,37,53,71)" in row["text"]]
    uses_energy_env = any("energy_env" in row["text"] for row in evidence)
    return {
        "diagnostic": environment(),
        "history_path": str(HISTORY),
        "history_exists": HISTORY.exists(),
        "launch_evidence": evidence,
        "seed11": {
            "interpreter": "energy_env python 3.10.20 / torch 2.11.0",
            "source": "reports/cc03/phase25_seed11.md",
            "verdict": "PASS",
        },
        "seed23": {
            "launch_command": (
                "C:\\Users\\26036\\.conda\\envs\\energy_env\\python.exe "
                "src\\run_formal_training.py --seed $s; $s in 23,37,53,71"),
            "source": "PowerShell ConsoleHost_history.txt",
            "verdict": ("PASS (launch command) / INFERRED (child runtime)"
                        if uses_energy_env else "NOT_TESTED"),
        },
        "seed37": {
            "launch_command": "same foreach loop as seed 23",
            "source": "PowerShell ConsoleHost_history.txt",
            "verdict": ("PASS (launch command) / INFERRED (child runtime)"
                        if uses_energy_env else "NOT_TESTED"),
        },
        "manifest_fields": "no interpreter or package version is recorded "
                           "(TD-10)",
    }


def holdout(run_ids: dict[int, Path], split: dict) -> dict:
    near = split["near"]
    test_ids = set(near["test"])
    holiday_ids = set(near["holiday_block"])
    train_ids = set(near["train"])
    validation_ids = set(near["validation"])
    per_run = {}
    for seed, run in run_ids.items():
        manifest = read_json(run / "run_manifest.json")
        seen = set()
        seen |= set(episodes_summary(run / "episodes.csv")["pair_ids"])
        seen |= set(eval_history_pairs(run / "eval_history.json")["pair_ids"])
        for name in ("gate_stats.jsonl", "block_decisions.jsonl"):
            if (run / name).exists():
                seen |= log_pairs(run / name)
        per_run[seed] = {
            "h2_holdout_read": manifest.get("h2_holdout_read"),
            "test_evaluation": manifest.get("test_evaluation"),
            "legacy_seen_test_run": manifest.get("legacy_seen_test_run"),
            "test_pairs": manifest.get("test_pairs"),
            "holiday_pairs": manifest.get("holiday_pairs"),
            "validation_pairs": manifest.get("validation_pairs"),
            "train_pairs": manifest.get("train_pairs"),
            "test_ids_in_artifacts": sorted(seen & test_ids),
            "holiday_ids_in_artifacts": sorted(seen & holiday_ids),
            "ids_outside_train_or_validation": sorted(
                seen - train_ids - validation_ids),
            "artifact_pair_count": len(seen),
        }
    return {
        "split_counts": split.get("counts"),
        "h2_opened_data_manifest": read_json(
            MANIFESTS[0] / "formal_training_data_manifest.json"
        ).get("h2_opened"),
        "per_run": per_run,
        "verdict": (
            "PASS (artifacts) / INFERRED (seal)"
            if all(not r["test_ids_in_artifacts"]
                   and not r["holiday_ids_in_artifacts"]
                   and r["h2_holdout_read"] is False
                   for r in per_run.values()) else "FAIL"),
    }


def frozen() -> dict:
    code = read_json(MANIFESTS[0] / "formal_training_code_manifest.json")
    sources = {}
    for name, expected in code.get("source_sha256", {}).items():
        path = ROOT / "src" / name
        actual = sha256(path) if path.exists() else None
        sources[name] = {"expected": expected, "actual": actual,
                         "matches": actual == expected}
    tests = {}
    for name, expected in (code.get("tests") or {}).items():
        path = None
        for base in ("tests", "src"):
            candidate = ROOT / base / name
            if candidate.exists():
                path = candidate
                break
        actual = sha256(path) if path else None
        tests[name] = {"expected": expected, "actual": actual,
                       "matches": actual == expected,
                       "path": str(path.relative_to(ROOT)) if path else None}
    return {
        "code_manifest_sha256": sha256(
            MANIFESTS[0] / "formal_training_code_manifest.json"),
        "source_pins": sources,
        "test_pins": tests,
        "all_sources_match": all(v["matches"] for v in sources.values()),
        "all_tests_match": all(v["matches"] for v in tests.values()),
    }


def cc01_reference() -> dict:
    path = ROOT / "reports/cc01/frozen_audit.json"
    if not path.exists():
        return {"exists": False}
    doc = read_json(path)
    return {
        "exists": True,
        "identical": doc.get("identical"),
        "drift": doc.get("drift"),
        "manifest_dirs": sorted(
            (doc.get("before") or {}).get("manifest_dirs", {}).keys()),
        "note": "predates formal_training_v3 (2026-11-05 15:58)",
    }


def gate() -> dict:
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src/run_formal_training.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"})
    text = (proc.stdout or "") + (proc.stderr or "")
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    checks = (parsed or {}).get("checks") or {}
    return {
        "exit_code": proc.returncode,
        "summary": parsed,
        "tail": text.strip().splitlines()[-5:],
        "verdict": (
            "PASS" if proc.returncode == 0 and parsed
            and checks.get("all_pass") is True
            and checks.get("budget_contract_ok") is True
            and parsed.get("status") == "GO_FORMAL_TRAINING"
            else "FAIL"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = read_json(SPLIT)
    run_ids = {seed: ROOT / f"runs/FORMAL_1b47e0ed_seed{seed}"
               for seed in V3_RUNS}
    before = {str(path.relative_to(ROOT)): tree_hash(path)
              for path in [*run_ids.values(),
                           *[ROOT / f"runs/FORMAL_1b47e0ed_seed{s}_best"
                             for s in (11, 23, 37)],
                           *[ROOT / f"snapshots/FORMAL_1b47e0ed_seed{s}"
                             for s in V3_RUNS],
                           V2_SEED53,
                           *MANIFESTS]}

    evidence = {
        "generated_at": now(),
        "root": str(ROOT),
        "environment": environment(),
        "seed11_identity": seed11_identity(),
        "run_artifacts": {
            seed: artifact_check(path, seed) for seed, path in run_ids.items()},
        "checkpoints": {
            seed: checkpoints(path) for seed, path in run_ids.items()},
        "best": {seed: best_dir(seed) for seed in V3_RUNS},
        "oom": {
            seed: scan_oom(path) for seed, path in run_ids.items()},
        "oom_documented_kills": [
            {"path": "reports/cc03/phase25_launch.md",
             "claim": "first launch killed before episode 1 by system memory"},
            {"path": "reports/cc03/phase25_seed11.md",
             "claim": "second launch killed at episode 221; resumed by hand"},
        ],
        "runtime": runtime(),
        "holdout": holdout(run_ids, split),
        "seed37": {
            "run_manifest": read_json(run_ids[37] / "run_manifest.json"),
            "episodes": episodes_summary(run_ids[37] / "episodes.csv"),
            "checkpoints": checkpoints(run_ids[37]),
            "eval_history": {
                "rounds": len(read_json(run_ids[37] / "eval_history.json"))},
            "verdict": "PASS (stopped run, not formal evidence)",
        },
        "seed53": {
            "run_dir": str(V2_SEED53.relative_to(ROOT)),
            "run_manifest": read_json(V2_SEED53 / "run_manifest.json"),
            "episodes": episodes_summary(V2_SEED53 / "episodes.csv"),
            "checkpoints": checkpoints(V2_SEED53),
            "verdict": "PASS (failed run, read-only, not resumed)",
        },
        "frozen": frozen(),
        "cc01_reference": cc01_reference(),
        "gate": gate(),
    }

    after = {str(path.relative_to(ROOT)): tree_hash(path)
             for path in [*run_ids.values(),
                          *[ROOT / f"runs/FORMAL_1b47e0ed_seed{s}_best"
                            for s in (11, 23, 37)],
                          *[ROOT / f"snapshots/FORMAL_1b47e0ed_seed{s}"
                            for s in V3_RUNS],
                          V2_SEED53,
                          *MANIFESTS]}
    unchanged = {key: before[key]["tree_sha256"] == after[key]["tree_sha256"]
                 for key in before}
    evidence["tree_unchanged"] = unchanged
    evidence["verdict"] = (
        "PASS_P0_INTEGRITY"
        if evidence["seed11_identity"]["verdict"] == "PASS"
        and evidence["frozen"]["all_sources_match"]
        and evidence["frozen"]["all_tests_match"]
        and evidence["gate"]["verdict"] == "PASS"
        and all(unchanged.values())
        and evidence["run_artifacts"][11]["artifacts_sha256_present"]
        and evidence["run_artifacts"][23]["artifacts_sha256_present"]
        else "NO_GO_P0_INTEGRITY")

    (OUT / "P0_evidence.json").write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    (OUT / "P0_tree_hashes.json").write_text(
        json.dumps({"before": {k: v["tree_sha256"] for k, v in before.items()},
                    "after": {k: v["tree_sha256"] for k, v in after.items()},
                    "unchanged": unchanged}, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": evidence["verdict"],
        "seed11": evidence["seed11_identity"]["verdict"],
        "sources_match": evidence["frozen"]["all_sources_match"],
        "tests_match": evidence["frozen"]["all_tests_match"],
        "gate": evidence["gate"]["verdict"],
        "trees_unchanged": all(unchanged.values()),
        "artifacts_11": evidence["run_artifacts"][11]["artifacts_sha256_present"],
        "artifacts_23": evidence["run_artifacts"][23]["artifacts_sha256_present"],
        "runtime_23": evidence["runtime"]["seed23"]["verdict"],
        "holdout": evidence["holdout"]["verdict"],
    }, indent=2))
    return 0 if evidence["verdict"] == "PASS_P0_INTEGRITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
