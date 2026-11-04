"""P22: pause and resume without duplicate counters or log rows.

Read-only with respect to historical runs. Two rounds over the five-day smoke
subset are written to a temporary run root; the second call resumes the first.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import train_da_gate as entry

RUN_ID = "PHASE22_RESUME_CHECK"
DAYS = "2022-08-14,2022-10-13,2022-11-01,2022-12-17,2022-12-27"
OUT = Path("reports/v2_cy_jn_2025/formal_training_ready/"
           "phase22_resume.json")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    old_runs, old_snaps = entry.RUNS, entry.SNAPSHOTS
    with tempfile.TemporaryDirectory(prefix="makerb_resume_") as tmp:
        root = Path(tmp)
        entry.RUNS = root / "runs"
        entry.SNAPSHOTS = root / "snapshots"
        common = [
            "--split", "common", "--train-days", DAYS,
            "--exposures", "2", "--seed", "11", "--run-id", RUN_ID,
            "--skip-eval", "--trace-blocks", "--early-stop-patience", "0",
        ]
        try:
            first = entry.main(common + ["--pause-after-rounds", "1"])
            run = root / "runs" / RUN_ID
            paused_manifest = json.loads(
                (run / "run_manifest.json").read_text(encoding="utf-8"))
            paused_episodes = read_csv(run / "episodes.csv")
            paused_blocks = jsonl(run / "block_decisions.jsonl")
            second = entry.main(common + ["--resume"])
            final_manifest = json.loads(
                (run / "run_manifest.json").read_text(encoding="utf-8"))
            episodes = read_csv(run / "episodes.csv")
            gates = jsonl(run / "gate_stats.jsonl")
            samplers = jsonl(run / "sampler_stats.jsonl")
            blocks = jsonl(run / "block_decisions.jsonl")
            counter = Counter(int(e["episode"]) for e in episodes)
            gate_counter = Counter(int(e["episode"]) for e in gates)
            sampler_counter = Counter(int(e["episode"]) for e in samplers)
            block_counter = Counter(int(e["episode"]) for e in blocks)
        finally:
            entry.RUNS, entry.SNAPSHOTS = old_runs, old_snaps
    checks = {
        "first_return_zero": first == 0,
        "second_return_zero": second == 0,
        "paused_status": paused_manifest.get("status") == "paused",
        "paused_episodes_five": len(paused_episodes) == 5,
        "paused_blocks_480": len(paused_blocks) == 5 * 24 * 4,
        "completed_status": final_manifest.get("status") == "completed",
        "episodes_ten": len(episodes) == 10,
        "episodes_unique": sorted(counter) == list(range(1, 11))
        and all(v == 1 for v in counter.values()),
        "gates_unique": sorted(gate_counter) == list(range(1, 11))
        and all(v == 1 for v in gate_counter.values()),
        "samplers_unique": sorted(sampler_counter) == list(range(1, 11))
        and all(v == 1 for v in sampler_counter.values()),
        "blocks_unique": sorted(block_counter) == list(range(1, 11))
        and all(v == 96 for v in block_counter.values()),
        "counters_ten": final_manifest.get("episodes_run") == 10,
        "steps_240": final_manifest.get("total_steps") == 240,
        "buffer_240": final_manifest.get("buffer_len") == 240,
        "artifacts_hashed": bool(final_manifest.get("artifacts_sha256")),
    }
    payload = {
        "boundary": ("temporary run root only; no historical run is touched"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "paused": {
            "status": paused_manifest.get("status"),
            "episodes": len(paused_episodes),
            "blocks": len(paused_blocks),
            "counters": {
                key: paused_manifest.get(key)
                for key in ("episodes_run", "total_steps", "updates",
                            "buffer_len")},
        },
        "final": {
            "status": final_manifest.get("status"),
            "episodes_run": final_manifest.get("episodes_run"),
            "total_steps": final_manifest.get("total_steps"),
            "buffer_len": final_manifest.get("buffer_len"),
            "episode_counts": dict(sorted(counter.items())),
            "block_rows": len(blocks),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": payload["all_pass"], "checks": checks},
                     indent=2))
    print(f"wrote {OUT}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
