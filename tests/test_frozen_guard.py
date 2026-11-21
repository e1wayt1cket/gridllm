"""Contracts for the PreToolUse frozen-path guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude/hooks/protect_frozen.py"


def _prefix(name: str) -> str:
    manifest = json.loads(
        (ROOT / f"reports/v2_cy_jn_2025/{name}/formal_training_manifest.json")
        .read_text(encoding="utf-8"))
    return manifest["config_sha256"][:8]


# The live protocol's pins are the ones the guard enforces through the code
# manifest; every earlier protocol is frozen as a directory and as the run and
# snapshot trees its own config names.
V7_PREFIX = _prefix("formal_training_v7")
V6_PREFIX = _prefix("formal_training_v6")
V4_PREFIX = _prefix("formal_training_v4")
PREFIX = _prefix("formal_training_v3")
V2_PREFIX = _prefix("formal_training_v2")
SUPERSEDED_PREFIX = _prefix("formal_training_ready")


def _call(tool: str, file_path: str | None, cwd: str | None = None,
          raw: str | None = None):
    if raw is None:
        payload = {"tool_name": tool, "cwd": cwd or str(ROOT),
                   "tool_input": {} if file_path is None
                   else {"file_path": file_path}}
        raw = json.dumps(payload)
    return subprocess.run([sys.executable, str(HOOK)], input=raw, text=True,
                          capture_output=True, cwd=str(ROOT))


def _denied(result) -> bool:
    if result.returncode != 0 or not result.stdout.strip():
        return False
    payload = json.loads(result.stdout)
    return payload["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("relative", [
    "src/train_local_da_v7.py",
    "src/rl_da_env.py",
    "src/run_formal_training.py",
    "src/training_contract_v7.py",
    "src/make_formal_training_manifests.py",
    "tests/test_training_contract_v7.py",
    "tests/test_local_da_env_v7.py",
    "tests/test_frozen_guard.py",
    "reports/v2_cy_jn_2025/formal_training_v7/formal_training_config.yaml",
    "reports/v2_cy_jn_2025/formal_training_v6/formal_training_config.yaml",
    "reports/v2_cy_jn_2025/formal_training_v4/formal_training_config.yaml",
    "reports/v2_cy_jn_2025/formal_training_v3/formal_training_config.yaml",
    "reports/v2_cy_jn_2025/formal_training_v2/formal_training_config.yaml",
    "reports/v2_cy_jn_2025/formal_training_ready/formal_training_config.yaml",
    f"runs/FORMAL_{V7_PREFIX}_seed11/train.log",
    f"runs/FORMAL_{V7_PREFIX}_seed11_best/checkpoints",
    f"snapshots/FORMAL_{V7_PREFIX}_seed11",
    f"runs/FORMAL_{V6_PREFIX}_seed11/train.log",
    f"snapshots/FORMAL_{V6_PREFIX}_seed11",
    f"runs/FORMAL_{V4_PREFIX}_seed11/train.log",
    f"runs/FORMAL_{V4_PREFIX}_seed11_best/checkpoints",
    f"snapshots/FORMAL_{V4_PREFIX}_seed11",
    f"runs/FORMAL_{PREFIX}_seed11/train.log",
    f"runs/FORMAL_{PREFIX}_seed11_best/checkpoints",
    f"snapshots/FORMAL_{PREFIX}_seed11",
    f"runs/FORMAL_{V2_PREFIX}_seed11/train.log",
    f"snapshots/FORMAL_{V2_PREFIX}_seed11",
    f"runs/FORMAL_{SUPERSEDED_PREFIX}_seed11/train.log",
    f"snapshots/FORMAL_{SUPERSEDED_PREFIX}_seed11",
])
def test_frozen_paths_are_denied(relative):
    result = _call("Write", str(ROOT / relative))
    assert _denied(result), result.stdout + result.stderr
    reason = json.loads(result.stdout)["hookSpecificOutput"][
        "permissionDecisionReason"]
    assert "frozen" in reason


@pytest.mark.parametrize("relative", [
    "src/dashboard.py",
    # A superseded protocol's source no longer carries the live pin.
    "src/train_da_gate.py",
    "tests/test_baseline.py",
    "reports/cc01/README.md",
    ".claude/settings.json",
])
def test_unprotected_paths_are_allowed(relative):
    result = _call("Edit", str(ROOT / relative))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_relative_path_is_resolved_against_cwd():
    result = _call("Edit", "src/train_local_da_v7.py", cwd=str(ROOT))
    assert _denied(result)


def test_traversal_cannot_escape_the_check():
    result = _call("Edit", str(ROOT / "src/../src/train_local_da_v7.py"))
    assert _denied(result)


def test_other_tools_are_ignored():
    result = _call("Read", str(ROOT / "src/train_local_da_v7.py"))
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_missing_file_path_fails_closed():
    result = _call("Write", None)
    assert _denied(result)


def test_malformed_input_fails_closed():
    result = _call("Write", None, raw="{not json")
    assert _denied(result)
