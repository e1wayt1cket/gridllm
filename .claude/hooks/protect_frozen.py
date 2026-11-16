"""PreToolUse guard: refuse Edit/Write on the frozen formal-training set.

The set is derived at call time, never hardcoded:

* `src/<name>` for every entry in the live code manifest's `source_sha256`;
* the manifest's pinned tests/audits;
* the two manifest directories, and for each of them the seed run directories
  and `_best` / `snapshot` trees its own config can name.

Failures are closed for the affected zone: an unreadable code manifest
protects the whole `src/` tree, and an unresolvable seed list protects `runs/`
and `snapshots/`. Any unexpected exception becomes a deny.

Input is the PreToolUse JSON on stdin. A denial is exit 0 with
`hookSpecificOutput.permissionDecision = "deny"`; an allowed call prints
nothing and exits 0.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The first directory is the live protocol; the rest are historical and are
# frozen as directories and run/snapshot trees, not as source pins.
MANIFEST_DIRS = (
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_v6",
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_v5",
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_v4",
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_v3",
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_v2",
    PROJECT_ROOT / "reports/v2_cy_jn_2025/formal_training_ready",
)
CODE_MANIFEST = MANIFEST_DIRS[0] / "formal_training_code_manifest.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attrs = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _has_link_on_path(path: Path) -> bool:
    current = path
    while True:
        if _is_reparse(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def frozen_targets() -> tuple[set[str], list[Path]]:
    """Return (exact protected files, protected directories)."""
    files: set[str] = set()
    dirs: list[Path] = list(MANIFEST_DIRS)

    try:
        code_manifest = _read_json(CODE_MANIFEST)
        for name in code_manifest.get("source_sha256", {}):
            files.add(_norm(PROJECT_ROOT / "src" / name))
        for name in code_manifest.get("tests", {}):
            for base in ("tests", "src"):
                candidate = PROJECT_ROOT / base / name
                if candidate.exists():
                    files.add(_norm(candidate))
                    break
    except Exception:
        dirs.append(PROJECT_ROOT / "src")

    # Every manifest directory contributes the runs it can name, so the
    # superseded config's runs stay frozen too.
    names_resolved = 0
    for manifest_dir in MANIFEST_DIRS:
        try:
            config = _read_yaml(manifest_dir / "formal_training_config.yaml")
            seeds = config["formal_training"]["budget"].get("seeds", [])
            prefix = str(_read_json(
                manifest_dir / "formal_training_manifest.json",
            ).get("config_sha256", ""))[:8]
            if not prefix:
                raise ValueError("manifest has no config_sha256")
            for seed in seeds:
                run_id = f"FORMAL_{prefix}_seed{seed}"
                dirs.append(PROJECT_ROOT / "runs" / run_id)
                dirs.append(PROJECT_ROOT / "runs" / f"{run_id}_best")
                dirs.append(PROJECT_ROOT / "snapshots" / run_id)
            names_resolved += 1
        except Exception:
            continue
    if names_resolved != len(MANIFEST_DIRS):
        dirs.append(PROJECT_ROOT / "runs")
        dirs.append(PROJECT_ROOT / "snapshots")

    return files, dirs


def match(target: Path) -> tuple[str | None, str | None]:
    """Return (protected source, reason) or (None, None)."""
    resolved = target.resolve(strict=False)
    if _has_link_on_path(resolved):
        return str(resolved), "a symlink or reparse point sits on the path"
    files, dirs = frozen_targets()
    key = _norm(resolved)
    if key in files:
        return str(resolved), (
            "file is pinned by the live formal-training code manifest")
    for directory in dirs:
        base = _norm(directory)
        if key == base or key.startswith(base + os.sep):
            return str(directory), (
                "path is inside a frozen manifest, run or snapshot directory")
    return None, None


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }), flush=True)


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        if event.get("tool_name") not in ("Edit", "Write"):
            return 0
        file_path = (event.get("tool_input") or {}).get("file_path")
        if not file_path:
            deny("Edit/Write call has no file_path; refusing to guess.")
            return 0
        target = Path(file_path)
        if not target.is_absolute():
            target = Path(event.get("cwd") or PROJECT_ROOT) / target
        source, reason = match(target)
        if source is None:
            return 0
        deny(
            f"Refused: {target} is frozen. Source: {source}. {reason}. "
            "Create a new version directory, or request an explicit "
            "re-freeze; do not edit the frozen file."
        )
        return 0
    except Exception as exc:
        try:
            deny(f"Frozen-path guard failed closed: {exc}")
            return 0
        except Exception:
            print(f"frozen-path guard failed: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
