"""Executable invariants for the v7 formal-training budget.

v7 keeps every v6 rule that still describes the run: the round floor, the
exposure and block-step arithmetic, the seed list and the test/holiday
boundary. Four rules change, and only these are replaced here.

  * the train block is the v3 local day-ahead library, not the NEAR split;
  * the day ends on a free SOC, valued at the last external price, so there is
    no pinned endpoint and no day-end window to declare;
  * the 40 unpaired 2025 price dates are declared used for price-date
    adaptation, so the H2 seal is replaced by a declaration that bounds the
    claim, not by silence;
  * the wall-clock projection is recorded for the reader and is not a gate.

Read-only: it opens the config and the data manifest, never a run directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

import training_contract as v6

DEFAULT_CONFIG_DIR = Path("reports/v2_cy_jn_2025/formal_training_v7")
EXPECTED_COUNTS = {"train": 90, "validation": 18, "test": 18, "holiday": 10}
#: The unpaired 2025 price dates inside the v3 library, by the H2 definition.
H2_PRICE_DATES_USED = 40
#: What the manifest may claim once those dates are in training.
H2_USE = ("price-date adaptation only: the 40 unpaired 2025 price dates the H2 "
          "definition enumerates also entered training, so no result supports "
          "joint source-load generalisation")

#: v6 rule codes that describe a pinned day-end SOC, the NEAR split, or the
#: sealed holdout. Each is replaced by a v7 rule below.
SUPERSEDED = {"terminal_soc_not_neutral", "terminal_soc_window_not_exclusive",
              "split_counts_mismatch", "boundary_h2_opened",
              "data_legacy_label_matches_declared", "data_h2_opened"}


def violation(code: str, where: str, expected: Any, actual: Any,
              message: str) -> dict:
    return v6.violation(code, where, expected, actual, message)


def validate_contract(config: dict, data_manifest: dict) -> list[dict]:
    """Every v6 violation except the superseded three, plus the v7 rules."""
    violations = [item for item in v6.validate_contract(config, data_manifest)
                  if item["code"] not in SUPERSEDED]
    formal = config.get("formal_training") or {}
    terminal = formal.get("terminal_soc") or {}
    counts = data_manifest.get("counts") or {}

    if terminal.get("mode") != "free":
        violations.append(violation(
            "terminal_soc_mode_not_free", "formal_training.terminal_soc.mode",
            "free", terminal.get("mode"),
            "v7 values the endpoint instead of pinning it"))
    if terminal.get("initial") != 0.5:
        violations.append(violation(
            "terminal_soc_initial", "formal_training.terminal_soc.initial",
            0.5, terminal.get("initial"),
            "the day must start at the declared initial SOC"))
    if "final" in terminal:
        violations.append(violation(
            "terminal_soc_final_declared",
            "formal_training.terminal_soc.final", None, terminal.get("final"),
            "a free endpoint declares no final value"))

    wall = formal.get("wall_clock") or {}
    values = [wall.get("hours_per_seed_min"), wall.get("hours_per_seed_max")]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value > 0 for value in values):
        violations.append(violation(
            "wall_clock_not_recorded", "formal_training.wall_clock",
            "finite positive min and max hours", wall,
            "the projection is recorded for the reader; it is not a gate"))
    elif wall["hours_per_seed_min"] > wall["hours_per_seed_max"]:
        violations.append(violation(
            "wall_clock_inverted", "formal_training.wall_clock",
            "min <= max", values,
            "the recorded projection must be ordered"))

    if counts != EXPECTED_COUNTS:
        violations.append(violation(
            "split_counts_mismatch", "data_manifest.counts", EXPECTED_COUNTS,
            counts, "the frozen v3 library has a different block size"))

    # The holdout is not sealed: the library's price dates are the same series
    # the H2 definition counts, so the manifest states how many entered
    # training and what that still supports.
    boundaries = formal.get("boundaries") or {}
    if (data_manifest.get("h2_opened") is not True
            or data_manifest.get("h2_price_dates_used") != H2_PRICE_DATES_USED
            or boundaries.get("h2_opened") is not True
            or data_manifest.get("independent_generalisation_claim") is not False):
        violations.append(violation(
            "h2_use_undeclared", "data_manifest.h2_price_dates_used",
            H2_PRICE_DATES_USED, data_manifest.get("h2_price_dates_used"),
            "the price dates the library drew on must be declared, and the "
            "claim bounded to price-date adaptation"))

    if data_manifest.get("legacy_label") != "LEGACY_SEEN_TEST":
        violations.append(violation(
            "legacy_label_undeclared", "data_manifest.legacy_label",
            "LEGACY_SEEN_TEST", data_manifest.get("legacy_label"),
            "the test and holiday blocks carry the legacy label"))
    return violations


def evaluate(config_dir: Path) -> dict:
    config_dir = Path(config_dir)
    config = yaml.safe_load(
        (config_dir / "formal_training_config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (config_dir / "formal_training_data_manifest.json").read_text(
            encoding="utf-8"))
    violations = validate_contract(config, manifest)
    return {"ok": not violations, "config_dir": str(config_dir),
            "violations": violations}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="accepted for symmetry with the launcher")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    args = parser.parse_args(argv)
    result = evaluate(Path(args.config_dir))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
