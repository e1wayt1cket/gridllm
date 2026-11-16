"""Executable invariants for the frozen formal-training budget.

The budget keys are hand-written in `formal_training_config.yaml`. Nothing in
the training entry checks that they agree with each other or with the split's
train-day count, so a contradictory manifest would still start a run. This
module turns those agreements into named violations that a launcher can refuse
on.

It is read-only: it opens the config and the data manifest, never a run
directory, checkpoint or log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

DEFAULT_CONFIG_DIR = Path("reports/v2_cy_jn_2025/formal_training_v6")
EXPECTED_COUNTS = {"train": 53, "validation": 11, "test": 11,
                   "holiday_block": 5}


def violation(code: str, where: str, expected: Any, actual: Any,
              message: str) -> dict:
    return {"code": code, "where": where, "expected": expected,
            "actual": actual, "message": message}


def _require_int(value: Any, where: str, code: str) -> list[dict]:
    if isinstance(value, int) and not isinstance(value, bool):
        return []
    return [violation(code, where, "integer", value,
                      f"{where} must be an integer")]


def validate_contract(config: dict, data_manifest: dict) -> list[dict]:
    """Return every budget/isolation violation; an empty list means GO.

    `config` is the parsed `formal_training_config.yaml` and
    `data_manifest` the parsed `formal_training_data_manifest.json`.
    """
    violations: list[dict] = []
    formal = config.get("formal_training") or {}
    budget = formal.get("budget") or {}
    episode = formal.get("episode") or {}
    validation = formal.get("validation") or {}
    counts = data_manifest.get("counts") or {}

    where = "formal_training.budget"
    min_rounds = budget.get("minimum_rounds")
    exp_min = budget.get("exposures_per_day_min")
    exp_max = budget.get("exposures_per_day_max")
    ep_min = budget.get("episodes_per_seed_min")
    ep_max = budget.get("episodes_per_seed_max")
    bs_min = budget.get("block_steps_per_seed_min")
    bs_max = budget.get("block_steps_per_seed_max")
    patience = validation.get("early_stop_patience")
    blocks = episode.get("blocks")
    train_days = counts.get("train")

    for check in (
        _require_int(min_rounds, f"{where}.minimum_rounds",
                     "minimum_rounds_not_integer"),
        _require_int(exp_min, f"{where}.exposures_per_day_min",
                     "exposures_per_day_min_not_integer"),
        _require_int(exp_max, f"{where}.exposures_per_day_max",
                     "exposures_per_day_max_not_integer"),
        _require_int(ep_min, f"{where}.episodes_per_seed_min",
                     "episodes_per_seed_min_not_integer"),
        _require_int(ep_max, f"{where}.episodes_per_seed_max",
                     "episodes_per_seed_max_not_integer"),
        _require_int(bs_min, f"{where}.block_steps_per_seed_min",
                     "block_steps_per_seed_min_not_integer"),
        _require_int(bs_max, f"{where}.block_steps_per_seed_max",
                     "block_steps_per_seed_max_not_integer"),
        _require_int(patience, "formal_training.validation.early_stop_patience",
                     "early_stop_patience_not_integer"),
        _require_int(blocks, "formal_training.episode.blocks",
                     "blocks_not_integer"),
        _require_int(train_days, "data_manifest.counts.train",
                     "train_days_not_integer"),
    ):
        violations.extend(check)

    if violations:
        return violations

    if min_rounds < 1:
        violations.append(violation(
            "minimum_rounds_below_one", f"{where}.minimum_rounds", ">= 1",
            min_rounds, "early stopping needs a positive round floor"))
    if min_rounds != exp_min:
        violations.append(violation(
            "min_rounds_exposures_mismatch",
            f"{where}.minimum_rounds vs exposures_per_day_min", exp_min,
            min_rounds,
            "the round floor and the minimum exposures per day must agree"))
    if min_rounds > exp_max:
        violations.append(violation(
            "min_rounds_above_max_exposures",
            f"{where}.minimum_rounds vs exposures_per_day_max", f"<= {exp_max}",
            min_rounds,
            "the round floor cannot exceed the maximum exposures per day"))
    if ep_min != min_rounds * train_days:
        violations.append(violation(
            "min_episodes_mismatch", f"{where}.episodes_per_seed_min",
            min_rounds * train_days, ep_min,
            "minimum episodes must equal minimum rounds times train days"))
    if ep_max != exp_max * train_days:
        violations.append(violation(
            "max_episodes_mismatch", f"{where}.episodes_per_seed_max",
            exp_max * train_days, ep_max,
            "maximum episodes must equal maximum exposures times train days"))
    if bs_min != ep_min * blocks:
        violations.append(violation(
            "min_block_steps_mismatch",
            f"{where}.block_steps_per_seed_min", ep_min * blocks, bs_min,
            "minimum block steps must equal minimum episodes times blocks"))
    if bs_max != ep_max * blocks:
        violations.append(violation(
            "max_block_steps_mismatch",
            f"{where}.block_steps_per_seed_max", ep_max * blocks, bs_max,
            "maximum block steps must equal maximum episodes times blocks"))
    if patience < 1:
        violations.append(violation(
            "patience_below_one",
            "formal_training.validation.early_stop_patience", ">= 1", patience,
            "a non-positive patience disables early stopping silently"))

    seeds = budget.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        violations.append(violation(
            "seeds_missing", f"{where}.seeds", "non-empty list", seeds,
            "at least one seed must be frozen"))
    elif len(set(seeds)) != len(seeds):
        duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
        violations.append(violation(
            "seeds_duplicated", f"{where}.seeds", "unique seeds", duplicates,
            "a repeated seed would overwrite its own run directory"))

    terminal = formal.get("terminal_soc") or {}
    if terminal.get("initial") != terminal.get("final"):
        violations.append(violation(
            "terminal_soc_not_neutral",
            "formal_training.terminal_soc.initial vs final",
            terminal.get("initial"), terminal.get("final"),
            "the day must start and end at the same SOC"))
    if terminal.get("day_end_window_only") is not True:
        violations.append(violation(
            "terminal_soc_window_not_exclusive",
            "formal_training.terminal_soc.day_end_window_only", True,
            terminal.get("day_end_window_only"),
            "the terminal constraint must apply only at the day-end window"))

    boundaries = formal.get("boundaries") or {}
    for key in ("test_evaluated", "holiday_evaluated", "h2_opened"):
        if boundaries.get(key) is not False:
            violations.append(violation(
                f"boundary_{key}", f"formal_training.boundaries.{key}", False,
                boundaries.get(key), f"{key} must remain false"))

    if counts != EXPECTED_COUNTS:
        violations.append(violation(
            "split_counts_mismatch", "data_manifest.counts", EXPECTED_COUNTS,
            counts, "the frozen split has a different block size"))
    for key in ("all_disjoint", "legacy_label_matches_declared"):
        if data_manifest.get(key) is not True:
            violations.append(violation(
                f"data_{key}", f"data_manifest.{key}", True,
                data_manifest.get(key), f"{key} must be true"))
    if data_manifest.get("h2_opened") is not False:
        violations.append(violation(
            "data_h2_opened", "data_manifest.h2_opened", False,
            data_manifest.get("h2_opened"), "the H2 holdout must stay sealed"))
    return violations


def load_contract(config_dir: Path) -> tuple[dict, dict]:
    config_dir = Path(config_dir)
    config = yaml.safe_load(
        (config_dir / "formal_training_config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (config_dir / "formal_training_data_manifest.json").read_text(
            encoding="utf-8"))
    return config, manifest


def evaluate(config_dir: Path) -> dict:
    config_dir = Path(config_dir)
    config, manifest = load_contract(config_dir)
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
