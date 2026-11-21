"""Contracts for the executable formal-training budget checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import training_contract as tc

V3 = Path("reports/v2_cy_jn_2025/formal_training_v3")


def _frozen():
    config, manifest = tc.load_contract(V3)
    return copy.deepcopy(config), copy.deepcopy(manifest)


def _codes(config, manifest):
    return [item["code"] for item in tc.validate_contract(config, manifest)]


def test_frozen_v3_contract_has_no_violations():
    config, manifest = _frozen()
    assert tc.validate_contract(config, manifest) == []
    assert config["formal_training"]["protocol_version"] == 3


def test_launcher_passes_the_whole_frozen_stop_rule():
    import run_formal_training as rft

    config, _ = _frozen()
    text = " ".join(rft.command(11, config, V3))
    assert "--exposures 15" in text
    assert "--min-rounds 10" in text
    assert "--max-rounds 15" in text
    assert "--max-episodes 795" in text


def test_v3_run_id_uses_a_new_prefix():
    import run_formal_training as rft

    ids = {
        rft.run_id(11, Path(f"reports/v2_cy_jn_2025/{name}"))
        for name in ("formal_training_v3", "formal_training_v2",
                     "formal_training_ready")
    }
    assert len(ids) == 3


def test_minimum_episodes_must_match_rounds_times_train_days():
    config, manifest = _frozen()
    config["formal_training"]["budget"]["episodes_per_seed_min"] = 500
    codes = _codes(config, manifest)
    assert "min_episodes_mismatch" in codes


def test_minimum_rounds_must_agree_with_exposures_floor():
    config, manifest = _frozen()
    config["formal_training"]["budget"]["exposures_per_day_min"] = 8
    assert "min_rounds_exposures_mismatch" in _codes(config, manifest)


def test_round_floor_cannot_exceed_maximum_exposures():
    config, manifest = _frozen()
    budget = config["formal_training"]["budget"]
    budget["minimum_rounds"] = 20
    budget["exposures_per_day_min"] = 20
    budget["episodes_per_seed_min"] = 20 * manifest["counts"]["train"]
    assert "min_rounds_above_max_exposures" in _codes(config, manifest)


def test_block_steps_are_derived_from_episodes():
    config, manifest = _frozen()
    config["formal_training"]["budget"]["block_steps_per_seed_max"] = 1
    assert "max_block_steps_mismatch" in _codes(config, manifest)


def test_non_positive_patience_is_rejected():
    config, manifest = _frozen()
    config["formal_training"]["validation"]["early_stop_patience"] = 0
    assert "patience_below_one" in _codes(config, manifest)


def test_duplicate_seeds_are_rejected():
    config, manifest = _frozen()
    config["formal_training"]["budget"]["seeds"] = [11, 11, 23]
    assert "seeds_duplicated" in _codes(config, manifest)


def test_terminal_soc_must_be_neutral_and_window_only():
    config, manifest = _frozen()
    terminal = config["formal_training"]["terminal_soc"]
    terminal["final"] = 0.4
    terminal["day_end_window_only"] = False
    codes = _codes(config, manifest)
    assert "terminal_soc_not_neutral" in codes
    assert "terminal_soc_window_not_exclusive" in codes


def test_test_and_h2_boundaries_must_stay_closed():
    config, manifest = _frozen()
    boundaries = config["formal_training"]["boundaries"]
    boundaries["test_evaluated"] = True
    boundaries["h2_opened"] = True
    manifest["h2_opened"] = True
    codes = _codes(config, manifest)
    assert "boundary_test_evaluated" in codes
    assert "boundary_h2_opened" in codes
    assert "data_h2_opened" in codes


def test_split_counts_must_be_the_frozen_four_blocks():
    config, manifest = _frozen()
    manifest["counts"]["train"] = 54
    assert "split_counts_mismatch" in _codes(config, manifest)


def test_missing_train_count_is_named_not_crashed():
    config, manifest = _frozen()
    manifest.pop("counts")
    codes = _codes(config, manifest)
    assert "train_days_not_integer" in codes


def test_cli_exit_codes_and_evidence(tmp_path: Path, capsys):
    config, manifest = _frozen()
    config_path = tmp_path / "formal_training_config.yaml"
    manifest_path = tmp_path / "formal_training_data_manifest.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert tc.main(["--check", "--config-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    config["formal_training"]["budget"]["episodes_per_seed_min"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert tc.main(["--check", "--config-dir", str(tmp_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert any(item["code"] == "min_episodes_mismatch"
               for item in result["violations"])
