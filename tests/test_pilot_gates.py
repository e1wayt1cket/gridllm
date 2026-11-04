# test_pilot_gates.py
"""The pre-training gates the pilot may not bypass."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_pilot_gates as gates

ROOT = Path(__file__).resolve().parents[1]


def test_no_observation_reads_the_price_or_the_counterfactual():
    facts = gates.timing_audit(ROOT)
    assert facts["obs_reads_price_or_baseline"] == []
    assert facts["obs_feature_count"] == 12


def test_the_cycle_cost_is_charged_once_and_the_penalty_is_in_the_actor():
    facts = gates.accounting_check(ROOT)
    assert facts["cycle_cost_charged_once"] is True
    assert facts["raw_profit_uses_participant_payoff"] is True
    assert facts["env_applies_deviation_penalty"] is False
    assert facts["actor_loss_adds_penalty"] is True


@pytest.mark.skipif(
    not (ROOT / "data/processed/v2_cy_jn_2025/load_price_pairing.csv").exists(),
    reason="V2-CY-JN-2025 artifacts are not built")
def test_the_truthful_baseline_is_state_matched():
    facts = gates.state_matching_check(ROOT)
    assert facts["blocks"] >= 2
    assert all(facts["checks"].values()), facts["checks"]
    assert facts["baseline_readable_from_obs"] is False
