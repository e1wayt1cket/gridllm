# test_reward_design_facts.py
"""The discount/action facts the seventh-phase design documents rely on."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_reward_design_facts as facts


@pytest.fixture(scope="module")
def payload():
    return facts.facts()


def test_the_discount_is_per_block_not_per_period(payload):
    assert payload["gamma_per_block"] == pytest.approx(0.99)
    assert payload["block_size_periods"] == 4
    assert payload["blocks_per_episode"] == 24
    assert payload["weight_last"] == pytest.approx(0.99 ** 23)
    assert payload["effective_discount_per_15min"] == pytest.approx(0.99 ** 0.25)
    assert 21.0 < payload["weight_sum"] < 22.0


def test_truthful_is_the_lower_corner_of_the_box(payload):
    assert payload["truthful"] == {"charge_price": 0.0, "discharge_price": 0.0}
    assert all(payload["truthful_is_lower_bound"].values())
    assert payload["truthful_at_tanh"] == {"charge_price": -1.0,
                                           "discharge_price": -1.0}
    assert payload["box_midpoint"] == {"charge_price": 75.0,
                                       "discharge_price": 150.0}


def test_terminal_soc_differs_between_training_and_settlement(payload):
    assert payload["terminal_soc_in_training_window"] is False
    assert payload["terminal_soc_in_settlement"] is True


def test_the_seven_design_documents_exist():
    root = Path(__file__).resolve().parents[1] / "reports" / "v2_cy_jn_2025"
    for name in ("daily_objective_definition.md",
                 "reward_design_comparison.md",
                 "discounted_return_alignment.md",
                 "action_parameterization_design.md",
                 "dataset_exposure_experiment_design.md",
                 "matd3_checkpoint_reproducibility_design.md",
                 "paper_method_definition_review.md"):
        assert (root / name).exists(), name
