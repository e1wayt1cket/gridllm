# test_action_mapping_gradients.py
"""Analytic derivatives of the two quote maps, offline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_action_mapping_gradients as audit


@pytest.mark.parametrize("z", [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
def test_the_analytic_derivatives_match_a_central_difference(z):
    h = 150.0
    eps = 1e-5
    old_num = (audit.old_map(z + eps, h) - audit.old_map(z - eps, h)) / (2 * eps)
    open_num = (audit.open_map(z + eps, h)
                - audit.open_map(z - eps, h)) / (2 * eps)
    assert audit.old_derivative(z, h) == pytest.approx(old_num, abs=1e-6)
    assert audit.open_derivative(z, h) == pytest.approx(open_num, abs=1e-6)


def test_the_maps_agree_at_the_midpoint_and_differ_at_truth():
    h = 150.0
    assert audit.old_map(0.0, h) == pytest.approx(h / 2)
    assert audit.open_map(0.0, h) == pytest.approx(h / 2)
    assert audit.old_map(-5.0, h) > 0.0
    assert audit.old_map(-50.0, h) == 0.0          # float tanh saturation only


def test_the_derivatives_have_the_expected_limits():
    h = 150.0
    assert audit.old_derivative(0.0, h) == pytest.approx(h / 2)
    assert audit.old_derivative(-50.0, h) == 0.0
    assert audit.open_derivative(0.0, h) == pytest.approx(h / 4)
    assert audit.open_derivative(-50.0, h) < 1e-15
    assert audit.open_derivative(50.0, h) == 0.0


def test_the_open_branch_is_monotone_on_the_grid():
    values = [audit.open_map(z, 300.0) for z in audit.GRID]
    assert np.all(np.diff(values) >= 0)
