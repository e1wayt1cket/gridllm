# test_action_gate_prototype.py
"""The truthful-gate mapping: exact truth, legal range, monotone, no clamping."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import action_gate as gate
from rl_spec import ACTION_STORAGE_QUOTE_V2


def test_the_prototype_uses_the_market_bounds():
    assert gate.LEG_HIGHS == (150.0, 300.0)
    assert ACTION_STORAGE_QUOTE_V2.bounds["charge_price"] == (0.0, 150.0)
    assert ACTION_STORAGE_QUOTE_V2.bounds["discharge_price"] == (0.0, 300.0)
    assert ACTION_STORAGE_QUOTE_V2.truthful == {"charge_price": 0.0,
                                                "discharge_price": 0.0}


def test_a_closed_gate_is_exactly_truthful_at_any_logit():
    for logit in (-1e6, -5.0, 0.0, 5.0, 1e6):
        q = gate.gated_quote([False, False], [logit, logit])
        assert q[0] == 0.0 and q[1] == 0.0


def test_an_open_gate_is_legal_and_non_negative():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 10, size=(1000, 2))
    q = gate.gated_quote(np.ones((1000, 2), dtype=bool), logits)
    assert np.all(q >= 0.0)
    assert np.all(q[:, 0] <= 150.0)
    assert np.all(q[:, 1] <= 300.0)
    assert np.all(np.isfinite(q))


def test_the_open_branch_is_monotone_in_the_logit():
    logits = np.linspace(-10, 10, 41)
    q = gate.gated_quote(np.ones((41, 2), dtype=bool),
                         np.stack([logits, logits], axis=1))
    assert np.all(np.diff(q[:, 0]) >= 0)
    assert np.all(np.diff(q[:, 1]) >= 0)


def test_the_gates_are_independent_per_leg():
    q = gate.gated_quote([False, True], [0.0, 0.0])
    assert q[0] == 0.0
    assert q[1] == pytest.approx(150.0)     # 300 * sigmoid(0)


def test_extreme_logits_stay_legal_without_overflow():
    with np.errstate(over="raise", invalid="raise"):
        q = gate.gated_quote([True, True], [1e6, -1e6])
    assert q[0] <= 150.0 and q[1] >= 0.0
    assert np.isfinite(q).all()


def test_the_mapping_does_not_clip_silently():
    # The open branch approaches the legal upper bound; it never exceeds it.
    q = gate.gated_quote([True, True], [60.0, 60.0])
    assert q[0] == pytest.approx(150.0)
    assert q[1] == pytest.approx(300.0)


def test_decode_round_trips_open_legs_and_marks_closed_ones():
    quote = [40.0, 120.0]
    g, z = gate.decode_quote(quote)
    assert g.tolist() == [True, True]
    np.testing.assert_allclose(gate.gated_quote(g, z), quote)
    g0, z0 = gate.decode_quote([0.0, 50.0])
    assert g0.tolist() == [False, True]
    assert np.isnan(z0[0])


def test_invalid_shapes_and_bounds_are_refused():
    with pytest.raises(ValueError):
        gate.gated_quote([True, False], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gate.gated_quote([True], [0.0], high=[-1.0])
    with pytest.raises(ValueError):
        gate.gated_quote([True], [float("inf")])
    with pytest.raises(ValueError):
        gate.decode_quote([200.0, 100.0])   # charge above its box


def test_truth_no_longer_requires_saturation():
    # The old tanh mapping needs an unbounded logit to reach 0.
    mid, half = 75.0, 75.0
    old_at_minus_5 = mid + np.tanh(-5.0) * half
    assert old_at_minus_5 > 0.0
    assert gate.gated_quote([False], [-5.0], high=[150.0])[0] == 0.0


def test_the_action_object_refuses_a_non_truthful_anchor():
    with pytest.raises(ValueError):
        gate.StorageGateAction(truthful=(1.0, 0.0))
