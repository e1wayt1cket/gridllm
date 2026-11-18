"""The price-and-quantity quote map: a gate, a price and a quantity per leg."""

import pytest
import torch

import rl_spec_v7  # noqa: F401  registers the v5 action spec
from da_quote_map_v5 import (ACTION_BOX_HIGH, QUOTE_DIM, TRUTHFUL,
                             forward_policy_v5, target_quotes_v5)
from rl_spec import get_action_spec

CLOSED = torch.tensor([[-1., -1., 2., -2., 3., 0.]])
OPEN_MID = torch.tensor([[1., 1., 0., 0., 0., 0.]])


def test_a_closed_gate_declares_the_truthful_vector():
    out = forward_policy_v5(CLOSED)
    assert out["quotes"].shape == (1, QUOTE_DIM)
    assert set(out) == {"quotes", "gates", "probs", "gate_logits",
                        "quote_logits", "mode"}
    assert out["quotes"].dtype == CLOSED.dtype
    # The map runs in the policy's dtype, where 0.4 is not exact; compare at
    # the precision the box is actually evaluated in.
    assert out["quotes"][0].tolist() == pytest.approx(list(TRUTHFUL))


def test_an_open_gate_at_zero_logit_withholds_half():
    out = forward_policy_v5(OPEN_MID)
    q = out["quotes"][0]
    assert abs(q[0].item() - 75.) < 1e-4 and abs(q[2].item() - 150.) < 1e-4
    assert abs(q[1].item() - 0.2) < 1e-4 and abs(q[3].item() - 0.2) < 1e-4


def test_withholding_is_reachable_and_differentiable():
    z = torch.tensor([[1., 1., 0., -3., 0., 3.]], requires_grad=True)
    q = forward_policy_v5(z)["quotes"][0]
    # A low quantity logit withholds more than a high one, and neither is 0.
    assert 0. < q[1].item() < q[3].item() < 0.4
    q.sum().backward()
    assert z.grad[0, 3].item() > 0. and z.grad[0, 5].item() > 0.


def test_the_target_action_is_the_hard_gate_map():
    assert torch.allclose(target_quotes_v5(OPEN_MID),
                          forward_policy_v5(OPEN_MID)["quotes"])


def test_the_map_and_the_spec_declare_one_box():
    spec = get_action_spec("storage_quote_v5")
    assert spec.truthful_array().tolist() == pytest.approx(list(TRUTHFUL))
    for name, high in zip(spec.action_names, ACTION_BOX_HIGH):
        assert tuple(spec.bounds[name]) == (0.0, high)
