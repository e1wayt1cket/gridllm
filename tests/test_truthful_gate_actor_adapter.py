# test_truthful_gate_actor_adapter.py
"""The gate policy adapter: exact truth, legal quotes, ST gradient, consistency."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import truthful_gate_actor_adapter as ga


def test_evaluation_gate_closed_is_exactly_truthful():
    out = torch.tensor([[-1.0, -5.0, 3.0, 9.0]])
    result = ga.forward_policy(out, train=False)
    assert result["quotes"].tolist() == [[0.0, 0.0]]
    assert result["gates"].tolist() == [[0.0, 0.0]]


def test_evaluation_open_gate_is_legal():
    out = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    result = ga.forward_policy(out, train=False)
    assert result["quotes"][0, 0].item() == 75.0
    assert result["quotes"][0, 1].item() == 150.0
    assert (result["quotes"] >= 0).all()
    assert result["quotes"][0, 0] <= ga.LEG_HIGHS[0]
    assert result["quotes"][0, 1] <= ga.LEG_HIGHS[1]


def test_training_sampling_is_reproducible_with_a_generator():
    out = torch.tensor([[0.0, 0.0, 0.0, 0.0]] * 8)
    a = ga.forward_policy(out, train=True,
                          generator=torch.Generator().manual_seed(3),
                          temperature=2.0)
    b = ga.forward_policy(out, train=True,
                          generator=torch.Generator().manual_seed(3),
                          temperature=2.0)
    assert torch.equal(a["hard_gates"], b["hard_gates"])
    assert set(torch.unique(a["hard_gates"]).tolist()) <= {0.0, 1.0}


def test_the_gate_gradient_is_the_biased_surrogate():
    out = torch.tensor([[-10.0, -10.0, 0.0, 0.0]], requires_grad=True)
    result = ga.forward_policy(
        out, train=True, generator=torch.Generator().manual_seed(0))
    assert result["hard_gates"].tolist() == [[0.0, 0.0]]
    result["quotes"].sum().backward()
    grad = out.grad
    # The hard forward is zero; the surrogate still gives the gate logits a
    # gradient through prob*(1-prob), scaled by the price branch.
    assert grad[0, 0].item() != 0.0
    # Closed gates give the price branch no gradient.
    assert grad[0, 2].item() == 0.0


def test_an_open_gate_gives_the_price_branch_a_gradient():
    out = torch.tensor([[2.0, 2.0, 0.0, 0.0]], requires_grad=True)
    result = ga.forward_policy(out, train=False)
    result["quotes"].sum().backward()
    assert out.grad[0, 2].item() > 0.0
    assert out.grad[0, 3].item() > 0.0


def test_the_target_actor_is_deterministic_and_matches_evaluation():
    out = torch.tensor([[0.4, -0.4, 1.0, -1.0]])
    first = ga.target_quotes(out)
    second = ga.target_quotes(out)
    assert torch.equal(first, second)
    assert torch.equal(first, ga.forward_policy(out, train=False)["quotes"])


def test_the_critic_sees_two_quotes_not_the_gate_probability():
    out = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
    action = ga.critic_action(out)
    assert action.shape == (1, 2)
    assert action[0, 0].item() == 75.0
    assert action[0, 1].item() == 0.0


def test_the_fleet_mode_is_base_four():
    assert ga.fleet_mode([0, 0, 0, 0]) == 0
    assert ga.fleet_mode([3, 3, 3, 3]) == 255
    assert ga.fleet_mode([1, 0, 2, 0]) == 1 + 2 * 16


def test_bad_policy_shape_is_refused():
    try:
        ga.forward_policy(torch.zeros(1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("a three-number policy vector must be refused")
