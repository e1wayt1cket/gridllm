"""The deviation penalty: what it measures, and how hard it pulls.

The penalty is defined on each action's distance from its own truthful point,
normalized by that action's box width. That normalization is the whole point:
it makes the two legs of a quote pair the same quantity, and it is what keeps
the penalty from pinning the policy at the truthful corner.
"""

import random

import numpy as np
import pytest
import torch

from rl_td3 import TD3, deviation_penalty, normalized_deviation

OBS_DIM = 4
ACT_DIM = 2
BOUNDS = torch.tensor([[0.6, 0.0], [1.4, 50.0]], dtype=torch.float32)

# BOUNDS is a load/prosumer box, whose own truthful point is (1.0, 0.0) by
# ACTION_BID_OFFER_V1 -- inside the box, unlike a battery's (0, 0) which is a
# corner of its own.
TRUTHFUL = np.array([1.0, 0.0], dtype=np.float32)

# The storage box, whose truthful point is the box corner. The interesting
# properties only show up here.
STORAGE_BOUNDS = torch.tensor([[0.0, 0.0], [150.0, 300.0]],
                              dtype=torch.float32)
STORAGE_TRUTHFUL = np.array([0.0, 0.0], dtype=np.float32)


def _make_td3(pen: float = 0.0) -> TD3:
    return TD3(OBS_DIM, ACT_DIM, BOUNDS, truthful_action=TRUTHFUL,
               dev_penalty=pen, batch_size=8, buffer_capacity=64,
               start_steps=0)


def _fill_buffer(td3: TD3, action: np.ndarray, n: int = 16):
    for _ in range(n):
        td3.buffer.add(np.zeros(OBS_DIM, dtype=np.float32),
                       action.astype(np.float32),
                       0.0,
                       np.zeros(OBS_DIM, dtype=np.float32),
                       False)


# ---------------------------------------------------------------------------
# What the deviation is
# ---------------------------------------------------------------------------

def test_the_deviation_is_the_distance_from_truthful_over_the_box_width():
    """The definition, stated as the arithmetic it is.

    For a truthful point at the box edge and a squashed action,
    `d = (1 + tanh(logit)) / 2`: zero at the edge, one at the far edge.
    """
    logits = torch.zeros(1, 2)
    d = normalized_deviation(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS)
    # tanh(0) = 0 -> the action sits at the box midpoint -> d = 0.5.
    assert torch.allclose(d, torch.full((1, 2), 0.5), atol=1e-6)

    far = torch.full((1, 2), 30.0)
    assert torch.all(normalized_deviation(
        far, STORAGE_TRUTHFUL, STORAGE_BOUNDS) > 0.99)

    near = torch.full((1, 2), -30.0)
    assert torch.all(normalized_deviation(
        near, STORAGE_TRUTHFUL, STORAGE_BOUNDS) < 0.01)


def test_the_deviation_is_bounded_in_the_unit_interval():
    """Whatever the logit, d stays in [0, 1] for a truthful point in the box.

    That bound is what lets the weight be chosen against a normalized reward:
    the penalty can never exceed the weight, however far the policy wanders.
    """
    logits = torch.linspace(-50.0, 50.0, 101).unsqueeze(1).repeat(1, 2)
    d = normalized_deviation(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS)
    assert float(d.min()) >= 0.0
    assert float(d.max()) <= 1.0 + 1e-6


def test_the_deviation_is_monotone_in_the_logit():
    """Further from truthful is always further, never less."""
    logits = torch.linspace(-20.0, 20.0, 200).unsqueeze(1).repeat(1, 2)
    d = normalized_deviation(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS)
    assert torch.all(d[1:] >= d[:-1] - 1e-7)


def test_the_deviation_is_the_same_on_both_legs_at_the_same_fraction():
    """The charge box is 150 wide and the discharge box 300.

    A raw deviation in CNY/MWh would call a 30 CNY/MWh move on the discharge
    leg twice the move it is on the charge leg. Normalized, the same fraction
    of each box is the same deviation, which is what makes one coefficient
    legitimate for both.
    """
    # 20% of each box, expressed as the logit that lands there.
    # d = (1 + tanh(x))/2 = 0.2  ->  tanh(x) = -0.6
    x = float(torch.atanh(torch.tensor(-0.6)))
    logits = torch.full((1, 2), x)
    d = normalized_deviation(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS)
    assert d[0, 0] == pytest.approx(d[0, 1], abs=1e-6)
    assert float(d[0, 0]) == pytest.approx(0.2, abs=1e-5)


def test_the_deviation_does_not_depend_on_the_units_the_box_is_in():
    """Scaling a box changes the prices, not the deviation.

    The same fractional position in a box twice as wide is the same deviation,
    so a penalty calibrated on one box transfers to a rescaled one.
    """
    x = float(torch.atanh(torch.tensor(0.4)))     # d = 0.7
    logits = torch.full((1, 2), x)
    d_small = normalized_deviation(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS)
    scaled = STORAGE_BOUNDS * 7.0
    d_big = normalized_deviation(logits, STORAGE_TRUTHFUL, scaled)
    assert torch.allclose(d_small, d_big, atol=1e-6)


# ---------------------------------------------------------------------------
# How it enters the actor loss
# ---------------------------------------------------------------------------

def test_the_penalty_is_the_weight_times_the_squared_deviation():
    logits = torch.zeros(4, 2)
    p = deviation_penalty(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 0.25)
    # d = 0.5 on both actions -> 0.25 * (0.25 + 0.25) = 0.125
    assert float(p) == pytest.approx(0.125, abs=1e-6)


def test_a_zero_weight_contributes_nothing():
    logits = torch.randn(8, 2) * 5.0
    p = deviation_penalty(logits, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 0.0)
    assert float(p) == 0.0


def test_the_penalty_gradient_vanishes_at_the_truthful_corner():
    """The property that stops it pinning the policy there.

    A penalty with a nonzero gradient at the truthful point would hold the
    policy against it. This one's pull weakens as the policy approaches
    truthful, because the deviation itself is going to zero.
    """
    at_corner = torch.full((4, 2), -30.0, requires_grad=True)
    p = deviation_penalty(at_corner, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 1.0)
    p.backward()
    assert float(at_corner.grad.abs().max()) < 1e-6

    mid = torch.zeros(4, 2, requires_grad=True)
    deviation_penalty(mid, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 1.0).backward()
    assert float(mid.grad.abs().max()) > 0.1


def test_the_penalty_gradient_is_bounded():
    """Unlike the logit-quadratic form, the pull cannot grow without bound.

    For one sample the penalty is `weight * sum_i d_i^2`. With a truthful point
    at a box edge, `d = (1 + t)/2` for `t = tanh(x)`, so
    `d(d^2)/dx = (1 + t)(1 - t^2)/2`, maximised at `t = 1/3` for the value
    `16/27`. It falls to zero at both edges -- so however large the weight, the
    regularizer cannot dominate the economic term arbitrarily far from
    truthful, which is exactly what pinned the old penalty at the corner.

    Checked one sample at a time: the penalty averages over the batch, so the
    per-element pull in a batch of B is this divided by B, the same scaling
    `-min(q1, q2).mean()` carries.
    """
    worst = 0.0
    for x in np.linspace(-20.0, 20.0, 2001):
        one = torch.full((1, 2), float(x), requires_grad=True)
        deviation_penalty(one, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 1.0).backward()
        worst = max(worst, float(one.grad.abs().max()))
    assert worst == pytest.approx(16.0 / 27.0, abs=1e-3)


def test_the_penalty_is_a_batch_mean_like_the_economic_term():
    """Both terms average over the batch, so the weight is batch-size free.

    Replicating a sample must not change what the penalty costs, or the
    coefficient calibrated at one batch size would mean something else at
    another.
    """
    one = torch.zeros(1, 2)
    many = torch.zeros(64, 2)
    assert float(deviation_penalty(
        one, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 1.0)) == pytest.approx(
        float(deviation_penalty(
            many, STORAGE_TRUTHFUL, STORAGE_BOUNDS, 1.0)), rel=1e-6)


# ---------------------------------------------------------------------------
# Through the trainer
# ---------------------------------------------------------------------------

def test_td3_accepts_the_deviation_penalty_arg():
    assert _make_td3(pen=0.25).dev_penalty == 0.25


def test_actor_loss_rises_with_the_penalty():
    random.seed(7)
    torch.manual_seed(7)
    baseline = _make_td3(pen=0.0)
    _fill_buffer(baseline, np.array([1.4, 0.0]))
    loss_baseline = baseline.update()["actor_loss"]
    assert loss_baseline is not None

    random.seed(7)
    torch.manual_seed(7)
    penalized = _make_td3(pen=0.5)
    _fill_buffer(penalized, np.array([1.4, 0.0]))
    assert penalized.update()["actor_loss"] > loss_baseline


def test_forward_logits_matches_forward():
    actor = _make_td3().actor
    obs = torch.zeros(1, OBS_DIM)
    logits, action = actor.forward_logits(obs)
    assert torch.allclose(action, actor(obs), atol=1e-6)
    assert logits.shape == (1, ACT_DIM)
    mid = (BOUNDS[0] + BOUNDS[1]) / 2.0
    half = (BOUNDS[1] - BOUNDS[0]) / 2.0
    assert torch.allclose(action[0], mid + torch.tanh(logits[0]) * half,
                          atol=1e-6)


def test_actor_loss_is_invariant_to_a_uniform_reward_rescale():
    """The weight is dimensionless, so rescaling the reward must not re-tune it.

    The reward is divided by its running std before the critic target, and the
    penalty is in those same units rather than scaled by the reward a second
    time.
    """
    def actor_loss_at(scale: float) -> float:
        random.seed(5)
        torch.manual_seed(5)
        td3 = _make_td3(pen=0.25)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(16):
            td3.buffer.add(obs, np.array([1.2, 20.0], dtype=np.float32),
                           scale * (1.0 + i % 3), obs, False)
        loss = td3.update()["actor_loss"]
        assert loss is not None
        return loss

    assert actor_loss_at(1.0) == pytest.approx(actor_loss_at(100.0), rel=1e-5)


def test_td3_refuses_to_invent_a_deviation_reference():
    """No truthful action is an error, not a default.

    The old fallback was the box midpoint, which for the storage box means
    "charge at 75, discharge at 150" -- the region the thin-margin batteries
    cannot trade in.
    """
    with pytest.raises(ValueError, match="truthful action"):
        TD3(OBS_DIM, ACT_DIM, BOUNDS, dev_penalty=0.25,
            batch_size=8, buffer_capacity=64, start_steps=0)
