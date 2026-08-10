"""Unit tests for the deviation penalty applied directly to the TD3 actor loss."""

import random

import numpy as np
import pytest
import torch

from rl_td3 import TD3

OBS_DIM = 4
ACT_DIM = 2
BOUNDS = torch.tensor([[0.6, 0.0], [1.4, 50.0]], dtype=torch.float32)


def _make_td3(bid_pen: float = 0.0, offer_pen: float = 0.0) -> TD3:
    return TD3(OBS_DIM, ACT_DIM, BOUNDS,
               bid_dev_penalty=bid_pen, offer_dev_penalty=offer_pen,
               batch_size=8, buffer_capacity=64, start_steps=0)


def _fill_buffer(td3: TD3, action: np.ndarray, n: int = 16):
    for _ in range(n):
        td3.buffer.add(np.zeros(OBS_DIM, dtype=np.float32),
                       action.astype(np.float32),
                       0.0,
                       np.zeros(OBS_DIM, dtype=np.float32),
                       False)


def test_td3_accepts_deviation_penalty_args():
    td3 = _make_td3(bid_pen=5.0, offer_pen=0.5)
    assert td3.bid_dev_penalty == 5.0
    assert td3.offer_dev_penalty == 0.5


def test_actor_loss_increases_with_bid_penalty():
    """A nonzero bid deviation penalty raises the actor loss, all else equal."""
    random.seed(7)
    torch.manual_seed(7)
    baseline = _make_td3(bid_pen=0.0)
    _fill_buffer(baseline, np.array([1.4, 0.0]))
    loss_baseline = baseline.update()["actor_loss"]
    assert loss_baseline is not None

    random.seed(7)
    torch.manual_seed(7)
    penalized = _make_td3(bid_pen=5.0)
    _fill_buffer(penalized, np.array([1.4, 0.0]))
    loss_penalized = penalized.update()["actor_loss"]

    # L2 penalty on the pre-tanh logit is strictly additive and non-negative.
    assert loss_penalized > loss_baseline


def test_actor_loss_increases_with_offer_penalty():
    random.seed(11)
    torch.manual_seed(11)
    baseline = _make_td3(offer_pen=0.0)
    _fill_buffer(baseline, np.array([1.0, 50.0]))
    loss_baseline = baseline.update()["actor_loss"]

    random.seed(11)
    torch.manual_seed(11)
    penalized = _make_td3(offer_pen=0.5)
    _fill_buffer(penalized, np.array([1.0, 50.0]))
    loss_penalized = penalized.update()["actor_loss"]

    assert loss_penalized > loss_baseline


def test_forward_logits_matches_forward():
    """forward() and forward_logits() return the same action."""
    actor = _make_td3().actor
    obs = torch.zeros(1, OBS_DIM)
    logits, action = actor.forward_logits(obs)
    assert torch.allclose(action, actor(obs), atol=1e-6)
    assert logits.shape == (1, ACT_DIM)
    # action = mid + tanh(logit) * half
    mid = (BOUNDS[0] + BOUNDS[1]) / 2.0
    half = (BOUNDS[1] - BOUNDS[0]) / 2.0
    expected = mid + torch.tanh(logits[0]) * half
    assert torch.allclose(action[0], expected, atol=1e-6)


def test_logit_penalty_survives_saturation():
    """The L2 logit penalty keeps a nonzero gradient even when the actor is
    saturated at the action bound, where d(tanh)/dx would vanish."""
    random.seed(3)
    torch.manual_seed(3)
    td3 = _make_td3(bid_pen=5.0, offer_pen=0.0)
    # Drive the actor's pre-tanh logits far from zero (saturated).
    with torch.no_grad():
        for p in td3.actor.net[4].parameters():
            p.fill_(50.0)
    obs = torch.zeros(8, OBS_DIM)
    logits, _ = td3.actor.forward_logits(obs)
    assert torch.all(torch.abs(logits[:, 0]) > 30.0)  # tanh fully saturated
    loss = td3.bid_dev_penalty * (logits[:, 0] ** 2).mean()
    loss.backward()
    grads = [p.grad for p in td3.actor.net[4].parameters()
             if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
