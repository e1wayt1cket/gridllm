"""Tests that the MATD3/TD3 trainers report twin-critic disagreement.

The critics already compute both Q estimates per update; surfacing their mean
and absolute gap costs no extra forward pass and makes critic collapse visible
on the training-monitor page. The gap is disagreement in the critic's
normalized reward units (rewards are divided by the buffer's reward std before
the critic loss), so it is not comparable across runs.
"""

import numpy as np
import pytest
import torch

from rl_bidding import MATD3
from rl_td3 import TD3


class _StubEnv:
    """Minimal BiddingEnv surface the MATD3 constructor touches."""

    def __init__(self, n_agents=2, obs_dim=4):
        from types import SimpleNamespace
        self.obs_spec = SimpleNamespace(total_dim=obs_dim)
        self.action_spec = SimpleNamespace(
            action_names=["bid_mult", "offer_adder"])
        self.rl_agents = [SimpleNamespace(name=f"A{i}") for i in range(n_agents)]
        self._obs_dim = obs_dim
        # The centralized critic splits each agent's observation into its
        # unique block and the shared block.
        self.unique_obs_dim = 1
        self.shared_obs_dim = obs_dim - 1

    def get_state_dim(self):
        return self._obs_dim

    def get_action_bounds(self):
        return torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32)


def _filled_matd3(batch_size=8, n_fill=20, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _StubEnv()
    t = MATD3(env, lr=3e-4, batch_size=batch_size)
    for _ in range(n_fill):
        t.buffer.add(
            obs=np.random.randn(2, 4).astype(np.float32),
            action=np.random.randn(2, 2).astype(np.float32),
            reward=np.random.randn(2).astype(np.float32),
            next_obs=np.random.randn(2, 4).astype(np.float32),
            done=np.zeros(2, dtype=np.float32))
    return t


def test_matd3_update_reports_q_statistics():
    info = _filled_matd3().update()

    assert "q1_mean" in info and "q2_mean" in info and "q_gap" in info
    assert np.isfinite(info["q1_mean"]) and np.isfinite(info["q2_mean"])
    # The gap is a magnitude, never negative.
    assert info["q_gap"] >= 0.0
    # The gap is the mean absolute disagreement, so it must bound the gap
    # between the two means.
    assert abs(info["q1_mean"] - info["q2_mean"]) <= info["q_gap"] + 1e-6


def test_matd3_update_reports_absent_q_stats_before_the_buffer_fills():
    env = _StubEnv()
    info = MATD3(env, batch_size=8).update()
    assert info["critic_loss"] is None
    assert info.get("q_gap") is None


def _filled_td3(batch_size=8, n_fill=20, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t = TD3(4, 2, torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32),
            lr=3e-4, batch_size=batch_size)
    for _ in range(n_fill):
        t.buffer.add(
            obs=np.random.randn(4).astype(np.float32),
            action=np.random.randn(2).astype(np.float32),
            reward=np.float32(np.random.randn()),
            next_obs=np.random.randn(4).astype(np.float32),
            done=np.float32(0.0))
    return t


def test_td3_update_reports_q_statistics():
    info = _filled_td3().update()
    assert "q1_mean" in info and "q2_mean" in info and "q_gap" in info
    assert info["q_gap"] >= 0.0
    assert abs(info["q1_mean"] - info["q2_mean"]) <= info["q_gap"] + 1e-6


def test_td3_update_reports_absent_q_stats_before_the_buffer_fills():
    t = TD3(4, 2, torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32),
            batch_size=8)
    info = t.update()
    assert info["critic_loss"] is None
    assert info.get("q_gap") is None
