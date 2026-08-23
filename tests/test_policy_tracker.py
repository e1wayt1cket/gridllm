"""Tests for in-training evaluation and policy tracking (rl_training)."""

import numpy as np
import torch
import pytest

from models import MarketConfig
from scenarios import get_scenario
from rl_td3 import Actor
from rl_training import deterministic_fleet_episode, PolicyTracker


def _stub_actor(obs_dim=12):
    bounds = torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32)
    return Actor(obs_dim, 2, bounds)


class _StubEnv:
    """Minimal BiddingEnv-like stub for episode aggregation tests."""

    def __init__(self, rl_names):
        self.rl_names = rl_names
        self.obs = {nm: np.zeros(12, dtype=np.float32) for nm in rl_names}
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return dict(self.obs)

    def step(self, actions):
        self.step_count += 1
        rewards = {nm: 10.0 for nm in self.rl_names}
        info = {"welfare": 100.0, "re_rate": 50.0}
        done = self.step_count >= 1
        return dict(self.obs), rewards, done, info


def test_deterministic_fleet_episode_aggregates():
    rl_names = ["Bus5R", "Bus6R"]
    env = _StubEnv(rl_names)
    actors = {"Bus5R": _stub_actor(), "Bus6R": _stub_actor()}
    out = deterministic_fleet_episode(env, actors, rl_names)
    # One step (done after 1), two agents each +10 -> mean_reward 10, welfare 100.
    assert out["mean_reward"] == pytest.approx(10.0)
    assert out["welfare"] == pytest.approx(100.0)
    assert out["re_rate"] == pytest.approx(50.0)
    assert out["profits"] == {"Bus5R": 10.0, "Bus6R": 10.0}


def test_deterministic_fleet_episode_missing_actor_uses_default():
    rl_names = ["Bus5R", "Bus6R"]
    env = _StubEnv(rl_names)
    # Bus6R has no actor -> falls back to truthful default [1.0, 0.0].
    actors = {"Bus5R": _stub_actor()}
    out = deterministic_fleet_episode(env, actors, rl_names)
    assert out["mean_reward"] == pytest.approx(10.0)


def test_tracker_build_eval_env():
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    tracker = PolicyTracker("policies/x", rl_names, obs_spec=None)
    env = tracker.build_eval_env(config)
    assert [a.name for a in env.rl_agents] == rl_names
    assert env.use_differential_reward is False


def test_tracker_best_and_last_with_early_stop(tmp_path):
    rl_names = ["Bus5R", "Bus6R"]
    save_dir = str(tmp_path / "policies")
    tracker = PolicyTracker(
        save_dir, rl_names, early_stopping_steps=2,
        early_stopping_threshold=0.01)
    actors = {nm: _stub_actor() for nm in rl_names}

    # First eval: improvement -> saves best/, not enough window for early stop.
    improving = {"mean_reward": 50.0, "welfare": 1000.0, "re_rate": 50.0}
    assert tracker.compare_and_save_policies(1, actors, improving) is False
    import os
    assert os.path.exists(os.path.join(tracker.best_dir, "Bus5R.pt"))
    assert tracker.best_episode == 1

    # Second eval: flat window -> improvement below threshold -> early stop.
    flat = {"mean_reward": 50.0, "welfare": 1000.0, "re_rate": 50.0}
    assert tracker.compare_and_save_policies(2, actors, flat) is True
    assert tracker.early_stopped is True
    assert os.path.exists(os.path.join(tracker.last_dir, "Bus5R.pt"))


def test_tracker_no_early_stop_when_disabled(tmp_path):
    rl_names = ["Bus5R"]
    tracker = PolicyTracker(str(tmp_path / "p"), rl_names,
                            early_stopping_steps=0)
    actors = {"Bus5R": _stub_actor()}
    flat = {"mean_reward": 10.0, "welfare": 100.0, "re_rate": 50.0}
    # Even flat metrics never trigger early stop when disabled.
    assert tracker.compare_and_save_policies(1, actors, flat) is False
    assert tracker.compare_and_save_policies(2, actors, flat) is False
