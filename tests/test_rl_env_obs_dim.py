"""Tests for the compact V1 observation space in rl_env.

The observation was reduced from the original 103-dim (four 24-period
sequences plus scalars) to a 9-dim vector whose first three entries are the
per-agent features (load, re-gen, SOC). MATD3's centralized critic slices the
leading `unique_obs_dim` entries of each other agent's observation, so the
ordering is part of the contract under test.
"""

import numpy as np

from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv


def _make_env(use_differential_reward: bool = False):
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names,
                     use_differential_reward=use_differential_reward)
    return env, rl_names


def test_obs_dim_is_compact_v1():
    env, _ = _make_env()
    assert env.get_state_dim() == 9
    assert env.unique_obs_dim == 3
    assert env.shared_obs_dim == 6


def test_obs_shape_and_leading_per_agent_features():
    env, _ = _make_env()
    obs = env.reset()
    assert len(obs) > 0
    for o in obs.values():
        assert o.shape == (9,)
        assert o.dtype == np.float32
        # Leading entries are the per-agent features: load[0], re_gen[0], soc
        assert o[0] >= 0.0          # load cannot be negative
        assert 0.0 <= o[2] <= 1.0   # SOC in [0, 1]


def test_differential_reward_flag():
    env, _ = _make_env(use_differential_reward=True)
    assert env.use_differential_reward is True
    env2, _ = _make_env(use_differential_reward=False)
    assert env2.use_differential_reward is False
