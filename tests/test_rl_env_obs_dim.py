"""Tests for the compact V2 observation space in rl_env.

The observation was reduced from the original 103-dim (four 24-period
sequences plus scalars) to a 12-dim vector whose first three entries are the
per-agent features (block-mean load, re-gen, SOC) normalized to [0, 1]; the
shared block adds LMP/system indicators (expressed as O(1) ratios of the
day-average price) and the three price-prediction features (EMA deviation, LMP
trend, and a direct next-block price forecast). MATD3's centralized critic
slices the leading `unique_obs_dim` entries of each other agent's observation,
so the ordering is part of the contract under test.
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


def test_obs_dim_is_compact_v2():
    env, _ = _make_env()
    assert env.get_state_dim() == 12
    assert env.unique_obs_dim == 3
    assert env.shared_obs_dim == 9


def test_obs_shape_and_leading_per_agent_features():
    env, _ = _make_env()
    obs = env.reset()
    assert len(obs) > 0
    for o in obs.values():
        assert o.shape == (12,)
        assert o.dtype == np.float32
        # Leading entries are the per-agent features: load, re_gen, soc.
        # Load/RE are block means normalized by the day-peak load -> [0, 1].
        assert 0.0 <= o[0] <= 1.0    # normalized load
        assert 0.0 <= o[1] <= 1.0    # normalized RE generation
        assert 0.0 <= o[2] <= 1.0    # SOC in [0, 1]
        # LMP features are O(1) ratios of the day-average price.
        assert -1.0 <= o[3] <= 3.0   # last LMP / day-average LMP
        assert 0.0 <= o[5] <= 3.0    # day-average LMP / 420
        assert o[6] <= 5.0           # bounded system load/RE ratio
        # Price-prediction features (EMA deviation, LMP trend) are bounded
        # and finite; at reset the forecaster is empty, so both are zero.
        assert -1.0 <= o[9] <= 1.0
        assert -1.0 <= o[10] <= 1.0
        assert np.isfinite(o[9]) and np.isfinite(o[10])
        # Direct next-block price forecast is a bounded ratio of the day
        # average (the default 420 reference at reset is always in range).
        assert -1.0 <= o[11] <= 3.0
        assert np.isfinite(o[11])


def test_differential_reward_flag():
    env, _ = _make_env(use_differential_reward=True)
    assert env.use_differential_reward is True
    env2, _ = _make_env(use_differential_reward=False)
    assert env2.use_differential_reward is False
