"""Tests for the pluggable observation/action space (rl_spec) integration."""

import torch
import numpy as np
import pytest

from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv
from rl_spec import (
    OBS_V3, OBS_V4, ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2,
    ObservationSpec, ActionSpec, FeatureSpec,
    register_obs_spec, get_obs_spec,
)
from rl_td3 import Actor, save_policy, load_policy


def _make_env(**kwargs):
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    return BiddingEnv(agents, config, rl_agent_names=rl_names, **kwargs)


def test_obs_v4_dims():
    assert OBS_V4.total_dim == 12
    assert OBS_V4.unique_dim == 3
    assert OBS_V4.shared_dim == 9


def test_obs_v4_feature_order_matches_env_layout():
    env = _make_env()
    assert env.obs_spec is OBS_V4
    assert env.get_state_dim() == 12
    assert env.unique_obs_dim == 3
    assert env.shared_obs_dim == 9
    # The env's feature order must be the active spec's, and the price-average
    # slot must be the CAUSAL one: `avg_lmp_hist_norm`, not the retired
    # whole-day `avg_lmp_norm`.
    assert env.obs_spec.feature_order == OBS_V4.feature_order
    assert OBS_V4.feature_order == (
        "load_feat", "re_feat", "soc",
        "last_lmp_norm", "block_pos", "avg_lmp_hist_norm", "slr_norm",
        "avg_other_bid", "bid_std", "ema_dev", "price_trend",
        "pred_lmp_norm")
    assert "avg_lmp_norm" not in OBS_V4.feature_order


def test_the_retired_v3_spec_survives_so_old_policies_can_be_identified():
    """Old checkpoints record `v3_12d`. Keeping it registered is what makes
    `load_policy` raise on the name mismatch rather than silently reinterpret an
    actor trained on a different layout."""
    assert OBS_V3.name == "v3_12d"
    assert OBS_V3.total_dim == 12
    assert "avg_lmp_norm" in OBS_V3.feature_order
    assert OBS_V3 != OBS_V4


def test_observation_spec_dict_round_trip():
    d = OBS_V4.to_dict()
    assert d["name"] == "v4_12d"
    assert d["version"] == 4
    assert d["total_dim"] == 12
    back = ObservationSpec.from_dict(d)
    assert back == OBS_V4


# ---------------------------------------------------------------------------
# No future prices in the observation
# ---------------------------------------------------------------------------

def test_the_observation_ignores_prices_after_the_decision():
    """Change only the day's LATER prices; the observation taken before the
    first decision must not move.

    This is the regression test for the V3 defect. There, `avg_lmp_norm` was
    `mean(self.wholesale)` -- the whole day's realized price -- and the same
    mean was the denominator of `last_lmp_norm` and `pred_lmp_norm` and the
    block-0 value of `last_lmp` (hence of `ema_dev`). Four of the twelve
    features therefore read periods the agent had not reached, and this test
    fails on that contract.
    """
    early = np.full(96, 300.0, dtype=float)
    late = early.copy()
    late[60:] = 900.0            # beyond every window the early blocks clear

    np.random.seed(7)
    obs_a = _make_env().reset(wholesale=early)
    np.random.seed(7)
    obs_b = _make_env().reset(wholesale=late)

    assert set(obs_a) == set(obs_b)
    for name in obs_a:
        np.testing.assert_array_equal(obs_a[name], obs_b[name])


def test_the_episode_wholesale_array_does_not_reach_any_feature():
    """`self.wholesale` may be stored, but no feature may read it.

    Checked directly so it holds at every block, not only at block 0: the
    history and the reference price are the only price inputs.
    """
    import rl_env
    env = _make_env()
    env.reset(wholesale=np.full(96, 100.0, dtype=float))
    agent = env.rl_agents[0]
    before = env._get_agent_obs(agent, 0, None, rl_env.PRICE_REF_CNY_MWH)
    env.wholesale = np.full(96, 5000.0, dtype=float)
    after = env._get_agent_obs(agent, 0, None, rl_env.PRICE_REF_CNY_MWH)
    np.testing.assert_array_equal(before, after)


def test_the_price_reference_is_a_frozen_constant():
    """The normalizer must not be an episode statistic, or the leak returns
    through the denominator even with the feature itself renamed."""
    import rl_env
    assert isinstance(rl_env.PRICE_REF_CNY_MWH, float)
    assert rl_env.PRICE_REF_CNY_MWH > 0
    # It is a module attribute, not an instance one: an episode cannot set it.
    env = _make_env()
    assert not hasattr(env, "PRICE_REF_CNY_MWH"), \
        "the reference price must not be per-episode state"
    env.reset(wholesale=np.full(96, 100.0, dtype=float))
    assert env.wholesale is not None      # stored, just never read as a feature


def test_a_block_zero_observation_carries_no_day_information():
    """The first observation must be reproducible from the frozen constant
    alone, with no history to draw on."""
    import rl_env
    env = _make_env()
    obs = env.reset(wholesale=np.full(96, 12345.0, dtype=float))
    # A wild wholesale level must not show up: the causal average is the
    # constant, so the ratio is exactly 1.0 rather than 12345/420.
    idx = env.obs_spec.feature_order.index("avg_lmp_hist_norm")
    for v in obs.values():
        assert v[idx] == pytest.approx(
            rl_env.PRICE_REF_CNY_MWH / rl_env.PRICE_REF_CNY_MWH)


def test_action_spec_dict_round_trip():
    d = ACTION_BID_OFFER_V1.to_dict()
    assert d["action_names"] == ["bid_mult", "offer_adder"]
    back = ActionSpec.from_dict(d)
    assert back == ACTION_BID_OFFER_V1


def test_action_bounds_reflect_ctor_overrides():
    """The constructor's bid bounds still reach the non-storage action space.

    They no longer set the environment's own bounds: the agents a policy acts
    on are the batteries, and a battery declares prices, so the default spec is
    the storage one. The multiplier bounds still apply to the space a load or a
    prosumer bids in, which is what they always described.
    """
    env = _make_env(bid_mult_low=0.4, bid_mult_high=1.6)
    np.testing.assert_allclose(
        env._legacy_action_spec.bounds["bid_mult"], (0.4, 1.6))
    np.testing.assert_allclose(
        env.get_action_bounds().numpy(),
        [[ACTION_STORAGE_QUOTE_V2.bounds[n][0] for n in
          ACTION_STORAGE_QUOTE_V2.action_names],
         [ACTION_STORAGE_QUOTE_V2.bounds[n][1] for n in
          ACTION_STORAGE_QUOTE_V2.action_names]])


def test_register_obs_spec_duplicate_guard():
    dup = ObservationSpec(
        name="v3_12d", version=99,
        unique_features=(FeatureSpec("a"),), shared_features=())
    with pytest.raises(ValueError):
        register_obs_spec(dup)


def test_get_obs_spec_unknown():
    with pytest.raises(ValueError):
        get_obs_spec("does_not_exist")


def _stub_actor():
    bounds = torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32)
    return Actor(12, 2, bounds)


def test_save_load_policy_with_meta(tmp_path):
    actor = _stub_actor()
    path = str(tmp_path / "Bus5R.pt")
    save_policy(actor, path, obs_spec=OBS_V3, action_spec=ACTION_BID_OFFER_V1)
    loaded = load_policy(path, 12, torch.tensor([[0.3, 0.0], [1.8, 50.0]]),
                         obs_spec=OBS_V3, action_spec=ACTION_BID_OFFER_V1)
    # Network weights survive the round trip (identical forward output).
    obs = torch.randn(1, 12)
    torch.testing.assert_close(loaded(obs), actor(obs))


def test_load_policy_spec_mismatch_raises(tmp_path):
    actor = _stub_actor()
    path = str(tmp_path / "Bus5R.pt")
    save_policy(actor, path, obs_spec=OBS_V3)
    other = ObservationSpec(
        name="v4_15d", version=4,
        unique_features=(FeatureSpec("a"), FeatureSpec("b"),
                         FeatureSpec("c"), FeatureSpec("d")),
        shared_features=tuple(FeatureSpec(f"f{i}") for i in range(11)),
    )
    with pytest.raises(ValueError, match="v3_12d"):
        load_policy(path, 12, torch.tensor([[0.3, 0.0], [1.8, 50.0]]),
                    obs_spec=other)


def test_load_policy_action_spec_mismatch_raises(tmp_path):
    """The check the action-space rename exists for.

    A checkpoint trained on the retired multiplier pair has the same two
    weights as a storage actor, so nothing downstream notices: `load_state_dict`
    succeeds and `bid_mult` is read as a CNY/MWh charge price. Only the spec
    name in the header distinguishes them.
    """
    actor = _stub_actor()
    path = str(tmp_path / "ESS6.pt")
    save_policy(actor, path, obs_spec=OBS_V3, action_spec=ACTION_BID_OFFER_V1)

    with pytest.raises(ValueError, match="bid_offer_v1"):
        load_policy(path, 12, torch.tensor([[0.0, 0.0], [150.0, 300.0]]),
                    obs_spec=OBS_V3, action_spec=ACTION_STORAGE_QUOTE_V2)


def test_load_policy_without_action_spec_skips_that_check(tmp_path):
    """A caller that passes no action spec is not refused on one."""
    actor = _stub_actor()
    path = str(tmp_path / "ESS6.pt")
    save_policy(actor, path, obs_spec=OBS_V3, action_spec=ACTION_BID_OFFER_V1)

    loaded = load_policy(path, 12, torch.tensor([[0.3, 0.0], [1.8, 50.0]]),
                         obs_spec=OBS_V3)
    obs = torch.randn(1, 12)
    torch.testing.assert_close(loaded(obs), actor(obs))


def test_load_legacy_policy_without_meta(tmp_path):
    """A pre-spec .pt (no metadata header) must still load unchanged."""
    actor = _stub_actor()
    path = str(tmp_path / "Bus5R.pt")
    torch.save(actor.state_dict(), path)
    loaded = load_policy(path, 12, torch.tensor([[0.3, 0.0], [1.8, 50.0]]),
                         obs_spec=OBS_V3)
    obs = torch.randn(1, 12)
    torch.testing.assert_close(loaded(obs), actor(obs))
