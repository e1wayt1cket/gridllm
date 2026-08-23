"""Tests for the pluggable observation/action space (rl_spec) integration."""

import torch
import numpy as np
import pytest

from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv
from rl_spec import (
    OBS_V3, ACTION_BID_OFFER_V1,
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


def test_obs_v3_dims():
    assert OBS_V3.total_dim == 12
    assert OBS_V3.unique_dim == 3
    assert OBS_V3.shared_dim == 9


def test_obs_v3_feature_order_matches_env_layout():
    env = _make_env()
    assert env.obs_spec is OBS_V3
    assert env.get_state_dim() == 12
    assert env.unique_obs_dim == 3
    assert env.shared_obs_dim == 9
    # env feature order must reproduce the historical vector layout.
    assert OBS_V3.feature_order == (
        "load_feat", "re_feat", "soc",
        "last_lmp_norm", "block_pos", "avg_lmp_norm", "slr_norm",
        "avg_other_bid", "bid_std", "ema_dev", "price_trend",
        "pred_lmp_norm")


def test_observation_spec_dict_round_trip():
    d = OBS_V3.to_dict()
    assert d["name"] == "v3_12d"
    assert d["total_dim"] == 12
    back = ObservationSpec.from_dict(d)
    assert back == OBS_V3


def test_action_spec_dict_round_trip():
    d = ACTION_BID_OFFER_V1.to_dict()
    assert d["action_names"] == ["bid_mult", "offer_adder"]
    back = ActionSpec.from_dict(d)
    assert back == ACTION_BID_OFFER_V1


def test_action_bounds_reflect_ctor_overrides():
    env = _make_env(bid_mult_low=0.4, bid_mult_high=1.6)
    bounds = env.get_action_bounds().numpy()
    np.testing.assert_allclose(bounds, [[0.4, 0.0], [1.6, 50.0]])


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


def test_load_legacy_policy_without_meta(tmp_path):
    """A pre-spec .pt (no metadata header) must still load unchanged."""
    actor = _stub_actor()
    path = str(tmp_path / "Bus5R.pt")
    torch.save(actor.state_dict(), path)
    loaded = load_policy(path, 12, torch.tensor([[0.3, 0.0], [1.8, 50.0]]),
                         obs_spec=OBS_V3)
    obs = torch.randn(1, 12)
    torch.testing.assert_close(loaded(obs), actor(obs))
