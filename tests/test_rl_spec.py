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


# ---------------------------------------------------------------------------
# Causality regression gaps: mid-block multi-channel perturbation, history
# update timing, cross-episode reset, load/RE normalization, opponent-bid
# timing, and an end-to-end future-price shock. These exercise the
# observation at blocks other than 0, through the real clearing loop.
# ---------------------------------------------------------------------------

def _quote_actions(env, charge, discharge):
    """The same (charge_price, discharge_price) pair for every RL agent, in
    the agent's own spec order (truthful values fill anything else)."""
    vec = {"charge_price": charge, "discharge_price": discharge}
    return {a.name: np.array(
        [vec.get(n, env.spec_for(a).truthful[n])
         for n in env.spec_for(a).action_names], dtype=float)
            for a in env.rl_agents}


def _recompute_obs(env, name):
    """Recompute one agent's observation for the env's current block through
    the same internal path step() uses, from the env's current state."""
    import rl_env
    agent = next(a for a in env.rl_agents if a.name == name)
    hist = np.array(list(env.hist_lmp)) if env.hist_lmp else None
    avg = float(np.mean(hist)) if hist is not None and hist.size \
        else rl_env.PRICE_REF_CNY_MWH
    slr = env._compute_system_load_re_ratio(env.block_idx * rl_env.BLOCK_SIZE)
    avg_oth, bid_std = env._compute_opponent_features(name)
    return env._get_agent_obs(agent, env.block_idx, hist, avg, slr,
                              avg_oth, bid_std)


def test_the_block_zero_price_features_are_canonical():
    """No history at block 0: every price-level feature is the frozen
    reference (ratio exactly 1.0) and every deviation feature is 0.0 -- for
    any wholesale level, since none of it may be read."""
    env = _make_env()
    obs = env.reset(wholesale=np.full(96, 777.0, dtype=float))
    order = env.obs_spec.feature_order
    for v in obs.values():
        assert v[order.index("last_lmp_norm")] == pytest.approx(1.0)
        assert v[order.index("avg_lmp_hist_norm")] == pytest.approx(1.0)
        assert v[order.index("pred_lmp_norm")] == pytest.approx(1.0)
        assert v[order.index("ema_dev")] == pytest.approx(0.0)
        assert v[order.index("price_trend")] == pytest.approx(0.0)


def test_a_mid_block_observation_ignores_every_future_channel():
    """The block-3 decision must be a function of blocks 0-2 alone.

    Two identically seeded envs step three blocks with identical quotes; then
    env B's future is rewritten -- later load/PV forecast slices and the whole
    episode wholesale array -- and its recomputed observation must equal A's.
    A sanity leg proves the observation is not deaf: rewriting the current
    block's load forecast moves it."""
    def _advance(env, n_blocks):
        obs = env.reset(wholesale=np.full(96, 300.0, dtype=float))
        for _ in range(n_blocks):
            obs, _, done, _ = env.step(_quote_actions(env, 40.0, 200.0))
            assert not done
        return obs

    np.random.seed(11)
    env_a = _make_env()
    np.random.seed(11)
    env_b = _make_env()
    obs_a = _advance(env_a, 3)
    obs_b = _advance(env_b, 3)
    for name in obs_a:
        np.testing.assert_array_equal(obs_a[name], obs_b[name])

    import rl_env
    k = env_b.block_idx                 # the block the observation decides
    cut = (k + 1) * rl_env.BLOCK_SIZE   # strictly future periods
    for a in env_b.all_agents:
        a.load_forecast[cut:] = 1.0e6
        a.pv_forecast[cut:] = 1.0e6
        wf = getattr(a, "wind_forecast", None)
        if a.has_wind and wf is not None:
            wf[cut:] = 1.0e6
    env_b.wholesale = np.full(96, 9999.0, dtype=float)
    for name in obs_a:
        np.testing.assert_array_equal(obs_a[name], _recompute_obs(env_b, name))

    # Sanity: the current block's load forecast feeds the system ratio, so a
    # broken test would be caught -- the equality above is not vacuous. Zero
    # the whole population's current-block load (the ratio channel is clipped
    # at 5.0, so a mere rescale can hide inside the saturation).
    for a in env_b.all_agents:
        a.load_forecast[k * rl_env.BLOCK_SIZE:cut] = 0.0
    probe = _recompute_obs(env_b, env_b.rl_agents[0].name)
    assert not np.array_equal(obs_a[env_b.rl_agents[0].name], probe)


def test_the_lmp_history_enters_the_observation_only_after_clearing():
    """Block 0 decides with an empty history; block 1's observation carries
    exactly what block 0's clearing published -- every appended value is a
    per-period nodal LMP from that clearing, and the price features agree
    with the history."""
    import rl_env
    env = _make_env()
    obs0 = env.reset(wholesale=np.full(96, 300.0, dtype=float))
    assert len(env.hist_lmp) == 0
    order = env.obs_spec.feature_order
    for v in obs0.values():
        assert v[order.index("last_lmp_norm")] == pytest.approx(1.0)

    obs1, _, done, _ = env.step(_quote_actions(env, 40.0, 200.0))
    assert not done
    result = env._last_result
    assert result is not None, "the clearing must succeed for the timing test"
    n_commit = rl_env.BLOCK_SIZE
    allowed = {float(result["lmp"][d, a.bus])
               for a in env.rl_agents for d in range(n_commit)}
    assert len(env.hist_lmp) == n_commit * len(env.rl_agents)
    assert all(float(v) in allowed for v in env.hist_lmp)

    hist = np.array(list(env.hist_lmp))
    for v in obs1.values():
        assert v[order.index("last_lmp_norm")] == pytest.approx(
            float(np.clip(hist[-1] / rl_env.PRICE_REF_CNY_MWH, -1.0, 3.0)))
        assert v[order.index("avg_lmp_hist_norm")] == pytest.approx(
            float(np.clip(hist.mean() / rl_env.PRICE_REF_CNY_MWH, 0.0, 3.0)))


def test_reset_leaves_no_trace_of_the_previous_episode():
    """A day traded with aggressive quotes must not tint the next day's first
    observation: block 0 after reset is bit-identical to a fresh env's, even
    though the wholesale level differs between the two days."""
    np.random.seed(5)
    env = _make_env()
    env.reset(wholesale=np.full(96, 300.0, dtype=float))
    env.step(_quote_actions(env, 130.0, 290.0))
    env.step(_quote_actions(env, 10.0, 60.0))
    assert env._prev_block_actions is not None      # the day really traded

    np.random.seed(5)
    fresh = _make_env()
    obs_fresh = fresh.reset(wholesale=np.full(96, 640.0, dtype=float))
    obs_day2 = env.reset(wholesale=np.full(96, 640.0, dtype=float))

    assert env._prev_block_actions is None
    assert env._prev_block_index is None
    assert len(env.hist_lmp) == 0
    assert all(fc._ema is None and not fc._history
               for fc in env.forecasters.values())
    assert not env.prev_soc
    assert set(obs_day2) == set(obs_fresh)
    for name in obs_fresh:
        np.testing.assert_array_equal(obs_day2[name], obs_fresh[name])


def test_load_and_re_normalization_reads_only_available_forecasts():
    """DA stage: realized curves are invisible to the observation; only the
    decision block's own forecast slice feeds load_feat/re_feat; the
    denominator is the agent's own day-ahead forecast peak, cached at reset.
    """
    import rl_env
    env = _make_env()
    env.reset(wholesale=np.full(96, 300.0, dtype=float))
    agent = env.rl_agents[0]

    def _obs0():
        return env._get_agent_obs(
            agent, 0, None, rl_env.PRICE_REF_CNY_MWH,
            env._compute_system_load_re_ratio(0),
            *env._compute_opponent_features(agent.name))

    before = _obs0().copy()

    # Realized channels: a day-ahead observation must not read them at all.
    for a in env.all_agents:
        for attr in ("load_real", "pv_real", "wind_real"):
            arr = getattr(a, attr, None)
            if arr is not None:
                arr[:] = 1.0e6
    np.testing.assert_array_equal(before, _obs0())

    # Future forecast slices are equally invisible.
    for a in env.all_agents:
        a.load_forecast[rl_env.BLOCK_SIZE:] = 1.0e6
        a.pv_forecast[rl_env.BLOCK_SIZE:] = 1.0e6
    np.testing.assert_array_equal(before, _obs0())

    # Formula pin on the agent with the most load: block mean over its own
    # forecast day-peak (the peak was cached before the mutations above). The
    # heaviest load sits on a non-RL bus, which owns no price forecaster --
    # register a fresh one so the shared feature path runs; it only feeds
    # features this pin does not assert.
    from price_forecaster import NodalPriceForecaster
    heavy = max(env.all_agents, key=lambda a: env._agent_peaks[a.name][0])
    peak = env._agent_peaks[heavy.name][0]
    assert peak > 0, "the formula pin needs one agent with real load"
    env.forecasters.setdefault(heavy.name,
                               NodalPriceForecaster(alpha=0.3, history_len=24))
    feats = env._compute_obs_features(
        heavy, 0, None, rl_env.PRICE_REF_CNY_MWH, 1.0)
    expected = float(np.clip(
        np.mean(heavy.load_forecast[0:rl_env.BLOCK_SIZE]) / peak, 0.0, 1.0))
    assert feats["load_feat"] == pytest.approx(expected)


def test_opponent_features_carry_the_previous_blocks_quotes():
    """Synchronous clearing: the block-k observation may only know quotes
    submitted for block k-1. Block 0 sees the default declarations; block 2
    must reflect block 1's quotes -- not block 0's."""
    import rl_env
    env = _make_env()
    order = env.obs_spec.feature_order
    i_oth, i_std = order.index("avg_other_bid"), order.index("bid_std")
    BS = rl_env.BLOCK_SIZE

    def _expected(src, blk, observer):
        t0 = blk * BS
        others, everyone = [], []
        for a in env.all_agents:
            spec = env.spec_for(a)
            name = spec.action_names[0]
            lo, hi = spec.bounds[name]
            v = float(np.mean(src[a.name][name][t0:t0 + BS]))
            s = (v - lo) / (hi - lo) if hi > lo else 0.0
            everyone.append(s)
            if a.name != observer:
                others.append(s)
        return float(np.mean(others)), float(np.std(everyone))

    obs0 = env.reset(wholesale=np.full(96, 300.0, dtype=float))
    defaults = {nm: {k: v.copy() for k, v in act.items()}
                for nm, act in env.current_actions.items()}
    obs1, _, _, _ = env.step(_quote_actions(env, 30.0, 210.0))
    obs2, _, _, _ = env.step(_quote_actions(env, 120.0, 280.0))
    final_actions = env.current_actions

    for name in obs0:                            # block 0: defaults
        e_oth, e_std = _expected(defaults, 0, name)
        assert obs0[name][i_oth] == pytest.approx(e_oth)
        assert obs0[name][i_std] == pytest.approx(e_std)
    for name in obs1:                            # block 1 sees block 0
        e_oth, _ = _expected(final_actions, 0, name)
        assert obs1[name][i_oth] == pytest.approx(e_oth)
    for name in obs2:                            # block 2 sees block 1
        e_oth, e_std = _expected(final_actions, 1, name)
        assert obs2[name][i_oth] == pytest.approx(e_oth)
        assert obs2[name][i_std] == pytest.approx(e_std)
        # Non-vacuous: block 1's quotes differ from block 0's, so the feature
        # must have moved (the frozen-at-block-0 defect fails here).
        assert not np.isclose(obs2[name][i_oth], obs1[name][i_oth])


def test_a_future_price_shock_leaves_the_committed_prefix_unchanged(
        monkeypatch):
    """Two envs see the same day except the wholesale curve's tail. Every
    block whose rolling window ends before the shock must clear and observe
    bit-identically; the tails really differ as inputs, so the equality is
    causality, not a vacuous comparison of two failures."""
    import grid
    import rl_env
    horizon = 16        # BiddingEnv's default roll_horizon
    shock_at = 24       # block k's window is [4k, 4k+16): k <= 2 untouched
    state = {"shock": False}

    def fake_price(T, agents=None, config=None, t_start=0, day_id=None):
        idx = t_start + np.arange(T)
        if state["shock"]:
            return np.where(idx >= shock_at, 900.0, 300.0)
        return np.full(T, 300.0)

    monkeypatch.setattr(grid, "day_ahead_price_china", fake_price)

    def _roll(n_steps):
        np.random.seed(13)
        env = _make_env()
        env.reset(wholesale=np.full(96, 300.0, dtype=float))
        out = []
        for _ in range(n_steps):
            obs, rew, done, info = env.step(_quote_actions(env, 40.0, 200.0))
            assert env._last_result is not None, \
                "the clearing must succeed for the regression to mean anything"
            out.append((obs, rew, info["raw_profit"]))
        return out

    n_clean = (shock_at - horizon) // rl_env.BLOCK_SIZE + 1   # blocks 0..2
    state["shock"] = False
    run_a = _roll(n_clean)
    state["shock"] = True
    run_b = _roll(n_clean)

    # Input sanity: the shock really enters later windows.
    assert (fake_price(horizon, t_start=shock_at - 4) == 900.0).any()

    for (obs_a, rew_a, prof_a), (obs_b, rew_b, prof_b) in zip(run_a, run_b):
        assert rew_a == rew_b
        assert prof_a == prof_b
        assert set(obs_a) == set(obs_b)
        for name in obs_a:
            np.testing.assert_array_equal(obs_a[name], obs_b[name])


def test_the_system_ratio_describes_the_block_being_decided():
    """`slr_norm` in the block-k+1 observation must come from block k+1's own
    slice, not from the just-cleared block k's -- the step path used to lag
    one block behind the reset path here."""
    import rl_env
    env = _make_env()
    env.reset(wholesale=np.full(96, 300.0, dtype=float))
    # Make block 1's system ratio unmistakably different from block 0's:
    # no load at all in block 1's slice.
    for a in env.all_agents:
        a.load_forecast[rl_env.BLOCK_SIZE:2 * rl_env.BLOCK_SIZE] = 0.0
    obs1, _, done, _ = env.step(_quote_actions(env, 40.0, 200.0))
    assert not done
    expected = min(env._compute_system_load_re_ratio(rl_env.BLOCK_SIZE), 5.0)
    assert expected == 0.0, "the mutation must really separate the two blocks"
    order = env.obs_spec.feature_order
    for v in obs1.values():
        assert v[order.index("slr_norm")] == pytest.approx(expected)
