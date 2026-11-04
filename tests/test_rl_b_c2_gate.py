"""The B/C2/gate trainer branch: buffer, rewards, gates, checkpoint."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import config_loader
import rl_b_c2_gate as mod
import v2_cy_jn_env as envmod
import wholesale_price
from daily_reward_b import B1Cache, B1Day
from truthful_gate_actor_adapter import forward_policy, target_quotes


@pytest.fixture(autouse=True)
def _restore_global_provider():
    """`build_env_for_pair` points the global price source at JN2025."""
    yield
    config_loader.reload_defaults()
    wholesale_price.reset_cache()


def _pair(rows=None):
    from pathlib import Path as P
    rows = rows or envmod._read_csv(P(envmod.PAIRING))
    return {"load_date": rows[0]["load_date"],
            "price_date": rows[0]["price_date"]}


def test_closed_gate_is_exactly_zero_even_with_target_smoothing():
    policy_out = torch.tensor([[-5.0, -5.0, 0.0, 0.0]])
    assert torch.equal(target_quotes(policy_out),
                       torch.zeros(1, 2))


def test_straight_through_gate_reaches_the_actor():
    torch.manual_seed(0)
    policy_out = torch.tensor([[5.0, 5.0, 0.0, 0.0]], requires_grad=True)
    out = forward_policy(policy_out, train=True, temperature=1.0)
    out["quotes"].sum().backward()
    assert policy_out.grad is not None
    assert float(policy_out.grad[0, :2].abs().sum()) > 0.0
    assert float(policy_out.grad[0, 2:].abs().sum()) > 0.0


def test_gate_buffer_round_trip():
    buf = mod.GateReplayBuffer(8)
    for i in range(4):
        buf.add(np.full((2, 3), i), np.full((2, 2), i),
                np.full(2, i, dtype=float), np.full((2, 3), i + 1), False,
                {"gates": np.full((2, 2), i), "probs": np.full((2, 2), 0.5),
                 "gate_logits": np.full((2, 2), i),
                 "price_logits": np.full((2, 2), -i),
                 "price_quotes": np.full((2, 2), 1.0),
                 "sampled": np.full((2, 2), i)})
    arrays = buf.extra_arrays
    assert arrays["gates"].shape == (4, 2, 2)
    other = mod.GateReplayBuffer(8)
    other.load_arrays({
        "obs": np.asarray([row[0] for row in buf.buffer]),
        "actions": np.asarray([row[1] for row in buf.buffer]),
        "rewards": np.asarray([row[2] for row in buf.buffer]),
        "next_obs": np.asarray([row[3] for row in buf.buffer]),
        "done": np.asarray([row[4] for row in buf.buffer]),
        **arrays,
    })
    assert len(other) == 4
    batch = other.sample(4)
    assert batch["gates"].shape == (4, 2, 2)
    assert sorted(batch["actions"][:, 0, 0].tolist()) == [0.0, 1.0, 2.0, 3.0]


def _cache_for(pair, experiments=1):
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    cache = B1Cache()
    cache.add(B1Day(
        pair_id=pair_id, soc0=0.5, config_hash="c" * 64,
        data_hash="d" * 64,
        block_payoffs={"ESS9": [0.0] * 24, "ESS14": [0.0] * 24,
                       "ESS17": [0.0] * 24, "ESS27": [0.0] * 24},
        soc_end={"ESS9": 0.5, "ESS14": 0.5, "ESS17": 0.5, "ESS27": 0.5}))
    return cache


def _env(pair):
    return envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)


def test_b_only_pays_once_at_the_last_block():
    pair = _pair()
    env = _env(pair)
    trainer = mod.MATD3BC2Gate(env, mod.EXPERIMENTS["NEAR_B_ONLY_GATE"])
    out = trainer.train_day(
        env, pair, _cache_for(pair),
        config_hash_value="c" * 64, data_hash="d" * 64)
    reward = out["b"]["reward_sum"]
    # The day's reward is entirely the last block; the sum is the delta/scale.
    for unit in trainer.agent_names:
        assert abs(out["b"]["discounted_return"][unit]
                   - 0.99 ** 23 * reward[unit]) < 1e-9


def test_b_c2_shaping_closes_the_discount_identity():
    pair = _pair()
    env = _env(pair)
    trainer = mod.MATD3BC2Gate(env, mod.EXPERIMENTS["NEAR_B_C2_GATE"])
    out = trainer.train_day(
        env, pair, _cache_for(pair),
        config_hash_value="c" * 64, data_hash="d" * 64)
    assert out["c2"] is not None
    for unit in trainer.agent_names:
        boundary = out["c2"]["boundary"][unit]
        # C2 changes the return by a policy-independent boundary term only.
        assert abs(out["c2"]["episode_return"][unit]
                   - (out["b"]["discounted_return"][unit] + boundary)) < 1e-9


def test_checkpoint_restores_action_buffer_rng_and_update(tmp_path):
    pair = _pair()
    env = _env(pair)
    experiment = mod.ExperimentConfig("roundtrip", "B_C2", batch_size=8)
    trainer = mod.MATD3BC2Gate(env, experiment)
    trainer.total_steps = 16
    rng = np.random.default_rng(0)
    for i in range(16):
        obs = rng.normal(size=(4, trainer.obs_dim)).astype(np.float32)
        act = rng.random((4, 2)).astype(np.float32)
        trainer.buffer.add(
            obs, act, np.full(4, i / 100.0, dtype=np.float32),
            rng.normal(size=(4, trainer.obs_dim)).astype(np.float32), False,
            {"gates": np.ones((4, 2)), "probs": np.full((4, 2), 0.5),
             "gate_logits": np.zeros((4, 2)),
             "price_logits": np.zeros((4, 2)),
             "price_quotes": np.ones((4, 2)),
             "sampled": np.ones((4, 2))})
    obs0 = rng.normal(size=trainer.obs_dim).astype(np.float32)
    trainer.save(tmp_path, level="resume", dataset="roundtrip", seed=7,
                 config_hash_value="a" * 64)
    sample_expected = trainer.buffer.sample(4)
    action_expected = trainer.select_action(obs0, trainer.agent_names[0])
    update_expected = trainer.update()
    params_expected = {nm: copy.deepcopy(trainer.actors[nm].state_dict())
                       for nm in trainer.agent_names}
    exposure_expected = dict(trainer.exposure)
    del trainer

    restored = mod.MATD3BC2Gate(env, experiment)
    bundle = restored.load(tmp_path)
    assert bundle["manifest"]["level"] == "resume"
    sample_actual = restored.buffer.sample(4)
    for key in sample_expected:
        assert torch.equal(sample_expected[key], sample_actual[key]), key
    action_actual = restored.select_action(obs0, restored.agent_names[0])
    for key in ("quotes", "gates", "probs"):
        assert np.allclose(action_expected[key], action_actual[key]), key
    update_actual = restored.update()
    for key in ("critic_loss", "actor_loss", "q1_mean", "q2_mean"):
        assert abs(update_expected[key] - update_actual[key]) < 1e-6, key
    for nm in restored.agent_names:
        for k, v in params_expected[nm].items():
            assert torch.equal(v, restored.actors[nm].state_dict()[k]), (nm, k)
    assert restored.total_steps == 16
    assert len(restored.buffer) == 16
    assert dict(restored.exposure) == exposure_expected
