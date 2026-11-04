# test_daily_reward_b.py
"""B: terminal-only payment, preserved components, strict cache, fixed scale."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import daily_reward_b as b

UNITS = b.UNITS
T = b.T


def _paths(offset=0.0):
    rng = np.random.default_rng(1)
    return ({u: rng.normal(100 + offset, 10, T) for u in UNITS},
            {u: rng.normal(50 + offset, 10, T) for u in UNITS})


def test_only_the_last_block_pays():
    policy, baseline = _paths()
    day = b.day_reward_b(policy, baseline, pair_id="p")
    for arr in day.reward.values():
        assert np.all(arr[:-1] == 0.0)
        assert arr[-1] != 0.0
    assert day.paid_at == T - 1


def test_components_are_preserved_and_scaled_fixedly():
    policy, baseline = _paths()
    cfg = b.RewardBConfig(fixed_scale=1000.0)
    day = b.day_reward_b(policy, baseline, pair_id="p", config=cfg)
    for u in UNITS:
        assert day.G_policy[u] == pytest.approx(np.sum(policy[u]))
        assert day.G_b1[u] == pytest.approx(np.sum(baseline[u]))
        assert day.delta[u] == pytest.approx(np.sum(policy[u] - baseline[u]))
        assert day.scaled_delta[u] == pytest.approx(day.delta[u] / 1000.0)
        assert day.discounted_return[u] == pytest.approx(
            cfg.gamma ** (T - 1) * day.scaled_delta[u])


def test_the_cache_key_is_strict():
    policy, baseline = _paths()
    day = b.day_reward_b(policy, baseline, pair_id="p")
    cache = b.B1Cache()
    cache.add(b.B1Day(pair_id="p", soc0=0.5, config_hash="c", data_hash="d",
                      block_payoffs=baseline, soc_end={u: 0.5 for u in UNITS}))
    assert b.day_reward_b_from_cache(policy, cache, "p", 0.5, "c", "d")
    with pytest.raises(b.CacheKeyError):
        b.day_reward_b_from_cache(policy, cache, "p", 0.6, "c", "d")
    with pytest.raises(b.CacheKeyError):
        b.day_reward_b_from_cache(policy, cache, "p", 0.5, "other", "d")


def test_bad_paths_are_refused():
    policy, baseline = _paths()
    policy = dict(policy)
    policy["ESS9"] = policy["ESS9"][:-1]
    with pytest.raises(ValueError):
        b.day_reward_b(policy, baseline, pair_id="p")


def test_no_running_reward_standardization():
    code = (inspect.getsource(b.day_reward_b)
            + inspect.getsource(b.day_reward_b_from_cache))
    assert "reward_std" not in code
    assert "np.std" not in code
