"""Tests that the environment retains its differential-reward baseline clear.

The truthful baseline re-clear already runs every block to compute the shaped
reward. Retaining it lets per-episode accounting read the truthful day without
paying for a third clearing, which is what makes training-time consumer
metrics free rather than a second full simulation.
"""

import numpy as np
import pytest
from unittest import mock

import market
from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv


def _make_env(use_differential_reward=True):
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names,
                     use_differential_reward=use_differential_reward)
    return env, rl_names


def _fixed_price(T: int, agents=None, config=None) -> np.ndarray:
    return np.full(T, 400.0, dtype=float)


def _run_one_block(env, rl_names):
    """Drive one block, recording every clear_market result in call order."""
    env.reset()
    acts = {nm: np.array([1.0, 0.0], dtype=np.float32) for nm in rl_names}
    seen = []
    real_clear = market.clear_market

    def recording_clear(*args, **kwargs):
        res = real_clear(*args, **kwargs)
        seen.append(res)
        return res

    with mock.patch("grid.day_ahead_price_china", side_effect=_fixed_price):
        with mock.patch("market.clear_market", side_effect=recording_clear):
            env.step(acts)
    return seen


def test_env_retains_the_baseline_clear_separately_from_the_real_clear():
    env, rl_names = _make_env(use_differential_reward=True)
    seen = _run_one_block(env, rl_names)

    # One block runs exactly two clears: the real one, then the truthful
    # differential baseline. Retaining the second must not disturb the first.
    assert len(seen) == 2, f"expected 2 clears, saw {len(seen)}"
    assert env._last_result is seen[0]
    assert env._last_base_result is seen[1]
    assert env._last_base_result is not env._last_result

    base = env._last_base_result
    assert "schedules" in base and "lmp" in base
    assert base["lmp"].shape[1] > 0


def test_baseline_result_is_absent_without_differential_reward():
    # Without the shaped reward there is no truthful baseline to compare
    # against, so the attribute stays None rather than holding a stale or
    # fabricated clear.
    env, rl_names = _make_env(use_differential_reward=False)
    seen = _run_one_block(env, rl_names)

    assert len(seen) == 1, f"expected 1 clear, saw {len(seen)}"
    assert env._last_base_result is None
    assert env._last_result is seen[0]


def test_reset_clears_a_retained_baseline():
    env, rl_names = _make_env(use_differential_reward=True)
    _run_one_block(env, rl_names)
    assert env._last_base_result is not None

    env.reset()
    assert env._last_base_result is None
