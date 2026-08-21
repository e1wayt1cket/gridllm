"""Tests that the differential-reward baseline re-clear shares one wholesale
curve with the real clear.

The training signal is reward = profit(price A) - profit(price B). If the two
clears each regenerate the wholesale curve with fresh unseeded noise (grid.py
adds N(0, 30) per draw), the baseline subtracts profits under a different price
realization and injects noise directly into the optimization target.
"""

import numpy as np
from unittest import mock

import grid
import market
from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv


def _make_diff_reward_env():
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names,
                     use_differential_reward=True)
    return env, rl_names


def _fixed_price(T: int, agents=None, config=None) -> np.ndarray:
    """Deterministic flat wholesale curve; realizations are irrelevant here."""
    return np.full(T, 400.0, dtype=float)


def test_real_and_baseline_clears_share_one_wholesale():
    env, rl_names = _make_diff_reward_env()
    env.reset()
    acts = {nm: np.array([1.0, 0.0], dtype=np.float32) for nm in rl_names}

    seen_wholesale = []
    real_clear = market.clear_market

    def recording_clear(*args, **kwargs):
        seen_wholesale.append(kwargs.get("wholesale"))
        return real_clear(*args, **kwargs)

    with mock.patch("grid.day_ahead_price_china",
                    side_effect=_fixed_price) as gen:
        with mock.patch("market.clear_market", side_effect=recording_clear):
            env.step(acts)

    # The wholesale curve is drawn once per block and reused by both clears;
    # regenerating inside each clear would triple this count.
    assert gen.call_count == 1
    # Both the real clear and the differential baseline re-clear ran, and they
    # received the identical wholesale array (same object).
    assert len(seen_wholesale) == 2, seen_wholesale
    assert seen_wholesale[0] is seen_wholesale[1]
    assert seen_wholesale[0] is not None
