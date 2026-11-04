"""Absolute day-end SOC on the rolling interface.

The rolling windows that reach the day's last period are the only place the
day-end constraint can bind. Every earlier window must remain unpinned, or the
pin would forbid the trajectory the window was opened to choose.
"""

import numpy as np
import pytest

import config_loader
import v2_cy_jn_env as envmod
import wholesale_price
from dispatch_core import pins_terminal_soc
from models import MarketConfig


@pytest.fixture(autouse=True)
def _restore_global_provider():
    yield
    config_loader.reload_defaults()
    wholesale_price.reset_cache()


def test_day_end_is_a_settled_horizon_and_window_is_not():
    config = MarketConfig()
    assert pins_terminal_soc(config, "day_end") is True
    assert pins_terminal_soc(config, "window") is False
    assert pins_terminal_soc(config, "full_day") is True


def _first_pair():
    from pathlib import Path
    rows = envmod._read_csv(Path(envmod.PAIRING))
    return {"load_date": rows[0]["load_date"],
            "price_date": rows[0]["price_date"]}


def test_day_end_target_pins_each_unit_to_absolute_soc():
    pair = _first_pair()
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    truth = {a.name: np.array([0.0, 0.0]) for a in env.rl_agents}
    env.reset()
    for _ in range(24):
        _, _, done, _ = env.step(truth)
        if done:
            break
    result = env._last_result
    assert result is not None
    for name in envmod.RL_NAMES:
        sched = result["schedules"][name]
        assert abs(float(sched["soc_boundary"][-1]) - 0.5) < 1e-5, name
