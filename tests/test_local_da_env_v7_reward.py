import numpy as np

from local_da_days import load_pair
from rl_da_env import BLOCK_SIZE, LocalDABiddingEnv

TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def truthful(env):
    return {nm: TRUTHFUL.copy() for nm in env.agent_names}


def walk(env):
    infos = []
    env.reset(load_pair("2021-06-04"))
    for _ in range(24):
        _, _, done, info = env.step(truthful(env))
        infos.append(info)
        if done:
            break
    return infos


def test_the_declared_point_pays_exactly_its_baseline():
    env = LocalDABiddingEnv()
    infos = walk(env)
    assert len(infos) == 24
    for info in infos:
        for nm in env.agent_names:
            assert abs(info["payoff"][nm] - info["baseline_payoff"][nm]) < 1e-9


def test_one_inventory_term_per_episode_at_the_declared_price():
    env = LocalDABiddingEnv()
    infos = walk(env)
    non_zero = [i for i, info in enumerate(infos)
                if any(abs(v) > 1e-9 for v in info["inventory_cny"].values())]
    assert non_zero == [23]
    e96 = infos[23]["inventory_cny"]["ESS9"] / env.pair["external_price"][-1] + 1.6 * 0.5
    assert 0.1 * 1.6 <= e96 <= 0.9 * 1.6


def test_the_degradation_is_throughput_times_the_declared_rate():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    _, _, _, info = env.step(truthful(env))
    for nm in env.agent_names:
        assert abs(info["degradation_cny"][nm]
                   - 100. * 0.25 * (info["charge_mwh"][nm]
                                    + info["discharge_mwh"][nm])) < 1e-6


def test_a_carried_soc_on_the_band_edge_keeps_the_band():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    env.prev_soc = {nm: 0.9 for nm in env.agent_names}
    _, _, _, info = env.step(truthful(env))
    assert all(0.1 - 1e-6 <= s <= 0.9 + 1e-6 for s in env.prev_soc.values())
    assert "ESS9" in info["payoff"]
