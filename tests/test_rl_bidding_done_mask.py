"""Tests for terminal-transition (done) handling in MATD3.

The replay buffer must store the per-transition done flag and MATD3.update must
mask the bootstrapping term on terminal transitions; otherwise the critic
regresses against the all-zero next observation emitted at episode end.
"""

import numpy as np

from models import MarketConfig
from scenarios import get_scenario
from rl_bidding import MATD3, ReplayBuffer


def _small_matd3():
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None][:2]
    from rl_env import BiddingEnv
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)
    return MATD3(env), rl_names


def test_replay_buffer_round_trips_done():
    buf = ReplayBuffer(capacity=10)
    obs = np.zeros((3, 11), dtype=np.float32)
    act = np.zeros((3, 2), dtype=np.float32)
    rew = np.zeros(3, dtype=np.float32)
    next_obs = np.zeros((3, 11), dtype=np.float32)
    done = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    buf.add(obs, act, rew, next_obs, done)

    o, a, r, no, d = buf.sample(10)
    assert d.shape == (1, 3)
    assert float(d[0, 0]) == 1.0
    assert float(d[0, 1]) == 0.0
    assert float(d[0, 2]) == 0.0


def test_matd3_update_runs_with_terminal_transitions():
    matd3, rl_names = _small_matd3()
    obs_dim = matd3.obs_dim
    n = matd3.n_agents

    # Fill the buffer past the batch size, with the last transition terminal.
    for i in range(140):
        obs = np.random.randn(n, obs_dim).astype(np.float32)
        act = np.random.uniform(0.3, 1.8, size=(n, 2)).astype(np.float32)
        rew = np.random.randn(n).astype(np.float32)
        next_obs = np.zeros((n, obs_dim), dtype=np.float32)
        done = np.ones(n, dtype=np.float32) if i == 139 \
            else np.zeros(n, dtype=np.float32)
        matd3.buffer.add(obs, act, rew, next_obs, done)

    info = matd3.update()
    # 140 >= batch_size(128) so both losses are computed; actor updates on
    # total_steps % policy_delay == 0 (0 % 2 == 0).
    assert info["critic_loss"] is not None
    assert info["actor_loss"] is not None
