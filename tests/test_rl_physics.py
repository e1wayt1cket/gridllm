"""Tests for physics-guided critic weights and injection (src/rl_physics.py)."""

import numpy as np
import torch
import pytest

from rl_physics import (pairwise_resistance_distance,
                        storage_interaction_weights,
                        PhysicsCentralizedCritic)


class _FakeLine:
    """Minimal net.line-like container for distance tests."""

    def __init__(self, rows):
        # rows: list of (from, to, r_ohm_per_km, length_km)
        import pandas as pd
        self.line = pd.DataFrame(
            rows, columns=["from_bus", "to_bus", "r_ohm_per_km", "length_km"])


def test_pairwise_resistance_distance_radial_tree():
    # 0-1-2 chain: r = 0.5 per edge.
    net = _FakeLine([(0, 1, 1.0, 0.5), (1, 2, 1.0, 0.5)])
    D = pairwise_resistance_distance(net, [0, 1, 2])
    assert D[0, 1] == pytest.approx(0.5)
    assert D[1, 2] == pytest.approx(0.5)
    assert D[0, 2] == pytest.approx(1.0)   # 0.5 + 0.5 via bus 1
    assert D[0, 0] == pytest.approx(0.0)
    assert D[1, 0] == pytest.approx(0.5)   # symmetric (undirected graph)


def test_storage_interaction_weights_zero_diag_symmetric_row_normalized(monkeypatch):
    # Reuse a tiny net; buses any subset present in the graph.
    net = _FakeLine([(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)])
    import rl_physics as rp
    monkeypatch.setattr(rp, "build_base_network", lambda config: net)
    W = storage_interaction_weights(None, [0, 1, 2])
    assert W.shape == (3, 3)
    assert np.allclose(np.diag(W), 0.0)
    # Row normalization (per-agent neighbor weights sum to 1) is used, so the
    # matrix is NOT column-symmetric - only zero-diagonal, non-negative, rows 1.
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(W >= 0.0)
    # Closer buses get larger weights.
    assert W[0, 1] > W[0, 2]


def test_physics_critic_forward_shape_and_grad():
    W = np.array([[0.0, 0.5, 0.5],
                  [0.5, 0.0, 0.5],
                  [0.5, 0.5, 0.0]], dtype=np.float32)
    crit = PhysicsCentralizedCritic(n_agents=3, obs_dim=4, act_dim=2,
                                    W=W, agent_idx=0, out_dim=1)
    obs = torch.randn(5, 3 * 4)
    acts = torch.randn(5, 3 * 2)
    q1, q2 = crit(obs, acts)
    assert q1.shape == (5, 1)
    assert q2.shape == (5, 1)
    loss = (q1 ** 2).sum() + (q2 ** 2).sum()
    loss.backward()
    g = next(p.grad for p in crit.encoder.parameters() if p.grad is not None)
    assert g is not None and float(g.abs().sum()) > 0


def test_physics_critic_quantile_heads():
    W = np.eye(2) * 0.0
    W[0, 1] = W[1, 0] = 1.0
    crit = PhysicsCentralizedCritic(n_agents=2, obs_dim=3, act_dim=2,
                                    W=W, agent_idx=1, out_dim=8)
    obs = torch.randn(4, 6)
    acts = torch.randn(4, 4)
    q1, q2 = crit(obs, acts)
    assert q1.shape == (4, 8)
    assert q2.shape == (4, 8)


def test_matd3_physics_critic_update_runs():
    """Physics critic drop-in via critic_factory on the real env/update."""
    from models import MarketConfig
    from scenarios import get_scenario
    from rl_env import BiddingEnv
    from rl_bidding import MATD3
    import rl_physics as rp

    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None][:2]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)
    buses = [a.bus for a in env.rl_agents]
    # Compute weights from the real (cached) network topology.
    W = storage_interaction_weights(config, buses)
    assert W.shape == (2, 2)
    factory = (lambda n, od, ad, i: PhysicsCentralizedCritic(
        n, od, ad, W, i, out_dim=1))
    m = MATD3(env, critic_factory=factory)
    obs_dim = m.obs_dim
    n = m.n_agents
    for i in range(140):
        obs = np.random.randn(n, obs_dim).astype(np.float32)
        act = np.random.uniform(0.3, 1.8, size=(n, 2)).astype(np.float32)
        rew = np.random.randn(n).astype(np.float32)
        next_obs = np.zeros((n, obs_dim), dtype=np.float32)
        done = np.ones(n, dtype=np.float32) if i == 139 \
            else np.zeros(n, dtype=np.float32)
        m.buffer.add(obs, act, rew, next_obs, done)
    info = m.update()
    assert info["critic_loss"] is not None
    assert info["actor_loss"] is not None
    assert len(info["critic_loss_by_agent"]) == 2
