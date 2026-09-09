"""Unit tests for the quantile distributional critic (src/rl_quantile.py).

Fast pure-torch checks (no OPF) for the quantile grid and quantile-Huber loss,
plus the critic shape/twin contract. One smoke mirrors test_rl_bidding_done_mask
by building a 2-agent BiddingEnv and running an update with terminal transitions.
"""

import numpy as np
import torch
import pytest

from rl_quantile import (quantile_huber_loss, QuantileCentralizedCritic,
                         QuantileMATD3)


def test_tau_grid_symmetric_and_length_n():
    qt = (torch.arange(32, dtype=torch.float32) + 0.5) / 32
    assert qt.shape == (32,)
    assert qt[0].item() == pytest.approx(0.5 / 32)
    assert qt[-1].item() == pytest.approx(31.5 / 32)
    assert all(qt[i + 1] > qt[i] for i in range(31))
    assert (qt + qt.flip(0)).max().item() == pytest.approx(1.0)


def test_quantile_huber_closed_form():
    tau = torch.tensor([0.5])
    # delta = +0.75 inside Huber region, kappa=1, weight |0.5-0| = 0.5.
    pred = torch.tensor([[0.25]])
    y = torch.tensor([1.0])
    assert quantile_huber_loss(pred, y, tau) == pytest.approx(
        0.5 * 0.5 * 0.75 ** 2)
    # over-prediction delta = -0.75 -> weight |0.5-1| = 0.5.
    pred2 = torch.tensor([[1.75]])
    assert quantile_huber_loss(pred2, y, tau) == pytest.approx(
        0.5 * 0.5 * 0.75 ** 2)


def test_quantile_huber_asymmetry():
    tau = torch.tensor([0.25])
    y = torch.tensor([1.0])
    under = quantile_huber_loss(torch.tensor([[0.0]]), y, tau)   # delta>0, w=tau
    over = quantile_huber_loss(torch.tensor([[2.0]]), y, tau)    # delta<0, w=1-tau
    assert under < over
    tau5 = torch.tensor([0.5])
    a = quantile_huber_loss(torch.tensor([[0.0]]), y, tau5)
    b = quantile_huber_loss(torch.tensor([[2.0]]), y, tau5)
    assert a == pytest.approx(b)


def test_quantile_huber_convexity():
    tau = torch.tensor([0.3])
    y = torch.tensor([1.0])
    preds = torch.linspace(-2.0, 4.0, 61)
    vals = torch.stack([quantile_huber_loss(p.reshape(1, 1), y, tau)
                        for p in preds])
    sec = vals[2:] - 2 * vals[1:-1] + vals[:-2]
    assert (sec >= -1e-6).all()


def test_quantile_critic_shapes_and_twins():
    crit = QuantileCentralizedCritic(n_agents=2, obs_dim=3, act_dim=2,
                                     n_quantiles=8)
    obs = torch.randn(4, 6)
    acts = torch.randn(4, 4)
    q1, q2 = crit(obs, acts)
    assert q1.shape == (4, 8)
    assert q2.shape == (4, 8)
    assert id(crit.q1) != id(crit.q2)
    loss = q1.sum() + q2.sum()
    loss.backward()
    g1 = next(p.grad for p in crit.q1.parameters() if p.grad is not None)
    assert g1 is not None and float(g1.abs().sum()) > 0


def test_qmatd3_update_with_terminal_transitions():
    from models import MarketConfig
    from scenarios import get_scenario
    from rl_env import BiddingEnv

    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None][:2]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)
    qm = QuantileMATD3(env)
    obs_dim = qm.obs_dim
    n = qm.n_agents
    for i in range(140):
        obs = np.random.randn(n, obs_dim).astype(np.float32)
        act = np.random.uniform(0.3, 1.8, size=(n, 2)).astype(np.float32)
        rew = np.random.randn(n).astype(np.float32)
        next_obs = np.zeros((n, obs_dim), dtype=np.float32)
        done = np.ones(n, dtype=np.float32) if i == 139 \
            else np.zeros(n, dtype=np.float32)
        qm.buffer.add(obs, act, rew, next_obs, done)

    info = qm.update()
    assert info["critic_loss"] is not None
    assert info["actor_loss"] is not None
    assert len(info["critic_loss_by_agent"]) == 2
    assert len(info["actor_loss_by_agent"]) == 2
