"""Tests for the stochastic SAC actor and MASAC trainer (src/rl_masac.py)."""

import numpy as np
import torch
import pytest

from rl_masac import StochasticActor, MASAC

BOUNDS = torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32)


def test_stochastic_actor_mean_forward_in_bounds():
    act = StochasticActor(4, 2, BOUNDS)
    obs = torch.randn(3, 4)
    a = act(obs).detach()
    assert a.shape == (3, 2)
    assert torch.all(a >= BOUNDS[0]) and torch.all(a <= BOUNDS[1])
    mu, mean_a = act.forward_logits(obs)
    assert mu.shape == (3, 2)
    assert torch.allclose(mean_a.detach(), a)


def test_stochastic_actor_samples_in_bounds():
    act = StochasticActor(4, 2, BOUNDS)
    obs = torch.randn(2000, 4)
    s, lp = act.sample(obs)
    assert s.shape == (2000, 2)
    assert torch.all(s >= BOUNDS[0]) and torch.all(s <= BOUNDS[1])
    assert torch.isfinite(lp).all()
    assert lp.dim() == 1


def test_stochastic_actor_higher_std_gives_more_entropy():
    obs = torch.randn(2000, 4)
    low = StochasticActor(4, 2, BOUNDS)
    low.log_std.data.fill_(-2.0)   # clamped minimum std -> low entropy
    high = StochasticActor(4, 2, BOUNDS)
    high.log_std.data.fill_(0.0)   # std ~1 -> higher entropy
    _, lp_low = low.sample(obs)
    _, lp_high = high.sample(obs)
    assert float((-lp_high).mean().detach()) > float((-lp_low).mean().detach())


def test_stochastic_actor_save_load_roundtrip(tmp_path):
    from rl_td3 import save_policy, load_policy
    act = StochasticActor(4, 2, BOUNDS)
    p = str(tmp_path / "a.pt")
    save_policy(act, p)  # no spec: state dict only, but add meta manually below
    # Re-save with spec so the actor_type marker is embedded.
    act2 = StochasticActor(4, 2, BOUNDS)
    save_policy(act2, p, obs_spec=None, action_spec=None)  # still no meta
    # Force metadata tagging via a spec-less path is not possible in save_policy
    # (meta only added when a spec is given). Use a stub spec.
    class _S:
        def to_dict(self):
            return {"name": "v3_12d"}
    save_policy(act, p, obs_spec=_S(), action_spec=_S())
    loaded = load_policy(p, 4, BOUNDS)
    assert type(loaded).__name__ == "StochasticActor"
    obs = torch.randn(2, 4)
    assert torch.allclose(loaded(obs).detach(), act(obs).detach(), atol=1e-6)


def test_masac_update_with_terminal_transitions():
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
    m = MASAC(env)
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
    assert len(info["actor_loss_by_agent"]) == 2
    # Temperature is trainable and got a gradient this step.
    la = next(iter(m.log_alpha.values()))
    assert la.grad is not None and float(la.grad.abs().sum()) >= 0
