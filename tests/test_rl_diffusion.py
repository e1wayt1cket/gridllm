# test_rl_diffusion.py
"""Unit tests for the MAD3PG diffusion scheduler and critic (pure torch, no OPF)."""

import torch

from rl_diffusion import DiffusionCritic, DiffusionScheduler


def test_scheduler_shapes_and_monotonicity():
    T = 10
    s = DiffusionScheduler(T=T, device="cpu")
    assert s.alpha_bars.shape == (T,)
    assert s.sigma_q.shape == (T,)
    assert torch.all(s.alpha_bars[1:] <= s.alpha_bars[:-1] + 1e-9)
    r0 = torch.ones(8, 1)
    t = torch.full((8,), T - 1, dtype=torch.long)
    r_t, noise = s.q_sample(r0, t)
    assert r_t.shape == (8, 1)
    assert noise.shape == (8, 1)


def test_diffusion_critic_output_and_grad():
    cond_dim = 16
    B = 8
    net = DiffusionCritic(cond_dim=cond_dim, hidden=32, T=5)
    r_t = torch.randn(B, 1)
    t = torch.randint(0, 5, (B,))
    c = torch.randn(B, cond_dim)
    pred = net(r_t, t, c)
    assert pred.shape == (B, 1)
    loss = pred.pow(2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in net.parameters())
    assert has_grad, "critic must be trainable w.r.t. its parameters"
