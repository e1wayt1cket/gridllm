"""The D+A actor loss must differentiate the quote map it executes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import da_quote_map
import rl_b_c2_gate as base
import rl_da_gate as da
import truthful_gate_actor_adapter as v2

NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def _env():
    return SimpleNamespace(
        rl_agents=[SimpleNamespace(name=n) for n in NAMES],
        get_state_dim=lambda: 12,
    )


def _same_draw_update_and_execution(trainer, policy_out):
    torch.manual_seed(0)
    update_quotes = trainer._actor_loss_quotes(policy_out)
    torch.manual_seed(0)
    execution_quotes = da_quote_map.forward_policy_v3(
        policy_out, train=True,
        gate_temperature=trainer.current_gate_temperature())["quotes"]
    return update_quotes, execution_quotes


def test_update_path_equals_execution_path_on_batched_open_gates():
    trainer = da.DAGateTrainer(
        _env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    policy = torch.tensor([
        [10.0, 10.0, -5.0, -5.0],
        [10.0, 10.0, 0.0, 0.0],
        [10.0, 10.0, 2.5, 2.5],
        [10.0, 10.0, 5.0, 5.0],
        [10.0, 10.0, -5.0, 5.0],
        [10.0, 10.0, 5.0, -5.0],
    ])
    update_quotes, execution_quotes = _same_draw_update_and_execution(
        trainer, policy)
    assert torch.allclose(update_quotes, execution_quotes, atol=1e-6)
    assert update_quotes.shape == (6, 2)
    # Same price logit on both legs: the 300 box is twice the 150 box.
    assert torch.all(update_quotes[:4, 1] > update_quotes[:4, 0])


def test_closed_gate_is_exactly_zero_in_the_update_path():
    trainer = da.DAGateTrainer(
        _env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    policy = torch.tensor([[-100.0, 100.0, 2.5, 2.5]])
    torch.manual_seed(0)
    quotes = trainer._actor_loss_quotes(policy)
    assert float(quotes[0, 1]) > 0.0
    assert float(quotes[0, 0]) == 0.0


def test_update_path_uses_the_scheduled_gate_temperature():
    trainer = da.DAGateTrainer(
        _env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    policy = torch.tensor([[10.0, 10.0, 0.0, 0.0]])
    for episode in (0, da.DAExperimentConfig.gate_warmup_episodes // 2,
                    da.DAExperimentConfig.gate_warmup_episodes):
        trainer._current_episode = episode
        torch.manual_seed(3)
        got = trainer._actor_loss_quotes(policy)
        torch.manual_seed(3)
        expected = da_quote_map.forward_policy_v3(
            policy, train=True,
            gate_temperature=trainer.current_gate_temperature())["quotes"]
        assert torch.allclose(got, expected, atol=1e-6)


def test_update_path_is_v3_not_v2():
    trainer = da.DAGateTrainer(
        _env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    policy = torch.tensor([[10.0, 10.0, 2.5, 2.5]])
    torch.manual_seed(0)
    got = trainer._actor_loss_quotes(policy)
    with torch.no_grad():
        old = v2.forward_policy(policy, train=False)["quotes"]
    assert torch.allclose(got, torch.tensor([[148.5, 297.0]]), atol=1e-4)
    assert torch.max(torch.abs(got - old)) > 1.0


def test_quote_map_autograd_matches_finite_difference():
    points = torch.tensor([-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0],
                          dtype=torch.float64)
    z = points.clone().requires_grad_(True)
    fractions = da_quote_map.map_fraction(z)
    grad, = torch.autograd.grad(fractions.sum(), z)
    h = 1e-6
    numeric = (da_quote_map.map_fraction_np(points.numpy() + h)
               - da_quote_map.map_fraction_np(points.numpy() - h)) / (2 * h)
    assert np.allclose(grad.detach().numpy(), numeric, atol=1e-5)


def _fill_buffer(trainer, size=128):
    obs = np.zeros((4, 12), dtype=np.float32)
    action = np.array([[75.0, 150.0]] * 4, dtype=np.float32)
    reward = np.zeros(4, dtype=np.float32)
    extras = {
        "gates": np.ones((4, 2), dtype=np.float32),
        "probs": np.full((4, 2), 0.5, dtype=np.float32),
        "gate_logits": np.zeros((4, 2), dtype=np.float32),
        "price_logits": np.zeros((4, 2), dtype=np.float32),
        "price_quotes": action.copy(),
        "sampled": np.ones((4, 2), dtype=np.float32),
        "explored": np.zeros((4, 2), dtype=np.float32),
        "exploration_source": np.zeros((4, 2), dtype=np.float32),
    }
    for _ in range(size):
        trainer.buffer.add(obs, action, reward, obs, 0.0, extras)


def test_update_does_not_call_the_v2_map(monkeypatch):
    trainer = da.DAGateTrainer(
        _env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    _fill_buffer(trainer)

    def forbidden(*args, **kwargs):
        raise AssertionError("actor update called the v2 quote map")

    monkeypatch.setattr(base, "forward_policy", forbidden)
    torch.manual_seed(0)
    out = trainer.update()
    assert out["actor_loss"] is not None
    assert trainer.updates == 1


def test_base_trainer_keeps_the_v2_actor_loss_map():
    trainer = base.MATD3BC2Gate(
        _env(), base.EXPERIMENTS["NEAR_B_C2_GATE"],
        agent_names=list(NAMES))
    policy = torch.tensor([[10.0, 10.0, 2.5, 2.5]])
    torch.manual_seed(4)
    got = trainer._actor_loss_quotes(policy)
    torch.manual_seed(4)
    expected = v2.forward_policy(
        policy, train=True,
        temperature=trainer.experiment.gate_temperature)["quotes"]
    assert torch.allclose(got, expected, atol=1e-6)
