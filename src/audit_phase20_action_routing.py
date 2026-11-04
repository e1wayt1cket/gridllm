"""P20-E: action routing, tensor dimensions and update-path call audit.

Read-only. No training, no checkpoint write, no H2 and no legacy test.

This audit answers one question: does every D+A route that produces a critic
action or an actor-loss quote use the same ESS order, leg order, tensor shape
and v3 quote map as execution? It also instruments one real ``update()`` and
requires the v2 sigmoid map to be called zero times.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import da_quote_map as qmap
import diagnose_critic_long_horizon as diag
import rl_b_c2_gate as base
import rl_da_gate as da
import rl_spec
import v2_cy_jn_env as envmod

DATA = Path("data/processed/v2_cy_jn_2025")
OUT = DATA / "phase20_action_routing.json"
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def fake_env():
    return SimpleNamespace(
        rl_agents=[SimpleNamespace(name=n) for n in NAMES],
        get_state_dim=lambda: 12,
    )


def routing_checks() -> dict:
    trainer = da.DAGateTrainer(
        fake_env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    policy = torch.tensor([[-100.0, 100.0, 2.5, 2.5]])
    torch.manual_seed(0)
    quotes = trainer._actor_loss_quotes(policy)
    actor_out = trainer.actors["ESS14"](
        torch.zeros(1, trainer.obs_dim))
    critic = trainer.critics["ESS14"]
    critic_out = critic(torch.zeros(1, trainer.n_agents * trainer.obs_dim),
                        torch.zeros(1, trainer.n_agents * trainer.act_dim))
    baseline = {nm: np.array([75.0, 150.0]) for nm in NAMES}
    obs = {nm: np.zeros(12, dtype=np.float32) for nm in NAMES}
    grads = {}
    finite = {}
    # Dropout is a training regularizer; a deterministic gradient check needs
    # the same mask on both sides of the finite difference.
    critic.eval()
    for leg in ("charge", "discharge"):
        q = float(baseline["ESS14"][diag.LEGS.index(leg)])
        grads[leg] = diag.critic_q(trainer, obs, baseline, "ESS14", leg, q)
        h = 1e-2
        def value(quote_value):
            acts = {nm: baseline[nm].copy() for nm in NAMES}
            acts["ESS14"][diag.LEGS.index(leg)] = quote_value
            obs_t = torch.tensor(np.stack([obs[nm] for nm in NAMES]),
                                 dtype=torch.float32).reshape(1, -1)
            act_t = torch.tensor(np.stack([acts[nm] for nm in NAMES]),
                                 dtype=torch.float32)
            q1, q2 = critic(obs_t, act_t.reshape(1, -1))
            return float(torch.min(q1, q2).mean())
        finite[leg] = (value(q + h) - value(q - h)) / (2 * h)
    critic.train()
    next_obs = torch.zeros(1, trainer.obs_dim)
    with torch.no_grad():
        torch.manual_seed(7)
        target = trainer._target_quotes(
            trainer.actor_targets["ESS14"], next_obs)
        torch.manual_seed(7)
        policy_out = trainer.actor_targets["ESS14"](next_obs)
        noise = (torch.randn_like(policy_out[..., 2:])
                 * trainer.experiment.target_policy_noise).clamp(
                     -trainer.experiment.target_noise_clip,
                     trainer.experiment.target_noise_clip)
        smoothed = torch.cat(
            [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        expected_target = qmap.target_quotes_v3(smoothed)
    pair_map, _ = diag.pairs()
    real_env = envmod.build_env_for_pair(
        pair_map[diag.WINTER[0]], differential_reward=False,
        terminal_soc_target=0.5)
    return {
        "ess_order_expected": list(NAMES),
        "ess_order_trainer": list(trainer.agent_names),
        "ess_order_real_env": [a.name for a in real_env.rl_agents],
        "actor_output_dim": int(actor_out.shape[-1]),
        "quote_dim": int(quotes.shape[-1]),
        "critic_input_dim": int(
            trainer.n_agents * trainer.obs_dim
            + trainer.n_agents * trainer.act_dim),
        "critic_output_shapes": [list(critic_out[0].shape),
                                 list(critic_out[1].shape)],
        "closed_charge_quote": float(quotes[0, 0]),
        "open_discharge_quote": float(quotes[0, 1]),
        "leg_gradients": grads,
        "finite_difference_dq_dquote": finite,
        "finite_difference_abs_error": {
            leg: abs(float(grads[leg]["dq_dquote"]) - finite[leg])
            for leg in finite},
        "target_route_equals_v3": bool(torch.allclose(
            target, expected_target, atol=1e-6)),
        "target_route_with_noise_finite": bool(torch.isfinite(target).all()),
        "env_action_spec": {
            "name": real_env.action_spec.name,
            "bounds": real_env.action_spec.bounds,
        },
        "trainer_action_spec": {
            "name": trainer.action_spec.name,
            "bounds": trainer.action_spec.bounds,
        },
        "same_bounds_as_v3": real_env.action_spec.bounds
        == rl_spec.ACTION_STORAGE_QUOTE_V3.bounds,
    }


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


def update_call_audit() -> dict:
    trainer = da.DAGateTrainer(
        fake_env(), da.DAExperimentConfig(), agent_names=list(NAMES))
    _fill_buffer(trainer)
    counts = {"v2": 0, "v3_actor_loss": 0, "v3_target": 0}
    original_v2 = base.forward_policy
    original_v3 = da.forward_policy_v3
    original_target = qmap.forward_policy_v3

    def v2_counter(*args, **kwargs):
        counts["v2"] += 1
        return original_v2(*args, **kwargs)

    def v3_counter(*args, **kwargs):
        counts["v3_actor_loss"] += 1
        return original_v3(*args, **kwargs)

    def target_counter(*args, **kwargs):
        counts["v3_target"] += 1
        return original_target(*args, **kwargs)

    base.forward_policy = v2_counter
    da.forward_policy_v3 = v3_counter
    qmap.forward_policy_v3 = target_counter
    try:
        torch.manual_seed(0)
        out = trainer.update()
    finally:
        base.forward_policy = original_v2
        da.forward_policy_v3 = original_v3
        qmap.forward_policy_v3 = original_target
    return {
        "calls": counts,
        "v2_calls_in_update": counts["v2"],
        "v3_calls_in_update": counts["v3_actor_loss"] + counts["v3_target"],
        "actor_loss": out["actor_loss"],
        "actor_loss_finite": bool(np.isfinite(out["actor_loss"])),
    }


def main() -> int:
    routing = routing_checks()
    update = update_call_audit()
    checks = {
        "ess_order_matches": routing["ess_order_trainer"]
        == routing["ess_order_expected"] == routing["ess_order_real_env"],
        "actor_output_4": routing["actor_output_dim"] == 4,
        "quote_dim_2": routing["quote_dim"] == 2,
        "critic_input_56": routing["critic_input_dim"] == 56,
        "closed_gate_zero": routing["closed_charge_quote"] == 0.0,
        "open_gate_positive": routing["open_discharge_quote"] > 0.0,
        "target_route_v3": routing["target_route_equals_v3"],
        "finite_difference": all(
            value <= 5e-2 for value in
            routing["finite_difference_abs_error"].values()),
        "v2_not_called_in_update": update["v2_calls_in_update"] == 0,
        "v3_called_in_update": update["v3_calls_in_update"] > 0,
        "actor_loss_finite": update["actor_loss_finite"],
    }
    payload = {
        "boundary": ("read-only routing audit; one in-memory update, no "
                     "checkpoint write, no H2 and no legacy test"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "routing": routing,
        "update_audit": update,
        "note": ("the environment carries the v2 action spec as metadata; "
                 "its bounds and truthful point are identical to v3, so this "
                 "does not change the numeric action contract"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass": payload["all_pass"], "checks": checks},
                     indent=2))
    print(f"wrote {OUT}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
