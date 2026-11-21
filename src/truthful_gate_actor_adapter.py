# truthful_gate_actor_adapter.py
"""Adapter between a gate policy vector and the market quote action.

Per ESS the policy outputs four numbers:

    [gate_logit_charge, gate_logit_discharge, price_logit_charge,
     price_logit_discharge]

Forward (evaluation):

    g_k = 1[gate_logit_k > 0]
    quote_k = 0              if g_k = 0
            = h_k * sigmoid(price_logit_k)   if g_k = 1

Training uses a hard forward gate and a **biased** straight-through surrogate
gradient through the gate probability; the price branch keeps its exact
sigmoid derivative. The critic sees the **executed quote** (two numbers), not
the gate probability, so its action space is the market's.

This module is independent of the environment, the action spec and any trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

LEG_HIGHS = (150.0, 300.0)
POLICY_DIM = 4
QUOTE_DIM = 2
GATES_PER_UNIT = 2


class _STGate(torch.autograd.Function):
    """Hard forward, sigmoid-derivative surrogate backward (biased)."""

    @staticmethod
    def forward(ctx, logit: torch.Tensor, prob: torch.Tensor):
        ctx.save_for_backward(prob)
        return (logit > 0.0).to(logit.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        prob, = ctx.saved_tensors
        return grad_output * prob * (1.0 - prob), None


def gate_probability(gate_logits: torch.Tensor,
                     temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(gate_logits / temperature)


def _split(policy_out: torch.Tensor):
    if policy_out.shape[-1] != POLICY_DIM:
        raise ValueError(
            f"policy output must end in {POLICY_DIM}; got "
            f"{tuple(policy_out.shape)}")
    return policy_out[..., :GATES_PER_UNIT], policy_out[..., GATES_PER_UNIT:]


def _hard_gate(gate_logits: torch.Tensor) -> torch.Tensor:
    return (gate_logits > 0.0).to(gate_logits.dtype)


def forward_policy(policy_out: torch.Tensor, *, train: bool = False,
                   generator: Optional[torch.Generator] = None,
                   temperature: float = 1.0,
                   straight_through: bool = True) -> Dict[str, torch.Tensor]:
    """Map the policy vector to quotes, gates and diagnostics.

    `train=True` samples the gate from Bernoulli(prob). The executed gate is
    hard; with `straight_through` the gate's gradient is the surrogate
    `prob * (1 - prob)`, which is biased.
    """
    gate_logits, price_logits = _split(policy_out)
    probs = gate_probability(gate_logits, temperature)
    if train:
        draw = torch.rand(probs.shape, generator=generator,
                          dtype=probs.dtype)
        hard = (draw < probs).to(probs.dtype)
        gate = hard + (probs - probs.detach()) if straight_through else hard
    else:
        gate = _hard_gate(gate_logits)
    highs = torch.as_tensor(LEG_HIGHS, dtype=policy_out.dtype,
                            device=policy_out.device)
    quotes = gate * highs * torch.sigmoid(price_logits)
    mode = (gate[..., 0] + 2.0 * gate[..., 1]).to(torch.long)
    return {"quotes": quotes, "gates": gate, "hard_gates":
            _hard_gate(gate_logits) if train else gate,
            "probs": probs, "price_logits": price_logits,
            "price_quotes": highs * torch.sigmoid(price_logits),
            "mode": mode}


def target_quotes(policy_out: torch.Tensor) -> torch.Tensor:
    """Deterministic target-actor action: hard gate, no sampling."""
    return forward_policy(policy_out, train=False)["quotes"]


def critic_action(policy_out: torch.Tensor) -> torch.Tensor:
    """What the centralized critic consumes: the executed quote, not the gate."""
    return target_quotes(policy_out)


def fleet_mode(modes: Sequence[int]) -> int:
    """Base-4 encoding of per-unit modes (0..3), four units -> 0..255."""
    out = 0
    for idx, mode in enumerate(modes):
        if not 0 <= int(mode) <= 3:
            raise ValueError("per-unit mode must be 0..3")
        out += int(mode) * (4 ** idx)
    return out


@dataclass
class GateMonitor:
    """The fields a training log needs per unit and per gate."""

    probs: torch.Tensor
    sampled: torch.Tensor
    executed: torch.Tensor
    price_logits: torch.Tensor
    price_quotes: torch.Tensor
    mode: torch.Tensor

    def row(self, unit: str, index: int = 0) -> dict:
        highs = LEG_HIGHS
        executed = self.executed[index].tolist()
        return {
            "unit": unit,
            "gate_prob_charge": float(self.probs[index, 0]),
            "gate_prob_discharge": float(self.probs[index, 1]),
            "gate_sampled_charge": float(self.sampled[index, 0]),
            "gate_sampled_discharge": float(self.sampled[index, 1]),
            "price_logit_charge": float(self.price_logits[index, 0]),
            "price_logit_discharge": float(self.price_logits[index, 1]),
            "price_quote_charge": float(self.price_quotes[index, 0]),
            "price_quote_discharge": float(self.price_quotes[index, 1]),
            "executed_charge": executed[0],
            "executed_discharge": executed[1],
            "near_upper_charge": executed[0] >= 0.99 * highs[0],
            "near_upper_discharge": executed[1] >= 0.99 * highs[1],
            "near_lower_charge": executed[0] <= 0.01 * highs[0],
            "near_lower_discharge": executed[1] <= 0.01 * highs[1],
            "mode": int(self.mode[index]),
        }
