"""D1: linear-core / quadratic-tail continuous quote map.

The map is the candidate C audited in
`reports/v2_cy_jn_2025/continuous_quote_redesign_comparison.md`:

    g(z) = 0.5 + 0.196 z                               |z| <= 2.5
    g(z) = 1 - 0.01 / (1 + 9.8 (z - 2.5))^2             z > 2.5
    g(z) = 0.01 / (1 + 9.8 (-z - 2.5))^2                z < -2.5

with `quote = high * g(z)` on an open gate and exactly 0 on a closed gate.
The map is monotone, bounded, finite and needs no quote clipping. The logit is
clamped only at a numerical guard far outside the usable range so that the
quadratic denominator cannot overflow.

This module is independent of the trainer. The gate estimator is the same
hard-forward / straight-through-backward Bernoulli as the phase-16 adapter;
only the price branch changes.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

LEG_HIGHS = (150.0, 300.0)
POLICY_DIM = 4
QUOTE_DIM = 2
GATES_PER_UNIT = 2

#: v4: gate logits are bounded as B * tanh(raw / B). With B = 4 the open
#: probability sigma(bound / tau) stays inside [sigmoid(-4), sigmoid(4)] at
#: tau = 1 — a closed gate keeps a ~2% reopen draw per block and the
#: straight-through gradient p * (1 - p) never vanishes, so the v3 absorbing
#: gate collapse (closed forever once the logit crossed zero far enough) is
#: removed by construction. The hard threshold at 0 is untouched, so
#: evaluation semantics do not change.
GATE_LOGIT_BOUND = 4.0


def bound_gate_logits(gate_logits: torch.Tensor) -> torch.Tensor:
    """Smooth B * tanh(raw / B) bound; differentiable everywhere."""
    return GATE_LOGIT_BOUND * torch.tanh(gate_logits / GATE_LOGIT_BOUND)

Q_HI = 0.99
Z_CORE = 2.5
SLOPE = (Q_HI - 0.5) / Z_CORE
TAIL_B = SLOPE / (2.0 * (1.0 - Q_HI))

# The quadratic tail denominator grows like z^2; clamping at 1e6 keeps it
# finite in float32/float64 without changing any quote a policy can use.
Z_GUARD = 1.0e6


def _guard(z):
    return torch.clamp(z, -Z_GUARD, Z_GUARD)


def map_fraction(z: torch.Tensor) -> torch.Tensor:
    """g(z) in (0, 1); scalar tensors and arbitrary shapes are accepted."""
    z = _guard(z)
    core = 0.5 + SLOPE * z
    upper = 1.0 - (1.0 - Q_HI) / (1.0 + TAIL_B * (z - Z_CORE)) ** 2
    lower = (1.0 - Q_HI) / (1.0 + TAIL_B * (-z - Z_CORE)) ** 2
    return torch.where(z > Z_CORE, upper,
                       torch.where(z < -Z_CORE, lower, core))


def map_derivative(z: torch.Tensor) -> torch.Tensor:
    """dg/dz in (0, SLOPE]; finite for every finite z."""
    z = _guard(z)
    upper = SLOPE / (1.0 + TAIL_B * (z - Z_CORE)) ** 3
    lower = SLOPE / (1.0 + TAIL_B * (-z - Z_CORE)) ** 3
    return torch.where(z > Z_CORE, upper,
                       torch.where(z < -Z_CORE, lower,
                                   torch.full_like(z, SLOPE)))


def map_fraction_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -Z_GUARD, Z_GUARD)
    core = 0.5 + SLOPE * z
    upper = 1.0 - (1.0 - Q_HI) / (1.0 + TAIL_B * (z - Z_CORE)) ** 2
    lower = (1.0 - Q_HI) / (1.0 + TAIL_B * (-z - Z_CORE)) ** 2
    return np.where(z > Z_CORE, upper, np.where(z < -Z_CORE, lower, core))


def map_derivative_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -Z_GUARD, Z_GUARD)
    upper = SLOPE / (1.0 + TAIL_B * (z - Z_CORE)) ** 3
    lower = SLOPE / (1.0 + TAIL_B * (-z - Z_CORE)) ** 3
    return np.where(z > Z_CORE, upper,
                    np.where(z < -Z_CORE, lower, np.full_like(z, SLOPE)))


class _STGate(torch.autograd.Function):
    """Hard forward; sigmoid-derivative surrogate backward (biased)."""

    @staticmethod
    def forward(ctx, logit: torch.Tensor, prob: torch.Tensor):
        ctx.save_for_backward(prob)
        return (logit > 0.0).to(logit.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        prob, = ctx.saved_tensors
        return grad_output * prob * (1.0 - prob), None


def _split(policy_out: torch.Tensor):
    if policy_out.shape[-1] != POLICY_DIM:
        raise ValueError(
            f"policy output must end in {POLICY_DIM}; got "
            f"{tuple(policy_out.shape)}")
    return policy_out[..., :GATES_PER_UNIT], policy_out[..., GATES_PER_UNIT:]


def forward_policy_v3(policy_out: torch.Tensor, *, train: bool = False,
                      generator: Optional[torch.Generator] = None,
                      gate_temperature: float = 1.0,
                      straight_through: bool = True) -> Dict[str, torch.Tensor]:
    """Map the four policy logits to quotes, gates and diagnostics.

    `gate_temperature` acts on the gate logits only. It never touches the
    continuous price map. Evaluation always uses the hard threshold at 0.
    """
    if gate_temperature <= 0:
        raise ValueError("gate_temperature must be positive")
    gate_logits, price_logits = _split(policy_out)
    gate_logits = bound_gate_logits(gate_logits)
    probs = torch.sigmoid(gate_logits / gate_temperature)
    if train:
        draw = torch.rand(probs.shape, generator=generator,
                          dtype=probs.dtype)
        hard = (draw < probs).to(probs.dtype)
        gate = hard + (probs - probs.detach()) if straight_through else hard
    else:
        gate = (gate_logits > 0.0).to(policy_out.dtype)
    highs = torch.as_tensor(LEG_HIGHS, dtype=policy_out.dtype,
                            device=policy_out.device)
    price_quotes = highs * map_fraction(price_logits)
    quotes = gate * price_quotes
    mode = (gate[..., 0] + 2.0 * gate[..., 1]).to(torch.long)
    return {
        "quotes": quotes,
        "gates": gate,
        "hard_gates": (gate_logits > 0.0).to(policy_out.dtype)
        if train else gate,
        "probs": probs,
        "price_logits": price_logits,
        "price_quotes": price_quotes,
        "mode": mode,
    }


def target_quotes_v3(policy_out: torch.Tensor) -> torch.Tensor:
    """Deterministic target/evaluation action: hard gate, gate τ=1."""
    return forward_policy_v3(policy_out, train=False,
                             gate_temperature=1.0)["quotes"]


def critic_action_v3(policy_out: torch.Tensor) -> torch.Tensor:
    return target_quotes_v3(policy_out)


def fleet_mode(modes: Sequence[int]) -> int:
    out = 0
    for idx, mode in enumerate(modes):
        if not 0 <= int(mode) <= 3:
            raise ValueError("per-unit mode must be 0..3")
        out += int(mode) * (4 ** idx)
    return out


if __name__ == "__main__":
    z = np.linspace(-50.0, 50.0, 200001)
    g = map_fraction_np(z)
    d = map_derivative_np(z)
    assert np.all(np.diff(g) >= -1e-15), "not monotone"
    assert np.all(g >= 0.0) and np.all(g <= 1.0), "outside [0,1]"
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(d))
    assert abs(map_fraction_np(np.array([0.0]))[0] - 0.5) < 1e-15
    # Numerical derivative at a variety of points, including the joint.
    for point in (-10.0, -3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0, 10.0):
        h = 1e-6
        num = (map_fraction_np(np.array([point + h]))[0]
               - map_fraction_np(np.array([point - h]))[0]) / (2 * h)
        ana = map_derivative_np(np.array([point]))[0]
        # The second derivative jumps at the joint, so a central difference
        # there is first-order accurate; elsewhere it is second-order.
        tol = 1e-5 if abs(abs(point) - Z_CORE) < 1e-12 else 1e-8
        assert abs(num - ana) < tol, (point, num, ana)
    # Torch parity and gate-closed exact zero.
    zt = torch.tensor([-10.0, -2.5, 0.0, 2.5, 10.0])
    assert np.allclose(map_fraction(zt).numpy(), map_fraction_np(zt.numpy()))
    out = forward_policy_v3(torch.tensor([[10.0, 10.0, 5.0, 5.0]]),
                            train=False)
    closed = forward_policy_v3(torch.tensor([[-10.0, -10.0, 5.0, 5.0]]),
                               train=False)
    assert float(closed["quotes"].sum()) == 0.0
    assert float(out["quotes"].sum()) > 0.0
    print("da_quote_map self-check ok")
