"""V5: the storage quote as a price and a quantity per leg.

`da_quote_map` prices a quote on one continuous branch and leaves the offered
quantity at the unit's rated power. This map adds a second fraction per leg, so
a policy can withhold as well as shade:

    charge_price    = gate_ch * 150 * g(z_q0)
    charge_quantity = 0.4 * (1 - gate_ch * (1 - g(z_q1)))
    discharge_price    = gate_dis * 300 * g(z_q2)
    discharge_quantity = 0.4 * (1 - gate_dis * (1 - g(z_q3)))

A closed gate therefore falls to the truthful declaration -- price 0, full
quantity available -- rather than to a zero-price zero-quantity bid that no
mechanism would clear. The quantity fraction is subtracted from 1 so that a
high logit offers more, matching the price branch's reading of a high logit as
more aggressive.

The gate estimator, its bound and the evaluation threshold are the ones in
`da_quote_map`; only the leg structure differs.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from da_quote_map import bound_gate_logits, map_fraction

GATES_PER_UNIT = 2
#: Two gate logits plus one price and one quantity logit per leg.
POLICY_DIM = 6
QUOTE_DIM = 4
#: Upper bound of each quote in `QUOTE_DIM` order.
ACTION_BOX_HIGH = (150.0, 0.4, 300.0, 0.4)
#: The closed-gate quote, and the point a truthful declaration sits at.
TRUTHFUL = (0.0, 0.4, 0.0, 0.4)


def _split(policy_out: torch.Tensor):
    if policy_out.shape[-1] != POLICY_DIM:
        raise ValueError(
            f"policy output must end in {POLICY_DIM}; got "
            f"{tuple(policy_out.shape)}")
    return policy_out[..., :GATES_PER_UNIT], policy_out[..., GATES_PER_UNIT:]


def forward_policy_v5(policy_out: torch.Tensor, *, train: bool = False,
                      generator: Optional[torch.Generator] = None,
                      gate_temperature: float = 1.0,
                      straight_through: bool = True) -> Dict[str, torch.Tensor]:
    """Map the six policy logits to quotes, gates and diagnostics.

    `gate_temperature` acts on the gate logits only; evaluation always uses the
    hard threshold at 0.
    """
    if gate_temperature <= 0:
        raise ValueError("gate_temperature must be positive")
    gate_logits, quote_logits = _split(policy_out)
    gate_logits = bound_gate_logits(gate_logits)
    probs = torch.sigmoid(gate_logits / gate_temperature)
    if train:
        draw = torch.rand(probs.shape, generator=generator,
                          dtype=probs.dtype)
        hard = (draw < probs).to(probs.dtype)
        gate = hard + (probs - probs.detach()) if straight_through else hard
    else:
        gate = (gate_logits > 0.0).to(policy_out.dtype)
    frac = map_fraction(quote_logits)
    charge_price = gate[..., 0] * ACTION_BOX_HIGH[0] * frac[..., 0]
    charge_qty = ACTION_BOX_HIGH[1] * (1.0 - gate[..., 0] * (1.0 - frac[..., 1]))
    discharge_price = gate[..., 1] * ACTION_BOX_HIGH[2] * frac[..., 2]
    discharge_qty = ACTION_BOX_HIGH[3] * (1.0 - gate[..., 1] * (1.0 - frac[..., 3]))
    quotes = torch.stack([charge_price, charge_qty,
                          discharge_price, discharge_qty], -1)
    mode = (gate[..., 0] + 2.0 * gate[..., 1]).to(torch.long)
    return {
        "quotes": quotes,
        "gates": gate,
        "probs": probs,
        "gate_logits": gate_logits,
        "quote_logits": quote_logits,
        "mode": mode,
    }


def target_quotes_v5(policy_out: torch.Tensor) -> torch.Tensor:
    """Deterministic target/evaluation action: hard gate at 0, gate tau = 1."""
    return forward_policy_v5(policy_out, train=False)["quotes"]
