# action_gate.py
"""Pure-function prototype of the truthful gate + continuous quotation map.

Not wired into any policy, action spec or trainer. It exists to make the
proposed mapping exact and testable before it is adopted.

For each quote dimension k (charge, discharge):

    a_k = 0                if g_k = 0              (exactly truthful)
    a_k = h_k * sigmoid(z_k)  if g_k = 1           (legal, non-truthful)

The gate selects the **declaration mode**, not a physical charge/discharge
switch. It is independent per leg, so a unit may declare truthful on charge and
quote a premium on discharge (or the reverse). The continuous branch lies in
(0, h_k] for finite z: it never produces a negative price and never exceeds the
legal upper bound, so no post-hoc clipping is needed.

The truthful point is exact by construction when the gate is closed. It does
not rely on sigmoid/tanh saturation: `g=0` returns 0.0 at any finite z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Upper bounds of the storage quote box, CNY/MWh. These are the existing
#: market bounds; the prototype does not change them.
CHARGE_HIGH = 150.0
DISCHARGE_HIGH = 300.0
LEG_HIGHS = (CHARGE_HIGH, DISCHARGE_HIGH)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic; never overflows for finite input."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def gated_quote(gate: Sequence, logit: Sequence[float],
                high: Sequence[float] = LEG_HIGHS) -> np.ndarray:
    """Map `(gate, logit)` to the quote vector, one leg per dimension.

    `gate` is cast to bool (0/1 accepted). `logit` and `high` must have the
    same shape; `high` must be strictly positive.
    """
    try:
        gate_arr, logit_arr, high_arr = np.broadcast_arrays(
            np.asarray(gate), np.asarray(logit, dtype=np.float64),
            np.asarray(high, dtype=np.float64))
    except ValueError as exc:
        raise ValueError(
            f"gate/logit/high shapes do not broadcast: {exc}") from exc
    gate_arr = gate_arr.astype(bool)
    if not np.all(np.isfinite(logit_arr)):
        raise ValueError("logits must be finite")
    if np.any(high_arr <= 0.0):
        raise ValueError(f"upper bounds must be positive; got {high_arr}")
    continuous = high_arr * _sigmoid(logit_arr)
    return np.where(gate_arr, continuous, 0.0)


def decode_quote(quote: Sequence[float],
                 high: Sequence[float] = LEG_HIGHS) -> tuple:
    """Inverse for analysis: `(gate, logit)` with `nan` on closed legs.

    A quote of exactly 0 is decoded as gate 0. Any positive legal quote is
    decoded as gate 1 with `logit = logit(q/high)`, which is finite only for
    `q < high`; a quote equal to `high` decodes to `+inf`, which the mapping
    accepts as a limit but the trainer would not sample.
    """
    q = np.asarray(quote, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    try:
        q, h = np.broadcast_arrays(q, h)
    except ValueError as exc:
        raise ValueError(f"quote/high shapes do not broadcast: {exc}") from exc
    if np.any(q < 0.0) or np.any(q > h):
        raise ValueError("quote outside the legal box")
    gate = q > 0.0
    logit = np.full_like(q, np.nan)
    with np.errstate(divide="ignore"):
        logit[gate] = np.log(q[gate] / (h[gate] - q[gate]))
    return gate, logit


@dataclass(frozen=True)
class StorageGateAction:
    """The two-leg mapping as a small, testable object."""

    high: tuple = LEG_HIGHS
    truthful: tuple = (0.0, 0.0)

    def __post_init__(self):
        if len(self.high) != 2 or len(self.truthful) != 2:
            raise ValueError("expected charge and discharge legs")
        if any(h <= 0 for h in self.high):
            raise ValueError("upper bounds must be positive")
        if tuple(self.truthful) != (0.0, 0.0):
            raise ValueError("the truthful storage quote is (0, 0)")

    def map(self, gate: Sequence, logit: Sequence[float]) -> np.ndarray:
        return gated_quote(gate, logit, self.high)

    def decode(self, quote: Sequence[float]) -> tuple:
        return decode_quote(quote, self.high)
