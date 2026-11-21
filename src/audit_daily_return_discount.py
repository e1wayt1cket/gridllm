# audit_daily_return_discount.py
"""Numerical checks of the discount facts and the C2 telescope. No training."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

from rl_bidding import MATD3
from rl_env import BLOCK_SIZE, N_BLOCKS

OUT = "data/processed/v2_cy_jn_2025/pilot/c2_identity_check.json"


def main(argv=None) -> int:
    gamma = float(inspect.signature(MATD3.__init__).parameters["gamma"].default)
    T = N_BLOCKS
    weights = np.array([gamma ** t for t in range(T)])
    rng = np.random.default_rng(20261101)

    # B: one terminal reward.
    day_rewards = rng.normal(0.0, 500.0, size=1000)
    base_returns = np.array([gamma ** (T - 1) * r for r in day_rewards])

    # C2: random potentials along random paths; the telescope must hold exactly.
    errors = []
    boundary_same = []
    for _ in range(100):
        phi = rng.normal(0.0, 100.0, size=T + 1)
        shaped = np.array([
            gamma * phi[t + 1] - phi[t] for t in range(T)])
        left = float(np.sum(weights * shaped))
        right = float(gamma ** T * phi[T] - phi[0])
        errors.append(abs(left - right))
        # Two paths with the same boundary but different intermediate states
        # must have the same total shaping.
        phi2 = phi.copy()
        phi2[1:-1] = rng.normal(0.0, 100.0, size=T - 1)
        shaped2 = np.array([
            gamma * phi2[t + 1] - phi2[t] for t in range(T)])
        boundary_same.append(
            abs(float(np.sum(weights * shaped2)) - right))

    # Terminal-zero variant: with Phi(s_T)=0 the total is -Phi(s_0).
    phi = rng.normal(0.0, 100.0, size=T + 1)
    phi[T] = 0.0
    shaped = np.array([gamma * phi[t + 1] - phi[t] for t in range(T)])
    terminal_zero_error = abs(float(np.sum(weights * shaped)) + phi[0])

    payload = {
        "gamma_per_block": gamma,
        "block_size_periods": BLOCK_SIZE,
        "blocks_per_episode": T,
        "block_hours": BLOCK_SIZE * 0.25,
        "weight_sum": float(weights.sum()),
        "weight_first": float(weights[0]),
        "weight_last": float(weights[-1]),
        "first_over_last": float(weights[0] / weights[-1]),
        "effective_discount_per_15min": float(gamma ** 0.25),
        "terminal_block_discount": float(gamma ** (T - 1)),
        "base_return_over_day_reward": float(
            np.mean(base_returns / day_rewards)),
        "telescope_max_abs_error": float(max(errors)),
        "telescope_boundary_only_max_abs_error": float(max(boundary_same)),
        "terminal_zero_max_abs_error": float(terminal_zero_error),
        "identity": "sum_t gamma^t (gamma Phi(s_{t+1}) - Phi(s_t)) "
                    "= gamma^T Phi(s_T) - Phi(s_0)",
        "episode_length_fixed": True,
        "early_termination_in_env": False,
        "reward_standardization": "buffer.reward_std() divides the batch at "
                                  "update time; it is not a fixed scale",
        "terminal_soc_training_window": False,
        "terminal_soc_settlement": True,
    }
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
