# audit_reward_design_facts.py
"""Read-only facts behind the seventh-phase design review.

Extracts the discount, horizon, action bounds and terminal treatment from the
code so the design documents cite numbers rather than memory. No model is built
and no clearing runs.

Usage:
    PYTHONPATH=src python src/audit_reward_design_facts.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

import dispatch_core
import models
from rl_bidding import MATD3
from rl_env import BLOCK_SIZE, N_BLOCKS
from rl_spec import ACTION_STORAGE_QUOTE_V2

OUT = "data/processed/v2_cy_jn_2025/pilot/reward_design_facts.json"


def facts() -> dict:
    signature = inspect.signature(MATD3.__init__)
    gamma = float(signature.parameters["gamma"].default)
    weights = [gamma ** t for t in range(N_BLOCKS)]
    bounds = ACTION_STORAGE_QUOTE_V2.bounds
    truthful = ACTION_STORAGE_QUOTE_V2.truthful
    return {
        "gamma_per_block": gamma,
        "block_size_periods": BLOCK_SIZE,
        "blocks_per_episode": N_BLOCKS,
        "episode_periods": BLOCK_SIZE * N_BLOCKS,
        "block_hours": BLOCK_SIZE * 0.25,
        "discount_weights": weights,
        "weight_sum": float(sum(weights)),
        "weight_last": float(weights[-1]),
        "first_over_last": float(weights[0] / weights[-1]),
        "effective_discount_per_15min": float(gamma ** 0.25),
        "action_bounds": {k: list(v) for k, v in bounds.items()},
        "truthful": dict(truthful),
        "truthful_is_lower_bound": {
            k: abs(float(truthful[k]) - float(bounds[k][0])) < 1e-12
            for k in truthful},
        "box_midpoint": {k: (float(bounds[k][0]) + float(bounds[k][1])) / 2.0
                         for k in bounds},
        "truthful_at_tanh": {
            k: (2.0 * (float(truthful[k]) - float(bounds[k][0]))
                / (float(bounds[k][1]) - float(bounds[k][0])) - 1.0)
            for k in bounds},
        "reward_standardized": "buffer.reward_std() divides every reward "
                               "before the critic target",
        "terminal_bootstrap": "gamma * (1 - done) * min(target Q)",
        "terminal_soc_in_training_window": bool(
            dispatch_core.pins_terminal_soc(
                models.MarketConfig.from_yaml(), "window")),
        "terminal_soc_in_settlement": bool(
            dispatch_core.pins_terminal_soc(
                models.MarketConfig.from_yaml(), "full_day")),
        "reward_unit": "CNY per block (4 periods), before standardization",
    }


def main(argv=None) -> int:
    payload = facts()
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "gamma_per_block", "weight_sum", "weight_last", "first_over_last",
        "effective_discount_per_15min", "truthful", "box_midpoint",
        "truthful_at_tanh")}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
