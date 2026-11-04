# daily_reward_b.py
"""Whole-day differential reward B under the 24-block rolling interface.

The policy arm and the B1 arm both run the same 24 blocks, commit four periods
per block, and settle each block through `participant_payoff` on the committed
periods. B1 is therefore the **rolling** truthful arm registered by
`b1_daily_baseline.py`, not the one-shot JN-Base clearing.

    R_i = G_i^policy - G_i^B1

is paid once, at the last block. The original components are kept; a fixed
positive scale may be applied, never a running `reward_std()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

T = 24
GAMMA = 0.99
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")


class CacheKeyError(KeyError):
    """A B1 lookup whose key does not match the registered one."""


@dataclass(frozen=True)
class RewardBConfig:
    """Frozen reward configuration. `fixed_scale` is a declared constant."""

    gamma: float = GAMMA
    blocks: int = T
    fixed_scale: float = 1.0
    terminal_block: int = T - 1

    def __post_init__(self):
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.blocks != T:
            raise ValueError(f"the episode is {T} blocks")
        if self.fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive")
        if self.terminal_block != self.blocks - 1:
            raise ValueError("the day reward is paid at the last block")


@dataclass
class B1Day:
    """One registered rolling-truthful day."""

    pair_id: str
    soc0: float
    config_hash: str
    data_hash: str
    block_payoffs: Dict[str, Sequence[float]]
    soc_end: Dict[str, float]


def cache_key(pair_id: str, soc0: float, config_hash: str,
              data_hash: str) -> tuple:
    return (str(pair_id), float(soc0), str(config_hash), str(data_hash))


class B1Cache:
    """Strict, never-fallback B1 lookup."""

    def __init__(self):
        self._days: Dict[tuple, B1Day] = {}

    def add(self, day: B1Day) -> None:
        key = cache_key(day.pair_id, day.soc0, day.config_hash, day.data_hash)
        if key in self._days:
            raise ValueError(f"duplicate B1 key {key}")
        self._days[key] = day

    def get(self, pair_id: str, soc0: float, config_hash: str,
            data_hash: str) -> B1Day:
        key = cache_key(pair_id, soc0, config_hash, data_hash)
        if key not in self._days:
            raise CacheKeyError(
                f"no B1 for {key}; a changed day, SOC, config or data hash "
                "must be re-solved, not approximated")
        return self._days[key]

    def __len__(self):
        return len(self._days)


@dataclass
class DayReward:
    """All components of one day, kept apart."""

    pair_id: str
    G_policy: Dict[str, float]
    G_b1: Dict[str, float]
    delta: Dict[str, float]
    scaled_delta: Dict[str, float]
    reward: Dict[str, np.ndarray]
    discounted_return: Dict[str, float]
    paid_at: int
    fixed_scale: float
    gamma: float

    def as_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "G_policy": dict(self.G_policy),
            "G_b1": dict(self.G_b1),
            "delta": dict(self.delta),
            "scaled_delta": dict(self.scaled_delta),
            "reward_sum": {k: float(np.sum(v)) for k, v in self.reward.items()},
            "discounted_return": dict(self.discounted_return),
            "paid_at": self.paid_at,
            "fixed_scale": self.fixed_scale,
            "gamma": self.gamma,
        }


def _sum_paths(paths: Mapping[str, Sequence[float]], blocks: int,
               name: str) -> Dict[str, float]:
    out = {}
    for unit, values in paths.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (blocks,):
            raise ValueError(f"{name} {unit}: expected {blocks} blocks, "
                             f"got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} {unit}: non-finite payoff")
        out[unit] = float(arr.sum())
    return out


def day_reward_b(policy_block_payoffs: Mapping[str, Sequence[float]],
                 b1_block_payoffs: Mapping[str, Sequence[float]],
                 *, pair_id: str,
                 config: Optional[RewardBConfig] = None) -> DayReward:
    """Construct the day differential from the two rolling arms."""
    config = config or RewardBConfig()
    policy = _sum_paths(policy_block_payoffs, config.blocks, "policy")
    baseline = _sum_paths(b1_block_payoffs, config.blocks, "b1")
    units = sorted(policy)
    if set(units) != set(baseline):
        raise ValueError("policy and B1 must cover the same units")
    delta = {u: policy[u] - baseline[u] for u in units}
    scaled = {u: delta[u] / config.fixed_scale for u in units}
    reward = {}
    discounted = {}
    for u in units:
        arr = np.zeros(config.blocks, dtype=float)
        arr[config.terminal_block] = scaled[u]
        reward[u] = arr
        discounted[u] = float(config.gamma ** config.terminal_block * scaled[u])
    return DayReward(pair_id=pair_id, G_policy=policy, G_b1=baseline,
                     delta=delta, scaled_delta=scaled, reward=reward,
                     discounted_return=discounted,
                     paid_at=config.terminal_block,
                     fixed_scale=config.fixed_scale, gamma=config.gamma)


def day_reward_b_from_cache(policy_block_payoffs, cache: B1Cache,
                            pair_id: str, soc0: float, config_hash: str,
                            data_hash: str, config: Optional[RewardBConfig] = None,
                            ) -> DayReward:
    day = cache.get(pair_id, soc0, config_hash, data_hash)
    return day_reward_b(policy_block_payoffs, day.block_payoffs,
                        pair_id=pair_id, config=config)
