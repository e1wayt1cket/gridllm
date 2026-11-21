# c2_reward.py
"""C2 potential-based shaping on top of the day differential B.

    r_{i,t} = r^B_{i,t} + gamma * Phi_i(s_{t+1}, t+1) - Phi_i(s_t, t)

with the time-decaying SOC potential

    Phi_i(soc, t) = (T - t) / T * kappa * e_max * (soc - soc0).

It is zero at t = T for any terminal SOC, bounded by
`kappa * e_max * max(soc0 - soc_lo, soc_hi - soc0)`, and uses only the agent's
own SOC and the block index. No actual future price, opponent action or
counterfactual result enters it.

The module does not import the old block-local reward A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

T = 24
GAMMA = 0.99
#: The frozen reference price used elsewhere in the project. Declared here so
#: this module stays a leaf; the value is a constant, not fitted per run.
KAPPA = 288.19757259048833
E_MAX = 1.6
SOC0 = 0.5
SOC_LO = 0.10
SOC_HI = 0.90


@dataclass(frozen=True)
class C2Config:
    gamma: float = GAMMA
    blocks: int = T
    kappa: float = KAPPA
    e_max: float = E_MAX
    soc0: float = SOC0
    soc_lo: float = SOC_LO
    soc_hi: float = SOC_HI

    def __post_init__(self):
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.blocks != T:
            raise ValueError(f"the episode is {T} blocks")
        if not self.soc_lo < self.soc0 < self.soc_hi:
            raise ValueError("initial SOC must lie inside the band")
        if self.kappa <= 0 or self.e_max <= 0:
            raise ValueError("kappa and e_max must be positive")

    @property
    def bound(self) -> float:
        return float(self.kappa * self.e_max
                     * max(self.soc0 - self.soc_lo, self.soc_hi - self.soc0))


def f_soc(soc, config: C2Config):
    soc_arr = np.asarray(soc, dtype=float)
    # The clearing enforces [soc_lo, soc_hi] on its own variables. The
    # committed SOC read back from a solved window can overshoot that bound by
    # solver tolerance, which is not a policy violation; absorb only that
    # numerical overshoot and clip. A real excursion is still refused.
    tol = 1e-6
    if np.any(soc_arr < config.soc_lo - tol) \
            or np.any(soc_arr > config.soc_hi + tol):
        raise ValueError(
            "SOC outside the declared band: "
            f"min={float(np.min(soc_arr)):.9f}, "
            f"max={float(np.max(soc_arr)):.9f}, "
            f"band=[{config.soc_lo}, {config.soc_hi}]")
    soc_arr = np.clip(soc_arr, config.soc_lo, config.soc_hi)
    return config.kappa * config.e_max * (soc_arr - config.soc0)


def potential(soc, t, config: Optional[C2Config] = None):
    """Phi(soc, t); exactly 0 at t = blocks."""
    config = config or C2Config()
    if np.any(np.asarray(t) < 0) or np.any(np.asarray(t) > config.blocks):
        raise ValueError("t must lie in [0, blocks]")
    return (config.blocks - np.asarray(t, dtype=float)) / config.blocks \
        * f_soc(soc, config)


def shaping(soc_t, soc_next, t, config: Optional[C2Config] = None):
    config = config or C2Config()
    return config.gamma * potential(soc_next, t + 1, config) \
        - potential(soc_t, t, config)


@dataclass
class C2Reward:
    reward: Dict[str, np.ndarray]
    base: Dict[str, np.ndarray]
    shaping_terms: Dict[str, np.ndarray]
    episode_return: Dict[str, float]
    boundary: Dict[str, float]
    potential_start: Dict[str, float]

    def as_dict(self) -> dict:
        return {
            "base_sum": {k: float(np.sum(v)) for k, v in self.base.items()},
            "shaping_sum": {k: float(np.sum(v))
                            for k, v in self.shaping_terms.items()},
            "episode_return": dict(self.episode_return),
            "boundary": dict(self.boundary),
            "potential_start": dict(self.potential_start),
        }


def assert_complete(soc_path: Mapping[str, Sequence[float]], done=None,
                    blocks: int = T) -> None:
    """Refuse a path that does not cover all blocks or ended early."""
    for unit, values in soc_path.items():
        if len(values) != blocks + 1:
            raise ValueError(f"{unit}: SOC path must have {blocks + 1} states")
    if done is not None:
        done_arr = np.asarray(done, dtype=bool)
        if done_arr.shape != (blocks,):
            raise ValueError(f"done must have {blocks} entries")
        if done_arr[:-1].any():
            raise ValueError("mid-episode termination is not supported; define "
                             "a terminal handling before using C2")


def c2_reward(base_reward: Mapping[str, Sequence[float]],
              soc_path: Mapping[str, Sequence[float]], *,
              done=None, config: Optional[C2Config] = None) -> C2Reward:
    """Add the shaping to the B reward. `base_reward` is paid at the end."""
    config = config or C2Config()
    assert_complete(soc_path, done, config.blocks)
    reward, base_out, terms = {}, {}, {}
    episode_return, boundary, pot0 = {}, {}, {}
    for unit, values in base_reward.items():
        b = np.asarray(values, dtype=float)
        if b.shape != (config.blocks,):
            raise ValueError(f"{unit}: base reward must have {config.blocks} "
                             "blocks")
        soc = np.asarray(soc_path[unit], dtype=float)
        if abs(soc[0] - config.soc0) > 1e-9:
            raise ValueError(
                f"{unit}: initial SOC {soc[0]} differs from the declared "
                f"{config.soc0}; a different initial state is a different "
                "episode and must be registered as such")
        shaping_arr = np.array([
            float(shaping(soc[t], soc[t + 1], t, config))
            for t in range(config.blocks)])
        shaped = b + shaping_arr
        reward[unit] = shaped
        base_out[unit] = b
        terms[unit] = shaping_arr
        weights = config.gamma ** np.arange(config.blocks)
        episode_return[unit] = float(np.sum(weights * shaped))
        boundary[unit] = float(
            config.gamma ** config.blocks
            * potential(soc[config.blocks], config.blocks, config)
            - potential(soc[0], 0, config))
        pot0[unit] = float(potential(soc[0], 0, config))
    return C2Reward(reward=reward, base=base_out, shaping_terms=terms,
                    episode_return=episode_return, boundary=boundary,
                    potential_start=pot0)
