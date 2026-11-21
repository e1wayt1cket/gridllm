"""D2: replay buffer with joint-cell stratified sampling.

The buffer stores one joint cell per transition. The cell is built only from
the OBS_V6 observation and the executed action, so it cannot use a future LMP:

* block band (0-7, 8-15, 16-23);
* historical price band (`avg_lmp_hist_norm` < 0.5, 0.5-1.0, >= 1.0);
* load band (mean `load_feat` < 0.5 / >= 0.5);
* fleet SOC band (mean SOC < 0.5 / >= 0.5);
* discharge quote band (max quote / 300 < 0.33, 0.33-0.66, >= 0.66);
* gate band (no open discharge gate / at least one);
* reward-magnitude band (max |scaled reward| > 0.02 / not).

Sampling is a declared mixture: `uniform_fraction` uniform plus the remainder
from joint cells with weight proportional to `count ** beta`. The per-cell
weight is capped relative to the uniform distribution, so a cell with a handful
of transitions cannot dominate. The critic target is still the conditional Q;
the mixture is the declared training distribution. The sampler records the
per-transition probability, the uniform-ratio (the importance weight that would
recover a uniform objective) and its effective sample size, but does not apply
the ratio as a loss weight.

Read-only with respect to the environment; the buffer is the trainer's own.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Mapping

import numpy as np

from rl_b_c2_gate import GateReplayBuffer

N_CELLS = 3 * 3 * 2 * 2 * 3 * 2 * 2


def _band(value: float, cuts) -> int:
    for i, cut in enumerate(cuts):
        if value < cut:
            return i
    return len(cuts)


def stratum_key(obs, action, reward, gates) -> int:
    """Joint cell key from one transition's observation and executed action."""
    obs = np.asarray(obs, dtype=float)
    action = np.asarray(action, dtype=float)
    reward = np.asarray(reward, dtype=float)
    gates = np.asarray(gates, dtype=float)
    block = int(round(float(obs[0, 4]) * 23.0))
    block_band = min(block // 8, 2)
    price_band = _band(float(obs[0, 5]), (0.5, 1.0))
    load_band = _band(float(np.mean(obs[:, 0])), (0.5,))
    soc_band = _band(float(np.mean(obs[:, 2])), (0.5,))
    quote_band = _band(float(np.max(action[:, 1]) / 300.0), (0.33, 0.66))
    gate_band = 1 if np.any(gates[:, 1] > 0) else 0
    reward_band = 1 if float(np.max(np.abs(reward))) > 0.02 else 0
    return (((((block_band * 3 + price_band) * 2 + load_band) * 2
              + soc_band) * 3 + quote_band) * 2 + gate_band) * 2 \
        + reward_band


def decode_key(key: int) -> dict:
    key = int(key)
    reward_band = key % 2
    key //= 2
    gate_band = key % 2
    key //= 2
    quote_band = key % 3
    key //= 3
    soc_band = key % 2
    key //= 2
    load_band = key % 2
    key //= 2
    price_band = key % 3
    key //= 3
    block_band = key
    return {"block_band": block_band, "price_band": price_band,
            "load_band": load_band, "soc_band": soc_band,
            "quote_band": quote_band, "gate_band": gate_band,
            "reward_band": reward_band}


class DAReplayBuffer(GateReplayBuffer):
    """Gate replay plus the joint cell, the exploration flag and the sampler."""

    def __init__(self, capacity: int = 100_000, *,
                 uniform_fraction: float = 0.5, beta: float = 0.5,
                 max_upweight: float = 5.0):
        super().__init__(capacity)
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0,1]")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if max_upweight < 1.0:
            raise ValueError("max_upweight must be at least 1")
        self.extras["explored"] = deque(maxlen=self.capacity)
        self.extras["exploration_source"] = deque(maxlen=self.capacity)
        self.extras["stratum"] = deque(maxlen=self.capacity)
        self.config = {"uniform_fraction": float(uniform_fraction),
                       "beta": float(beta),
                       "max_upweight": float(max_upweight)}
        self.last_stats: Dict[str, object] = {}

    def add(self, obs, action, reward, next_obs, done, extras: dict) -> None:
        obs = np.asarray(obs, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float32).copy()
        reward_arr = np.asarray(reward, dtype=np.float32).copy()
        gates = np.asarray(extras["gates"], dtype=np.float32)
        extras = dict(extras)
        extras["stratum"] = np.asarray(
            [stratum_key(obs, action, reward_arr, gates)], dtype=np.float32)
        self.buffer.append((
            obs, action, reward_arr,
            np.asarray(next_obs, dtype=np.float32).copy(),
            float(done),
        ))
        for key in self.extras:
            self.extras[key].append(
                np.asarray(extras[key], dtype=np.float32).copy())

    def sample(self, batch_size: int, *, uniform_fraction: float | None = None,
               beta: float | None = None,
               max_upweight: float | None = None) -> dict:
        n = len(self.buffer)
        if n == 0:
            raise ValueError("cannot sample an empty buffer")
        m = min(batch_size, n)
        uf = self.config["uniform_fraction"] if uniform_fraction is None \
            else float(uniform_fraction)
        b = self.config["beta"] if beta is None else float(beta)
        cap = self.config["max_upweight"] if max_upweight is None \
            else float(max_upweight)
        keys = np.asarray(self.extras["stratum"], dtype=float).reshape(-1)
        keys = np.rint(keys).astype(int)
        counts = np.bincount(keys, minlength=N_CELLS)
        nonempty = np.flatnonzero(counts > 0)
        cell_weight = counts[nonempty].astype(float) ** b
        cell_weight /= cell_weight.sum()
        p_cell = dict(zip(nonempty.tolist(), cell_weight.tolist()))
        p_strat = np.zeros(n, dtype=float)
        for cell in nonempty:
            idx = np.flatnonzero(keys == cell)
            p_strat[idx] = p_cell[int(cell)] / len(idx)
        # Cap each transition's stratified share against the uniform share.
        p_strat = np.minimum(p_strat, cap / n)
        p_strat /= p_strat.sum()
        p = (1.0 - uf) / n + uf * p_strat
        p /= p.sum()
        idx = np.random.choice(n, size=m, replace=True, p=p)
        rows = [self.buffer[i] for i in idx]
        obs, act, rew, next_obs, done = zip(*rows)
        import torch
        out = {
            "obs": torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            "actions": torch.as_tensor(np.asarray(act), dtype=torch.float32),
            "rewards": torch.as_tensor(np.asarray(rew), dtype=torch.float32),
            "next_obs": torch.as_tensor(np.asarray(next_obs),
                                        dtype=torch.float32),
            "done": torch.as_tensor(np.asarray(done), dtype=torch.float32),
            "indices": idx,
            "sample_prob": p[idx],
            "uniform_ratio": (1.0 / n) / p[idx],
            "stratum": keys[idx],
        }
        for key, values in self.extras.items():
            out[key] = torch.as_tensor(
                np.asarray([values[i] for i in idx]), dtype=torch.float32)
        ess = float(1.0 / np.sum(p ** 2))
        out["sampler_stats"] = {
            "n": int(n), "batch": int(m),
            "uniform_fraction": uf, "beta": b, "max_upweight": cap,
            "cells_nonempty": int(len(nonempty)),
            "cell_count_min": int(counts[nonempty].min()),
            "cell_count_median": float(np.median(counts[nonempty])),
            "cell_count_max": int(counts[nonempty].max()),
            "sample_prob_min": float(np.min(p)),
            "sample_prob_max": float(np.max(p)),
            "uniform_ratio_min": float(np.min((1.0 / n) / p)),
            "uniform_ratio_max": float(np.max((1.0 / n) / p)),
            "ess": ess,
            "ess_fraction": ess / n,
            "sampled_cells": int(len(np.unique(keys[idx]))),
        }
        self.last_stats = out["sampler_stats"]
        return out

    def sampler_state(self) -> dict:
        keys = np.asarray(self.extras["stratum"], dtype=float).reshape(-1)
        keys = np.rint(keys).astype(int)
        counts = np.bincount(keys, minlength=N_CELLS) if len(keys) else \
            np.zeros(N_CELLS, dtype=int)
        return {"config": dict(self.config),
                "n": int(len(keys)),
                "counts": {str(i): int(c) for i, c in
                           enumerate(counts) if c > 0},
                "decoded": {str(i): decode_key(i) for i, c in
                            enumerate(counts) if c > 0},
                "last_stats": dict(self.last_stats)}
