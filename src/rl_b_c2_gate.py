"""MATD3_B_C2_GATE: daily differential reward plus truthful-gate quotes.

This trainer is a separate branch. It does not modify the old MATD3 entry
points, the pilot artifacts, or their reward definition.

Components
----------
- Actors emit four logits per ESS: two gate logits and two continuous price
  logits. `truthful_gate_actor_adapter` turns that into the executed quote.
  Training samples the gate from Bernoulli(prob) and uses the straight-through
  surrogate for the gate gradient; evaluation uses the deterministic hard gate.
- The centralized critic consumes the **executed quote** for every ESS, so its
  action space is the market's, not the gate's.
- The reward is the whole-day differential B, paid at block 23. The C2 mode adds
  the time-decaying SOC potential every block. A fixed positive scale is
  applied; no running reward standardisation is used.
- `GateReplayBuffer` stores the executed quote, gate value, gate probability,
  gate logit, price logit and the Bernoulli draw, so a resumed run can continue
  the same estimator.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from c2_reward import C2Config, c2_reward
from checkpoint_loader_v2 import load_checkpoint, restore_checkpoint
from checkpoint_writer_v2 import save_checkpoint
from daily_reward_b import (B1Cache, B1Day, RewardBConfig,
                            day_reward_b_from_cache)
from rl_bidding import CentralizedCritic
from rl_env import BLOCK_SIZE, BiddingEnv, N_BLOCKS
from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V6
from truthful_gate_actor_adapter import (LEG_HIGHS, POLICY_DIM, QUOTE_DIM,
                                         fleet_mode, forward_policy,
                                         target_quotes)

B1_RESULTS = "data/processed/v2_cy_jn_2025/b1_terminal_neutral_results.csv"
B1_BLOCKS = "data/processed/v2_cy_jn_2025/b1_terminal_neutral_block_payoffs.json"
B1_MANIFEST = "data/processed/v2_cy_jn_2025/b1_terminal_neutral_manifest.json"
PAIRING = "data/processed/v2_cy_jn_2025/load_price_pairing.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen knobs of one trainer branch.

    `reward_mode` is the only thing that differs between the two declared
    experiments; everything else is shared so a comparison changes one thing.
    """

    experiment_id: str
    reward_mode: str                 # "B" or "B_C2"
    gamma: float = 0.99
    fixed_scale: float = 1000.0
    gate_temperature: float = 1.0
    price_noise_std: float = 0.5
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    lr: float = 3e-4
    tau: float = 0.005
    policy_delay: int = 2
    batch_size: int = 128
    buffer_capacity: int = 100_000
    start_steps: int = 0

    def __post_init__(self):
        if self.reward_mode not in ("B", "B_C2"):
            raise ValueError("reward_mode must be 'B' or 'B_C2'")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive")
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

    @property
    def uses_c2(self) -> bool:
        return self.reward_mode == "B_C2"

    def as_dict(self) -> dict:
        return asdict(self)


# The two declared branches differ only in the reward mode and its id.
EXPERIMENTS = {
    "NEAR_B_C2_GATE": ExperimentConfig("NEAR_B_C2_GATE", "B_C2"),
    "NEAR_B_ONLY_GATE": ExperimentConfig("NEAR_B_ONLY_GATE", "B"),
}


def config_hash(config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_data_hash(pair_row: Mapping[str, str]) -> str:
    """Bind a B1 lookup to the two curves the day actually clears on."""
    payload = (str(pair_row["load_curve_sha256"])
               + "|" + str(pair_row["price_curve_sha256"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_csv(path: str | Path) -> List[dict]:
    import csv
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_b1_cache(results_path: str = B1_RESULTS,
                  blocks_path: str = B1_BLOCKS,
                  manifest_path: str = B1_MANIFEST,
                  pairing_path: str = PAIRING) -> B1Cache:
    """Build the strict B1 cache from the terminal-neutral run.

    A day that did not clear, did not pin, or is missing any unit's block
    payoffs is refused rather than approximated.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mechanism_hash = manifest.get("mechanism_hash")
    if not mechanism_hash:
        raise ValueError(
            "the B1 manifest has no mechanism_hash; the old rolling B1 "
            "(no terminal pin) cannot be reused as the terminal-neutral "
            "benchmark")
    results = {r["pair_id"]: r for r in _read_csv(results_path)}
    blocks = json.loads(Path(blocks_path).read_text(encoding="utf-8"))
    pairing = {f"{r['load_date']}|{r['price_date']}": r
               for r in _read_csv(pairing_path)}
    cache = B1Cache()
    for pair_id, payoffs in blocks.items():
        row = results.get(pair_id)
        if row is None:
            raise KeyError(f"B1 result missing for {pair_id}")
        if str(row.get("feasible")).lower() not in ("true", "1"):
            raise ValueError(f"B1 day {pair_id} was not feasible")
        pair = pairing.get(pair_id)
        if pair is None:
            raise KeyError(f"pairing row missing for {pair_id}")
        cache.add(B1Day(
            pair_id=pair_id,
            soc0=float(row["soc0"]),
            # The field name is historical; the value is the mechanism hash,
            # so a no-pin B1 with the same market config cannot collide.
            config_hash=mechanism_hash,
            data_hash=pair_data_hash(pair),
            block_payoffs={u: list(v) for u, v in payoffs.items()},
            soc_end={u: float(row[f"soc_end_{u}"]) for u in payoffs},
        ))
    return cache


class GateActor(nn.Module):
    """obs -> four logits: [gate_charge, gate_discharge, price_charge,
    price_discharge]. No tanh: the adapter owns the gate and sigmoid mapping."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, POLICY_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GateReplayBuffer:
    """Transitions plus the gate branch's own quantities.

    `buffer` keeps the (obs, action, reward, next_obs, done) tuple the schema-v2
    writer already knows; `extras` carries the gate, logit and sampling fields.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.buffer: deque = deque(maxlen=self.capacity)
        self.extras: Dict[str, deque] = {
            "gates": deque(maxlen=self.capacity),
            "probs": deque(maxlen=self.capacity),
            "gate_logits": deque(maxlen=self.capacity),
            "price_logits": deque(maxlen=self.capacity),
            "price_quotes": deque(maxlen=self.capacity),
            "sampled": deque(maxlen=self.capacity),
        }

    def add(self, obs, action, reward, next_obs, done, extras: dict) -> None:
        self.buffer.append((
            np.asarray(obs, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy(),
            np.asarray(reward, dtype=np.float32).copy(),
            np.asarray(next_obs, dtype=np.float32).copy(),
            float(done),
        ))
        for key in self.extras:
            self.extras[key].append(
                np.asarray(extras[key], dtype=np.float32).copy())

    def sample(self, batch_size: int) -> dict:
        n = min(batch_size, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), n)
        rows = [self.buffer[i] for i in idx]
        obs, act, rew, next_obs, done = zip(*rows)
        out = {
            "obs": torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            "actions": torch.as_tensor(np.asarray(act), dtype=torch.float32),
            "rewards": torch.as_tensor(np.asarray(rew), dtype=torch.float32),
            "next_obs": torch.as_tensor(np.asarray(next_obs),
                                        dtype=torch.float32),
            "done": torch.as_tensor(np.asarray(done), dtype=torch.float32),
        }
        for key, values in self.extras.items():
            out[key] = torch.as_tensor(
                np.asarray([values[i] for i in idx]), dtype=torch.float32)
        return out

    @property
    def extra_arrays(self) -> Dict[str, np.ndarray]:
        return {key: np.asarray(list(values)) for key, values in
                self.extras.items()}

    def load_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.buffer.clear()
        for values in self.extras.values():
            values.clear()
        n = len(arrays["obs"])
        for i in range(n):
            self.buffer.append((
                np.asarray(arrays["obs"][i], dtype=np.float32).copy(),
                np.asarray(arrays["actions"][i], dtype=np.float32).copy(),
                np.asarray(arrays["rewards"][i], dtype=np.float32).copy(),
                np.asarray(arrays["next_obs"][i], dtype=np.float32).copy(),
                float(arrays["done"][i]),
            ))
        for key in self.extras:
            if key in arrays:
                for row in arrays[key]:
                    self.extras[key].append(
                        np.asarray(row, dtype=np.float32).copy())

    def __len__(self) -> int:
        return len(self.buffer)


class MATD3BC2Gate:
    """Twin-delayed MATD3 with the B/C2 reward and per-leg truthful gates."""

    def __init__(self, env: BiddingEnv, experiment: ExperimentConfig,
                 *, agent_names: Optional[Sequence[str]] = None,
                 obs_spec=None, action_spec=None):
        self.env = env
        self.experiment = experiment
        self.obs_spec = obs_spec or getattr(env, "obs_spec", OBS_V6)
        self.action_spec = action_spec or getattr(
            env, "action_spec", ACTION_STORAGE_QUOTE_V2)
        self.agent_names = list(agent_names or [a.name for a in env.rl_agents])
        self.obs_dim = env.get_state_dim()
        self.act_dim = QUOTE_DIM
        self.n_agents = len(self.agent_names)

        self.actors = {nm: GateActor(self.obs_dim) for nm in self.agent_names}
        self.actor_targets = {
            nm: GateActor(self.obs_dim) for nm in self.agent_names}
        self.critics = {
            nm: CentralizedCritic(self.n_agents, self.obs_dim, self.act_dim)
            for nm in self.agent_names}
        self.critic_targets = {
            nm: CentralizedCritic(self.n_agents, self.obs_dim, self.act_dim)
            for nm in self.agent_names}
        for nm in self.agent_names:
            self.actor_targets[nm].load_state_dict(
                self.actors[nm].state_dict())
            self.critic_targets[nm].load_state_dict(
                self.critics[nm].state_dict())
            self.critic_targets[nm].eval()

        self.actor_opts = {
            nm: optim.Adam(self.actors[nm].parameters(), lr=experiment.lr,
                           weight_decay=1e-5)
            for nm in self.agent_names}
        self.critic_opts = {
            # weight_decay 0: with Q at O(1) or below, 1e-4 decays the very
            # weights that carry the between-action differences the actor
            # needs; the v3 audits trace the ranking failure partly here.
            nm: optim.Adam(self.critics[nm].parameters(), lr=experiment.lr,
                           weight_decay=0.0)
            for nm in self.agent_names}

        self.buffer = GateReplayBuffer(experiment.buffer_capacity)
        self.total_steps = 0
        self.updates = 0
        self.episodes = 0
        # pair_id -> number of full-day exposures. Saved beside every
        # checkpoint so a resumed run continues the log rather than restarting
        # it.
        self.exposure: Counter = Counter()
        self.gate_stats = self._empty_gate_stats()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _empty_gate_stats(self) -> dict:
        return {
            "open": {(nm, leg): 0 for nm in self.agent_names
                     for leg in ("charge", "discharge")},
            "prob_sum": {(nm, leg): 0.0 for nm in self.agent_names
                         for leg in ("charge", "discharge")},
            "steps": 0,
            "modes": Counter(),
            "saturated_upper": 0,
            "saturated_lower": 0,
        }

    def _policy_action(self, actor: nn.Module, obs: np.ndarray,
                       *, train: bool) -> dict:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        policy_out = actor(obs_t)
        if train and self.experiment.price_noise_std > 0:
            noise = torch.randn_like(policy_out[..., 2:]) \
                * self.experiment.price_noise_std
            policy_out = torch.cat(
                [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        out = forward_policy(
            policy_out, train=train,
            temperature=self.experiment.gate_temperature)
        return {
            "quotes": out["quotes"].squeeze(0).detach().numpy(),
            "gates": out["gates"].squeeze(0).detach().numpy(),
            "probs": out["probs"].squeeze(0).detach().numpy(),
            "gate_logits": policy_out[..., :2].squeeze(0).detach().numpy(),
            "price_logits": policy_out[..., 2:].squeeze(0).detach().numpy(),
            "price_quotes": out["price_quotes"].squeeze(0).detach().numpy(),
            # In train mode `gates` is the hard Bernoulli draw (the
            # straight-through term adds zero in the forward pass).
            "sampled": out["gates"].squeeze(0).detach().numpy(),
            "mode": int(out["mode"].reshape(-1)[0].item()),
        }

    def select_action(self, obs: np.ndarray, agent_name: str,
                      *, train: bool = False) -> dict:
        return self._policy_action(self.actors[agent_name], obs, train=train)

    def _fleet_action(self, obs: Mapping[str, np.ndarray], *,
                      train: bool) -> tuple[dict, dict]:
        actions, diags = {}, {}
        for nm in self.agent_names:
            d = self._policy_action(self.actors[nm], obs[nm], train=train)
            actions[nm] = np.asarray(d["quotes"], dtype=float)
            diags[nm] = d
        return actions, diags

    def _target_quotes(self, actor: nn.Module,
                       next_obs: torch.Tensor) -> torch.Tensor:
        """Smooth the price branch only; a closed gate stays exactly zero."""
        policy_out = actor(next_obs)
        noise = (torch.randn_like(policy_out[..., 2:])
                 * self.experiment.target_policy_noise).clamp(
                     -self.experiment.target_noise_clip,
                     self.experiment.target_noise_clip)
        policy_out = torch.cat(
            [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        return target_quotes(policy_out)

    def _actor_loss_quotes(self, policy_out: torch.Tensor) -> torch.Tensor:
        """Quote map used by the actor loss; the subclass owns its map."""
        return forward_policy(
            policy_out, train=True,
            temperature=self.experiment.gate_temperature)["quotes"]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _day_rewards(self, block_payoffs: Mapping[str, Sequence[float]],
                     soc_path: Mapping[str, Sequence[float]],
                     done: Sequence[bool],
                     b1_cache: B1Cache, pair_id: str, config_hash: str,
                     data_hash: str) -> dict:
        base = day_reward_b_from_cache(
            block_payoffs, b1_cache, pair_id=pair_id, soc0=0.5,
            config_hash=config_hash, data_hash=data_hash,
            config=RewardBConfig(gamma=self.experiment.gamma,
                                 fixed_scale=1.0))
        out = {"b": base.as_dict(),
               "b_reward_by_block": {nm: list(base.reward[nm])
                                     for nm in base.reward}}
        if self.experiment.uses_c2:
            shaped = c2_reward(
                base.reward, soc_path, done=np.asarray(done, dtype=bool),
                config=C2Config(gamma=self.experiment.gamma))
            out["c2"] = shaped.as_dict()
            out["c2_shaping_by_block"] = {
                nm: list(shaped.shaping_terms[nm])
                for nm in shaped.shaping_terms}
            per_block = shaped.reward
        else:
            per_block = base.reward
        out["per_block"] = per_block
        return out

    @staticmethod
    def _stack_obs(obs: Mapping[str, np.ndarray],
                   names: Sequence[str]) -> np.ndarray:
        return np.stack([np.asarray(obs[nm], dtype=np.float32)
                         for nm in names])

    def _accumulate_gate_stats(self, diags: Mapping[str, dict]) -> None:
        modes = []
        for nm in self.agent_names:
            d = diags[nm]
            modes.append(d["mode"])
            for leg_idx, leg in enumerate(("charge", "discharge")):
                self.gate_stats["open"][(nm, leg)] += int(
                    d["gates"][leg_idx] > 0)
                self.gate_stats["prob_sum"][(nm, leg)] += float(
                    d["probs"][leg_idx])
            executed = np.asarray(d["quotes"], dtype=float)
            opened = np.asarray(d["gates"], dtype=float) > 0
            # A closed gate is an exact zero by construction, not a quote
            # pressed against its lower bound; saturation is counted only on
            # legs that actually declared a price.
            if opened.any():
                open_quotes = executed[opened]
                open_highs = np.asarray(LEG_HIGHS)[opened]
                self.gate_stats["saturated_upper"] += int(
                    np.sum(open_quotes >= 0.99 * open_highs))
                self.gate_stats["saturated_lower"] += int(
                    np.sum(open_quotes <= 0.01 * open_highs))
        self.gate_stats["modes"][fleet_mode(modes)] += 1
        self.gate_stats["steps"] += 1

    # ------------------------------------------------------------------
    # One training day
    # ------------------------------------------------------------------
    def train_day(self, env: BiddingEnv, pair: dict, b1_cache: B1Cache,
                  *, config_hash_value: str, data_hash: str) -> dict:
        if getattr(env, "terminal_soc_target", None) != 0.5:
            raise ValueError(
                "the B/C2 trainer requires the terminal-neutral 24-block "
                "environment (SOC0=SOC24=0.5); refusing a different mechanism")
        pair_id = f"{pair['load_date']}|{pair['price_date']}"
        obs = env.reset()
        soc_path = {
            nm: [float(next(a for a in env.all_agents
                            if a.name == nm).storage.soc0)]
            for nm in self.agent_names}
        block_payoffs = {nm: [] for nm in self.agent_names}
        pending = []
        done_flags = []
        checks = {"clears": 0, "fallbacks": 0, "lmp_fallbacks": 0,
                  "all_optimal": True, "voltage_violation_cells": 0,
                  "line_overload_cells": 0, "losses_mwh": 0.0,
                  "charge_mwh": {nm: 0.0 for nm in self.agent_names},
                  "discharge_mwh": {nm: 0.0 for nm in self.agent_names}}
        v_lo = float(env.config.network.v_min_pu)
        v_hi = float(env.config.network.v_max_pu)
        for block in range(N_BLOCKS):
            actions, diags = self._fleet_action(obs, train=True)
            next_obs, _, done, info = env.step(actions)
            if env._last_result is None:
                raise RuntimeError(
                    f"{pair_id} block {block}: clearing failed; training stops")
            result = env._last_result
            checks["clears"] += 1
            checks["lmp_fallbacks"] += int(result.get("lmp_fallbacks", 0))
            checks["all_optimal"] &= bool(
                result.get("solver", {}).get("final", {}).get("optimal"))
            # The result covers the whole look-ahead window; only the
            # committed periods are part of the executed day.
            n_commit = min(BLOCK_SIZE, env.T - block * BLOCK_SIZE)
            voltage = np.asarray(result.get("bus_voltage", []),
                                 dtype=float)[:n_commit]
            if voltage.size:
                checks["voltage_violation_cells"] += int(
                    ((voltage < v_lo - 1e-7) | (voltage > v_hi + 1e-7)).sum())
            util = np.asarray(result.get("line_utilization", []),
                              dtype=float)[:n_commit]
            if util.size:
                checks["line_overload_cells"] += int(
                    (util > 1.0 + 1e-7).sum())
            loss = result.get("line_loss_mw")
            if loss is not None:
                checks["losses_mwh"] += float(
                    np.sum(np.asarray(loss, dtype=float)[:n_commit]) * 0.25)
            for nm in self.agent_names:
                ws = (result.get("schedules") or {}).get(nm)
                if ws is not None:
                    checks["charge_mwh"][nm] += float(
                        np.sum(np.asarray(ws["p_ch"][:n_commit])) * 0.25)
                    checks["discharge_mwh"][nm] += float(
                        np.sum(np.asarray(ws["p_dis"][:n_commit])) * 0.25)
                block_payoffs[nm].append(
                    float(info.get("raw_profit", {}).get(nm, 0.0)))
                soc_path[nm].append(
                    float(env.prev_soc.get(nm, soc_path[nm][-1])))
            pending.append((obs, actions, next_obs, done, diags))
            done_flags.append(bool(done))
            obs = next_obs
            if done:
                break
        if len(pending) != N_BLOCKS:
            raise RuntimeError(f"{pair_id}: episode ended after "
                               f"{len(pending)} blocks")

        rewards = self._day_rewards(
            block_payoffs, soc_path, done_flags, b1_cache, pair_id,
            config_hash_value, data_hash)
        per_block = rewards["per_block"]
        for t, (obs_t, act_t, next_obs_t, done_t, diag_t) in enumerate(pending):
            nm_list = self.agent_names
            reward = np.array(
                [per_block[nm][t] / self.experiment.fixed_scale
                 for nm in nm_list], dtype=np.float32)
            extras = {
                "gates": np.stack([diag_t[nm]["gates"] for nm in nm_list]),
                "probs": np.stack([diag_t[nm]["probs"] for nm in nm_list]),
                "gate_logits": np.stack(
                    [diag_t[nm]["gate_logits"] for nm in nm_list]),
                "price_logits": np.stack(
                    [diag_t[nm]["price_logits"] for nm in nm_list]),
                "price_quotes": np.stack(
                    [diag_t[nm]["price_quotes"] for nm in nm_list]),
                "sampled": np.stack(
                    [diag_t[nm]["sampled"] for nm in nm_list]),
            }
            self.buffer.add(
                self._stack_obs(obs_t, nm_list),
                np.stack([act_t[nm] for nm in nm_list]),
                reward,
                self._stack_obs(next_obs_t, nm_list),
                done_t, extras)
            self.total_steps += 1
            if len(self.buffer) >= self.experiment.batch_size \
                    and self.total_steps > self.experiment.start_steps:
                self.update()
            self._accumulate_gate_stats(diag_t)
        self.episodes += 1
        self.exposure[pair_id] += 1
        e_max = {nm: float(next(a for a in env.all_agents
                                if a.name == nm).storage.e_max)
                 for nm in self.agent_names}
        checks["cycles"] = {
            nm: ((checks["charge_mwh"][nm] + checks["discharge_mwh"][nm])
                 / (2.0 * e_max[nm]) if e_max[nm] > 0 else 0.0)
            for nm in self.agent_names}
        checks["terminal_soc"] = {
            nm: soc_path[nm][-1] for nm in self.agent_names}
        return {
            "pair_id": pair_id,
            "b": rewards["b"],
            "c2": rewards.get("c2"),
            "b_reward_by_block": rewards["b_reward_by_block"],
            "c2_shaping_by_block": rewards.get("c2_shaping_by_block"),
            "checks": checks,
            "soc_end": {nm: soc_path[nm][-1] for nm in self.agent_names},
            "gate": self.gate_stats_snapshot(),
        }

    def gate_stats_snapshot(self) -> dict:
        steps = max(self.gate_stats["steps"], 1)
        return {
            "steps": self.gate_stats["steps"],
            "open_frequency": {
                f"{nm}.{leg}": self.gate_stats["open"][(nm, leg)] / steps
                for nm in self.agent_names
                for leg in ("charge", "discharge")},
            "mean_probability": {
                f"{nm}.{leg}": self.gate_stats["prob_sum"][(nm, leg)] / steps
                for nm in self.agent_names
                for leg in ("charge", "discharge")},
            "mode_counts": {str(k): int(v)
                            for k, v in sorted(
                                self.gate_stats["modes"].items())},
            "distinct_modes": len(self.gate_stats["modes"]),
            "saturated_upper": self.gate_stats["saturated_upper"],
            "saturated_lower": self.gate_stats["saturated_lower"],
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> dict:
        if len(self.buffer) < self.experiment.batch_size:
            return {"critic_loss": None, "actor_loss": None,
                    "q1_mean": None, "q2_mean": None, "q_gap": None,
                    "reward_scale": self.experiment.fixed_scale}
        batch = self.buffer.sample(self.experiment.batch_size)
        obs = batch["obs"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        batch_size = obs.shape[0]
        obs_flat = obs.reshape(batch_size, -1)
        critic_losses, q1s, q2s, q_gaps = [], [], [], []

        for i, nm in enumerate(self.agent_names):
            critic = self.critics[nm]
            target_critic = self.critic_targets[nm]
            with torch.no_grad():
                target_actions = torch.stack(
                    [self._target_quotes(self.actor_targets[nm2],
                                         next_obs[:, j, :])
                     for j, nm2 in enumerate(self.agent_names)], dim=1)
                target_q1, target_q2 = target_critic(
                    obs_flat, target_actions.reshape(batch_size, -1))
                target = rew[:, i:i + 1] + self.experiment.gamma \
                    * (1.0 - done[:, None]) * torch.min(target_q1, target_q2)
            current_q1, current_q2 = critic(obs_flat, act.reshape(
                batch_size, -1))
            loss = nn.functional.mse_loss(current_q1, target) \
                + nn.functional.mse_loss(current_q2, target)
            self.critic_opts[nm].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            self.critic_opts[nm].step()
            critic_losses.append(float(loss.detach()))
            q1s.append(float(current_q1.mean().detach()))
            q2s.append(float(current_q2.mean().detach()))
            q_gaps.append(float((current_q1 - current_q2).abs().mean()
                                .detach()))

        actor_losses = {}
        anchor_logs = {}
        hinge_adv_logs = {}
        hinge_active_logs = {}
        hinge_twin_logs = {}
        anchor_weight = float(getattr(self.experiment, "anchor_weight", 0.0))
        hinge_delta = float(getattr(self.experiment,
                                    "advantage_hinge_delta", 0.0))
        twin_weight = float(getattr(self.experiment,
                                    "hinge_twin_gap_weight", 0.0))
        highs = torch.as_tensor(LEG_HIGHS, dtype=torch.float32)
        if self.total_steps % max(self.experiment.policy_delay, 1) == 0:
            for i, nm in enumerate(self.agent_names):
                actor = self.actors[nm]
                critic = self.critics[nm]
                policy_out = actor(obs[:, i, :])
                quotes_i = self._actor_loss_quotes(policy_out)
                replaced = act.clone()
                replaced[:, i, :] = quotes_i
                q1, q2 = critic(obs_flat, replaced.reshape(batch_size, -1))
                q_policy = torch.min(q1, q2)
                if hinge_delta > 0.0:
                    # W7 advantage hinge: the econ term fires only where the
                    # critic sees an advantage over the truthful quote that
                    # clears the noise floor delta. The truthful
                    # counterfactual reuses the batch's joint action with
                    # this agent's quotes zeroed, the same reference the
                    # anchor pulls toward. On flat-Q states (winter days,
                    # G1c) the spurious sub-delta slope no longer moves the
                    # actor; a genuinely better-than-delta edge still does.
                    # Q(a_truthful) carries no gradient.
                    with torch.no_grad():
                        truthful = act.clone()
                        truthful[:, i, :] = 0.0
                        tq1, tq2 = critic(
                            obs_flat, truthful.reshape(batch_size, -1))
                        q_truth = torch.min(tq1, tq2)
                        # W9 (b/W8标定暴露_G1重测_2026-11-09.md §5): never
                        # trust an advantage smaller than the two critics'
                        # own disagreement at the evaluated action. In the
                        # acceptance-blind plateau the payoff signal cannot
                        # differentiate quotes and the twins' extrapolation
                        # diverges (measured twin_gap 0.12-0.17 there vs
                        # 0.02-0.09 where payoff differentiates), so the
                        # effective floor rises exactly where the critic is
                        # underdetermined. weight 0 restores the W7 constant
                        # floor.
                        twin_gap = (q1 - q2).abs()
                        delta_eff = torch.clamp(
                            twin_weight * twin_gap, min=hinge_delta)
                    advantage = q_policy - q_truth
                    loss = -nn.functional.relu(
                        advantage - delta_eff).mean()
                    hinge_adv_logs[nm] = float(advantage.mean().detach())
                    hinge_active_logs[nm] = float(
                        (advantage > delta_eff).float().mean().detach())
                    hinge_twin_logs[nm] = float(twin_gap.mean().detach())
                else:
                    loss = -q_policy.mean()
                # Anchor toward the truthful quote (0, 0): box-normalised
                # squared deviation of the executed quotes, exactly zero at
                # the truthful point. On a flat or mis-ranked Q surface this
                # is the only term that keeps the policy from drifting away
                # from truthful; the v3 actor loss had no such term and the
                # fleet drifted into corner collapse (phase 25).
                if anchor_weight > 0.0:
                    anchor = ((quotes_i / highs) ** 2).mean()
                    loss = loss + anchor_weight * anchor
                    anchor_logs[nm] = float(anchor.detach())
                self.actor_opts[nm].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                self.actor_opts[nm].step()
                actor_losses[nm] = float(loss.detach())

        for nm in self.agent_names:
            for target, source in (
                    (self.actor_targets[nm], self.actors[nm]),
                    (self.critic_targets[nm], self.critics[nm])):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.experiment.tau * sp.data
                                  + (1.0 - self.experiment.tau) * tp.data)
        self.updates += 1
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": (float(np.mean(list(actor_losses.values())))
                           if actor_losses else None),
            "actor_loss_by_agent": actor_losses,
            "anchor_by_agent": anchor_logs,
            "anchor_weight": anchor_weight,
            "hinge_advantage_by_agent": hinge_adv_logs,
            "hinge_active_fraction_by_agent": hinge_active_logs,
            "hinge_twin_gap_by_agent": hinge_twin_logs,
            "hinge_delta": hinge_delta,
            "q1_mean": float(np.mean(q1s)),
            "q2_mean": float(np.mean(q2s)),
            "q_gap": float(np.mean(q_gaps)),
            "reward_scale": self.experiment.fixed_scale,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save(self, root: str | Path, *, level: str, dataset: str, seed: int,
             config_hash_value: str, data: Optional[dict] = None,
             code_revision: str = "working-tree",
             exposure_log: Optional[str] = None,
             eval_provenance: Optional[str] = None) -> Path:
        if level == "bundle" and not data:
            raise ValueError("a bundle checkpoint needs the four data hashes")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        exposure_path = root / "exposure.json"
        exposure_path.write_text(
            json.dumps(dict(sorted(self.exposure.items())), indent=2),
            encoding="utf-8")
        exposure_log = exposure_log or str(exposure_path)
        reward = {
            "definition": self.experiment.reward_mode,
            "gamma": self.experiment.gamma,
            "scale": self.experiment.fixed_scale,
            "potential": (C2Config(gamma=self.experiment.gamma).__dict__
                          if self.experiment.uses_c2 else None),
            "normalization": "fixed_scale_only",
            "terminal_rule": "soc_end_0.5_settlement",
            "b1_source": "b1_terminal_neutral",
            "gate": {
                "policy_dim": POLICY_DIM,
                "quote_dim": QUOTE_DIM,
                "leg_highs": list(LEG_HIGHS),
                "temperature": self.experiment.gate_temperature,
                "estimator": "straight_through_bernoulli",
                "initial_probability": 0.5,
            },
            "price_noise_std": self.experiment.price_noise_std,
        }
        counters = {
            "total_steps": self.total_steps,
            "episode": self.episodes,
            "updates": self.updates,
            "buffer_len": len(self.buffer),
        }
        optimizers = {}
        for nm in self.agent_names:
            optimizers[f"actor_{nm}"] = self.actor_opts[nm]
            optimizers[f"critic_{nm}"] = self.critic_opts[nm]
        return save_checkpoint(
            root, level=level, run_id=self.experiment.experiment_id,
            dataset=dataset, seed=seed, code_revision=code_revision,
            config_hash=config_hash_value,
            specs={
                "obs_spec": {"name": self.obs_spec.name,
                             "version": self.obs_spec.version},
                "action_spec": {"name": self.action_spec.name,
                                "version": self.action_spec.version},
            },
            actors=self.actors, actor_targets=self.actor_targets,
            critics=self.critics, critic_targets=self.critic_targets,
            optimizers=optimizers, counters=counters, reward=reward,
            data=data or {},
            hyperparameters=self.experiment.as_dict(),
            exposure_log=exposure_log, eval_provenance=eval_provenance,
            buffer=self.buffer)

    def load(self, root: str | Path) -> dict:
        bundle = load_checkpoint(root)
        saved = bundle["manifest"]["reward"]
        if saved.get("definition") != self.experiment.reward_mode:
            raise ValueError(
                f"checkpoint reward {saved.get('definition')!r} does not match "
                f"this trainer's {self.experiment.reward_mode!r}")
        if float(saved.get("scale", -1)) != self.experiment.fixed_scale:
            raise ValueError("checkpoint reward scale does not match")
        if float(saved.get("gamma", -1)) != self.experiment.gamma:
            raise ValueError("checkpoint gamma does not match")
        optimizers = {}
        for nm in self.agent_names:
            optimizers[f"actor_{nm}"] = self.actor_opts[nm]
            optimizers[f"critic_{nm}"] = self.critic_opts[nm]
        restore_checkpoint(
            bundle, actors=self.actors, actor_targets=self.actor_targets,
            critics=self.critics, critic_targets=self.critic_targets,
            optimizers=optimizers)
        self.critic_targets = {nm: mod.eval()
                              for nm, mod in self.critic_targets.items()}
        counters = bundle["counters"]
        self.total_steps = int(counters["total_steps"])
        self.episodes = int(counters["episode"])
        self.updates = int(counters["updates"])
        exposure_path = Path(root) / "exposure.json"
        if exposure_path.exists():
            self.exposure = Counter(
                {str(k): int(v) for k, v in
                 json.loads(exposure_path.read_text(encoding="utf-8")).items()})
        if "buffer" in bundle:
            self.buffer.load_arrays(bundle["buffer"])
        return bundle
