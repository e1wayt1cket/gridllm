"""MATD3_DA_GATE_v2: D1 quote map + D2 stratified replay + D3 exploration + A1.

The economic core is inherited unchanged from `rl_b_c2_gate`: whole-day B,
the C2 potential, gamma=0.99, fixed scale 1000, terminal SOC 0.5, twin
centralised critics and the existing clearing engine. This module changes only:

* D1: the open-gate quote map (`da_quote_map.forward_policy_v3`);
* D2: the replay sampler (`da_replay.DAReplayBuffer`);
* D3: bounded random quote exploration on training episodes;
* A1: a gate-logit temperature warmup with evaluation at τ=1.

A v2 checkpoint is refused. The price logits mean something different under
the new map, so R2/R10 are baseline references only and are never loaded as
initialisations.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import rl_b_c2_gate as base
from c2_reward import C2Config
from checkpoint_loader_v2 import load_checkpoint, restore_checkpoint
from checkpoint_writer_v2 import save_checkpoint
from da_quote_map import (LEG_HIGHS, POLICY_DIM, QUOTE_DIM, fleet_mode,
                          forward_policy_v3, target_quotes_v3)
from da_replay import DAReplayBuffer
from daily_reward_b import B1Cache, B1Day, RewardBConfig, day_reward_b_from_cache
from rl_env import BLOCK_SIZE, N_BLOCKS
from rl_spec import ACTION_STORAGE_QUOTE_V4, OBS_V6

# Re-export the provenance helpers the training script reads.
B1_RESULTS = base.B1_RESULTS
B1_BLOCKS = base.B1_BLOCKS
B1_MANIFEST = base.B1_MANIFEST
PAIRING = base.PAIRING
load_b1_cache = base.load_b1_cache
pair_data_hash = base.pair_data_hash
config_hash = base.config_hash
GateActor = base.GateActor
GateReplayBuffer = base.GateReplayBuffer


class EpisodeClearingError(RuntimeError):
    """One block failed to clear inside `train_day`.

    The episode is atomic: block transitions only reach the buffer after the
    whole day is rewarded, so an aborted episode leaves the trainer state
    clean and the run may continue with the next episode (TD-02).
    """


@dataclass(frozen=True)
class DAExperimentConfig:
    """Frozen D+A knobs. The inherited economic knobs keep their old values.

    v4 changes (see `b/v4修复方案_2026-11-08.md`): `fixed_scale` 1000 -> 100
    (Q to O(1)); `buffer_capacity` 100k -> 6000 (a sliding window that only
    just covers the anneal era); `start_steps` 0 -> 480 (pure collection
    before the first update); `anchor_weight` adds the truthful-quote anchor
    to the actor loss; `gate_logit_bound` records the map's tanh bound.
    """

    experiment_id: str = "MATD3_DA_GATE_v2"
    reward_mode: str = "B_C2"
    gamma: float = 0.99
    fixed_scale: float = 100.0
    gate_temperature: float = 1.0       # evaluation value
    price_noise_std: float = 0.5
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    lr: float = 3e-4
    tau: float = 0.005
    policy_delay: int = 2
    batch_size: int = 128
    buffer_capacity: int = 6_000
    start_steps: int = 480
    # `rl_td3.CALIBRATED_DEV_PENALTY`, the weight the deviation calibration
    # selected for the sum-of-squares regularizer. The gate actor used an
    # uncalibrated 0.01 against a leg-averaged penalty, which is neither the
    # documented form nor the calibrated weight.
    anchor_weight: float = 0.10
    # W7 (b/G1结果与机制分析_2026-11-08.md §3): advantage-hinge actor
    # objective. The econ term is -ReLU(Q(a) - Q(a_truthful) - delta): when
    # the critic's estimated advantage over truthful bidding is inside the
    # noise floor delta, the actor receives no econ gradient and the anchor
    # alone holds the quotes at truthful. delta was calibrated from the
    # measured Q-surface noise on provably flat winter states
    # (reports/algorithm_diagnostics/v4_hinge_calibration.json), never on
    # profit. 0 disables the hinge (pre-W7 behaviour).
    advantage_hinge_delta: float = 0.10
    # W8 (b/G1重测_V4B_结果与机制分析_2026-11-09.md §5): periodic calibration
    # exposures. The hinge pins the actor near the truthful quote, so without
    # intervention the replay buffer's quote distribution collapses and the
    # critic's unchecked extrapolation grows fake peaks far above the noise
    # floor. Every `calibration_interval`-th episode redraws the quote of
    # every open gate uniformly over the full leg box for the whole day,
    # keeping supervised payoff signal across the box no matter where the
    # actor sits. Gates follow the policy as usual (the state distribution
    # stays natural). 0 disables (pre-W8 behaviour).
    calibration_interval: int = 8
    # W9 (b/W8标定暴露_G1重测_2026-11-09.md §5): twin-gap-adaptive hinge
    # floor, delta_eff = max(advantage_hinge_delta, k * |Q1 - Q2|) evaluated
    # at the policy action. The floor is never trusted below the critics'
    # own measured disagreement; k = 1.0 encodes exactly that principle and
    # was read off the G1b twin_gap measurements (0.02-0.09 where payoff
    # differentiates quotes, 0.12-0.17 in the acceptance-blind plateau), a
    # non-profit rule in the same spirit as the delta calibration.
    # 0 restores the W7 constant floor.
    hinge_twin_gap_weight: float = 1.0
    gate_logit_bound: float = 4.0
    action_schema: str = "storage_quote_v4"
    quote_map: str = "linear_core_v1"
    gate_temperature_start: float = 4.0
    gate_temperature_end: float = 1.0
    gate_warmup_episodes: int = 106
    exploration_epsilon_start: float = 0.20
    exploration_epsilon_end: float = 0.05
    exploration_decay_episodes: int = 159
    exploration_charge_scale: float = 0.5
    exploration_high_fraction: float = 0.5
    uniform_fraction: float = 0.5
    sampling_beta: float = 0.5
    sampling_max_upweight: float = 5.0

    def __post_init__(self):
        if self.reward_mode not in ("B", "B_C2"):
            raise ValueError("reward_mode must be 'B' or 'B_C2'")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive")
        if self.gate_temperature_start < 1.0:
            raise ValueError("gate_temperature_start must be >= 1")
        if not 0.0 <= self.exploration_epsilon_start <= 1.0:
            raise ValueError("exploration epsilon must be in [0,1]")
        if not 0.0 <= self.uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0,1]")
        if self.action_schema != "storage_quote_v4":
            raise ValueError("this trainer only builds storage_quote_v4")
        if self.anchor_weight < 0.0:
            raise ValueError("anchor_weight must be non-negative")
        if self.advantage_hinge_delta < 0.0:
            raise ValueError("advantage_hinge_delta must be non-negative")
        if self.calibration_interval < 0:
            raise ValueError("calibration_interval must be non-negative")
        if self.hinge_twin_gap_weight < 0.0:
            raise ValueError("hinge_twin_gap_weight must be non-negative")
        from da_quote_map import GATE_LOGIT_BOUND
        if abs(self.gate_logit_bound - GATE_LOGIT_BOUND) > 1e-12:
            raise ValueError(
                f"gate_logit_bound {self.gate_logit_bound} disagrees with "
                f"da_quote_map.GATE_LOGIT_BOUND {GATE_LOGIT_BOUND}; the map "
                "is the single source of the bound")

    @property
    def uses_c2(self) -> bool:
        return self.reward_mode == "B_C2"

    def as_dict(self) -> dict:
        return asdict(self)

    def gate_temperature_at(self, episode: int) -> float:
        if self.gate_warmup_episodes <= 0:
            return self.gate_temperature_end
        progress = min(1.0, max(0.0, float(episode)
                                / float(self.gate_warmup_episodes)))
        return float(self.gate_temperature_start
                     + (self.gate_temperature_end
                        - self.gate_temperature_start) * progress)

    def exploration_epsilon_at(self, episode: int) -> float:
        if self.exploration_decay_episodes <= 0:
            return self.exploration_epsilon_end
        progress = min(1.0, max(0.0, float(episode)
                                / float(self.exploration_decay_episodes)))
        return float(self.exploration_epsilon_start
                     + (self.exploration_epsilon_end
                        - self.exploration_epsilon_start) * progress)


class DAGateTrainer(base.MATD3BC2Gate):
    """The B/C2 gate trainer with D+A behind the new action schema."""

    def __init__(self, env, experiment: DAExperimentConfig,
                 *, agent_names: Optional[Sequence[str]] = None,
                 obs_spec=None, action_spec=None):
        super().__init__(env, experiment, agent_names=agent_names,
                         obs_spec=obs_spec or OBS_V6,
                         action_spec=action_spec or ACTION_STORAGE_QUOTE_V4)
        self.action_spec = ACTION_STORAGE_QUOTE_V4
        self.obs_spec = OBS_V6
        self.buffer = DAReplayBuffer(
            experiment.buffer_capacity,
            uniform_fraction=experiment.uniform_fraction,
            beta=experiment.sampling_beta,
            max_upweight=experiment.sampling_max_upweight)
        self._current_episode = 0
        self._calibrating = False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def current_gate_temperature(self) -> float:
        return self.experiment.gate_temperature_at(self._current_episode)

    def _policy_action(self, actor: nn.Module, obs: np.ndarray,
                       *, train: bool) -> dict:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        policy_out = actor(obs_t)
        if train and self.experiment.price_noise_std > 0:
            noise = torch.randn_like(policy_out[..., 2:]) \
                * self.experiment.price_noise_std
            policy_out = torch.cat(
                [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        gate_temp = self.current_gate_temperature() if train else 1.0
        out = forward_policy_v3(policy_out, train=train,
                                gate_temperature=gate_temp)
        quotes = out["quotes"].squeeze(0).detach().numpy().copy()
        gates = out["gates"].squeeze(0).detach().numpy()
        probs = out["probs"].squeeze(0).detach().numpy()
        actor_quotes = quotes.copy()
        explored = np.zeros(2, dtype=np.float32)
        source = np.zeros(2, dtype=np.float32)
        if train:
            eps = self.experiment.exploration_epsilon_at(
                self._current_episode)
            for leg in (1, 0):
                p = eps if leg == 1 else eps * \
                    self.experiment.exploration_charge_scale
                if p > 0.0 and np.random.rand() < p and gates[leg] > 0.0:
                    high = LEG_HIGHS[leg]
                    if np.random.rand() < \
                            self.experiment.exploration_high_fraction:
                        quotes[leg] = high * (0.8 + 0.2 * np.random.rand())
                        source[leg] = 2.0
                    else:
                        quotes[leg] = high * np.random.rand()
                        source[leg] = 1.0
                    explored[leg] = 1.0
        if train and self._calibrating:
            # W8 calibration exposure: redraw every open gate's quote
            # uniformly over the full leg box. This runs after the epsilon
            # exploration and overrides it, so the whole day covers the box
            # regardless of where the hinge+anchor pin the actor.
            for leg in (1, 0):
                if gates[leg] > 0.0:
                    quotes[leg] = LEG_HIGHS[leg] * np.random.rand()
                    explored[leg] = 1.0
                    source[leg] = 3.0
        return {
            "quotes": quotes,
            "gates": gates,
            "probs": probs,
            "gate_logits": policy_out[..., :2].squeeze(0).detach().numpy(),
            "price_logits": policy_out[..., 2:].squeeze(0).detach().numpy(),
            "price_quotes": out["price_quotes"].squeeze(0).detach().numpy(),
            "sampled": gates.copy(),
            "mode": int(out["mode"].reshape(-1)[0].item()),
            "explored": explored,
            "exploration_source": source,
            "actor_quotes": actor_quotes,
            "gate_temperature": float(gate_temp),
        }

    def _target_quotes(self, actor: nn.Module,
                       next_obs: torch.Tensor) -> torch.Tensor:
        policy_out = actor(next_obs)
        noise = (torch.randn_like(policy_out[..., 2:])
                 * self.experiment.target_policy_noise).clamp(
                     -self.experiment.target_noise_clip,
                     self.experiment.target_noise_clip)
        policy_out = torch.cat(
            [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        return target_quotes_v3(policy_out)

    def _actor_loss_quotes(self, policy_out: torch.Tensor) -> torch.Tensor:
        """The actor loss must differentiate the map the actor executes."""
        return forward_policy_v3(
            policy_out, train=True,
            gate_temperature=self.current_gate_temperature())["quotes"]

    # ------------------------------------------------------------------
    # One training day (same accounting as the parent; new extras only)
    # ------------------------------------------------------------------
    def train_day(self, env, pair: dict, b1_cache: B1Cache, *,
                  config_hash_value: str, data_hash: str,
                  trace: bool = False) -> dict:
        if getattr(env, "terminal_soc_target", None) != 0.5:
            raise ValueError(
                "the B/C2 trainer requires the terminal-neutral 24-block "
                "environment (SOC0=SOC24=0.5); refusing a different mechanism")
        self._current_episode = self.episodes
        interval = int(self.experiment.calibration_interval)
        self._calibrating = bool(
            interval > 0 and (self.episodes + 1) % interval == 0)
        pair_id = f"{pair['load_date']}|{pair['price_date']}"
        obs = env.reset()
        soc_path = {
            nm: [float(next(a for a in env.all_agents
                            if a.name == nm).storage.soc0)]
            for nm in self.agent_names}
        block_payoffs = {nm: [] for nm in self.agent_names}
        pending = []
        done_flags = []
        trace_rows = []
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
                # TD-02: the episode is atomic — nothing from this day has
                # reached the buffer yet, so the caller may skip the episode
                # instead of aborting the whole seed.
                raise EpisodeClearingError(
                    f"{pair_id} block {block}: clearing failed; episode "
                    "aborted atomically")
            result = env._last_result
            checks["clears"] += 1
            checks["lmp_fallbacks"] += int(result.get("lmp_fallbacks", 0))
            checks["all_optimal"] &= bool(
                result.get("solver", {}).get("final", {}).get("optimal"))
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
            if trace:
                for nm in self.agent_names:
                    d = diags[nm]
                    trace_rows.append({
                        "pair_id": pair_id,
                        "block": block,
                        "agent": nm,
                        "gate_logits": np.asarray(
                            d["gate_logits"], dtype=float).tolist(),
                        "price_logits": np.asarray(
                            d["price_logits"], dtype=float).tolist(),
                        "probs": np.asarray(d["probs"], dtype=float).tolist(),
                        "gates": np.asarray(d["gates"], dtype=float).tolist(),
                        "quotes": np.asarray(actions[nm],
                                             dtype=float).tolist(),
                        "explored": np.asarray(
                            d["explored"], dtype=float).tolist(),
                        "exploration_source": np.asarray(
                            d["exploration_source"], dtype=float).tolist(),
                        "mode": int(d["mode"]),
                        "payoff": float(
                            info.get("raw_profit", {}).get(nm, 0.0)),
                    })
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
        hinge_active_sum = {nm: 0.0 for nm in self.agent_names}
        hinge_advantage_sum = {nm: 0.0 for nm in self.agent_names}
        hinge_twin_sum = {nm: 0.0 for nm in self.agent_names}
        hinge_updates = 0
        # The anchor is the only term in the actor loss that is not the
        # critic's advantage; its size is what says whether an update was
        # learning or being regularized.
        anchor_terms = {nm: 0.0 for nm in self.agent_names}
        anchor_updates = 0
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
                "explored": np.stack(
                    [diag_t[nm]["explored"] for nm in nm_list]),
                "exploration_source": np.stack(
                    [diag_t[nm]["exploration_source"] for nm in nm_list]),
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
                stats = self.update()
                fr = stats.get("hinge_active_fraction_by_agent") or {}
                adv = stats.get("hinge_advantage_by_agent") or {}
                tw = stats.get("hinge_twin_gap_by_agent") or {}
                if fr:
                    hinge_updates += 1
                    for nm in self.agent_names:
                        hinge_active_sum[nm] += fr.get(nm, 0.0)
                        hinge_advantage_sum[nm] += adv.get(nm, 0.0)
                        hinge_twin_sum[nm] += tw.get(nm, 0.0)
                anchor = stats.get("anchor_by_agent") or {}
                if anchor:
                    anchor_updates += 1
                    for nm in self.agent_names:
                        anchor_terms[nm] += float(anchor.get(nm) or 0.0)
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
            "block_decisions": trace_rows,
            "b": rewards["b"],
            "c2": rewards.get("c2"),
            "b_reward_by_block": rewards["b_reward_by_block"],
            "c2_shaping_by_block": rewards.get("c2_shaping_by_block"),
            "checks": checks,
            "soc_end": {nm: soc_path[nm][-1] for nm in self.agent_names},
            "gate": self.gate_stats_snapshot(),
            "gate_temperature": self.current_gate_temperature(),
            "exploration_epsilon": self.experiment.exploration_epsilon_at(
                self._current_episode),
            "calibration": bool(self._calibrating),
            "hinge_active_fraction": {
                nm: (hinge_active_sum[nm] / hinge_updates
                     if hinge_updates else None)
                for nm in self.agent_names},
            "hinge_advantage": {
                nm: (hinge_advantage_sum[nm] / hinge_updates
                     if hinge_updates else None)
                for nm in self.agent_names},
            "hinge_twin_gap": {
                nm: (hinge_twin_sum[nm] / hinge_updates
                     if hinge_updates else None)
                for nm in self.agent_names},
            "sampler": dict(self.buffer.last_stats),
            "anchor_by_agent": {
                nm: (anchor_terms[nm] / anchor_updates
                     if anchor_updates else None)
                for nm in self.agent_names},
            "anchor_weight": float(getattr(self.experiment,
                                           "anchor_weight", 0.0)),
        }

    def update(self) -> dict:
        out = super().update()
        out["sampler"] = dict(self.buffer.last_stats)
        return out

    # ------------------------------------------------------------------
    # Checkpoint v3
    # ------------------------------------------------------------------
    def _da_metadata(self) -> dict:
        return {
            "da_gate": {
                "experiment": self.experiment.as_dict(),
                "action_schema": self.experiment.action_schema,
                "quote_map": "linear_core_v1",
                "actor_loss_map": "linear_core_v1",
                "actor_loss_gate_temperature": "current_gate_temperature",
                "gate_temperature_schedule": {
                    "start": self.experiment.gate_temperature_start,
                    "end": self.experiment.gate_temperature_end,
                    "warmup_episodes": self.experiment.gate_warmup_episodes,
                    "evaluation_temperature": 1.0,
                },
                "exploration": {
                    "start": self.experiment.exploration_epsilon_start,
                    "end": self.experiment.exploration_epsilon_end,
                    "decay_episodes": self.experiment.exploration_decay_episodes,
                    "charge_scale": self.experiment.exploration_charge_scale,
                    "high_fraction": self.experiment.exploration_high_fraction,
                    "bounded": [0.0, 1.0],
                },
            },
            "sampler": self.buffer.sampler_state(),
        }

    def save(self, root, *, level: str, dataset: str, seed: int,
             config_hash_value: str, data: Optional[dict] = None,
             code_revision: str = "working-tree",
             exposure_log: Optional[str] = None,
             eval_provenance: Optional[str] = None):
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
                "temperature": 1.0,
                "estimator": "straight_through_bernoulli",
                "initial_probability": 0.5,
                "temperature_schedule": {
                    "start": self.experiment.gate_temperature_start,
                    "end": self.experiment.gate_temperature_end,
                    "warmup_episodes": self.experiment.gate_warmup_episodes,
                },
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
                "action_spec": {"name": ACTION_STORAGE_QUOTE_V4.name,
                                "version": ACTION_STORAGE_QUOTE_V4.version,
                                "quote_map": "linear_core_v1"},
            },
            actors=self.actors, actor_targets=self.actor_targets,
            critics=self.critics, critic_targets=self.critic_targets,
            optimizers=optimizers, counters=counters, reward=reward,
            data=data or {},
            hyperparameters=self.experiment.as_dict(),
            exposure_log=exposure_log, eval_provenance=eval_provenance,
            buffer=self.buffer, schema_version="matd3-checkpoint-v3",
            extra_manifest=self._da_metadata())

    def load(self, root) -> dict:
        bundle = load_checkpoint(root)
        manifest = bundle["manifest"]
        if manifest.get("schema_version") != "matd3-checkpoint-v3":
            raise ValueError(
                f"checkpoint schema {manifest.get('schema_version')!r} is not "
                "matd3-checkpoint-v3; the v3 quote map cannot reuse v2 actors")
        spec = manifest.get("specs", {}).get("action_spec", {})
        if spec.get("name") != ACTION_STORAGE_QUOTE_V4.name:
            raise ValueError(
                f"action spec {spec.get('name')!r} is not "
                f"{ACTION_STORAGE_QUOTE_V4.name!r}; v4 bounds the gate "
                "logits, so v3 checkpoints cannot be reused")
        saved = manifest["reward"]
        if saved.get("definition") != self.experiment.reward_mode:
            raise ValueError("checkpoint reward definition does not match")
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
        self._current_episode = self.episodes
        return bundle
