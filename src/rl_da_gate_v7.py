"""MATD3_DA_GATE_v7: the v5 storage quote as a price and a quantity per leg.

The economic core is inherited unchanged from `rl_b_c2_gate`: whole-day
differential B, the C2 potential, gamma=0.99, terminal SOC 0.5, twin
centralised critics and the existing clearing engine. The action box is the
v5 one -- four entries, `charge_price, charge_quantity, discharge_price,
discharge_quantity` -- so what this module replaces is everything the parent
sizes from the two-dimension box:

* the actor head, six logits instead of four;
* the centralised critics, one input per quote entry;
* the actor loss, over the executed four-entry quote;
* the truthful anchor and the hinge's truthful counterfactual, both taken
  from `da_quote_map_v5.TRUTHFUL` rather than the parent's zeros.

The parent is a hashed source in the live formal manifest, so it is imported
and extended, never edited. Its `update` hard-codes the two-dimension truth
and is therefore copied below with those three changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rl_b_c2_gate as base
from c2_reward import C2Config, c2_reward
from checkpoint_loader_v2 import load_checkpoint
from checkpoint_writer_v2 import save_checkpoint
from da_quote_map import GATE_LOGIT_BOUND, fleet_mode
from da_quote_map_v5 import (ACTION_BOX_HIGH, POLICY_DIM, QUOTE_DIM,
                             TRUTHFUL, forward_policy_v5, target_quotes_v5)
from rl_bidding import CentralizedCritic
from rl_da_env import N_BLOCKS
from rl_da_gate import DAExperimentConfig
from rl_spec_v7 import ACTION_STORAGE_QUOTE_V5, OBS_V7


class GateActorV5(nn.Module):
    """obs -> six logits: two gates, then a price and a quantity per leg."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, POLICY_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass(frozen=True)
class DAV7ExperimentConfig(DAExperimentConfig):
    """The parent's D+A knobs over the v5 quote box.

    The parent refuses any schema but `storage_quote_v4` in its own
    `__post_init__`, so the inherited checks are repeated here against the v5
    literal; every other field keeps the inherited default.
    """

    experiment_id: str = "MATD3_DA_GATE_v7"
    action_schema: str = "storage_quote_v5"

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
        if self.action_schema != "storage_quote_v5":
            raise ValueError("this trainer only builds storage_quote_v5")
        if self.anchor_weight < 0.0:
            raise ValueError("anchor_weight must be non-negative")
        if self.advantage_hinge_delta < 0.0:
            raise ValueError("advantage_hinge_delta must be non-negative")
        if self.calibration_interval < 0:
            raise ValueError("calibration_interval must be non-negative")
        if self.hinge_twin_gap_weight < 0.0:
            raise ValueError("hinge_twin_gap_weight must be non-negative")
        if abs(self.gate_logit_bound - GATE_LOGIT_BOUND) > 1e-12:
            raise ValueError(
                f"gate_logit_bound {self.gate_logit_bound} disagrees with "
                f"da_quote_map.GATE_LOGIT_BOUND {GATE_LOGIT_BOUND}; the map "
                "is the single source of the bound")


class DAGateTrainerV7(base.MATD3BC2Gate):
    """Twin-delayed MATD3 over the four-entry storage quote."""

    def __init__(self, env, experiment: DAV7ExperimentConfig,
                 *, agent_names: Optional[Sequence[str]] = None,
                 obs_spec=None, action_spec=None):
        super().__init__(env, experiment, agent_names=agent_names,
                         obs_spec=obs_spec or OBS_V7,
                         action_spec=action_spec or ACTION_STORAGE_QUOTE_V5)
        self.obs_spec = OBS_V7
        self.action_spec = ACTION_STORAGE_QUOTE_V5
        self.act_dim = QUOTE_DIM
        # The parent builds every head from the two-entry box; the v5 quote is
        # four wide, so the maps are rebuilt here instead of in the pinned file.
        self.actors = {nm: GateActorV5(self.obs_dim)
                       for nm in self.agent_names}
        self.actor_targets = {nm: GateActorV5(self.obs_dim)
                              for nm in self.agent_names}
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
            nm: optim.Adam(self.critics[nm].parameters(), lr=experiment.lr,
                           weight_decay=0.0)
            for nm in self.agent_names}
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
            # The four quote logits are the exploration branch; the gates
            # stay clean so a sampled leg is still the policy's own.
            noise = torch.randn_like(policy_out[..., 2:]) \
                * self.experiment.price_noise_std
            policy_out = torch.cat(
                [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        gate_temp = self.current_gate_temperature() if train else 1.0
        out = forward_policy_v5(policy_out, train=train,
                                gate_temperature=gate_temp)
        quotes = out["quotes"].squeeze(0).detach().numpy().copy()
        gates = out["gates"].squeeze(0).detach().numpy()
        explored = np.zeros(2, dtype=np.float32)
        source = np.zeros(2, dtype=np.float32)
        if train:
            eps = self.experiment.exploration_epsilon_at(
                self._current_episode)
            for leg in (1, 0):
                p = eps if leg == 1 else eps * \
                    self.experiment.exploration_charge_scale
                if p > 0.0 and np.random.rand() < p and gates[leg] > 0.0:
                    price_high = ACTION_BOX_HIGH[2 * leg]
                    quantity_high = ACTION_BOX_HIGH[2 * leg + 1]
                    if np.random.rand() < \
                            self.experiment.exploration_high_fraction:
                        quotes[2 * leg] = price_high * (
                            0.8 + 0.2 * np.random.rand())
                        quotes[2 * leg + 1] = quantity_high * (
                            0.8 + 0.2 * np.random.rand())
                        source[leg] = 2.0
                    else:
                        quotes[2 * leg] = price_high * np.random.rand()
                        quotes[2 * leg + 1] = quantity_high * np.random.rand()
                        source[leg] = 1.0
                    explored[leg] = 1.0
            if self._calibrating:
                # W8 calibration exposure: redraw every open leg's price and
                # quantity uniformly over its box, so the replay buffer keeps
                # payoff signal across the box wherever the actor sits.
                for leg in (1, 0):
                    if gates[leg] > 0.0:
                        quotes[2 * leg] = \
                            ACTION_BOX_HIGH[2 * leg] * np.random.rand()
                        quotes[2 * leg + 1] = \
                            ACTION_BOX_HIGH[2 * leg + 1] * np.random.rand()
                        explored[leg] = 1.0
                        source[leg] = 3.0
        return {
            "quotes": quotes,
            "gates": gates,
            "probs": out["probs"].squeeze(0).detach().numpy(),
            "gate_logits": out["gate_logits"].squeeze(0).detach().numpy(),
            # The v5 map names the four quote logits `quote_logits`; the
            # replay buffer's fields keep their historical names.
            "price_logits": out["quote_logits"].squeeze(0).detach().numpy(),
            "price_quotes": quotes.copy(),
            "sampled": gates.copy(),
            "mode": int(out["mode"].reshape(-1)[0].item()),
            "explored": explored,
            "exploration_source": source,
        }

    def _target_quotes(self, actor: nn.Module,
                       next_obs: torch.Tensor) -> torch.Tensor:
        """Smooth the quote logits only; a closed gate stays at truthful."""
        policy_out = actor(next_obs)
        noise = (torch.randn_like(policy_out[..., 2:])
                 * self.experiment.target_policy_noise).clamp(
                     -self.experiment.target_noise_clip,
                     self.experiment.target_noise_clip)
        policy_out = torch.cat(
            [policy_out[..., :2], policy_out[..., 2:] + noise], dim=-1)
        return target_quotes_v5(policy_out)

    def _actor_loss_quotes(self, policy_out: torch.Tensor) -> torch.Tensor:
        return forward_policy_v5(
            policy_out, train=True,
            gate_temperature=self.current_gate_temperature())["quotes"]

    def _anchor_tensor(self, quotes: torch.Tensor) -> torch.Tensor:
        """Sum of squared box-normalised deviations from the truthful quote.

        Exactly zero at `TRUTHFUL` and the sum over the four entries, which is
        the form the calibrated anchor weight was selected for. The bounds are
        the map's own box, so the regularizer cannot pull outside the box it
        measures distance in.
        """
        truth = torch.as_tensor(TRUTHFUL, dtype=quotes.dtype,
                                device=quotes.device)
        high = torch.as_tensor(ACTION_BOX_HIGH, dtype=quotes.dtype,
                               device=quotes.device)
        return (((quotes - truth) / high) ** 2).sum(dim=-1).mean()

    def _accumulate_gate_stats(self, diags) -> None:
        """The parent compares one quote per gate; a v5 leg posts two.

        An open leg's price and quantity are each counted against their own
        bound, and a closed leg posts the truthful vector, so saturation stays
        a statement about declared entries only.
        """
        highs = np.asarray(ACTION_BOX_HIGH, dtype=float)
        modes = []
        for nm in self.agent_names:
            d = diags[nm]
            modes.append(d["mode"])
            gates = np.asarray(d["gates"], dtype=float) > 0.0
            for leg_idx, leg in enumerate(("charge", "discharge")):
                self.gate_stats["open"][(nm, leg)] += int(gates[leg_idx])
                self.gate_stats["prob_sum"][(nm, leg)] += float(
                    d["probs"][leg_idx])
            executed = np.asarray(d["quotes"], dtype=float)
            open_idx = np.array(
                [2 * leg + k for leg in np.flatnonzero(gates) for k in (0, 1)],
                dtype=int)
            if open_idx.size:
                open_quotes = executed[open_idx]
                open_highs = highs[open_idx]
                self.gate_stats["saturated_upper"] += int(
                    np.sum(open_quotes >= 0.99 * open_highs))
                self.gate_stats["saturated_lower"] += int(
                    np.sum(open_quotes <= 0.01 * open_highs))
        self.gate_stats["modes"][fleet_mode(modes)] += 1
        self.gate_stats["steps"] += 1

    # ------------------------------------------------------------------
    # One training day
    # ------------------------------------------------------------------
    def train_day(self, env, pair: dict, *, blocks: int = N_BLOCKS) -> dict:
        """Clear one day's blocks and reward it per block.

        The differential is the settled payoff minus the same window cleared
        at the truthful declaration; C2 shaping is added on a complete day.
        The transitions are held locally and reach the buffer only after the
        day has cleared, so an `EpisodeClearingError` leaves the buffer empty
        and the aborted day's probe records on the environment as evidence.
        """
        self._current_episode = self.episodes
        interval = int(self.experiment.calibration_interval)
        self._calibrating = bool(
            interval > 0 and (self.episodes + 1) % interval == 0)
        pair_id = f"{pair['load_date']}|{pair['price_date']}"
        obs = env.reset(pair)
        soc_path = {nm: [float(env.prev_soc[nm])] for nm in self.agent_names}
        payoff = {nm: [] for nm in self.agent_names}
        baseline = {nm: [] for nm in self.agent_names}
        pending = []
        done_flags = []
        for _ in range(blocks):
            actions, diags = self._fleet_action(obs, train=True)
            next_obs, _, done, info = env.step(actions)
            for nm in self.agent_names:
                payoff[nm].append(float(info["payoff"][nm]))
                baseline[nm].append(float(info["baseline_payoff"][nm]))
                soc_path[nm].append(float(env.prev_soc[nm]))
            pending.append((obs, actions, next_obs, done, diags))
            done_flags.append(bool(done))
            obs = next_obs
            if done:
                break

        diff = {nm: [payoff[nm][t] - baseline[nm][t]
                     for t in range(len(payoff[nm]))]
                for nm in self.agent_names}
        if len(done_flags) == N_BLOCKS:
            shaped = c2_reward(diff, soc_path, done=done_flags,
                               config=C2Config(gamma=self.experiment.gamma))
            per_block = {nm: np.asarray(shaped.reward[nm], dtype=float)
                         / self.experiment.fixed_scale
                         for nm in self.agent_names}
        else:
            # C2's potential is defined on the whole 24-block day; a shortened
            # walk is rewarded by its own differential alone.
            per_block = {nm: np.asarray(diff[nm], dtype=float)
                         / self.experiment.fixed_scale
                         for nm in self.agent_names}

        anchor_terms = {nm: 0.0 for nm in self.agent_names}
        anchor_updates = 0
        for t, (obs_t, act_t, next_obs_t, done_t, diag_t) in enumerate(pending):
            options = self.agent_names
            reward = np.array([per_block[nm][t] for nm in options],
                              dtype=np.float32)
            extras = {
                "gates": np.stack([diag_t[nm]["gates"] for nm in options]),
                "probs": np.stack([diag_t[nm]["probs"] for nm in options]),
                "gate_logits": np.stack(
                    [diag_t[nm]["gate_logits"] for nm in options]),
                "price_logits": np.stack(
                    [diag_t[nm]["price_logits"] for nm in options]),
                "price_quotes": np.stack(
                    [diag_t[nm]["price_quotes"] for nm in options]),
                "sampled": np.stack(
                    [diag_t[nm]["sampled"] for nm in options]),
            }
            self.buffer.add(
                self._stack_obs(obs_t, options),
                np.stack([act_t[nm] for nm in options]),
                reward,
                self._stack_obs(next_obs_t, options),
                done_t, extras)
            self.total_steps += 1
            if len(self.buffer) >= self.experiment.batch_size \
                    and self.total_steps > self.experiment.start_steps:
                stats = self.update()
                for nm, value in (stats.get("anchor_by_agent") or {}).items():
                    if value is not None:
                        anchor_terms[nm] += float(value)
                anchor_updates += 1
            self._accumulate_gate_stats(diag_t)
        self.episodes += 1
        self.exposure[pair_id] += 1
        return {
            "pair_id": pair_id,
            "per_block": {nm: list(per_block[nm]) for nm in self.agent_names},
            "soc_end": {nm: soc_path[nm][-1] for nm in self.agent_names},
            "probe": [dict(record) for record in env.probe_records],
            "gate": self.gate_stats_snapshot(),
            "anchor_by_agent": {
                nm: (anchor_terms[nm] / anchor_updates
                     if anchor_updates else None)
                for nm in self.agent_names},
            "anchor_weight": float(self.experiment.anchor_weight),
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> dict:
        """The parent's update over the v5 quote.

        Copied rather than subclassed because the parent hard-codes the
        two-dimension truthful vector: its anchor normalises by the two-leg
        box and its hinge's counterfactual writes zeros. Both are the v5
        `TRUTHFUL` here; the twin-gap hinge and the critic losses are the
        parent's, unchanged.
        """
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
        anchor_weight = float(self.experiment.anchor_weight)
        hinge_delta = float(self.experiment.advantage_hinge_delta)
        twin_weight = float(self.experiment.hinge_twin_gap_weight)
        truthful_quote = torch.as_tensor(TRUTHFUL, dtype=act.dtype)
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
                    # counterfactual is the batch's joint action with this
                    # agent's quote set to the v5 declaration, the same
                    # reference the anchor pulls toward.
                    with torch.no_grad():
                        truthful = act.clone()
                        truthful[:, i, :] = truthful_quote
                        tq1, tq2 = critic(
                            obs_flat, truthful.reshape(batch_size, -1))
                        q_truth = torch.min(tq1, tq2)
                        # W9: never trust an advantage smaller than the two
                        # critics' own disagreement at the evaluated action.
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
                # Anchor toward the truthful quote, exactly zero there. On a
                # flat or mis-ranked Q surface this is the only term that
                # keeps the policy from drifting into corner collapse.
                if anchor_weight > 0.0:
                    anchor = self._anchor_tensor(quotes_i)
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
        # A non-finite loss means the parameters this step just moved are
        # already garbage; continuing would train on top of it.
        losses = [float(np.mean(critic_losses))] \
            + [float(v) for v in actor_losses.values()]
        if not np.all(np.isfinite(losses)):
            raise FloatingPointError(
                f"update {self.updates}: non-finite loss "
                f"{[v for v in losses if not np.isfinite(v)]}")
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
    def save(self, root, *, level: str, dataset: str, seed: int,
             config_hash_value: str, data: Optional[dict] = None,
             code_revision: str = "working-tree",
             exposure_log: Optional[str] = None,
             eval_provenance: Optional[str] = None):
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
            "b1_source": "local_da_env_truthful_window",
            "gate": {
                "policy_dim": POLICY_DIM,
                "quote_dim": QUOTE_DIM,
                "leg_highs": list(ACTION_BOX_HIGH),
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

    def load(self, root) -> dict:
        bundle = load_checkpoint(root)
        specs = bundle["manifest"].get("specs", {})
        action = specs.get("action_spec", {}).get("name")
        obs = specs.get("obs_spec", {}).get("name")
        if action != self.action_spec.name or obs != self.obs_spec.name:
            raise ValueError(
                f"checkpoint specs obs {obs!r} / action {action!r} are not "
                f"{self.obs_spec.name!r} / {self.action_spec.name!r}; the "
                "v5 quote box cannot reuse an earlier one")
        return super().load(root)
