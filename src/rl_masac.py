# rl_masac.py
"""Multi-agent SAC (MASAC V1): stochastic Gaussian actors + SAC twin-critic.

Drop-in CTDE trainer mirroring the MATD3 interface used by `_train_matd3`
(actors/actor_targets/actor_opts, critics/critic_targets/critic_opts, buffer,
total_steps, start_steps, select_action, remember, update). Only the policy
parameterization and the TD target/actor objective change:

  - deterministic actor + external annealed noise  -> stochastic Gaussian actor
    (pre-tanh mu + learned log_std, tanh squashed to action bounds); exploration
    comes from sampling, not injected noise.
  - TD target subtracts the acting agent's own entropy: the centralized critic i
    conditions on a joint next action sampled from the target policies, and
    targets y = r + gamma*(1-done)*(min(q1,q2) - alpha_i * log pi_i(next a_i)).
  - actor maximizes min(q1,q2) plus an automatic-temperature entropy bonus;
    log_alpha is tuned per agent toward a target entropy.

Rewards stay on MATD3's dynamic scale (update() divides by buffer.reward_std()
and reuses the same scale for any logit deviation penalties), so critic/actor
magnitudes and alpha live in normalized units.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_bidding import CentralizedCritic, ReplayBuffer

LOG_STD_MIN = -2.0
LOG_STD_MAX = 2.0


class StochasticActor(nn.Module):
    """Gaussian policy over pre-tanh logits, tanh-squashed to action bounds.

    forward() returns the deterministic MEAN action (used by evaluation and
    policy saving, matching the deterministic-Actor contract); sampling is
    exposed through sample(). The state dict keeps the `net` trunk (so
    save_policy can read net[0].in_features) plus a `log_std` parameter and the
    action_low/action_high buffers used by the rest of the pipeline.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 action_bounds: torch.Tensor):
        super().__init__()
        self.act_dim = act_dim
        # Shared trunk to a per-dim pre-tanh mean (no final Tanh).
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("action_low", action_bounds[0].clone())
        self.register_buffer("action_high", action_bounds[1].clone())
        mid = (action_bounds[0] + action_bounds[1]) / 2.0
        half = (action_bounds[1] - action_bounds[0]) / 2.0
        self.register_buffer("_mid", mid.clone())
        self.register_buffer("_half", half.clone())
        self.register_buffer("_log_half", torch.log(half))

    def _mu_std(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return mu, std

    def _affine(self, raw: torch.Tensor) -> torch.Tensor:
        return self._mid + raw * self._half

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action (eval/save semantics)."""
        mu, _ = self._mu_std(obs)
        return self._affine(torch.tanh(mu))

    def forward_logits(self, obs: torch.Tensor):
        """Return (pre-tanh mean mu, mean action) for a deviation penalty."""
        mu, _ = self._mu_std(obs)
        return mu, self._affine(torch.tanh(mu))

    def _log_prob(self, mu: torch.Tensor, std: torch.Tensor,
                  z: torch.Tensor) -> torch.Tensor:
        """SAC log-probability of a sampled action (per-dim then summed)."""
        gauss = (-0.5 * ((z - mu) / std).pow(2)
                 - torch.log(std) - 0.5 * math.log(2.0 * math.pi))
        pre = torch.tanh(z)
        correction = torch.log(1.0 - pre.pow(2) + 1e-6)
        return (gauss - correction - self._log_half).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample -> (action, log_prob); action in bounds."""
        mu, std = self._mu_std(obs)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        action = self._affine(torch.tanh(z))
        return action, self._log_prob(mu, std, z)


class MASAC:
    """Multi-agent SAC with centralized critics (CTDE). Drop-in for MATD3."""

    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 batch_size: int = 128, buffer_capacity: int = 100_000,
                 start_steps: int = 500, bid_dev_penalty: float = 0.0,
                 offer_dev_penalty: float = 0.0,
                 alpha_init: float = 0.1, target_entropy: float = -2.0,
                 critic_factory=None):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.bid_dev_penalty = bid_dev_penalty
        self.offer_dev_penalty = offer_dev_penalty
        self.target_entropy = target_entropy

        obs_dim = env.get_state_dim()
        act_dim = 2
        action_bounds = env.get_action_bounds()
        n_agents = len(env.rl_agents)

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_names: List[str] = [a.name for a in env.rl_agents]

        # Stochastic actors per agent (target copies for the TD target).
        self.actors: Dict[str, StochasticActor] = {}
        self.actor_targets: Dict[str, StochasticActor] = {}
        self.actor_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            self.actors[a.name] = StochasticActor(obs_dim, act_dim,
                                                  action_bounds)
            self.actor_targets[a.name] = StochasticActor(obs_dim, act_dim,
                                                         action_bounds)
            self.actor_targets[a.name].load_state_dict(
                self.actors[a.name].state_dict())
            self.actor_opts[a.name] = optim.Adam(
                self.actors[a.name].parameters(), lr=lr, weight_decay=1e-5)

        # Centralized twin critics per agent (scalar Q), same as MATD3.
        self.critics: Dict[str, CentralizedCritic] = {}
        self.critic_targets: Dict[str, CentralizedCritic] = {}
        self.critic_opts: Dict[str, optim.Adam] = {}
        def make_critic(idx: int):
            if critic_factory is not None:
                return critic_factory(n_agents, obs_dim, act_dim, idx)
            return CentralizedCritic(n_agents, obs_dim, act_dim)
        for idx, a in enumerate(env.rl_agents):
            self.critics[a.name] = make_critic(idx)
            self.critic_targets[a.name] = make_critic(idx)
            self.critic_targets[a.name].load_state_dict(
                self.critics[a.name].state_dict())
            self.critic_targets[a.name].eval()
            self.critic_opts[a.name] = optim.Adam(
                self.critics[a.name].parameters(), lr=lr, weight_decay=1e-4)

        # Automatic temperature per agent (log_alpha scalar).
        self.log_alpha: Dict[str, torch.Tensor] = {}
        self.alpha_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            la = torch.tensor([math.log(max(alpha_init, 1e-6))],
                              requires_grad=True)
            self.log_alpha[a.name] = la
            self.alpha_opts[a.name] = optim.Adam([la], lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

    def _alpha(self, nm: str) -> torch.Tensor:
        return self.log_alpha[nm].exp().detach()

    def _build_global_state(self, obs_batch, agent_idx):
        return obs_batch.reshape(obs_batch.shape[0], -1)

    def select_action(self, obs: np.ndarray, agent_name: str,
                      add_noise: bool = False) -> np.ndarray:
        """Sample from the stochastic policy (exploration is intrinsic)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actors[agent_name].sample(obs_t)
            return action.squeeze(0).numpy()

    def remember(self, obs, act, rew, next_obs, done):
        """Store a transition with raw rewards (update() normalizes)."""
        self.buffer.add(obs, act, rew, next_obs, done)

    def _target_next(self, next_obs, i):
        """Sample joint next actions from target actors (+ own log_prob)."""
        next_actions = []
        logp_next = None
        for j, nm2 in enumerate(self.agent_names):
            a, logp = self.actor_targets[nm2].sample(next_obs[:, j, :])
            next_actions.append(a)
            if j == i:
                logp_next = logp
        return torch.stack(next_actions, dim=1), logp_next

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": None, "actor_loss": None,
                    "critic_loss_by_agent": {}, "actor_loss_by_agent": {}}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        B = obs.shape[0]
        scale = self.buffer.reward_std()
        rew = rew / scale

        critic_losses = []
        critic_loss_by_agent = {}
        for i, nm in enumerate(self.agent_names):
            critic = self.critics[nm]
            critic_target = self.critic_targets[nm]
            opt = self.critic_opts[nm]
            with torch.no_grad():
                next_act_all, logp_next = self._target_next(next_obs, i)
                tq1, tq2 = critic_target(
                    self._build_global_state(next_obs, i),
                    next_act_all.view(B, -1))
                alpha_i = self._alpha(nm)
                target_q = torch.min(tq1, tq2) - alpha_i * logp_next.unsqueeze(1)
                y = rew[:, i].unsqueeze(1) + self.gamma * \
                    (1.0 - done[:, i].unsqueeze(1)) * target_q
            cur1, cur2 = critic(
                self._build_global_state(obs, i), act.view(B, -1))
            loss = nn.functional.mse_loss(cur1, y) + \
                nn.functional.mse_loss(cur2, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            critic_losses.append(loss.detach().item())
            critic_loss_by_agent[nm] = loss.detach().item()

        actor_losses = []
        actor_loss_by_agent = {}
        if self.total_steps % self.policy_delay == 0:
            for i, nm in enumerate(self.agent_names):
                actor = self.actors[nm]
                critic = self.critics[nm]
                opt = self.actor_opts[nm]
                alpha_i = self._alpha(nm)

                new_a, logp = actor.sample(obs[:, i, :])
                new_actions = []
                for j, nm2 in enumerate(self.agent_names):
                    if j == i:
                        new_actions.append(new_a)
                    else:
                        new_actions.append(act[:, j, :])
                new_act_all = torch.stack(new_actions, dim=1)
                q1, q2 = critic(
                    self._build_global_state(obs, i),
                    new_act_all.view(B, -1))
                value = torch.min(q1, q2).squeeze(-1)          # (B,)
                a_loss = (alpha_i * logp - value).mean()
                # Optional logit deviation penalty on the pre-tanh mean.
                mu, _ = actor.forward_logits(obs[:, i, :])
                if self.bid_dev_penalty > 0:
                    a_loss = a_loss + (self.bid_dev_penalty / scale) \
                        * (mu[:, 0] ** 2).mean()
                if self.offer_dev_penalty > 0:
                    a_loss = a_loss + (self.offer_dev_penalty / scale) \
                        * (mu[:, 1] ** 2).mean()
                opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()

                # Temperature: keep entropy near target_entropy.
                logp_det = logp.detach()
                alpha_loss = -(self.log_alpha[nm]
                               * (logp_det.mean() + self.target_entropy))
                aopt = self.alpha_opts[nm]
                aopt.zero_grad()
                alpha_loss.backward()
                aopt.step()

                actor_losses.append(a_loss.detach().item())
                actor_loss_by_agent[nm] = a_loss.detach().item()

        # Polyak update of actor/critic targets.
        for nm in self.agent_names:
            for target, source in [
                (self.actor_targets[nm], self.actors[nm]),
                (self.critic_targets[nm], self.critics[nm]),
            ]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data
                                  + (1 - self.tau) * tp.data)

        avg_critic = float(np.mean(critic_losses))
        avg_actor = float(np.mean(actor_losses)) if actor_losses else None
        return {"critic_loss": avg_critic, "actor_loss": avg_actor,
                "critic_loss_by_agent": critic_loss_by_agent,
                "actor_loss_by_agent": actor_loss_by_agent}
