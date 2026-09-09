# rl_quantile.py
"""Fixed-grid Quantile Distributional Critic (qmatd3), a drop-in MATD3 variant.

Replaces each centralized critic's single point-estimate Q with N quantile
heads (fixed symmetric grid tau_i = (i + 0.5) / N). The critic is otherwise
unchanged: one twin (q1/q2) centralised critic per agent over the full joint
observation and actions, clipped double-Q on the value used by the actor.

Locked design decisions (see docs/plan):
  - The TD target is the scalar expectation-min approximation
        y = r + gamma * (1 - done) * min(mean Z1_target, mean Z2_target)
    and every online quantile head of BOTH twins regresses to that scalar via
    quantile-Huber. Against a scalar target the learned distribution
    degenerates toward the scalar; the method's value is robust/asymmetric
    regression (quantile-weighted, outlier-tolerant), not calibrated quantiles.
  - Fixed grid N=32 (CLI-configurable via --n-quantiles).
  - Reward normalization identical to MATD3: dynamic buffer.reward_std() scale
    inside update(), with the actor-logit deviation penalty divided by the same
    scale. Actor/ReplayBuffer/obs/action spec are reused unchanged, so saved
    policies stay actors and the eval path is untouched.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple

from rl_bidding import Actor, ReplayBuffer


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                        tau: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile-Huber loss, vectorized over the N-head output.

    pred: (B, N) online quantile-head predictions.
    target: (B,) or (B, 1) scalar regression target y.
    tau: (N,) fixed grid in (0, 1).
    kappa: Huber threshold.

    rho_tau^kappa(delta) = |tau - I(delta < 0)| * L_kappa(delta) / kappa
    with L_kappa = 0.5*delta^2 for |delta| < kappa, else
    kappa*(|delta| - 0.5*kappa). Scalar mean over all B*N elements.
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)               # (B, 1)
    delta = target - pred                            # (B, N)
    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta < kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa))           # (B, N)
    indicator = (delta < 0.0).float()                # (B, N)
    weight = (tau.view(1, -1) - indicator).abs()     # (B, N)
    return (weight * huber / kappa).mean()


class QuantileCentralizedCritic(nn.Module):
    """Per-agent centralized critic with twin quantile heads.

    Mirrors `CentralizedCritic` (rl_bidding) topology and joint input
    (obs*n_agents + act*n_agents); only the final head width becomes
    n_quantiles. Dropout lives in the online heads only (target copies .eval()).
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int,
                 n_quantiles: int = 32):
        super().__init__()
        self.critic_in = obs_dim * n_agents + act_dim * n_agents
        self.n_quantiles = n_quantiles
        h = 256 if n_agents <= 20 else 512
        def head():
            return nn.Sequential(
                nn.Linear(self.critic_in, h), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, h // 2), nn.ReLU(),
                nn.Linear(h // 2, n_quantiles),
            )
        self.q1 = head()
        self.q2 = head()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """obs (B, obs*n_agents), actions (B, act*n_agents). -> (q1, q2) (B, N)."""
        xu = torch.cat([obs, actions], dim=-1)
        return self.q1(xu), self.q2(xu)


class QuantileMATD3:
    """Multi-agent quantile distributional critic (CTDE).

    Mirrors the MATD3 trainer interface (actors/actor_targets/actor_opts,
    critics/critic_targets/critic_opts, buffer, total_steps, select_action,
    remember, update) so the shared _train_matd3 loop is drop-in.
    """

    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_std: float = 0.2, noise_clip: float = 0.5,
                 batch_size: int = 128, buffer_capacity: int = 100_000,
                 start_steps: int = 500, bid_dev_penalty: float = 0.0,
                 offer_dev_penalty: float = 0.0,
                 final_noise_std: float = 0.05,
                 noise_anneal_steps: int = 5000,
                 n_quantiles: int = 32, critic_factory=None):
        """critic_factory: optional (n_agents, obs_dim, act_dim, agent_idx)
        -> nn.Module (N quantile heads, e.g. physics) instead of the default
        QuantileCentralizedCritic."""
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.bid_dev_penalty = bid_dev_penalty
        self.offer_dev_penalty = offer_dev_penalty
        self.final_noise_std = final_noise_std
        self.noise_anneal_steps = max(1, noise_anneal_steps)
        self.n_quantiles = n_quantiles

        obs_dim = env.get_state_dim()
        act_dim = 2
        action_bounds = env.get_action_bounds()
        n_agents = len(env.rl_agents)

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_names: List[str] = [a.name for a in env.rl_agents]
        # Fixed symmetric quantile grid.
        self.qt = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) \
            / n_quantiles

        # One Actor per agent (identical to MATD3).
        self.actors: Dict[str, Actor] = {}
        self.actor_targets: Dict[str, Actor] = {}
        self.actor_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            self.actors[a.name] = Actor(obs_dim, act_dim, action_bounds)
            self.actor_targets[a.name] = Actor(obs_dim, act_dim, action_bounds)
            self.actor_targets[a.name].load_state_dict(
                self.actors[a.name].state_dict())
            self.actor_opts[a.name] = optim.Adam(
                self.actors[a.name].parameters(), lr=lr, weight_decay=1e-5)

        # One Quantile critic per agent; target copies eval() to disable dropout.
        self.critics: Dict[str, QuantileCentralizedCritic] = {}
        self.critic_targets: Dict[str, QuantileCentralizedCritic] = {}
        self.critic_opts: Dict[str, optim.Adam] = {}
        def make_critic(idx: int):
            if critic_factory is not None:
                return critic_factory(n_agents, obs_dim, act_dim, idx)
            return QuantileCentralizedCritic(
                n_agents, obs_dim, act_dim, n_quantiles)
        for idx, a in enumerate(env.rl_agents):
            self.critics[a.name] = make_critic(idx)
            self.critic_targets[a.name] = make_critic(idx)
            self.critic_targets[a.name].load_state_dict(
                self.critics[a.name].state_dict())
            self.critic_targets[a.name].eval()
            self.critic_opts[a.name] = optim.Adam(
                self.critics[a.name].parameters(), lr=lr, weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

    def _build_global_state(self, obs_batch, agent_idx):
        """Full joint observation (batch, obs_dim*n_agents); index unused."""
        return obs_batch.reshape(obs_batch.shape[0], -1)

    def select_action(self, obs: np.ndarray, agent_name: str,
                      add_noise: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actors[agent_name](obs_t).squeeze(0)
            if add_noise:
                ratio = min(1.0, self.total_steps / self.noise_anneal_steps)
                noise_std = (self.noise_std * (1 - ratio)
                             + self.final_noise_std * ratio)
                noise = torch.randn_like(action) * noise_std
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise).clamp(self.action_low,
                                                self.action_high)
            return action.numpy()

    def remember(self, obs, act, rew, next_obs, done):
        """Store a transition with raw rewards (update() normalizes by scale)."""
        self.buffer.add(obs, act, rew, next_obs, done)

    def update(self) -> dict:
        """One TD3-style update with a quantile distributional critic."""
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": None, "actor_loss": None,
                    "critic_loss_by_agent": {}, "actor_loss_by_agent": {}}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        B = obs.shape[0]
        n = self.n_agents
        obs_dim = self.obs_dim

        # Dynamic reward scale, shared by the actor-logit penalties.
        scale = self.buffer.reward_std()
        rew = rew / scale

        critic_losses = []
        critic_loss_by_agent = {}
        for i, nm in enumerate(self.agent_names):
            critic = self.critics[nm]
            critic_target = self.critic_targets[nm]
            opt = self.critic_opts[nm]

            with torch.no_grad():
                next_actions = []
                for j, nm2 in enumerate(self.agent_names):
                    na = self.actor_targets[nm2](next_obs[:, j, :])
                    noise = (torch.randn_like(na) * self.noise_std
                             * 0.5).clamp(-self.noise_clip * 0.5,
                                          self.noise_clip * 0.5)
                    na = (na + noise).clamp(self.action_low, self.action_high)
                    next_actions.append(na)
                next_act_all = torch.stack(next_actions, dim=1)
                tq1, tq2 = critic_target(
                    self._build_global_state(next_obs, i),
                    next_act_all.view(B, -1))
                # Expectation of each target distribution, then clipped double-Q.
                tq1 = tq1.mean(dim=-1, keepdim=True)
                tq2 = tq2.mean(dim=-1, keepdim=True)
                y = rew[:, i].unsqueeze(1) + self.gamma * \
                    (1.0 - done[:, i].unsqueeze(1)) * torch.min(tq1, tq2)

            cur_q1, cur_q2 = critic(
                self._build_global_state(obs, i), act.view(B, -1))
            loss = quantile_huber_loss(cur_q1, y, self.qt) \
                + quantile_huber_loss(cur_q2, y, self.qt)

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

                logits, new_act = actor.forward_logits(obs[:, i, :])
                new_actions = []
                for j, nm2 in enumerate(self.agent_names):
                    if j == i:
                        new_actions.append(new_act)
                    else:
                        new_actions.append(act[:, j, :])
                new_act_all = torch.stack(new_actions, dim=1)

                q1, q2 = critic(
                    self._build_global_state(obs, i),
                    new_act_all.view(B, -1))
                # Value = min of the twin means over the quantile heads.
                value = torch.min(q1.mean(dim=-1), q2.mean(dim=-1))
                a_loss = -value.mean()
                if self.bid_dev_penalty > 0:
                    a_loss = a_loss + (self.bid_dev_penalty / scale) \
                        * (logits[:, 0] ** 2).mean()
                if self.offer_dev_penalty > 0:
                    a_loss = a_loss + (self.offer_dev_penalty / scale) \
                        * (logits[:, 1] ** 2).mean()

                opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
                actor_losses.append(a_loss.detach().item())
                actor_loss_by_agent[nm] = a_loss.detach().item()

        # Polyak update of target networks.
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
