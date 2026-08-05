# rl_bidding_per.py
"""MATD3 with Prioritized Experience Replay (PER) for agent bidding.

Same architecture as rl_bidding.py (Tanh output, L2-regularized Adam),
with the uniform ReplayBuffer replaced by a SumTree-based prioritized
buffer. Joint transitions carry per-agent TD-errors, so priorities are
max-aggregated across agents to keep joint-sampling intact.

This module exists for A/B comparison against the uniform baseline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from collections import deque

from models import Agent, MarketConfig
from rl_env import BiddingEnv, N_BLOCKS


# ---------------------------------------------------------------------------
# Actor network — decentralized, sees only local observations
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """MLP: obs → 256 → 128 → act_dim (tanh-squashed to action bounds)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 action_bounds: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )
        self.register_buffer("action_low", action_bounds[0])
        self.register_buffer("action_high", action_bounds[1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        mid = (self.action_low + self.action_high) / 2.0
        half = (self.action_high - self.action_low) / 2.0
        return mid + raw * half


# ---------------------------------------------------------------------------
# Centralized Critic — sees all agents' unique obs + all actions
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """Twin Q-networks taking global state + global actions → Q-value.

    Input: [agent_i_full_obs | all_other_unique_obs | all_actions]
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int,
                 unique_obs_dim: int):
        super().__init__()
        # Total input: obs_dim (self) + unique_obs_dim*(n_agents-1) (others)
        #            + act_dim * n_agents (all actions)
        self.critic_in = (obs_dim + unique_obs_dim * (n_agents - 1)
                          + act_dim * n_agents)
        h = 256 if n_agents <= 20 else 512
        self.q1 = nn.Sequential(
            nn.Linear(self.critic_in, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.critic_in, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        xu = torch.cat([obs, actions], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor):
        return self.q1(torch.cat([obs, actions], dim=-1))


# ---------------------------------------------------------------------------
# SumTree — flat-array binary tree for O(log N) priority sampling
# ---------------------------------------------------------------------------

class SumTree:
    """Segment-tree over priorities; leaves hold data indices.

    Node 0 is the root; node i has children 2i+1 (left) and 2i+2 (right).
    Leaves occupy nodes [capacity-1, 2*capacity-1).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.size = 0
        self.next_idx = 0

    def add(self, priority: float):
        """Insert a new leaf with the given priority, evicting oldest if full."""
        idx = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._update_leaf(idx, priority)

    def _update_leaf(self, idx: int, priority: float):
        node = idx + self.capacity - 1
        delta = priority - self.tree[node]
        self.tree[node] = priority
        while node > 0:
            node = (node - 1) // 2
            self.tree[node] += delta

    def total(self) -> float:
        return float(self.tree[0])

    def get(self, s: float) -> int:
        """Return leaf index whose cumulative priority covers mass s."""
        node = 0
        while node < self.capacity - 1:
            left = 2 * node + 1
            if s <= self.tree[left]:
                node = left
            else:
                s -= self.tree[left]
                node = left + 1
        return node - (self.capacity - 1)

    def update(self, idx: int, priority: float):
        self._update_leaf(idx, priority)


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer — shared across agents
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """SumTree-backed replay buffer with proportional priority sampling.

    Priorities are max-aggregated across agents (one priority per joint
    transition), preserving joint-sampling for the centralized critics.
    Importance-sampling weights correct the sampling bias; beta anneals
    from beta_start to 1.0 as the MATD3 trainer progresses.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6,
                 beta_start: float = 0.4, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.eps = eps
        self.tree = SumTree(capacity)
        self.data = deque(maxlen=capacity)
        self.max_priority = 1.0

    def set_beta(self, beta: float):
        self.beta = beta

    def add(self, obs, action, reward, next_obs):
        self.data.append((
            obs.copy(), action.copy(), reward.copy(), next_obs.copy()))
        self.tree.add(self.max_priority)

    def sample(self, batch_size: int):
        """Return (obs, act, rew, next_obs, indices, is_weights).

        is_weights are normalized to max weight 1.0 (bounded bias
        correction as in Schaul et al. 2016).
        """
        n = min(batch_size, len(self.data))
        indices, batch, total = [], [], self.tree.total()
        segment = total / n
        for i in range(n):
            # Jitter within each segment to decorrelate consecutive samples
            s = np.random.uniform(i * segment, (i + 1) * segment)
            idx = self.tree.get(s)
            indices.append(idx)
            batch.append(self.data[idx])
        obs, act, rew, next_obs = zip(*batch)
        prios = np.array([self._leaf_priority(i) for i in indices])
        probs = prios / (self.tree.total() or 1.0)
        weights = (len(self.data) * probs) ** (-self.beta)
        weights = weights / (weights.max() or 1.0)
        return (torch.as_tensor(np.array(obs), dtype=torch.float32),
                torch.as_tensor(np.array(act), dtype=torch.float32),
                torch.as_tensor(np.array(rew), dtype=torch.float32),
                torch.as_tensor(np.array(next_obs), dtype=torch.float32),
                indices,
                torch.as_tensor(weights, dtype=torch.float32))

    def _leaf_priority(self, idx: int) -> float:
        return float(self.tree.tree[idx + self.capacity - 1])

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Write back per-sample priorities; clamp small ones above eps."""
        for idx, p in zip(indices, priorities):
            p = max(p, self.eps)
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# MATD3 Trainer (PER variant) — centralized critic, decentralized actors
# ---------------------------------------------------------------------------

class MATD3:
    """Multi-Agent TD3 with PER. API-compatible with rl_bidding.MATD3.

    Differences from the uniform baseline:
      - buffer is a PrioritizedReplayBuffer
      - critic loss is scaled by importance-sampling weights
      - TD-errors computed during critic updates write priorities back
    """

    def __init__(self, env: BiddingEnv, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_delay: int = 2, noise_std: float = 0.2,
                 noise_clip: float = 0.5, batch_size: int = 128,
                 buffer_capacity: int = 100_000, start_steps: int = 500,
                 beta_anneal_steps: int = 20_000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.beta_anneal_steps = beta_anneal_steps

        obs_dim = env.get_state_dim()
        act_dim = 2
        action_bounds = env.get_action_bounds()
        n_agents = len(env.rl_agents)
        unique_dim = env.unique_obs_dim

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.unique_obs_dim = unique_dim
        self.agent_names = [a.name for a in env.rl_agents]

        # One Actor per agent
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

        # One Centralized Critic per agent
        self.critics: Dict[str, CentralizedCritic] = {}
        self.critic_targets: Dict[str, CentralizedCritic] = {}
        self.critic_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            self.critics[a.name] = CentralizedCritic(
                n_agents, obs_dim, act_dim, unique_dim)
            self.critic_targets[a.name] = CentralizedCritic(
                n_agents, obs_dim, act_dim, unique_dim)
            self.critic_targets[a.name].load_state_dict(
                self.critics[a.name].state_dict())
            self.critic_opts[a.name] = optim.Adam(
                self.critics[a.name].parameters(), lr=lr, weight_decay=1e-4)

        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.total_steps = 0
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

    def _build_global_state(self, obs_batch, agent_idx):
        """Build centralized critic input for agent agent_idx.

        obs_batch: (batch, n_agents, obs_dim)
        Returns: (batch, obs_dim + unique_dim*(n-1)) — full obs for self
                 + unique parts from all others.
        """
        batch_size = obs_batch.shape[0]
        n = self.n_agents
        u_dim = self.unique_obs_dim
        # Agent's own full observation
        own = obs_batch[:, agent_idx, :]
        # Other agents' unique observations
        other_unique = []
        for j in range(n):
            if j != agent_idx:
                # unique part is at the START of the observation
                other_unique.append(obs_batch[:, j, :u_dim])
        others = torch.cat(other_unique, dim=-1) if other_unique else \
            torch.zeros(batch_size, 0, device=obs_batch.device)
        return torch.cat([own, others], dim=-1)

    def select_action(self, obs: np.ndarray, agent_name: str,
                      add_noise: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actors[agent_name](obs_t).squeeze(0)
            if add_noise:
                noise = torch.randn_like(action) * self.noise_std
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise).clamp(self.action_low,
                                                self.action_high)
            return action.numpy()

    def update(self) -> dict:
        """Execute one TD3 update step. Returns critic_loss and actor_loss
        (actor_loss is None on steps without actor update)."""
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": None, "actor_loss": None}

        obs, act, rew, next_obs, indices, is_weights = \
            self.buffer.sample(self.batch_size)
        # obs: (batch, n_agents, obs_dim), act: (batch, n_agents, act_dim)
        # rew: (batch, n_agents), next_obs: (batch, n_agents, obs_dim)
        # is_weights: (batch,) — max-aggregated importance correction

        n = self.n_agents
        u_dim = self.unique_obs_dim
        batch = self.batch_size

        # Max-aggregated TD-error per transition across all agents
        td_errs = np.zeros(batch, dtype=np.float64)

        # ---- Update each agent's critic ----
        critic_losses = []
        for i, nm in enumerate(self.agent_names):
            critic = self.critics[nm]
            critic_target = self.critic_targets[nm]
            opt = self.critic_opts[nm]

            with torch.no_grad():
                # Target actions with noise
                next_actions = []
                for j, nm2 in enumerate(self.agent_names):
                    na = self.actor_targets[nm2](next_obs[:, j, :])
                    noise = (torch.randn_like(na) * self.noise_std
                             * 0.5).clamp(-self.noise_clip * 0.5,
                                          self.noise_clip * 0.5)
                    na = (na + noise).clamp(self.action_low, self.action_high)
                    next_actions.append(na)
                next_act_all = torch.stack(next_actions, dim=1)

                target_q1, target_q2 = critic_target(
                    self._build_global_state(next_obs, i),
                    next_act_all.view(batch, -1))
                target_q = rew[:, i].unsqueeze(1) + self.gamma * \
                    torch.min(target_q1, target_q2)

            current_q1, current_q2 = critic(
                self._build_global_state(obs, i),
                act.view(batch, -1))
            # TD-error from the first critic head; max-aggregate across agents
            td_errs = np.maximum(
                td_errs,
                (target_q - current_q1).abs().squeeze(1)
                .detach().cpu().numpy())
            weighted_loss = (
                nn.functional.mse_loss(current_q1, target_q,
                                       reduction="none").mean(dim=1)
                + nn.functional.mse_loss(current_q2, target_q,
                                         reduction="none").mean(dim=1))
            critic_loss = (is_weights * weighted_loss).mean()

            opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            critic_losses.append(critic_loss.detach().item())

        # Write priorities back (proportional to |TD| ** alpha)
        self.buffer.update_priorities(indices, td_errs ** 0.6)

        # ---- Delayed actor update ----
        actor_losses = []
        if self.total_steps % self.policy_delay == 0:
            for i, nm in enumerate(self.agent_names):
                actor = self.actors[nm]
                critic = self.critics[nm]
                opt = self.actor_opts[nm]

                # Current actions with this agent's actor
                new_actions = []
                for j, nm2 in enumerate(self.agent_names):
                    if j == i:
                        new_actions.append(actor(obs[:, j, :]))
                    else:
                        new_actions.append(act[:, j, :])
                new_act_all = torch.stack(new_actions, dim=1)

                actor_loss = -critic.q1_forward(
                    self._build_global_state(obs, i),
                    new_act_all.view(batch, -1)).mean()

                opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
                actor_losses.append(actor_loss.detach().item())

        # ---- Polyak update target networks ----
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
        return {"critic_loss": avg_critic, "actor_loss": avg_actor}

    def train(self, n_episodes: int = 100, verbose: bool = True) \
            -> Dict[str, list]:
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        log_dir = os.path.join("runs", datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + "-per")
        writer = SummaryWriter(log_dir)

        history = {nm: {"reward": []} for nm in self.agent_names}
        history["welfare"] = []
        history["re_rate"] = []

        for ep in range(n_episodes):
            obs = self.env.reset()
            ep_rewards = {nm: 0.0 for nm in self.agent_names}

            for _ in range(N_BLOCKS):
                # Select actions
                actions = {}
                for nm in self.agent_names:
                    actions[nm] = self.select_action(
                        obs[nm], nm, add_noise=True)

                next_obs, rewards, done, info = self.env.step(actions)
                for nm in self.agent_names:
                    ep_rewards[nm] += rewards.get(nm, 0.0)

                # Store transitions in replay buffer
                obs_arr = np.stack([obs[nm] for nm in self.agent_names])
                act_arr = np.stack([actions[nm] for nm in self.agent_names])
                rew_arr = np.array([rewards.get(nm, 0.0)
                                   for nm in self.agent_names])
                next_obs_arr = np.stack(
                    [next_obs.get(nm, np.zeros(self.obs_dim))
                     for nm in self.agent_names])
                self.buffer.add(obs_arr, act_arr, rew_arr, next_obs_arr)
                self.total_steps += 1
                # Anneal importance-sampling beta with progress
                progress = min(1.0, self.total_steps / self.beta_anneal_steps)
                self.buffer.set_beta(0.4 + 0.6 * progress)

                obs = next_obs

            # Update after each episode
            if self.total_steps >= self.start_steps:
                for _ in range(N_BLOCKS):
                    self.update()

            for nm in self.agent_names:
                history[nm]["reward"].append(ep_rewards[nm])

            history["welfare"].append(info.get("welfare", 0.0))
            history["re_rate"].append(info.get("re_rate", 0.0))

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg_r = np.mean([ep_rewards[nm]
                                for nm in self.agent_names])
                print(f"  Ep {ep+1}/{n_episodes}: avg_reward={avg_r:.3f}, "
                      f"welfare={info.get('welfare', 0):.0f}, "
                      f"RE={info.get('re_rate', 0):.1f}%")

            # TensorBoard logging
            writer.add_scalar("Reward/mean", np.mean(
                [ep_rewards[nm] for nm in self.agent_names]), ep)
            writer.add_scalar("Welfare", info.get("welfare", 0), ep)
            writer.add_scalar("RE_Rate", info.get("re_rate", 0), ep)

        writer.close()
        return history


# Global cache for trained policies
_TRAINED_POLICIES: Dict[str, Actor] = {}


def train_rl_agents(agents: List[Agent], config: MarketConfig,
                    n_episodes: int = 100, verbose: bool = True) \
                    -> Tuple[Dict[str, Actor], dict]:
    """Train MATD3-PER for all prosumer agents."""
    env = BiddingEnv(agents, config)
    matd3 = MATD3(env)
    history = matd3.train(n_episodes, verbose=verbose)

    global _TRAINED_POLICIES
    _TRAINED_POLICIES.update(matd3.actors)

    return matd3.actors, history


def save_policies(policies: dict, path: str):
    """Save trained Actor policies to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {nm: net.state_dict() for nm, net in policies.items()}
    torch.save(state, path)


def load_policies(path: str, obs_dim: int,
                  action_bounds: torch.Tensor) -> Dict[str, Actor]:
    """Load trained Actor policies from disk."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    policies = {}
    for nm, sd in state.items():
        net = Actor(obs_dim, 2, action_bounds)
        net.load_state_dict(sd)
        policies[nm] = net
    global _TRAINED_POLICIES
    _TRAINED_POLICIES.update(policies)
    return policies
