# rl_bidding.py
"""MATD3 with CTDE for agent bidding in electricity markets.

Centralized Training with Decentralized Execution: each agent has an
Actor (local obs → action) and a Centralized Critic (global obs+act → Q).
The Critic sees all agents' unique observations and actions during training,
resolving the non-stationarity problem of independent learners.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

from models import Agent, MarketConfig
from rl_env import BiddingEnv, BLOCK_SIZE, N_BLOCKS


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
# Replay Buffer — shared across all agents, supports off-policy training
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size replay buffer storing (obs, action, reward, next_obs)."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs):
        """obs/next_obs: (n_agents, obs_dim), action: (n_agents, act_dim),
           reward: (n_agents,)"""
        self.buffer.append((
            obs.copy(), action.copy(), reward.copy(), next_obs.copy()))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, rew, next_obs = zip(*batch)
        return (torch.as_tensor(np.array(obs), dtype=torch.float32),
                torch.as_tensor(np.array(act), dtype=torch.float32),
                torch.as_tensor(np.array(rew), dtype=torch.float32),
                torch.as_tensor(np.array(next_obs), dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# MATD3 Trainer — centralized critic, decentralized actors
# ---------------------------------------------------------------------------

class MATD3:
    """Multi-Agent TD3 with centralized critics.

    Parameters
    ----------
    env : BiddingEnv
    lr : float               Learning rate for actor and critic.
    gamma : float            Discount factor.
    tau : float              Polyak averaging coefficient.
    policy_delay : int       Actor update frequency (critic updates every step).
    noise_std : float        Exploration noise standard deviation.
    noise_clip : float       Exploration noise clipping.
    batch_size : int         Mini-batch size.
    buffer_capacity : int    Replay buffer size.
    start_steps : int        Steps of random exploration before learning.
    """

    def __init__(self, env: BiddingEnv, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_delay: int = 2, noise_std: float = 0.2,
                 noise_clip: float = 0.5, batch_size: int = 128,
                 buffer_capacity: int = 100_000, start_steps: int = 500):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.start_steps = start_steps

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
                self.actors[a.name].parameters(), lr=lr)

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
                self.critics[a.name].parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
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

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, act, rew, next_obs = self.buffer.sample(self.batch_size)
        # obs: (batch, n_agents, obs_dim), act: (batch, n_agents, act_dim)
        # rew: (batch, n_agents), next_obs: (batch, n_agents, obs_dim)

        n = self.n_agents
        u_dim = self.unique_obs_dim

        # ---- Update each agent's critic ----
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
                    next_act_all.view(self.batch_size, -1))
                target_q = rew[:, i].unsqueeze(1) + self.gamma * \
                    torch.min(target_q1, target_q2)

            current_q1, current_q2 = critic(
                self._build_global_state(obs, i),
                act.view(self.batch_size, -1))
            critic_loss = nn.functional.mse_loss(
                current_q1, target_q) + nn.functional.mse_loss(
                current_q2, target_q)

            opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()

        # ---- Delayed actor update ----
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
                    new_act_all.view(self.batch_size, -1)).mean()

                opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()

        # ---- Polyak update target networks ----
        for nm in self.agent_names:
            for target, source in [
                (self.actor_targets[nm], self.actors[nm]),
                (self.critic_targets[nm], self.critics[nm]),
            ]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data
                                  + (1 - self.tau) * tp.data)

    def train(self, n_episodes: int = 100, verbose: bool = True) \
            -> Dict[str, list]:
        from torch.utils.tensorboard import SummaryWriter
        import datetime, os
        log_dir = os.path.join("runs", datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)

        history = {nm: {"reward": []} for nm in self.agent_names}
        history["welfare"] = []
        history["re_rate"] = []

        for ep in range(n_episodes):
            obs = self.env.reset()
            ep_rewards = {nm: 0.0 for nm in self.agent_names}
            ep_obs = {nm: [] for nm in self.agent_names}
            ep_act = {nm: [] for nm in self.agent_names}
            ep_rew = {nm: [] for nm in self.agent_names}

            for _ in range(N_BLOCKS):
                # Select actions
                actions = {}
                for nm in self.agent_names:
                    add_noise = self.total_steps < self.start_steps
                    if not add_noise:
                        add_noise = True  # always explore during training
                    actions[nm] = self.select_action(
                        obs[nm], nm, add_noise=add_noise)

                next_obs, rewards, done, info = self.env.step(actions)

                # Store in per-agent episode buffers
                for nm in self.agent_names:
                    ep_obs[nm].append(obs[nm])
                    ep_act[nm].append(actions[nm])
                    ep_rew[nm].append(rewards.get(nm, 0.0))
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

                obs = next_obs

            # Update after each episode
            if self.total_steps >= self.start_steps:
                for _ in range(N_BLOCKS):  # multiple updates per episode
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


# ---------------------------------------------------------------------------
# RL bidding strategy integration
# ---------------------------------------------------------------------------

def rl_bidding_strategy(agents: List[Agent], config: MarketConfig,
                        market_history=None, T: int = 96) -> Dict:
    return _rl_bidding_impl(agents, config, _TRAINED_POLICIES,
                            market_history, T)


def _rl_bidding_impl(agents: List[Agent], config: MarketConfig,
                     policies: dict, market_history=None,
                     T: int = 96) -> Dict:
    """Core RL bidding logic — per-block forward pass with Actor networks."""
    if not policies:
        from strategies.random_bidding import RandomStrategy
        return RandomStrategy().formulate(agents, config, market_history, T)

    env_temp = BiddingEnv(agents, config)
    avg_price = 420.0
    if market_history is not None and hasattr(market_history, "get"):
        avg_price = float(np.mean(market_history.get("price", [420.0])))
    action_low, action_high = env_temp.get_action_bounds()
    action_bounds = env_temp.get_action_bounds()

    actions = {}
    for a in agents:
        nm = a.name
        # Consumers control bid_mult only; offer_adder is always zero for them
        actions[nm] = {
            "bid_mult": np.full(T, 1.0),
            "offer_adder": np.full(T, 0.0),
        }

    # Seed opponent features with defaults for block 0
    env_temp._prev_block_actions = {
        nm: {"bid_mult": np.array([1.0]), "offer_adder": np.array([0.0])}
        for nm in actions
    }

    for block_idx in range(N_BLOCKS):
        t_start = block_idx * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, T)
        slr = env_temp._compute_system_load_re_ratio(t_start)

        for a in agents:
            nm = a.name
            if nm not in policies:
                continue
            avg_oth, bid_std = env_temp._compute_opponent_features(nm)
            obs = env_temp._get_agent_obs(
                a, block_idx, None, avg_price, slr, 0.0, avg_oth, bid_std)
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act_arr = policies[nm](obs_t).squeeze(0).numpy()
            bid_m = float(np.clip(act_arr[0], action_low[0], action_high[0]))
            offer_a = float(np.clip(act_arr[1], action_low[1], action_high[1]))
            actions[nm]["bid_mult"][t_start:t_end] = bid_m
            actions[nm]["offer_adder"][t_start:t_end] = offer_a

        # Snapshot this block's actions so the next block sees consistent
        # opponent features from all agents rather than stale partial data.
        env_temp._prev_block_actions = {
            nm: {"bid_mult": act["bid_mult"].copy(),
                 "offer_adder": act.get("offer_adder",
                                        np.zeros(T)).copy()}
            for nm, act in actions.items()
        }

    # Fill random actions for agents without trained policies (all agents, not just prosumers)
    for block_idx in range(N_BLOCKS):
        t_start = block_idx * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, T)
        for a in agents:
            nm = a.name
            if nm in policies:
                continue
            rng = np.random.RandomState(hash(nm + str(block_idx)) % (2**31))
            bid_m = rng.uniform(float(action_low[0]), float(action_high[0]))
            offer_a = rng.uniform(float(action_low[1]), float(action_high[1]))
            if nm in actions:
                actions[nm]["bid_mult"][t_start:t_end] = bid_m
                actions[nm]["offer_adder"][t_start:t_end] = offer_a

    return actions


# Global cache for trained policies
_TRAINED_POLICIES: Dict[str, Actor] = {}


def train_rl_agents(agents: List[Agent], config: MarketConfig,
                    n_episodes: int = 100, verbose: bool = True) \
                    -> Tuple[Dict[str, Actor], dict]:
    """Train MATD3 for all prosumer agents."""
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
