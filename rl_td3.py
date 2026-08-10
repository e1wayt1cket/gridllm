# rl_td3.py
"""Single-agent TD3 (Twin Delayed DDPG) for agent bidding.

Each agent is trained independently. The environment designates one agent
as the RL learner; all others use fixed default strategy (bid_mult=1.0,
offer_adder=0.0). This is the Independent Learner paradigm — each agent
optimizes its own profit without centralized coordination.

Architecture:
  - Actor: obs → 256 → 128 → 2 (tanh-squashed to action bounds)
  - Twin Critic: (obs + action) → Q-value  (no multi-agent input)
  - Standard TD3: clipped double Q, delayed actor, target policy smoothing
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, List
from collections import deque
import random

from models import Agent, MarketConfig
from rl_env import BiddingEnv, N_BLOCKS, BID_MULT_LOW, BID_MULT_HIGH, \
    OFFER_ADDER_LOW, OFFER_ADDER_HIGH


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
# Twin Critic — single-agent, sees (obs, action) only
# ---------------------------------------------------------------------------

class TwinCritic(nn.Module):
    """Twin Q-networks: (obs + action) → Q-value.

    Unlike the MATD3 CentralizedCritic, this takes only the single
    agent's observation and action — no global state or other-agent
    observations are concatenated.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + act_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, action], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) \
            -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


# ---------------------------------------------------------------------------
# Replay Buffer — single-agent, flat tensors
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size replay buffer for single-agent TD3.

    Each transition: (obs, action, reward, next_obs, done).
    Unlike the MATD3 buffer, tensors are flat (no multi-agent stacking).
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool):
        self.buffer.append((
            obs.copy(), action.copy(), float(reward),
            next_obs.copy(), done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, rew, next_obs, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(np.array(act), dtype=torch.float32),
            torch.as_tensor(np.array(rew), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(next_obs), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.bool),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# TD3 Trainer — standard single-agent TD3
# ---------------------------------------------------------------------------

class TD3:
    """Single-agent TD3 with twin critics and delayed actor updates.

    Parameters
    ----------
    obs_dim : int
    act_dim : int
    action_bounds : torch.Tensor  shape (2, act_dim): [low, high]
    lr : float                    Learning rate.
    gamma : float                 Discount factor.
    tau : float                   Polyak averaging coefficient.
    policy_delay : int            Actor update frequency.
    noise_std : float             Exploration noise std (during action selection).
    target_noise : float          Target policy smoothing noise std.
    noise_clip : float            Target noise clipping.
    batch_size : int
    buffer_capacity : int
    start_steps : int             Random exploration steps before learning.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 action_bounds: torch.Tensor,
                 lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_std: float = 0.2, target_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 batch_size: int = 128, buffer_capacity: int = 100_000,
                 start_steps: int = 500,
                 bid_dev_penalty: float = 0.0,
                 offer_dev_penalty: float = 0.0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.bid_dev_penalty = bid_dev_penalty
        self.offer_dev_penalty = offer_dev_penalty

        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

        # Networks
        self.actor = Actor(obs_dim, act_dim, action_bounds)
        self.actor_target = Actor(obs_dim, act_dim, action_bounds)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr,
                                    weight_decay=1e-5)

        self.critic = TwinCritic(obs_dim, act_dim)
        self.critic_target = TwinCritic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr,
                                     weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    def select_action(self, obs: np.ndarray,
                      add_noise: bool = False) -> np.ndarray:
        """Select action given observation.

        Parameters
        ----------
        obs : np.ndarray  shape (obs_dim,)
        add_noise : bool   If True, add exploration noise.

        Returns
        -------
        np.ndarray  shape (act_dim,)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0)
            if add_noise:
                noise = torch.randn_like(action) * self.noise_std
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise).clamp(self.action_low,
                                                self.action_high)
            return action.numpy()

    def update(self) -> dict:
        """Execute one TD3 update step.

        Returns
        -------
        dict with keys "critic_loss" (float or None) and "actor_loss"
        (float or None).
        """
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": None, "actor_loss": None}

        obs, act, rew, next_obs, dones = self.buffer.sample(self.batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            # Target actions with smoothing noise
            next_action = self.actor_target(next_obs)
            noise = (torch.randn_like(next_action) * self.target_noise) \
                .clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(self.action_low,
                                                      self.action_high)

            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            # Terminal states have zero future value
            target_q[dones] = 0.0
            target_q = rew + self.gamma * target_q

        current_q1, current_q2 = self.critic(obs, act)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) \
            + nn.functional.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---- Delayed actor update ----
        actor_loss = None
        if self.total_steps % self.policy_delay == 0:
            actor_action = self.actor(obs)
            actor_loss = -self.critic.q1_forward(obs, actor_action).mean()
            # Deviation penalty: apply directly to the actor objective so the
            # gradient toward moderate bids does not depend on the critic
            # learning the penalty through the reward signal.
            if self.bid_dev_penalty > 0:
                bid = actor_action[:, 0]
                actor_loss = actor_loss \
                    + self.bid_dev_penalty * torch.abs(bid - 1.0).mean()
            if self.offer_dev_penalty > 0:
                offer = actor_action[:, 1]
                actor_loss = actor_loss \
                    + self.offer_dev_penalty * offer.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # ---- Polyak update target networks ----
            for target, source in [
                (self.actor_target, self.actor),
                (self.critic_target, self.critic),
            ]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data
                                  + (1 - self.tau) * tp.data)

        return {"critic_loss": float(critic_loss.detach().item()),
                "actor_loss": float(actor_loss.detach().item())
                if actor_loss is not None else None}

    def train(self, env: BiddingEnv, n_episodes: int = 200,
              scenario_names: Optional[List[str]] = None,
              verbose: bool = True) -> Dict[str, list]:
        """Train TD3 on the given environment.

        Parameters
        ----------
        env : BiddingEnv
            Training environment. Must have exactly one RL agent.
        n_episodes : int
            Number of episodes.
        scenario_names : list of str or None
            Scenario names to cycle through. If provided, a scenario is
            randomly sampled each episode and env.set_agents() is called.
            If None, uses the agents already in the env.
        verbose : bool

        Returns
        -------
        dict with keys "reward", "welfare", "re_rate", "scenario".
        """
        import copy as _copy
        from scenarios import get_scenario

        history = {"reward": [], "welfare": [], "re_rate": [],
                   "scenario": []}
        agent_name = env.rl_agents[0].name

        for ep in range(n_episodes):
            # ---- Scenario switching ----
            sc_name = None
            if scenario_names:
                sc_name = random.choice(scenario_names)
                config_copy = _copy.deepcopy(env.config)
                agents_sc, wholesale = get_scenario(sc_name, T=env.T,
                                                    config=config_copy)
                env.set_agents(agents_sc, wholesale)
                env.config = config_copy
            obs = env.reset()

            ep_reward = 0.0
            ep_welfare = 0.0
            ep_re_rate = 0.0

            for _ in range(N_BLOCKS):
                add_noise = True  # always explore during training
                if self.total_steps < self.start_steps:
                    # Pure random during initial exploration
                    action = np.random.uniform(
                        self.action_low.numpy(), self.action_high.numpy())
                else:
                    action = self.select_action(obs[agent_name],
                                                add_noise=add_noise)

                next_obs, rewards, done, info = env.step(
                    {agent_name: action})
                ep_reward += rewards.get(agent_name, 0.0)
                ep_welfare = info.get("welfare", 0.0)
                ep_re_rate = info.get("re_rate", 0.0)

                self.buffer.add(
                    obs[agent_name], action,
                    rewards.get(agent_name, 0.0),
                    next_obs.get(agent_name, np.zeros(self.obs_dim)),
                    done)
                self.total_steps += 1
                obs = next_obs

            # ---- Post-episode updates ----
            ep_c_losses = []
            ep_a_losses = []
            if self.total_steps >= self.start_steps:
                for _ in range(N_BLOCKS):
                    loss_info = self.update()
                    if loss_info["critic_loss"] is not None:
                        ep_c_losses.append(loss_info["critic_loss"])
                    if loss_info["actor_loss"] is not None:
                        ep_a_losses.append(loss_info["actor_loss"])

            history["reward"].append(ep_reward)
            history["welfare"].append(ep_welfare)
            history["re_rate"].append(ep_re_rate)
            history["scenario"].append(sc_name or "fixed")

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                c_str = f"critic={np.mean(ep_c_losses):.4f}" \
                    if ep_c_losses else "critic=N/A"
                a_str = f"actor={np.mean(ep_a_losses):.4f}" \
                    if ep_a_losses else "actor=N/A"
                print(f"  Ep {ep+1}/{n_episodes}: reward={ep_reward:+.1f} | "
                      f"welfare={ep_welfare:.0f} | RE={ep_re_rate:.1f}% | "
                      f"{c_str} {a_str} | sc={sc_name or 'fixed'}",
                      flush=True)

        return history


# ---------------------------------------------------------------------------
# Policy persistence
# ---------------------------------------------------------------------------

def save_policy(actor: Actor, path: str):
    """Save a single Actor network to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(actor.state_dict(), path)


def load_policy(path: str, obs_dim: int,
                action_bounds: torch.Tensor) -> Actor:
    """Load a single Actor network from disk."""
    net = Actor(obs_dim, 2, action_bounds)
    state = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(state)
    return net
