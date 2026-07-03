# rl_bidding.py
"""Independent PPO for agent bidding in electricity markets.

Each prosumer agent trains its own policy network with discrete actions
(24 = 6 bid_mult x 4 offer_adder combos) on 4-period decision blocks.

Uses a lightweight numpy-based PPO with small MLP networks suitable for
the modest state/action dimensions of this problem.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from collections import deque

from models import Agent, MarketConfig
from rl_env import BiddingEnv, action_to_params, BLOCK_SIZE, N_BLOCKS, BID_MULTS, OFFER_ADDERS


# ---------------------------------------------------------------------------
# Neural network helpers (numpy-based)
# ---------------------------------------------------------------------------

def _init_weights(fan_in: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """Xavier initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    w = np.random.uniform(-limit, limit, (fan_in, fan_out))
    b = np.zeros(fan_out)
    return w, b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class MLPActorCritic:
    """Small MLP with separate actor (policy) and critic (value) heads.

    Architecture:
      shared:  obs_dim → 64 (ReLU) → 32 (ReLU)
      actor:   32 → n_actions → softmax
      critic:  32 → 1
    """

    def __init__(self, obs_dim: int, n_actions: int, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Layer 1
        self.w1, self.b1 = _init_weights(obs_dim, 64)
        self.w2, self.b2 = _init_weights(64, 32)
        # Actor head
        self.wa, self.ba = _init_weights(32, n_actions)
        # Critic head
        self.wc, self.bc = _init_weights(32, 1)

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (action_probs, state_value)."""
        h = _relu(np.dot(obs, self.w1) + self.b1)
        h = _relu(np.dot(h, self.w2) + self.b2)
        logits = np.dot(h, self.wa) + self.ba
        probs = _softmax(logits)
        value = float(np.dot(h, self.wc) + self.bc)
        return probs, value

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """Sample action from policy. Returns (action_idx, action_probs)."""
        probs, _ = self.forward(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))
        return action, probs

    def get_params(self) -> List[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2, self.wa, self.ba, self.wc, self.bc]

    def set_params(self, params: List[np.ndarray]):
        self.w1, self.b1, self.w2, self.b2, self.wa, self.ba, self.wc, self.bc = params


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores trajectories for PPO update."""

    def __init__(self, max_size: int = 2048):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.max_size = max_size

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obs)


# ---------------------------------------------------------------------------
# Independent PPO Trainer
# ---------------------------------------------------------------------------

class IndependentPPO:
    """Independent PPO: each agent trains its own policy.

    Parameters
    ----------
    env : BiddingEnv
    lr_actor : float
    lr_critic : float
    gamma : float
        Discount factor.
    lam : float
        GAE lambda.
    clip_epsilon : float
        PPO clipping range.
    entropy_coef : float
        Entropy bonus weight.
    epochs : int
        PPO update epochs per rollout.
    batch_size : int
        Mini-batch size for updates.
    """

    def __init__(self, env: BiddingEnv, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, gamma: float = 0.99,
                 lam: float = 0.95, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01, epochs: int = 10,
                 batch_size: int = 64):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

        obs_dim = env.get_state_dim()
        n_actions = env.get_action_dim()

        # One policy per RL agent
        self.policies: Dict[str, MLPActorCritic] = {}
        for a in env.rl_agents:
            self.policies[a.name] = MLPActorCritic(obs_dim, n_actions,
                                                    lr_actor, lr_critic)
        # Buffers per agent
        self.buffers: Dict[str, RolloutBuffer] = {
            nm: RolloutBuffer() for nm in self.policies}

    def train(self, n_episodes: int = 100, verbose: bool = True,
              early_stop_welfare: Optional[float] = None) -> Dict[str, list]:
        """Train all agents with Independent PPO.

        Returns dict of training history per agent.
        """
        history = {nm: {"reward": []} for nm in self.policies}
        history["welfare"] = []
        history["re_rate"] = []

        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            ep_rewards = {nm: 0.0 for nm in self.policies}

            for step in range(N_BLOCKS):
                actions = {}
                for nm in self.policies:
                    a_idx, probs = self.policies[nm].act(obs[nm])
                    log_prob = np.log(probs[a_idx] + 1e-8)
                    _, value = self.policies[nm].forward(obs[nm])
                    actions[nm] = a_idx
                    self.buffers[nm].add(obs[nm], a_idx, log_prob, 0.0,
                                         value, False)

                next_obs, rewards, done, info = self.env.step(actions)

                # Store rewards (delayed)
                for nm in self.policies:
                    self.buffers[nm].rewards[-1] = rewards.get(nm, 0.0)
                    ep_rewards[nm] += rewards.get(nm, 0.0)
                obs = next_obs

            # Episode done — compute returns and update
            for nm in self.policies:
                buf = self.buffers[nm]
                if len(buf) == 0:
                    continue
                # Bootstrapped last value = 0 (episode end)
                returns = self._compute_gae(buf, 0.0)
                self._ppo_update(self.policies[nm], buf, returns)
                buf.clear()
                history[nm]["reward"].append(ep_rewards[nm])

            history["welfare"].append(info.get("welfare", 0.0))
            history["re_rate"].append(info.get("re_rate", 0.0))

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg_r = np.mean([ep_rewards[nm] for nm in self.policies])
                print(f"  Ep {ep+1}/{n_episodes}: avg_reward={avg_r:.1f}, "
                      f"welfare={info.get('welfare', 0):.0f}, "
                      f"RE={info.get('re_rate', 0):.1f}%")

        return history

    def _compute_gae(self, buf: RolloutBuffer, last_value: float) -> np.ndarray:
        """Compute GAE returns."""
        n = len(buf)
        returns = np.zeros(n)
        adv = 0.0
        for t in reversed(range(n)):
            delta = buf.rewards[t] + self.gamma * (last_value if t == n - 1
                        else buf.values[t + 1]) - buf.values[t]
            adv = delta + self.gamma * self.lam * adv
            returns[t] = adv + buf.values[t]
            last_value = buf.values[t]
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _ppo_update(self, policy: MLPActorCritic, buf: RolloutBuffer,
                    returns: np.ndarray):
        """Single PPO update epoch."""
        n = len(buf)
        if n < 2:
            return
        obs_arr = np.array(buf.obs)
        act_arr = np.array(buf.actions)
        old_logp = np.array(buf.log_probs)
        old_vals = np.array(buf.values)
        advantages = returns - old_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(n)
        batch_size = min(self.batch_size, n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                if len(batch_idx) == 0:
                    continue
                self._update_batch(policy, obs_arr[batch_idx],
                                   act_arr[batch_idx],
                                   old_logp[batch_idx],
                                   advantages[batch_idx],
                                   returns[batch_idx])

    def _update_batch(self, policy: MLPActorCritic, obs_batch: np.ndarray,
                      act_batch: np.ndarray, old_logp_batch: np.ndarray,
                      adv_batch: np.ndarray, ret_batch: np.ndarray):
        """Single mini-batch update via manual backprop (2-layer MLP)."""
        # Forward
        h1 = _relu(np.dot(obs_batch, policy.w1) + policy.b1)
        h2 = _relu(np.dot(h1, policy.w2) + policy.b2)
        logits = np.dot(h2, policy.wa) + policy.ba
        probs = _softmax(logits)
        values = np.dot(h2, policy.wc) + policy.bc

        # Actor loss (PPO clipped)
        new_logp = np.log(probs[np.arange(len(act_batch)), act_batch] + 1e-8)
        ratio = np.exp(new_logp - old_logp_batch)
        surr1 = ratio * adv_batch
        surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
        actor_loss = -np.mean(np.minimum(surr1, surr2))
        # Entropy bonus
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1))
        # Critic loss (MSE)
        critic_loss = np.mean((values.flatten() - ret_batch) ** 2)
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # Manual gradient computation (simplified: gradient clipping + SGD)
        # Policy gradient on last layer
        d_logits = probs.copy()
        for i, a in enumerate(act_batch):
            d_logits[i, a] -= 1.0
        d_logits /= len(act_batch)
        d_logits *= (ratio * adv_batch).reshape(-1, 1)

        d_h2_a = np.dot(d_logits, policy.wa.T)
        d_wa = np.dot(h2.T, d_logits)
        d_ba = np.sum(d_logits, axis=0)

        # Value gradient
        d_v = 2.0 * (values.flatten() - ret_batch) / len(ret_batch)
        d_h2_c = np.dot(d_v.reshape(-1, 1), policy.wc.T)
        d_wc = np.dot(h2.T, d_v.reshape(-1, 1))
        d_bc = np.sum(d_v)

        # Merge gradients at h2
        d_h2 = d_h2_a + 0.5 * d_h2_c
        d_h2[h2 <= 0] = 0  # ReLU backward

        # Layer 2
        d_w2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0)
        d_h1 = np.dot(d_h2, policy.w2.T)
        d_h1[h1 <= 0] = 0  # ReLU backward

        # Layer 1
        d_w1 = np.dot(obs_batch.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0)

        # SGD update with gradient clipping
        lr_a = policy.lr_actor
        lr_c = policy.lr_critic

        grad_list = [d_w1, d_b1, d_w2, d_b2, d_wa, d_ba, d_wc, d_bc]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grad_list) + 1e-8)
        clip = 0.5
        if total_norm > clip:
            scale = clip / total_norm
            grad_list = [g * scale for g in grad_list]

        # Apply updates (actor lr for all shared params, critic lr for value head)
        policy.w1 -= lr_a * grad_list[0]
        policy.b1 -= lr_a * grad_list[1]
        policy.w2 -= lr_a * grad_list[2]
        policy.b2 -= lr_a * grad_list[3]
        policy.wa -= lr_a * grad_list[4]
        policy.ba -= lr_a * grad_list[5]
        policy.wc -= lr_c * grad_list[6]
        policy.bc -= lr_c * grad_list[7]


# ---------------------------------------------------------------------------
# RL bidding strategy (integrated with market.py)
# ---------------------------------------------------------------------------

def rl_bidding_strategy(agents: List[Agent], config: MarketConfig,
                        market_history=None, T: int = 96) -> Dict:
    """Bidding strategy entry point for market.py.

    Uses trained PPO policies if available, otherwise falls back to heuristic.
    Called by adaptive_bidding when strategy="rl".
    """
    from market import _bootstrap_actions

    base = _bootstrap_actions(agents, config, T)
    return base


# Global cache for trained policies
_TRAINED_POLICIES: Dict[str, MLPActorCritic] = {}


def train_rl_agents(agents: List[Agent], config: MarketConfig,
                    n_episodes: int = 100, verbose: bool = True) \
                    -> Tuple[Dict[str, MLPActorCritic], dict]:
    """Train Independent PPO for all prosumer agents.

    Returns trained policies and training history.
    """
    env = BiddingEnv(agents, config)
    ppo = IndependentPPO(env)
    history = ppo.train(n_episodes, verbose=verbose)

    # Save trained policies to global cache
    global _TRAINED_POLICIES
    _TRAINED_POLICIES.update(ppo.policies)

    return ppo.policies, history
