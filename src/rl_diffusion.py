# rl_diffusion.py
"""Diffusion value-distribution multi-agent RL (MAD3PG / Diffusion-MATD).

Replaces the MLP point-estimate critic of MATD3 with a denoising-diffusion
model of the per-agent return distribution, following the MAD3PG idea
(Information Fusion, 2026): the critic predicts the raw return R0 conditioned
on the global (state, action); the TD target uses the minimum over K reverse
samples of the target critic as a conservative future-value estimate.

Design choices locked with the project owner (see docs/plan):
  - one diffusion critic per agent (mirrors the existing per-agent
    CentralizedCritic), conditioned on the full global state and actions;
  - actor gradient passes through a K=1 reverse trajectory of the ONLINE
    critic with re-parameterized (fixed) noise; min-over-K is reserved for the
    critic TD target only;
  - returns are affinely standardized per agent with FROZEN (mu, sigma)
    measured from a short random-policy pre-roll (an affine transform preserves
    distribution shape); the actor-logit deviation penalty uses a fixed
    coefficient divided by the same frozen sigma so its balance vs the
    standardized value matches the unnormalized setting;
  - linear beta schedule (1e-4 -> 0.02), T reverse steps, K min samples, and
    batch size are CLI-configurable; paper-faithful defaults are T=50, K=4.

Only the critic changes: actors, the shared ReplayBuffer and the obs/action
spec are reused from the MATD3 pipeline, so saved policies stay actors and the
evaluation path is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional

from rl_bidding import Actor, ReplayBuffer
from rl_env import N_BLOCKS


class DiffusionScheduler:
    """Linear-beta noise scheduler for 1-D diffusion on standardized returns."""

    def __init__(self, T: int = 50, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        # sigma_q(t) = sqrt(beta_t) is one standard DDPM reverse-noise choice.
        self.sigma_q = torch.sqrt(self.betas)

    def q_sample(self, r0: torch.Tensor, t: torch.Tensor):
        """Forward noising: r_t = sqrt(alpha_bar_t) r0 + sqrt(1-alpha_bar_t) e."""
        noise = torch.randn_like(r0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_1ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        r_t = sqrt_ab * r0 + sqrt_1ab * noise
        return r_t, noise


class DiffusionCritic(nn.Module):
    """Denoising network predicting the raw return R0 given (r_t, t, c).

    c is the conditioning context = [all agents' full obs | all agents'
    actions]. Directly predicts R0 (not the noise). No dropout: denoising
    must stay deterministic so that reverse trajectories used by the actor
    gradient and the K-sample target min are stable.
    """

    def __init__(self, cond_dim: int, hidden: int = 256, T: int = 50):
        super().__init__()
        self.T = T
        t_dim = 64
        self.time_proj = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = 32
        freqs = torch.exp(torch.linspace(
            0.0, np.log(10000.0), half, device=t.device))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.time_proj(
            torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))

    def forward(self, r_t: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        if r_t.dim() == 1:
            r_t = r_t.unsqueeze(-1)
        t_emb = self._time_embedding(t)
        x = torch.cat([r_t, c, t_emb], dim=-1)
        return self.net(x)


class MAD3PG:
    """Multi-agent diffusion value critic with decentralized actors.

    Mirrors the MATD3 trainer interface (actors / actor_targets / actor_opts,
    critics / critic_targets / critic_opts, buffer, total_steps,
    select_action, update, remember) so the training loop is drop-in.
    """

    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_std: float = 0.2, noise_clip: float = 0.5,
                 batch_size: int = 128, buffer_capacity: int = 100_000,
                 start_steps: int = 500, bid_dev_penalty: float = 5.0,
                 offer_dev_penalty: float = 0.5,
                 final_noise_std: float = 0.05,
                 noise_anneal_steps: int = 5000,
                 diff_steps: int = 50, diff_k: int = 4,
                 scale_episodes: int = 4, device: str = "cpu"):
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
        self.T = diff_steps
        self.K = diff_k
        self.device = device

        obs_dim = env.get_state_dim()
        act_dim = 2
        action_bounds = env.get_action_bounds()
        n_agents = len(env.rl_agents)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_names: List[str] = [a.name for a in env.rl_agents]
        self.cond_dim = obs_dim * n_agents + act_dim * n_agents

        self.actors: Dict[str, Actor] = {}
        self.actor_targets: Dict[str, Actor] = {}
        self.actor_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            self.actors[a.name] = Actor(obs_dim, act_dim, action_bounds)
            self.actor_targets[a.name] = Actor(obs_dim, act_dim,
                                               action_bounds)
            self.actor_targets[a.name].load_state_dict(
                self.actors[a.name].state_dict())
            self.actor_opts[a.name] = optim.Adam(
                self.actors[a.name].parameters(), lr=lr, weight_decay=1e-5)

        self.critics: Dict[str, DiffusionCritic] = {}
        self.critic_targets: Dict[str, DiffusionCritic] = {}
        self.critic_opts: Dict[str, optim.Adam] = {}
        for a in env.rl_agents:
            net = DiffusionCritic(self.cond_dim, T=diff_steps).to(device)
            net_t = DiffusionCritic(self.cond_dim, T=diff_steps).to(device)
            net_t.load_state_dict(net.state_dict())
            self.critics[a.name] = net
            self.critic_targets[a.name] = net_t
            self.critic_opts[a.name] = optim.Adam(
                net.parameters(), lr=lr, weight_decay=1e-4)

        self.sched = DiffusionScheduler(T=diff_steps, device=device)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

        # Frozen per-agent affine standardization of the (differential)
        # reward. Calibrated from a short random-policy pre-roll.
        self.mu, self.sigma = self._calibrate_scale(env, scale_episodes)

    # ---- scale calibration -------------------------------------------------

    def _calibrate_scale(self, env, episodes: int = 4):
        """Measure per-agent (mean, std) of the differential reward under
        random actions and freeze them; the affine transform preserves the
        distribution shape and keeps the diffusion targets ~O(1)."""
        lows = env.get_action_bounds()[0].numpy()
        highs = env.get_action_bounds()[1].numpy()
        acc: Dict[str, List[float]] = {nm: [] for nm in self.agent_names}
        for ep in range(max(1, episodes)):
            np.random.seed(100 + ep)
            env.reset()
            for _ in range(N_BLOCKS):
                acts = {a.name: np.random.uniform(lows, highs)
                        for a in env.rl_agents}
                _, rew, done, _ = env.step(acts)
                for nm in self.agent_names:
                    acc[nm].append(float(rew.get(nm, 0.0)))
                if done:
                    break
        mu = np.zeros(len(self.agent_names), dtype=float)
        sd = np.ones(len(self.agent_names), dtype=float)
        for i, nm in enumerate(self.agent_names):
            vals = np.asarray(acc[nm])
            if len(vals) > 1:
                mu[i] = float(np.mean(vals))
                s = float(np.std(vals))
                sd[i] = s if s > 1e-3 else 1.0
        return mu, sd

    def remember(self, obs, act, rew, next_obs, done):
        """Store a transition, standardizing each agent's reward."""
        std_rew = (np.asarray(rew, dtype=float) - self.mu) / self.sigma
        self.buffer.add(obs, act, std_rew, next_obs, done)

    # ---- exploration --------------------------------------------------------

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

    # ---- diffusion inference ------------------------------------------------

    def _denoise_min(self, net: DiffusionCritic, c: torch.Tensor,
                     K: int) -> torch.Tensor:
        """Conservative future-value estimate: min over K reverse samples."""
        B = c.shape[0]
        with torch.no_grad():
            ck = c.repeat_interleave(K, dim=0)
            r = torch.randn(B * K, 1, device=self.device)
            final = None
            for t in reversed(range(self.T)):
                t_idx = torch.full((B * K,), t, dtype=torch.long,
                                   device=self.device)
                pred = net(r, t_idx, ck)
                if t == 0:
                    final = pred
                    break
                z = torch.randn_like(r)
                r = (self.sched.sqrt_alpha_bars[t - 1] * pred
                     + (self.sched.sqrt_one_minus_alpha_bars[t - 1]
                        / self.sched.sqrt_one_minus_alpha_bars[t])
                     * (r - self.sched.sqrt_alpha_bars[t] * pred)
                     + self.sched.sigma_q[t] * z)
            out = final.view(B, K).min(dim=1)[0]
        return out.unsqueeze(-1)

    def _value_grad(self, net: DiffusionCritic, c: torch.Tensor) \
            -> torch.Tensor:
        """Differentiable single-reverse value estimate for the actor update.

        The reverse noises are re-parameterized (fixed, non-trainable) random
        inputs, so the trajectory is a deterministic function of (c, noise);
        gradient flows to c (hence the actor's action) through the denoising
        chain without being polluted by backprop through the noise.
        """
        B = c.shape[0]
        r = torch.randn(B, 1, device=self.device)
        final = None
        for t in reversed(range(self.T)):
            t_idx = torch.full((B,), t, dtype=torch.long, device=self.device)
            pred = net(r, t_idx, c)
            if t == 0:
                final = pred
                break
            z = torch.randn_like(r)
            r = (self.sched.sqrt_alpha_bars[t - 1] * pred
                 + (self.sched.sqrt_one_minus_alpha_bars[t - 1]
                    / self.sched.sqrt_one_minus_alpha_bars[t])
                 * (r - self.sched.sqrt_alpha_bars[t] * pred)
                 + self.sched.sigma_q[t] * z)
        return final

    # ---- update -------------------------------------------------------------

    def update(self) -> dict:
        """One TD3-style update of critics and (delayed) actors."""
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": None, "actor_loss": None,
                    "critic_loss_by_agent": {}, "actor_loss_by_agent": {}}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        B = obs.shape[0]
        n = self.n_agents
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        act_t = torch.as_tensor(act, dtype=torch.float32)
        rew_t = torch.as_tensor(rew, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        done_t = torch.as_tensor(done, dtype=torch.float32)

        state = obs_t.reshape(B, -1)
        next_state = next_obs_t.reshape(B, -1)
        cur_joint = act_t.reshape(B, -1)
        c_now = torch.cat([state, cur_joint], dim=-1)

        # Deterministic next joint actions from the target actors.
        with torch.no_grad():
            nxt = torch.stack(
                [self.actor_targets[nm](next_obs_t[:, j, :])
                 for j, nm in enumerate(self.agent_names)], dim=1)
        next_c = torch.cat([next_state, nxt.reshape(B, -1)], dim=-1)

        critic_losses = []
        critic_loss_by_agent = {}
        for i, nm in enumerate(self.agent_names):
            critic = self.critics[nm]
            tnet = self.critic_targets[nm]
            zmin = self._denoise_min(tnet, next_c, self.K)
            y0 = (rew_t[:, i].unsqueeze(1)
                  + self.gamma * (1.0 - done_t[:, i].unsqueeze(1)) * zmin)
            t_rand = torch.randint(0, self.T, (B,), device=self.device)
            r_t, _ = self.sched.q_sample(y0, t_rand)
            pred = critic(r_t, t_rand, c_now)
            loss = nn.functional.mse_loss(pred, y0)
            opt = self.critic_opts[nm]
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
                logits, new_act = actor.forward_logits(obs_t[:, i, :])
                cur = act_t.clone()
                cur[:, i, :] = new_act
                c_new = torch.cat([state, cur.reshape(B, -1)], dim=-1)
                val = self._value_grad(critic, c_new)
                a_loss = -val.mean()
                # Penalty rescaled by the frozen sigma so the balance vs the
                # standardized value matches the raw-profit formulation.
                sig = float(self.sigma[i]) if self.sigma[i] > 0 else 1.0
                if self.bid_dev_penalty > 0:
                    a_loss = (a_loss
                              + (self.bid_dev_penalty / sig)
                              * (logits[:, 0] ** 2).mean())
                if self.offer_dev_penalty > 0:
                    a_loss = (a_loss
                              + (self.offer_dev_penalty / sig)
                              * (logits[:, 1] ** 2).mean())
                opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
                actor_losses.append(a_loss.detach().item())
                actor_loss_by_agent[nm] = a_loss.detach().item()

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
