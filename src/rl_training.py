# rl_training.py
"""In-training evaluation and policy tracking for RL bidding.

Port of ASSUME's learning-role orchestration (periodic evaluation episodes,
best/last policy persistence, early stopping) onto the MakerB training loop.

Evaluation uses deterministic episodes (no exploration noise) so the reported
metric reflects the learned policy, and defaults to raw profit (differential
reward off) to halve the per-block clearing cost.
"""

import os
import copy

import numpy as np

from eval_agents import actor_predict
from rl_env import BiddingEnv, N_BLOCKS
from rl_td3 import save_policy
from scenarios import get_scenario


def deterministic_fleet_episode(env: BiddingEnv, actors: dict,
                                rl_agent_names: list) -> dict:
    """Run one deterministic episode (no exploration noise) for a policy fleet.

    actors maps agent name to a torch Actor (or any obs->action callable).
    Agents without an actor fall back to truthful-bidding defaults.
    """
    obs = env.reset()
    total_welfare = 0.0
    re_rate = 0.0
    profits = {nm: 0.0 for nm in rl_agent_names}
    for _ in range(N_BLOCKS):
        acts = {}
        for nm in rl_agent_names:
            actor = actors.get(nm)
            acts[nm] = actor_predict(actor, obs[nm]) if actor is not None \
                else np.array([1.0, 0.0], dtype=np.float32)
        next_obs, rewards, done, info = env.step(acts)
        for nm in rl_agent_names:
            profits[nm] += rewards.get(nm, 0.0)
        total_welfare += info.get("welfare", 0.0)
        re_rate = info.get("re_rate", 0.0)
        if done:
            break
        obs = next_obs
    mean_reward = sum(profits.values()) / max(len(profits), 1)
    return {"mean_reward": mean_reward, "welfare": total_welfare,
            "re_rate": re_rate, "profits": profits}


class PolicyTracker:
    """Evaluates policies during training and manages best/last checkpoints.

    Mirrors ASSUME's compare_and_save_policies: an improvement in the tracked
    metric saves the current actors to best/, and a sustained flat or
    declining window of `early_stopping_steps` evaluations triggers early
    stopping. Early stopping is opt-in (early_stopping_steps=0 disables it),
    so a default run keeps training for the full episode count.
    """

    def __init__(self, save_dir: str, agent_names: list,
                 eval_scenario: str = "baseline", metric: str = "mean_reward",
                 eval_episodes: int = 1, early_stopping_steps: int = 0,
                 early_stopping_threshold: float = 0.05,
                 use_differential_reward: bool = False,
                 obs_spec=None, action_spec=None):
        self.save_dir = save_dir
        self.agent_names = agent_names
        self.eval_scenario = eval_scenario
        self.metric = metric
        self.eval_episodes = eval_episodes
        self.early_stopping_steps = early_stopping_steps
        self.early_stopping_threshold = early_stopping_threshold
        self.use_differential_reward = use_differential_reward
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.best_dir = os.path.join(save_dir, "best")
        self.last_dir = os.path.join(save_dir, "last")
        self.eval_history = {metric: []}
        self.max_eval = {metric: float("-inf")}
        self.best_episode = None
        self.early_stopped = False

    def build_eval_env(self, config):
        """Build a fresh evaluation environment on the eval scenario."""
        agents, _ = get_scenario(
            self.eval_scenario, T=96, config=copy.deepcopy(config))
        return BiddingEnv(
            agents, config, rl_agent_names=self.agent_names,
            use_differential_reward=self.use_differential_reward,
            obs_spec=self.obs_spec, action_spec=self.action_spec)

    def evaluate(self, env: BiddingEnv, actors: dict) -> dict:
        """Average deterministic_fleet_episode over eval_episodes."""
        n = max(1, self.eval_episodes)
        agg = {"mean_reward": 0.0, "welfare": 0.0, "re_rate": 0.0}
        for _ in range(n):
            r = deterministic_fleet_episode(env, actors, self.agent_names)
            agg["mean_reward"] += r["mean_reward"]
            agg["welfare"] += r["welfare"]
            agg["re_rate"] += r["re_rate"]
        return {k: v / n for k, v in agg.items()}

    def compare_and_save_policies(self, episode: int, actors: dict,
                                  metrics: dict) -> bool:
        """Port of ASSUME compare_and_save_policies.

        Saves best policies when the tracked metric improves, saves last
        policies on early stop, and returns True when early stopping should
        trigger.
        """
        value = metrics[self.metric]
        self.eval_history[self.metric].append(value)
        if value > self.max_eval[self.metric]:
            self.max_eval[self.metric] = value
            self.best_episode = episode
            self._save_best(actors)
            print(f"New best {self.metric}={value:.1f} at episode {episode}",
                  flush=True)

        if self.early_stopping_steps <= 0 or \
                len(self.eval_history[self.metric]) < self.early_stopping_steps:
            return False
        window = self.eval_history[self.metric][-self.early_stopping_steps:]
        denom = max(abs(min(window)), 1e-8)
        improvement = (window[-1] - window[0]) / denom
        if improvement < self.early_stopping_threshold:
            self._save_last(actors)
            self.early_stopped = True
            print(f"Early stopping: no improvement > "
                  f"{self.early_stopping_threshold:.0%} in last "
                  f"{self.early_stopping_steps} evals for {self.metric}",
                  flush=True)
            return True
        return False

    def save_last(self, actors: dict):
        """Persist the current actors as the final/last policies."""
        self._save_last(actors)

    def _save_best(self, actors: dict):
        os.makedirs(self.best_dir, exist_ok=True)
        for nm in self.agent_names:
            actor = actors.get(nm)
            if actor is None:
                continue
            save_policy(actor, os.path.join(self.best_dir, f"{nm}.pt"),
                        obs_spec=self.obs_spec, action_spec=self.action_spec)
        print(f"  -> best policies: {self.best_dir}", flush=True)

    def _save_last(self, actors: dict):
        os.makedirs(self.last_dir, exist_ok=True)
        for nm in self.agent_names:
            actor = actors.get(nm)
            if actor is None:
                continue
            save_policy(actor, os.path.join(self.last_dir, f"{nm}.pt"),
                        obs_spec=self.obs_spec, action_spec=self.action_spec)
