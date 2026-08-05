# compare_per.py
"""A/B experiment: uniform replay vs Prioritized Experience Replay.

Runs both MATD3 variants from identical initial conditions (same seed,
same hyperparameters) and records per-episode reward, welfare and loss
to decide whether PER earns its place in the main pipeline.

Usage:
    python compare_per.py                      # default: 150 episodes
    python compare_per.py --episodes 200       # override
    python compare_per.py --out outputs/per_ab.png
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from scenarios import get_scenario
from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS

import rl_bidding        # uniform baseline (Softsign + L2)
import rl_bidding_per    # PER variant


def run_training(matd3_cls, env, n_episodes: int, seed: int, tag: str,
                 log_every: int = 10) -> dict:
    """Train one MATD3 variant; record per-episode stats."""
    # Same initial state for both runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    matd3 = matd3_cls(env)
    hist = {"reward": [], "welfare": [], "re_rate": [],
            "critic_loss": [], "actor_loss": []}

    t0 = time.time()
    for ep in range(n_episodes):
        obs = env.reset()
        ep_rewards = {nm: 0.0 for nm in matd3.agent_names}

        for _ in range(N_BLOCKS):
            actions = {}
            for nm in matd3.agent_names:
                actions[nm] = matd3.select_action(
                    obs[nm], nm, add_noise=True)

            next_obs, rewards, done, info = env.step(actions)
            for nm in matd3.agent_names:
                ep_rewards[nm] += rewards.get(nm, 0.0)

            obs_arr = np.stack([obs[nm] for nm in matd3.agent_names])
            act_arr = np.stack([actions[nm] for nm in matd3.agent_names])
            rew_arr = np.array([rewards.get(nm, 0.0)
                               for nm in matd3.agent_names])
            next_obs_arr = np.stack(
                [next_obs.get(nm, np.zeros(matd3.obs_dim))
                 for nm in matd3.agent_names])
            matd3.buffer.add(obs_arr, act_arr, rew_arr, next_obs_arr)
            matd3.total_steps += 1
            obs = next_obs

        c_losses, a_losses = [], []
        if matd3.total_steps >= matd3.start_steps:
            for _ in range(N_BLOCKS):
                li = matd3.update()
                if li["critic_loss"] is not None:
                    c_losses.append(li["critic_loss"])
                if li["actor_loss"] is not None:
                    a_losses.append(li["actor_loss"])

        avg_r = np.mean([ep_rewards[nm] for nm in matd3.agent_names])
        hist["reward"].append(avg_r)
        hist["welfare"].append(info.get("welfare", 0.0))
        hist["re_rate"].append(info.get("re_rate", 0.0))
        hist["critic_loss"].append(float(np.mean(c_losses)) if c_losses
                                   else np.nan)
        hist["actor_loss"].append(float(np.mean(a_losses)) if a_losses
                                  else np.nan)

        if (ep + 1) % log_every == 0:
            print(f"  [{tag}] Ep {ep+1}/{n_episodes}: "
                  f"avg_reward={avg_r:+.3f} welfare={info.get('welfare', 0):.0f} "
                  f"critic={hist['critic_loss'][-1]:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    hist["episodes"] = n_episodes
    hist["seed"] = seed
    return hist


def summarize(name: str, hist: dict, tail: int = 20) -> dict:
    """Final-window mean reward/welfare and convergence episode."""
    r = np.array(hist["reward"])
    w = np.array(hist["welfare"])
    tail_r = float(np.mean(r[-tail:]))
    tail_w = float(np.mean(w[-tail:]))
    # Convergence: first episode whose 10-ep running mean reaches
    # 90% of the final running mean (reward, smoothed).
    window = min(10, len(r))
    running = np.convolve(r, np.ones(window) / window, mode="valid")
    target = 0.9 * running[-1]
    above = np.where(running >= target)[0]
    conv_ep = int(above[0] + window) if len(above) else len(r)
    print(f"\n=== {name} (seed={hist['seed']}) ===")
    print(f"  final {tail}-ep mean reward: {tail_r:+.3f}")
    print(f"  final {tail}-ep mean welfare: {tail_w:.0f}")
    print(f"  convergence episode: {conv_ep}/{hist['episodes']}")
    return {"name": name, "reward": tail_r, "welfare": tail_w,
            "conv_ep": conv_ep}


def plot_comparison(hist_a: dict, hist_b: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(hist_a["reward"]), len(hist_b["reward"]))
    eps = np.arange(1, n + 1)
    window = min(10, n)

    def smooth(x):
        if len(x) < window * 2:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(eps, smooth(hist_a["reward"][:n]), label="uniform", lw=1.5)
    axes[0].plot(eps, smooth(hist_b["reward"][:n]), label="PER", lw=1.5)
    axes[0].set_ylabel("avg reward (smoothed)")
    axes[0].legend()
    axes[0].set_title("A/B: uniform vs Prioritized Experience Replay")

    axes[1].plot(eps, smooth(hist_a["welfare"][:n]), label="uniform", lw=1.5)
    axes[1].plot(eps, smooth(hist_b["welfare"][:n]), label="PER", lw=1.5)
    axes[1].set_ylabel("welfare (smoothed)")
    axes[1].set_xlabel("episode")
    axes[1].legend()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PER vs uniform A/B test")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/per_ab.png")
    args = parser.parse_args()

    print(f"Building environment (seed={args.seed})...", flush=True)
    config = MarketConfig(opf_mode="socp", verbose=False)
    agents, _ = get_scenario("baseline", T=96, config=config)
    env = BiddingEnv(agents, config)
    print(f"RL agents: {len(env.rl_agents)}", flush=True)

    print(f"\n--- Run 1/2: uniform replay ({args.episodes} eps) ---")
    hist_uniform = run_training(rl_bidding.MATD3, env, args.episodes,
                                args.seed, "uniform")
    s_uniform = summarize("uniform", hist_uniform)

    print(f"\n--- Run 2/2: PER ({args.episodes} eps) ---")
    hist_per = run_training(rl_bidding_per.MATD3, env, args.episodes,
                            args.seed, "PER")
    s_per = summarize("PER", hist_per)

    # Verdict on the data
    print("\n=== verdict ===")
    dr = s_per["reward"] - s_uniform["reward"]
    dw = s_per["welfare"] - s_uniform["welfare"]
    print(f"  PER - uniform final reward: {dr:+.3f} "
          f"({100*dr/abs(s_uniform['reward']):+.1f}% if nonzero)")
    print(f"  PER - uniform final welfare: {dw:+.1f}")
    better = "PER" if dw > 0 else "uniform"
    verdict = ("PER improves welfare; consider merging" if dw > 0
               else "no evidence PER helps; keep uniform")
    print(f"  -> {verdict} ({better})")

    plot_comparison(hist_uniform, hist_per, args.out)


if __name__ == "__main__":
    main()
