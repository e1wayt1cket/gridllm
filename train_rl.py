# train_rl.py
"""Standalone RL training script with checkpointing, TensorBoard logging
and reproducible seeds."""
import os, time, argparse, random, datetime
import numpy as np
import torch

from scenarios import get_scenario
from models import MarketConfig
from rl_env import BiddingEnv
from rl_bidding import MATD3, save_policies

N_EPISODES = 200
SAVE_PATH = "policies/default.pt"

def main():
    parser = argparse.ArgumentParser(description="Train MATD3 bidding agents")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    # Seed all RNGs before scenario/profile generation for reproducible runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Building environment (seed={args.seed})...", flush=True)
    config = MarketConfig(opf_mode="socp", verbose=False)
    agents, _ = get_scenario("baseline", T=96, config=config)

    env = BiddingEnv(agents, config)
    matd3 = MATD3(env)
    print(f"RL agents: {len(env.rl_agents)}", flush=True)
    for a in env.rl_agents:
        print(f"  {a.name} bus={a.bus} type={a.load_type}", flush=True)

    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join("runs", "train-" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    for ep in range(N_EPISODES):
        obs = env.reset()
        ep_rewards = {nm: 0.0 for nm in matd3.agent_names}

        for _ in range(24):  # N_BLOCKS
            actions = {}
            for nm in matd3.agent_names:
                add_noise = True
                actions[nm] = matd3.select_action(obs[nm], nm, add_noise=add_noise)
            next_obs, rewards, done, info = env.step(actions)
            for nm in matd3.agent_names:
                ep_rewards[nm] += rewards.get(nm, 0.0)

            obs_arr = np.stack([obs[nm] for nm in matd3.agent_names])
            act_arr = np.stack([actions[nm] for nm in matd3.agent_names])
            rew_arr = np.array([rewards.get(nm, 0.0) for nm in matd3.agent_names])
            next_obs_arr = np.stack([next_obs.get(nm, np.zeros(matd3.obs_dim))
                                     for nm in matd3.agent_names])
            matd3.buffer.add(obs_arr, act_arr, rew_arr, next_obs_arr)
            matd3.total_steps += 1
            obs = next_obs

        if matd3.total_steps >= matd3.start_steps:
            for _ in range(24):
                matd3.update()

        # Collect loss stats from the most recent update
        loss_info = matd3.update()  # one extra step to get latest loss values
        c_loss = loss_info.get("critic_loss")
        a_loss = loss_info.get("actor_loss")

        avg_r = np.mean([ep_rewards[nm] for nm in matd3.agent_names])
        elapsed = time.time() - t0
        eta = (elapsed / (ep + 1)) * (N_EPISODES - ep - 1) if ep > 0 else 0

        if (ep + 1) % 10 == 0 or ep == 0:
            loss_str = ""
            if c_loss is not None:
                loss_str += f"critic={c_loss:.4f} "
            if a_loss is not None:
                loss_str += f"actor={a_loss:.4f}"
            print(f"Ep {ep+1}/{N_EPISODES} | "
                  f"avg_reward={avg_r:+.3f} | "
                  f"welfare={info.get('welfare', 0):.0f} | "
                  f"RE={info.get('re_rate', 0):.1f}% | "
                  f"{loss_str} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

        # TensorBoard logging (every episode)
        writer.add_scalar("Reward/mean", avg_r, ep)
        writer.add_scalar("Welfare", info.get("welfare", 0), ep)
        writer.add_scalar("RE_Rate", info.get("re_rate", 0), ep)
        if c_loss is not None:
            writer.add_scalar("Loss/critic", c_loss, ep)
        if a_loss is not None:
            writer.add_scalar("Loss/actor", a_loss, ep)

        # Checkpoint every 50 episodes
        if (ep + 1) % 50 == 0:
            ckpt_path = f"policies/ckpt_{ep+1}.pt"
            save_policies(matd3.actors, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}", flush=True)

    writer.close()

    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    save_policies(matd3.actors, SAVE_PATH)
    elapsed = time.time() - t0
    print(f"Training complete: {N_EPISODES} episodes in {elapsed:.0f}s "
          f"({elapsed/N_EPISODES:.1f}s/ep)", flush=True)
    print(f"Model saved to {SAVE_PATH}", flush=True)
    print(f"TensorBoard log: {log_dir}  (run `tensorboard --logdir runs`)",
          flush=True)


if __name__ == "__main__":
    main()
