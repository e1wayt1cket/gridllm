# train_rl_chunked.py
"""Chunked RL training: run N episodes, auto-resume from latest checkpoint."""
import sys, os, time
import numpy as np
import torch

from scenarios import get_scenario
from models import MarketConfig
from rl_env import BiddingEnv
from rl_bidding import MATD3, save_policies, load_policies, _TRAINED_POLICIES

CHECKPOINT = "policies/latest.pt"

def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("Building environment...", flush=True)
    config = MarketConfig(opf_mode="socp", verbose=False)
    agents, _ = get_scenario("baseline", T=96, config=config)
    env = BiddingEnv(agents, config)
    matd3 = MATD3(env)

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT):
        print(f"Loading checkpoint: {CHECKPOINT}", flush=True)
        load_policies(CHECKPOINT, env.get_state_dim(), env.get_action_bounds())
        for nm in matd3.agent_names:
            if nm in _TRAINED_POLICIES:
                matd3.actors[nm].load_state_dict(_TRAINED_POLICIES[nm].state_dict())
                matd3.actor_targets[nm].load_state_dict(_TRAINED_POLICIES[nm].state_dict())
        print("Checkpoint loaded.", flush=True)

    print(f"RL agents: {len(env.rl_agents)}, episodes={n_episodes}, chunk={chunk}", flush=True)

    t0 = time.time()
    for ep in range(n_episodes):
        obs = env.reset()
        ep_rewards = {nm: 0.0 for nm in matd3.agent_names}

        for _ in range(24):
            actions = {}
            for nm in matd3.agent_names:
                actions[nm] = matd3.select_action(obs[nm], nm, add_noise=True)
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

        avg_r = np.mean([ep_rewards[nm] for nm in matd3.agent_names])
        elapsed = time.time() - t0
        eta = (elapsed / (ep + 1)) * (n_episodes - ep - 1) if ep > 0 else 0

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Ep {ep+1}/{n_episodes} | "
                  f"avg_reward={avg_r:+.3f} | "
                  f"welfare={info.get('welfare', 0):.0f} | "
                  f"RE={info.get('re_rate', 0):.1f}% | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

        # Checkpoint every chunk
        if (ep + 1) % chunk == 0:
            os.makedirs("policies", exist_ok=True)
            save_policies(matd3.actors, CHECKPOINT)
            ep_path = f"policies/ep_{ep+1}.pt"
            save_policies(matd3.actors, ep_path)
            print(f"  -> saved: {CHECKPOINT} & {ep_path}", flush=True)

    # Final save
    os.makedirs("policies", exist_ok=True)
    save_policies(matd3.actors, "policies/default.pt")
    save_policies(matd3.actors, CHECKPOINT)
    elapsed = time.time() - t0
    print(f"DONE: {n_episodes} eps in {elapsed:.0f}s ({elapsed/n_episodes:.1f}s/ep)", flush=True)
    print("Model saved to policies/default.pt", flush=True)


if __name__ == "__main__":
    main()
