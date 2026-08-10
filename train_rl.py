# train_rl.py
"""Multi-agent RL training with TD3 for all storage agents.

Each storage agent is an independent learner with its own TD3 policy. All
policies are stepped together in one BiddingEnv each block; each agent
updates from its own replay buffer (Independent Learner paradigm).

RL controls the bid/offer PRICE (bid_mult, offer_adder). Storage SOC is
dispatched by the market-clearing optimizer using the declared prices:
self_schedule=False disables the MPC pre-scheduler so the OPF objective
(which prices storage at the agent's declared bid/offer) steers storage.
"""

import os
import time
import argparse
import random
import datetime
import copy
import numpy as np
import torch

from scenarios import get_scenario, list_scenarios
from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS
from rl_td3 import TD3, save_policy

N_EPISODES = 200


def list_agents_command():
    """Print all trainable agents for a reference scenario."""
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    agents, _ = get_scenario("baseline", T=96, config=config)
    print(f"{'Name':<25s} {'Bus':>4s} {'Type':<15s} {'Storage':>8s} "
          f"{'Wind':>5s} {'Prosumer':>10s}")
    print("-" * 72)
    for a in agents:
        has_stor = "yes" if a.storage is not None else "no"
        has_wind = "yes" if a.has_wind else "no"
        print(f"{a.name:<25s} {a.bus:>4d} {a.load_type:<15s} "
              f"{has_stor:>8s} {has_wind:>5s} {str(a.is_prosumer):>10s}")


def build_parser():
    """Construct the training CLI parser (testable in isolation)."""
    parser = argparse.ArgumentParser(
        description="Train independent TD3 bidding policies for all storage agents")
    parser.add_argument("--agent-names", type=str, default=None,
                        help="Comma-separated agent names to train. "
                             "Default: all agents with storage.")
    parser.add_argument("--list-agents", action="store_true",
                        help="Print all available agent names and exit")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario names for training. "
                             "Default: all scenarios except re_ramp variants")
    parser.add_argument("--eval-scenarios", type=str, default=None,
                        help="Comma-separated scenario names held out for "
                             "post-training evaluation")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for actor and critic")
    parser.add_argument("--noise-std", type=float, default=0.2,
                        help="Exploration noise standard deviation")
    parser.add_argument("--bid-dev-penalty", type=float, default=5.0,
                        help="Penalty per unit |bid_mult - 1.0| per period "
                             "(0 = no penalty)")
    parser.add_argument("--offer-dev-penalty", type=float, default=0.5,
                        help="Penalty per unit offer_adder per period "
                             "(0 = no penalty)")
    parser.add_argument("--bid-mult-low", type=float, default=0.6,
                        help="Lower bound for bid_mult action space")
    parser.add_argument("--bid-mult-high", type=float, default=1.4,
                        help="Upper bound for bid_mult action space")
    parser.add_argument("--start-steps", type=int, default=500,
                        help="Random exploration steps before TD3 learning")
    parser.add_argument("--save-dir", type=str, default="policies/multi_agent",
                        help="Directory for saved policy files")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_agents:
        list_agents_command()
        return

    # ---- Seed RNGs ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Build config (pure economic welfare objective, RL-dispatched storage) ----
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    # Storage SOC is dispatched by the market optimizer using declared prices;
    # the MPC pre-scheduler and the nodal re-solve pass are disabled.
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False

    # ---- Build initial agents from baseline scenario ----
    agents, _ = get_scenario("baseline", T=96, config=config)

    # ---- Determine RL agents ----
    if args.agent_names:
        rl_agent_names = [s.strip() for s in args.agent_names.split(",")]
    else:
        rl_agent_names = [a.name for a in agents if a.storage is not None]
    known = {a.name for a in agents}
    missing = [n for n in rl_agent_names if n not in known]
    if missing:
        available = [a.name for a in agents if a.storage is not None]
        print(f"Agents not found: {missing}. Available storage agents: "
              f"{available}")
        return

    print(f"Training {len(rl_agent_names)} storage agents: {rl_agent_names}",
          flush=True)
    print(f"Action bounds: bid_mult=[{args.bid_mult_low}, "
          f"{args.bid_mult_high}], offer_adder=[0, 50]", flush=True)

    # ---- Determine training and evaluation scenarios ----
    all_scenarios = list_scenarios()
    if args.scenarios:
        train_scenarios = [s.strip() for s in args.scenarios.split(",")]
    else:
        train_scenarios = [s for s in all_scenarios
                           if not s.startswith("re_ramp")]
    if args.eval_scenarios:
        eval_scenarios = [s.strip() for s in args.eval_scenarios.split(",")]
    else:
        eval_scenarios = []

    print(f"Training scenarios: {train_scenarios}")
    if eval_scenarios:
        print(f"Eval scenarios (held out): {eval_scenarios}")
    print(f"Episodes: {args.episodes}  |  LR: {args.lr}  |  "
          f"Noise std: {args.noise_std}", flush=True)

    # ---- Create environment (all storage agents are RL) ----
    env = BiddingEnv(agents, config, rl_agent_names=rl_agent_names,
                     bid_dev_penalty=args.bid_dev_penalty,
                     offer_dev_penalty=args.offer_dev_penalty,
                     bid_mult_low=args.bid_mult_low,
                     bid_mult_high=args.bid_mult_high)
    print(f"RL agents in env: {[a.name for a in env.rl_agents]}", flush=True)

    action_bounds = env.get_action_bounds()
    obs_dim = env.get_state_dim()
    act_dim = 2

    td3s = {name: TD3(obs_dim, act_dim, action_bounds,
                      lr=args.lr, noise_std=args.noise_std,
                      start_steps=args.start_steps)
            for name in rl_agent_names}

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join("runs",
                           f"train-multi-"
                           + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # ---- Training loop ----
    t0 = time.time()
    for ep in range(args.episodes):
        sc_name = random.choice(train_scenarios)
        config_copy = copy.deepcopy(config)
        agents_sc, wholesale = get_scenario(sc_name, T=96, config=config_copy)
        env.set_agents(agents_sc, wholesale)
        env.config = config_copy
        obs = env.reset()

        ep_rewards = {name: 0.0 for name in rl_agent_names}
        ep_welfare = 0.0
        ep_re_rate = 0.0

        for _ in range(N_BLOCKS):
            actions = {}
            for name, td3 in td3s.items():
                if td3.total_steps < td3.start_steps:
                    actions[name] = np.random.uniform(
                        action_bounds[0].numpy(), action_bounds[1].numpy())
                else:
                    actions[name] = td3.select_action(
                        obs[name], add_noise=True)

            next_obs, rewards, done, info = env.step(actions)

            for name, td3 in td3s.items():
                r = rewards.get(name, 0.0)
                ep_rewards[name] += r
                nxt = next_obs.get(name, np.zeros(obs_dim, dtype=np.float32))
                td3.buffer.add(obs[name], actions[name], r, nxt, done)
                td3.total_steps += 1

            ep_welfare = info.get("welfare", 0.0)
            ep_re_rate = info.get("re_rate", 0.0)
            obs = next_obs

        # Post-episode updates and loss stats per agent
        for name, td3 in td3s.items():
            if td3.total_steps >= td3.start_steps:
                for _ in range(N_BLOCKS):
                    td3.update()
            loss_info = td3.update()
            c_loss = loss_info.get("critic_loss")
            a_loss = loss_info.get("actor_loss")
            if c_loss is not None:
                writer.add_scalar(f"Loss/critic/{name}", c_loss, ep)
            if a_loss is not None:
                writer.add_scalar(f"Loss/actor/{name}", a_loss, ep)

        # TensorBoard
        for name in rl_agent_names:
            writer.add_scalar(f"Reward/{name}", ep_rewards[name], ep)
        writer.add_scalar("Welfare", ep_welfare, ep)
        writer.add_scalar("RE_Rate", ep_re_rate, ep)
        writer.add_scalar("Reward/mean",
                          sum(ep_rewards.values()) / len(rl_agent_names), ep)

        # Console logging
        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
            elapsed = time.time() - t0
            mean_r = sum(ep_rewards.values()) / len(rl_agent_names)
            print(f"Ep {ep+1}/{args.episodes} | mean_reward={mean_r:+.1f} | "
                  f"welfare={ep_welfare:.0f} | RE={ep_re_rate:.1f}% | "
                  f"sc={sc_name} | elapsed={elapsed:.0f}s", flush=True)

        # Checkpoint
        if (ep + 1) % 50 == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            for name, td3 in td3s.items():
                ckpt_path = os.path.join(
                    args.save_dir, f"{name}_ckpt_{ep+1}.pt")
                save_policy(td3.actor, ckpt_path)
            print(f"  -> checkpoint: {args.save_dir}", flush=True)

    writer.close()

    # ---- Save final policies ----
    os.makedirs(args.save_dir, exist_ok=True)
    for name, td3 in td3s.items():
        save_path = os.path.join(args.save_dir, f"{name}.pt")
        save_policy(td3.actor, save_path)
    elapsed = time.time() - t0
    print(f"Training complete: {args.episodes} episodes in {elapsed:.0f}s "
          f"({elapsed / args.episodes:.1f}s/ep)", flush=True)
    print(f"Models saved: {args.save_dir}/{{name}}.pt "
          f"({len(rl_agent_names)} agents)", flush=True)
    print(f"TensorBoard: {log_dir}", flush=True)

    if eval_scenarios:
        print(f"\nHeld-out scenarios: {eval_scenarios}")
        print("Run `python eval_agents.py --policies "
              f"{args.save_dir}` for detailed evaluation.")


if __name__ == "__main__":
    main()
