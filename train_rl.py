# train_rl.py
"""Multi-agent RL training for all storage agents.

Two algorithms:
  - TD3 (default): independent learners, one policy and replay buffer per
    agent.
  - MATD3 (--algo matd3): centralized training with decentralized execution
    (CTDE). One shared replay buffer; each agent has its own actor and a
    centralized critic that sees all agents' unique observations and actions.

RL controls the bid/offer PRICE (bid_mult, offer_adder). Storage SOC is
dispatched by the market-clearing optimizer using the declared prices:
self_schedule=False disables the MPC pre-scheduler so the OPF objective
(which prices storage at the agent's declared bid/offer) steers storage.

The per-block reward is raw profit minus the truthful-bidding baseline
profit (differential reward, on by default; disable with --no-diff-reward).
"""

import os
import time
import argparse
import random
import datetime
import copy
import numpy as np
import torch

from scenarios import get_scenario
from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS
from rl_td3 import TD3, save_policy

N_EPISODES = 200
# Each training run trains on a single fixed scenario by default, so the
# reward curve is free of scenario-rotation noise and per-run behavior is
# scenario-specific. Pass --scenarios with a comma-separated list to rotate.
DEFAULT_TRAIN_SCENARIO = "baseline"


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
        description="Train RL bidding policies for all storage agents "
                    "(independent TD3 or MATD3/CTDE)")
    parser.add_argument("--algo", type=str, default="matd3",
                        choices=["td3", "matd3"],
                        help="Training algorithm: matd3 (centralized critic, "
                             "CTDE) or td3 (independent learners)")
    parser.add_argument("--agent-names", type=str, default=None,
                        help="Comma-separated agent names to train. "
                             "Default: all agents with storage.")
    parser.add_argument("--list-agents", action="store_true",
                        help="Print all available agent names and exit")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario names for training. "
                             "Default: a single fixed scenario (baseline); "
                             "pass a list to rotate across them")
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
                        help="L2 penalty on the pre-tanh bid logit to keep "
                             "bids away from action bounds (0 = no penalty)")
    parser.add_argument("--offer-dev-penalty", type=float, default=0.5,
                        help="L2 penalty on the pre-tanh offer logit to keep "
                             "offers away from action bounds (0 = no penalty)")
    parser.add_argument("--bid-mult-low", type=float, default=None,
                        help="Lower bound for bid_mult action space. "
                             "Default: from config.market_design.bid_mult_range")
    parser.add_argument("--bid-mult-high", type=float, default=None,
                        help="Upper bound for bid_mult action space. "
                             "Default: from config.market_design.bid_mult_range")
    parser.add_argument("--no-diff-reward", action="store_false",
                        dest="diff_reward", default=True,
                        help="Disable differential reward (use raw profit "
                             "instead of profit minus truthful baseline)")
    parser.add_argument("--noise-anneal-steps", type=int, default=5000,
                        help="Steps over which MATD3 exploration noise anneals "
                             "from --noise-std to a floor of 0.05")
    parser.add_argument("--start-steps", type=int, default=500,
                        help="Random exploration steps before learning")
    parser.add_argument("--save-dir", type=str, default="policies/multi_agent",
                        help="Directory for saved policy files")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Evaluate policies every N episodes during "
                             "training (0 disables in-training evaluation)")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="Deterministic evaluation episodes per eval step")
    parser.add_argument("--early-stop-steps", type=int, default=0,
                        help="Early stop when the eval metric shows no "
                             "relative improvement for this many consecutive "
                             "evals (0 disables early stopping)")
    parser.add_argument("--early-stop-threshold", type=float, default=0.05,
                        help="Relative improvement required to avoid early "
                             "stopping")
    parser.add_argument("--eval-metric", type=str, default="mean_reward",
                        help="Metric tracked for best-policy selection and "
                             "early stopping")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable in-training evaluation entirely")
    parser.add_argument("--eval-use-diff-reward", action="store_true",
                        help="Use differential reward in evaluation episodes "
                             "(default off: evaluates raw profit)")
    return parser


def _resolve_train_scenarios(scenarios_arg):
    """Resolve the training scenario list from a --scenarios argument.

    Default is a single fixed scenario (DEFAULT_TRAIN_SCENARIO) so that each
    training run trains on one scenario only, keeping the reward curve free of
    scenario-rotation noise. Passing an explicit comma-separated list restores
    rotation across that subset.
    """
    if scenarios_arg:
        return [s.strip() for s in scenarios_arg.split(",")]
    return [DEFAULT_TRAIN_SCENARIO]


def _train_matd3(args, env, action_bounds, rl_agent_names,
                 train_scenarios, writer, tracker=None):
    """CTDE training loop: one MATD3 with a shared replay buffer.

    Each agent keeps its own actor; a centralized critic per agent sees all
    agents' unique observations plus all actions during training, which
    resolves the non-stationarity that makes independent critics flat along
    the action direction.

    Returns True when the run was stopped early by the PolicyTracker.
    """
    from rl_bidding import MATD3
    from rl_td3 import save_policy

    obs_dim = env.get_state_dim()
    matd3 = MATD3(env, lr=args.lr, noise_std=args.noise_std,
                  start_steps=args.start_steps,
                  bid_dev_penalty=args.bid_dev_penalty,
                  offer_dev_penalty=args.offer_dev_penalty,
                  noise_anneal_steps=args.noise_anneal_steps)
    low = action_bounds[0].numpy()
    high = action_bounds[1].numpy()

    stopped_early = False
    eval_env = tracker.build_eval_env(env.config) if tracker is not None \
        else None

    t0 = time.time()
    for ep in range(args.episodes):
        sc_name = random.choice(train_scenarios)
        config_copy = copy.deepcopy(env.config)
        agents_sc, wholesale = get_scenario(sc_name, T=env.T,
                                            config=config_copy)
        env.set_agents(agents_sc, wholesale)
        env.config = config_copy
        obs = env.reset()

        ep_rewards = {nm: 0.0 for nm in rl_agent_names}
        ep_welfare = 0.0
        ep_re_rate = 0.0

        for _ in range(N_BLOCKS):
            actions = {}
            for nm in rl_agent_names:
                if matd3.total_steps < matd3.start_steps:
                    actions[nm] = np.random.uniform(low, high)
                else:
                    actions[nm] = matd3.select_action(
                        obs[nm], nm, add_noise=True)

            next_obs, rewards, done, info = env.step(actions)

            obs_arr = np.stack([obs[nm] for nm in rl_agent_names])
            act_arr = np.stack([actions[nm] for nm in rl_agent_names])
            rew_arr = np.array([rewards.get(nm, 0.0) for nm in rl_agent_names])
            next_obs_arr = np.stack([
                next_obs.get(nm, np.zeros(obs_dim, dtype=np.float32))
                for nm in rl_agent_names])
            done_arr = np.full(len(rl_agent_names), done, dtype=np.float32)
            matd3.buffer.add(obs_arr, act_arr, rew_arr, next_obs_arr, done_arr)
            matd3.total_steps += 1

            for nm in rl_agent_names:
                ep_rewards[nm] += rewards.get(nm, 0.0)
            ep_welfare = info.get("welfare", 0.0)
            ep_re_rate = info.get("re_rate", 0.0)
            obs = next_obs

        # Post-episode updates
        if matd3.total_steps >= matd3.start_steps:
            c_losses, a_losses = [], []
            c_losses_by_agent = {nm: [] for nm in rl_agent_names}
            a_losses_by_agent = {nm: [] for nm in rl_agent_names}
            for _ in range(N_BLOCKS):
                li = matd3.update()
                if li["critic_loss"] is not None:
                    c_losses.append(li["critic_loss"])
                if li["actor_loss"] is not None:
                    a_losses.append(li["actor_loss"])
                for nm in rl_agent_names:
                    cl = li["critic_loss_by_agent"].get(nm)
                    if cl is not None:
                        c_losses_by_agent[nm].append(cl)
                    al = li["actor_loss_by_agent"].get(nm)
                    if al is not None:
                        a_losses_by_agent[nm].append(al)
            if c_losses:
                writer.add_scalar("Loss/critic", np.mean(c_losses), ep)
            if a_losses:
                writer.add_scalar("Loss/actor", np.mean(a_losses), ep)
            for nm in rl_agent_names:
                if c_losses_by_agent[nm]:
                    writer.add_scalar(f"Loss/critic/{nm}",
                                      np.mean(c_losses_by_agent[nm]), ep)
                if a_losses_by_agent[nm]:
                    writer.add_scalar(f"Loss/actor/{nm}",
                                      np.mean(a_losses_by_agent[nm]), ep)

        # TensorBoard
        for nm in rl_agent_names:
            writer.add_scalar(f"Reward/{nm}", ep_rewards[nm], ep)
        writer.add_scalar("Welfare", ep_welfare, ep)
        writer.add_scalar("RE_Rate", ep_re_rate, ep)
        writer.add_scalar("Reward/mean",
                          sum(ep_rewards.values()) / len(rl_agent_names), ep)

        # In-training evaluation + early stopping (opt-in via tracker)
        if tracker is not None and args.eval_interval > 0 \
                and (ep + 1) % args.eval_interval == 0:
            metrics = tracker.evaluate(eval_env, matd3.actors)
            writer.add_scalar("Eval/mean_reward", metrics["mean_reward"], ep)
            writer.add_scalar("Eval/welfare", metrics["welfare"], ep)
            if tracker.compare_and_save_policies(ep + 1, matd3.actors,
                                                 metrics):
                stopped_early = True
                break

        # Console logging
        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
            elapsed = time.time() - t0
            mean_r = sum(ep_rewards.values()) / len(rl_agent_names)
            print(f"Ep {ep+1}/{args.episodes} | mean_reward={mean_r:+.1f} | "
                  f"welfare={ep_welfare:.0f} | RE={ep_re_rate:.1f}% | "
                  f"sc={sc_name} | elapsed={elapsed:.0f}s | algo=matd3",
                  flush=True)

        # Checkpoint
        if (ep + 1) % 50 == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            for nm in rl_agent_names:
                ckpt_path = os.path.join(
                    args.save_dir, f"{nm}_ckpt_{ep+1}.pt")
                save_policy(matd3.actors[nm], ckpt_path,
                            obs_spec=env.obs_spec,
                            action_spec=env.action_spec)
            print(f"  -> checkpoint: {args.save_dir}", flush=True)

    # ---- Save final policies ----
    os.makedirs(args.save_dir, exist_ok=True)
    for nm in rl_agent_names:
        save_path = os.path.join(args.save_dir, f"{nm}.pt")
        save_policy(matd3.actors[nm], save_path,
                    obs_spec=env.obs_spec, action_spec=env.action_spec)
    if tracker is not None:
        tracker.save_last(matd3.actors)
    elapsed = time.time() - t0
    done_ep = ep + 1 if stopped_early else args.episodes
    print(f"MATD3 training complete: {done_ep} episodes in "
          f"{elapsed:.0f}s ({elapsed / max(done_ep, 1):.1f}s/ep)"
          + (" (early stopped)" if stopped_early else ""), flush=True)
    print(f"Models saved: {args.save_dir}/{{name}}.pt "
          f"({len(rl_agent_names)} agents)", flush=True)
    return stopped_early


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

    # ---- Action bounds: single source of truth in the market config ----
    if args.bid_mult_low is None or args.bid_mult_high is None:
        low, high = config.market_design.bid_mult_range
        args.bid_mult_low, args.bid_mult_high = low, high

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
    train_scenarios = _resolve_train_scenarios(args.scenarios)
    if args.eval_scenarios:
        eval_scenarios = [s.strip() for s in args.eval_scenarios.split(",")]
    else:
        eval_scenarios = []

    sc_mode = "fixed" if len(train_scenarios) == 1 else "rotation"
    print(f"Training scenario(s): {train_scenarios} ({sc_mode})")
    if eval_scenarios:
        print(f"Eval scenarios (held out): {eval_scenarios}")
    print(f"Episodes: {args.episodes}  |  LR: {args.lr}  |  "
          f"Noise std: {args.noise_std}", flush=True)

    # ---- Create environment (all storage agents are RL) ----
    # Reward is market profit, optionally shaped by subtracting the truthful-
    # bidding baseline for the same block (differential reward). The deviation
    # penalty lives in the actor loss, not the environment reward, so the
    # critic learns the real profit objective.
    env = BiddingEnv(agents, config, rl_agent_names=rl_agent_names,
                     use_differential_reward=args.diff_reward,
                     bid_mult_low=args.bid_mult_low,
                     bid_mult_high=args.bid_mult_high)
    print(f"RL agents in env: {[a.name for a in env.rl_agents]}", flush=True)

    # ---- In-training policy tracker (evaluation + best/last + early stop) ----
    if args.no_eval or args.eval_interval <= 0:
        tracker = None
    else:
        from rl_training import PolicyTracker
        eval_sc = eval_scenarios[0] if eval_scenarios \
            else DEFAULT_TRAIN_SCENARIO
        tracker = PolicyTracker(
            args.save_dir, rl_agent_names, eval_scenario=eval_sc,
            metric=args.eval_metric, eval_episodes=args.eval_episodes,
            early_stopping_steps=args.early_stop_steps,
            early_stopping_threshold=args.early_stop_threshold,
            use_differential_reward=args.eval_use_diff_reward,
            obs_spec=env.obs_spec, action_spec=env.action_spec)
        print(f"Tracker: eval every {args.eval_interval} eps on "
              f"'{eval_sc}' | early stop: "
              f"{'on' if args.early_stop_steps > 0 else 'off'}",
              flush=True)

    action_bounds = env.get_action_bounds()
    obs_dim = env.get_state_dim()
    act_dim = 2

    td3s = {name: TD3(obs_dim, act_dim, action_bounds,
                      lr=args.lr, noise_std=args.noise_std,
                      start_steps=args.start_steps,
                      bid_dev_penalty=args.bid_dev_penalty,
                      offer_dev_penalty=args.offer_dev_penalty)
            for name in rl_agent_names}

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join("runs",
                           f"train-multi-"
                           + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # ---- CTDE path: MATD3 with a centralized critic ----
    if args.algo == "matd3":
        stopped_early = _train_matd3(
            args, env, action_bounds, rl_agent_names, train_scenarios,
            writer, tracker)
        writer.close()
        if eval_scenarios:
            print(f"\nHeld-out scenarios: {eval_scenarios}")
            print("Run `python eval_agents.py --policies "
                  f"{args.save_dir}` for detailed evaluation.")
        return

    # ---- Independent TD3 training loop ----
    stopped_early = False
    eval_env = tracker.build_eval_env(config) if tracker is not None else None
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

        # In-training evaluation + early stopping (opt-in via tracker)
        if tracker is not None and args.eval_interval > 0 \
                and (ep + 1) % args.eval_interval == 0:
            actors = {name: td3s[name].actor for name in rl_agent_names}
            metrics = tracker.evaluate(eval_env, actors)
            writer.add_scalar("Eval/mean_reward", metrics["mean_reward"], ep)
            writer.add_scalar("Eval/welfare", metrics["welfare"], ep)
            if tracker.compare_and_save_policies(ep + 1, actors, metrics):
                stopped_early = True
                break

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
                save_policy(td3.actor, ckpt_path,
                            obs_spec=env.obs_spec,
                            action_spec=env.action_spec)
            print(f"  -> checkpoint: {args.save_dir}", flush=True)

    writer.close()

    # ---- Save final policies ----
    os.makedirs(args.save_dir, exist_ok=True)
    for name, td3 in td3s.items():
        save_path = os.path.join(args.save_dir, f"{name}.pt")
        save_policy(td3.actor, save_path,
                    obs_spec=env.obs_spec, action_spec=env.action_spec)
    if tracker is not None:
        actors = {name: td3s[name].actor for name in rl_agent_names}
        tracker.save_last(actors)
    elapsed = time.time() - t0
    done_ep = ep + 1 if stopped_early else args.episodes
    print(f"Training complete: {done_ep} episodes in {elapsed:.0f}s "
          f"({elapsed / max(done_ep, 1):.1f}s/ep)"
          + (" (early stopped)" if stopped_early else ""), flush=True)
    print(f"Models saved: {args.save_dir}/{{name}}.pt "
          f"({len(rl_agent_names)} agents)", flush=True)
    print(f"TensorBoard: {log_dir}", flush=True)

    if eval_scenarios:
        print(f"\nHeld-out scenarios: {eval_scenarios}")
        print("Run `python eval_agents.py --policies "
              f"{args.save_dir}` for detailed evaluation.")


if __name__ == "__main__":
    main()
