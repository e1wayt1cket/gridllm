# eval_agents.py
"""Evaluate trained RL policies against baseline (bid_mult=1.0, offer_adder=0.0).

Load any combination of trained policy files and run market simulations to
measure:
  - Per-agent profit vs baseline
  - Social welfare change
  - RE consumption rate change
  - Carbon emissions change

Usage:
  python eval_agents.py --policies policies/Agent_A.pt --scenarios baseline
  python eval_agents.py --policies policies/ --scenarios all --output results.csv
"""

import os
import argparse
import copy
import numpy as np
import torch
from typing import Dict, List, Optional

from scenarios import get_scenario, list_scenarios
from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS, BLOCK_SIZE
from rl_td3 import Actor, load_policy
from rl_profit_diagnostics import valuation_artifact
from surplus_metrics import (consumer_metrics, load_payment_weighted_markup,
                             market_power_split)

# Schedule keys stitched per day when capture=True (mirrors empty_schedules).
_CAPTURE_KEYS = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
                 "pv_used", "wind_used"]
# CSV columns appended to the combined (ALL) row under --consumer-metrics.
_CONSUMER_COLUMNS = ["cs_baseline", "cs_rl", "cs_delta",
                     "cp_baseline", "cp_rl", "cp_delta",
                     "lmp_markup_baseline", "lmp_markup_rl",
                     "lmp_markup_delta", "market_power_arb",
                     "market_power_power"]


def _n_commit(env, t_start: int) -> int:
    """Committed periods of the rolling window starting at t_start."""
    return min(BLOCK_SIZE, min(t_start + env.roll_horizon, env.T) - t_start)


def compute_agent_profit(schedule: dict, lmp_node: np.ndarray,
                         agent, config: MarketConfig,
                         n_periods: int) -> float:
    """Compute raw profit for one agent from a market result.

    Profit = consumer surplus - generation cost + market net payment
             - unserved penalty - storage cycle cost.

    Parameters
    ----------
    schedule : dict  Per-agent schedule arrays (p_buy, p_sell, served, ...).
    lmp_node : np.ndarray  (T,) nodal LMP for this agent's bus.
    agent : Agent
    config : MarketConfig
    n_periods : int  Number of committed periods to sum over.

    Returns
    -------
    float  Total profit in CNY.
    """
    profit = 0.0
    cycle_cost = float(config.storage.cycle_cost)
    for d in range(n_periods):
        cons_val = agent.bid_value * schedule["served"][d]
        gen_cost = agent.offer_cost * (schedule["pv_used"][d]
                                       + schedule["wind_used"][d])
        mkt_pmt = (schedule["p_sell"][d] * lmp_node[d]
                   - schedule["p_buy"][d] * lmp_node[d])
        penalty = config.market_design.penalty_unserved \
            * schedule["unserved"][d]
        step = float(cons_val - gen_cost + mkt_pmt - penalty)
        if agent.storage is not None:
            step -= cycle_cost * (schedule["p_ch"][d]
                                  + schedule["p_dis"][d])
        profit += step
    return profit


def actor_predict(actor: Actor, obs: np.ndarray) -> np.ndarray:
    """Run Actor inference on a single observation, return numpy action."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return actor(obs_t).squeeze(0).numpy()


def run_episode(env: BiddingEnv, policy: Optional[Actor] = None,
                n_blocks: int = N_BLOCKS, capture: bool = False) -> dict:
    """Run an episode of up to *n_blocks* rolling-window blocks.

    Parameters
    ----------
    env : BiddingEnv
    policy : Actor or None
        If None, all agents use defaults (baseline run).
    n_blocks : int
        Number of blocks (BLOCK_SIZE periods each) to run, starting from the
        beginning of the day. Must not exceed N_BLOCKS; the default is the
        full day.
    capture : bool
        When True, also stitch the committed-period schedules of every agent,
        the (T, n_buses) nodal LMP and the per-block wholesale curve actually
        fed to each clear into full-day arrays, returned under "sched",
        "lmp", "wholesale". Default False keeps the legacy return unchanged.

    Returns
    -------
    dict with keys: welfare, re_rate, carbon, profits (per-agent dict), plus
    sched/lmp/wholesale when capture is True.
    """
    obs = env.reset()
    total_welfare = 0.0
    total_re_rate = 0.0
    agent_profits: Dict[str, float] = {}
    sched = None
    lmp = None
    wholesale_day = None
    if capture:
        sched = {a.name: {k: np.zeros(env.T) for k in _CAPTURE_KEYS}
                 for a in env.all_agents}
        wholesale_day = np.zeros(env.T)

    for block in range(n_blocks):
        t_start = block * BLOCK_SIZE
        if policy is not None:
            # RL agent uses trained policy; others use defaults from env
            agent_name = env.rl_agents[0].name
            action = actor_predict(policy, obs[agent_name])
            acts = {agent_name: action}
        else:
            # Baseline: all agents use defaults
            default_act = np.array([1.0, 0.0], dtype=np.float32)
            acts = {a.name: default_act for a in env.all_agents}

        next_obs, rewards, done, info = env.step(acts)
        total_welfare += info.get("welfare", 0.0)
        total_re_rate = info.get("re_rate", 0.0)  # last block's value

        res = env._last_result
        if capture and res is not None:
            n_commit = _n_commit(env, t_start)
            for a in env.all_agents:
                ws = res["schedules"].get(a.name)
                if ws is None:
                    continue
                for key in sched[a.name]:
                    sched[a.name][key][t_start:t_start + n_commit] = \
                        ws[key][:n_commit]
            if lmp is None:
                lmp = np.zeros((env.T, res["lmp"].shape[1]))
            lmp[t_start:t_start + n_commit, :] = res["lmp"][:n_commit, :]
            wholesale_day[t_start:t_start + n_commit] = \
                env._last_wholesale[:n_commit]

        # Accumulate per-agent profit from per-block rewards
        if info.get("lmp") is not None:
            for a in env.all_agents:
                nm = a.name
                if nm in rewards:
                    agent_profits[nm] = agent_profits.get(nm, 0.0) \
                        + rewards.get(nm, 0.0)

        if done:
            break
        obs = next_obs

    carbon = 0.0  # will be extracted from last info if available
    out = {"welfare": total_welfare,
           "re_rate": total_re_rate,
           "carbon": carbon,
           "profits": agent_profits}
    if capture:
        out["sched"] = sched
        out["lmp"] = lmp
        out["wholesale"] = wholesale_day
    return out


def run_combined_episode(env: BiddingEnv, policies: Dict[str, Actor],
                         n_blocks: int = N_BLOCKS,
                         capture: bool = False) -> dict:
    """Run a combined-fleet episode: every matched policy acts simultaneously.

    Each block the trained actors pick bids/offers from their own observation,
    and the market clears once for the whole fleet.

    Parameters
    ----------
    env : BiddingEnv
        Constructed with rl_agent_names covering the agents present in
        *policies*; all other agents bid truthfully via the env defaults.
    policies : dict  {agent_name: Actor}.
    n_blocks : int  Number of blocks to run (default full day).

    Returns
    -------
    dict with keys:
      profits  per-rl-agent accumulated reward
      welfare  accumulated system welfare
      re_rate  renewable consumption rate from the last block
      sched    per-agent committed schedules padded to length env.T
      declared per-rl-agent bid_mult/offer_adder padded to length env.T
    """
    obs = env.reset()
    rl_names = [a.name for a in env.rl_agents]
    profits: Dict[str, float] = {nm: 0.0 for nm in rl_names}
    welfare = 0.0
    re_rate = 0.0
    T = env.T
    sched = {a.name: {k: np.zeros(T) for k in _CAPTURE_KEYS}
             for a in env.all_agents}
    declared = {nm: {"bid_mult": np.zeros(T), "offer_adder": np.zeros(T)}
                for nm in rl_names}
    lmp = None
    wholesale_day = np.zeros(T) if capture else None
    for block in range(n_blocks):
        t_start = block * BLOCK_SIZE
        acts = {nm: actor_predict(policies[nm], obs[nm]) for nm in rl_names}
        next_obs, rewards, done, info = env.step(acts)
        res = env._last_result
        n_commit = min(BLOCK_SIZE,
                       min(t_start + env.roll_horizon, T) - t_start)
        if res is not None:
            for nm in sched:
                ws = res["schedules"].get(nm)
                if ws is None:
                    continue
                for key in sched[nm]:
                    sched[nm][key][t_start:t_start + n_commit] = \
                        ws[key][:n_commit]
            if capture:
                if lmp is None:
                    lmp = np.zeros((T, res["lmp"].shape[1]))
                lmp[t_start:t_start + n_commit, :] = res["lmp"][:n_commit, :]
                wholesale_day[t_start:t_start + n_commit] = \
                    env._last_wholesale[:n_commit]
        for nm in rl_names:
            ca = env.current_actions.get(nm)
            if ca is not None:
                declared[nm]["bid_mult"][t_start:t_start + n_commit] = \
                    ca["bid_mult"][t_start:t_start + n_commit]
                declared[nm]["offer_adder"][t_start:t_start + n_commit] = \
                    ca["offer_adder"][t_start:t_start + n_commit]
        welfare += info.get("welfare", 0.0)
        re_rate = info.get("re_rate", 0.0)
        for nm in rl_names:
            profits[nm] = profits.get(nm, 0.0) + rewards.get(nm, 0.0)
        if done:
            break
        obs = next_obs
    out = {"profits": profits, "welfare": welfare, "re_rate": re_rate,
           "sched": sched, "declared": declared}
    if capture:
        out["lmp"] = lmp
        out["wholesale"] = wholesale_day
    return out


def load_policies_from_dir(dir_path: str, obs_dim: int,
                           action_bounds: torch.Tensor,
                           checkpoint: Optional[int] = None,
                           obs_spec=None, action_spec=None) \
        -> Dict[str, Actor]:
    """Load all .pt policy files from a directory.

    Filename is used as agent name: "Agent_X.pt" → agent "Agent_X". When
    checkpoint is given, loads "{name}_ckpt_{N}.pt" for each agent instead
    of the final policy files. obs_spec/action_spec are forwarded to
    load_policy for the metadata mismatch check.
    """
    policies = {}
    if not os.path.isdir(dir_path):
        return policies
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".pt"):
            continue
        if checkpoint is not None:
            if f"_ckpt_{checkpoint}." not in fname:
                continue
            agent_name = fname.replace(".pt", "")
            agent_name = agent_name.replace(f"_ckpt_{checkpoint}", "")
        else:
            agent_name = fname.replace(".pt", "")
            # Skip checkpoint files
            if "_ckpt_" in agent_name:
                continue
        path = os.path.join(dir_path, fname)
        try:
            net = load_policy(path, obs_dim, action_bounds,
                              obs_spec=obs_spec, action_spec=action_spec)
            policies[agent_name] = net
            print(f"  Loaded: {agent_name} from {path}")
        except Exception as e:
            print(f"  Failed to load {path}: {e}")
    return policies


def default_eval_config() -> MarketConfig:
    """Config recipe used for policy evaluation.

    Storage is dispatched by the market optimizer using declared prices (same
    mechanism as training); MPC pre-scheduling is disabled so that baseline
    and RL runs are comparable.
    """
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    return config


def evaluate_policy_dir(policy_dir: str, scenarios: List[str],
                        n_blocks: int = N_BLOCKS, seed: int = 42) -> dict:
    """Evaluate a directory of trained per-agent policies over scenarios.

    Runs entirely in memory (writes no files). For each scenario a truthful
    baseline episode and a combined-fleet episode are cleared, and the return
    mirrors the "ALL" rows the CLI writes to CSV. Best-response regret is not
    evaluated here.

    Parameters
    ----------
    policy_dir : str  Directory of {agent_name}.pt policy files.
    scenarios : list of str
    n_blocks : int  Blocks per episode; reducing it cuts runtime roughly
        linearly (a partial-day window with identical envs on both sides).
    seed : int  Accepted for parity with the CLI; episodes are deterministic
        given the scenario RNG state.

    Returns
    -------
    dict with keys policy_dir, loaded, obs_spec_name and scenarios (one entry
    per scenario). Each scenario entry carries the aggregate columns
    baseline_profit / rl_profit / profit_delta / welfare_baseline /
    welfare_rl / welfare_delta / valuation_artifact / genuine_welfare_delta /
    re_rate_baseline / re_rate_rl and n_rl (matched policies).
    """
    config = default_eval_config()

    # Recover the observation/action spec from a scratch baseline env so policy
    # files are validated against the same spec they were saved with.
    scratch_agents, _ = get_scenario("baseline", T=96,
                                     config=copy.deepcopy(config))
    scratch_env = BiddingEnv(scratch_agents, config)
    obs_dim = scratch_env.get_state_dim()
    obs_spec = scratch_env.obs_spec
    action_spec = scratch_env.action_spec
    bounds = scratch_env.get_action_bounds()
    bid_mult_low = float(bounds[0, 0])
    bid_mult_high = float(bounds[1, 0])

    policies = load_policies_from_dir(policy_dir, obs_dim, bounds,
                                      obs_spec=obs_spec,
                                      action_spec=action_spec)
    if not policies:
        raise ValueError(
            f"No compatible policy files loaded from '{policy_dir}' "
            f"(obs spec '{getattr(obs_spec, 'name', '?')}').")

    out = {
        "policy_dir": policy_dir,
        "loaded": len(policies),
        "obs_spec_name": getattr(obs_spec, "name", ""),
        "scenarios": [],
    }
    for sc_name in scenarios:
        cfg = copy.deepcopy(config)
        agents, _ = get_scenario(sc_name, T=96, config=cfg)
        present = {a.name for a in agents}
        rl_names = [nm for nm in policies if nm in present]
        if not rl_names:
            raise ValueError(
                f"No policy agent is present in scenario '{sc_name}'. "
                f"Loaded agents: {list(policies)}")

        # Truthful baseline episode.
        env_base = BiddingEnv(agents, cfg,
                              bid_mult_low=bid_mult_low,
                              bid_mult_high=bid_mult_high)
        base = run_episode(env_base, policy=None, n_blocks=n_blocks)

        # Combined-fleet episode (all matched policies act together).
        env_comb = BiddingEnv(agents, cfg, rl_agent_names=rl_names,
                              bid_mult_low=bid_mult_low,
                              bid_mult_high=bid_mult_high)
        comb = run_combined_episode(env_comb, policies, n_blocks=n_blocks)

        artifact = valuation_artifact(comb["sched"], comb["declared"],
                                      agents, cfg)
        welfare_delta = comb["welfare"] - base["welfare"]
        total_base_profit = sum(base["profits"].get(nm, 0.0)
                                for nm in rl_names)
        total_rl_profit = sum(comb["profits"].get(nm, 0.0)
                              for nm in rl_names)
        out["scenarios"].append({
            "scenario": sc_name,
            "n_rl": len(rl_names),
            "baseline_profit": total_base_profit,
            "rl_profit": total_rl_profit,
            "profit_delta": total_rl_profit - total_base_profit,
            "welfare_baseline": base["welfare"],
            "welfare_rl": comb["welfare"],
            "welfare_delta": welfare_delta,
            "valuation_artifact": artifact,
            "genuine_welfare_delta": welfare_delta - artifact,
            "re_rate_baseline": base["re_rate"],
            "re_rate_rl": comb["re_rate"],
            "n_blocks": n_blocks,
        })
    return out


def run_regret_test(agents, config, declared: dict, T: int,
                    output_path: str = None) -> dict:
    """Run a best-response regret test against a trained bid profile.

    Mirrors marl_clearing_and_bidding's test_for_ne_shared: each agent's
    best-response payoff is compared with its payoff under the trained fleet
    (declared), and the per-agent regret (best - base) is aggregated. Writes
    final_regret.csv and returns compute_regret_summary.

    Note: one best-response search per agent, so this is an explicit,
    potentially slow post-training step.
    """
    from nash import NashEquilibriumTester, compute_regret_summary

    base_strategy = {}
    for a in agents:
        if a.name in declared:
            bid = np.asarray(declared[a.name]["bid_mult"], dtype=float)
            offer = np.asarray(declared[a.name].get("offer_adder",
                                                    np.zeros(T)),
                               dtype=float)
        else:
            bid = np.full(T, 1.0, dtype=float)
            offer = np.full(T, 0.0, dtype=float)
        base_strategy[a.name] = {"bid_mult": bid, "offer_adder": offer}

    # On Windows the tester runs serial (parallel defaults to os.name != 'nt')
    # and the COBYLA best-response search costs ~4-8h for the full fleet.
    # Sampling-based BR with a process Pool gives a tractable first-pass
    # regret check; the exhaustive COBYLA path stays available by
    # constructing NashEquilibriumTester directly.
    tester = NashEquilibriumTester(agents, config, T=T, stage="DA",
                                   use_optimization=False, parallel=True)
    is_nash, improvements = tester.test_nash_equilibrium(base_strategy)
    summary = compute_regret_summary(improvements)

    import csv
    rows = [{
        "agent": nm,
        "base_payoff": imp["base_payoff"],
        "best_payoff": imp["best_payoff"],
        "regret": imp["regret"],
        "relative_regret": imp["relative_regret"],
        "profitable": imp["profitable"],
    } for nm, imp in improvements.items()]
    path = output_path or "final_regret.csv"
    with open(path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["agent"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nRegret test: {'NASH' if is_nash else 'NOT NASH'} | "
          f"total_regret={summary['total_regret']:+.1f} | "
          f"profitable={summary['n_profitable']}/{len(improvements)}")
    print(f"Regret written to: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL bidding policies")
    parser.add_argument("--policies", type=str, required=True,
                        help="Path to a policy .pt file or a directory of "
                             ".pt files")
    parser.add_argument("--scenarios", type=str, default="baseline",
                        help="Comma-separated scenario names or 'all'")
    parser.add_argument("--bid-mult-low", type=float, default=0.3,
                        help="Lower bound for bid_mult action space")
    parser.add_argument("--bid-mult-high", type=float, default=1.8,
                        help="Upper bound for bid_mult action space")
    parser.add_argument("--combined-only", action="store_true",
                        help="Only run the combined all-policies evaluation, "
                             "skipping per-agent runs")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Load {name}_ckpt_{N}.pt from --policies directory "
                             "instead of the final {name}.pt files "
                             "(requires --policies to be a directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path (default: print to console)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="Number of evaluation episodes per (policy, "
                             "scenario). Each episode draws a different "
                             "wholesale-price day and results are averaged. "
                             "With >1 the truthful baseline and the RL fleet "
                             "are re-seeded with the same seed per episode, so "
                             "both clear against the same price day (paired).")
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="Seed for multi-episode evaluation. When set, "
                             "episode k re-seeds numpy with eval_seed + k "
                             "before the baseline and again before the RL run, "
                             "pairing them on the same wholesale price day. "
                             "Default: None keeps the legacy unseeded behavior "
                             "for a single episode.")
    parser.add_argument("--nash-regret", action="store_true",
                        help="After the combined fleet evaluation, run a "
                             "best-response regret test against the trained "
                             "bid profile and write final_regret.csv (may be "
                             "slow: one best-response search per agent)")
    parser.add_argument("--consumer-metrics", action="store_true",
                        help="Also compute and report the three-layer "
                             "accounting metrics (consumer surplus / consumer "
                             "payment / LMP markup / fleet market-power "
                             "arbitrage split) on the combined fleet row. "
                             "Appends cs_*/cp_*/lmp_markup_*/market_power_* "
                             "columns to the CSV.")
    parser.add_argument("--capacity", type=float, default=None,
                        help="Override network.line_capacity_multiplier for "
                             "the evaluated network (scenarios otherwise pin "
                             "it to 1.5).")
    args = parser.parse_args()

    # ---- Determine scenarios ----
    if args.scenarios == "all":
        eval_scenarios = list_scenarios()
    else:
        eval_scenarios = [s.strip() for s in args.scenarios.split(",")]

    # ---- Build config ----
    config = default_eval_config()

    # ---- Load policies ----
    action_bounds = torch.tensor(
        [[args.bid_mult_low, 0.0], [args.bid_mult_high, 50.0]],
        dtype=torch.float32)
    # Use a temp env to get obs_dim and the active observation/action spec
    temp_agents, _ = get_scenario("baseline", T=96, config=copy.deepcopy(config))
    temp_env = BiddingEnv(temp_agents, config)
    obs_dim = temp_env.get_state_dim()
    obs_spec = temp_env.obs_spec
    action_spec = temp_env.action_spec

    if os.path.isdir(args.policies):
        policies = load_policies_from_dir(
            args.policies, obs_dim, action_bounds, checkpoint=args.checkpoint,
            obs_spec=obs_spec, action_spec=action_spec)
    else:
        if args.checkpoint is not None:
            print("--checkpoint requires --policies to be a directory")
            return
        policy = load_policy(args.policies, obs_dim, action_bounds,
                             obs_spec=obs_spec, action_spec=action_spec)
        # Extract agent name from filename
        fname = os.path.splitext(os.path.basename(args.policies))[0]
        if "_ckpt_" in fname:
            # Parse "Agent_X_ckpt_N" → "Agent_X"
            parts = fname.rsplit("_ckpt_", 1)
            agent_name = parts[0]
        else:
            agent_name = fname
        policies = {agent_name: policy}

    if not policies:
        print("No policies loaded. Check --policies path.")
        return

    print(f"\nLoaded {len(policies)} policy(s): {list(policies.keys())}")
    print(f"Scenarios: {eval_scenarios}\n")

    # ---- Evaluate ----
    results = []
    header = ["scenario", "agent", "baseline_profit", "rl_profit",
              "profit_delta", "welfare_baseline", "welfare_rl",
              "welfare_delta", "valuation_artifact", "genuine_welfare_delta",
              "re_rate_baseline", "re_rate_rl", "total_regret"]
    if args.consumer_metrics:
        header += _CONSUMER_COLUMNS

    # Multi-episode paired evaluation protocol: for each scenario and episode
    # the truthful baseline and the RL fleet are re-seeded with the SAME seed
    # so both clear against the same wholesale-price day; episodes then sweep
    # the seed to average out wholesale-day noise. Single-episode runs without
    # --eval-seed keep the legacy unseeded path unchanged.
    n_eps = max(1, args.eval_episodes)
    paired = (args.eval_seed is not None) or n_eps > 1
    base_seed = args.eval_seed if args.eval_seed is not None else args.seed

    _regret_done = False  # regret test runs once, on the first combined fleet
    for sc_name in eval_scenarios:
        print(f"--- {sc_name} ---")
        config_copy = copy.deepcopy(config)
        agents, wholesale = get_scenario(sc_name, T=96, config=config_copy)
        if args.capacity is not None:
            # Re-apply after get_scenario: scenarios re-pin capacity to 1.5.
            config_copy.network.line_capacity_multiplier = args.capacity
        agent_names = [a.name for a in agents]
        rl_names = [n for n in policies if n in agent_names]

        env_base = BiddingEnv(agents, config_copy,
                              bid_mult_low=args.bid_mult_low,
                              bid_mult_high=args.bid_mult_high)
        env_comb = None
        if len(rl_names) > 1:
            env_comb = BiddingEnv(agents, config_copy,
                                  rl_agent_names=rl_names,
                                  bid_mult_low=args.bid_mult_low,
                                  bid_mult_high=args.bid_mult_high)

        # Per-episode rows, averaged per agent/ALL across episodes below.
        rows_ep: list = []
        last_declared = None

        for k in range(n_eps):
            S = base_seed + k if paired else None
            if S is not None:
                np.random.seed(S)

            # ---- Baseline run (all fixed) for this price day ----
            base_result = run_episode(env_base, policy=None,
                                      capture=args.consumer_metrics)
            if k == 0:
                print(f"  Baseline: welfare={base_result['welfare']:.0f}  "
                      f"RE={base_result['re_rate']:.1f}%")

            # ---- RL run (with trained policies) ----
            # If multiple policies, run them separately or combined.
            # --combined-only skips per-agent runs; the fleet run below is
            # then the sole output.
            if not args.combined_only:
                for pol_agent_name, pol_net in policies.items():
                    # Check if this agent exists in the scenario
                    if pol_agent_name not in agent_names:
                        if k == 0:
                            print(f"  Agent {pol_agent_name} not in scenario "
                                  f"{sc_name}, skipping")
                        continue

                    # Re-seed so each per-agent run pairs with the same-price
                    # baseline day as the combined run.
                    if S is not None:
                        np.random.seed(S)
                    env_rl = BiddingEnv(agents, config_copy,
                                        rl_agent_names=[pol_agent_name],
                                        bid_mult_low=args.bid_mult_low,
                                        bid_mult_high=args.bid_mult_high)
                    rl_result = run_episode(env_rl, policy=pol_net)
                    if k == 0:
                        print(f"  RL ({pol_agent_name}): "
                              f"welfare={rl_result['welfare']:.0f}  "
                              f"RE={rl_result['re_rate']:.1f}%")

                    base_profit = base_result["profits"].get(pol_agent_name, 0.0)
                    rl_profit = rl_result["profits"].get(pol_agent_name, 0.0)
                    profit_delta = rl_profit - base_profit
                    welfare_delta = rl_result["welfare"] - base_result["welfare"]
                    re_delta = rl_result["re_rate"] - base_result["re_rate"]
                    if k == 0:
                        print(f"    Profit: baseline={base_profit:.1f}  "
                              f"RL={rl_profit:.1f}  delta={profit_delta:+.1f}  "
                              f"welfare_delta={welfare_delta:+.0f}  "
                              f"RE_delta={re_delta:+.1f}pp")

                    rows_ep.append({
                        "agent": pol_agent_name,
                        "baseline_profit": base_profit, "rl_profit": rl_profit,
                        "profit_delta": profit_delta,
                        "welfare_baseline": base_result["welfare"],
                        "welfare_rl": rl_result["welfare"],
                        "welfare_delta": welfare_delta,
                        "re_rate_baseline": base_result["re_rate"],
                        "re_rate_rl": rl_result["re_rate"],
                    })

            # ---- Combined run (all trained policies together) ----
            if env_comb is not None:
                if S is not None:
                    np.random.seed(S)
                comb = run_combined_episode(env_comb, policies,
                                            capture=args.consumer_metrics)
                comb_welfare = comb["welfare"]
                comb_re_rate = comb["re_rate"]
                comb_profits = comb["profits"]
                comb_welfare_delta = comb_welfare - base_result["welfare"]
                comb_re_delta = comb_re_rate - base_result["re_rate"]
                artifact = valuation_artifact(comb["sched"], comb["declared"],
                                              agents, config_copy)
                genuine = comb_welfare_delta - artifact
                if k == 0:
                    print(f"  Combined ({len(rl_names)} policies): "
                          f"welfare_delta={comb_welfare_delta:+.0f}  "
                          f"artifact={artifact:+.0f}  "
                          f"genuine={genuine:+.0f}  "
                          f"RE_delta={comb_re_delta:+.1f}pp")
                    for nm in rl_names:
                        print(f"    {nm}: profit={comb_profits[nm]:.1f}  "
                              f"delta={comb_profits[nm] - base_result['profits'].get(nm, 0):+.1f}")

                # Three-layer accounting metrics (only on the combined fleet,
                # computed on the captured full-day schedule/LMP/wholesale).
                metric = {}
                if args.consumer_metrics and comb.get("lmp") is not None \
                        and base_result.get("lmp") is not None:
                    cm_b = consumer_metrics(base_result["sched"],
                                            base_result["lmp"], agents)
                    cm_r = consumer_metrics(comb["sched"], comb["lmp"], agents)
                    mk_b = load_payment_weighted_markup(
                        base_result["sched"], base_result["lmp"],
                        base_result["wholesale"], agents)
                    mk_r = load_payment_weighted_markup(
                        comb["sched"], comb["lmp"], comb["wholesale"], agents)
                    splits = market_power_split(
                        base_result["sched"], comb["sched"],
                        base_result["lmp"], comb["lmp"], agents, config_copy)
                    mk_delta = (mk_r - mk_b) if (not np.isnan(mk_r)
                                                 and not np.isnan(mk_b)) \
                        else None
                    metric = {
                        "cs_baseline": cm_b["cs"], "cs_rl": cm_r["cs"],
                        "cs_delta": cm_r["cs"] - cm_b["cs"],
                        "cp_baseline": cm_b["cp"], "cp_rl": cm_r["cp"],
                        "cp_delta": cm_r["cp"] - cm_b["cp"],
                        "lmp_markup_baseline": mk_b, "lmp_markup_rl": mk_r,
                        "lmp_markup_delta": mk_delta,
                        "market_power_arb": sum(s["arb"] for s in splits),
                        "market_power_power": sum(
                            s["market_power"] for s in splits),
                    }
                    if k == 0:
                        print(f"    CS delta={metric['cs_delta']:+.1f}  "
                              f"CP delta={metric['cp_delta']:+.1f}  "
                              f"markup {mk_b:.4f}->{mk_r:.4f}  "
                              f"fleet arb/power="
                              f"{metric['market_power_arb']:+.1f}/"
                              f"{metric['market_power_power']:+.1f}")
                        for s in splits:
                            print(f"      {s['name']}: dp={s['profit_delta']:+.1f}"
                                  f" arb={s['arb']:+.1f} "
                                  f"power={s['market_power']:+.1f} "
                                  f"other={s['other']:+.1f} net={s['net']:+.1f}")

                # Aggregate combined row for the CSV
                total_base_profit = sum(base_result["profits"].get(nm, 0.0)
                                        for nm in rl_names)
                total_rl_profit = sum(comb_profits[nm] for nm in rl_names)
                rows_ep.append({
                    "agent": "ALL",
                    "baseline_profit": total_base_profit,
                    "rl_profit": total_rl_profit,
                    "profit_delta": total_rl_profit - total_base_profit,
                    "welfare_baseline": base_result["welfare"],
                    "welfare_rl": comb_welfare,
                    "welfare_delta": comb_welfare_delta,
                    "valuation_artifact": artifact,
                    "genuine_welfare_delta": genuine,
                    "re_rate_baseline": base_result["re_rate"],
                    "re_rate_rl": comb_re_rate,
                    **metric,
                })
                last_declared = comb["declared"]

        # ---- Average per-episode rows into one row per agent / ALL ----
        by_agent: dict = {}
        for r in rows_ep:
            by_agent.setdefault(r["agent"], []).append(r)
        num_keys = ["baseline_profit", "rl_profit", "profit_delta",
                    "welfare_baseline", "welfare_rl", "welfare_delta",
                    "valuation_artifact", "genuine_welfare_delta",
                    "re_rate_baseline", "re_rate_rl"]
        if args.consumer_metrics:
            num_keys += _CONSUMER_COLUMNS
        for agent, rows in by_agent.items():
            agg = {"scenario": sc_name, "agent": agent,
                   "total_regret": ""}
            for key in num_keys:
                vals = [r[key] for r in rows if r.get(key) is not None]
                agg[key] = float(np.mean(vals)) if vals else None
            results.append(agg)

        # Optional best-response regret test (once, explicit opt-in).
        if args.nash_regret and not _regret_done and last_declared is not None:
            _regret_done = True
            out_dir = os.path.dirname(args.output) if args.output else "."
            regret_path = os.path.join(out_dir, "final_regret.csv")
            regret = run_regret_test(agents, config_copy, last_declared,
                                     T, output_path=regret_path)
            for r in results:
                if r.get("scenario") == sc_name and r.get("agent") == "ALL":
                    r["total_regret"] = regret["total_regret"]

    # ---- Output ----
    if args.output and results:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")
    elif results:
        print(f"\nSummary: {len(results)} evaluations across "
              f"{len(eval_scenarios)} scenario(s)")


if __name__ == "__main__":
    main()
