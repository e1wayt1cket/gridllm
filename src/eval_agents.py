# eval_agents.py
"""Evaluate trained RL policies against the truthful-declaration baseline.

Load any combination of trained policy files and run market simulations to
measure:
  - Per-agent profit vs baseline
  - Clearing-objective change
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

import money
import participant_payoff

from scenarios import get_scenario, list_scenarios
from models import MarketConfig
from rl_env import BiddingEnv, DayCapture, N_BLOCKS, BLOCK_SIZE
from rl_td3 import Actor, load_policy
from rl_spec import action_spec_for
from rl_profit_diagnostics import valuation_artifact
from counterfactual_eval import settle_day, summarize as summarize_payoff
from surplus_metrics import consumer_metrics, day_econ_metrics, \
    DAY_ECON_COLUMNS

# CSV columns appended to the combined (ALL) row under --consumer-metrics.
# The names belong to the shared day-metric vocabulary so the evaluation CSV
# and the per-episode training log can be read as one schema.
_CONSUMER_COLUMNS = list(DAY_ECON_COLUMNS)

#: What the evaluation CSV's two families of money column mean. Recorded here
#: because both appear in the same file and only one of them is a research
#: result; a reader of `results/*_eval.csv` should not have to recover the
#: distinction from the source.
#:
#:   `*_profit`            The day's **settled payoff**: the whole horizon
#:                         re-cleared at once with both state-of-charge
#:                         endpoints pinned, priced at the nodal LMP, from the
#:                         quotes the policy actually declared. Computed by
#:                         `counterfactual_eval.settle_day`, the same call
#:                         `evaluate_policy_dir` makes, so this file and the
#:                         programmatic evaluator agree by construction.
#:   `*_rolling_reward`    The episode's accumulated per-block reward. A
#:                         **learning signal**, not a payoff. It is NOT
#:                         endpoint-pinned, so a policy that runs its battery
#:                         down collects more cash across blocks and scores
#:                         higher here. Never quote it as an economic result.
#:   `settled_clean`       Whether both arms settled without a degraded or
#:                         fallen-back clear. A row that is not clean reports
#:                         NaN profits rather than zeros, and is not a result.
MONEY_COLUMN_MEANINGS = {
    "baseline_profit": "settled payoff of the truthful arm",
    "rl_profit": "settled payoff of the policy arm",
    "profit_delta": ("settled payoff difference, rl_profit - baseline_profit; "
                     "the research KPI"),
    "baseline_rolling_reward": "training signal, not a payoff",
    "rl_rolling_reward": "training signal, not a payoff",
    "rolling_reward_delta": "training signal, not a payoff",
}


def _n_commit(env, t_start: int) -> int:
    """Committed periods of the rolling window starting at t_start."""
    return DayCapture.committed_periods_for(env, t_start)


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
    return participant_payoff.participant_payoff(
        schedule, lmp_node, agent, config, n_periods).total


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
    dict with keys: clearing_objective, re_rate, carbon, profits (per-agent
    dict), plus sched/lmp/wholesale when capture is True.
    """
    obs = env.reset()
    total_clearing_objective = 0.0
    total_re_rate = 0.0
    agent_profits: Dict[str, float] = {}
    cap = DayCapture.from_env(env) if capture else None

    for block in range(n_blocks):
        t_start = block * BLOCK_SIZE
        if policy is not None:
            # RL agent uses trained policy; others use defaults from env
            agent_name = env.rl_agents[0].name
            action = actor_predict(policy, obs[agent_name])
            acts = {agent_name: action}
        else:
            # Baseline: every agent stays at the truthful declaration reset()
            # already installed, so no action is sent. A literal pair here
            # would be read positionally as a storage unit's two prices, and
            # `(1.0, 0.0)` is not the truthful point of that box.
            acts = {}

        next_obs, rewards, done, info = env.step(acts)
        total_clearing_objective += info.get("clearing_objective", 0.0)
        total_re_rate = info.get("re_rate", 0.0)  # last block's value

        if cap is not None:
            cap.add_block(env._last_result, t_start, env._last_wholesale)
            # Captured for the baseline arm as well, so both arms are settled
            # from their own declared quotes by the same code. For this arm
            # they are the env's default (truthful) declaration, which reset()
            # already installed.
            cap.add_declared(env.current_actions, t_start)

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
    out = {"clearing_objective": total_clearing_objective,
           "re_rate": total_re_rate,
           "carbon": carbon,
           "profits": agent_profits}
    if cap is not None:
        out.update(cap.finish())
        out["capture_complete"] = out["complete"]
    return out


def settle_capture(agents, config, rl_names, capture: dict, T: int = 96):
    """Settle one captured arm the way the paired evaluator does.

    `capture` is a `run_episode(..., capture=True)` result. Returns
    `(per-agent settled profit, clean)`.

    This calls the same `counterfactual_eval.settle_day` that
    `evaluate_policy_dir` calls, on the same inputs, so the CLI and the
    programmatic entry point cannot publish two different payoffs for the same
    day under the same column name. Before this existed the CLI reported the
    episode's accumulated per-block rewards instead -- a quantity that rewards
    a policy for ending the day with an emptier battery, because it collects
    cash across blocks and is never corrected for the energy it banked.

    A capture that is not `usable` (a fallen-back or failed block) settles to
    NaN rather than to zeros: a degraded clear is reported as degraded, never
    averaged in as a payoff.
    """
    if not capture or not capture.get("usable"):
        return {nm: float("nan") for nm in rl_names}, False
    profits, clean, _ = settle_day(agents, config, rl_names,
                                   capture["declared"], capture["wholesale"], T)
    if not clean:
        return {nm: float("nan") for nm in rl_names}, False
    return profits, True


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
      profits             per-rl-agent accumulated reward
      clearing_objective  accumulated clearing objective
      re_rate             renewable consumption rate from the last block
      sched    per-agent committed schedules padded to length env.T
      declared per-rl-agent quotes padded to length env.T, in each agent's
               own action space
    """
    obs = env.reset()
    rl_names = [a.name for a in env.rl_agents]
    profits: Dict[str, float] = {nm: 0.0 for nm in rl_names}
    clearing_objective = 0.0
    re_rate = 0.0
    cap = DayCapture.from_env(env)
    for block in range(n_blocks):
        t_start = block * BLOCK_SIZE
        acts = {nm: actor_predict(policies[nm], obs[nm]) for nm in rl_names}
        next_obs, rewards, done, info = env.step(acts)
        cap.add_block(env._last_result, t_start, env._last_wholesale)
        cap.add_declared(env.current_actions, t_start)
        clearing_objective += info.get("clearing_objective", 0.0)
        re_rate = info.get("re_rate", 0.0)
        for nm in rl_names:
            profits[nm] = profits.get(nm, 0.0) + rewards.get(nm, 0.0)
        if done:
            break
        obs = next_obs
    out = {"profits": profits, "clearing_objective": clearing_objective,
           "re_rate": re_rate}
    out.update(cap.finish())
    out["capture_complete"] = out["complete"]
    if not capture:
        # Capture is cheap (bookkeeping over already-solved blocks) and the
        # caller may need the declared actions regardless; only the bulky
        # day arrays are dropped.
        for key in ("sched", "lmp", "wholesale", "declared"):
            out.pop(key, None)
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
                        n_blocks: int = N_BLOCKS, seed: int = 42,
                        checkpoint: Optional[int] = None,
                        consumer_metrics: bool = False) -> dict:
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
    seed : int
        Paired-evaluation seed. The baseline and the RL fleet are re-seeded
        with the same value before each episode so both clear against the
        identical wholesale-price day; without that, a delta mixes the
        strategy effect with price-day noise.
    checkpoint : int or None
        Load {agent_name}_ckpt_{N}.pt instead of the final {agent_name}.pt
        files, for comparing checkpoints of one run.
    consumer_metrics : bool
        Also compute the three-layer day metrics (cs_*/cp_*/lmp_markup_*/
        market_power_*) for each scenario.

    Returns
    -------
    dict with keys policy_dir, loaded, obs_spec_name and scenarios (one entry
    per scenario). Each scenario entry carries the aggregate columns
    baseline_profit / rl_profit / profit_delta / clearing_objective_baseline /
    clearing_objective_rl / clearing_objective_delta / valuation_artifact / genuine_clearing_objective_delta /
    re_rate_baseline / re_rate_rl and n_rl (matched policies), plus the
    consumer columns when consumer_metrics is set.
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

    policies = load_policies_from_dir(policy_dir, obs_dim, bounds,
                                      checkpoint=checkpoint,
                                      obs_spec=obs_spec,
                                      action_spec=action_spec)
    if not policies:
        which = f"checkpoint {checkpoint}" if checkpoint is not None \
            else "final policies"
        raise ValueError(
            f"No compatible policy files loaded from '{policy_dir}' "
            f"({which}; obs spec '{getattr(obs_spec, 'name', '?')}'). "
            f"The directory may lack {which} — check its available roles.")

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

        # Truthful baseline episode. Both episodes are re-seeded with the same
        # value: grid.day_ahead_price_china adds unseeded noise, so without
        # this the two clears see different wholesale days and every delta
        # carries that spread.
        np.random.seed(seed)
        env_base = BiddingEnv(agents, cfg)
        # Capture always: both arms are re-settled from their captured declared
        # quotes below, so the baseline's capture is not only for the consumer
        # columns.
        base = run_episode(env_base, policy=None, n_blocks=n_blocks,
                           capture=True)

        # Combined-fleet episode (all matched policies act together).
        np.random.seed(seed)
        env_comb = BiddingEnv(agents, cfg, rl_agent_names=rl_names)
# The objective artifact is revalued from the captured day schedule, so
        # the combined run must always capture it; `consumer_metrics`
        # controls only the consumer accounting columns below.
        comb = run_combined_episode(env_comb, policies, n_blocks=n_blocks,
                                    capture=True)

        artifact = valuation_artifact(comb["sched"], comb["declared"],
                                      agents, cfg)
        clearing_objective_delta = comb["clearing_objective"] - base["clearing_objective"]

        # Both arms settled the same way: one joint clear of the whole day with
        # the state of charge pinned at both ends. The block-by-block sums the
        # episodes carry are a *learning* signal, not a payoff -- a policy that
        # runs its battery down collects more cash across blocks and is not
        # corrected for it there, so reporting them as profit let a policy win
        # by ending the day emptier. Reused from counterfactual_eval so the two
        # reported evaluations cannot settle the same day differently.
        base_settled, base_clean, _ = settle_day(
            agents, cfg, rl_names, base["declared"], base["wholesale"], 96)
        comb_settled, comb_clean, _ = settle_day(
            agents, cfg, rl_names, comb["declared"], comb["wholesale"], 96)
        settled_clean = base_clean and comb_clean
        if settled_clean:
            total_base_profit = sum(base_settled.get(nm, 0.0) for nm in rl_names)
            total_rl_profit = sum(comb_settled.get(nm, 0.0) for nm in rl_names)
        else:
            # A degraded settlement is reported as degraded, never averaged in
            # as if it were a payoff.
            total_base_profit = float("nan")
            total_rl_profit = float("nan")
        row = {
            "scenario": sc_name,
            "n_rl": len(rl_names),
            "settled_clean": settled_clean,
            "baseline_profit": total_base_profit,
            "rl_profit": total_rl_profit,
            "profit_delta": total_rl_profit - total_base_profit,
            "clearing_objective_baseline": base["clearing_objective"],
            "clearing_objective_rl": comb["clearing_objective"],
            "clearing_objective_delta": clearing_objective_delta,
            "valuation_artifact": artifact,
            "genuine_clearing_objective_delta": clearing_objective_delta - artifact,
            "re_rate_baseline": base["re_rate"],
            "re_rate_rl": comb["re_rate"],
            "n_blocks": n_blocks,
            "checkpoint": checkpoint,
        }
        if consumer_metrics:
            try:
                row.update(day_econ_metrics(base, comb, agents, cfg))
            except ValueError:
                # Incomplete capture (a fallback or failed block): report the
                # gap rather than a number built on zero-filled periods.
                row.update({c: None for c in DAY_ECON_COLUMNS})
        out["scenarios"].append(row)
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
        spec = action_spec_for(a)
        d = declared.get(a.name)
        # The profile under test is whatever the policies declared, in each
        # agent's own action space; an agent with no declaration sits at its
        # spec's truthful point rather than at a literal written here.
        base_strategy[a.name] = {
            n: (np.asarray(d[n], dtype=float) if d is not None
                else np.full(T, spec.truthful[n], dtype=float))
            for n in spec.action_names
        }

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
    # The box a policy was trained in is the one its action spec declares, so
    # it is read from the spec rather than assembled from flags. `load_policy`
    # checks the spec name, but a checkpoint whose spec matches and whose box
    # does not would be rescaled silently.
    temp_agents, _ = get_scenario("baseline", T=96, config=copy.deepcopy(config))
    temp_env = BiddingEnv(temp_agents, config)
    obs_dim = temp_env.get_state_dim()
    obs_spec = temp_env.obs_spec
    action_spec = temp_env.action_spec
    action_bounds = temp_env.get_action_bounds()

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
              "profit_delta", "settled_clean",
              "baseline_rolling_reward", "rl_rolling_reward",
              "rolling_reward_delta",
              "clearing_objective_baseline", "clearing_objective_rl",
              "clearing_objective_delta", "valuation_artifact", "genuine_clearing_objective_delta",
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

        env_base = BiddingEnv(agents, config_copy)
        env_comb = None
        if len(rl_names) > 1:
            env_comb = BiddingEnv(agents, config_copy,
                                  rl_agent_names=rl_names)

        # Per-episode rows, averaged per agent/ALL across episodes below.
        rows_ep: list = []
        last_declared = None

        for k in range(n_eps):
            S = base_seed + k if paired else None
            if S is not None:
                np.random.seed(S)

            # ---- Baseline run (all fixed) for this price day ----
            # Capture always: this arm is settled from its own captured
            # declarations below, the same way `evaluate_policy_dir` settles it,
            # so the CLI's profit columns are the same quantity the API
            # reports. `consumer_metrics` now controls only the consumer
            # accounting columns.
            base_result = run_episode(env_base, policy=None, capture=True)
            # Settled once for the whole matched set. The baseline arm declares
            # truthfully for every agent, so naming a subset would clear
            # identically; one settlement therefore serves both the per-agent
            # and the combined rows.
            base_settled, base_settled_clean = settle_capture(
                agents, config_copy, rl_names, base_result)
            if k == 0:
                print(f"  Baseline: objective={base_result['clearing_objective']:.0f}  "
                      f"RE={base_result['re_rate']:.1f}%  "
                      f"settled={sum(base_settled.values()):+.1f}")

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
                                        rl_agent_names=[pol_agent_name])
                    rl_result = run_episode(env_rl, policy=pol_net, capture=True)
                    rl_settled, rl_settled_clean = settle_capture(
                        agents, config_copy, [pol_agent_name], rl_result)
                    if k == 0:
                        print(f"  RL ({pol_agent_name}): "
                              f"objective={rl_result['clearing_objective']:.0f}  "
                              f"RE={rl_result['re_rate']:.1f}%")

                    # `profit` is the settled, endpoint-pinned payoff -- the same
                    # quantity `evaluate_policy_dir` reports under this name. The
                    # episode's accumulated per-block reward is reported beside
                    # it under its own name, because it is a learning signal and
                    # not a payoff.
                    base_profit = base_settled.get(pol_agent_name, float("nan"))
                    rl_profit = rl_settled.get(pol_agent_name, float("nan"))
                    profit_delta = rl_profit - base_profit
                    base_rolling = base_result["profits"].get(pol_agent_name, 0.0)
                    rl_rolling = rl_result["profits"].get(pol_agent_name, 0.0)
                    clearing_objective_delta = rl_result["clearing_objective"] - base_result["clearing_objective"]
                    re_delta = rl_result["re_rate"] - base_result["re_rate"]
                    if k == 0:
                        print(f"    Profit: baseline={base_profit:.1f}  "
                              f"RL={rl_profit:.1f}  delta={profit_delta:+.1f}  "
                              f"clearing_objective_delta={clearing_objective_delta:+.0f}  "
                              f"RE_delta={re_delta:+.1f}pp")

                    rows_ep.append({
                        "agent": pol_agent_name,
                        "baseline_profit": base_profit, "rl_profit": rl_profit,
                        "profit_delta": profit_delta,
                        "settled_clean": bool(base_settled_clean
                                              and rl_settled_clean),
                        "baseline_rolling_reward": base_rolling,
                        "rl_rolling_reward": rl_rolling,
                        "rolling_reward_delta": rl_rolling - base_rolling,
                        "clearing_objective_baseline": base_result["clearing_objective"],
                        "clearing_objective_rl": rl_result["clearing_objective"],
                        "clearing_objective_delta": clearing_objective_delta,
                        "re_rate_baseline": base_result["re_rate"],
                        "re_rate_rl": rl_result["re_rate"],
                    })

            # ---- Combined run (all trained policies together) ----
            if env_comb is not None:
                if S is not None:
                    np.random.seed(S)
                # Always capture: the objective artifact below is revalued from
                # the captured day schedule, and valuation_artifact cannot be
                # computed without it. `consumer_metrics` controls only the
                # consumer accounting columns.
                comb = run_combined_episode(env_comb, policies,
                                            capture=True)
                comb_settled, comb_settled_clean = settle_capture(
                    agents, config_copy, rl_names, comb)
                comb_clearing_objective = comb["clearing_objective"]
                comb_re_rate = comb["re_rate"]
                comb_profits = comb["profits"]
                comb_clearing_objective_delta = comb_clearing_objective - base_result["clearing_objective"]
                comb_re_delta = comb_re_rate - base_result["re_rate"]
                artifact = valuation_artifact(comb["sched"], comb["declared"],
                                              agents, config_copy)
                genuine = comb_clearing_objective_delta - artifact
                if k == 0:
                    print(f"  Combined ({len(rl_names)} policies): "
                          f"clearing_objective_delta={comb_clearing_objective_delta:+.0f}  "
                          f"artifact={artifact:+.0f}  "
                          f"genuine={genuine:+.0f}  "
                          f"RE_delta={comb_re_delta:+.1f}pp")
                    for nm in rl_names:
                        print(f"    {nm}: settled={comb_settled[nm]:.1f}  "
                              f"delta={comb_settled[nm] - base_settled.get(nm, float('nan')):+.1f}  "
                              f"(rolling={comb_profits[nm]:.1f})")

                # Three-layer accounting metrics (only on the combined fleet,
                # computed on the captured full-day schedule/LMP/wholesale).
                metric = {}
                if args.consumer_metrics and comb.get("lmp") is not None \
                        and base_result.get("lmp") is not None:
                    metric = day_econ_metrics(base_result, comb, agents,
                                              config_copy)
                    if k == 0:
                        print(f"    CS delta={metric['cs_delta']:+.1f}  "
                              f"CP delta={metric['cp_delta']:+.1f}  "
                              f"markup {metric['lmp_markup_baseline']:.4f}"
                              f"->{metric['lmp_markup_rl']:.4f}  "
                              f"fleet arb/power="
                              f"{metric['market_power_arb']:+.1f}/"
                              f"{metric['market_power_power']:+.1f}")

                # Aggregate combined row for the CSV. The `*_profit` columns are
                # the fleet's settled payoff; the rolling episode rewards ride
                # alongside under their own names.
                total_base_profit = sum(base_settled.get(nm, float("nan"))
                                        for nm in rl_names)
                total_rl_profit = sum(comb_settled.get(nm, float("nan"))
                                      for nm in rl_names)
                total_base_rolling = sum(base_result["profits"].get(nm, 0.0)
                                         for nm in rl_names)
                total_rl_rolling = sum(comb_profits[nm] for nm in rl_names)
                rows_ep.append({
                    "agent": "ALL",
                    "baseline_profit": total_base_profit,
                    "rl_profit": total_rl_profit,
                    "profit_delta": total_rl_profit - total_base_profit,
                    "settled_clean": bool(base_settled_clean
                                          and comb_settled_clean),
                    "baseline_rolling_reward": total_base_rolling,
                    "rl_rolling_reward": total_rl_rolling,
                    "rolling_reward_delta": total_rl_rolling - total_base_rolling,
                    "clearing_objective_baseline": base_result["clearing_objective"],
                    "clearing_objective_rl": comb_clearing_objective,
                    "clearing_objective_delta": comb_clearing_objective_delta,
                    "valuation_artifact": artifact,
                    "genuine_clearing_objective_delta": genuine,
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
                    "settled_clean",
                    "baseline_rolling_reward", "rl_rolling_reward",
                    "rolling_reward_delta",
                    "clearing_objective_baseline", "clearing_objective_rl", "clearing_objective_delta",
                    "valuation_artifact", "genuine_clearing_objective_delta",
                    "re_rate_baseline", "re_rate_rl"]
        if args.consumer_metrics:
            num_keys += _CONSUMER_COLUMNS
        for agent, rows in by_agent.items():
            agg = {"scenario": sc_name, "agent": agent,
                   "total_regret": ""}
            for key in num_keys:
                vals = [r[key] for r in rows if r.get(key) is not None]
                # The mean alone cannot answer whether a gain is stable: a mean
                # lifted by two lucky episodes and a share-near-half are
                # different findings. The distribution is reported alongside it
                # under the same `{quantity}_{stat}` spelling the paired
                # evaluator uses.
                if not vals:
                    agg[key] = None
                    continue
                stats = summarize_payoff(vals)
                agg[key] = stats["mean"]
                for stat, value in stats.items():
                    if stat != "mean":
                        agg[f"{key}_{stat}"] = value
            results.append(agg)

        # Optional best-response regret test (once, explicit opt-in).
        if args.nash_regret and not _regret_done and last_declared is not None:
            _regret_done = True
            out_dir = os.path.dirname(args.output) if args.output else "."
            regret_path = os.path.join(out_dir, "final_regret.csv")
            regret = run_regret_test(agents, config_copy, last_declared,
                                     T=96, output_path=regret_path)
            for r in results:
                if r.get("scenario") == sc_name and r.get("agent") == "ALL":
                    r["total_regret"] = regret["total_regret"]

    # ---- Output ----
    if args.output and results:
        import csv
        # The declared columns first, in their declared order, then whatever
        # the per-episode aggregation produced beyond them. The distribution
        # columns are named by `summarize_payoff`, so declaring them here means
        # declaring them twice and letting the two drift -- which is how this
        # write came to raise on its own output rather than emit it.
        fieldnames = list(header)
        for row in results:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")
    elif results:
        print(f"\nSummary: {len(results)} evaluations across "
              f"{len(eval_scenarios)} scenario(s)")


if __name__ == "__main__":
    main()
