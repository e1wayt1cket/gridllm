# nash.py
"""Nash equilibrium solver.

Methods implemented:
  - Fictitious play (FP): parallel across agents, exponentially-smoothed updates.
  - Diagonalization (Gauss-Seidel): sequential best response, immediate per-agent updates.
    Cited in electricity-market EPEC literature (Alidoust et al. 2024).
  - Jacobi (parallel diagonalization): all agents update simultaneously from a shared
    snapshot; useful as a faster but less stable alternative.

All methods use random-sampling best-response approximation via Gurobi market clearing.
"""

import numpy as np
import copy
from multiprocessing import Pool
from typing import Dict, List, Tuple
from models import Agent, MarketConfig
from market import clear_market


def _evaluate_payoffs(args):
    """Evaluate payoff for a single strategy variant (multiprocessing target)."""
    agents, config, T, stage, action_variant, target_name = args
    try:
        result = clear_market(agents, T, stage, action_variant, config)
    except Exception:
        return target_name, -1e12, None
    sched = result["schedules"][target_name]
    lmp = result["lmp"]
    bus = next(a.bus for a in agents if a.name == target_name)
    node_price = lmp[:, bus]
    sell = np.sum(sched["p_sell"] * node_price)
    buy = np.sum(sched["p_buy"] * node_price)
    penalty = np.sum(sched["unserved"] * config.penalty_unserved)
    payoff = sell - buy - penalty
    return target_name, float(payoff), action_variant.get(target_name)


def _evaluate_best_response(args):
    """Find best response for one agent: sample N variations, return best.
    Variants are evaluated serially to avoid nested multiprocessing pools
    that risk deadlock on Windows (spawn) when called via pool.map.
    """
    (agent, agents, config, T, stage, base_strategy, num_variations,
     exploration_scale) = args
    name = agent.name
    variants = _generate_variations(base_strategy, agent, config,
                                    num_variations, exploration_scale)

    best_pay = -1e12
    best_strat = copy.deepcopy(base_strategy[name])
    for var in variants:
        r_name, payoff, _ = _evaluate_payoffs(
            (agents, config, T, stage, var, name))
        if r_name == name and payoff > best_pay:
            best_pay = payoff
            best_strat = var[name]

    return name, best_pay, best_strat


def _generate_variations(base_strategy, agent, config, num_variations=5,
                         exploration_scale=None):
    """Generate perturbed strategy variants.

    exploration_scale: multiplier on perturbation std. Larger = wider search.
    If None, defaults to 1.0 for bid_mult sigma=0.03, offer sigma=2.0.
    """
    variants = []
    base_bid = np.array(base_strategy[agent.name]["bid_mult"])
    is_prosumer = agent.is_prosumer
    if is_prosumer:
        base_offer = np.array(base_strategy[agent.name]["offer_adder"])

    bid_sigma = 0.03 * (exploration_scale or 1.0)
    offer_sigma = 2.0 * (exploration_scale or 1.0)

    for _ in range(num_variations):
        new_strat = copy.deepcopy(base_strategy)
        if is_prosumer:
            bid_mult = np.clip(np.random.normal(base_bid, bid_sigma),
                               *config.bid_mult_range)
            offer_adder = np.clip(np.random.normal(base_offer, offer_sigma),
                                  *config.offer_adder_range)
            new_strat[agent.name] = {"bid_mult": bid_mult, "offer_adder": offer_adder}
        else:
            bid_mult = np.clip(np.random.normal(base_bid, bid_sigma),
                               *config.bid_mult_range)
            new_strat[agent.name] = {"bid_mult": bid_mult}
        variants.append(new_strat)
    return variants


def _strategy_distance(s1, s2, agent):
    """Compute normalized distance between two strategy dicts for one agent."""
    name = agent.name
    bid_diff = np.abs(np.array(s1[name]["bid_mult"]) - np.array(s2[name]["bid_mult"])).mean()
    if agent.is_prosumer:
        offer_diff = np.abs(np.array(s1[name]["offer_adder"]) - np.array(s2[name]["offer_adder"])).mean()
        return (bid_diff + offer_diff) / 2.0
    return bid_diff


def _blend_strategy(current, name, br_strat, is_prosumer, alpha):
    """Blend current strategy toward best response with weight alpha."""
    if is_prosumer:
        current[name]["bid_mult"] = (1 - alpha) * current[name]["bid_mult"] + alpha * br_strat["bid_mult"]
        current[name]["offer_adder"] = (1 - alpha) * current[name]["offer_adder"] + alpha * br_strat["offer_adder"]
    else:
        current[name]["bid_mult"] = (1 - alpha) * current[name]["bid_mult"] + alpha * br_strat["bid_mult"]


class NashEquilibriumTester:
    def __init__(self, agents, config, T=96, stage="DA"):
        self.agents = agents
        self.config = config
        self.T = T
        self.stage = stage

    # ------------------------------------------------------------------
    # Diagonalization (Gauss-Seidel sequential best response)
    # ------------------------------------------------------------------
    def diagonalization(self, init_strategy, max_iter=8, num_variations=5,
                        alpha=1.0, tol_distance=0.003, history=None):
        """Gauss-Seidel diagonalization.

        Iterates through agents sequentially. Each agent finds its best response
        given the *already-updated* strategies of previous agents in this round.

        alpha=1.0: pure Gauss-Seidel (full best-response adoption).
        alpha<1.0: damped Gauss-Seidel (safer but slower).

        If history is a list, per-iteration stats are appended to it:
          {"iter": N, "avg_payoff": float, "max_distance": float}.
        """
        current = copy.deepcopy(init_strategy)

        for it in range(max_iter):
            prev_round = copy.deepcopy(current)
            total_pay = 0.0

            for a in self.agents:
                br = _evaluate_best_response(
                    (a, self.agents, self.config, self.T, self.stage,
                     current, num_variations, None))
                name, payoff, strat = br
                total_pay += payoff
                _blend_strategy(current, name, strat, a.is_prosumer, alpha)

            avg_pay = total_pay / len(self.agents)
            max_dist = max(_strategy_distance(prev_round, current, a) for a in self.agents)

            print(f"  diagonalization {it+1}/{max_iter}: "
                  f"avg_pay={avg_pay:.1f}  max_dist={max_dist:.4f}")

            if history is not None:
                history.append({"iter": it + 1, "avg_payoff": avg_pay,
                                "max_distance": max_dist})

            if max_dist < tol_distance:
                print(f"  converged (strategy distance < {tol_distance})")
                return current, it + 1

        return current, max_iter

    # ------------------------------------------------------------------
    # Jacobi (parallel diagonalization)
    # ------------------------------------------------------------------
    def jacobi(self, init_strategy, max_iter=10, num_variations=5,
               alpha=0.6, tol_distance=0.003, history=None):
        """Jacobi parallel best response.

        All agents find best responses simultaneously from a shared snapshot,
        then all update at once. Faster per iteration than GS (parallel),
        but can oscillate — use alpha < 1.0 to damp.
        """
        current = copy.deepcopy(init_strategy)

        for it in range(max_iter):
            prev_snapshot = copy.deepcopy(current)

            # Evaluate all best responses in parallel from the snapshot
            tasks = [(a, self.agents, self.config, self.T, self.stage,
                      prev_snapshot, num_variations, None) for a in self.agents]
            with Pool(processes=min(len(self.agents), 4)) as pool:
                br_results = pool.map(_evaluate_best_response, tasks)

            payoffs = {}
            best_strategies = {}
            for name, payoff, strat in br_results:
                payoffs[name] = payoff
                best_strategies[name] = strat

            # Apply all updates simultaneously
            for a in self.agents:
                _blend_strategy(current, a.name, best_strategies[a.name],
                                a.is_prosumer, alpha)

            avg_pay = np.mean(list(payoffs.values()))
            max_dist = max(_strategy_distance(prev_snapshot, current, a) for a in self.agents)

            print(f"  jacobi {it+1}/{max_iter}: "
                  f"avg_pay={avg_pay:.1f}  max_dist={max_dist:.4f}")

            if history is not None:
                history.append({"iter": it + 1, "avg_payoff": avg_pay,
                                "max_distance": max_dist})

            if max_dist < tol_distance:
                print(f"  converged (strategy distance < {tol_distance})")
                return current, it + 1

        return current, max_iter

    # ------------------------------------------------------------------
    # Fictitious play (parallel, smoothed)  -- original + adaptive alpha
    # ------------------------------------------------------------------
    def iter_fictitious_play(self, init_strategy, max_iter=10, num_variations=5,
                             alpha=0.3, tol_relative=0.001, adaptive_alpha=True,
                             history=None):
        """Fictitious play with optional adaptive alpha.

        When adaptive_alpha=True, alpha starts at 0.7 and linearly decays to
        the final alpha over max_iter rounds. This accelerates early convergence
        while maintaining stability near equilibrium.
        """
        current = copy.deepcopy(init_strategy)
        prev_avg_pay = None

        for it in range(max_iter):
            # Adaptive alpha: start large, decay linearly
            if adaptive_alpha:
                r = it / max(1, max_iter - 1)  # 0 → 1
                alpha_t = 0.7 * (1 - r) + alpha * r
            else:
                alpha_t = alpha

            print(f"\n==========  fictitious play {it+1}/{max_iter}  "
                  f"(alpha={alpha_t:.3f}) ==========")

            tasks = [(a, self.agents, self.config, self.T, self.stage,
                      current, num_variations, None) for a in self.agents]
            with Pool(processes=min(len(self.agents), 4)) as pool:
                br_results = pool.map(_evaluate_best_response, tasks)

            payoffs = {}
            best_strategies = {}
            for name, payoff, strat in br_results:
                payoffs[name] = payoff
                best_strategies[name] = strat

            # Blend toward best response
            for a in self.agents:
                _blend_strategy(current, a.name, best_strategies[a.name],
                                a.is_prosumer, alpha_t)

            avg_pay = np.mean(list(payoffs.values()))
            print(f"  avg payoff: {avg_pay:.2f}")

            if history is not None:
                history.append({"iter": it + 1, "avg_payoff": avg_pay,
                                "max_distance": 0.0})

            if prev_avg_pay is not None and (abs(avg_pay - prev_avg_pay) /
                                             (abs(prev_avg_pay) + 1e-6) < tol_relative):
                print("  converged (payoff)")
                return current, it + 1
            prev_avg_pay = avg_pay

        return current, max_iter

    # ------------------------------------------------------------------
    # Nash equilibrium test
    # ------------------------------------------------------------------
    def test_nash_equilibrium(self, base_strategy, threshold=30.0, num_variations=5):
        """Test whether current strategy profile is a Nash equilibrium.
        All agent deviation checks run in parallel.

        Returns:
            is_nash: bool
            improvements: dict of {agent_name: {"gain": float, "base_payoff": float,
                                                 "best_payoff": float, "is_prosumer": bool}}
        """
        print("\nNash equilibrium test...")

        tasks = [(a, self.agents, self.config, self.T, self.stage,
                  base_strategy, num_variations, None) for a in self.agents]
        with Pool(processes=min(len(self.agents), 4)) as pool:
            br_results = pool.map(_evaluate_best_response, tasks)

        base_tasks = [(self.agents, self.config, self.T, self.stage,
                       base_strategy, a.name) for a in self.agents]
        with Pool(processes=min(len(self.agents), 4)) as pool:
            base_results = pool.map(_evaluate_payoffs, base_tasks)

        is_nash = True
        improvements = {}
        for name_b, base_pay, _ in base_results:
            for name_br, best_pay, _ in br_results:
                if name_br == name_b:
                    agent = next(a for a in self.agents if a.name == name_b)
                    gain = best_pay - base_pay
                    improvements[name_b] = {
                        "gain": gain,
                        "base_payoff": base_pay,
                        "best_payoff": best_pay,
                        "is_prosumer": agent.is_prosumer,
                    }
                    if gain > threshold:
                        is_nash = False
                    break
        return is_nash, improvements
