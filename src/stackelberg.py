# stackelberg.py
"""Stackelberg leader bidding: one agent optimizes bids anticipating market response.

The leader searches over its bid parameters to maximize its own profit,
treating the market clearing (OPF) as a known response function.
This implements the Stackelberg game structure from the IES papers without DRL.
"""

import numpy as np
import copy
from scipy.optimize import minimize, differential_evolution

from models import Agent, MarketConfig
from market import clear_market, adaptive_bidding
from rl_spec import action_spec_for


def _leader_payoff(leader_name, leader_bus, actions, agents, T, config):
    """Compute leader's DA payoff from a market clearing result."""
    result = clear_market(agents, T, "DA", actions, config)
    if result is None:
        return None, -1e12
    sched = result["schedules"][leader_name]
    node_price = result["lmp"][:, leader_bus]
    sell = np.sum(sched["p_sell"] * node_price)
    buy = np.sum(sched["p_buy"] * node_price)
    penalty = np.sum(sched["unserved"] * config.market_design.penalty_unserved)
    return result, sell - buy - penalty


def _param_to_actions(x, n_blocks, T, leader, spec):
    """Turn a flat parameter vector into this leader's own action dict.

    The leader searches over whatever its action space is: a battery searches
    its two reservation prices, a load its multiplier and adder. Naming the
    keys here rather than in each caller is what keeps a storage leader from
    being handed a multiplier it can no longer use.
    """
    block_size = T // n_blocks
    out = {}
    for i, name in enumerate(spec.action_names):
        block = x[i * n_blocks:(i + 1) * n_blocks]
        out[name] = np.repeat(np.array(block), block_size)[:T]
    return out


def _build_bounds(config, n_blocks, spec):
    """Build bounds list for scipy.optimize, one block per action component."""
    bounds = []
    for name in spec.action_names:
        bounds += [spec.bounds[name]] * n_blocks
    return bounds


def _coarse_grid_search(objective, bounds, n_pts=5):
    """Coarse grid scan to find a good starting point for local refinement."""
    n_params = len(bounds)
    best_x = np.array([np.mean(b) for b in bounds])
    best_val = objective(best_x)

    # Sample each dimension independently
    for dim in range(n_params):
        lo, hi = bounds[dim]
        for val in np.linspace(lo, hi, n_pts):
            x_try = best_x.copy()
            x_try[dim] = val
            f_val = objective(x_try)
            if f_val < best_val:
                best_val = f_val
                best_x = x_try

    return best_x, best_val


def stackelberg_bidding(agents, config, leader_name,
                        follower_strategy="rl",
                        T=96, n_blocks=1):
    """Stackelberg leader bidding: optimize one agent's bids via direct search.

    The leader anticipates market clearing. Other agents use a fixed strategy.

    Uses differential evolution (robust to numerical noise) for low-dim
    search, with Nelder-Mead refinement at the end.

    Args:
        agents: list of Agent
        config: MarketConfig
        leader_name: name of the Stackelberg leader agent
        follower_strategy: fixed strategy for all non-leader agents
        T: time periods
        n_blocks: 1 for scalar bid (same all day), 4 for piecewise constant

    Returns:
        (actions dict, info dict)
    """
    leader = next(a for a in agents if a.name == leader_name)
    spec = action_spec_for(leader)

    # Compute follower strategies once (fixed during leader optimization)
    follower_actions = adaptive_bidding(agents, config,
                                        strategy=follower_strategy, T=T)

    n_params = n_blocks * spec.dim
    bounds = _build_bounds(config, n_blocks, spec)

    def objective(x):
        actions = copy.deepcopy(follower_actions)
        actions[leader_name] = _param_to_actions(x, n_blocks, T, leader, spec)
        _, payoff = _leader_payoff(leader_name, leader.bus,
                                   actions, agents, T, config)
        return -payoff

    # Stage 1: coarse grid for a good starting point
    x0, _ = _coarse_grid_search(objective, bounds, n_pts=7)

    # Stage 2: differential evolution for global robustness
    # popsize=8, maxiter=8 → ~72 evals (popsize * (maxiter+1))
    de_result = differential_evolution(
        objective, bounds, seed=1, polish=False,
        popsize=8, maxiter=8, tol=1.0, atol=5.0,
        x0=x0,
    )

    # Stage 3: Nelder-Mead refinement from DE best point
    nm_result = minimize(
        objective, de_result.x, method='Nelder-Mead', bounds=bounds,
        options={'xatol': 0.002, 'fatol': 2.0, 'maxiter': 30, 'maxfev': 80},
    )

    # Pick the better of DE and NM results
    if nm_result.fun < de_result.fun:
        best_x = nm_result.x
        best_fun = nm_result.fun
    else:
        best_x = de_result.x
        best_fun = de_result.fun

    # Build final actions
    actions = copy.deepcopy(follower_actions)
    actions[leader_name] = _param_to_actions(best_x, n_blocks, T, leader, spec)

    info = {
        "leader": leader_name,
        "optimal_payoff": -best_fun,
        "optimal_params": best_x.tolist(),
        "de_nfev": de_result.nfev,
        "nm_nfev": nm_result.nfev if hasattr(nm_result, 'nfev') else 0,
    }
    return actions, info


def stackelberg_nash(agents, config, leader_names=None,
                     follower_strategy="rl",
                     T=96, max_rounds=5):
    """Multi-leader Stackelberg-Nash: each storage agent acts as leader in turn.

    Iterates until strategies stabilize or max_rounds reached.
    """
    if leader_names is None:
        leader_names = [a.name for a in agents if a.storage is not None]
    if not leader_names:
        raise ValueError("No leader agents specified")

    current_actions = adaptive_bidding(agents, config,
                                       strategy=follower_strategy, T=T)
    history = []

    for r in range(max_rounds):
        print(f"\nStackelberg-Nash round {r + 1}/{max_rounds}")
        prev_payoffs = {}
        for name in leader_names:
            leader = next(a for a in agents if a.name == name)
            actions, info = stackelberg_bidding(
                agents, config, name,
                follower_strategy=follower_strategy, T=T)
            # Update this leader's strategy in current_actions
            current_actions[name] = actions[name]
            prev_payoffs[name] = info["optimal_payoff"]
            print(f"  {name}: payoff={info['optimal_payoff']:.1f}, "
                  f"params={[f'{p:.3f}' for p in info['optimal_params']]}, "
                  f"fev={info['de_nfev'] + info['nm_nfev']}")

        history.append({"round": r + 1, "payoffs": dict(prev_payoffs)})

        # Check convergence: all payoffs stable within tolerance
        if r > 0:
            prev = history[r - 1]["payoffs"]
            curr = history[r]["payoffs"]
            max_change = max(abs(curr[n] - prev[n]) for n in leader_names)
            if max_change < 10.0:
                print(f"  converged (max payoff change {max_change:.1f})")
                break

    return current_actions, history
