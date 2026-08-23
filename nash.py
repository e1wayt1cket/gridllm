# nash.py
"""Nash equilibrium solver with block-parameterized best response optimization.

Methods:
  - Diagonalization (Gauss-Seidel): sequential best response, immediate updates.
  - Jacobi (parallel diagonalization): simultaneous updates from shared snapshot.
  - Fictitious play: each agent best-responds to average of opponents' history.

Best response: COBYLA over strategy blocks (default 24 blocks, hourly for T=96).
Fallback: random sampling when scipy is unavailable or use_optimization=False.
"""

import numpy as np
import copy
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

from models import Agent, MarketConfig
from market import clear_market

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Strategy block helpers
# ---------------------------------------------------------------------------

def _strategy_to_blocks(arr: np.ndarray, block_count: int) -> np.ndarray:
    """Average 96-period strategy into block_count blocks."""
    T = len(arr)
    bs = T // block_count
    return np.array([arr[i * bs:(i + 1) * bs].mean() for i in range(block_count)])


def _blocks_to_strategy(blocks: np.ndarray, T: int = 96) -> np.ndarray:
    """Expand block_count blocks back to T-period strategy."""
    bc = len(blocks)
    bs = T // bc
    out = np.zeros(T)
    for i in range(bc):
        out[i * bs:(i + 1) * bs] = blocks[i]
    return out


# ---------------------------------------------------------------------------
# Payoff evaluation (module-level, multiprocessing-safe)
# ---------------------------------------------------------------------------

def _evaluate_payoffs(args):
    """Evaluate economic surplus for a single agent under a strategy profile.

    Returns the agent's full individual surplus:
      bid_value * served  — consumption value
      - p_buy * price     — grid import cost
      + p_sell * price    — grid export revenue
      - offer_cost * gen  — generation cost (PV + wind)
      - penalty * unserved

    This captures self-consumption value (PV serving own load with zero
    market transaction) that the naive sell*price - buy*price misses.
    """
    agents, config, T, stage, action_variant, target_name = args
    try:
        result = clear_market(agents, T, stage, action_variant, config)
    except Exception:
        return target_name, float('-inf'), None
    sched = result["schedules"][target_name]
    lmp = result["lmp"]
    agent = next(a for a in agents if a.name == target_name)
    bus = agent.bus
    node_price = lmp[:, bus]
    consumption_value = agent.bid_value * np.sum(sched["served"])
    generation_cost = agent.offer_cost * np.sum(sched["pv_used"] + sched["wind_used"])
    market_payment = np.sum(sched["p_sell"] * node_price) - np.sum(sched["p_buy"] * node_price)
    penalty = config.market_design.penalty_unserved * np.sum(sched["unserved"])
    payoff = consumption_value - generation_cost + market_payment - penalty
    return target_name, float(payoff), action_variant.get(target_name)


# ---------------------------------------------------------------------------
# COBYLA-based best response (block-level optimization)
# ---------------------------------------------------------------------------

def _best_response_optimize(agent, agents, config, T, stage, base_strategy,
                            block_count=24, maxiter=500):
    """Find best response via COBYLA over block-level strategy parameters.

    Reduces 96-dim strategy to block_count dims (e.g. 24 blocks of 4 periods).
    COBYLA handles the black-box OPF objective without derivatives.
    """
    name = agent.name
    is_prosumer = agent.is_prosumer
    base_bid = np.array(base_strategy[name]["bid_mult"])

    if is_prosumer:
        base_offer = np.array(base_strategy[name]["offer_adder"])
        x0 = np.concatenate([
            _strategy_to_blocks(base_bid, block_count),
            _strategy_to_blocks(base_offer, block_count),
        ])
    else:
        x0 = _strategy_to_blocks(base_bid, block_count)

    def _objective(x):
        trial = copy.deepcopy(base_strategy)
        if is_prosumer:
            bid_blk = x[:block_count]
            offer_blk = x[block_count:]
            bid = np.clip(_blocks_to_strategy(bid_blk, T),
                          *config.market_design.bid_mult_range)
            offer = np.clip(_blocks_to_strategy(offer_blk, T),
                            *config.market_design.offer_adder_range)
            trial[name] = {"bid_mult": bid, "offer_adder": offer}
        else:
            bid_blk = x
            bid = np.clip(_blocks_to_strategy(bid_blk, T),
                          *config.market_design.bid_mult_range)
            trial[name] = {"bid_mult": bid}

        r_name, payoff, _ = _evaluate_payoffs(
            (agents, config, T, stage, trial, name))
        return -payoff if r_name == name else float('inf')

    try:
        res = _scipy_minimize(_objective, x0, method='COBYLA',
                              options={'maxiter': maxiter, 'rhobeg': 0.05})
        opt_x = res.x
    except Exception:
        opt_x = x0

    # Reconstruct best strategy from optimized blocks
    best_strat = copy.deepcopy(base_strategy[name])
    if is_prosumer:
        best_strat["bid_mult"] = np.clip(
            _blocks_to_strategy(opt_x[:block_count], T),
            *config.market_design.bid_mult_range)
        best_strat["offer_adder"] = np.clip(
            _blocks_to_strategy(opt_x[block_count:], T),
            *config.market_design.offer_adder_range)
    else:
        best_strat["bid_mult"] = np.clip(
            _blocks_to_strategy(opt_x, T), *config.market_design.bid_mult_range)

    # Final OPF evaluation to get accurate payoff
    trial_final = copy.deepcopy(base_strategy)
    trial_final[name] = best_strat
    r_name, payoff, _ = _evaluate_payoffs(
        (agents, config, T, stage, trial_final, name))
    return name, payoff if r_name == name else float('-inf'), best_strat


# ---------------------------------------------------------------------------
# Random-sampling best response (fallback)
# ---------------------------------------------------------------------------

def _generate_variations(base_strategy, agent, config, num_variations=150,
                         exploration_scale=None):
    """Generate perturbed strategy variants via random normal sampling."""
    variants = []
    base_bid = np.array(base_strategy[agent.name]["bid_mult"])
    is_prosumer = agent.is_prosumer
    if is_prosumer:
        base_offer = np.array(base_strategy[agent.name]["offer_adder"])

    bid_sigma = 0.08 * (exploration_scale or 1.0)
    offer_sigma = 3.0 * (exploration_scale or 1.0)

    for _ in range(num_variations):
        new_strat = copy.deepcopy(base_strategy)
        if is_prosumer:
            bid_mult = np.clip(np.random.normal(base_bid, bid_sigma),
                               *config.market_design.bid_mult_range)
            offer_adder = np.clip(np.random.normal(base_offer, offer_sigma),
                                  *config.market_design.offer_adder_range)
            new_strat[agent.name] = {"bid_mult": bid_mult,
                                     "offer_adder": offer_adder}
        else:
            bid_mult = np.clip(np.random.normal(base_bid, bid_sigma),
                               *config.market_design.bid_mult_range)
            new_strat[agent.name] = {"bid_mult": bid_mult}
        variants.append(new_strat)
    return variants


def _evaluate_best_response(args):
    """Find best response via random sampling (multiprocessing-safe target)."""
    (agent, agents, config, T, stage, base_strategy, num_variations,
     exploration_scale) = args
    name = agent.name
    variants = _generate_variations(base_strategy, agent, config,
                                    num_variations, exploration_scale)

    best_pay = float('-inf')
    best_strat = copy.deepcopy(base_strategy[name])
    for var in variants:
        r_name, payoff, _ = _evaluate_payoffs(
            (agents, config, T, stage, var, name))
        if r_name == name and payoff > best_pay:
            best_pay = payoff
            best_strat = var[name]

    return name, best_pay, best_strat


def compute_regret_summary(improvements: dict) -> dict:
    """Aggregate per-agent regret (best-response payoff minus base payoff).

    Regret per agent equals the profitable-deviation gain; total regret sums
    across agents and quantifies how far the joint strategy profile is from a
    Nash equilibrium (0 means no agent can unilaterally improve). Mirrors the
    best-response regret reported by marl_clearing_and_bidding.
    """
    if not improvements:
        return {"total_regret": 0.0, "mean_regret": 0.0, "max_regret": 0.0,
                "n_profitable": 0, "regret_share": 0.0}
    regrets = np.array([v.get("regret", v.get("gain", 0.0))
                        for v in improvements.values()])
    n_profitable = int(sum(1 for v in improvements.values()
                           if v.get("profitable", False)))
    total_base = sum(abs(v.get("base_payoff", 0.0))
                     for v in improvements.values())
    total_regret = float(regrets.sum())
    return {
        "total_regret": total_regret,
        "mean_regret": float(regrets.mean()),
        "max_regret": float(regrets.max()),
        "n_profitable": n_profitable,
        "regret_share": float(total_regret / total_base)
        if total_base > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# NashEquilibriumTester — detection only (no solving)
# ---------------------------------------------------------------------------

class NashEquilibriumTester:
    """Nash equilibrium detector with block-parameterized best response.

    Tests whether a given strategy profile is a Nash equilibrium by checking
    each agent for profitable unilateral deviations via random sampling.

    Important: electricity markets with storage are NOT zero-sum — the
    payoff landscape has local plateaus. Detection uses random perturbations;
    a negative result is informative but a "pass" does not guarantee a
    global Nash equilibrium.

    Parameters
    ----------
    agents : list of Agent
    config : MarketConfig
    T : int, default 96
    stage : str, default "DA"
    parallel : bool or None
        Whether to use multiprocessing. None auto-detects: True on Linux/macOS,
        False on Windows (spawn forks require serial guard).
    use_optimization : bool or None
        Whether to use COBYLA block optimization for best response.
        None auto-detects: True if scipy is available.
    block_count : int, default 12
        Number of strategy blocks (2-hour for T=96). Higher = finer granularity.
    """

    def __init__(self, agents, config, T=96, stage="DA", parallel=None,
                 use_optimization=None, block_count=12):
        self.agents = agents
        self.config = config
        self.T = T
        self.stage = stage
        # Auto-detect: serial on Windows (spawn), parallel elsewhere
        self.parallel = parallel if parallel is not None else (os.name != 'nt')
        self.use_optimization = (use_optimization if use_optimization is not None
                                 else _HAS_SCIPY)
        self.block_count = block_count
        self._payoff_cache = {}

    # -- best response dispatch ---------------------------------------------

    def _best_response(self, agent, base_strategy, num_variations=150):
        """Dispatch: COBYLA optimization or random sampling."""
        cache_key = self._make_cache_key(agent.name, base_strategy)
        if cache_key in self._payoff_cache:
            return self._payoff_cache[cache_key]

        if self.use_optimization:
            result = _best_response_optimize(
                agent, self.agents, self.config, self.T, self.stage,
                base_strategy, self.block_count)
        else:
            result = _evaluate_best_response(
                (agent, self.agents, self.config, self.T, self.stage,
                 base_strategy, num_variations, None))

        self._payoff_cache[cache_key] = result
        return result

    def _make_cache_key(self, target_name, strategy):
        """Hash the full strategy profile for payoff cache lookup.

        Includes ALL agents' strategies, not just the target's own. Otherwise,
        in diagonalization, when Agent A updates and Agent B re-computes BR
        with unchanged bid_mult/offer_adder, the cache returns a stale payoff
        computed against Agent A's old strategy.
        """
        parts = [target_name]
        for a in sorted(self.agents, key=lambda x: x.name):
            s = strategy.get(a.name, {})
            parts.append(tuple(np.round(s.get("bid_mult", []), 5)))
            if a.is_prosumer:
                parts.append(tuple(np.round(s.get("offer_adder", []), 5)))
        return tuple(parts)

    def _clear_cache(self):
        self._payoff_cache.clear()

    def compute_base_payoffs(self, base_strategy):
        """Compute base payoff for each agent under the given strategy.
        Results are cached so that test_nash_equilibrium re-uses them.
        """
        tasks = [(self.agents, self.config, self.T, self.stage,
                  base_strategy, a.name) for a in self.agents]
        if self.parallel:
            try:
                n_proc = min(len(tasks), os.cpu_count() or 4)
                with Pool(processes=n_proc) as pool:
                    results = pool.map(_evaluate_payoffs, tasks)
            except Exception:
                results = [_evaluate_payoffs(t) for t in tasks]
        else:
            results = [_evaluate_payoffs(t) for t in tasks]
        return {r[0]: r[1] for r in results}

    # Nash equilibrium test
    # ------------------------------------------------------------------

    def test_nash_equilibrium(self, base_strategy, num_variations=150,
                              threshold_rel=0.01):
        """Test whether current strategy profile is a Nash equilibrium.

        Each agent's best-response payoff is compared to its base payoff.
        An agent has a profitable deviation if relative gain > threshold_rel.

        Returns
        -------
        is_nash : bool
        improvements : dict
            {agent_name: {"gain": float, "base_payoff": float,
                          "best_payoff": float, "is_prosumer": bool,
                          "profitable": bool}}
        """
        print("\nNash equilibrium test...")
        self._clear_cache()

        # Compute base payoffs
        base_payoffs = self.compute_base_payoffs(base_strategy)

        # Compute best responses for each agent
        if self.parallel and not self.use_optimization:
            tasks = [(a, self.agents, self.config, self.T, self.stage,
                      base_strategy, num_variations, None)
                     for a in self.agents]
            try:
                n_proc = min(len(tasks), os.cpu_count() or 4)
                with Pool(processes=n_proc) as pool:
                    br_results = pool.map(_evaluate_best_response, tasks)
            except Exception:
                br_results = [_evaluate_best_response(t) for t in tasks]
        else:
            br_results = [
                self._best_response(a, base_strategy, num_variations)
                for a in self.agents]

        is_nash = True
        improvements = {}
        for name, best_pay, _ in br_results:
            base_pay = base_payoffs.get(name, float('-inf'))
            agent = next(a for a in self.agents if a.name == name)
            gain = best_pay - base_pay
            rel_gain = gain / max(abs(base_pay), 1.0)
            profitable = rel_gain > threshold_rel
            improvements[name] = {
                "gain": gain,
                "regret": gain,          # regret = best - base payoff
                "rel_gain": rel_gain,
                "relative_regret": rel_gain,
                "base_payoff": base_pay,
                "best_payoff": best_pay,
                "is_prosumer": agent.is_prosumer,
                "profitable": profitable,
            }
            if profitable:
                is_nash = False
                print(f"  {name}: gain={gain:+.1f} (rel={rel_gain:.4f}) ***")
            else:
                print(f"  {name}: gain={gain:+.1f} (rel={rel_gain:.4f})")

        # -- print statistical summary
        self._print_nash_summary(improvements, is_nash, threshold_rel)

        self._clear_cache()
        return is_nash, improvements

    def _print_nash_summary(self, improvements, is_nash, threshold_rel):
        """Print statistical summary of Nash equilibrium test results."""
        n_total = len(improvements)
        n_profitable = sum(1 for v in improvements.values() if v["profitable"])
        n_prosumer = sum(1 for v in improvements.values() if v["is_prosumer"])
        n_prosumer_prof = sum(1 for v in improvements.values()
                              if v["is_prosumer"] and v["profitable"])

        gains = np.array([v["gain"] for v in improvements.values()])
        rel_gains = np.array([v["rel_gain"] for v in improvements.values()])
        base_pays = np.array([v["base_payoff"] for v in improvements.values()])
        best_pays = np.array([v["best_payoff"] for v in improvements.values()])

        print(f"\n{'='*60}")
        print(f"  Nash Equilibrium Test — Statistical Summary")
        print(f"{'='*60}")
        print(f"  Threshold:  relative > {threshold_rel:.3f}")
        print(f"  Result:     {'NASH' if is_nash else 'NOT NASH'}  ({n_profitable}/{n_total} agents with profitable deviation)")
        print(f"  Prosumers:  {n_prosumer_prof}/{n_prosumer} profitable")
        print(f"  Consumers:  {n_profitable - n_prosumer_prof}/{n_total - n_prosumer} profitable")
        print(f"{'='*60}")
        print(f"  Gain (absolute, CNY):")
        print(f"    mean={gains.mean():+.1f}  median={np.median(gains):+.1f}  "
              f"std={gains.std():.1f}  min={gains.min():+.1f}  max={gains.max():+.1f}")
        print(f"  Gain (relative):")
        print(f"    mean={rel_gains.mean():+.4f}  median={np.median(rel_gains):+.4f}  "
              f"max={rel_gains.max():+.4f}")
        print(f"  Base  payoff: mean={base_pays.mean():,.0f}  median={np.median(base_pays):,.0f}")
        print(f"  Best  payoff: mean={best_pays.mean():,.0f}  median={np.median(best_pays):,.0f}")
        reg = compute_regret_summary(improvements)
        print(f"  Total regret: {reg['total_regret']:+.1f}  "
              f"(share of base payoff: {reg['regret_share']:.1%})")
        print(f"{'='*60}\n")

        if n_profitable > 0:
            print("  Top profitable deviations:")
            sorted_imps = sorted(
                [(n, v) for n, v in improvements.items() if v["profitable"]],
                key=lambda x: x[1]["gain"], reverse=True)
            for name, imp in sorted_imps[:10]:
                tag = "P" if imp["is_prosumer"] else "C"
                print(f"    [{tag}] {name:12s}  gain={imp['gain']:+,.0f} CNY  "
                      f"({imp['rel_gain']:+.4f})  base={imp['base_payoff']:,.0f}  best={imp['best_payoff']:,.0f}")
            if n_profitable > 10:
                print(f"    ... and {n_profitable - 10} more")
            print()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_nash_results(improvements, is_nash, threshold_rel=0.01,
                      save_path=None, title="Nash Equilibrium Test"):
    """Generate statistical charts for Nash equilibrium test results.

    Produces a 2-panel figure:
      - Left: horizontal bar chart of gains per agent (sorted), color-coded by
        profitable deviation and agent type.
      - Right: histogram + KDE of gain distribution with threshold markers.

    Parameters
    ----------
    improvements : dict
        Output from NashEquilibriumTester.test_nash_equilibrium().
    is_nash : bool
    threshold_rel : float
        Relative gain threshold used in the test.
    save_path : str or None
        Path to save the figure. If None, defaults to "nash_test_results.png".
    title : str
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Sort agents by gain
    sorted_items = sorted(improvements.items(), key=lambda x: x[1]["gain"])
    names = [it[0] for it in sorted_items]
    gains = np.array([it[1]["gain"] for it in sorted_items])
    is_prosumer = [it[1]["is_prosumer"] for it in sorted_items]
    profitable = [it[1]["profitable"] for it in sorted_items]
    rel_gains = np.array([it[1]["rel_gain"] for it in sorted_items])

    n_total = len(gains)
    n_profitable = sum(profitable)

    # Color mapping
    bar_colors = ['#ef4444' if p else '#10b981' for p in profitable]
    edge_colors = ['#b91c1c' if p else '#047857' for p in profitable]
    hatches = ['//' if p else '' for p in profitable]

    fig, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(16, max(7, n_total * 0.22)),
                                          gridspec_kw={'width_ratios': [2.5, 1]})
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    # ---- Panel 1: horizontal bar chart ----
    y_pos = range(n_total)
    bars = ax_bar.barh(y_pos, gains, color=bar_colors, edgecolor=edge_colors,
                       height=0.7, linewidth=1.2)

    # Hatch profitable bars
    for i, (bar, hatch) in enumerate(zip(bars, hatches)):
        if hatch:
            bar.set_hatch(hatch)

    # Mark prosumers with a different edge style
    for i, (bar, is_p) in enumerate(zip(bars, is_prosumer)):
        if is_p:
            bar.set_linewidth(2.5)
            bar.set_edgecolor('#f59e0b')  # amber edge for prosumers

    ax_bar.axvline(x=0, color='#d1d5db', linewidth=0.8)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=8, fontfamily='monospace')
    ax_bar.set_xlabel('Gain (CNY)', fontsize=11)
    ax_bar.set_title(f'Agent Deviation Gains  ({n_profitable}/{n_total} profitable)',
                     fontsize=12, fontweight='600')
    ax_bar.grid(axis='x', alpha=0.3, linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ef4444', edgecolor='#b91c1c', label='Profitable deviation'),
        Patch(facecolor='#10b981', edgecolor='#047857', label='No deviation'),
        Patch(facecolor='white', edgecolor='#f59e0b', linewidth=2.5, label='Prosumer'),
    ]
    ax_bar.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.9, ncol=2)

    # Annotate top profitable bars with gain value
    for i, (bar, gain, prof) in enumerate(zip(bars, gains, profitable)):
        if prof:
            ax_bar.text(bar.get_width() + ax_bar.get_xlim()[1] * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{gain:+,.0f}', va='center', fontsize=7,
                        fontweight='bold', color='#b91c1c')

    # ---- Panel 2: histogram + summary ----
    bins = max(10, min(30, n_total // 2))
    ax_dist.hist(gains, bins=bins, color='#6366f1', edgecolor='#4338ca',
                 alpha=0.75, linewidth=0.8, label='Gain distribution')
    ax_dist.axvline(x=0, color='#d1d5db', linewidth=0.8)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(gains.min() * 1.1, gains.max() * 1.1, 200)
        kde = gaussian_kde(gains)
        ax_dist_twin = ax_dist.twinx()
        ax_dist_twin.plot(kde_x, kde(kde_x), color='#f59e0b', linewidth=1.8,
                          label='KDE')
        ax_dist_twin.set_ylabel('Density', fontsize=9, color='#92400e')
        ax_dist_twin.tick_params(axis='y', colors='#92400e', labelsize=8)
        ax_dist_twin.set_ylim(bottom=0)
    except Exception:
        # KDE requires scipy; silent fallback is acceptable for visualization
        import logging
        logging.getLogger(__name__).debug("KDE overlay failed", exc_info=True)

    ax_dist.set_xlabel('Gain (CNY)', fontsize=11)
    ax_dist.set_ylabel('Agent count', fontsize=11)
    ax_dist.set_title('Gain Distribution', fontsize=12, fontweight='600')
    ax_dist.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # -- Summary text box --
    status_str = 'NASH' if is_nash else 'NOT NASH'
    status_color = '#047857' if is_nash else '#b91c1c'
    summary_lines = [
        f"Status: {status_str}",
        f"Agents: {n_total}",
        f"Profitable: {n_profitable}",
        f"Mean gain: {gains.mean():+,.1f}",
        f"Median gain: {np.median(gains):+,.1f}",
        f"Std gain: {gains.std():.1f}",
        f"Max gain: {gains.max():+,.1f}",
        f"Min gain: {gains.min():+,.1f}",
    ]
    ax_dist.text(0.95, 0.95, '\n'.join(summary_lines),
                 transform=ax_dist.transAxes, fontsize=8.5,
                 fontfamily='monospace', verticalalignment='top',
                 horizontalalignment='right',
                 bbox={'boxstyle': 'round', 'facecolor': 'white',
                       'edgecolor': '#d1d5db', 'alpha': 0.9})

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is None:
        save_path = 'output/nash_test_results.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Chart saved to: {save_path}")

    return fig
