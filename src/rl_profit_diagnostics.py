# rl_profit_diagnostics.py
"""Shared profit/welfare diagnostics for RL bidding policies.

Separates the genuine dispatch effect of an RL policy from the valuation
artifact caused by bid shading, so evaluation reports an honest "consistent
welfare delta" instead of the raw OPF objective-value delta.

Used by eval_agents.py (evaluation CSV) and diagnose_profit.py
(decomposition report); both import from here as the single source of truth.
"""

import numpy as np


def valuation_artifact(sched: dict, declared: dict, agents, config) -> float:
    """Welfare (ObjVal) drop caused purely by RL bid shading.

    The OPF objective values load ``served`` and storage charge/discharge at
    the agent's DECLARED bid/offer. A shaded bid_mult<1 therefore understates
    both the storage agent's load value and its charging value in the metric,
    even when the real dispatch is unchanged. Revaluing the same RL dispatch
    at truthful bid_value/offer_cost isolates that distortion.

    Mirrors dispatch_socp.py: ``bid*served`` is undiscounted; the storage
    ``bid*ch - offer*dis`` term carries the discount factor.

    Parameters
    ----------
    sched : dict
        {agent_name: {key: (T,) array}} for p_ch, p_dis, served, ...
    declared : dict
        {agent_name: {"bid_mult": (T,), "offer_adder": (T,)}} — the bid/offer
        the optimizer actually applied per period.
    agents : list of Agent
    config : MarketConfig
    """
    gamma = config.storage.discount_factor
    A = 0.0
    for a in agents:
        if a.storage is None:
            continue
        s = sched[a.name]
        bm = declared[a.name]["bid_mult"]
        oa = declared[a.name]["offer_adder"]
        # load-value (served) term: undiscounted
        A += float(np.sum(a.bid_value * (bm - 1.0) * s["served"]))
        # storage charge/discharge valuation: discounted
        disc = gamma ** np.arange(len(s["p_ch"]))
        A += float(np.sum(
            disc * (a.bid_value * (bm - 1.0) * s["p_ch"] - oa * s["p_dis"])))
    return A
