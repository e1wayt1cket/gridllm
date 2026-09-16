# rl_profit_diagnostics.py
"""Shared profit/welfare diagnostics for RL bidding policies.

Separates the genuine dispatch effect of an RL policy from the valuation
artifact caused by bid shading, so evaluation reports an honest "consistent
welfare delta" instead of the raw OPF objective-value delta.

Used by eval_agents.py (evaluation CSV) and diagnose_profit.py
(decomposition report); both import from here as the single source of truth.
"""

import numpy as np

import money


def valuation_artifact(sched: dict, declared: dict, agents, config) -> float:
    """Welfare (ObjVal) drop caused purely by the quotes the fleet declared.

    The OPF objective values charge and discharge at the agent's DECLARED
    quotes. A unit that shades them therefore changes the metric even when the
    real dispatch is unchanged. Revaluing the same RL dispatch at the truthful
    declaration isolates that distortion.

    The quote ontology is the one `dispatch_core.quote_arrays` applies, keyed on
    `participant_type`:

    * a storage unit declares its two reservation prices in CNY/MWh and has
      neither load nor generation, so only its charge and discharge legs move;
    * a prosumer declares a multiplier on its own `bid_value` and an adder to
      its own `offer_cost`, and revaluing at truthful means the pair
      `(1.0, 0.0)`.

    Mirrors the clearing objective: the ``bid*served`` term is undiscounted
    while the ``bid*ch + offer*dis`` term carries the discount factor, and both
    are energy-valued, so the whole artifact is scaled by the period length.

    Parameters
    ----------
    sched : dict
        {agent_name: {key: (T,) array}} for p_ch, p_dis, served, ...
    declared : dict
        {agent_name: {action_name: (T,) array}} — the quotes the optimizer
        actually applied per period, in the agent's own action space.
    agents : list of Agent
    config : MarketConfig
    """
    gamma = config.storage.discount_factor
    A = 0.0
    for a in agents:
        if a.storage is None:
            continue
        d = declared.get(a.name)
        if d is None:
            continue
        s = sched[a.name]
        disc = gamma ** np.arange(len(s["p_ch"]))
        if getattr(a, "participant_type", "prosumer") == "storage":
            # Truthful is (0, 0) -- declaring no premium over `cycle_cost`,
            # which the objective already charges separately. The declared
            # prices are therefore themselves the whole distortion.
            ch = np.asarray(d["charge_price"], dtype=float)
            dis = np.asarray(d["discharge_price"], dtype=float)
            A += float(np.sum(disc * (ch * s["p_ch"] + dis * s["p_dis"])))
        else:
            bm = np.asarray(d["bid_mult"], dtype=float)
            oa = np.asarray(d["offer_adder"], dtype=float)
            # load-value (served) term: undiscounted
            A += float(np.sum(a.bid_value * (bm - 1.0) * s["served"]))
            # battery legs: discounted
            A += float(np.sum(
                disc * (a.bid_value * (bm - 1.0) * s["p_ch"] - oa * s["p_dis"])))
    return A * money.DT_HOURS
