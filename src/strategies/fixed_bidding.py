# strategies/fixed_bidding.py
"""Fixed baseline bidding — all agents use truthful/default parameters."""

import numpy as np
from typing import List, Optional

from models import Agent, MarketConfig
from rl_spec import action_spec_for
from strategies import BiddingStrategy, register_strategy


@register_strategy("fixed")
class FixedStrategy(BiddingStrategy):
    """Every participant at its truthful declaration, in its own action space.

    Used as the comparison baseline for evaluating RL-trained policies. The
    declaration is taken from each agent's spec rather than written out, so a
    storage unit declares zero reservation prices and a load declares its own
    prices unshaded -- which is what "no strategic manipulation" means for each
    of them, and is not the same vector.
    """

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        return {
            a.name: {name: np.full(T, action_spec_for(a).truthful[name],
                                   dtype=float)
                     for name in action_spec_for(a).action_names}
            for a in agents
        }
