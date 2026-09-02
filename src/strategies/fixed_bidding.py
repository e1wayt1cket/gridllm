# strategies/fixed_bidding.py
"""Fixed baseline bidding — all agents use truthful/default parameters."""

import numpy as np
from typing import List, Optional

from models import Agent, MarketConfig
from strategies import BiddingStrategy, register_strategy


@register_strategy("fixed")
class FixedStrategy(BiddingStrategy):
    """Fixed baseline: bid_mult=1.0, offer_adder=0.0 for all agents.

    Used as the comparison baseline for evaluating RL-trained policies.
    Represents truthful bidding with no strategic manipulation.
    """

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        return {
            a.name: {
                "bid_mult": np.full(T, 1.0, dtype=float),
                "offer_adder": np.full(T, 0.0, dtype=float),
            }
            for a in agents
        }
