# strategies/random_bidding.py
"""Random exploration bidding — each period gets independent random parameters."""

import numpy as np
from typing import List, Optional

from models import Agent, MarketConfig
from strategies import BiddingStrategy, register_strategy


@register_strategy("random")
class RandomStrategy(BiddingStrategy):
    """Random bid_mult and offer_adder, independently sampled per period.

    Prosumers draw bid_mult from [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    and offer_adder from [0, 15, 30, 45].  Non-prosumers use the
    midpoint of the configured bid_mult range.
    """

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        bid_choices = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
        offer_choices = [0, 15, 30, 45]
        actions = {}
        for a in agents:
            rng = np.random.RandomState(hash(a.name) % (2**31))
            if a.is_prosumer:
                actions[a.name] = {
                    "bid_mult": np.array([rng.choice(bid_choices) for _ in range(T)]),
                    "offer_adder": np.array([rng.choice(offer_choices) for _ in range(T)]),
                }
            else:
                actions[a.name] = {
                    "bid_mult": np.full(T, np.mean(config.market_design.bid_mult_range)),
                    "offer_adder": np.zeros(T),
                }
        return actions
