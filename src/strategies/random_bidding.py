# strategies/random_bidding.py
"""Random exploration bidding — each period gets independent random parameters."""

import numpy as np
from typing import List, Optional

from models import Agent, MarketConfig
from rl_spec import action_spec_for
from strategies import BiddingStrategy, register_strategy


@register_strategy("random")
class RandomStrategy(BiddingStrategy):
    """Each agent's own action components drawn independently per period.

    The space comes from the agent's spec, so a storage unit draws two
    reservation prices and a load draws a multiplier and an adder -- sampling
    one ontology and handing it to a participant of the other is what the
    clearing now rejects outright. Five levels per component, evenly spaced
    across that component's own range, so the sample spans the space
    including its truthful corner. Seeded per agent name, so a run repeats.
    """

    LEVELS = 5

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        actions = {}
        for a in agents:
            rng = np.random.RandomState(hash(a.name) % (2**31))
            spec = action_spec_for(a)
            actions[a.name] = {}
            for name in spec.action_names:
                lo, hi = spec.bounds[name]
                choices = np.linspace(lo, hi, self.LEVELS)
                if spec.truthful[name] not in choices:
                    choices[0] = spec.truthful[name]
                actions[a.name][name] = np.array(
                    [rng.choice(choices) for _ in range(T)], dtype=float)
        return actions
