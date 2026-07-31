# strategies/mpc_bidding.py
"""MPC-based bidding strategy — thin wrapper around mpc_storage module."""

from typing import List, Optional

from models import Agent, MarketConfig
from strategies import BiddingStrategy, register_strategy


@register_strategy("mpc")
class MPCBiddingStrategy(BiddingStrategy):
    """Storage agents use look-ahead LP; others use random fallback."""

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        from mpc_storage import mpc_storage_bidding
        return mpc_storage_bidding(agents, config, market_history, T)
