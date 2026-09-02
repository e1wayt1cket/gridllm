# strategies/__init__.py
"""Bidding strategy registry with pluggable strategy classes.

Each strategy is a BiddingStrategy subclass registered via the decorator.
Callers use adaptive_bidding() which looks up the registry.
"""

from typing import Dict, List, Optional
import numpy as np

from models import Agent, MarketConfig


class BiddingStrategy:
    """Base class for agent bidding strategies.

    Subclasses must set `name` and implement `formulate()`.
    Strategy-specific configuration (e.g. leader name for Stackelberg)
    is passed at construction time.
    """

    name: str = ""

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        """Return {agent_name: {"bid_mult": array(T,), "offer_adder": array(T,)}}."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, BiddingStrategy] = {}


def register_strategy(name: str):
    """Decorator that registers a BiddingStrategy subclass by name."""
    def decorator(cls):
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


# Import strategy modules so decorators register them
from strategies import random_bidding  # noqa: E402, F401
from strategies import rl_bidding      # noqa: E402, F401
from strategies import mpc_bidding     # noqa: E402, F401
from strategies import fixed_bidding   # noqa: E402, F401


# ---------------------------------------------------------------------------
# Public entry point — replaces market.adaptive_bidding
# ---------------------------------------------------------------------------

def adaptive_bidding(
    agents: List[Agent],
    config: MarketConfig,
    strategy: str = "rl",
    market_history: Optional[dict] = None,
    T: int = 96,
) -> dict:
    """Dispatch to registered strategy by name.

    Keeps the Stackelberg branches for backwards compatibility.
    Once Stackelberg strategies are migrated to classes, those branches
    can be removed.
    """
    # ---- Stackelberg (temporarily kept in-place) ----
    if strategy.startswith("stackelberg"):
        from stackelberg import stackelberg_bidding, stackelberg_nash
        parts = strategy.split(":", 1)
        if strategy.startswith("stackelberg_nash"):
            actions, history = stackelberg_nash(agents, config, T=T)
            return actions
        if len(parts) > 1:
            leader_name = parts[1]
        else:
            storage_agents = [a for a in agents if a.storage is not None]
            if not storage_agents:
                raise ValueError("No storage agent for Stackelberg leader")
            leader_name = storage_agents[0].name
        actions, info = stackelberg_bidding(agents, config, leader_name, T=T)
        if config.verbose:
            print(f"Stackelberg leader={info['leader']} "
                  f"payoff={info['optimal_payoff']:.1f}")
        return actions

    # ---- Registered strategies ----
    if strategy in STRATEGY_REGISTRY:
        strat_instance = STRATEGY_REGISTRY[strategy]()
        return strat_instance.formulate(agents, config, market_history, T)

    raise ValueError(f"Unknown strategy '{strategy}'. "
                     f"Available: {list(STRATEGY_REGISTRY.keys())} + stackelberg*")
