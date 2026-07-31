# strategies/rl_bidding.py
"""RL-based bidding strategy using MATD3 Actor networks."""
import numpy as np
import torch
from typing import Dict, List, Optional

from models import Agent, MarketConfig
from strategies import BiddingStrategy, register_strategy


@register_strategy("rl")
class RLBiddingStrategy(BiddingStrategy):
    """RL-driven bidding using trained Actor networks (MATD3)."""

    def __init__(self):
        self.policies: dict = {}

    def set_policies(self, policies: dict):
        """Inject trained Actor instances."""
        self.policies = policies

    def formulate(
        self, agents: List[Agent], config: MarketConfig,
        market_history: Optional[dict] = None, T: int = 96,
    ) -> dict:
        policies = self.policies
        if not policies:
            from rl_bidding import _TRAINED_POLICIES
            policies = _TRAINED_POLICIES
        if not policies:
            # Try auto-loading from default path
            import os as _os
            default_path = _os.path.join("policies", "default.pt")
            if _os.path.exists(default_path):
                from rl_env import BiddingEnv
                _env = BiddingEnv(agents, config)
                from rl_bidding import load_policies
                load_policies(default_path, _env.get_state_dim(),
                             _env.get_action_bounds())
                from rl_bidding import _TRAINED_POLICIES as _p
                policies = _p
        if not policies:
            from strategies.random_bidding import RandomStrategy
            return RandomStrategy().formulate(agents, config, market_history, T)

        from rl_env import (BiddingEnv, BID_MULT_LOW, BID_MULT_HIGH,
                            OFFER_ADDER_LOW, OFFER_ADDER_HIGH,
                            BLOCK_SIZE, N_BLOCKS)

        env_temp = BiddingEnv(agents, config)
        avg_price = 420.0
        if market_history is not None and hasattr(market_history, "get"):
            avg_price = float(np.mean(market_history.get("price", [420.0])))
        action_low = BID_MULT_LOW
        action_high = BID_MULT_HIGH

        actions = {}
        for a in agents:
            nm = a.name
            if nm in policies or a.is_prosumer:
                actions[nm] = {"bid_mult": np.full(T, 1.0),
                               "offer_adder": np.full(T, 0.0)}
            else:
                actions[nm] = {"bid_mult": np.full(
                    T, np.mean(config.market_design.bid_mult_range))}

        for block_idx in range(N_BLOCKS):
            t_start = block_idx * BLOCK_SIZE
            t_end = min(t_start + BLOCK_SIZE, T)
            slr = env_temp._compute_system_load_re_ratio(t_start)

            for a in agents:
                nm = a.name
                if nm not in policies:
                    continue
                avg_oth, bid_std = env_temp._compute_opponent_features(nm)
                obs = env_temp._get_agent_obs(
                    a, block_idx, None, avg_price, slr, 0.0, avg_oth, bid_std)
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act_arr = policies[nm](obs_t).squeeze(0).numpy()
                bid_m = float(np.clip(act_arr[0], action_low, action_high))
                offer_a = float(np.clip(act_arr[1], OFFER_ADDER_LOW,
                                        OFFER_ADDER_HIGH))
                if nm in actions:
                    actions[nm]["bid_mult"][t_start:t_end] = bid_m
                    actions[nm]["offer_adder"][t_start:t_end] = offer_a

            for a in agents:
                nm = a.name
                if nm in policies or not a.is_prosumer:
                    continue
                rng = np.random.RandomState(
                    hash(nm + str(block_idx)) % (2**31))
                bid_m = rng.uniform(action_low, action_high)
                offer_a = rng.uniform(OFFER_ADDER_LOW, OFFER_ADDER_HIGH)
                if nm in actions:
                    actions[nm]["bid_mult"][t_start:t_end] = bid_m
                    actions[nm]["offer_adder"][t_start:t_end] = offer_a

        return actions
