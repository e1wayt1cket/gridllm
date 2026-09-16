# strategies/rl_bidding.py
"""RL-based bidding strategy using MATD3 Actor networks."""
import numpy as np
import torch
from typing import Dict, List, Optional

from models import Agent, MarketConfig
from rl_spec import action_spec_for
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
            # Try auto-loading from default path (skipped during training)
            import os as _os
            default_path = _os.path.join("policies", "default.pt")
            if (_os.path.exists(default_path)
                    and _os.environ.get("GRIDLLM_NO_POLICY_LOAD") != "1"):
                from rl_env import BiddingEnv
                _env = BiddingEnv(agents, config)
                from rl_bidding import load_policies
                try:
                    load_policies(default_path, _env.get_state_dim(),
                                  _env.get_action_bounds())
                except Exception as e:
                    # A saved checkpoint from an older observation space
                    # (e.g. the 103-dim pre-V1 encoding) is incompatible;
                    # fall back to fixed bidding rather than crashing.
                    print(f"[strategies/rl] policy load failed ({e}); "
                          f"falling back to fixed bidding", flush=True)
                    from strategies.fixed_bidding import FixedStrategy
                    return FixedStrategy().formulate(
                        agents, config, market_history, T)
                from rl_bidding import _TRAINED_POLICIES as _p
                policies = _p
        if not policies:
            # No trained policies available — use fixed defaults for all agents
            from strategies.fixed_bidding import FixedStrategy
            return FixedStrategy().formulate(agents, config, market_history, T)

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
            # Every agent starts at its own truthful declaration; trained
            # policies overwrite their blocks in the loop below. The two
            # ontologies differ -- a storage unit declares prices and a load
            # declares multipliers -- so the vector is built from the spec.
            spec = action_spec_for(a)
            actions[nm] = {name: np.full(T, spec.truthful[name])
                           for name in spec.action_names}

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
                    a, block_idx, None, avg_price, slr, avg_oth, bid_std)
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act_arr = policies[nm](obs_t).squeeze(0).numpy()
                if nm in actions:
                    spec = action_spec_for(a)
                    for i, name in enumerate(spec.action_names):
                        lo, hi = spec.bounds[name]
                        actions[nm][name][t_start:t_end] = \
                            float(np.clip(act_arr[i], lo, hi))

            for a in agents:
                nm = a.name
                if nm in policies or nm in actions:
                    continue
                # No policy and no action yet — use fixed defaults
                spec = action_spec_for(a)
                actions[nm] = {name: np.full(T, spec.truthful[name])
                               for name in spec.action_names}

        return actions
