# rl_env.py
"""Gym-style RL environment for agent bidding in the electricity market.

Each episode = one 96-period day. Agents act on 4-period blocks (24 steps).
State includes local forecasts, historical LMP, price predictions, and SOC.
Action is discrete: chooses (bid_mult, offer_adder) pair from a fixed grid.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque

from models import Agent, MarketConfig
from market import clear_market, adaptive_bidding
from price_forecaster import NodalPriceForecaster

# Discrete action grid: (bid_mult, offer_adder)
BID_MULTS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
OFFER_ADDERS = [0, 15, 30, 45]
N_ACTIONS = len(BID_MULTS) * len(OFFER_ADDERS)


def action_to_params(action_idx: int):
    """Map discrete action index to (bid_mult, offer_adder)."""
    b = action_idx // len(OFFER_ADDERS)
    o = action_idx % len(OFFER_ADDERS)
    return BID_MULTS[b], OFFER_ADDERS[o]


BLOCK_SIZE = 4   # periods per decision block
N_BLOCKS = 24    # 96 / 4


class BiddingEnv:
    """Multi-agent bidding environment with Independent PPO.

    Each agent trains its own policy. The environment processes all agents
    jointly through a single market clearing call per step.

    Parameters
    ----------
    agents : list of Agent
    config : MarketConfig
    stage : str
        "DA" or "RT".
    nash_tester : callable or None
        Optional function to evaluate Nash equilibrium quality.
    """

    def __init__(self, agents: List[Agent], config: MarketConfig,
                 stage: str = "DA"):
        self.all_agents = agents
        self.config = config
        self.T = 96
        self.stage = stage
        self.wholesale = None  # set at episode start

        # Only train RL for prosumers (agents with DER or storage)
        self.rl_agents = [a for a in agents if a.is_prosumer]
        self.n_agents = len(self.rl_agents)

        # Per-agent forecasters (one per RL agent)
        self.forecasters = {a.name: NodalPriceForecaster(alpha=0.3, history_len=16)
                           for a in self.rl_agents}

        # Track SOC across blocks for observation freshness
        self.prev_soc: Dict[str, float] = {}

        # Observation dimension
        self.obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        """Compute flattened observation dimension per agent."""
        # 4 load + 4 pv/wind + 4 hist LMP + 4 forecast + 1 SOC + 1 avg LMP
        return 4 + 4 + 4 + 4 + 1 + 1

    def _get_agent_obs(self, agent: Agent, block_idx: int,
                       hist_lmp: np.ndarray, avg_lmp: float) -> np.ndarray:
        """Build observation vector for one agent at a decision block."""
        t_start = block_idx * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, self.T)

        # Local forecasts for the upcoming block
        if self.stage == "DA":
            load = agent.load_forecast[t_start:t_end]
            pv = agent.pv_forecast[t_start:t_end]
            wind = agent.get_wind_forecast()[t_start:t_end] if agent.has_wind else np.zeros(4)
        else:
            load = agent.load_real[t_start:t_end]
            pv = agent.pv_real[t_start:t_end]
            wind = agent.get_wind_real()[t_start:t_end] if agent.has_wind else np.zeros(4)

        # Pad to BLOCK_SIZE
        load = np.pad(load, (0, BLOCK_SIZE - len(load)), mode='edge')
        pv = np.pad(pv, (0, BLOCK_SIZE - len(pv)), mode='edge')
        wind = np.pad(wind, (0, BLOCK_SIZE - len(wind)), mode='edge')

        # Historical LMP (last 4 block averages)
        if hist_lmp is not None and len(hist_lmp) >= 4:
            lmp_hist = hist_lmp[-4:]
        else:
            lmp_hist = np.full(4, avg_lmp)

        # Price forecast for upcoming block
        fc = self.forecasters[agent.name]
        price_fc = np.array(fc.forecast(4))

        # SOC — use tracked SOC from market results, falling back to initial
        if agent.storage is not None:
            soc = np.array([self.prev_soc.get(agent.name, agent.storage.soc0)])
        else:
            soc = np.array([0.0])

        obs = np.concatenate([load, pv+wind, lmp_hist, price_fc, soc, [avg_lmp]])
        return obs.astype(np.float32)

    def reset(self, wholesale=None) -> Dict[str, np.ndarray]:
        """Reset environment for a new episode. Returns initial obs per agent."""
        if wholesale is None:
            from grid import day_ahead_price_china
            self.wholesale = day_ahead_price_china(self.T)
        else:
            self.wholesale = wholesale

        self.block_idx = 0
        self.prev_soc = {}
        self.hist_lmp = deque(maxlen=16)
        # Reset forecasters
        for fc in self.forecasters.values():
            fc._ema = None
            fc._history = []

        # Initialize with heuristic bidding
        self.current_actions = adaptive_bidding(
            self.all_agents, self.config, strategy="rl")

        obs = {}
        avg_lmp = np.mean(self.wholesale)
        for a in self.rl_agents:
            obs[a.name] = self._get_agent_obs(a, 0, None, avg_lmp)
        return obs

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray],
                                                       Dict[str, float],
                                                       bool, dict]:
        """Execute one decision block (4 periods).

        Parameters
        ----------
        actions : dict
            {agent_name: discrete_action_index}

        Returns
        -------
        obs : dict
            Next observations per agent.
        rewards : dict
            Per-agent rewards.
        done : bool
            Episode finished.
        info : dict
            Additional info (LMP, schedules, welfare).
        """
        block = self.block_idx
        t_start = block * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, self.T)

        # Apply RL actions to the action params for the upcoming block
        for a in self.rl_agents:
            nm = a.name
            if nm not in actions:
                continue
            bid_m, offer_a = action_to_params(actions[nm])
            # Set block-level strategy
            if nm not in self.current_actions:
                self.current_actions[nm] = {
                    "bid_mult": np.full(self.T, 1.0),
                    "offer_adder": np.full(self.T, 0.0),
                }
            self.current_actions[nm]["bid_mult"][t_start:t_end] = bid_m
            self.current_actions[nm]["offer_adder"][t_start:t_end] = offer_a

        # Run market clearing for the full day (simplified: run once after all
        # block actions are set; rewards are computed from the result)
        # For efficiency, we only clear once at the end of the episode
        # But for proper RL, we need per-step rewards
        # Strategy: clear market with current partial actions, extract per-agent payoff

        try:
            result = clear_market(self.all_agents, self.T, self.stage,
                                 self.current_actions, self.config)
        except Exception:
            result = None

        # Compute rewards from market result
        rewards = {}
        if result is not None:
            for a in self.rl_agents:
                nm = a.name
                sched = result["schedules"][nm]
                lmp_node = result["lmp"][:, a.bus]
                # Agent payoff for this block
                cons_val = a.bid_value * np.sum(sched["served"][t_start:t_end])
                gen_cost = a.offer_cost * np.sum(sched["pv_used"][t_start:t_end]
                                                  + sched["wind_used"][t_start:t_end])
                mkt_pmt = (np.sum(sched["p_sell"][t_start:t_end]
                                  * lmp_node[t_start:t_end])
                           - np.sum(sched["p_buy"][t_start:t_end]
                                    * lmp_node[t_start:t_end]))
                penalty = self.config.penalty_unserved * np.sum(sched["unserved"]
                                                                [t_start:t_end])
                rewards[nm] = float(cons_val - gen_cost + mkt_pmt - penalty)

                # Update forecaster with this block's average LMP
                block_lmp = np.mean(lmp_node[t_start:t_end])
                self.forecasters[nm].update(block_lmp)
                self.hist_lmp.append(block_lmp)

                # Track end-of-block SOC for next observation
                if a.storage and nm in result["schedules"]:
                    soc_arr = result["schedules"][nm]["soc"]
                    block_end = min((block + 1) * BLOCK_SIZE, self.T)
                    if block_end < self.T:
                        self.prev_soc[nm] = float(soc_arr[block_end])
                    elif len(soc_arr) > 0:
                        self.prev_soc[nm] = float(soc_arr[-1])
        else:
            for a in self.rl_agents:
                rewards[a.name] = 0.0

        self.block_idx += 1
        done = self.block_idx >= N_BLOCKS

        # Build next observations
        obs = {}
        avg_lmp = np.mean(self.wholesale) if self.wholesale is not None else 420.0
        hist_arr = np.array(list(self.hist_lmp)) if self.hist_lmp else None
        if not done:
            for a in self.rl_agents:
                obs[a.name] = self._get_agent_obs(a, self.block_idx,
                                                  hist_arr, avg_lmp)
        else:
            for a in self.rl_agents:
                obs[a.name] = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"lmp": result["lmp"] if result else None,
                "welfare": result["welfare"] if result else 0.0,
                "re_rate": result["re_consumption_rate"] if result else 0.0}

        return obs, rewards, done, info

    def get_state_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return N_ACTIONS
