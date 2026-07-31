# rl_env.py
"""Gym-style RL environment for continuous-action agent bidding.

Each episode = one 96-period day. Agents act on 4-period blocks (24 steps).
State includes 24h-ahead local forecasts, LMP history, price predictions,
SOC, and system-level indicators.
Action is continuous: (bid_mult in [0.3, 1.8], offer_adder in [0, 50]).
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import Agent, MarketConfig
from market import adaptive_bidding
from price_forecaster import NodalPriceForecaster

# Continuous action space bounds
BID_MULT_LOW, BID_MULT_HIGH = 0.3, 1.8
OFFER_ADDER_LOW, OFFER_ADDER_HIGH = 0.0, 50.0

# Decision granularity
BLOCK_SIZE = 4        # periods per decision block
N_BLOCKS = 24         # 96 / 4
LOOKAHEAD_BLOCKS = 6  # 24 periods ahead for observations


class RunningMoments:
    """EMA-tracked mean and variance for dynamic reward normalization.

    Normalizes per-agent rewards to zero-mean unit-variance so that
    PPO training is robust to changes in reward magnitude across
    scenarios without a hand-tuned reward_scale.
    """

    def __init__(self, alpha: float = 0.01):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.alpha = alpha
        self._n: int = 0

    def update(self, x: float):
        self._n += 1
        delta = x - self.mean
        self.mean += self.alpha * delta
        # Bias-corrected variance: use (1 - alpha) on var, alpha on delta²
        self.var = (1.0 - self.alpha) * self.var + self.alpha * delta * delta

    def normalize(self, x: float) -> float:
        return (x - self.mean) / (np.sqrt(max(self.var, 1e-8)) + 1e-8)


class BiddingEnv:
    """Multi-agent bidding environment with continuous actions.

    Each agent trains its own policy. The environment processes all agents
    jointly through a single market clearing call per block.

    Parameters
    ----------
    agents : list of Agent
    config : MarketConfig
    stage : str
        "DA" or "RT".
    """

    def __init__(self, agents: List[Agent], config: MarketConfig,
                 stage: str = "DA", roll_horizon: int = 16):
        self.all_agents = agents
        self.config = config
        self.T = 96
        self.stage = stage
        self.wholesale = None
        self.roll_horizon = roll_horizon  # look-ahead periods per window

        self.rl_agents = list(agents)  # all agents learn, not just prosumers
        self.n_agents = len(self.rl_agents)

        self.forecasters = {
            a.name: NodalPriceForecaster(alpha=0.3, history_len=24)
            for a in self.rl_agents
        }
        self.reward_norms = {
            a.name: RunningMoments(alpha=0.01) for a in self.rl_agents
        }
        self.prev_soc: Dict[str, float] = {}
        self.obs_dim = self._compute_obs_dim()
        # Snapshot of all agents' actions from the previous block, used
        # for opponent-feature computation to avoid self-reference bias
        # when agents are evaluated sequentially within a block.
        self._prev_block_actions: Optional[Dict] = None

    def get_action_bounds(self) -> torch.Tensor:
        """Return (low, high) bounds tensor for the 2-dim continuous action."""
        return torch.tensor([[BID_MULT_LOW, OFFER_ADDER_LOW],
                             [BID_MULT_HIGH, OFFER_ADDER_HIGH]],
                            dtype=torch.float32)

    def get_state_dim(self) -> int:
        return self.obs_dim

    @property
    def unique_obs_dim(self) -> int:
        """Per-agent observation features (load, re_gen, soc)."""
        L = LOOKAHEAD_BLOCKS * BLOCK_SIZE
        return L + L + 1  # load(24) + re_gen(24) + soc(1) = 49

    @property
    def shared_obs_dim(self) -> int:
        """Shared observation features (LMP, price, system indicators)."""
        return self.obs_dim - self.unique_obs_dim  # 103 - 49 = 54

    def _compute_obs_dim(self) -> int:
        """Flattened observation dimension per agent."""
        L = LOOKAHEAD_BLOCKS * BLOCK_SIZE  # 24 periods
        # 4*24 (load/re_gen/lmp_hist/price_fc) + soc + avg_lmp + load_re_ratio
        # + congestion_idx + block_pos + 2 opponent features = 103
        return L + L + L + L + 1 + 1 + 1 + 1 + 1 + 2

    def _get_agent_obs(self, agent: Agent, block_idx: int,
                       hist_lmp: np.ndarray | None,
                       avg_lmp: float,
                       system_load_re_ratio: float = 1.0,
                       congestion_idx: float = 0.0,
                       avg_other_bid: float = 1.0,
                       bid_std: float = 0.0) -> np.ndarray:
        """Build observation vector for one agent at a decision block."""
        t_start = block_idx * BLOCK_SIZE
        lookahead = LOOKAHEAD_BLOCKS * BLOCK_SIZE  # 24 periods
        t_end = min(t_start + lookahead, self.T)
        n_actual = t_end - t_start

        if self.stage == "DA":
            load = agent.load_forecast[t_start:t_end]
            pv = agent.pv_forecast[t_start:t_end]
            wind = agent.get_wind_forecast()[t_start:t_end] \
                if agent.has_wind else np.zeros(n_actual)
        else:
            load = agent.load_real[t_start:t_end]
            pv = agent.pv_real[t_start:t_end]
            wind = agent.get_wind_real()[t_start:t_end] \
                if agent.has_wind else np.zeros(n_actual)

        re_gen = pv + wind

        # Pad to LOOKAHEAD periods
        load = np.pad(load, (0, lookahead - len(load)), mode="edge")
        re_gen = np.pad(re_gen, (0, lookahead - len(re_gen)), mode="edge")

        # Historical LMP
        if hist_lmp is not None and len(hist_lmp) >= lookahead:
            lmp_hist = hist_lmp[-lookahead:]
        else:
            lmp_hist = np.full(lookahead, avg_lmp)

        # Price forecast
        fc = self.forecasters[agent.name]
        price_fc = np.array(fc.forecast(lookahead))
        if len(price_fc) < lookahead:
            price_fc = np.pad(price_fc, (0, lookahead - len(price_fc)),
                              mode="edge")

        # SOC
        if agent.storage is not None:
            soc = np.array([self.prev_soc.get(agent.name, agent.storage.soc0)])
        else:
            soc = np.array([0.0])

        # Positional encoding — normalize block index to [0, 1]
        block_pos = np.array([block_idx / float(N_BLOCKS - 1)])

        obs = np.concatenate([
            load, re_gen, lmp_hist, price_fc, soc,
            [avg_lmp], [system_load_re_ratio], [congestion_idx],
            block_pos, [avg_other_bid], [bid_std],
        ])
        return obs.astype(np.float32)

    def _compute_system_load_re_ratio(self, t_start: int) -> float:
        t_end = min(t_start + BLOCK_SIZE, self.T)
        total_load = 0.0
        total_re = 0.0
        for a in self.all_agents:
            if self.stage == "DA":
                total_load += np.sum(a.load_forecast[t_start:t_end])
                if a.is_prosumer:
                    total_re += np.sum(np.maximum(
                        a.pv_forecast[t_start:t_end], 0))
                if a.has_wind:
                    total_re += np.sum(np.maximum(
                        a.wind_forecast[t_start:t_end], 0))
            else:
                total_load += np.sum(a.load_real[t_start:t_end])
                if a.is_prosumer:
                    total_re += np.sum(np.maximum(
                        a.pv_real[t_start:t_end], 0))
                if a.has_wind:
                    total_re += np.sum(np.maximum(
                        a.wind_real[t_start:t_end], 0))
        return float(total_load / max(total_re, 1e-6))

    def _compute_opponent_features(self, agent_name: str) -> tuple:
        """Return (avg_other_bid_mult, bid_mult_std) from previous-block snapshot.

        Uses _prev_block_actions (snapshot from the last completed block)
        rather than current_actions (which is partially filled when agents
        are evaluated sequentially within a block). This avoids a self-reference
        bias where later agents see already-updated actions of earlier agents.
        """
        src = self._prev_block_actions
        if src is None:
            # Fallback: use current actions if no previous block exists (block 0)
            src = getattr(self, 'current_actions', None)
        if not src:
            return 1.0, 0.0
        other_bids = []
        all_bids = []
        for a in self.rl_agents:
            nm = a.name
            if nm in src:
                bm = float(src[nm].get("bid_mult",
                           np.array([1.0])).flat[0])
                all_bids.append(bm)
                if nm != agent_name:
                    other_bids.append(bm)
        avg_other = float(np.mean(other_bids)) if other_bids else 1.0
        bid_std = float(np.std(all_bids)) if len(all_bids) >= 2 else 0.0
        return avg_other, bid_std

    def reset(self, wholesale=None) -> Dict[str, np.ndarray]:
        if wholesale is None:
            from grid import day_ahead_price_china
            self.wholesale = day_ahead_price_china(
                self.T, agents=self.all_agents, config=self.config)
        else:
            self.wholesale = wholesale

        self.block_idx = 0
        self.prev_soc = {}
        self.hist_lmp = deque(maxlen=96)
        for fc in self.forecasters.values():
            fc._ema = None
            fc._history = []

        self.current_actions = adaptive_bidding(
            self.all_agents, self.config, strategy="rl")

        obs = {}
        avg_lmp = np.mean(self.wholesale)
        slr = self._compute_system_load_re_ratio(0)
        for a in self.rl_agents:
            avg_oth, bid_std = self._compute_opponent_features(a.name)
            obs[a.name] = self._get_agent_obs(
                a, 0, None, avg_lmp, slr, 0.0, avg_oth, bid_std)
        return obs

    def step(self, actions: Dict[str, np.ndarray]) \
            -> Tuple[Dict[str, np.ndarray],
                     Dict[str, float], bool, dict]:
        """Execute one decision block with rolling-window OPF.

        Only a window of self.roll_horizon periods is solved, preventing
        the OPF from using perfect foresight of all 96 periods to cheat
        on storage SOC trajectories.

        Parameters
        ----------
        actions : dict
            {agent_name: np.array([bid_mult, offer_adder])}

        Returns
        -------
        obs, rewards, done, info
        """
        from market import _make_window_agents, clear_market
        import dataclasses

        block = self.block_idx
        t_start = block * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, self.T)

        # Update current_actions with this block's decisions
        for a in self.rl_agents:
            nm = a.name
            if nm not in actions:
                continue
            act = actions[nm]
            bid_m = float(np.clip(act[0], BID_MULT_LOW, BID_MULT_HIGH))
            offer_a = float(np.clip(act[1], OFFER_ADDER_LOW, OFFER_ADDER_HIGH))
            if nm not in self.current_actions:
                self.current_actions[nm] = {
                    "bid_mult": np.full(self.T, 1.0),
                    "offer_adder": np.full(self.T, 0.0),
                }
            self.current_actions[nm]["bid_mult"][t_start:t_end] = bid_m
            self.current_actions[nm]["offer_adder"][t_start:t_end] = offer_a

        # Snapshot this block's actions for opponent features in the next block.
        self._prev_block_actions = {
            nm: {"bid_mult": act["bid_mult"].copy(),
                 "offer_adder": act["offer_adder"].copy()}
            for nm, act in self.current_actions.items()
        }

        # ---- Rolling-window OPF ----
        window_end = min(t_start + self.roll_horizon, self.T)
        window_T = window_end - t_start
        n_commit = min(BLOCK_SIZE, window_T)

        # Slice agents and actions to the window
        window_agents = _make_window_agents(
            self.all_agents, t_start, window_end, self.prev_soc,
            self.current_actions)

        # Slice action params to window length
        window_actions = {}
        for nm, act in self.current_actions.items():
            window_actions[nm] = {
                k: v[t_start:window_end].copy()
                for k, v in act.items()
            }

        window_config = dataclasses.replace(self.config, verbose=False)

        try:
            result = clear_market(window_agents, window_T, self.stage,
                                  window_actions, window_config)
        except Exception:
            result = None

        # ---- Extract committed-period rewards and SOC ----
        rewards = {}
        if result is not None:
            for a in self.rl_agents:
                nm = a.name
                ws = result["schedules"].get(nm)
                if ws is None:
                    rewards[nm] = 0.0
                    continue
                lmp_node = result["lmp"][:, a.bus]

                raw_reward = 0.0
                for d in range(n_commit):
                    cons_val = a.bid_value * ws["served"][d]
                    gen_cost = a.offer_cost * (ws["pv_used"][d]
                                               + ws["wind_used"][d])
                    mkt_pmt = (ws["p_sell"][d] * lmp_node[d]
                               - ws["p_buy"][d] * lmp_node[d])
                    penalty = self.config.market_design.penalty_unserved \
                        * ws["unserved"][d]
                    raw_reward += float(cons_val - gen_cost + mkt_pmt - penalty)

                # Normalize reward through RunningMoments
                norm = self.reward_norms[nm]
                norm.update(raw_reward)
                rewards[nm] = norm.normalize(raw_reward)

                # Update forecaster and LMP history from committed period
                block_lmp = float(np.mean(lmp_node[:n_commit]))
                self.forecasters[nm].update(block_lmp)
                for d in range(n_commit):
                    self.hist_lmp.append(float(np.mean(lmp_node[d])))

                # Update prev_soc from last committed period
                if a.storage and nm in result["schedules"]:
                    soc_arr = result["schedules"][nm].get("soc")
                    if soc_arr is not None and len(soc_arr) > n_commit:
                        self.prev_soc[nm] = float(soc_arr[n_commit])
                    elif soc_arr is not None and len(soc_arr) > 0:
                        self.prev_soc[nm] = float(soc_arr[-1])
        else:
            for a in self.rl_agents:
                rewards[a.name] = 0.0

        self.block_idx += 1
        done = self.block_idx >= N_BLOCKS

        # ---- Build next observation ----
        obs = {}
        avg_lmp = np.mean(self.wholesale) if self.wholesale is not None \
            else 420.0
        hist_arr = np.array(list(self.hist_lmp)) \
            if self.hist_lmp else None
        if result is not None and result["lmp"] is not None:
            # Congestion index from the last period of the committed block
            last_lmp = result["lmp"][min(n_commit - 1, window_T - 1)]
            cong_idx = float(np.std(last_lmp) / max(np.mean(last_lmp), 1e-6))
        else:
            cong_idx = 0.0
        slr = self._compute_system_load_re_ratio(t_start)
        if not done:
            for a in self.rl_agents:
                avg_oth, bid_std = self._compute_opponent_features(a.name)
                obs[a.name] = self._get_agent_obs(
                    a, self.block_idx, hist_arr, avg_lmp, slr, cong_idx,
                    avg_oth, bid_std)
        else:
            for a in self.rl_agents:
                obs[a.name] = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"lmp": result["lmp"] if result else None,
                "welfare": result["welfare"] if result else 0.0,
                "re_rate": result["re_consumption_rate"] if result else 0.0}

        return obs, rewards, done, info
