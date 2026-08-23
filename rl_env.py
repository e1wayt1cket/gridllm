# rl_env.py
"""Gym-style RL environment for continuous-action agent bidding.

Each episode = one 96-period day. Agents act on 4-period blocks (24 steps).
Observation is a compact V2 vector (12-dim): block-mean load/RE and SOC
normalized to [0, 1], LMP expressed as ratios of the day-average price, block
position, load/RE ratio, opponent-bid statistics, plus price-prediction
features (LMP deviation from EMA, recent LMP trend, and a direct next-block
price forecast).
Action is continuous: (bid_mult in [0.3, 1.8], offer_adder in [0, 50]).

Optional differential reward: reward = raw profit - profit under truthful
bidding (bid=1.0, offer=0.0) for the same block, anchoring the truthful
policy at zero advantage and sharpening the critic gradient.

Supports both multi-agent (all agents are RL) and single-agent
(rl_agent_names filters to specific agents) modes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import Agent, MarketConfig
from market import adaptive_bidding
from price_forecaster import NodalPriceForecaster
from rl_spec import OBS_V3, ACTION_BID_OFFER_V1, ObservationSpec, ActionSpec

# Continuous action space bounds
BID_MULT_LOW, BID_MULT_HIGH = 0.3, 1.8
OFFER_ADDER_LOW, OFFER_ADDER_HIGH = 0.0, 50.0

# Decision granularity
BLOCK_SIZE = 4        # periods per decision block
N_BLOCKS = 24         # 96 / 4


class BiddingEnv:
    """Multi-agent bidding environment with continuous actions.

    Supports two modes:
    - Multi-agent (rl_agent_names=None): all agents are RL-controlled.
      Backward compatible with the original MATD3 pipeline.
    - Single-agent (rl_agent_names=["Agent_X"]): only the named agent
      uses RL actions; all other agents use fixed default strategy
      (bid_mult=1.0, offer_adder=0.0).

    Reward is raw profit in CNY (not normalized). Storage agents have
    cycle_cost deducted from their reward. When use_differential_reward is
    set, each block's reward is raw profit minus the truthful-bidding
    baseline profit for the same block.

    Parameters
    ----------
    agents : list of Agent
    config : MarketConfig
    stage : str
        "DA" or "RT".
    rl_agent_names : list of str or None
        Agent names to treat as RL learners. None = all agents.
    """

    def __init__(self, agents: List[Agent], config: MarketConfig,
                 stage: str = "DA", roll_horizon: int = 16,
                 rl_agent_names: Optional[List[str]] = None,
                 use_differential_reward: bool = False,
                 bid_mult_low: float = BID_MULT_LOW,
                 bid_mult_high: float = BID_MULT_HIGH,
                 obs_spec: Optional[ObservationSpec] = None,
                 action_spec: Optional[ActionSpec] = None):
        self.all_agents = agents
        self.config = config
        self.T = 96
        self.stage = stage
        self.wholesale = None
        self.roll_horizon = roll_horizon  # look-ahead periods per window

        # Per-agent action bounds (configurable per training run)
        self.bid_mult_low = bid_mult_low
        self.bid_mult_high = bid_mult_high

        # Pluggable observation/action layout. Defaults reproduce the V3
        # 12-dim observation and the (bid_mult, offer_adder) action space.
        self.obs_spec = obs_spec if obs_spec is not None else OBS_V3
        if action_spec is not None:
            self.action_spec = action_spec
        else:
            self.action_spec = ActionSpec(
                ACTION_BID_OFFER_V1.name,
                ACTION_BID_OFFER_V1.version,
                ACTION_BID_OFFER_V1.action_names,
                {"bid_mult": (bid_mult_low, bid_mult_high),
                 "offer_adder": ACTION_BID_OFFER_V1.bounds["offer_adder"]},
            )

        # Differential reward: per-block reward = raw profit - baseline profit,
        # where the baseline is the same window under truthful bidding for all
        # RL agents. Anchors the truthful policy at zero advantage.
        self.use_differential_reward = use_differential_reward

        if rl_agent_names is not None:
            self.rl_agents = [a for a in agents if a.name in rl_agent_names]
        else:
            self.rl_agents = list(agents)  # all agents are RL (backward compat)
        self.n_agents = len(self.rl_agents)

        self.forecasters = {
            a.name: NodalPriceForecaster(alpha=0.3, history_len=24)
            for a in self.rl_agents
        }
        self.prev_soc: Dict[str, float] = {}
        self.obs_dim = self.obs_spec.total_dim
        # Last market-clearing result, exposed for post-hoc analysis (e.g.
        # profit decomposition); not used by the training loop itself.
        self._last_result = None
        # Snapshot of all agents' actions from the previous block, used
        # for opponent-feature computation to avoid self-reference bias
        # when agents are evaluated sequentially within a block.
        self._prev_block_actions: Optional[Dict] = None
        # Cumulative RE consumption over committed periods of the day, used
        # to report a day-aggregate re_rate in step() info (not the last
        # rolling-window value).
        self._re_used = 0.0
        self._re_avail = 0.0
        # Per-agent (day-peak load, day-peak RE) for the stage, cached at reset
        # so the normalized load/RE observation features stay in [0, 1] without
        # recomputing the max every block.
        self._agent_peaks: Dict[str, Tuple[float, float]] = {}

    def get_action_bounds(self) -> torch.Tensor:
        """Return (low, high) bounds tensor for the continuous action space."""
        low = [self.action_spec.bounds[nm][0]
               for nm in self.action_spec.action_names]
        high = [self.action_spec.bounds[nm][1]
                for nm in self.action_spec.action_names]
        return torch.tensor([low, high], dtype=torch.float32)

    def get_state_dim(self) -> int:
        return self.obs_spec.total_dim

    @property
    def unique_obs_dim(self) -> int:
        """Per-agent observation features at the START of the obs vector.

        MATD3's centralized critic slices the first `unique_obs_dim` entries
        of each other agent's observation to build the global state, so the
        per-agent features must be contiguous at position 0.
        """
        return self.obs_spec.unique_dim

    @property
    def shared_obs_dim(self) -> int:
        """Shared observation features (LMP, price, system indicators)."""
        return self.obs_spec.shared_dim

    def _compute_obs_dim(self) -> int:
        """Flattened observation dimension per agent (retained shim)."""
        return self.obs_spec.total_dim

    # ------------------------------------------------------------------
    # Default actions for non-RL agents
    # ------------------------------------------------------------------

    def _default_actions(self) -> dict:
        """Return truthful/default bidding params for all agents.

        bid_mult=1.0, offer_adder=0.0 — the baseline strategy used by
        non-RL agents during single-agent training and by the evaluation
        framework for baseline comparison.
        """
        return {
            a.name: {
                "bid_mult": np.full(self.T, 1.0, dtype=float),
                "offer_adder": np.full(self.T, 0.0, dtype=float),
            }
            for a in self.all_agents
        }

    # ------------------------------------------------------------------
    # Scenario switching
    # ------------------------------------------------------------------

    def set_agents(self, agents: List[Agent],
                   wholesale: Optional[np.ndarray] = None):
        """Replace the agent population for scenario switching.

        Rebuilds per-RL-agent forecasters and resets SOC/history state.
        The RL agent list is re-filtered from the new population using
        the names originally passed to __init__.

        Call this between episodes when cycling scenarios.
        """
        self.all_agents = agents
        # Re-filter RL agents from the new population
        rl_names = {a.name for a in self.rl_agents}
        self.rl_agents = [a for a in agents if a.name in rl_names]
        self.n_agents = len(self.rl_agents)
        # Rebuild forecasters for RL agents (new profiles = fresh EMAs)
        self.forecasters = {
            a.name: NodalPriceForecaster(alpha=0.3, history_len=24)
            for a in self.rl_agents
        }
        if wholesale is not None:
            self.wholesale = wholesale
        self._agent_peaks = self._compute_agent_peaks()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _compute_agent_peaks(self) -> Dict[str, Tuple[float, float]]:
        """Per-agent (day-peak load, day-peak RE) for the current stage.

        Load and RE are normalized against their OWN day peaks: a prosumer's
        PV/wind capacity can exceed its load several-fold, so normalizing RE by
        the load peak would peg the RE feature at 1.0 for most daylight periods.
        """
        peaks = {}
        for a in self.all_agents:
            load_src = a.load_forecast if self.stage == "DA" else a.load_real
            re_src = (a.pv_forecast if self.stage == "DA" else a.pv_real)
            if a.has_wind:
                re_src = re_src + (a.wind_forecast if self.stage == "DA"
                                   else a.wind_real)
            peaks[a.name] = (float(np.max(load_src)),
                             float(np.max(re_src)))
        return peaks

    def _compute_obs_features(self, agent: Agent, block_idx: int,
                              hist_lmp: np.ndarray | None,
                              avg_lmp: float,
                              system_load_re_ratio: float = 1.0,
                              avg_other_bid: float = 1.0,
                              bid_std: float = 0.0) -> Dict[str, float]:
        """Compute all named observation features for one agent.

        Returns the superset of features known to the environment, keyed by
        feature name; `_get_agent_obs` selects the subset the active
        ObservationSpec declares. Load/RE are the block means (the action
        commits a whole block), each divided by its own day peak (load peak
        for load, RE peak for RE), so they stay in [0, 1] without saturating;
        absolute LMP levels are expressed as ratios of the day-average price
        so all features are O(1) for the MLP.
        """
        t_start = block_idx * BLOCK_SIZE
        t_block_end = min(t_start + BLOCK_SIZE, self.T)
        n_block = t_block_end - t_start

        if self.stage == "DA":
            load = agent.load_forecast[t_start:t_block_end]
            pv = agent.pv_forecast[t_start:t_block_end]
            wind = agent.get_wind_forecast()[t_start:t_block_end] \
                if agent.has_wind else np.zeros(n_block)
        else:
            load = agent.load_real[t_start:t_block_end]
            pv = agent.pv_real[t_start:t_block_end]
            wind = agent.get_wind_real()[t_start:t_block_end] \
                if agent.has_wind else np.zeros(n_block)

        load_peak, re_peak = self._agent_peaks.get(agent.name, (1.0, 1.0))
        load_feat = float(np.clip(
            np.mean(load) / max(load_peak, 1e-6), 0.0, 1.0))
        re_feat = float(np.clip(
            np.mean(pv + wind) / max(re_peak, 1e-6), 0.0, 1.0))

        # Last observed LMP, or the day average if none is available yet
        if hist_lmp is not None and len(hist_lmp) > 0:
            last_lmp = float(hist_lmp[-1])
        else:
            last_lmp = avg_lmp

        # SOC
        if agent.storage is not None:
            soc = float(self.prev_soc.get(agent.name, agent.storage.soc0))
        else:
            soc = 0.0

        # Positional encoding — normalize block index to [0, 1]
        block_pos = block_idx / float(N_BLOCKS - 1)

        # Price-prediction features from the per-agent forecaster: how far the
        # current LMP deviates from its EMA, and the recent LMP slope. Both are
        # normalized to stay O(1) and clipped for stability; the forecaster's
        # forecast() itself returns EMA persistence, so the deviation already
        # carries that signal and only a history slope adds new information.
        fc = self.forecasters[agent.name]
        ema = fc._ema if fc._ema is not None else last_lmp
        ema_dev = (last_lmp - ema) / ema if ema != 0 else 0.0
        trend = 0.0
        if len(fc._history) >= 2:
            k = min(4, len(fc._history))
            pts = fc._history[-k:]
            slope = np.polyfit(np.arange(k), pts, 1)[0]
            trend = slope / ema if ema != 0 else 0.0
        ema_dev = float(np.clip(ema_dev, -1.0, 1.0))
        trend = float(np.clip(trend, -1.0, 1.0))

        # Absolute price levels as O(1) ratios of the day average; the
        # negative tail (excess-RE prices) is preserved by a lower clip.
        price_ref = avg_lmp if avg_lmp > 0 else 420.0
        last_lmp_norm = float(np.clip(last_lmp / price_ref, -1.0, 3.0))
        avg_lmp_norm = float(np.clip(avg_lmp / 420.0, 0.0, 3.0))
        slr_norm = float(min(system_load_re_ratio, 5.0))

        # Direct next-block price forecast (EMA persistence) from the same
        # forecaster, normalized on the same scale as the realized LMP.
        pred_lmp = fc.forecast(1)[0]
        pred_lmp_norm = float(np.clip(pred_lmp / price_ref, -1.0, 3.0))

        return {
            "load_feat": load_feat,
            "re_feat": re_feat,
            "soc": soc,
            "last_lmp_norm": last_lmp_norm,
            "block_pos": block_pos,
            "avg_lmp_norm": avg_lmp_norm,
            "slr_norm": slr_norm,
            "avg_other_bid": avg_other_bid,
            "bid_std": bid_std,
            "ema_dev": ema_dev,
            "price_trend": trend,
            "pred_lmp_norm": pred_lmp_norm,
        }

    def _get_agent_obs(self, agent: Agent, block_idx: int,
                       hist_lmp: np.ndarray | None,
                       avg_lmp: float,
                       system_load_re_ratio: float = 1.0,
                       avg_other_bid: float = 1.0,
                       bid_std: float = 0.0) -> np.ndarray:
        """Build the observation vector for one agent from the active spec.

        Order matters: the per-agent features must be the first entries so
        MATD3's centralized critic can slice them out when building the global
        state.
        """
        feats = self._compute_obs_features(
            agent, block_idx, hist_lmp, avg_lmp, system_load_re_ratio,
            avg_other_bid, bid_std)
        return np.array(
            [feats[f.name]
             for f in (*self.obs_spec.unique_features,
                       *self.obs_spec.shared_features)],
            dtype=np.float32)

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

    def _committed_re_avail(self, t_start: int, n_commit: int) -> float:
        """RE energy available over the committed slice of the current window.

        Mirrors the availability accounting in dispatch (sum pv/wind forecast
        over prosumer and wind-capable agents), but only for the periods this
        block commits, so the accumulated day-aggregate rate is exact.
        """
        avail = 0.0
        for a in self.all_agents:
            if self.stage == "DA":
                if a.is_prosumer:
                    avail += float(np.sum(np.maximum(
                        a.pv_forecast[t_start:t_start + n_commit], 0)))
                if a.has_wind:
                    avail += float(np.sum(np.maximum(
                        a.wind_forecast[t_start:t_start + n_commit], 0)))
            else:
                if a.is_prosumer:
                    avail += float(np.sum(np.maximum(
                        a.pv_real[t_start:t_start + n_commit], 0)))
                if a.has_wind:
                    avail += float(np.sum(np.maximum(
                        a.wind_real[t_start:t_start + n_commit], 0)))
        return avail * 0.25  # dt_h

    def _compute_opponent_features(self, agent_name: str) -> tuple:
        """Return (avg_other_bid_mult, bid_mult_std) from previous-block snapshot.

        Iterates over self.all_agents (not just rl_agents) so that the RL
        agent can observe what ALL other market participants bid, including
        non-RL agents using default values.
        """
        src = self._prev_block_actions
        if src is None:
            # Fallback: use current actions if no previous block exists (block 0)
            src = getattr(self, 'current_actions', None)
        if not src:
            return 1.0, 0.0
        other_bids = []
        all_bids = []
        for a in self.all_agents:
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

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, wholesale: Optional[np.ndarray] = None) \
            -> Dict[str, np.ndarray]:
        if wholesale is None:
            from grid import day_ahead_price_china
            self.wholesale = day_ahead_price_china(
                self.T, agents=self.all_agents, config=self.config)
        else:
            self.wholesale = wholesale

        self.block_idx = 0
        self.prev_soc = {}
        self.hist_lmp = deque(maxlen=96)
        self._re_used = 0.0
        self._re_avail = 0.0
        self._agent_peaks = self._compute_agent_peaks()
        for fc in self.forecasters.values():
            fc._ema = None
            fc._history = []

        # Initialize all agents with default actions
        self.current_actions = self._default_actions()

        obs = {}
        avg_lmp = np.mean(self.wholesale)
        slr = self._compute_system_load_re_ratio(0)
        for a in self.rl_agents:
            avg_oth, bid_std = self._compute_opponent_features(a.name)
            obs[a.name] = self._get_agent_obs(
                a, 0, None, avg_lmp, slr, avg_oth, bid_std)
        return obs

    def _agent_block_profit(self, schedule: dict, lmp_node: np.ndarray,
                            agent: Agent, n_periods: int) -> float:
        """Profit (CNY) for one agent summed over committed periods.

        consumer surplus - generation cost + net market payment
        - unserved penalty - storage cycle cost.
        """
        profit = 0.0
        cycle_cost = float(self.config.storage.cycle_cost)
        for d in range(n_periods):
            cons_val = agent.bid_value * schedule["served"][d]
            gen_cost = agent.offer_cost * (schedule["pv_used"][d]
                                           + schedule["wind_used"][d])
            mkt_pmt = (schedule["p_sell"][d] * lmp_node[d]
                       - schedule["p_buy"][d] * lmp_node[d])
            penalty = self.config.market_design.penalty_unserved \
                * schedule["unserved"][d]
            step_reward = float(cons_val - gen_cost + mkt_pmt - penalty)
            if agent.storage is not None:
                step_reward -= cycle_cost * (schedule["p_ch"][d]
                                             + schedule["p_dis"][d])
            profit += step_reward
        return profit

    def step(self, actions: Dict[str, np.ndarray]) \
            -> Tuple[Dict[str, np.ndarray],
                     Dict[str, float], bool, dict]:
        """Execute one decision block with rolling-window OPF.

        Parameters
        ----------
        actions : dict
            {agent_name: np.array([bid_mult, offer_adder])}
            Only RL agents need to provide actions; non-RL agents use
            defaults set during reset().

        Returns
        -------
        obs, rewards, done, info
        """
        from market import _make_window_agents, clear_market
        from grid import day_ahead_price_china
        import dataclasses

        block = self.block_idx
        t_start = block * BLOCK_SIZE
        t_end = min(t_start + BLOCK_SIZE, self.T)

        # Update current_actions with this block's RL decisions
        for a in self.rl_agents:
            nm = a.name
            if nm not in actions:
                continue
            act = actions[nm]
            bid_m = float(np.clip(act[0], self.bid_mult_low, self.bid_mult_high))
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

        # One wholesale curve per block, shared by the real clear and the
        # differential-reward baseline re-clear. Regenerating it inside each
        # clear_market call would draw different unseeded price noise for the
        # two clears, corrupting reward = profit(price A) - profit(price B).
        wholesale = day_ahead_price_china(
            window_T, agents=window_agents, config=window_config)

        try:
            result = clear_market(window_agents, window_T, self.stage,
                                  window_actions, window_config,
                                  wholesale=wholesale)
        except Exception:
            result = None
        self._last_result = result

        # Surface silent degradation: a fallback to the single-period solver
        # bypasses the RL bidding mechanism; a None result zeroes rewards.
        if result is not None and result.get("fell_back"):
            print(f"[rl_env] WARNING: block {block} clear_market fell back to "
                  f"single-period solver; RL bids bypassed.", flush=True)
        elif result is None:
            print(f"[rl_env] WARNING: block {block} clear_market returned "
                  f"None; rewards set to 0.", flush=True)

        # ---- Differential-reward baseline ----
        # Baseline = this same window under truthful bidding (bid=1.0,
        # offer=0.0) for every RL agent's committed block, with opponents kept
        # at their RL actions and identical prev_soc. It must NOT advance
        # prev_soc or the forecasters; it only supplies the profit to subtract.
        base_rewards = {}
        if self.use_differential_reward and result is not None:
            base_window_agents = _make_window_agents(
                self.all_agents, t_start, window_end, self.prev_soc,
                self.current_actions)
            base_actions = {}
            for nm, act in window_actions.items():
                base_actions[nm] = {k: v.copy() for k, v in act.items()}
            for a in self.rl_agents:
                nm = a.name
                if nm in base_actions:
                    base_actions[nm]["bid_mult"][:n_commit] = 1.0
                    base_actions[nm]["offer_adder"][:n_commit] = 0.0
            try:
                base_result = clear_market(
                    base_window_agents, window_T, self.stage,
                    base_actions, window_config, wholesale=wholesale)
            except Exception:
                base_result = None
            if base_result is not None and not base_result.get("fell_back"):
                for a in self.rl_agents:
                    nm = a.name
                    ws = base_result["schedules"].get(nm)
                    if ws is None:
                        base_rewards[nm] = 0.0
                        continue
                    base_rewards[nm] = self._agent_block_profit(
                        ws, base_result["lmp"][:, a.bus], a, n_commit)
            else:
                print(f"[rl_env] WARNING: block {block} differential baseline "
                      f"unavailable; falling back to unshaped rewards.",
                      flush=True)

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

                raw_reward = self._agent_block_profit(ws, lmp_node, a, n_commit)

                # Differential reward: profit above the truthful baseline
                rewards[nm] = raw_reward - base_rewards.get(nm, 0.0)

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

        # ---- Accumulate committed-period RE for the day-aggregate rate ----
        if result is not None:
            for a in self.all_agents:
                ws = result["schedules"].get(a.name)
                if ws is None:
                    continue
                self._re_used += float(np.sum(
                    (ws["pv_used"][:n_commit] + ws["wind_used"][:n_commit])
                    * 0.25))
            self._re_avail += self._committed_re_avail(t_start, n_commit)

        self.block_idx += 1
        done = self.block_idx >= N_BLOCKS

        # ---- Build next observation ----
        obs = {}
        avg_lmp = np.mean(self.wholesale) if self.wholesale is not None \
            else 420.0
        hist_arr = np.array(list(self.hist_lmp)) \
            if self.hist_lmp else None
        slr = self._compute_system_load_re_ratio(t_start)
        if not done:
            for a in self.rl_agents:
                avg_oth, bid_std = self._compute_opponent_features(a.name)
                obs[a.name] = self._get_agent_obs(
                    a, self.block_idx, hist_arr, avg_lmp, slr,
                    avg_oth, bid_std)
        else:
            for a in self.rl_agents:
                obs[a.name] = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"lmp": result["lmp"] if result else None,
                "welfare": result["welfare"] if result else 0.0,
                "re_rate": (self._re_used / self._re_avail * 100.0)
                           if self._re_avail > 0 else 100.0}

        return obs, rewards, done, info
