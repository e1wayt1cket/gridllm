# rl_env.py
"""Gym-style RL environment for continuous-action agent bidding.

Each episode = one 96-period day. Agents act on 4-period blocks (24 steps).
Observation is a compact V2 vector (12-dim): block-mean load/RE and SOC
normalized to [0, 1], LMP expressed as ratios of the day-average price, block
position, load/RE ratio, opponent-bid statistics, plus price-prediction
features (LMP deviation from EMA, recent LMP trend, and a direct next-block
price forecast).
The action space is the acting agent's `ActionSpec`: an independent battery
declares its two reservation prices in CNY/MWh, and a load or prosumer
declares the multiplier/adder pair its own `bid_value`/`offer_cost` live in.

Optional differential reward: reward = raw profit - profit under truthful
bidding for the same block, anchoring the truthful policy at zero advantage
and sharpening the critic gradient. The truthful point is each agent's own
spec's, not a literal: it is (0, 0) for a battery's two declared prices and
(1, 0) for a load's multiplier pair.

Supports both multi-agent (all agents are RL) and single-agent
(rl_agent_names filters to specific agents) modes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque

import money
import participant_payoff

from models import Agent, MarketConfig
from market import adaptive_bidding
from price_forecaster import NodalPriceForecaster
from rl_spec import (OBS_V4, ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2,
                    ObservationSpec, ActionSpec, action_spec_for)

#: Frozen reference price, CNY/MWh, used to put every price feature on an O(1)
#: scale and to initialize them before any LMP has been observed.
#:
#: **It is a constant on purpose.** The observation must be a function of the
#: past only, so no episode-derived quantity may stand in for it -- that is what
#: V3 did, using the mean of the whole day's realized price as the denominator
#: of `last_lmp_norm` and `pred_lmp_norm` and as the block-0 value of
#: `last_lmp`, which made four of the twelve features carry whole-day
#: information at the first decision.
#:
#: **Provenance.** Calibrated on 2026-10-13 from the **training split only** of
#: the synthetic family `fam_a` via `synthetic_scenarios.calibrate_price_ref`:
#: each training curve's daily mean, then a representative price per family,
#: then an equal-weighted mean across families. 11 training curves, one family.
#: Artifact `data/wholesale/price_ref_calibration.json` (sha256
#: 400200a36d0a5668f6388070add98dbe28305cb7a88b8e5819974971d89073), data digest
#: `89e1ea4afc2c3288de65dc4bd6ce699a39182b2622aba62a6fb425750f1a3eef`.
#:
#: The validation split, the test split and the single real day `261009` were
#: **excluded from the calibration**. The value is frozen: it must not be
#: re-fitted, because a scale re-derived per split leaks across splits even when
#: no day is shared. Re-calibrating is a deliberate act that overwrites this
#: constant and records a new artifact, not something a run does implicitly.
#:
#: It replaces the inherited `420.0`, which was the pre-V4 fallback denominator
#: and was never verified against any distribution.
PRICE_REF_CNY_MWH = 288.19757259048833

# Bounds of the non-storage action space, read off the spec rather than
# restated. A second copy here would be a box the policies could be trained in
# while the clearing reads a different one; these names survive because older
# callers import them.
BID_MULT_LOW, BID_MULT_HIGH = ACTION_BID_OFFER_V1.bounds["bid_mult"]
OFFER_ADDER_LOW, OFFER_ADDER_HIGH = ACTION_BID_OFFER_V1.bounds["offer_adder"]

# Decision granularity
BLOCK_SIZE = 4        # periods per decision block
N_BLOCKS = 24         # 96 / 4


def bounds_tensor(spec: ActionSpec) -> torch.Tensor:
    """A spec's (low, high) bounds as a tensor, in `action_names` order.

    The actor parameterizes an action as `mid + tanh(logit) * half`, so a
    policy is built in one agent's box. Deriving that box from the spec here
    keeps the two from drifting apart.
    """
    low = [spec.bounds[n][0] for n in spec.action_names]
    high = [spec.bounds[n][1] for n in spec.action_names]
    return torch.tensor([low, high], dtype=torch.float32)

# Schedule keys stitched when assembling a captured day (mirrors the keys
# every clear engine returns per agent).
CAPTURE_SCHED_KEYS = ("p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
                      "pv_used", "wind_used")


class DayCapture:
    """Stitch the committed periods of successive rolling clears into one day.

    A rolling-horizon episode solves a look-ahead window per block and commits
    only its first ``BLOCK_SIZE`` periods; the rest are re-planned by the next
    window. A captured day is therefore the concatenation of those committed
    slices, not one joint optimization over the day.

    Training (per episode) and evaluation (per checkpoint) both record through
    this class, so a consumer metric computed during training is the same
    quantity as the one written to the evaluation CSV rather than a second
    implementation that can drift.

    ``finish()`` reports ``complete`` and ``fell_backs`` so callers can refuse
    to compute metrics over un-captured periods: a block whose solve returned
    None leaves zeros, and a fallback clear solves single-period and bypasses
    the RL bidding mechanism entirely.
    """

    def __init__(self, agents, T: int, roll_horizon: int):
        self.T = int(T)
        self.roll_horizon = int(roll_horizon)
        names = [a.name for a in agents]
        self.sched = {nm: {k: np.zeros(self.T) for k in CAPTURE_SCHED_KEYS}
                      for nm in names}
        self.lmp = None
        self.wholesale = np.zeros(self.T)
        self.declared = {}
        self.fell_backs = 0
        self.periods_captured = 0
        self.blocks_captured = 0

    @classmethod
    def from_env(cls, env) -> "DayCapture":
        return cls(env.all_agents, env.T, env.roll_horizon)

    def committed_periods(self, t_start: int) -> int:
        """Periods a window starting at ``t_start`` commits to the day."""
        return min(BLOCK_SIZE, min(t_start + self.roll_horizon, self.T) - t_start)

    @staticmethod
    def committed_periods_for(env, t_start: int) -> int:
        """Same count read off an env, for callers holding the env not a
        capture. Keeps the arithmetic in one place."""
        return min(BLOCK_SIZE, min(t_start + env.roll_horizon, env.T) - t_start)

    def add_block(self, res, t_start: int, wholesale=None) -> bool:
        """Record one cleared window's committed periods.

        Returns False when the block contributed nothing usable (no result, or
        a single-period fallback that bypassed the bidding mechanism).
        """
        if res is None:
            return False
        if res.get("fell_back"):
            self.fell_backs += 1
            return False

        n = self.committed_periods(t_start)
        for nm, day_sched in self.sched.items():
            ws = res["schedules"].get(nm)
            if ws is None:
                continue
            for key in day_sched:
                day_sched[key][t_start:t_start + n] = ws[key][:n]
        if self.lmp is None:
            self.lmp = np.zeros((self.T, res["lmp"].shape[1]))
        self.lmp[t_start:t_start + n, :] = res["lmp"][:n, :]
        if wholesale is not None:
            self.wholesale[t_start:t_start + n] = wholesale[:n]
        self.periods_captured += n
        self.blocks_captured += 1
        return True

    def add_declared(self, current_actions: dict, t_start: int) -> None:
        """Record the actions actually declared over the committed periods.

        The slots come from whatever the caller declared rather than from a
        fixed pair of names, so a record of one action space cannot be silently
        reread as another.
        """
        n = self.committed_periods(t_start)
        for nm, act in (current_actions or {}).items():
            slots = self.declared.setdefault(
                nm, {k: np.zeros(self.T) for k in act})
            for k, arr in act.items():
                slots[k][t_start:t_start + n] = \
                    np.asarray(arr)[t_start:t_start + n]

    def finish(self) -> dict:
        """Report the capture plus whether the recorded periods are sound.

        ``complete`` means the whole horizon was captured; ``usable`` means
        every captured block recorded real schedule data, ignoring how much of
        the horizon was covered. A deliberately short episode (fewer blocks
        than a full day) is usable but not complete, whereas a failed or
        fallen-back block is neither.
        """
        return {"sched": self.sched,
                "lmp": self.lmp,
                "wholesale": self.wholesale,
                "declared": self.declared,
                "fell_backs": self.fell_backs,
                "n_blocks_captured": self.blocks_captured,
                "n_periods_captured": self.periods_captured,
                "complete": (self.periods_captured == self.T
                             and self.fell_backs == 0),
                "usable": self.periods_captured == self.blocks_captured
                          * BLOCK_SIZE and self.fell_backs == 0}


class BiddingEnv:
    """Multi-agent bidding environment with continuous actions.

    Supports two modes:
    - Multi-agent (rl_agent_names=None): all agents are RL-controlled.
      Backward compatible with the original MATD3 pipeline.
    - Single-agent (rl_agent_names=["Agent_X"]): only the named agent
      uses RL actions; all other agents stay at their spec's truthful
      declaration.

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
                 market_impact_penalty: float = 0.0,
                 bid_mult_low: float = BID_MULT_LOW,
                 bid_mult_high: float = BID_MULT_HIGH,
                 obs_spec: Optional[ObservationSpec] = None,
                 action_spec: Optional[ActionSpec] = None):
        self.all_agents = agents
        self.config = config
        self.T = 96
        self.stage = stage
        self.wholesale = None
        # Which day of the wholesale library this episode is priced on; set by
        # `set_agents`. None means "whatever the configuration says", which is
        # what every caller that predates day rotation expects.
        self.day_id = None
        self.roll_horizon = roll_horizon  # look-ahead periods per window

        # Per-agent action bounds (configurable per training run)
        self.bid_mult_low = bid_mult_low
        self.bid_mult_high = bid_mult_high

        # Pluggable observation/action layout. The observation defaults to V3,
        # and the action to the storage quote space: the agents a policy acts
        # on are the independent batteries, and a battery declares prices.
        #
        # `bid_mult_low/high` still override the non-storage action's bounds,
        # which is what they always did; they have no meaning for a storage
        # agent, whose prices are absolute.
        self.obs_spec = obs_spec if obs_spec is not None else OBS_V4
        if action_spec is not None:
            self.action_spec = action_spec
        else:
            self.action_spec = ACTION_STORAGE_QUOTE_V2
        self._legacy_action_spec = ActionSpec(
            ACTION_BID_OFFER_V1.name,
            ACTION_BID_OFFER_V1.version,
            ACTION_BID_OFFER_V1.action_names,
            {"bid_mult": (bid_mult_low, bid_mult_high),
             "offer_adder": ACTION_BID_OFFER_V1.bounds["offer_adder"]},
            truthful=dict(ACTION_BID_OFFER_V1.truthful),
        )

        # Differential reward: per-block reward = raw profit - baseline profit,
        # where the baseline is the same window under truthful bidding for all
        # RL agents. Anchors the truthful policy at zero advantage.
        self.use_differential_reward = use_differential_reward
        # Market-impact penalty: per block, subtract
        #   lambda * sum_t ((q_rl + q_base)/2 * (lmp_rl - lmp_base))_t
        # (the agent's price-impact "power" term between its own clear and the
        # truthful baseline clear in the same window). 0 = disabled (current
        # differential-reward behavior). Only meaningful with the differential
        # baseline available (use_differential_reward).
        self.market_impact_penalty = float(market_impact_penalty)

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
        # Last differential-reward baseline clear (truthful bids for the
        # committed block, opponents and SOC held identical). Exposed so
        # per-episode accounting can read the truthful day without paying for
        # a third clearing. None when differential reward is off: there is
        # then no truthful baseline to compare against.
        self._last_base_result = None
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
        return bounds_tensor(self.action_spec)

    def action_bounds_for(self, agent) -> torch.Tensor:
        """The action box of one agent, from that agent's own spec.

        The environment can hold a mixed population, and a prosumer's
        multiplier pair is not the battery's price pair, so the box a policy is
        built in has to follow the agent rather than the environment.
        """
        return bounds_tensor(self.spec_for(agent))

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

    # ------------------------------------------------------------------
    # Default actions for non-RL agents
    # ------------------------------------------------------------------

    def _default_actions(self) -> dict:
        """Every agent at its truthful declaration, in its own action space.

        This is the reference every comparison is measured from, so it is
        taken from the spec rather than written out here: a baseline policy
        that declares something other than the agent's own reservation is not
        a baseline. Storage agents get their two prices; load and prosumer
        agents get the multiplier pair their declared prices live in.
        """
        return {
            a.name: {name: np.full(self.T, self.spec_for(a).truthful[name],
                                   dtype=float)
                     for name in self.spec_for(a).action_names}
            for a in self.all_agents
        }

    def spec_for(self, agent) -> ActionSpec:
        """Which action space this agent's quotes live in.

        Delegates to `rl_spec.action_spec_for` with this environment's two
        configured specs. The rule -- chosen by participant type, so a prosumer
        that happens to carry a battery keeps the declared-price ontology its
        load and generation give it, and only a battery alone declares
        reservation prices -- lives there, so it cannot drift between the
        environment, the trainers and the clearing.
        """
        return action_spec_for(agent, storage_spec=self.action_spec,
                               other_spec=self._legacy_action_spec)

    # ------------------------------------------------------------------
    # Scenario switching
    # ------------------------------------------------------------------

    def set_agents(self, agents: List[Agent],
                   wholesale: Optional[np.ndarray] = None,
                   day_id: Optional[int] = None):
        """Replace the agent population for scenario switching.

        Rebuilds per-RL-agent forecasters and resets SOC/history state.
        The RL agent list is re-filtered from the new population using
        the names originally passed to __init__.

        Call this between episodes when cycling scenarios.

        `day_id` is which day of the wholesale library this episode is priced
        on. It is carried as environment *metadata*: it selects the price, and
        it is logged and fingerprint-checked, but it never enters an
        observation -- the layout of OBS_V4 is the same on every day, so a
        policy cannot learn to recognise a day rather than to bid. Both
        `reset` and `step` pass it down, because each regenerates the price and
        a window priced off the wrong day is silent. None leaves whatever the
        configuration says, which is the behaviour every caller had before
        rotation existed.
        """
        self.day_id = day_id
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
                              avg_lmp_hist: float,
                              system_load_re_ratio: float = 1.0,
                              avg_other_bid: float = 1.0,
                              bid_std: float = 0.0) -> Dict[str, float]:
        """Compute all named observation features for one agent.

        Returns the superset of features known to the environment, keyed by
        feature name; `_get_agent_obs` selects the subset the active
        ObservationSpec declares. Load/RE are the block means (the action
        commits a whole block), each divided by its own day peak (load peak
        for load, RE peak for RE), so they stay in [0, 1] without saturating.

        **Every price feature is a function of the past alone.** The load and RE
        peaks come from the agent's own *forecasts*, which a day-ahead
        participant holds; the LMP history is the sequence already cleared; and
        the reference price is the frozen constant `PRICE_REF_CNY_MWH`. No
        realized price from a period this decision has not reached enters any
        feature, which is what V4 exists to guarantee and what
        `test_the_observation_ignores_prices_after_the_decision` pins.
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

        # Last observed LMP, or the frozen reference if none exists yet. The
        # fallback is a constant rather than the day's mean: at block 0 the
        # day's mean is the future, and using it seeded `last_lmp` -- and
        # through it `ema_dev` -- with information from later periods.
        if hist_lmp is not None and len(hist_lmp) > 0:
            last_lmp = float(hist_lmp[-1])
        else:
            last_lmp = PRICE_REF_CNY_MWH

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

        # Absolute price levels as O(1) ratios of a FROZEN constant; the
        # negative tail (excess-RE prices) is preserved by a lower clip. The
        # denominator used to be the day's realized mean, which put that mean
        # into `last_lmp_norm` and `pred_lmp_norm` as well as into
        # `avg_lmp_hist_norm` -- three more features than the one that looked
        # like the culprit.
        price_ref = PRICE_REF_CNY_MWH
        last_lmp_norm = float(np.clip(last_lmp / price_ref, -1.0, 3.0))
        avg_lmp_hist_norm = float(np.clip(avg_lmp_hist / price_ref, 0.0, 3.0))
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
            "avg_lmp_hist_norm": avg_lmp_hist_norm,
            "slr_norm": slr_norm,
            "avg_other_bid": avg_other_bid,
            "bid_std": bid_std,
            "ema_dev": ema_dev,
            "price_trend": trend,
            "pred_lmp_norm": pred_lmp_norm,
        }

    def _get_agent_obs(self, agent: Agent, block_idx: int,
                       hist_lmp: np.ndarray | None,
                       avg_lmp_hist: float,
                       system_load_re_ratio: float = 1.0,
                       avg_other_bid: float = 1.0,
                       bid_std: float = 0.0) -> np.ndarray:
        """Build the observation vector for one agent from the active spec.

        Order matters: the per-agent features must be the first entries so
        MATD3's centralized critic can slice them out when building the global
        state.
        """
        feats = self._compute_obs_features(
            agent, block_idx, hist_lmp, avg_lmp_hist, system_load_re_ratio,
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
        """Summary of what the other participants declared, last block.

        Each agent contributes the first component of its own action space,
        scaled into its own bounds so the two ontologies are comparable: for a
        storage unit that is its charge price over the charge-price range, and
        for a load or prosumer it is the multiplier over the multiplier range.
        Comparing raw values across the two would be meaningless, and comparing
        only storage units would hide most of the market.

        Iterates over self.all_agents (not just rl_agents) so the RL agent
        observes every other participant, including those on defaults.

        The observation features these feed are still named `avg_other_bid`
        and `bid_std`; the names are legacy from when every action was a
        multiplier, and the rename belongs with the observation revision.
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
            if nm not in src or not src[nm]:
                continue
            spec = self.spec_for(a)
            name = spec.action_names[0]
            lo, hi = spec.bounds[name]
            value = float(np.asarray(src[nm][name]).flat[0])
            scaled = (value - lo) / (hi - lo) if hi > lo else 0.0
            all_bids.append(scaled)
            if nm != agent_name:
                other_bids.append(scaled)
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
                self.T, agents=self.all_agents, config=self.config,
                day_id=self.day_id)
        else:
            self.wholesale = wholesale

        # Per-block wholesale actually fed to clear_market; set on each step.
        self._last_wholesale = None
        # Clears from the previous episode must not leak into this one.
        self._last_result = None
        self._last_base_result = None
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
        # No LMP has been observed yet, so the causal average is the frozen
        # reference. It is deliberately NOT `mean(self.wholesale)`: at the first
        # decision that is the whole day's realized price, which is not knowable
        # then and made four features non-causal under V3.
        avg_lmp_hist = PRICE_REF_CNY_MWH
        slr = self._compute_system_load_re_ratio(0)
        for a in self.rl_agents:
            avg_oth, bid_std = self._compute_opponent_features(a.name)
            obs[a.name] = self._get_agent_obs(
                a, 0, None, avg_lmp_hist, slr, avg_oth, bid_std)
        return obs

    def _agent_block_profit(self, schedule: dict, lmp_node: np.ndarray,
                            agent: Agent, n_periods: int) -> float:
        """Settled profit (CNY) for one agent over its committed periods.

        This is the environment's view of what the agent earned — the quantity
        the differential reward is built from — and it delegates to the
        participant's own payoff model so the training signal and the reported
        benefit are the same accounting.
        """
        return participant_payoff.participant_payoff(
            schedule, lmp_node, agent, self.config, n_periods).total

    def step(self, actions: Dict[str, np.ndarray]) \
            -> Tuple[Dict[str, np.ndarray],
                     Dict[str, float], bool, dict]:
        """Execute one decision block with rolling-window OPF.

        Parameters
        ----------
        actions : dict
            {agent_name: np.array of the agent's spec dim, in
             `spec.action_names` order}
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

        # Update current_actions with this block's RL decisions. The action
        # vector is positional and its names come from the agent's own spec, so
        # nothing here assumes the storage ontology is the only one.
        for a in self.rl_agents:
            nm = a.name
            if nm not in actions:
                continue
            spec = self.spec_for(a)
            act = np.asarray(actions[nm], dtype=float)
            if act.shape != (spec.dim,):
                raise ValueError(
                    f"{nm} produced an action of shape {act.shape}, but its "
                    f"spec {spec.name!r} declares {spec.dim} components "
                    f"{spec.action_names}")
            if nm not in self.current_actions:
                self.current_actions[nm] = {
                    name: np.full(self.T, spec.truthful[name], dtype=float)
                    for name in spec.action_names}
            for i, name in enumerate(spec.action_names):
                lo, hi = spec.bounds[name]
                self.current_actions[nm][name][t_start:t_end] = \
                    float(np.clip(act[i], lo, hi))

        # Snapshot this block's actions for opponent features in the next block.
        self._prev_block_actions = {
            nm: {name: np.asarray(values).copy()
                 for name, values in act.items()}
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
        #
        # `t_start` is what makes the window the window's own periods. The
        # retired net-load map could leave it out because it derived the price
        # from the agents it was handed, and a window's agents carry the
        # window's own net load -- so asking for `window_T` periods landed on
        # the right ones by construction. An external market does not read the
        # agents, so without an offset every block would be priced off the
        # first `window_T` periods of the day and block 20 would be charged
        # 05:00's prices.
        wholesale = day_ahead_price_china(
            window_T, agents=window_agents, config=window_config,
            t_start=t_start, day_id=self.day_id)
        self._last_wholesale = wholesale

        try:
            result = clear_market(window_agents, window_T, self.stage,
                                  window_actions, window_config,
                                  wholesale=wholesale, horizon_type="window")
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
        # Baseline = this same window under truthful bidding for every RL
        # agent's committed block, with opponents kept at their RL actions and
        # identical prev_soc. The truthful point comes from each agent's spec,
        # since it is a different pair of numbers per action space. It must NOT
        # advance prev_soc or the forecasters; it only supplies the profit to
        # subtract.
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
                if nm not in base_actions:
                    continue
                # The baseline arm is the same window with every RL agent at
                # its truthful declaration and every other agent unchanged, so
                # the difference between the two clears is the policy's doing.
                for name, value in self.spec_for(a).truthful.items():
                    base_actions[nm][name][:n_commit] = value
            try:
                base_result = clear_market(
                    base_window_agents, window_T, self.stage,
                    base_actions, window_config, wholesale=wholesale,
                    horizon_type="window")
            except Exception:
                base_result = None
            self._last_base_result = base_result
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
        # Two quantities with different roles, kept apart on purpose:
        #   raw_reward     the agent's settled profit this block (CNY)
        #   local_baseline what it would have settled under truthful bidding in
        #                  this same window, holding the other agents' bids at
        #                  their RL values
        #   reward         raw minus local baseline: the learning signal
        # The signal is a *local* one, its counterfactual is inside the training
        # window and depends on the other agents' concurrent actions. The
        # benefit the paper reports is a different quantity on a different
        # counterfactual: every agent truthful, over the whole day, both arms
        # facing the same day. A reward is not a benefit and must never be
        # logged or read as one.
        rewards = {}
        raw_profits = {}
        local_baselines = {}
        if result is not None:
            for a in self.rl_agents:
                nm = a.name
                ws = result["schedules"].get(nm)
                if ws is None:
                    rewards[nm] = 0.0
                    raw_profits[nm] = 0.0
                    local_baselines[nm] = 0.0
                    continue
                lmp_node = result["lmp"][:, a.bus]

                raw_reward = self._agent_block_profit(ws, lmp_node, a, n_commit)
                raw_profits[nm] = raw_reward
                local_baselines[nm] = base_rewards.get(nm, 0.0)

                # Differential reward: profit above the truthful baseline
                rewards[nm] = raw_reward - base_rewards.get(nm, 0.0)

                # Market-impact penalty on the same differential basis.
                if self.market_impact_penalty > 0.0 \
                        and base_result is not None \
                        and nm in base_result.get("schedules", {}):
                    bws = base_result["schedules"][nm]
                    q_rl = (ws["p_sell"][:n_commit] - ws["p_buy"][:n_commit])
                    q_bs = (bws["p_sell"][:n_commit] - bws["p_buy"][:n_commit])
                    lmp_bs = base_result["lmp"][:n_commit, a.bus]
                    power_i = float(np.sum(
                        (q_rl + q_bs) / 2.0 * (lmp_node[:n_commit] - lmp_bs)))
                    rewards[nm] -= self.market_impact_penalty * power_i

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

        # Every RL agent gets an entry in all three, so a training log can
        # report profit and reward side by side rather than inferring one from
        # the other.
        for a in self.rl_agents:
            rewards.setdefault(a.name, 0.0)
            raw_profits.setdefault(a.name, 0.0)
            local_baselines.setdefault(a.name, 0.0)

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
        hist_arr = np.array(list(self.hist_lmp)) \
            if self.hist_lmp else None
        # The mean of the LMPs observed SO FAR. Not the day's realized mean:
        # that is not knowable at this decision, and using it made
        # `avg_lmp_hist_norm`, `last_lmp_norm`, `pred_lmp_norm` and `ema_dev`
        # all depend on prices from periods the agent had not reached.
        avg_lmp_hist = (float(np.mean(hist_arr))
                        if hist_arr is not None and hist_arr.size
                        else PRICE_REF_CNY_MWH)
        slr = self._compute_system_load_re_ratio(t_start)
        if not done:
            for a in self.rl_agents:
                avg_oth, bid_std = self._compute_opponent_features(a.name)
                obs[a.name] = self._get_agent_obs(
                    a, self.block_idx, hist_arr, avg_lmp_hist, slr,
                    avg_oth, bid_std)
        else:
            for a in self.rl_agents:
                obs[a.name] = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"lmp": result["lmp"] if result else None,
                "clearing_objective": result["clearing_objective"] if result else 0.0,
                # Per-RL-agent block accounting, kept separate: `reward` is the
                # learning signal, `raw_profit` the settled money it is measured
                # against, `local_baseline_profit` the in-window truthful
                # counterfactual it subtracts. None of these is the paper's
                # benefit; see participant_payoff and the counterfactual
                # evaluator for that.
                "reward": dict(rewards),
                "raw_profit": dict(raw_profits),
                "local_baseline_profit": dict(local_baselines),
                "re_rate": (self._re_used / self._re_avail * 100.0)
                           if self._re_avail > 0 else 100.0}

        return obs, rewards, done, info
