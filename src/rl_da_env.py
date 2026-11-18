"""Clear one training block at a time on the local day-ahead engine.

A block is a window cut from a v3 library pair and cleared by
`local_da_clearing`: an exact complementary dispatch, then a conditional price
solve with the directions frozen. The payoff a unit receives here is the
settlement ledger's -- energy bought and sold at its own bus LMP, less the
declared throughput cost -- not the clearing objective.

The baseline arm re-clears the same window with every unit's quote at its
truthful declaration. It is what the day reward is measured from, so the two
arms must describe identical bids: both quote exact Python floats, never the
float32 values a policy or `truthful_array()` hands over (a float32 0.4 is
greater than the device rating and the market refuses it).

The payoff here is a training signal, never a settlement claim: the physical
audits, AC replay and price probes that `local_da_clearing.clear_day` requires
before settlement are not run per block.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from da_quote_map_v5 import TRUTHFUL
from local_da_clearing import ClearingModel, choose_directions
from local_da_days import STORAGE_BUSES, window_day
from models import Agent, StorageSpec
from rl_env import PRICE_REF_CNY_MWH, BiddingEnv
from rl_spec_v7 import ACTION_STORAGE_QUOTE_V5, OBS_V7

BLOCK_SIZE = 4
N_BLOCKS = 24
ROLL_HORIZON = 20
PROBE_BLOCK = 11

DT_HOURS = 0.25
CYCLE_COST_CNY_MWH = 100.0
E_MAX_MWH = 1.6
P_MAX_MW = 0.4
SOC0 = 0.5

FAILURE_CODES = ("CLEAR_INFEASIBLE", "DIRECTIONS_FAILED",
                 "PRICE_SOLVE_FAILED", "BASELINE_INFEASIBLE")


class EpisodeClearingError(RuntimeError):
    """A block the market could not clear, price, or compare against."""

    def __init__(self, failure_code, date, block):
        super().__init__(f"{failure_code} on {date} block {block}")
        self.failure_code = failure_code
        self.date = date
        self.block = block


def directions_consistent(directions, idle, pricing, *, tolerance=1e-6):
    """Whether a priced dispatch opens exactly the leg the exact one did.

    On an idle period the exact dispatch used neither leg, so the priced one
    must too; on an active period it must open the leg the direction names.

    This is a leg-integrity check on this block's own quotes, not the
    accounting layer's `idle_direction_stable`, which re-solves the idle
    direction and compares the two price vectors.
    """
    for name, direction in directions.items():
        charge = np.asarray(pricing["charge"][name], dtype=float)
        discharge = np.asarray(pricing["discharge"][name], dtype=float)
        idle_mask = np.asarray(idle[name], dtype=bool)
        if (np.any(np.abs(charge[idle_mask]) > tolerance)
                or np.any(np.abs(discharge[idle_mask]) > tolerance)):
            return False
        active = ~idle_mask
        if np.any(np.sign(charge[active] - discharge[active])
                  != np.asarray(direction)[active]):
            return False
    return True


def clear_window(day, *, seconds=120.0):
    """Clear one window exactly, then re-solve it with its directions frozen."""
    output = {"dispatch": None, "pricing": None, "directions": None,
              "idle": None, "failure": None, "direction_unstable": False}
    exact = ClearingModel(day, exact=True, seconds=seconds)
    try:
        dispatch = exact.solve()
    finally:
        exact.close()
    output["dispatch"] = dispatch
    if not dispatch["optimal"]:
        output["failure"] = "CLEAR_INFEASIBLE"
        return output
    try:
        directions, idle = choose_directions(dispatch)
    except ValueError:
        output["failure"] = "DIRECTIONS_FAILED"
        return output
    output["directions"], output["idle"] = directions, idle
    pricing_model = ClearingModel(day, directions=directions, exact=False,
                                  seconds=seconds)
    try:
        pricing = pricing_model.solve(prices=True)
    finally:
        pricing_model.close()
    output["pricing"] = pricing
    if not pricing["optimal"] or "lmp_cny_mwh" not in pricing:
        output["failure"] = "PRICE_SOLVE_FAILED"
        return output
    # The conditional solve must reproduce the exact objective, on the same
    # tolerance `clear_day` uses; otherwise its prices describe another
    # dispatch. Both objectives are already in hand, so this costs no solve.
    tolerance = max(.01, abs(dispatch["clearing_objective"]) * 1e-6)
    if (abs(pricing["clearing_objective"] - dispatch["clearing_objective"])
            > tolerance):
        output["failure"] = "PRICE_SOLVE_FAILED"
        return output
    output["direction_unstable"] = not directions_consistent(
        directions, idle, pricing)
    return output


def storage_agents():
    """The four independent batteries the local market clears, with no load."""
    return [Agent(name=f"ESS{bus}", bus=bus, is_prosumer=False,
                  load_forecast=np.zeros(96), pv_forecast=np.zeros(96),
                  load_real=np.zeros(96), pv_real=np.zeros(96),
                  bid_value=0., offer_cost=0.,
                  storage=StorageSpec(e_max=E_MAX_MWH, p_ch_max=P_MAX_MW,
                                      p_dis_max=P_MAX_MW, eta_ch=.92,
                                      eta_dis=.92, soc0=SOC0, soc_min=.1,
                                      soc_max=.9),
                  participant_type="storage")
            for bus in STORAGE_BUSES]


class LocalDABiddingEnv(BiddingEnv):
    """A day of 24 blocks, each a cleared window of one library pair."""

    def __init__(self, *, seconds=120.0, roll_horizon=ROLL_HORIZON):
        super().__init__(storage_agents(), None, roll_horizon=roll_horizon,
                         obs_spec=OBS_V7, action_spec=ACTION_STORAGE_QUOTE_V5)
        self.seconds = float(seconds)

    @property
    def agent_names(self):
        return [a.name for a in self.rl_agents]

    def reset(self, pair):
        self.pair = pair
        self._last_result = None
        self._last_base_result = None
        self._prev_block_actions = None
        self._prev_block_index = None
        self.block_idx = 0
        self.prev_soc = {nm: SOC0 for nm in self.agent_names}
        self.hist_lmp = deque(maxlen=self.T)
        self.bus_lmp_history = {nm: [] for nm in self.agent_names}
        self.probe_records = []
        for fc in self.forecasters.values():
            fc._ema = None
            fc._history = []
        self.current_actions = self._default_actions()
        slr = self._compute_system_load_re_ratio(0)
        return {a.name: self._get_agent_obs(
                    a, 0, None, PRICE_REF_CNY_MWH, slr,
                    *self._compute_opponent_features(a.name))
                for a in self.rl_agents}

    def _compute_obs_features(self, agent, block_idx, hist_lmp, avg_lmp_hist,
                              system_load_re_ratio=1.0, avg_other_dev=0.0,
                              all_dev_std=0.0):
        """The inherited features plus this unit's own bus-price history."""
        feats = super()._compute_obs_features(
            agent, block_idx, hist_lmp, avg_lmp_hist, system_load_re_ratio,
            avg_other_dev, all_dev_std)
        history = self.bus_lmp_history[agent.name]
        last = history[-1] if history else PRICE_REF_CNY_MWH
        # No committed period yet reads as the frozen reference, the same
        # no-history convention the inherited price features use.
        average = float(np.mean(history)) if history else PRICE_REF_CNY_MWH
        feats["lmp_bus_norm"] = float(
            np.clip(last / PRICE_REF_CNY_MWH, -1., 3.))
        feats["avg_lmp_bus_hist_norm"] = float(
            np.clip(average / PRICE_REF_CNY_MWH, -1., 3.))
        return feats

    def _update_price_history(self, result, n_commit):
        """Append the shared cross-node means, then each unit's own bus prices."""
        if result is None:
            return
        super()._update_price_history({"lmp": result["lmp_cny_mwh"]}, n_commit)
        prices = np.asarray(result["lmp_cny_mwh"], dtype=float)[:n_commit]
        for a in self.rl_agents:
            self.bus_lmp_history[a.name].extend(
                float(p) for p in prices[:, a.bus])

    def step(self, actions):
        block = self.block_idx
        t_start = block * BLOCK_SIZE
        n_commit = min(BLOCK_SIZE, self.T - t_start)
        t_end = min(t_start + self.roll_horizon, self.T)

        quotes = {}
        self._prev_block_actions = {}
        for a in self.rl_agents:
            spec = self.spec_for(a)
            act = np.asarray(actions[a.name], dtype=float)
            if act.shape != (spec.dim,):
                raise ValueError(
                    f"{a.name} produced an action of shape {act.shape}, but "
                    f"its spec {spec.name!r} declares {spec.dim} components "
                    f"{spec.action_names}")
            # Project into the declared box, as the parent does: a policy
            # output is a float32 in the box's units, and 0.4 rounds above the
            # device rating before the projection makes it exact.
            quotes[a.name] = np.array(
                [np.clip(act[i], *spec.bounds[name])
                 for i, name in enumerate(spec.action_names)])
            self._prev_block_actions[a.name] = {
                name: np.full(self.T, quotes[a.name][i])
                for i, name in enumerate(spec.action_names)}
        self._prev_block_index = block

        day = window_day(self.pair, t_start, t_end, soc=self.prev_soc,
                         quotes=quotes)
        result = clear_window(day, seconds=self.seconds)
        self._last_result = result
        if result["failure"] is not None:
            raise EpisodeClearingError(result["failure"],
                                       self.pair["load_date"], block)

        truthful = {nm: np.array(TRUTHFUL, dtype=float)
                    for nm in self.agent_names}
        base_day = window_day(self.pair, t_start, t_end, soc=self.prev_soc,
                              quotes=truthful)
        baseline = clear_window(base_day, seconds=self.seconds)
        self._last_base_result = baseline
        if baseline["failure"] is not None:
            raise EpisodeClearingError("BASELINE_INFEASIBLE",
                                       self.pair["load_date"], block)

        pricing = result["pricing"]
        payoff, baseline_payoff = {}, {}
        charge_mwh, discharge_mwh = {}, {}
        for a in self.rl_agents:
            nm = a.name
            lmp = pricing["lmp_cny_mwh"][:n_commit, a.bus]
            charge = pricing["charge"][nm][:n_commit]
            discharge = pricing["discharge"][nm][:n_commit]
            payoff[nm] = DT_HOURS * float(np.sum(
                lmp * (discharge - charge)
                - CYCLE_COST_CNY_MWH * (charge + discharge)))
            charge_mwh[nm] = DT_HOURS * float(np.sum(charge))
            discharge_mwh[nm] = DT_HOURS * float(np.sum(discharge))
            base_lmp = baseline["pricing"]["lmp_cny_mwh"][:n_commit, a.bus]
            base_charge = baseline["pricing"]["charge"][nm][:n_commit]
            base_discharge = baseline["pricing"]["discharge"][nm][:n_commit]
            baseline_payoff[nm] = DT_HOURS * float(np.sum(
                base_lmp * (base_discharge - base_charge)
                - CYCLE_COST_CNY_MWH * (base_charge + base_discharge)))
            self.prev_soc[nm] = float(
                pricing["energy"][nm][n_commit]) / E_MAX_MWH

        buses = sorted(a.bus for a in self.rl_agents)
        mean_lmp = float(np.mean(pricing["lmp_cny_mwh"][:n_commit, buses]))
        for fc in self.forecasters.values():
            fc.update(mean_lmp)
        self._update_price_history(pricing, n_commit)

        self.block_idx += 1
        done = self.block_idx >= N_BLOCKS
        hist = np.array(list(self.hist_lmp)) if self.hist_lmp else None
        avg_lmp_hist = (float(np.mean(hist))
                        if hist is not None and hist.size else PRICE_REF_CNY_MWH)
        slr = self._compute_system_load_re_ratio(self.block_idx * BLOCK_SIZE)
        obs = {}
        for a in self.rl_agents:
            if done:
                obs[a.name] = np.zeros(self.obs_dim, dtype=np.float32)
                continue
            obs[a.name] = self._get_agent_obs(
                a, self.block_idx, hist, avg_lmp_hist, slr,
                *self._compute_opponent_features(a.name))

        info = {"payoff": payoff,
                "baseline_payoff": baseline_payoff,
                "charge_mwh": charge_mwh,
                "discharge_mwh": discharge_mwh,
                "lmp": pricing["lmp_cny_mwh"][:n_commit],
                "directions": result["directions"],
                "direction_unstable": result["direction_unstable"]}
        return obs, dict(payoff), done, info
