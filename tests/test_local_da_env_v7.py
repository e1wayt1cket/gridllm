import numpy as np
import pytest

import rl_da_env
import rl_spec_v7
from local_da_contracts import validate_day
from local_da_days import load_pair, window_day
from rl_da_env import (E_MAX_MWH, EpisodeClearingError, LocalDABiddingEnv,
                       carried_soc, directions_consistent)

PAIR = "2021-06-04"
TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def truthful(env):
    return {nm: TRUTHFUL.copy() for nm in env.agent_names}


class DriftingClearingModel:
    """Fixed results whose conditional objective does not reproduce the exact one."""

    def __init__(self, day, *, directions=None, exact=True, seconds=120.0):
        self.day = day
        self.exact = exact

    def solve(self, prices=False):
        names = [unit.name for unit in self.day.storage]
        periods = self.day.periods
        result = {"optimal": True,
                  "clearing_objective": 0.0 if self.exact else 1.0,
                  "charge": {nm: np.zeros(periods) for nm in names},
                  "discharge": {nm: np.zeros(periods) for nm in names},
                  "energy": {nm: np.full(periods + 1, 0.8) for nm in names}}
        if prices:
            result["lmp_cny_mwh"] = np.zeros(
                (periods, self.day.network.buses))
        return result

    def close(self):
        pass


def test_a_truthful_block_returns_finite_rewards_and_committed_lmps():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    obs, rewards, done, info = env.step(truthful(env))
    assert set(rewards) == set(env.agent_names)
    assert all(np.isfinite(v) for v in rewards.values())
    assert info["lmp"].shape == (4, 33)
    assert all(0.1 <= s <= 0.9 for s in env.prev_soc.values())
    assert obs["ESS9"].shape == (12,)
    assert done is False


def test_the_observation_reads_no_price_after_the_decision():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    env.step(truthful(env))
    agent = env.all_agents[0]
    hist = np.array(list(env.hist_lmp))
    before = env._compute_obs_features(agent, 1, hist, float(hist.mean()))
    saved = env.pair["external_price"].copy()
    env.pair["external_price"][8:] = 9999.
    after = env._compute_obs_features(agent, 1, hist, float(hist.mean()))
    env.pair["external_price"] = saved
    assert set(before) == set(after)
    assert all(np.allclose(before[k], after[k]) for k in before)
    assert before["lmp_bus_norm"] == pytest.approx(
        env.bus_lmp_history["ESS9"][-1] / rl_da_env.PRICE_REF_CNY_MWH)


def test_the_bus_history_holds_only_committed_periods():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    env.step(truthful(env))
    assert len(env.bus_lmp_history["ESS9"]) == 4
    assert np.allclose(env.bus_lmp_history["ESS9"],
                       env._last_result["pricing"]["lmp_cny_mwh"][:4, 9])


def test_an_unpriced_block_names_its_failure(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    monkeypatch.setattr(rl_da_env, "clear_window",
                        lambda day, **kw: {"failure": "PRICE_SOLVE_FAILED"})
    with pytest.raises(EpisodeClearingError) as caught:
        env.step(truthful(env))
    assert caught.value.failure_code == "PRICE_SOLVE_FAILED"


def test_a_conditional_objective_that_drifts_names_the_price_failure(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    monkeypatch.setattr(rl_da_env, "ClearingModel", DriftingClearingModel)
    with pytest.raises(EpisodeClearingError) as caught:
        env.step(truthful(env))
    assert caught.value.failure_code == "PRICE_SOLVE_FAILED"


def test_zero_quantity_quotes_either_clear_or_name_the_failure():
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    zero = {nm: np.array([0., 0., 0., 0.]) for nm in env.agent_names}
    try:
        _, _, _, info = env.step(zero)
    except EpisodeClearingError as error:
        assert error.failure_code in ("CLEAR_INFEASIBLE", "BASELINE_INFEASIBLE")
    else:
        assert all(abs(v) < 1e-9 for v in info["payoff"].values())
        assert np.allclose(list(info["charge_mwh"].values()), 0.)


def test_direction_consistency_flags_a_flipped_leg():
    directions = {"ESS9": np.array([1, -1])}
    idle = {"ESS9": np.array([False, False])}
    same = {"charge": {"ESS9": np.array([0.2, 0.0])},
            "discharge": {"ESS9": np.array([0.0, 0.3])}}
    flipped = {"charge": {"ESS9": np.array([0.0, 0.3])},
               "discharge": {"ESS9": np.array([0.0, 0.0])}}
    assert directions_consistent(directions, idle, same) is True
    assert directions_consistent(directions, idle, flipped) is False


def test_the_carried_soc_is_held_on_the_band_edge():
    # A solve can leave a unit a fraction outside its declared band. The next
    # window must inherit the edge, because validate_day refuses anything else.
    env = LocalDABiddingEnv()
    env.reset(load_pair(PAIR))
    unit = env.rl_agents[0]
    below = carried_soc(0.1 * E_MAX_MWH - 1e-9, unit)
    above = carried_soc(0.9 * E_MAX_MWH + 1e-9, unit)
    for value in (below, above):
        env.prev_soc = {agent.name: value for agent in env.rl_agents}
        validate_day(window_day(env.pair, 0, 20, soc=env.prev_soc, quotes={}))
    assert below == unit.storage.soc_min
    assert above == unit.storage.soc_max


def test_the_quote_vector_reaches_the_window_in_leg_order():
    quote = rl_spec_v7.ACTION_STORAGE_QUOTE_V5.truthful_array()
    day = window_day(load_pair(PAIR), 44, 64,
                     soc={f"ESS{b}": 0.5 for b in (9, 14, 17, 27)},
                     quotes={"ESS9": quote})
    bid = day.bids["ESS9"]
    assert np.allclose(bid.charge_price_cny_mwh, 0.0)
    assert np.allclose(bid.charge_quantity_mw, 0.4)
    assert np.allclose(bid.discharge_price_cny_mwh, 0.0)
    assert np.allclose(bid.discharge_quantity_mw, 0.4)
