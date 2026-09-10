"""Unit tests for DayCapture, the stitched full-day record of rolling clears.

Training (per episode) and evaluation (per checkpoint) both build their
consumer/mechanism metrics from a DayCapture, so the two paths cannot drift
into computing different quantities under the same column names.
"""

import numpy as np
import pytest

from rl_env import DayCapture, BLOCK_SIZE, N_BLOCKS


T = 8            # two blocks of four, short enough to hand-check
ROLL = 16        # look-ahead longer than the day, as in a real short window


class _Agent:
    def __init__(self, name):
        self.name = name


def _res(n_periods, n_buses=2, fill=1.0):
    """A cleared-window result shaped like clear_market's return."""
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    sched = {}
    return {
        "schedules": sched,
        "lmp": np.full((n_periods, n_buses), fill),
        "fell_back": False,
    }, sched, keys


def _capture(T_total=T, roll=ROLL, names=("A",)):
    agents = [_Agent(n) for n in names]
    return DayCapture(agents, T_total, roll), agents


def _add(cap, sched, keys, res, t_start, n_commit, fill_by_agent=None):
    """Populate a window result's schedules then feed one committed block."""
    for nm in sched:
        for k in keys:
            sched[nm][k] = np.full(res["lmp"].shape[0], fill_by_agent[nm]
                                   if fill_by_agent else 1.0)
    return cap.add_block(res, t_start, wholesale=np.full(res["lmp"].shape[0],
                                                         fill_by_agent["_w"]
                                                         if fill_by_agent
                                                         else 100.0))


def test_stitches_committed_periods_into_a_full_day():
    cap, agents = _capture()
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    sched0 = {"A": {k: np.full(T, 5.0) for k in keys}}
    res0 = {"schedules": sched0, "lmp": np.full((ROLL, 2), 10.0),
            "fell_back": False}
    sched1 = {"A": {k: np.full(T, 7.0) for k in keys}}
    res1 = {"schedules": sched1, "lmp": np.full((ROLL, 2), 20.0),
            "fell_back": False}

    assert cap.add_block(res0, 0, wholesale=np.full(ROLL, 100.0)) is True
    assert cap.add_block(res1, BLOCK_SIZE, wholesale=np.full(ROLL, 200.0)) is True
    day = cap.finish()

    # Only the committed BLOCK_SIZE periods of each window are recorded; the
    # look-ahead periods are re-planned by the next window and must not be
    # double-counted.
    assert day["complete"] is True
    assert day["fell_backs"] == 0
    assert day["n_periods_captured"] == T
    assert day["sched"]["A"]["p_buy"].tolist() == [5.0] * 4 + [7.0] * 4
    assert day["lmp"][:, 0].tolist() == [10.0] * 4 + [20.0] * 4
    assert day["wholesale"].tolist() == [100.0] * 4 + [200.0] * 4


def test_partial_day_capture_is_usable_but_not_the_whole_day():
    # A deliberately short episode (n_blocks < N_BLOCKS) captures fewer
    # periods than the day without anything having failed. That is a valid
    # partial window and must be distinguishable from a degraded capture:
    # "no failures" and "covers the whole day" are different questions.
    cap, _ = _capture()
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    sched = {"A": {k: np.full(T, 5.0) for k in keys}}
    res = {"schedules": sched, "lmp": np.full((ROLL, 2), 10.0),
           "fell_back": False}
    assert cap.add_block(res, 0, wholesale=np.full(ROLL, 100.0)) is True
    day = cap.finish()

    assert day["complete"] is False          # not the whole day
    assert day["usable"] is True             # but nothing degraded
    assert day["fell_backs"] == 0
    assert day["n_periods_captured"] == BLOCK_SIZE


def test_missing_result_leaves_capture_incomplete():
    # A block whose solve returned None holds zeros for its periods; the
    # capture must report itself incomplete rather than presenting the zeros
    # as a cleared day.
    cap, _ = _capture()
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    sched = {"A": {k: np.full(T, 5.0) for k in keys}}
    res = {"schedules": sched, "lmp": np.full((ROLL, 2), 10.0),
           "fell_back": False}

    assert cap.add_block(res, 0, wholesale=np.full(ROLL, 100.0)) is True
    assert cap.add_block(None, BLOCK_SIZE) is False
    day = cap.finish()

    assert day["complete"] is False
    assert day["n_blocks_captured"] == 1
    assert day["n_periods_captured"] == BLOCK_SIZE


def test_fallback_clear_counts_as_a_fellback():
    # clear_market degrades to the single-period solver when the batch solve
    # fails; that path bypasses the RL bidding mechanism, so the block is
    # counted and the capture is marked incomplete.
    cap, _ = _capture()
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    sched = {"A": {k: np.full(T, 5.0) for k in keys}}
    res = {"schedules": sched, "lmp": np.full((ROLL, 2), 10.0),
           "fell_back": True}

    assert cap.add_block(res, 0, wholesale=np.full(ROLL, 100.0)) is False
    day = cap.finish()
    assert day["fell_backs"] == 1
    assert day["complete"] is False
    assert day["n_periods_captured"] == 0


def test_declared_actions_are_recorded_per_committed_period():
    cap, _ = _capture(names=("A",))
    T_day = T
    current = {"A": {"bid_mult": np.arange(T_day, dtype=float),
                     "offer_adder": np.arange(T_day, dtype=float) * 2.0}}
    cap.add_declared(current, 0)
    day = cap.finish()

    assert day["declared"]["A"]["bid_mult"].tolist() == [0.0, 1.0, 2.0, 3.0,
                                                        0.0, 0.0, 0.0, 0.0]
    assert day["declared"]["A"]["offer_adder"][:4].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_committed_periods_shrink_at_the_final_window():
    # The last window is shorter than the day, so fewer periods are committed.
    cap, _ = _capture(T_total=T, roll=ROLL)
    assert cap.committed_periods(0) == BLOCK_SIZE
    assert cap.committed_periods(T - BLOCK_SIZE) == BLOCK_SIZE
    short = DayCapture([_Agent("A")], 6, 16)
    assert short.committed_periods(4) == 2


def test_from_env_adopts_the_envs_geometry_and_agents():
    from types import SimpleNamespace
    env = SimpleNamespace(all_agents=[_Agent("A"), _Agent("B")], T=96,
                          roll_horizon=16)
    cap = DayCapture.from_env(env)
    day = cap.finish()
    assert set(day["sched"]) == {"A", "B"}
    assert day["complete"] is False       # nothing captured yet
    assert cap.committed_periods(0) == BLOCK_SIZE
    assert cap.committed_periods(92) == N_BLOCKS * BLOCK_SIZE - 92
