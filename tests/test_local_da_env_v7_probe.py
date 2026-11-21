import numpy as np

import rl_da_env
from local_da_days import load_pair
from rl_da_env import PROBE_BLOCK, LocalDABiddingEnv, abort_count

TRUTHFUL = np.array([0., 0.4, 0., 0.4])


def test_a_day_records_exactly_one_probe_at_block_eleven():
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    for _ in range(24):
        _, _, done, _ = env.step({nm: TRUTHFUL.copy() for nm in env.agent_names})
        if done:
            break
    assert len(env.probe_records) == 1
    assert env.probe_records[0]["block"] == PROBE_BLOCK
    assert env.probe_records[0]["date"] == "2021-06-04"


def test_a_failed_probe_is_recorded_and_does_not_abort(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    monkeypatch.setattr(rl_da_env, "clear_day",
                        lambda day, **kw: {"price_qualification": "FAILED",
                                           "idle_direction_stable": False,
                                           "qualified_for_settlement": False})
    record = env.probe_block()
    assert record["failure"] == "PROBE_FAILED"
    assert env.probe_records == []


def test_a_qualified_probe_reaches_clear_day_and_records_success(monkeypatch):
    env = LocalDABiddingEnv()
    env.reset(load_pair("2021-06-04"))
    calls = []

    def fake(day, **kw):
        calls.append(kw)
        return {"price_qualification": "SMOOTH_PASS",
                "idle_direction_stable": True,
                "qualified_for_settlement": True}

    monkeypatch.setattr(rl_da_env, "clear_day", fake)
    record = env.probe_block()
    assert record["failure"] is None
    assert record["qualified_for_settlement"] is True
    # The record must carry the pipeline's own verdicts, not stand-ins.
    assert record["price_qualification"] == "SMOOTH_PASS"
    assert record["idle_direction_stable"] is True
    assert len(calls) == 1
    assert calls[0]["price_probes"] is True


def test_more_than_five_percent_aborts_degrade_the_round():
    assert abort_count([{"abort": False}] * 19 + [{"abort": True}])["degraded"] is True
    assert abort_count([{"abort": False}] * 20 + [{"abort": True}])["degraded"] is False
