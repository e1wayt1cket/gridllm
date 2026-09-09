"""Unit tests for the three-layer accounting metrics (src/surplus_metrics.py).

Pure-toy tests build schedule dicts / LMP / wholesale arrays directly and
hand-check the formulas; two parity tests pin reuse against
`diagnose_profit.decompose` and `eval_agents.compute_agent_profit`.
The last test is marked slow (real SOCP clears).
"""

import numpy as np
import pytest

from surplus_metrics import (load_agents, consumer_metrics,
                             load_payment_weighted_markup, agent_profit,
                             market_power_split, reconciliation)


def _config():
    from types import SimpleNamespace
    return SimpleNamespace(
        market_design=SimpleNamespace(penalty_unserved=100.0),
        storage=SimpleNamespace(cycle_cost=0.5))


def _agent(name, bus, bid_value, load=None, offer_cost=0.0,
           storage=False):
    from types import SimpleNamespace
    load_arr = None if load is None else np.asarray(load, dtype=float)
    return SimpleNamespace(name=name, bus=bus, bid_value=bid_value,
                           offer_cost=offer_cost,
                           load_forecast=load_arr, load_real=None,
                           storage=object() if storage else None,
                           has_wind=False)


def _sched(nm_vals, T=3):
    """Schedule dict for one agent: arrays from a {key: list} mapping."""
    keys = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
            "pv_used", "wind_used"]
    d = {k: np.zeros(T) for k in keys}
    for k, v in nm_vals.items():
        d[k] = np.asarray(v, dtype=float)
    return d


# ---------------------------------------------------------------------------
def test_load_agents_includes_prosumer_excludes_pure_gen():
    a1 = _agent("C", 0, 50.0, load=[5, 5, 5])
    a2 = _agent("G", 1, 0.0, load=[0, 0, 0], storage=False)
    names = {x.name for x in load_agents([a1, a2])}
    assert names == {"C"}


def test_consumer_payment_surplus_pure_consumer():
    a = _agent("C", 0, 50.0, load=[5, 5, 5])
    sched = {"C": _sched({"served": [5, 5, 5], "p_buy": [5, 5, 5]})}
    lmp = np.full((3, 2), 100.0)
    m = consumer_metrics(sched, lmp, [a])
    assert m["cp"] == pytest.approx(100.0 * 15)      # sum lmp*served
    assert m["cs"] == pytest.approx(50.0 * 15 - 1500.0)


def test_consumer_payment_excludes_storage_charge():
    # Self-generating prosumer; charging extra energy does not inflate CP.
    a = _agent("P", 0, 60.0, load=[10, 10, 10])
    base = _sched({"served": [10, 10, 10], "pv_used": [4, 4, 4],
                   "p_buy": [6, 6, 6]})
    with_charge = _sched({"served": [10, 10, 10], "pv_used": [4, 4, 4],
                          "p_buy": [9, 9, 9], "p_ch": [3, 3, 3]})
    lmp = np.full((3, 2), 100.0)
    m1 = consumer_metrics({"P": base}, lmp, [a])
    m2 = consumer_metrics({"P": with_charge}, lmp, [a])
    assert m1["cp"] == m2["cp"]
    assert m1["cs"] == m2["cs"]
    assert m1["cp"] == pytest.approx(100.0 * 18)     # 3*6 load-import


def test_consumer_payment_ignores_storage_discharge():
    # Storage discharge does NOT offset the consumer import bill: qbuy is
    # served - (pv+wind), ignoring p_dis entirely.
    a = _agent("P", 0, 60.0, load=[10, 10, 10])
    no_dis = _sched({"served": [10, 10, 10], "pv_used": [4, 4, 4],
                     "p_dis": [0, 0, 0], "p_buy": [6, 6, 6]})
    dis = _sched({"served": [10, 10, 10], "pv_used": [4, 4, 4],
                  "p_dis": [6, 6, 6], "p_buy": [2, 2, 2]})
    lmp = np.full((3, 2), 100.0)
    m1 = consumer_metrics({"P": no_dis}, lmp, [a])
    m2 = consumer_metrics({"P": dis}, lmp, [a])
    assert m1["cp"] == m2["cp"] == pytest.approx(100.0 * 18)  # 3*(10-4), p_dis ignored
    assert m1["cs"] == m2["cs"]


def test_consumer_surplus_uses_truthful_bid_value_and_served():
    # Self-gen supplies part of load; CS values the FULL served at bid_value
    # but bills only the imported portion.
    a = _agent("P", 0, 50.0, load=[10, 10])
    sched = {"P": _sched({"served": [10, 10], "pv_used": [4, 4],
                          "p_buy": [6, 6]}, T=2)}
    lmp = np.full((2, 2), 100.0)
    m = consumer_metrics(sched, lmp, [a])
    assert m["cs"] == pytest.approx(50.0 * 20 - 100.0 * 12)


def test_markup_guards_low_wholesale():
    a = _agent("C", 0, 50.0, load=[1, 2, 3])
    sched = {"C": _sched({"served": [1, 2, 3], "p_buy": [1, 2, 3]})}
    lmp = np.array([[100.0], [140.0], [120.0]])
    w = np.array([-5.0, 100.0, 120.0])
    mk = load_payment_weighted_markup(sched, lmp, w, [a])
    num = (140 - 100) * 2 + (120 - 120) * 3
    den = 100 * 2 + 120 * 3
    assert mk == pytest.approx(num / den)
    w2 = np.array([0.0, 0.0, 0.0])
    assert np.isnan(load_payment_weighted_markup(sched, lmp, w2, [a]))


def test_markup_closed_form_constant_wholesale():
    a = _agent("C", 0, 50.0, load=[1, 2, 3])
    sched = {"C": _sched({"served": [1, 2, 3], "p_buy": [1, 2, 3]})}
    lmp = np.array([[100.0], [140.0], [120.0]])
    w = np.full(3, 120.0)
    mk = load_payment_weighted_markup(sched, lmp, w, [a])
    expected = ((100 - 120) * 1 + (140 - 120) * 2 + 0) / (120.0 * 6)
    assert mk == pytest.approx(expected)


def test_reconciliation_identity_lossless_toy():
    # Two-node lossless day, lmp == wholesale everywhere -> rent == 0 and the
    # consumer bill + residual business cash exactly pay the wholesale import.
    T = 3
    w = np.full(T, 100.0)
    lmp = np.tile(w.reshape(-1, 1), (1, 2))  # both buses at wholesale
    c = _agent("C", 0, 60.0, load=[5, 5, 5])
    s = _agent("S", 1, 0.0, load=[0, 0, 0])  # pure seller, no load
    sched = {
        "C": _sched({"served": [5, 5, 5], "p_buy": [5, 5, 5]}),
        "S": _sched({"p_sell": [2, 2, 2]}),
    }
    r = reconciliation(sched, lmp, w, [c, s])
    assert r["rent_total"] == pytest.approx(0.0, abs=1e-6)
    # consumer bill 1500, seller business cash -600, import 9 units
    assert r["cp_total"] == pytest.approx(1500.0)
    assert r["bc_total"] == pytest.approx(-600.0)
    assert r["bill_total"] == pytest.approx(900.0)
    assert r["max_identity_residual"] < 1e-6


def test_market_power_split_matches_diagnose_profit():
    from diagnose_profit import decompose
    T = 3
    cfg = _config()
    aA = _agent("A", 0, 0.0, load=None, storage=True)
    aB = _agent("B", 1, 0.0, load=None, storage=True)
    agents = [aA, aB]
    s_b = {"A": _sched({"p_sell": [0, 1, 1]}),
           "B": _sched({"p_sell": [0, 0, 0]})}
    s_r = {"A": _sched({"p_sell": [0, 2, 2]}),
           "B": _sched({"p_sell": [0, 0, 0]})}
    lmp_b = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    lmp_r = np.array([[110.0, 100.0], [110.0, 100.0], [110.0, 100.0]])
    split = market_power_split(s_b, s_r, lmp_b, lmp_r, agents, cfg)
    row = next(x for x in split if x["name"] == "A")
    rows = [[row["name"], row["profit_delta"]]]
    rows = decompose(rows, s_b, lmp_b, s_r, lmp_r, agents, cfg)
    arb, power, other, net = rows[0][2], rows[0][3], rows[0][4], rows[0][5]
    assert row["arb"] == pytest.approx(arb)
    assert row["market_power"] == pytest.approx(power)
    assert row["other"] == pytest.approx(other)
    assert row["net"] == pytest.approx(net)
    # Hand-checked A: arb=210, power=30, dp=240, net=4.
    assert row["arb"] == pytest.approx(210.0)
    assert row["market_power"] == pytest.approx(30.0)
    assert row["profit_delta"] == pytest.approx(240.0)
    assert row["net"] == pytest.approx(4.0)


def test_agent_profit_parity_with_eval_agents():
    from eval_agents import compute_agent_profit
    T = 3
    cfg = _config()
    a = _agent("S", 0, 50.0, load=None, offer_cost=20.0, storage=True)
    sa = _sched({"p_sell": [1, 2, 0], "p_buy": [0, 0, 2],
                 "served": [3, 3, 3], "pv_used": [1, 1, 1],
                 "p_ch": [0, 0, 2], "p_dis": [1, 2, 0],
                 "unserved": [0, 0, 0]})
    lmp_node = np.array([100.0, 110.0, 90.0])
    mine = agent_profit(sa, lmp_node, a, cfg)
    theirs = compute_agent_profit(sa, lmp_node, a, cfg, n_periods=T)
    assert mine == pytest.approx(theirs)


@pytest.mark.slow
def test_consumer_metrics_end_to_end_capture(tmp_path):
    """Real SOCP day: capture run_episode/run_combined_episode on an existing
    policy dir, assert new keys present and reconciliation rent bounded."""
    from rl_env import BiddingEnv
    from eval_agents import (run_episode, run_combined_episode,
                             default_eval_config, load_policies_from_dir,
                             _n_commit)
    from scenarios import get_scenario
    from surplus_metrics import consumer_metrics, reconciliation

    import torch
    config = default_eval_config()
    agents, _ = get_scenario("baseline", T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    act_bounds = torch.tensor([[0.3, 0.0], [1.8, 50.0]], dtype=torch.float32)
    policy_dir = "policies/matd3_cc_rot_seed42/best"
    policies = load_policies_from_dir(policy_dir, 12, act_bounds,
                                      obs_spec=None, action_spec=None)
    assert len(policies) == 12, "expected 12 policies from seed42 best"
    env_b = BiddingEnv(agents, config)
    base = run_episode(env_b, capture=True)
    env_c = BiddingEnv(agents, config, rl_agent_names=rl_names)
    comb = run_combined_episode(env_c, policies, capture=True)
    assert base["lmp"].shape == (96, 33)
    assert comb["lmp"].shape == (96, 33)
    m = consumer_metrics(comb["sched"], comb["lmp"], agents)
    assert np.isfinite(m["cs"]) and np.isfinite(m["cp"])
    r = reconciliation(comb["sched"], comb["lmp"], comb["wholesale"], agents)
    assert r["bill_total"] > 0
    assert r["rent_total"] > -1e-3 * r["bill_total"]
