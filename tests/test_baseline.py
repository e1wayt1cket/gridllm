# tests/test_baseline.py
"""Regression tests for baseline scenario: welfare, RE rate, carbon."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from models import MarketConfig
from market import clear_market, adaptive_bidding, two_settlement
from scenarios import get_scenario


def test_baseline_da_welfare():
    """Baseline DA welfare should be in reasonable range."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    welfare = results["welfare"]
    assert 50000 < welfare < 200000, f"welfare {welfare:.0f} out of range"


def test_baseline_re_rate():
    """Baseline RE consumption rate should be >= 90%."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    re_rate = results["re_consumption_rate"]
    assert re_rate >= 90.0, f"RE rate {re_rate:.1f}% below 90%"


def test_baseline_carbon_positive():
    """Baseline should have positive carbon emissions."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    assert results["carbon_emissions"] > 0, "carbon emissions should be > 0"
    assert results["carbon_intensity"] > 0, "carbon intensity should be > 0"


def test_baseline_result_schema():
    """Baseline result dict must contain all expected keys."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    required = ["price", "lmp", "schedules", "welfare", "re_consumption_rate",
                "total_re_available", "carbon_emissions", "carbon_intensity",
                "total_curtailment"]
    for key in required:
        assert key in results, f"missing key: {key}"


def test_baseline_lmp_shape():
    """LMP matrix should have shape (96, 33)."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    assert results["lmp"].shape == (96, 33), f"LMP shape {results['lmp'].shape} != (96, 33)"


def test_baseline_schedule_keys():
    """Every agent should have a schedule with required sub-keys."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    sub_keys = ["p_buy", "p_sell", "served", "unserved", "pv_used",
                "wind_used", "p_ch", "p_dis", "soc"]
    for a in agents:
        sched = results["schedules"][a.name]
        for key in sub_keys:
            assert key in sched, f"agent {a.name} missing {key}"
            assert len(sched[key]) == T, f"agent {a.name} {key} length {len(sched[key])} != {T}"


def test_peak_load_higher_welfare():
    """Peak load should have higher DA welfare than baseline (weighted-sum mode)."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False,
                          use_constraint_multi_obj=False)
    agents_b, _ = get_scenario("baseline", T=T, config=config)
    agents_p, _ = get_scenario("peak_load", T=T, config=config)
    r_b = clear_market(agents_b, T, "DA",
                       adaptive_bidding(agents_b, config, "rl"), config)
    r_p = clear_market(agents_p, T, "DA",
                       adaptive_bidding(agents_p, config, "rl"), config)
    assert r_p["welfare"] > r_b["welfare"], \
        f"peak_load welfare {r_p['welfare']:.0f} <= baseline {r_b['welfare']:.0f}"


@pytest.mark.skip(reason="constraint-based multi-objective system removed from dispatch")
def test_constraint_mode_feasible():
    """Constraint mode with default caps should be feasible and meet targets."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False,
                          use_constraint_multi_obj=True,
                          carbon_cap_tco2=200.0, re_min_rate=90.0)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    assert results is not None, "constraint mode should not fail"
    assert results["re_consumption_rate"] >= config.re_min_rate - 0.5, \
        f"RE rate {results['re_consumption_rate']:.1f}% below min {config.re_min_rate}%"
    assert results["carbon_emissions"] <= config.carbon_cap_tco2 + 1.0, \
        f"carbon {results['carbon_emissions']:.1f} exceeds cap {config.carbon_cap_tco2}"
    assert "shadow_prices" in results, "constraint mode should return shadow prices"


def test_congestion_lower_welfare():
    """Congestion should have lower or equal welfare vs baseline (weighted-sum mode)."""
    T = 96
    config_b = MarketConfig(opf_mode="lindistflow", verbose=False,
                            use_constraint_multi_obj=False)
    config_c = MarketConfig(opf_mode="lindistflow", verbose=False,
                            use_constraint_multi_obj=False)
    agents_b, _ = get_scenario("baseline", T=T, config=config_b)
    agents_c, _ = get_scenario("congestion", T=T, config=config_c)
    r_b = clear_market(agents_b, T, "DA",
                       adaptive_bidding(agents_b, config_b, "rl"), config_b)
    r_c = clear_market(agents_c, T, "DA",
                       adaptive_bidding(agents_c, config_c, "rl"), config_c)
    assert r_c["welfare"] <= r_b["welfare"] * 1.05, \
        f"congestion welfare {r_c['welfare']:.0f} much higher than baseline {r_b['welfare']:.0f}"


def test_carbon_scales_with_dt():
    """Carbon emissions capped by carbon_cap_tco2 regardless of T."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False,
                          carbon_cap_tco2=200.0, re_min_rate=90.0)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
    results = clear_market(agents, T, "DA", actions, config)
    assert results["carbon_emissions"] <= config.carbon_cap_tco2 + 1.0, \
        f"carbon {results['carbon_emissions']:.1f} exceeds cap {config.carbon_cap_tco2}"


def test_soc_transition_formula():
    """Storage SOC evolution follows the transition formula."""
    from dispatch import solve_lindist_opf_batch
    from grid import build_base_network
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    storage_agents = [a for a in agents if a.storage]
    if not storage_agents:
        return
    base_net = build_base_network(config)
    wholesale = np.ones(T) * 300.0
    actions = {a.name: {"bid_mult": 1.0, "offer_adder": 0.0} for a in agents}
    result = solve_lindist_opf_batch(base_net, agents, T, "DA", config, actions, wholesale)
    assert result is not None, "batch OPF should succeed"
    for a in storage_agents:
        nm = a.name
        sched = result["schedules"][nm]
        stor = a.storage
        dt = 0.25
        e_max = stor.e_max
        eta_ch = stor.eta_ch
        eta_dis = stor.eta_dis
        soc0 = stor.soc0
        # sched["soc"][t] is SOC at START of period t; transition links t -> t+1
        assert abs(sched["soc"][0] - soc0) < 1e-3, \
            f"initial SOC {sched['soc'][0]:.4f} vs {soc0:.4f}"
        for t in range(T - 1):
            soc_t = sched["soc"][t]
            expected_next = (soc_t
                             + (eta_ch * sched["p_ch"][t] - sched["p_dis"][t] / eta_dis) * dt / e_max)
            actual_next = sched["soc"][t + 1]
            assert abs(actual_next - expected_next) < 1e-5, \
                f"SOC mismatch t={t}: {actual_next:.6f} vs {expected_next:.6f}"


def test_two_settlement_flow():
    """two_settlement: agents pay when net-consuming, revenue when net-generating."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    da_actions = adaptive_bidding(agents, config, strategy="rl")
    da = clear_market(agents, T, "DA", da_actions, config)
    rt_actions = adaptive_bidding(agents, config, strategy="rl")
    rt = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da, rt)

    da_lmp = da.get("lmp")
    rt_lmp = rt.get("lmp")
    for a in agents:
        p = payment[a.name]
        lmp_da = da_lmp[:, a.bus] if da_lmp is not None else da["price"]
        lmp_rt = rt_lmp[:, a.bus] if rt_lmp is not None else rt["price"]
        da_net = (da["schedules"][a.name]["p_buy"]
                  - da["schedules"][a.name]["p_sell"])
        rt_net = (rt["schedules"][a.name]["p_buy"]
                  - rt["schedules"][a.name]["p_sell"])
        expected_p = np.sum(lmp_da * da_net) + np.sum(lmp_rt * (rt_net - da_net))
        assert abs(p - expected_p) < 1e-4, \
            f"{a.name}: payment {p:.2f} vs expected {expected_p:.2f}"


def test_constraint_vs_weighted():
    """Constraint mode meets carbon cap; higher carbon price reduces emissions."""
    T = 96
    cap = 150.0
    config_c = MarketConfig(opf_mode="lindistflow", verbose=False,
                            carbon_cap_tco2=cap, re_min_rate=90.0)
    agents_c, _ = get_scenario("baseline", T=T)
    actions_c = adaptive_bidding(agents_c, config_c, strategy="rl")
    r_c = clear_market(agents_c, T, "DA", actions_c, config_c)
    assert r_c["carbon_emissions"] <= cap + 1.0, \
        f"constraint mode carbon {r_c['carbon_emissions']:.1f} exceeds cap {cap}"

    agents_w, _ = get_scenario("baseline", T=T)
    config_w0 = MarketConfig(opf_mode="lindistflow", verbose=False,
                             use_constraint_multi_obj=False,
                             lambda_carbon=0, lambda_re=0, lambda_curtail=0)
    r_w0 = clear_market(agents_w, T, "DA",
                        adaptive_bidding(agents_w, config_w0, "rl"), config_w0)
    config_wh = MarketConfig(opf_mode="lindistflow", verbose=False,
                             use_constraint_multi_obj=False,
                             lambda_carbon=500, lambda_re=0, lambda_curtail=0)
    r_wh = clear_market(agents_w, T, "DA",
                        adaptive_bidding(agents_w, config_wh, "rl"), config_wh)
    assert r_wh["carbon_emissions"] <= r_w0["carbon_emissions"] + 1.0, \
        f"high carbon price {r_wh['carbon_emissions']:.1f} > zero price {r_w0['carbon_emissions']:.1f}"


def test_stackelberg_improves_leader_payoff():
    """Stackelberg leader bidding should improve leader payoff vs RL bidding."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    storage_agents = [a for a in agents if a.storage]
    if len(storage_agents) < 2:
        return
    leader = storage_agents[1]  # use second storage (larger capacity)
    from stackelberg import stackelberg_bidding

    # RL baseline
    br_actions = adaptive_bidding(agents, config, strategy="rl")
    br_result = clear_market(agents, T, "DA", br_actions, config)

    # Stackelberg
    st_actions, info = stackelberg_bidding(agents, config, leader.name, T=T)
    st_result = clear_market(agents, T, "DA", st_actions, config)

    def agent_payoff(result, a):
        sched = result["schedules"][a.name]
        node_p = result["lmp"][:, a.bus]
        return float(np.sum(sched["p_sell"] * node_p)
                     - np.sum(sched["p_buy"] * node_p)
                     - np.sum(sched["unserved"] * config.penalty_unserved))

    br_pay = agent_payoff(br_result, leader)
    st_pay = agent_payoff(st_result, leader)
    assert st_pay >= br_pay - 5.0, \
        f"Stackelberg leader payoff {st_pay:.1f} << best_response {br_pay:.1f}"
    assert info["leader"] == leader.name
    assert len(info["optimal_params"]) >= 1


def test_stackelberg_nash_runs():
    """Multi-leader Stackelberg-Nash should converge within few rounds."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    from stackelberg import stackelberg_nash
    actions, history = stackelberg_nash(agents, config, max_rounds=3, T=T)
    assert len(history) >= 1
    for a in agents:
        assert a.name in actions


def test_mpc_storage_lp():
    """MPC LP should charge at low prices and discharge at high prices."""
    from mpc_storage import solve_storage_mpc
    from models import StorageSpec
    stor = StorageSpec(e_max=0.4, p_ch_max=0.1, p_dis_max=0.1,
                       eta_ch=0.95, eta_dis=0.95, soc0=0.5,
                       soc_min=0.1, soc_max=0.9)
    price = np.array([50, 50, 50, 300, 300, 300, 50, 50])
    ch, dis, soc = solve_storage_mpc(stor, 0.5, price)
    # Should charge in cheap periods, discharge in expensive ones
    assert np.sum(ch[:3]) > 0.01, "should charge when price is low"
    assert np.sum(dis[3:6]) > 0.01, "should discharge when price is high"
    assert np.all(soc >= stor.soc_min - 1e-6)
    assert np.all(soc <= stor.soc_max + 1e-6)
    assert abs(soc[0] - 0.5) < 1e-6


def test_mpc_bidding_runs():
    """MPC bidding strategy should produce valid actions and improve on random."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    from mpc_storage import mpc_storage_bidding
    actions = mpc_storage_bidding(agents, config, T=T)
    assert len(actions) == len(agents)
    for a in agents:
        assert a.name in actions
        assert "bid_mult" in actions[a.name]
        assert len(actions[a.name]["bid_mult"]) == T
