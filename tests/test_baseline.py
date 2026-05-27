# tests/test_baseline.py
"""Regression tests for baseline scenario: welfare, RE rate, carbon."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models import MarketConfig
from market import clear_market, adaptive_bidding, two_settlement
from scenarios import get_scenario


def test_baseline_da_welfare():
    """Baseline DA welfare should be in reasonable range."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
    results = clear_market(agents, T, "DA", actions, config)
    welfare = results["welfare"]
    assert 50000 < welfare < 200000, f"welfare {welfare:.0f} out of range"


def test_baseline_re_rate():
    """Baseline RE consumption rate should be >= 90%."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
    results = clear_market(agents, T, "DA", actions, config)
    re_rate = results["re_consumption_rate"]
    assert re_rate >= 90.0, f"RE rate {re_rate:.1f}% below 90%"


def test_baseline_carbon_positive():
    """Baseline should have positive carbon emissions."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
    results = clear_market(agents, T, "DA", actions, config)
    assert results["carbon_emissions"] > 0, "carbon emissions should be > 0"
    assert results["carbon_intensity"] > 0, "carbon intensity should be > 0"


def test_baseline_result_schema():
    """Baseline result dict must contain all expected keys."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
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
    actions = adaptive_bidding(agents, config, strategy="random")
    results = clear_market(agents, T, "DA", actions, config)
    assert results["lmp"].shape == (96, 33), f"LMP shape {results['lmp'].shape} != (96, 33)"


def test_baseline_schedule_keys():
    """Every agent should have a schedule with required sub-keys."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
    results = clear_market(agents, T, "DA", actions, config)
    sub_keys = ["p_buy", "p_sell", "served", "unserved", "pv_used",
                "wind_used", "p_ch", "p_dis", "soc"]
    for a in agents:
        sched = results["schedules"][a.name]
        for key in sub_keys:
            assert key in sched, f"agent {a.name} missing {key}"
            assert len(sched[key]) == T, f"agent {a.name} {key} length {len(sched[key])} != {T}"


def test_peak_load_higher_welfare():
    """Peak load should have higher DA welfare than baseline."""
    T = 96
    config = MarketConfig(opf_mode="lindistflow", verbose=False)
    agents_b, _ = get_scenario("baseline", T=T)
    agents_p, _ = get_scenario("peak_load", T=T)
    r_b = clear_market(agents_b, T, "DA",
                       adaptive_bidding(agents_b, config, "random"), config)
    r_p = clear_market(agents_p, T, "DA",
                       adaptive_bidding(agents_p, config, "random"), config)
    assert r_p["welfare"] > r_b["welfare"], \
        f"peak_load welfare {r_p['welfare']:.0f} <= baseline {r_b['welfare']:.0f}"


def test_congestion_lower_welfare():
    """Congestion should have lower or equal welfare vs baseline."""
    T = 96
    config_b = MarketConfig(opf_mode="lindistflow", verbose=False)
    config_c = MarketConfig(opf_mode="lindistflow", verbose=False,
                            line_capacity_multiplier=0.5)
    agents_b, _ = get_scenario("baseline", T=T)
    agents_c, _ = get_scenario("congestion", T=T)
    r_b = clear_market(agents_b, T, "DA",
                       adaptive_bidding(agents_b, config_b, "random"), config_b)
    r_c = clear_market(agents_c, T, "DA",
                       adaptive_bidding(agents_c, config_c, "random"), config_c)
    assert r_c["welfare"] <= r_b["welfare"] * 1.05, \
        f"congestion welfare {r_c['welfare']:.0f} much higher than baseline {r_b['welfare']:.0f}"
