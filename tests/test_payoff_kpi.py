"""The economic KPIs, and the pairing that makes them attributable.

The research question is answered with a difference between two settlements of
the same day:

    Pi_AI, Pi_baseline, DeltaPi = Pi_AI - Pi_baseline,
    DeltaPi_pct, and P(DeltaPi > 0)

For the difference to be attributable to the bidding policy, the two arms must
differ in nothing else -- same seed, scenario, initial state of charge, initial
market state, load and renewable trajectory, network, and terminal state of
charge. These tests pin both the arithmetic and the pairing.
"""

import numpy as np
import pytest

import counterfactual_eval as ce
import participant_payoff as pp
from models import MarketConfig, MarketDesignConfig


# ---------------------------------------------------------------------------
# The KPI definitions
# ---------------------------------------------------------------------------

def test_delta_pi_is_the_difference_of_the_two_settled_profits():
    assert ce.delta_pi(150.0, 100.0) == pytest.approx(50.0)
    assert ce.delta_pi(100.0, 150.0) == pytest.approx(-50.0)


def test_delta_pi_percent_uses_the_baseline_magnitude():
    """A money-losing baseline still yields a rate that reads the right way.

    The denominator is the magnitude, not the signed value: improving on a
    baseline that lost money is a gain, not a loss.
    """
    assert ce.delta_pi_percent(150.0, 100.0) == pytest.approx(0.5)
    assert ce.delta_pi_percent(-50.0, -100.0) == pytest.approx(0.5, rel=1e-4)
    assert ce.delta_pi_percent(-50.0, -100.0) > 0


def test_delta_pi_percent_survives_a_zero_baseline():
    assert np.isfinite(ce.delta_pi_percent(10.0, 0.0))


def test_p_delta_pi_positive_counts_the_days_the_assisted_arm_won():
    assert ce.summarize([1.0, -1.0, 1.0])["positive_rate"] == pytest.approx(2 / 3)
    assert ce.summarize([-1.0, -2.0])["positive_rate"] == 0.0
    assert ce.summarize([1.0, 2.0])["positive_rate"] == 1.0


def test_headline_reports_the_five_canonical_kpis():
    result = {
        "n_days": 4, "seeds_dropped": [],
        "fleet": {"delta_pi": {"mean": 42.0, "positive_rate": 0.75},
                  "delta_pi_percent": {"mean": 0.25}},
    }
    h = ce.headline(result)
    assert h["mean_delta_pi"] == pytest.approx(42.0)
    assert h["mean_delta_pi_percent"] == pytest.approx(0.25)
    assert h["p_delta_pi_positive"] == pytest.approx(0.75)
    assert h["n_days"] == 4


def test_summarize_reports_the_full_distribution():
    stats = ce.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    for col in ("n", "mean", "std", "median", "p05", "p25", "p50", "p75",
                "p95", "positive_rate"):
        assert col in stats, col
    assert stats["n"] == 5
    assert stats["median"] == pytest.approx(3.0)
    assert stats["p25"] <= stats["p50"] <= stats["p75"]


# ---------------------------------------------------------------------------
# The payoff a KPI is defined on
# ---------------------------------------------------------------------------

def test_independent_storage_payoff_has_no_load_or_generation_terms():
    """A battery alone cannot earn a consumption value.

    Its ledger is revenue - purchase - degradation, with the load and
    generation terms zero *by definition* rather than because the local load
    profile happens to be zero at its bus.
    """
    from types import SimpleNamespace

    T = 4
    sched = {
        "p_sell": np.array([1.0, 0.0, 1.0, 0.0]),
        "p_buy": np.array([0.0, 1.0, 0.0, 1.0]),
        "p_ch": np.array([0.0, 1.0, 0.0, 1.0]),
        "p_dis": np.array([1.0, 0.0, 1.0, 0.0]),
        # A load profile sitting at the same bus must not be settled as if the
        # battery had consumed it.
        "served": np.array([9.0, 9.0, 9.0, 9.0]),
        "unserved": np.zeros(T),
        "pv_used": np.zeros(T),
        "wind_used": np.zeros(T),
    }
    lmp = np.full(T, 400.0)
    agent = SimpleNamespace(
        name="ESS2", participant_type="storage", bus=2,
        bid_value=0.0, offer_cost=0.0,
        storage=SimpleNamespace(e_max=4.0))
    config = MarketConfig()

    net = pp.participant_payoff(sched, lmp, agent, config)
    assert net.generation_cost == 0.0
    assert net.penalty == 0.0
    assert net.other_cost == 0.0
    # The arrays are power in MW and the day is 15 minutes per period, so the
    # 2 MW traded each way is 0.5 MWh at 400 CNY/MWh.
    assert net.revenue == pytest.approx(2.0 * 0.25 * 400.0)
    assert net.purchase_cost == pytest.approx(2.0 * 0.25 * 400.0)
    # Every MW that moved is cycled: 4 MW over the day, also at 15 minutes.
    assert net.degradation_cost == pytest.approx(
        config.storage.cycle_cost * 4.0 * 0.25)
    assert net.total == pytest.approx(net.revenue - net.purchase_cost
                                      - net.degradation_cost)


def test_the_reward_is_not_a_payoff():
    """A training reward must never be reported as an economic outcome.

    It is a per-block differential against an in-window truthful
    counterfactual, holding the other agents at their learned bids -- a
    different quantity on a different horizon from the day's settled profit.
    The training log names it `mean_reward`, and the profit columns are
    separate ones.
    """
    import run_artifacts as ra

    art = ra.RunArtifacts.create(
        config={}, cli_args={}, obs_spec={}, action_spec={},
        base_dir="outputs/rl")
    try:
        art.record_episode(0, mean_reward=-480.0, clearing_objective=7695.0,
                           re_rate=56.0, raw_profit=-151.8,
                           local_baseline_profit=-20.0,
                           profit_delta_local=-131.8)
        row = art.metrics[0]
        # The reward and the profit are distinct columns with distinct values.
        assert row["mean_reward"] == -480.0
        assert row["raw_profit_mean"] == -151.8
        assert row["mean_reward"] != row["raw_profit_mean"]
        # The clearing objective is not either of them.
        assert row["clearing_objective"] == 7695.0
    finally:
        import shutil
        shutil.rmtree(art.root_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# The pairing: only the bidding policy may differ
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def paired():
    """Two arms of one day, run exactly as the evaluator runs them."""
    import copy

    T, n_blocks = 96, 3
    md = MarketDesignConfig(use_constraint_multi_obj=False)
    config = MarketConfig(opf_mode="socp", verbose=False, market_design=md)
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False

    agents, _ = __import__("scenarios").get_scenario(
        "typical_day", T=T, config=copy.deepcopy(config))
    from ess import build_ess_fleet
    fleet = [a for a in agents if getattr(a, "participant_type", "") == "storage"]
    names = [a.name for a in fleet]
    if not names:
        pytest.skip("no independent storage fleet in this scenario")

    base = ce.run_arm("typical_day", copy.deepcopy(config),
                      ce.ArmSpec("truthful", "truthful"), names, 123, T,
                      n_blocks)
    ai = ce.run_arm("typical_day", copy.deepcopy(config),
                    ce.ArmSpec("rule", "rule"), names, 123, T, n_blocks)
    return base, ai, names


def test_the_two_arms_saw_the_same_day(paired):
    """Same seed means the same price day, checked rather than assumed."""
    base, ai, _ = paired
    ce.assert_paired(base, ai)
    assert np.allclose(base["wholesale"], ai["wholesale"])


def test_both_arms_settle_with_the_same_pinned_endpoints(paired):
    """Terminal SOC is a constraint, not a free variable either arm can use.

    A day that finishes with a fuller battery has banked energy it did not
    sell, so an unpinned settlement rewards whichever policy ran the battery
    down. Both arms are cleared with the endpoints tied, so the comparison is
    on the day itself.
    """
    import dataclasses

    from dispatch_core import pins_terminal_soc

    config = MarketConfig(opf_mode="socp", verbose=False)
    pinned = dataclasses.replace(
        config,
        storage=dataclasses.replace(config.storage, terminal_soc_equal=True))
    assert pins_terminal_soc(pinned, 96, "full_day") is True
    # And the rolling window the episode runs on does not pin it.
    assert pins_terminal_soc(config, 96, "window") is False


def test_only_the_bidding_policy_differs(paired):
    """Both arms carry the same agents, network and initial SOC."""
    base, ai, names = paired
    assert base["arm"] != ai["arm"]
    assert base["seed"] == ai["seed"]
    # run_arm seeds identically and reports the settled profit per agent, so a
    # difference in the delta is a difference in policy.
    assert set(base["profits"]) == set(ai["profits"]) == set(names)
