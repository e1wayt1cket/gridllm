"""Siting a battery by voltage self-sensitivity, and what that rules out.

The criterion is the **B0 operating-point-based finite-difference voltage
self-sensitivity**: how far a bus's own voltage moves under a small injection
at that same bus, on the benchmark network at its published load. No market
quantity takes part in the screen; market measurements appear only as a
feasibility diagnostic.
"""

import numpy as np
import pytest

import ess_siting as es
from config_loader import get_default


@pytest.fixture(scope="module")
def rows():
    return es.sensitivity_table()


@pytest.fixture(scope="module")
def stability():
    return es.check_eps_stability()


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_the_finite_difference_is_stable_against_its_step_size(stability):
    """The step size must not be doing any work.

    A finite difference is a derivative only while it stays in the linear
    regime, so the whole screen is re-run at twice the step and compared. If
    the ranking moved, the criterion would be a property of `eps` rather than
    of the network.
    """
    assert stability["ranking_identical"] is True
    assert stability["max_relative_difference"] < 0.05, stability


def test_the_weakest_bus_scores_highest(rows):
    """Bus 17 is the network's electrical end and must top the ranking.

    B0's voltage minimum is there (0.9131 pu), so the bus whose voltage is
    worst is also the bus where an injection helps most. A screen that missed
    it would be measuring something other than voltage sensitivity.
    """
    top = max(rows, key=lambda r: r["dV_dP_pu_per_mw"])
    assert top["bus"] == 17
    assert top["normalized_sensitivity"] == pytest.approx(1.0)


def test_a_bus_beside_the_slack_scores_near_zero(rows):
    """The network is stiff at the source, so an injection there does nothing."""
    bus1 = next(r for r in rows if r["bus"] == 1)
    assert bus1["normalized_sensitivity"] < 0.05
    assert bus1["candidate"] is False


def test_sensitivity_is_positive_at_every_bus(rows):
    """An injection raises voltage everywhere; a negative score is a bug."""
    assert all(r["dV_dP_pu_per_mw"] > 0.0 for r in rows)


# ---------------------------------------------------------------------------
# The candidate pool
# ---------------------------------------------------------------------------

def test_the_slack_is_not_a_candidate(rows):
    """The slack carries no load, so it is never measured and never selected.

    `case33bw` puts a load on buses 1-32 and none on the slack, which is why
    the table has 32 rows: the screen iterates the loads, and a bus with no
    load is not somewhere a battery could serve anything.
    """
    slack = es.slack_bus()
    assert slack not in {r["bus"] for r in rows}
    assert slack not in es.candidate_buses(rows)
    assert len(rows) == 32


def test_renewable_buses_are_not_candidates(rows):
    """A battery never shares a bus with a renewable, as the network always had."""
    re_buses = es.renewable_buses("typical_day")
    assert re_buses == {21, 23, 30}
    pool = set(es.candidate_buses(rows))
    assert pool.isdisjoint(re_buses), pool & re_buses


def test_bus_17_is_in_the_pool(rows):
    """It must not be excluded for being on the main trunk.

    It is the most sensitive bus on the feeder; excluding it on a topological
    label would discard the network's own voltage-critical end.
    """
    r17 = next(r for r in rows if r["bus"] == 17)
    assert r17["passes_sensitivity_floor"] is True
    assert r17["candidate"] is True


def test_the_floor_is_the_declared_share_of_the_maximum(rows):
    top = max(r["dV_dP_pu_per_mw"] for r in rows)
    for r in rows:
        expected = r["dV_dP_pu_per_mw"] >= es.SENSITIVITY_FLOOR * top
        assert r["passes_sensitivity_floor"] is expected, r["bus"]


def test_electrical_distance_is_a_diagnostic_and_not_the_criterion(rows):
    """It is reported, it is not the screen.

    The two disagree in the way that matters: `electrical_distance` is largest
    at bus 32 (the far end of the longest spur) while the sensitivity ranking is
    topped by bus 17. A regression that swapped them would show up here.
    """
    by_dist = max(rows, key=lambda r: r["electrical_distance"])
    by_sens = max(rows, key=lambda r: r["dV_dP_pu_per_mw"])
    assert by_dist["bus"] != by_sens["bus"]
    assert by_dist["bus"] == 32


# ---------------------------------------------------------------------------
# The selection rule
# ---------------------------------------------------------------------------

def test_selection_is_deterministic(rows):
    assert es.select_buses(rows) == es.select_buses(rows)


def test_selection_stays_inside_the_pool(rows):
    assert set(es.select_buses(rows)) <= set(es.candidate_buses(rows))


def test_selection_does_not_take_the_top_four_by_sensitivity(rows):
    """Four units on the four most sensitive buses would crowd each other.

    They arbitrage the same daily spread, so the fleet is spread by shared
    conductor instead. Only the seed is chosen on sensitivity.
    """
    selected = es.select_buses(rows)
    top4 = [r["bus"] for r in sorted(
        rows, key=lambda r: -r["dV_dP_pu_per_mw"])[:4]]
    assert sorted(selected) != sorted(top4)
    assert len(set(selected)) == es.FLEET_SIZE


def test_selection_spreads_the_fleet(rows):
    """Every chosen bus is further from the others than the pool's closest pair.

    The rule is max-min dispersion, so the minimum pairwise distance among the
    chosen four must beat what picking the four most sensitive would give.
    """
    from grid import build_base_network
    from models import MarketConfig
    from rl_physics import pairwise_resistance_distance

    net = build_base_network(MarketConfig(opf_mode="socp", verbose=False))
    pool = es.candidate_buses(rows)
    D = pairwise_resistance_distance(net, pool)
    idx = {b: i for i, b in enumerate(pool)}

    def min_pair(chosen):
        return min(D[idx[a], idx[b]] for i, a in enumerate(chosen)
                   for b in chosen[i + 1:])

    assert min_pair(es.select_buses(rows)) > 0.0
    top4 = [r["bus"] for r in sorted(
        rows, key=lambda r: -r["dV_dP_pu_per_mw"])[:4]]
    assert min_pair(es.select_buses(rows)) >= min_pair(top4)


# ---------------------------------------------------------------------------
# The fleet the config actually builds
# ---------------------------------------------------------------------------

def test_the_configured_fleet_is_the_selected_one():
    """The selection is only real if it reached the config."""
    rows = es.sensitivity_table()
    assert sorted(int(b) for b in
                  get_default("independent_storage.buses")) \
        == es.select_buses(rows)


def test_the_fleet_is_not_oversized_against_the_feeder():
    """The resize exists so the batteries cannot decide the day alone.

    At native scale the old 4 MWh units gave a 4 MW fleet against a 3.715 MW
    peak -- more power than every load on the network draws at its busiest
    instant. The declared fleet must stay under the peak.
    """
    capacity = float(get_default("independent_storage.capacity_mwh"))
    ratio = float(get_default("independent_storage.power_ratio"))
    n = len(get_default("independent_storage.buses"))
    from grid import build_base_network, native_nominal_demand_mw
    from models import MarketConfig
    peak = native_nominal_demand_mw(
        build_base_network(MarketConfig(opf_mode="socp", verbose=False)))
    assert n * capacity * ratio < peak
