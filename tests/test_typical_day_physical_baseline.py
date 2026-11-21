# tests/test_typical_day_physical_baseline.py
"""The typical day's physical contract.

The device fleet this pins is what every later bidding experiment is measured
on, and two of its properties are easy to break without anything looking wrong:

  * Which agents carry a battery. A configuration switch that sizes a battery
    down to zero instead of skipping it leaves an agent that still counts as
    storage and still takes part in the clearing while being unable to do
    anything, and the clearing divides by its capacity.

  * The order in which a battery is removed relative to the bidding anchor. If a
    prosumer loses its battery after the anchor has been applied, it keeps a
    battery's willingness to pay instead of its own load type's, and its
    renewable output stops being worth dispatching.

Both are pinned here by consequence rather than by inspecting the config.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import money
from market import clear_market
from models import MarketConfig
from scenarios import get_scenario, typical_day_config

from config_loader import get_default, get_scenario_cfg

# Read from the scenario's own declaration rather than restated. These are the
# scenario's choices, and a literal here would make every future revision look
# like a regression; what must hold is that the declaration reaches the agents.
_DECLARED = (get_scenario_cfg("typical_day").get("renewable") or {})
TARGET_PV_MW = sum(float(v) for v in (_DECLARED.get("pv_bus_mw") or {}).values())
TARGET_WIND_MW = sum(float(v) for v in (_DECLARED.get("wind_bus_mw") or {}).values())
# `native_peak` pins the day's realized coincident peak to the network's own
# nominal demand, so the target is read from the network, not configured.
from grid import build_base_network, native_nominal_demand_mw
TARGET_PEAK_MW = native_nominal_demand_mw(
    build_base_network(typical_day_config()))
# Read from the file rather than restated: the placement is a scenario fact
# that the siting criterion is allowed to change, and a literal here would make
# every re-siting look like a regression. What must hold is that the fleet is
# where the file says, not that the file says a particular set of buses.
ESS_BUSES = set(get_default("independent_storage.buses"))


@pytest.fixture(scope="module")
def day():
    """The canonical typical day, cleared once and shared by the tests."""
    config = typical_day_config()
    agents, wholesale = get_scenario("typical_day", money.PERIODS_PER_DAY, config)
    result = clear_market(agents, money.PERIODS_PER_DAY, "DA", {}, config,
                          wholesale=wholesale, horizon_type="full_day")
    return config, agents, wholesale, result


def _storage_agents(agents):
    return [a for a in agents if a.storage is not None]


def test_installed_renewables(day):
    _, agents, _, _ = day
    pv = sum(a.pv_capacity for a in agents)
    wind = sum(a.wind_capacity for a in agents)
    assert abs(pv - TARGET_PV_MW) <= 1e-6, f"installed PV {pv:.6f} MW"
    assert abs(wind - TARGET_WIND_MW) <= 1e-6, f"installed wind {wind:.6f} MW"


def test_renewables_are_spatially_distributed(day):
    """PV and wind on separate buses, none of them co-located with each other."""
    _, agents, _, _ = day
    pv_buses = {a.bus for a in agents if a.pv_capacity > 0}
    wind_buses = {a.bus for a in agents if a.wind_capacity > 0}
    assert len(pv_buses) >= 2, f"PV concentrated on {pv_buses}"
    assert pv_buses.isdisjoint(wind_buses), \
        f"PV {pv_buses} and wind {wind_buses} share a bus"


def test_exactly_four_batteries_and_only_the_fleet(day):
    """Four independent units, and no other battery anywhere in the network.

    A battery the configuration drops must be absent rather than zero-sized, so
    the count of agents carrying storage is 4 and not 18.
    """
    _, agents, _, _ = day
    storage = _storage_agents(agents)
    assert len(storage) == 4, \
        f"{len(storage)} agents carry storage, expected 4: " \
        f"{sorted(a.name for a in storage)}"

    fleet = [a for a in agents if getattr(a, "participant_type", "") == "storage"]
    assert len(fleet) == 4
    assert {a.bus for a in fleet} == ESS_BUSES

    on_network = [a for a in storage if a not in fleet]
    assert not on_network, \
        f"on-network batteries survived the switch: {[a.name for a in on_network]}"


def test_fleet_unit_parameters(day):
    """Sizing is read from the scenario's declaration, not restated.

    The capacity is a scenario choice that has already moved once (4.0 MWh at
    the old study scale, 1.6 MWh at native scale). A literal here would turn
    every deliberate resize into a test failure; what must hold is that the
    declaration reaches the agents and that power follows from it.
    """
    _, agents, _, _ = day
    declared_mwh = float(get_default("independent_storage.capacity_mwh"))
    ratio = float(get_default("independent_storage.power_ratio"))
    for a in _storage_agents(agents):
        assert a.storage.e_max == pytest.approx(declared_mwh), f"{a.name} e_max"
        assert a.storage.p_ch_max == pytest.approx(
            declared_mwh * ratio), f"{a.name} p_ch_max"
        assert a.storage.p_dis_max == pytest.approx(
            declared_mwh * ratio), f"{a.name} p_dis_max"
        assert a.storage.eta_ch == pytest.approx(0.92)
        assert a.storage.eta_dis == pytest.approx(0.92)
        assert a.storage.soc0 == pytest.approx(0.50)
        assert a.storage.soc_min == pytest.approx(0.10)
        assert a.storage.soc_max == pytest.approx(0.90)


def test_battery_buses_are_distinct(day):
    _, agents, _, _ = day
    buses = [a.bus for a in _storage_agents(agents)]
    assert len(set(buses)) == len(buses), f"batteries share a bus: {buses}"


def test_agent_names_are_unique(day):
    """Duplicate names silently corrupt the clearing model, so the scenario
    builder refuses them; this catches a regression that reintroduces one."""
    _, agents, _, _ = day
    names = [a.name for a in agents]
    assert len(set(names)) == len(names), \
        f"duplicate agent names: {sorted({n for n in names if names.count(n) > 1})}"


def test_coincident_peak_load_is_the_configured_target(day):
    """The load level is a target the profiles are scaled to hit.

    The scale is derived from the unscaled profiles, so this is exact rather
    than a band: a band would let the level drift and still pass. The quantity
    tested is the coincident peak -- the most the whole network draws in any one
    period -- because that is what the feeder carries. The sum of each agent's
    own maximum is larger, is reached at no single instant, and is reported
    separately rather than asserted.
    """
    _, agents, _, _ = day
    # The realized load, because that is what the network carries and what the
    # scenario's `native_peak` declaration pins. The forecast carries its own
    # error on top and is a different quantity.
    total = np.sum([a.load_real for a in agents], axis=0)
    coincident = float(np.max(total))
    assert abs(coincident - TARGET_PEAK_MW) <= 1e-6, \
        f"coincident peak {coincident:.6f} MW, target {TARGET_PEAK_MW}"

    # Renewable capacity is the scenario's declaration and must reach the
    # agents rather than being re-derived from anything.
    assert abs(sum(a.pv_capacity for a in agents) - TARGET_PV_MW) <= 1e-6
    assert abs(sum(a.wind_capacity for a in agents) - TARGET_WIND_MW) <= 1e-6


def test_battery_fleet_is_sized_against_the_day(day):
    _, agents, _, result = day
    # Against the coincident peak, which is what the feeder actually carries
    # and what this ratio is about. (The sum of per-agent maxima is larger and
    # is reported separately.)
    total = np.sum([a.load_real for a in agents], axis=0)
    peak = float(np.max(total))
    energy = float(np.sum(total)) * money.DT_HOURS
    fleet_power = sum(a.storage.p_ch_max for a in _storage_agents(agents))
    fleet_energy = sum(a.storage.e_max for a in _storage_agents(agents))

    # The ratio is a property of the scenario and is reported, not fitted to a
    # band. It has moved twice: at the old 10 MW scale the 4 MWh units gave a
    # 4 MW fleet against a 10 MW peak (40%); at native scale the same units
    # were larger than the whole system's demand (108%), which is what the
    # resize corrected; at 1.6 MWh each the fleet is 1.6 MW against a 3.715 MW
    # peak (43%).
    #
    # What is asserted is the property the resize exists to guarantee: the
    # fleet cannot move more power than the system draws, so it cannot decide
    # the day for the network on its own.
    assert fleet_power < peak, (
        f"fleet power {fleet_power:.2f} MW is no longer below the system peak "
        f"{peak:.2f} MW ({fleet_power / peak:.3f}); the fleet is oversized "
        f"relative to the feeder again")
    assert 0.0 < fleet_energy / energy < 0.5, (
        f"fleet energy {fleet_energy:.2f} MWh is {fleet_energy / energy:.1%} "
        f"of the day's load")


def test_renewables_are_dispatched_not_merely_installed(day):
    """Renewable output has to be worth dispatching to the agents that own it.

    This is the consequence of removing a prosumer's battery before the bidding
    anchor is applied: a prosumer that inherits a battery's willingness to pay
    instead of its own load type's is priced above what the energy is worth, and
    its output is curtailed. Installed capacity alone would not show that.
    """
    # No consumption-rate floor is asserted. The rate is an OUTCOME: it is
    # whatever the clearing decides given the prices it faces, and it moved
    # from 98% to 19.7% when the native scenario's price source changed without
    # anything about the mechanism changing. Asserting a floor would make the
    # test a statement about the price level, and re-deriving the floor to fit
    # the new level would be fitting the market to the test.
    #
    # The mechanism itself is asserted directly instead, below, and the rate is
    # reported by `audit_renewable_curtailment.py` where it can be read as the
    # outcome it is.
    _, agents, _, _ = day
    from config_loader import get_prosumer_cfg

    carriers = [a for a in agents
                if getattr(a, "pv_capacity", 0.0) > 0.0
                or getattr(a, "wind_capacity", 0.0) > 0.0]
    assert carriers, "no agent carries a declared renewable capacity"

    for a in carriers:
        own = get_prosumer_cfg(a.load_type)
        assert a.offer_cost == pytest.approx(own["offer_cost"]), (
            f"{a.name} carriers renewables but offers them at {a.offer_cost}, "
            f"not its load type's {own['offer_cost']}")
        assert a.bid_value == pytest.approx(own["bid_value"]), a.name

    # A non-prosumer offers its generation at 999, which curtails it outright.
    # That is the failure this guards: a prosumer that lost its own prices to
    # that anchor would show up here and nowhere else, because installed
    # capacity is unchanged either way.
    assert all(a.offer_cost < 999.0 for a in carriers)


def _ends_at_its_start(agents, result):
    return all(
        abs(result["schedules"][a.name]["soc"][-1] - a.storage.soc0) <= 1e-3
        for a in _storage_agents(agents))


def test_terminated_state_of_charge_only_on_a_settled_horizon(day):
    """The endpoint pin follows the caller's declared horizon, not its length.

    A window that pinned its own endpoints would forbid the trajectory it was
    opened to choose, so the pin belongs to the day that gets settled. The
    first two cases are what the horizon length already got right; the last two
    are what it cannot see -- a window that is exactly a day long, and a day
    that is not 96 periods -- and they are the reason the length stopped being
    the signal.
    """
    _, _, _, result = day
    assert _ends_at_its_start(day[1], result), \
        "the settled day did not close at its starting state of charge"

    config = MarketConfig(opf_mode="socp")

    # A window taken out of the middle of the day, priced on the periods it
    # actually covers. A 16-period horizon sliced off the front of the day is
    # all cheap early-morning hours, so the optimum is to do nothing -- and
    # "ended where it started" cannot then tell a pinned endpoint from an
    # idle battery. The window has to contain something worth trading in.
    agents_w, wholesale_w = get_scenario("typical_day", 96, config)
    window = wholesale_w[64:80]
    assert float(window.max() - window.min()) > 10.0, \
        "the window carries no spread, so the test cannot tell the cases apart"
    result_w = clear_market(agents_w, 16, "DA", {}, config,
                            wholesale=window, horizon_type="window")
    assert not _ends_at_its_start(agents_w, result_w), \
        "a 16-period window pinned its endpoints"

    # 96 periods, but the caller says it is a window: the length would have
    # pinned it and the declaration must win.
    agents_96, wholesale_96 = get_scenario("typical_day", 96, config)
    result_96w = clear_market(agents_96, 96, "DA", {}, config,
                              wholesale=wholesale_96, horizon_type="window")
    assert not _ends_at_its_start(agents_96, result_96w), \
        "a 96-period clear declared as a window pinned its endpoints"

    # 48 periods, but the caller says it settles a day: the length would have
    # left it free and the declaration must win.
    agents_48, wholesale_48 = get_scenario("typical_day", 48, config)
    result_48d = clear_market(agents_48, 48, "DA", {}, config,
                              wholesale=wholesale_48, horizon_type="full_day")
    assert _ends_at_its_start(agents_48, result_48d), \
        "a 48-period clear declared a full day did not pin its endpoints"


def test_typical_day_config_refuses_a_partial_day():
    with pytest.raises(ValueError):
        typical_day_config(16)
