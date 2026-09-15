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

TARGET_PV_MW = 3.5
TARGET_WIND_MW = 1.5
TARGET_PEAK_MW = 10.0
ESS_BUSES = {6, 20, 24, 31}


@pytest.fixture(scope="module")
def day():
    """The canonical typical day, cleared once and shared by the tests."""
    config = typical_day_config()
    agents, wholesale = get_scenario("typical_day", money.PERIODS_PER_DAY, config)
    result = clear_market(agents, money.PERIODS_PER_DAY, "DA", {}, config,
                          wholesale=wholesale)
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
    _, agents, _, _ = day
    for a in _storage_agents(agents):
        assert a.storage.e_max == pytest.approx(4.0), f"{a.name} e_max"
        assert a.storage.p_ch_max == pytest.approx(1.0), f"{a.name} p_ch_max"
        assert a.storage.p_dis_max == pytest.approx(1.0), f"{a.name} p_dis_max"
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
    total = np.sum([a.load_forecast for a in agents], axis=0)
    coincident = float(np.max(total))
    assert abs(coincident - TARGET_PEAK_MW) <= 1e-6, \
        f"coincident peak {coincident:.6f} MW, target {TARGET_PEAK_MW}"

    # The load scale must not have moved the devices: they are sized from the
    # per-bus rated load, which the scale deliberately does not touch.
    assert abs(sum(a.pv_capacity for a in agents) - TARGET_PV_MW) <= 1e-6
    assert abs(sum(a.wind_capacity for a in agents) - TARGET_WIND_MW) <= 1e-6


def test_battery_fleet_is_sized_against_the_day(day):
    _, agents, _, result = day
    peak = sum(float(np.max(a.load_forecast)) for a in agents)
    energy = sum(float(np.sum(a.load_forecast)) for a in agents) * money.DT_HOURS
    fleet_power = sum(a.storage.p_ch_max for a in _storage_agents(agents))
    fleet_energy = sum(a.storage.e_max for a in _storage_agents(agents))
    assert fleet_power / peak == pytest.approx(0.36, abs=0.05)
    assert fleet_energy / energy == pytest.approx(0.09, abs=0.02)


def test_renewables_are_dispatched_not_merely_installed(day):
    """Renewable output has to be worth dispatching to the agents that own it.

    This is the consequence of removing a prosumer's battery before the bidding
    anchor is applied: a prosumer that inherits a battery's willingness to pay
    instead of its own load type's is priced above what the energy is worth, and
    its output is curtailed. Installed capacity alone would not show that.
    """
    _, _, _, result = day
    assert result["re_consumption_rate"] >= 90.0, \
        f"RE consumption rate {result['re_consumption_rate']:.2f}% below 90%"


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

    agents, wholesale = get_scenario("typical_day", 16, config)
    result_w = clear_market(agents, 16, "DA", {}, config, wholesale=wholesale,
                            horizon_type="window")
    assert not _ends_at_its_start(agents, result_w), \
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
