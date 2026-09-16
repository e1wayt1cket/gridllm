# tests/test_der_relocation.py
"""Relocating a renewable site must conserve what it moves.

The zero-shedding search asks where an array should be, and that question is
only well posed if the answer is not also changing how big the array is. Every
candidate is a *move* of an existing site, so the installed MW and the energy
profile are conserved by construction; these tests hold that construction.

Nothing here clears a market: the invariants are properties of the built agents.
"""

import numpy as np
import pytest

import money
from scenarios import get_scenario, relocate_renewables, typical_day_config
from diagnose_typical_day import DER_LAYOUTS

PV_TOTAL, WIND_TOTAL = 3.5, 1.5


def _agents(der_moves=None):
    config = typical_day_config()
    agents, _ = get_scenario("typical_day", money.PERIODS_PER_DAY, config,
                             der_moves=der_moves)
    return agents


def _totals(agents):
    pv_mw = sum(a.pv_capacity for a in agents)
    wind_mw = sum(a.wind_capacity for a in agents)
    pv_mwh = sum(float(np.sum(a.pv_forecast)) for a in agents) * money.DT_HOURS
    wind_mwh = sum(float(np.sum(a.wind_forecast)) for a in agents
                   if a.has_wind) * money.DT_HOURS
    return pv_mw, wind_mw, pv_mwh, wind_mwh


def test_the_configured_layout_is_the_baseline_the_moves_are_measured_from():
    pv_mw, wind_mw, pv_mwh, wind_mwh = _totals(_agents())
    assert pv_mw == pytest.approx(PV_TOTAL)
    assert wind_mw == pytest.approx(WIND_TOTAL)
    assert pv_mwh > 0 and wind_mwh > 0


@pytest.mark.parametrize("name", sorted(DER_LAYOUTS))
def test_every_candidate_conserves_installed_mw_and_energy(name):
    """MW, and the energy under them, are the same before and after a move."""
    base_mw, base_wind_mw, base_mwh, base_wind_mwh = _totals(_agents())
    moved = _totals(_agents(DER_LAYOUTS[name]))

    assert moved[0] == pytest.approx(base_mw), f"{name}: PV MW changed"
    assert moved[1] == pytest.approx(base_wind_mw), f"{name}: wind MW changed"
    assert moved[2] == pytest.approx(base_mwh), f"{name}: PV energy changed"
    assert moved[3] == pytest.approx(base_wind_mwh), f"{name}: wind energy changed"


@pytest.mark.parametrize("name", sorted(DER_LAYOUTS))
def test_every_candidate_holds_the_layout_invariants(name):
    """The conditions the plan puts on a candidate, checked not assumed.

    PV 3.5 MW, wind 1.5 MW, exactly four batteries, at least two PV buses, and
    no bus carrying both renewables.
    """
    agents = _agents(DER_LAYOUTS[name])
    pv_mw, wind_mw, _, _ = _totals(agents)
    assert pv_mw == pytest.approx(PV_TOTAL), name
    assert wind_mw == pytest.approx(WIND_TOTAL), name

    fleet = [a for a in agents if getattr(a, "participant_type", "") == "storage"]
    assert len(fleet) == 4, f"{name}: {len(fleet)} batteries"

    pv_buses = {a.bus for a in agents if a.pv_capacity > 0}
    wind_buses = {a.bus for a in agents if a.wind_capacity > 0}
    assert len(pv_buses) >= 2, f"{name}: PV on {sorted(pv_buses)}"
    assert not (pv_buses & wind_buses), \
        f"{name}: {sorted(pv_buses & wind_buses)} carries both"


def test_an_emptied_wind_site_stops_being_a_wind_site():
    """`has_wind` is a None check, so zeroing the capacity is not enough."""
    agents = _agents([(30, 32, "wind")])
    source = next(a for a in agents if a.bus == 30)
    target = next(a for a in agents if a.bus == 32)

    assert source.wind_capacity == pytest.approx(0.0)
    assert not source.has_wind, "bus 30 still models a turbine with a zero rating"
    assert target.wind_capacity == pytest.approx(WIND_TOTAL)
    assert target.has_wind


def test_a_split_lands_where_it_says_and_empties_the_source():
    agents = _agents([(21, 17, "pv", 0.5), (21, 2, "pv", 1.0)])
    got = {a.bus: a.pv_capacity for a in agents if a.pv_capacity > 0}
    assert got == {17: pytest.approx(1.0), 2: pytest.approx(1.0),
                   23: pytest.approx(1.5)}


def test_moving_from_a_bus_that_has_nothing_raises():
    with pytest.raises(ValueError, match="carries no PV"):
        _agents([(25, 17, "pv")])


def test_moving_onto_an_occupied_bus_raises():
    with pytest.raises(ValueError, match="already carries"):
        _agents([(21, 23, "pv")])


def test_moving_to_a_bus_with_no_agent_raises():
    with pytest.raises(ValueError, match="no agent at bus"):
        _agents([(21, 999, "pv")])


def test_an_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        _agents([(21, 17, "geothermal")])


def test_a_fraction_outside_the_unit_interval_raises():
    with pytest.raises(ValueError, match="fraction must be"):
        relocate_renewables(_agents(), [(21, 17, "pv", 1.5)])
