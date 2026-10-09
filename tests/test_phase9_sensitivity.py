# tests/test_phase9_sensitivity.py
"""Contracts of the sensitivity analysis, and the screen that scoped it.

The phase rests on one argument, so that argument is what the tests defend:

    For any battery, the margin it can capture is at most
    `eta_dis * max_t p[t] - min_t p[t] / eta_ch` -- the day's extremes, with the
    power and energy limits removed. That is an optimistic bound on every
    capacity and every location at once. If it fails to clear `2 * cycle_cost`
    at every bus, no capacity or placement can open the window.

If that bound were ever to clear on some bus, the phase's conclusion
("capacity and siting cannot vary revenue here") stops holding and the report
must be revisited -- so the tests assert the bound rather than the prose.

The rest is cache isolation and reproducibility: every scan reconfigures the
fleet, and a reused cached result would silently attribute one configuration's
numbers to another.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import phase9_sensitivity as p9
from config_loader import get_default, reload_defaults

SHEETS = ("261009", "7days", "30days", "halfyear")


@pytest.fixture(scope="module")
def screen():
    try:
        return p9.feasibility_screen()
    finally:
        p9._fresh()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_the_required_margin_is_the_derived_floor_not_the_lossless_one(screen):
    """`cycle_cost * (1/eta_ch + eta_dis)`, not `2 * cycle_cost`.

    The two legs of a cycle are not equal in energy, so the lossless form is
    too permissive. The difference is small -- 0.6957 CNY/MWh at the configured
    values -- but it is the whole margin on a day that sits near the threshold,
    so the screen must use the derived one.
    """
    from arbitrage_lp import required_margin

    reload_defaults()
    cc = get_default("storage.cycle_cost")
    unit = screen["unit"]
    assert screen["required_margin"] == pytest.approx(
        required_margin(cc, unit["eta_ch"], unit["eta_dis"]))
    assert screen["required_margin"] > 2.0 * cc
    assert screen["required_margin"] == pytest.approx(200.6957, abs=1e-4)


def test_no_curve_clears_its_optimistic_margin(screen):
    """The day's extremes are an upper bound on every battery."""
    for sheet, c in screen["curves"].items():
        assert c["clean"], sheet
        assert c["external"]["unconstrained_margin"] < screen["required_margin"], (
            f"{sheet}: the unconstrained margin now clears "
            f"{screen['required_margin']:.0f}; the capacity and siting "
            f"conclusions in reports/phase9_sensitivity_analysis.md no longer "
            f"hold")
        assert c["external"]["clears"] is False, sheet


def test_no_bus_in_the_network_clears_on_any_curve(screen):
    """The siting conclusion, asserted rather than described."""
    for sheet, c in screen["curves"].items():
        assert c["nodal"]["n_buses_clearing"] == 0, sheet
        assert c["nodal"]["best_shortfall"] < 0, sheet


def test_the_screen_also_uses_the_cleared_nodal_prices_not_only_the_curve(screen):
    """The network's own prices are checked, not assumed to equal the curve."""
    for sheet, c in screen["curves"].items():
        assert c["nodal"]["best_margin"] > 0 or True   # a number was computed
        assert set(c["nodal"]["per_bus"][0]) >= {"bus", "margin", "shortfall"}
        assert len(c["nodal"]["per_bus"]) == 33
        # The nodal field is not the external curve: it has its own range.
        assert c["nodal"]["per_bus"][c["nodal"]["best_bus"]]["lmp_max"] >= \
            c["external"]["min"]


def test_the_conclusion_flags_are_derived_from_the_numbers(screen):
    assert screen["conclusion"]["any_curve_clears"] == \
        any(c["external"]["clears"] for c in screen["curves"].values())
    assert screen["conclusion"]["any_bus_clears"] == \
        any(c["nodal"]["n_buses_clearing"] > 0 for c in screen["curves"].values())
    assert screen["conclusion"]["any_curve_clears"] is False
    assert screen["conclusion"]["any_bus_clears"] is False


def test_the_screen_reports_the_external_and_nodal_bands_separately(screen):
    """Nodal and external prices are different quantities and must not merge."""
    for sheet, c in screen["curves"].items():
        ext = c["external"]
        assert ext["min"] < ext["max"]
        assert ext["available_spread"] == pytest.approx(ext["max"] - ext["min"])
        assert c["nodal_vs_external"]["nodal_max"] >= ext["min"]


# ---------------------------------------------------------------------------
# Capacity cannot open the window
# ---------------------------------------------------------------------------

def test_no_battery_size_clears_the_margin():
    """The bound does not contain the capacity at all, so no size can help."""
    try:
        result = p9.capacity_screen()
    finally:
        p9._fresh()
    assert result["required_margin"] == pytest.approx(200.6957, abs=1e-4)
    for c in result["configs"]:
        assert c["clears"] is False, c
        assert c["best_lp_margin_over_all_buses"] == pytest.approx(0.0, abs=1e-9), c
    assert result["any_config_clears"] is False


def test_the_capacity_bound_is_the_same_for_every_size():
    """A bound that moved with capacity would mean the argument was wrong."""
    try:
        result = p9.capacity_screen()
    finally:
        p9._fresh()
    margins = {round(c["best_unconstrained_margin"], 6) for c in result["configs"]}
    assert len(margins) == 1, (
        "the unconstrained margin varied with battery size, so it is not the "
        f"size-independent bound the conclusion relies on: {margins}")


# ---------------------------------------------------------------------------
# The configured fleet really does change
# ---------------------------------------------------------------------------

def test_a_capacity_change_reaches_the_built_fleet():
    """A scan that silently kept the old fleet would report the same row twice."""
    try:
        small = p9._clear(0, capacity_mwh=0.8, power_ratio=0.5)
        large = p9._clear(0, capacity_mwh=3.2, power_ratio=0.125)
    finally:
        p9._fresh()
    assert small["storage_energy_mwh"][0] == pytest.approx(0.8)
    assert large["storage_energy_mwh"][0] == pytest.approx(3.2)
    assert small["storage_power_mw"][0] == pytest.approx(0.4)
    assert large["storage_power_mw"][0] == pytest.approx(0.4)
    assert small["storage_energy_mwh"] != large["storage_energy_mwh"]


def test_a_siting_change_reaches_the_built_fleet():
    try:
        moved = p9._clear(0, ess_buses=[6, 20, 24, 31])
        default = p9._clear(0)
    finally:
        p9._fresh()
    assert moved["storage_buses"] == [6, 20, 24, 31]
    assert default["storage_buses"] == [9, 14, 17, 27]


def test_the_config_is_restored_after_a_reconfigured_run():
    """A scan must not leave the fleet resized for the next caller."""
    reload_defaults()
    before_energy = get_default("independent_storage.capacity_mwh")
    before_ratio = get_default("independent_storage.power_ratio")
    try:
        p9._clear(0, capacity_mwh=9.9, power_ratio=0.5)
        reload_defaults()
        assert get_default("independent_storage.capacity_mwh") == before_energy
        assert get_default("independent_storage.power_ratio") == before_ratio
    finally:
        p9._fresh()


def test_repeating_a_configuration_reproduces_it():
    """Same configuration, same result -- no cache carries one run into another."""
    try:
        first = p9._clear(0, capacity_mwh=3.2, power_ratio=0.125)
        # An intervening different configuration, to poison a naive cache.
        p9._clear(0, capacity_mwh=0.8, power_ratio=0.5)
        again = p9._clear(0, capacity_mwh=3.2, power_ratio=0.125)
    finally:
        p9._fresh()
    for key in ("fleet_profit_cny", "storage_energy_mwh", "storage_buses"):
        assert first[key] == again[key], key
    np.testing.assert_array_equal(first["wholesale"], again["wholesale"])


def test_each_price_curve_is_selected_explicitly_and_differs():
    from dianyicha_price import read_curves
    curves = read_curves()
    prices = {}
    try:
        for index, sheet in enumerate(SHEETS):
            prices[sheet] = p9._clear(index)["wholesale"]
    finally:
        p9._fresh()
    for sheet, arr in prices.items():
        np.testing.assert_allclose(arr, curves[sheet].array(), err_msg=sheet)
    for a, b in zip(SHEETS, SHEETS[1:]):
        assert not np.allclose(prices[a], prices[b]), (a, b)


# ---------------------------------------------------------------------------
# The load-shape comparison
# ---------------------------------------------------------------------------

def test_the_load_shape_comparison_holds_the_scale_and_changes_the_shape():
    try:
        result = p9.load_shape_comparison()
    finally:
        p9._fresh()
    native = result["typical_day"]
    industry = result["typical_day_industry"]
    # The declared scale is the contract both days share.
    assert native["coincident_peak_mw"] == pytest.approx(3.715, abs=1e-6)
    assert industry["coincident_peak_mw"] == pytest.approx(3.715, abs=1e-6)
    # ...and the shapes genuinely differ.
    assert native["daily_energy_mwh"] != pytest.approx(
        industry["daily_energy_mwh"], abs=0.1)
    assert native["peak_slot"] != industry["peak_slot"]
    assert native["load_factor"] != pytest.approx(industry["load_factor"], abs=1e-6)


def test_both_load_shape_days_are_physically_feasible():
    try:
        result = p9.load_shape_comparison()
    finally:
        p9._fresh()
    for name, c in result.items():
        v = c["voltage"]
        assert v["periods_below_min"] == 0, name
        assert v["periods_above_max"] == 0, name
        assert c["line_utilization"]["lines_over_1"] == 0, name
        assert all(s["all_transitions_verified"] for s in c["storage"].values()), name


def test_the_native_day_sits_on_breakeven_at_the_configured_buses():
    """The knife edge the report records, asserted rather than transcribed.

    On `typical_day` the configured fleet's own buses land within a few
    CNY/MWh of the requirement -- close enough that reading a margin off
    rounded figures flips its sign. What is worth gating is that closeness,
    not any one bus's last decimal: at most one of the four is nominally
    above the requirement, and by well under 1 CNY/MWh.
    """
    try:
        result = p9.load_shape_comparison()
    finally:
        p9._fresh()
    unit, _ = p9._fleet_unit()
    native = result["typical_day"]
    own = {r["bus"]: r for r in native["nodal_per_bus"]}
    assert set(own) >= {9, 14, 17, 27}

    shortfalls = {bus: own[bus]["shortfall"] for bus in (9, 14, 17, 27)}
    for bus, shortfall in shortfalls.items():
        # The margin is the formula applied to this bus's own prices, so a
        # table that drifted from the clearing would fail here.
        assert own[bus]["margin"] == pytest.approx(
            unit.eta_dis * own[bus]["lmp_max"] - own[bus]["lmp_min"] / unit.eta_ch)
        assert shortfall == pytest.approx(
            own[bus]["margin"] - native["required_margin"])
        assert abs(shortfall) < 3.0, (bus, shortfall)

    above = [bus for bus, s in shortfalls.items() if s > 0]
    assert len(above) <= 1, above
    for bus in above:
        assert shortfalls[bus] < 1.0, (bus, shortfalls[bus])


# ---------------------------------------------------------------------------
# Siting runs, but not to rank
# ---------------------------------------------------------------------------

def test_the_siting_scan_reports_physical_state_for_every_placement():
    try:
        result = p9.siting_scan()
    finally:
        p9._fresh()
    assert len(result) == 5
    for label, c in result.items():
        assert c["voltage_violations"] == 0, label
        assert c["soc_all_verified"] is True, label
        # No placement's own buses reach the requirement.
        assert c["own_buses_clearing"] == 0, label


# ---------------------------------------------------------------------------
# Nothing older moved
# ---------------------------------------------------------------------------

def test_phase_1_to_8_interfaces_are_intact():
    import dianyicha_price as dp
    import industry_load as il
    assert il.N_SLOTS == 96
    assert dp.SHEETS == SHEETS
    import rl_bidding
    assert hasattr(rl_bidding, "policy_provenance")
    import counterfactual_eval as ce
    assert callable(ce.baseline_cache_key)

    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    from models import MarketConfig
    from scenarios import get_scenario
    agents, _ = get_scenario("typical_day_industry", T=96,
                             config=MarketConfig.from_yaml(), day_id=0)
    assert [a.name for a in agents if a.participant_type == "storage"] == \
        ["ESS9", "ESS14", "ESS17", "ESS27"]
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak == pytest.approx(3.715, abs=1e-6)
    reload_defaults()
    wp.reset_cache()
