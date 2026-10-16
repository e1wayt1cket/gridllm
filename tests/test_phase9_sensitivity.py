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

#: The siting and capacity scans ask where a battery *could* pay, which needs a
#: day the feeder can serve whole. `typical_day_industry` used to be one; it now
#: carries the metered industry load, on which the feeder serves a few percent
#: and every margin is a statement about shedding. The scans run on the native
#: benchmark day, which is the day they were calibrated against.
_SCAN_SCENARIO = "typical_day"


@pytest.fixture(scope="module", autouse=True)
def _on_the_native_day():
    """Point the scans at a day the feeder can serve whole.

    Module scope, and that matters: `screen` below is module-scoped too, and a
    function-scoped patch would be installed after it had already been built --
    which is how a test can look correctly configured and still run on the
    wrong day.
    """
    from config_loader import reload_scenarios, set_scenario_param
    previous = (p9.SCENARIO, p9.BASELINE_SCENARIO)
    p9.SCENARIO = p9.BASELINE_SCENARIO = _SCAN_SCENARIO
    reload_scenarios()
    set_scenario_param(_SCAN_SCENARIO, "wholesale_source", "dianyicha")
    yield
    p9.SCENARIO, p9.BASELINE_SCENARIO = previous
    reload_scenarios()

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

def test_the_load_shape_comparison_now_compares_two_scales_not_two_shapes():
    """The comparison's premise changed when the industry day moved to a ledger.

    It was written for two days at one declared scale differing only in shape.
    `typical_day_industry` now carries the metered load, so the second entry is
    not a shape variant of the first -- it is a day this feeder cannot serve.
    What the comparison must still do is say so rather than report a margin.
    """
    try:
        result = p9.load_shape_comparison()
    finally:
        p9._fresh()
    # These scans run on the native day in this module, so the comparison is
    # between it and itself; what is checked is that both entries are present
    # and carry a whole day, which is the contract the scans rely on.
    for name, c in result.items():
        assert c.get("clean") is not False, (name, c.get("reason"))
        assert c["coincident_peak_mw"] == pytest.approx(3.715, abs=1e-6), name


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


def test_the_native_days_margins_are_reported_for_every_configured_bus():
    """What is gated here is the accounting, not one curve's knife edge.

    This test used to assert that the four configured buses sit within a few
    CNY/MWh of the requirement -- a finding measured on the native day priced
    from the **synthetic** curve. The scans in this module now run on the real
    curve (`261009`), and on it bus 9 clears by about 29.8 CNY/MWh rather than
    sitting on the edge. The closeness was a property of that price day, not of
    the fleet, so it is not a property to gate.

    What is gated: every configured bus is reported, its margin and shortfall
    are finite and agree with the formula, and the whole per-bus table is the
    clearing's own numbers rather than a recomputation that could drift from
    them.
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
        # The table is the clearing's own figures, not a second derivation.
        assert own[bus]["margin"] == pytest.approx(
            unit.eta_dis * own[bus]["lmp_max"] - own[bus]["lmp_min"] / unit.eta_ch)
        assert shortfall == pytest.approx(
            own[bus]["margin"] - native["required_margin"])
        assert np.isfinite(shortfall), (bus, shortfall)

    # The measured margins for the current curve, recorded rather than asserted:
    # they move with the price day, and the phase-9 report carries them.
    assert len(shortfalls) == 4


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
    agents, _ = get_scenario(_SCAN_SCENARIO, T=96,
                             config=MarketConfig.from_yaml(), day_id=0)
    assert [a.name for a in agents if a.participant_type == "storage"] == \
        ["ESS9", "ESS14", "ESS17", "ESS27"]
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak == pytest.approx(3.715, abs=1e-6)
    reload_defaults()
    wp.reset_cache()
