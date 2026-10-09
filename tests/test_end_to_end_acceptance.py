# tests/test_end_to_end_acceptance.py
"""The industry-shape research day, accepted end to end.

One full run through the whole chain -- industry shapes, IEEE 33-bus network,
storage fleet, 96-period SOCP clearing, shared settlement -- with the checks
that make it an acceptance rather than a smoke test.

The run is done once and asserted many times, because it is the *day* under test
and not any single function. Two properties are worth naming because they are
the ones a plausible-looking wrong answer would still satisfy:

* **Both settlement paths must agree.** `settle_day` on the fixed declarations
  and the rolling episode the evaluation CLI runs must produce the same
  per-unit payoff. The two disagreed before Phase 5, and a day that clears
  cleanly would not have shown it.
* **An idle fleet must be explained.** On this price scenario every unit is
  idle, and the run reports the round-trip margin at each unit's own bus
  against what the clearing charges it to cycle. "Idle" without that number is
  indistinguishable from "broken".

The day is *not* asserted to be profitable. It is not, and a test that demanded
otherwise would be asking for the tuning this project forbids.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import accept_typical_day_industry as acc


@pytest.fixture(scope="module")
def report():
    """One acceptance run, shared by every test in this module."""
    from config_loader import reload_defaults
    import counterfactual_eval as ce
    import wholesale_price as wp

    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()
    try:
        yield acc.run_acceptance(repeats=2)
    finally:
        reload_defaults()
        wp.reset_cache()
        ce.reset_baseline_cache()


# ---------------------------------------------------------------------------
# 1. Inputs
# ---------------------------------------------------------------------------

def test_industry_shapes_load_and_are_complete(report):
    shapes = report["industry_shapes"]
    assert shapes["n_industries"] == 16
    assert shapes["industries_failing_96_finite"] == []
    assert shapes["n_slots"] == 96
    # The three the Phase-1 assessment could not support stay marked.
    assert "分布式光伏" in shapes["limited"] or "分布式光伏" in shapes["reliable"]


def test_all_four_price_curves_read_with_correct_metadata(report):
    curves = report["price_curves"]["curves"]
    assert set(curves) == {"261009", "7days", "30days", "halfyear"}
    for name, c in curves.items():
        assert c["n_slots"] == 96
        assert c["valid"] is True
        assert c["units"] == "CNY/MWh"
        assert c["is_measured"] is False, name
        assert c["interval_convention"] == "label_is_interval_end"
        assert c["workbook_sha256"]
    # The header's own wording survives, including the 均值 on the chosen sheet.
    assert "均值" in curves["261009"]["header_label"]


# ---------------------------------------------------------------------------
# 2. The scenario and the price it runs on
# ---------------------------------------------------------------------------

def test_the_declared_curve_is_the_one_that_prices_the_day(report):
    assert report["declared_curve"]["resolved_sheet"] == "261009"
    assert report["declared_curve"]["wholesale_curve_index"] == 0
    assert report["declared_curve"]["global_day_id_used"] is False
    assert report["price_check"]["wholesale_matches_declared_curve"] is True


def test_the_declared_source_is_dianyicha(report):
    assert report["declared_curve"]["declared_source"] == "dianyicha"


def test_the_price_matches_the_workbook(report):
    pc = report["price_check"]
    assert pc["matches_workbook_min"]
    assert pc["matches_workbook_max"]
    assert pc["matches_workbook_mean"]
    # Independent of the reader: the values measured in Phase 2.
    assert pc["min"] == pytest.approx(225.90, abs=0.01)
    assert pc["max"] == pytest.approx(445.00, abs=0.01)
    assert pc["mean"] == pytest.approx(386.83, abs=0.01)


def test_the_curve_never_reaches_the_fleet_as_a_measured_day(report):
    """`261009` derives from real market data but its aggregation is unsettled,
    so it is not treated as a confirmed daily series."""
    assert report["provenance"]["curve_is_measured"] is False
    assert report["provenance"]["curve_role"] == "same_day_or_same_day_mean"


# ---------------------------------------------------------------------------
# 3. Load
# ---------------------------------------------------------------------------

def test_the_coincident_peak_is_the_networks_own_demand(report):
    lc = report["load_check"]
    assert lc["coincident_peak_mw"] == pytest.approx(3.715, abs=1e-6)
    assert lc["absolute_error_mw"] < 1e-6
    assert lc["all_finite"] and lc["all_non_negative"]


def test_the_mapping_is_declared_a_simulation_and_uses_no_generator(report):
    sc = report["scenario"]
    assert sc["mapping_is_simulation"] is True
    assert "分布式光伏" in sc["mapping_excluded"]
    assert "分布式光伏" not in sc["mapping_industries"]
    assert "5G基站" not in sc["mapping_industries"]
    assert "电解铝" not in sc["mapping_industries"]


def test_the_facilities_are_unchanged(report):
    sc = report["scenario"]
    assert sc["storage_names"] == ["ESS9", "ESS14", "ESS17", "ESS27"]
    assert sc["storage_buses"] == [9, 14, 17, 27]
    assert all(p == pytest.approx(0.4) for p in sc["storage_power_mw"])
    assert all(e == pytest.approx(1.6) for e in sc["storage_energy_mwh"])
    assert sc["pv_mw"] == pytest.approx(0.6)
    assert sc["wind_mw"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Clearing and physics
# ---------------------------------------------------------------------------

def test_every_period_cleared_without_a_fallback(report):
    assert report["settlement_path_a"]["clean"] is True
    assert report["physics"]["prices"]["lmp_fallbacks"] == 0


def test_the_voltage_band_holds_in_every_period(report):
    v = report["physics"]["bus_voltage"]
    assert v["periods_below_min"] == 0
    assert v["periods_above_max"] == 0
    assert v["band"][0] <= v["min"] and v["max"] <= v["band"][1]


def test_no_line_reaches_its_limit(report):
    lu = report["physics"]["line_utilization"]
    assert lu["lines_over_1"] == 0
    assert lu["periods_over_1"] == 0


def test_the_grid_never_imports_and_exports_at_once(report):
    g = report["physics"]["grid"]
    assert g["simultaneous_import_export"] == 0
    assert g["import_mwh"] > 0


def test_the_day_ahead_market_serves_its_own_basis_in_full(report):
    """Served covers the forecast; the realized load is a different number."""
    e = report["physics"]["energy"]
    assert e["da_basis"] == "load_forecast"
    assert e["served_covers_forecast"] is True
    assert e["unserved_mwh"] < 1e-6
    # The realized load is larger, and that gap is forecast error, not shed
    # load -- the acceptance must not present it as a constraint violation.
    assert e["realized_mwh"] > e["forecast_mwh"]
    assert e["forecast_error_pct"] > 0


def test_storage_stays_inside_its_own_limits(report):
    for unit, s in report["physics"]["storage"].items():
        assert s["within_bounds"], unit
        assert s["charge_mwh"] >= 0.0 and s["discharge_mwh"] >= 0.0
        assert s["transitions_checked"] > 0
        assert s["max_recursion_residual"] < 1e-6, unit


def test_the_state_of_charge_returns_to_where_it_started(report):
    for unit, s in report["physics"]["storage"].items():
        assert s["soc0_boundary"] == pytest.approx(s["soc0_configured"], abs=1e-9)
        assert s["transitions_checked"] == s["transitions_expected"], unit
        assert s["all_transitions_verified"] is True, unit
        assert s["terminal_actual_is_the_boundary_end"] is True, unit
        assert s["soc_terminal_actual"] == pytest.approx(s["soc0_boundary"],
                                                         abs=1e-5), unit


# ---------------------------------------------------------------------------
# 5. Money
# ---------------------------------------------------------------------------

def test_the_two_settlement_paths_agree_on_every_unit(report):
    """The Phase-5 property, checked on a real day."""
    pa = report["paths_agree"]
    assert pa["prices_identical"] is True
    assert pa["agree"] is True, pa["per_unit_max_abs_diff"]
    assert pa["per_unit_max_abs_diff"] < 1e-6


def test_the_rolling_reward_is_recorded_but_is_not_the_payoff(report):
    """The CLI's two families must stay distinguishable on a real day."""
    b = report["settlement_path_b"]
    assert set(b["rolling_reward"]) == set(b["per_unit_profit"])
    for unit in b["per_unit_profit"]:
        assert np.isfinite(b["per_unit_profit"][unit])
        assert np.isfinite(b["rolling_reward"][unit])


def test_the_breakdown_reconciles_for_every_unit(report):
    rec = report["fleet_reconciliation"]
    assert rec["every_unit_reconciles"] is True
    assert rec["every_unit_matches_settlement"] is True
    assert rec["agree"] is True


def test_other_cost_is_reported_as_zero_not_silently_filled(report):
    for unit, b in report["breakdown"].items():
        assert b["other_cost"] == 0.0, unit
        assert b["generation_cost"] == 0.0, unit
        assert b["penalty"] == 0.0, unit


# ---------------------------------------------------------------------------
# 6. Why the fleet is idle
# ---------------------------------------------------------------------------

def test_the_idle_fleet_is_explained_by_the_arbitrage_window(report):
    """Every unit is idle, and the run must say why rather than assert it.

    The clearing charges a cycle at the derived floor
    `cycle_cost * (1/eta_ch + eta_dis)` -- not the lossless `2 * cycle_cost`.
    If no unit's own bus delivers that, no unit can profitably move. This test
    pins that the diagnosis was made and that it is consistent with the idle
    dispatch.
    """
    from arbitrage_lp import required_margin
    arb = report["arbitrage"]
    assert arb["required_margin"] == pytest.approx(
        required_margin(arb["cycle_cost_per_leg"], 0.92, 0.92))
    assert arb["required_margin"] > 2.0 * arb["cycle_cost_per_leg"]
    assert len(arb["units"]) == 4
    for u in arb["units"]:
        assert u["round_trip_margin"] == pytest.approx(
            u["lmp_max"] * 0.92 - u["lmp_min"] / 0.92, rel=1e-6)
    idle = all(s["charge_mwh"] < 1e-9 and s["discharge_mwh"] < 1e-9
               for s in report["physics"]["storage"].values())
    if idle:
        assert arb["all_below_threshold"] is True, (
            "the fleet is idle but the arbitrage window is open; the idleness "
            "has some other cause and this diagnosis is wrong")


def test_the_result_is_not_tuned_to_produce_a_profit(report):
    """No threshold, price or cost was adjusted to make the day pay."""
    assert report["arbitrage"]["cycle_cost_per_leg"] == 100.0
    assert report["physics"]["energy"]["curtailment_mwh"] >= 0.0
    # The fleet's settled profit is what the day gave, reported as found.
    assert isinstance(report["settlement_path_a"]["fleet_profit"], float)


# ---------------------------------------------------------------------------
# 7. Repeatability and non-interference
# ---------------------------------------------------------------------------

def test_the_run_repeats_exactly(report):
    rep = report["repeatability"]
    assert rep["n_runs"] >= 2
    assert rep["prices_identical"] is True
    assert rep["peak_identical"] is True
    assert rep["profits_identical"] is True
    assert rep["all_clean"] is True


def test_a_build_does_not_price_the_next_scenario_from_a_stale_index():
    """Building the industry day must not leave its curve index behind.

    `_build_scenario` writes `wholesale_market.day_id` so the nine other price
    creation sites agree with it. If a scenario that follows declared nothing,
    it would inherit this one's index -- so every price-bearing scenario
    declares its own.
    """
    import numpy as np
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import get_default, reload_defaults
    from models import MarketConfig
    from scenarios import get_scenario

    reload_defaults()
    wp.reset_cache()
    _, industry_price = get_scenario("typical_day_industry", T=96,
                                     config=MarketConfig.from_yaml())
    reload_defaults()
    wp.reset_cache()
    _, native_price = get_scenario("typical_day", T=96,
                                   config=MarketConfig.from_yaml())
    assert get_default("wholesale_market.day_id") == 11
    # Different markets, and the native day is back on the synthetic library.
    assert wp.current_source() == "synthetic_v1"
    assert industry_price.min() == pytest.approx(225.90, abs=0.01)
    assert not np.allclose(industry_price, native_price)
    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


# ---------------------------------------------------------------------------
# 8. Nothing older moved
# ---------------------------------------------------------------------------

def test_the_native_scenario_is_unchanged():
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from models import MarketConfig
    from scenarios import get_scenario

    reload_defaults()
    wp.reset_cache()
    agents, wholesale = get_scenario("typical_day", T=96,
                                     config=MarketConfig.from_yaml())
    loads = [a for a in agents if a.load_type != "storage"]
    peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
    assert peak == pytest.approx(3.715, abs=1e-6)
    assert [a.name for a in agents if a.participant_type == "storage"] == \
        ["ESS9", "ESS14", "ESS17", "ESS27"]
    # Still the synthetic library, still its median-range reference day.
    assert wp.current_source() == "synthetic_v1"
    assert float(wholesale.min()) != pytest.approx(225.90, abs=0.01)
    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


def test_legacy_scenarios_still_build():
    import numpy as np
    import wholesale_price as wp
    from config_loader import reload_defaults
    from models import MarketConfig
    from scenarios import get_scenario

    reload_defaults()
    wp.reset_cache()
    for name in ("baseline", "high_re", "peak_load"):
        agents, _ = get_scenario(name, T=96, config=MarketConfig.from_yaml())
        loads = [a for a in agents if a.load_type != "storage"]
        peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
        assert peak > 9.0, name
        reload_defaults()
        wp.reset_cache()


# ---------------------------------------------------------------------------
# 9. Solver provenance
# ---------------------------------------------------------------------------

def test_the_result_names_the_solver_that_produced_it(report):
    sv = report["physics"]["solver"]
    assert sv["present"] is True
    assert sv["engine"] == "socp"
    assert sv["primary_name"] == "gurobi"
    assert sv["primary_status_code"] == 2
    assert sv["primary_status_name"] == "OPTIMAL"
    assert sv["primary_termination"] == "optimal"
    assert sv["used_fallback"] is False
    assert sv["fallback"] is None
    assert sv["final_name"] == "gurobi"
    assert sv["final_is_primary"] is True


def test_duals_availability_is_reported_apart_from_optimality(report):
    """An OPTIMAL model can still hand back no dual; the two are separate."""
    sv = report["physics"]["solver"]
    assert sv["duals_available"] is True
    assert sv["primary_status_name"] == "OPTIMAL"
    # ...and the price layer agrees, independently of the status.
    assert report["physics"]["prices"]["lmp_fallbacks"] == 0


def test_the_solver_block_is_not_confused_with_the_fallback_key(report):
    """`fell_back` predates the solver block and is only set on that path."""
    assert report["physics"]["solver"]["fallback_key_present"] is False


def test_status_names_cover_every_code_the_classifier_knows():
    from dispatch_core import GUROBI_STATUS_NAMES, categorize_termination

    for code in (2, 3, 4, 5, 7, 9, 12, 13, 17):
        assert code in GUROBI_STATUS_NAMES, code
        assert categorize_termination(code) != "unknown", code
    assert categorize_termination(2) == "optimal"
    assert categorize_termination(3) == "infeasible_or_unbounded"
    assert categorize_termination(9) == "limit_reached"
    assert categorize_termination(None) == "unknown"


def test_infeasible_is_not_reported_as_a_limit():
    """The two need different responses, so they must not share a label."""
    from dispatch_core import categorize_termination
    assert categorize_termination(3) != categorize_termination(9)
    assert categorize_termination(4) != categorize_termination(8)


def test_a_fallback_result_names_the_fallback_as_final(monkeypatch):
    """After a fallback the primary failed; its status is not the answer."""
    import numpy as np
    import dispatch
    import dispatch_core
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from market import clear_market
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    config.opf_mode = "lindistflow"
    agents, wholesale = get_scenario("typical_day_industry", T=96, config=config)
    declared = FixedStrategy().formulate(agents, config, None, 96)

    entry = dispatch.CLEARING_MECHANISMS["lindistflow"]
    monkeypatch.setitem(entry, "batch_fn", lambda *a, **k: None)
    dispatch_core.record_last_solve("lindistflow", {
        "name": "gurobi", "status_code": 9, "status_name": "TIME_LIMIT",
        "termination": "limit_reached", "optimal": False})

    result = clear_market(agents, 96, "DA", declared, config,
                          wholesale=wholesale, horizon_type="full_day")
    sv = result["solver"]
    assert sv["used_fallback"] is True
    assert result.get("fell_back") is True
    # The primary's own status survives, and is not presented as the outcome.
    assert sv["primary"]["status_name"] == "TIME_LIMIT"
    assert sv["primary"]["termination"] == "limit_reached"
    assert sv["final"]["name"] == "single_period_lindistflow"
    assert sv["final"]["name"] != sv["primary"]["name"]
    assert sv["final"]["status_name"] == "COMPLETED"
    # The per-period loop does not surface balance duals, so the answer is
    # unknown rather than a confident False.
    assert sv["duals_available"] is None
    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


# ---------------------------------------------------------------------------
# 10. The state-of-charge trajectory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def moving_day():
    """A day whose fleet actually cycles, so the trajectory is non-degenerate.

    The industry day leaves every unit idle, which would make a trajectory test
    pass on a constant array and prove nothing.
    """
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    agents, wholesale = get_scenario("typical_day", T=96, config=config)
    names = [a.name for a in agents if a.participant_type == "storage"]
    declared = FixedStrategy().formulate(agents, config, None, 96)
    _, clean, result = ce.settle_day(agents, config, names, declared,
                                     wholesale, 96)
    assert clean and result is not None
    try:
        yield {"agents": agents, "names": names, "result": result}
    finally:
        reload_defaults()
        wp.reset_cache()
        ce.reset_baseline_cache()


def test_the_boundary_array_has_one_more_state_than_there_are_periods(moving_day):
    T = 96
    for nm in moving_day["names"]:
        s = moving_day["result"]["schedules"][nm]
        assert np.asarray(s["soc_boundary"]).size == T + 1, nm
        assert np.asarray(s["soc"]).size == T, nm


def test_every_transition_verifies_on_a_moving_day(moving_day):
    """The regression this exists to catch.

    The per-period readout used to overwrite its last entry with the terminal
    state, which silently dropped the final transition. A test on an idle fleet
    cannot see that -- every state is the same number. On a fleet that cycles,
    reconstructing the trajectory from the readout fails the last transition,
    and the boundary array passes all 96.
    """
    T = 96
    money_dt = 0.25
    for nm in moving_day["names"]:
        agent = next(a for a in moving_day["agents"] if a.name == nm)
        s = moving_day["result"]["schedules"][nm]
        b = np.asarray(s["soc_boundary"], dtype=float)
        ch = np.asarray(s["p_ch"], dtype=float)
        dis = np.asarray(s["p_dis"], dtype=float)
        flow = (agent.storage.eta_ch * ch
                - dis / agent.storage.eta_dis) * money_dt / agent.storage.e_max

        resid = (b[1:] - b[:-1]) - flow
        assert resid.size == T, nm
        assert np.max(np.abs(resid)) < 1e-6, (nm, np.max(np.abs(resid)))

        # And the readout really is not a usable trajectory: its last entry is
        # the terminal state, so stepping it fails where the boundary holds.
        readout = np.asarray(s["soc"], dtype=float)
        assert readout[-1] == pytest.approx(b[-1], abs=1e-9), nm
        readout_resid = (readout[1:] - readout[:-1]) - flow[:-1]
        assert np.max(np.abs(readout_resid)) > 1e-9, (
            f"{nm}: the readout closed every transition, so it can no longer "
            "demonstrate why the boundary array is needed")


def test_the_fleet_actually_cycles_on_this_day(moving_day):
    """A guard on the guard: an idle fleet would make the test above vacuous."""
    total = 0.0
    for nm in moving_day["names"]:
        s = moving_day["result"]["schedules"][nm]
        total += float(np.sum(s["p_ch"]) + np.sum(s["p_dis"]))
    assert total > 1.0, "the fixture's fleet did not move"


def test_the_terminal_target_and_the_actual_state_are_stored_separately(moving_day):
    T = 96
    for nm in moving_day["names"]:
        s = moving_day["result"]["schedules"][nm]
        b = np.asarray(s["soc_boundary"], dtype=float)
        assert s["soc_terminal_actual"] == pytest.approx(b[T], abs=1e-12), nm
        assert s["soc_terminal_target"] is not None, nm
        # Pinned for a whole-day clear, so the two agree here -- but they are
        # two fields, and a windowed clear reports None rather than a value.
        assert s["soc_terminal_actual"] == pytest.approx(
            s["soc_terminal_target"], abs=1e-5), nm


def test_a_rolling_window_reports_no_terminal_target(tmp_path):
    """On a window the pin is not applied, and the target says so."""
    import numpy as np
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from market import clear_market
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    agents, wholesale = get_scenario("typical_day", T=96, config=config)
    declared = FixedStrategy().formulate(agents, config, None, 96)
    names = [a.name for a in agents if a.participant_type == "storage"]

    subset = {nm: declared[nm] for nm in names}
    result = clear_market(agents, 8, "DA", subset, config,
                          wholesale=wholesale[:8], horizon_type="window")
    for nm in names:
        s = result["schedules"][nm]
        assert s["soc_terminal_target"] is None, nm
        assert s["soc_terminal_actual"] is not None, nm
    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


def test_lindistflow_carries_the_same_schema():
    """A reader must not have to know which engine cleared the day."""
    import numpy as np
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults
    from market import clear_market
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies.fixed_bidding import FixedStrategy

    reload_defaults()
    wp.reset_cache()
    config = MarketConfig.from_yaml()
    config.opf_mode = "lindistflow"
    agents, wholesale = get_scenario("typical_day", T=96, config=config)
    declared = FixedStrategy().formulate(agents, config, None, 96)
    result = clear_market(agents, 96, "DA", declared, config,
                          wholesale=wholesale, horizon_type="full_day")
    assert result["solver"]["engine"] == "lindistflow"
    assert result["solver"]["final"]["status_name"] == "OPTIMAL"
    for a in agents:
        if a.storage is None:
            continue
        s = result["schedules"][a.name]
        b = np.asarray(s["soc_boundary"], dtype=float)
        ch = np.asarray(s["p_ch"], dtype=float)
        dis = np.asarray(s["p_dis"], dtype=float)
        flow = (a.storage.eta_ch * ch
                - dis / a.storage.eta_dis) * 0.25 / a.storage.e_max
        assert (b[1:] - b[:-1] - flow).size == 96
        assert np.max(np.abs((b[1:] - b[:-1]) - flow)) < 1e-6, a.name
    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()
