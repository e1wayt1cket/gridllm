"""The B0-B3 ladder, its two voltage bands, and the shed diagnostics.

B0 is the original IEEE 33-bus at its own nominal load: a benchmark reference,
not this study's operating point. B1-B3 are the project scenario with the
distributed generation and the storage fleet added one at a time. The rungs are
not required to share a load level, and B0's numbers are not a baseline for
B1's -- they answer different questions about different networks.
"""

import inspect
import json
import pathlib

import numpy as np
import pytest

import baseline_ladder as bl
from baseline_ladder import BANDS, PRIMARY_BAND, RUNGS, STRICT_BAND

_REPO = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The bands and the rung definitions
# ---------------------------------------------------------------------------

def test_the_two_bands_are_the_operational_and_the_sensitivity_one():
    assert PRIMARY_BAND == (0.93, 1.05)
    assert STRICT_BAND == (0.95, 1.05)
    assert set(BANDS) == {"primary", "strict"}
    # They differ in the lower bound alone, which is the one this network
    # binds on: an upper bound that separated them would be separating them on
    # a constraint that never fires, and the 1.07 the primary band used to
    # carry was exactly that.
    assert PRIMARY_BAND[0] < STRICT_BAND[0]
    assert PRIMARY_BAND[1] == STRICT_BAND[1]


def test_the_rungs_add_one_thing_at_a_time():
    assert RUNGS == ("B0", "B1", "B2", "B3")
    assert bl._RUNG_FLEETS["B1"] == {"der": False, "ess": False}
    assert bl._RUNG_FLEETS["B2"] == {"der": True, "ess": False}
    assert bl._RUNG_FLEETS["B3"] == {"der": True, "ess": True}


# ---------------------------------------------------------------------------
# B0 -- the native benchmark
# ---------------------------------------------------------------------------

def test_b0_is_the_original_ieee33_at_its_own_load():
    """P and Q are the published case33bw totals, untouched."""
    rec = bl.run_b0("primary", PRIMARY_BAND)
    assert rec["rung"] == "B0"
    assert rec["reference"] == "native_benchmark"
    assert rec["convergence"] is True
    assert rec["system"]["total_p_mw"] == pytest.approx(3.715)
    assert rec["system"]["total_q_mvar"] == pytest.approx(2.3)
    assert rec["system"]["n_buses"] == 33


def test_b0_reports_its_horizon_quantities_as_not_applicable():
    """A snapshot has no horizon, so its energy figures are null, not zero.

    Reporting 0.0 MWh would read as a day that consumed nothing; the honest
    statement is that the question does not apply to a power-flow snapshot.
    """
    rec = bl.run_b0("primary", PRIMARY_BAND)
    system = rec["system"]
    assert system["metric_not_applicable"] is True
    for key in ("peak_load", "energy_load", "pv_energy", "wind_energy",
                "renewable_consumption"):
        assert system[key] is None, key


def test_b0_does_not_claim_feasibility_from_an_absent_mechanism():
    """B0 has no shedding mechanism, so its zero shed proves nothing.

    The rung is a power flow: nothing can be curtailed, so `shed = 0` is a
    statement about the model. It is reported with the mechanism named absent
    so it cannot be read as a feasibility result.
    """
    rec = bl.run_b0("primary", PRIMARY_BAND)
    assert rec["network"]["shed"] == 0.0
    assert rec["network"]["shed_mechanism"] == "absent"
    assert rec["market"] is None
    assert rec["ess"] == []


def test_b0_voltage_violates_both_bands_at_native_load():
    """The measured property the band question turns on.

    The original IEEE 33-bus does not satisfy 0.95 pu, nor 0.93, at the load it
    was published with. This is why a band is a constraint the market clears
    against rather than a property of the network, and why B0 is not retuned to
    reach either one.
    """
    rec = bl.run_b0("primary", PRIMARY_BAND)
    vmin = rec["network"]["vmin"]
    assert vmin == pytest.approx(0.9131, abs=1e-3)
    assert rec["network"]["vmin_bus"] == 17
    assert rec["network"]["buses_below_0_95"] == 21
    assert rec["network"]["buses_below_0_93"] == 14


def test_b0_uses_a_meaningful_line_rating_and_says_which():
    """`case33bw` ships `max_i_ka = 99999`: no thermal limit at all.

    Loading against that is a current over infinity. The rung reports loading
    against the project's derived ampacity, and keeps the unbounded native
    figure visible so the substitution cannot pass unnoticed.
    """
    rec = bl.run_b0("primary", PRIMARY_BAND)
    net = rec["network"]
    assert net["native_line_rating_is_unbounded"] is True
    assert net["native_line_rating_ka"] >= 9999.0
    assert net["max_line_utilization_rating_used"] == "derived_ampacity_x1.0"
    assert 0.0 < net["max_line_utilization"] < 1.0


def test_a_rung_does_not_inherit_another_callers_scenario_override():
    """The config caches are process-global, so a rung must clear them.

    A sweep that writes a per-run override leaves it in the cache for the next
    scenario built in the same process. Without the clear, a rung's parameters
    depend on what ran before it -- which is not a hypothetical: it is what
    made the calibration pass alone and fail inside the full suite.
    """
    from config_loader import set_scenario_override

    _agents, _w, config = bl.build_rung_agents("B3", PRIMARY_BAND, 123)
    baseline = config.network.inverter_smax_multiplier

    set_scenario_override("typical_day", "network.inverter_smax_multiplier", 1.80)
    _agents2, _w2, config2 = bl.build_rung_agents("B3", PRIMARY_BAND, 123)
    assert config2.network.inverter_smax_multiplier == pytest.approx(baseline)
    assert config2.network.inverter_smax_multiplier != pytest.approx(1.80)


def test_b0_is_reproducible():
    a = bl.run_b0("primary", PRIMARY_BAND)
    b = bl.run_b0("primary", PRIMARY_BAND)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# B1-B3 -- the project scenario, one rung at a time
# ---------------------------------------------------------------------------

def _one_market_rung(rung):
    return bl.build_ladder(seed=123, bands=("primary",), rungs=(rung,),
                           policy="rule")


def test_b1_has_no_der_and_no_storage():
    """The rung is defined by what is absent from it."""
    doc = _one_market_rung("B1")
    rec = doc["rungs"][0]
    assert rec["rung"] == "B1"
    assert rec["reference"] == "project_scenario"
    assert rec["policy"] is None          # no fleet to bid
    assert rec["ess"] == []
    assert rec["system"]["pv_energy"] == pytest.approx(0.0)
    assert rec["system"]["wind_energy"] == pytest.approx(0.0)
    # The day is the benchmark's own demand, spread over 96 periods by one
    # shared peak-normalized profile. The peak is the network's rated demand
    # because that is what `native_peak` pins; the energy follows from the
    # profile and is not configured anywhere.
    from grid import native_nominal_demand_mw, build_base_network
    native = native_nominal_demand_mw(
        build_base_network(bl._config(PRIMARY_BAND)))
    assert rec["system"]["peak_load"] == pytest.approx(native, abs=0.02)
    # A day below its own peak: the profile's mean is under one by construction.
    assert 0.0 < rec["system"]["energy_load"] < native * 24.0


def test_b2_adds_the_renewables_and_still_has_no_storage():
    doc = _one_market_rung("B2")
    rec = doc["rungs"][0]
    from config_loader import get_scenario_cfg
    renewable = get_scenario_cfg(bl.BASE_SCENARIO).get("renewable") or {}
    pv_mw = sum(float(v) for v in (renewable.get("pv_bus_mw") or {}).values())
    wind_mw = sum(float(v) for v in (renewable.get("wind_bus_mw") or {}).values())

    assert rec["ess"] == []
    # A day's output is at most nameplate times 24 h, and is a real fraction
    # of it rather than a trace: the profiles are the same shapes as ever, so
    # rescaling the capacity must rescale the energy with them.
    assert 0.1 * pv_mw * 24.0 < rec["system"]["pv_energy"] < pv_mw * 24.0
    assert 0.1 * wind_mw * 24.0 < rec["system"]["wind_energy"] < wind_mw * 24.0
    # Renewable output is consumed, not merely available.
    assert rec["system"]["renewable_consumption"] > 0.0


def test_b3_carries_the_fleet_and_reports_its_economics():
    """The fleet is where the scenario says, and it does real work.

    The bus list is read from the declaration rather than restated, so a
    re-siting is a scenario change and not a test failure. What is asserted
    about activity is the property the siting exists to deliver: at least
    three of the four units move energy. A unit that sits out the day is
    named in `inactive_buses` rather than hidden inside a fleet total.
    """
    from config_loader import get_default

    doc = _one_market_rung("B3")
    rec = doc["rungs"][0]
    declared = sorted(int(b) for b in get_default("independent_storage.buses"))
    assert len(rec["ess"]) == len(declared)
    assert sorted(u["bus"] for u in rec["ess"]) == declared

    active = [u for u in rec["ess"]
              if u["charge_energy"] > 0.0 and u["discharge_energy"] > 0.0]
    assert len(active) >= 3, (
        f"only {len(active)} of {len(rec['ess'])} units have both a charge "
        f"and a discharge opportunity; inactive: "
        f"{rec['ess_fleet']['inactive_buses']}")

    for unit in active:
        assert unit["cycles"] > 0.0
        # The settled profit is the ledger's own identity, not a separate
        # number that could drift from its parts.
        assert unit["profit"] == pytest.approx(
            unit["revenue"] - unit["purchase_cost"] - unit["degradation_cost"])


def test_the_fleet_block_is_the_sum_of_the_per_unit_rows():
    """A fleet figure that disagreed with its parts would hide a quiet unit."""
    doc = _one_market_rung("B3")
    rec = doc["rungs"][0]
    fleet = rec["ess_fleet"]
    assert fleet["n_units"] == len(rec["ess"])
    assert fleet["charge_energy_mwh"] == pytest.approx(
        sum(u["charge_energy"] for u in rec["ess"]))
    assert fleet["discharge_energy_mwh"] == pytest.approx(
        sum(u["discharge_energy"] for u in rec["ess"]))
    assert fleet["profit"] == pytest.approx(sum(u["profit"] for u in rec["ess"]))
    # The same tolerance the fleet block uses. A unit whose energy is solver
    # noise rather than activity must not count as active, or a fleet total
    # would report four working units when one of them never traded.
    tol = inspect.signature(bl._ess_fleet_block).parameters["activity_tol"].default
    assert fleet["active_units"] == sum(
        1 for u in rec["ess"]
        if u["charge_energy"] > tol or u["discharge_energy"] > tol)


def test_b3_reports_its_own_counterfactual():
    """A profit only means something against what doing nothing would earn."""
    doc = _one_market_rung("B3")
    rec = doc["rungs"][0]
    assert rec["pi_ai"] is not None
    assert rec["pi_baseline"] is not None
    assert rec["delta_pi"] == pytest.approx(rec["pi_ai"] - rec["pi_baseline"])


# ---------------------------------------------------------------------------
# The ladder as a whole
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_whole_ladder_is_byte_for_byte_reproducible(tmp_path):
    """Same configuration, same seed, same scenario -> the same file."""
    a = bl.build_ladder(seed=123, bands=("primary", "strict"))
    b = bl.build_ladder(seed=123, bands=("primary", "strict"))
    pa = bl.write_ladder(a, str(tmp_path / "a.json"))
    pb = bl.write_ladder(b, str(tmp_path / "b.json"))
    assert pathlib.Path(pa).read_bytes() == pathlib.Path(pb).read_bytes()


@pytest.mark.slow
def test_every_rung_ran_at_both_bands(tmp_path):
    doc = bl.build_ladder(seed=123, bands=("primary", "strict"))
    got = {(r["rung"], r["voltage_band"]) for r in doc["rungs"]}
    assert got == {(r, b) for r in RUNGS for b in ("primary", "strict")}
    assert doc["voltage_band"] == {"primary": list(PRIMARY_BAND),
                                   "strict": list(STRICT_BAND)}
    # Each record carries the band it was actually cleared at, so a rung cannot
    # be filed under one band and computed at the other.
    for r in doc["rungs"]:
        assert r["voltage_band_bounds"] == list(BANDS[r["voltage_band"]])


@pytest.mark.slow
def test_the_strict_band_costs_shed_and_is_reported_as_such():
    """The measured answer to what 0.95 costs, which is the point of running it.

    Reported, never tuned away: no rung's scenario parameters differ between
    the two bands, so the difference is the band's own.
    """
    doc = bl.build_ladder(seed=123, bands=("primary", "strict"))
    sens = doc["band_sensitivity"]
    assert sens["B0"]["applicable"] is False   # no clearing to compare
    for rung in ("B1", "B2", "B3"):
        assert sens[rung]["applicable"] is True
        assert sens[rung]["delta_shed_95_vs_93"] > 0.0, rung


@pytest.mark.slow
def test_voltage_is_the_binding_constraint_at_every_market_rung():
    """Every clearing rung parks a voltage on its own lower bound.

    This is what makes shed a network result rather than an energy one, and it
    holds at both bands.
    """
    doc = bl.build_ladder(seed=123, bands=("primary", "strict"))
    for r in doc["rungs"]:
        if r["market"] is None:
            continue
        assert r["network"]["voltage_binding"] is True, r["rung"]
        lo = r["voltage_band_bounds"][0]
        assert r["network"]["vmin"] == pytest.approx(lo, abs=1e-6)


def _fake_records():
    def row(rung, shed, curt, lmp, obj, pi_ai=None, pi_base=None):
        rec = {"rung": rung, "voltage_band": "primary",
               "network": {"shed": shed, "curtailment": curt},
               "market": {"lmp_mean": lmp, "clearing_objective": obj}}
        if pi_ai is not None:
            rec.update({"pi_ai": pi_ai, "pi_baseline": pi_base,
                        "delta_pi": pi_ai - pi_base,
                        "delta_pi_percent": (pi_ai - pi_base)
                        / (abs(pi_base) + 1e-6)})
        return rec
    return [row("B1", 30.0, 0.0, 700.0, 10000.0),
            row("B2", 2.0, 1.6, 800.0, 12000.0),
            row("B3", 0.4, 1.6, 720.0, 19000.0, pi_ai=613.0, pi_base=664.0)]


def test_the_two_improvements_are_reported_separately():
    """Installing the batteries and bidding better are different claims.

    Summing them would credit the policy with the batteries, so they are
    returned under separate keys with their scope named.
    """
    sep = bl.physical_vs_strategic(_fake_records())
    physical = sep["physical_b2_to_b3"]
    strategic = sep["strategic_at_b3"]

    assert physical["attributable_to_bidding"] is False
    assert strategic["attributable_to_bidding"] is True
    assert "B2" in physical["comparison"] and "B3" in physical["comparison"]

    # The physical block compares the two facility rungs only.
    assert physical["delta_shed_mwh"] == pytest.approx(0.4 - 2.0)
    assert physical["shed_b2_mwh"] == pytest.approx(2.0)
    # The strategic block compares policies at one facility rung.
    assert strategic["pi_baseline_truthful"] == pytest.approx(664.0)
    assert strategic["pi_rule"] == pytest.approx(613.0)
    assert strategic["delta_pi_rule_vs_truthful"] == pytest.approx(-51.0)


def test_the_physical_block_carries_no_policy_figure():
    """A facility comparison has no policy in it to report."""
    physical = bl.physical_vs_strategic(_fake_records())["physical_b2_to_b3"]
    assert not any("pi" in k for k in physical), physical


@pytest.mark.slow
def test_the_real_ladder_separates_the_two_effects(tmp_path):
    doc = bl.build_ladder(seed=123, bands=("primary",))
    sep = bl.physical_vs_strategic(doc["rungs"])
    # The battery's effect is a real reduction in shed, and it is attributed
    # to the facility, not the bidding.
    assert sep["physical_b2_to_b3"]["delta_shed_mwh"] < 0
    assert sep["physical_b2_to_b3"]["attributable_to_bidding"] is False
    # The deterministic rule's effect is a policy result at fixed facilities.
    assert sep["strategic_at_b3"]["attributable_to_bidding"] is True
    assert sep["strategic_at_b3"]["pi_baseline_truthful"] is not None


# ---------------------------------------------------------------------------
# The shed diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_forced_serve_diagnostic_leaves_the_network_cache_alone():
    """`build_base_network` hands out one shared object to every caller.

    The diagnostic rewrites each bus's load and adds a generator per renewable
    bus, so doing that on the cached network leaves those edits in place for
    every later run in the same process. Measured, that is not subtle: it made
    an unrelated unit-invariance test fail inside the full suite while passing
    on its own.
    """
    from grid import build_base_network
    from models import MarketConfig
    from diagnose_typical_day import no_shed_voltage_diagnostic

    config = MarketConfig(opf_mode="socp", verbose=False)
    before = build_base_network(config)
    load_before = float(before.load.p_mw.sum())
    sgen_before = len(before.sgen)

    no_shed_voltage_diagnostic()

    after = build_base_network(config)
    assert float(after.load.p_mw.sum()) == pytest.approx(load_before)
    assert len(after.sgen) == sgen_before


@pytest.mark.slow
def test_min_shed_run_serves_at_least_as_much_as_the_market_did():
    """Minimising shed cannot serve less than the economic run.

    The constraints are identical and only the objective's ordering changes, so
    a smaller minimum would mean the economic solve was leaving load unserved
    that the physics allowed.
    """
    from diagnose_typical_day import min_shed_run

    out = min_shed_run()
    assert out["diagnostic_mode"] == "MIN_SHED_RUN"
    assert out["not_a_market_outcome"] is True
    assert out["shed_mwh"] <= out["shed_economic_mwh"] + 1e-9


@pytest.mark.slow
def test_the_residual_shed_is_the_network_and_not_the_market():
    """Almost all of the shed is unservable, not declined.

    Measured: the economic run sheds 0.1709 MWh and the minimum under identical
    physics is 0.1286 MWh, so the market declines about 0.04 MWh of a 179 MWh
    day. The rest is what the network cannot deliver.
    """
    from diagnose_typical_day import min_shed_run

    out = min_shed_run()
    assert out["shed_economic_mwh"] == pytest.approx(0.1709, abs=1e-3)
    declined = out["shed_economic_mwh"] - out["shed_mwh"]
    assert declined < 0.05, (
        f"the market declined {declined:.4f} MWh, more than the measured "
        f"0.042; the shed attribution may have changed")


@pytest.mark.slow
def test_forcing_zero_shed_breaks_voltage_and_not_line_capacity():
    """Which limit gives first if nothing may be shed.

    Every period violates the voltage band and no period overloads a line, so
    shed is a voltage-constraint result: not an energy shortage, not a line
    limit, and not an artifact of clearing.
    """
    from diagnose_typical_day import no_shed_voltage_diagnostic

    out = no_shed_voltage_diagnostic()
    assert out["diagnostic_mode"] == "NO_SHED_VOLTAGE_DIAGNOSTIC"
    assert out["not_a_market_outcome"] is True
    assert out["shed"] == 0.0 and out["shed_forced_zero"] is True
    assert out["converged"] is True
    assert out["binding_constraint"] == "voltage"
    assert out["periods_overloading_a_line"] == 0
    assert out["worst_line_loading"] < 1.0
    assert out["periods_violating_voltage"] == 96
    # The worst sag is far below either band, at the same bus B0 identified.
    assert out["worst_voltage"]["vmin"] < 0.75
    assert out["worst_voltage"]["vmin_bus"] == 17
