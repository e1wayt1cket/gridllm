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

#: Shed below this is the solver's residue rather than a decision to shed.
#:
#: The served/unserved split leaves ~1e-7 MWh behind at the native scale, and
#: the band-sensitivity differences inherit it. The one rung that genuinely
#: sheds -- B1 at the strict band, which has neither DER nor a battery and so
#: can only meet a tighter bound by dropping load -- sits at ~3.2e-2 MWh,
#: three orders above this line. Every tolerance below is stated against it
#: rather than against a transcribed measurement, so a rung whose shed moves
#: into the noise floor is reported rather than asserted to have changed.
SHED_TOL_MWH = 1e-5


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
    """What the tighter band costs, and at which rung it costs it.

    Reported, never tuned away: no rung's scenario parameters differ between
    the two bands, so the difference is the band's own.

    Two things are gated. The invariant: a tighter band is a strictly smaller
    feasible set, so it can never *reduce* shed at any rung. And one stress
    case: where a rung carries neither DER nor a battery, the tighter bound
    can only be met by dropping load, so the cost there is a decision rather
    than a residue. The rungs that can hold the bound with their own devices
    sit at the noise floor, and their differences are reported rather than
    asserted to be a cost -- reading one of them as evidence that 0.95 "costs
    shed" would be reading solver residue.
    """
    doc = bl.build_ladder(seed=123, bands=("primary", "strict"))
    sens = doc["band_sensitivity"]
    assert sens["B0"]["applicable"] is False   # no clearing to compare
    for rung in ("B1", "B2", "B3"):
        assert sens[rung]["applicable"] is True
        assert sens[rung]["delta_shed_95_vs_93"] >= -SHED_TOL_MWH, rung
    # B1 has no DER and no battery, so the tighter bound costs it load.
    assert sens["B1"]["delta_shed_95_vs_93"] > SHED_TOL_MWH, sens["B1"]

    print("  [diagnostic] delta_shed_95_vs_93: " + ", ".join(
        f"{r} {sens[r]['delta_shed_95_vs_93']:.3e}"
        for r in ("B1", "B2", "B3")))


@pytest.mark.slow
def test_voltage_is_the_binding_constraint_at_every_market_rung():
    """The band is a hard constraint, and the ladder reports where it pins.

    Two things are gated. First, the invariant that makes the band a constraint
    at all: no cleared rung puts a bus outside it in any period, and none
    approaches a line limit. Second, the consistency of `voltage_binding`
    itself -- it is the market clear's own active-constraint report and must
    agree with the cleared extremum rather than be asserted alongside it.

    Whether a *particular* rung pins is not gated, because it is a property of
    how stressed that rung is rather than of the mechanism. At the native scale
    most rungs sit well inside the band; the one that must pin is the rung that
    has nothing but shedding to meet the bound, and that is where the ladder is
    required to show it. The per-rung verdict is reported.
    """
    doc = bl.build_ladder(seed=123, bands=("primary", "strict"))
    pinned = {}
    for r in doc["rungs"]:
        if r["market"] is None:
            continue
        net = r["network"]
        lo, hi = r["voltage_band_bounds"]
        # The voltage variables are hard-bounded and carry no slack, so a
        # cleared day outside the band would mean the constraint was not being
        # enforced. This is a check on the mechanism, not a measurement.
        assert net["voltage_violation_bus_periods"] == 0, r["rung"]
        assert net["max_line_utilization"] < 1.0, r["rung"]
        # The flag reports the clear's active set, so it is checked against the
        # extremum the clear actually produced.
        on_edge = bool(net["vmin"] == pytest.approx(lo, abs=1e-6)
                       or net["vmax"] == pytest.approx(hi, abs=1e-6))
        assert net["voltage_binding"] is on_edge, r["rung"]
        pinned[f"{r['rung']}/{r['voltage_band']}"] = net["voltage_binding"]

    # B1 carries neither DER nor a battery, so a bound it does not naturally
    # satisfy can only be met by dropping load -- and a bound met that way is
    # met exactly. The strict band is such a bound, so B1 is where the ladder
    # has to show the constraint pinning rather than merely being respected.
    assert pinned["B1/strict"] is True, pinned

    print(f"  [diagnostic] voltage_binding by rung: {pinned}")


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
    """Facility and bidding effects are read from different comparisons.

    Installing the batteries is a change of *facilities* and its effect belongs
    to them, not to any policy; the rule's effect is read at B3's fixed
    facilities and is a policy result. The shed invariant the facility step
    has to keep is directional: adding voltage support cannot make the day
    shed more.

    It is stated as a bound rather than as a strict decrease because the
    decrease is only observable where there is shed to remove. At the native
    scale both rungs sit at the solver's floor, so a strict inequality would
    be asserting a quantity neither arm has.
    """
    doc = bl.build_ladder(seed=123, bands=("primary",))
    sep = bl.physical_vs_strategic(doc["rungs"])
    # The battery's effect is attributed to the facility, not the bidding, and
    # it cannot be a shed *increase*.
    assert sep["physical_b2_to_b3"]["delta_shed_mwh"] <= SHED_TOL_MWH
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
    # The tolerance is the noise floor of the served/unserved split. It was
    # 1e-9, which is below the floor the two arms themselves carry, so the
    # comparison was being made at a precision neither quantity has.
    assert out["shed_mwh"] <= out["shed_economic_mwh"] + SHED_TOL_MWH


@pytest.mark.slow
def test_the_residual_shed_is_the_network_and_not_the_market():
    """Where shed exists, it is the constraint's and not the objective's.

    Two arms on the same day under identical constraints: the economic
    clearing, and the same day with total shed ranked above every price.
    `shed_min` is what the constraints make unservable and the difference is
    what the market declined to buy. This is a statement about *market-clear*
    shed only.

    The forced-service diagnostic is a different object and is not an arm of
    this comparison -- it has no clearing at all, fixes shed at zero by
    construction, and answers which limit breaks first. Its answer is gated by
    `test_forcing_zero_shed_breaks_voltage_and_not_line_capacity`.

    At the native scale on the primary band both arms sit at the solver's
    floor, so the comparison holds trivially and establishes nothing. The
    configuration that genuinely sheds is therefore asserted on its own, so
    that "the market declined almost nothing" is never the output of
    differencing two zeros.
    """
    from diagnose_typical_day import min_shed_run

    out = min_shed_run()
    assert out["diagnostic_mode"] == "MIN_SHED_RUN"
    assert out["not_a_market_outcome"] is True

    # The same constraints with only the objective's ordering changed, so the
    # minimum can never serve less than the economic run did. The tolerance is
    # the noise floor of the served/unserved split, not a budget for the
    # market to decline load.
    assert out["shed_mwh"] <= out["shed_economic_mwh"] + SHED_TOL_MWH
    declined = out["shed_economic_mwh"] - out["shed_mwh"]
    assert declined <= SHED_TOL_MWH, (
        f"the market declined {declined:.6e} MWh against a total shed of "
        f"{out['shed_economic_mwh']:.6e} MWh; the attribution may have changed")

    # The stress case: a rung whose shed is a decision rather than residue. B1
    # at the strict band carries no DER and no battery, so a bound it does not
    # naturally satisfy can only be met by dropping load.
    doc = bl.build_ladder(seed=123, bands=("strict",))
    b1 = next(r for r in doc["rungs"] if r["rung"] == "B1")
    assert b1["network"]["shed"] > SHED_TOL_MWH, (
        "no rung sheds above the noise floor, so the attribution this test "
        "asserts is vacuous on this configuration")

    print(f"  [diagnostic] economic shed {out['shed_economic_mwh']:.6e} MWh, "
          f"minimum {out['shed_mwh']:.6e} MWh, declined {declined:.6e} MWh; "
          f"strict-band B1 sheds {b1['network']['shed']:.6e} MWh")


@pytest.mark.slow
def test_forcing_zero_shed_breaks_voltage_and_not_line_capacity():
    """Which limit gives first if nothing may be shed.

    The gate is the *identity* of the binding limit, not how hard it binds: a
    power flow on the day's must-serve injections, and whichever limit is
    reached first is what has to give if shed is to be zero. Voltage is that
    limit and no period overloads a line, so shed is a voltage-constraint
    result -- not an energy shortage, not a line limit, and not an artifact of
    the clearing's objective.

    How many periods violate, and how deep the sag goes, are properties of the
    load the scenario carries. They are reported here and deliberately not
    gated: pinning them would make this test a transcript of one scenario's
    stress level rather than a statement about which constraint binds.
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

    # The band is reachable only by declining service: forcing every unit
    # served drives at least one bus-period outside it, in at least one period.
    # That is the mechanism this diagnostic exists to establish.
    assert out["periods_violating_voltage"] >= 1
    lo = out["band"][0]
    assert out["worst_voltage"]["vmin"] <= lo, (
        "a period is reported outside the band, so the day's minimum must sit "
        "at or below the bound")
    # The worst bus is a property of where load and injection sit on this
    # feeder, which is why both B0 and the forced-serve day name the same one.
    assert out["worst_voltage"]["vmin_bus"] == 17

    print(f"  [diagnostic] forced-serve {out['periods_violating_voltage']}/"
          f"{len(out['rows'])} periods outside {out['band']}, "
          f"vmin {out['worst_voltage']['vmin']:.6f} at bus "
          f"{out['worst_voltage']['vmin_bus']}, worst line loading "
          f"{out['worst_line_loading']:.4f}")


# ---------------------------------------------------------------------------
# The rung's config is the scenario's config, at the rung's band
# ---------------------------------------------------------------------------

def _field_diffs(a, b, ignore=()):
    """Every dataclass field that differs between two configs."""
    import dataclasses
    out = []
    for section in ("network", "storage", "rt", "market_design"):
        sa, sb = getattr(a, section), getattr(b, section)
        for f in dataclasses.fields(sa):
            if f.name in ignore:
                continue
            va, vb = getattr(sa, f.name), getattr(sb, f.name)
            if va != vb:
                out.append(f"{section}.{f.name}: {va!r} != {vb!r}")
    return out


def test_a_rung_is_built_through_the_scenario_config_path():
    """The rung's model and the scenario's model are the same model.

    `_config` used to assemble a `MarketConfig` from the dataclass defaults,
    which agree with the file on most fields and not on the ones the scenario
    declares. The two entry points then cleared different markets, and the
    ladder's rung was not a rung of the scenario it claims to be.
    """
    from scenarios import scenario_model_config

    rung = bl._config(PRIMARY_BAND)
    scenario = scenario_model_config(bl.BASE_SCENARIO)
    assert _field_diffs(rung, scenario,
                        ignore=("v_min_pu", "v_max_pu")) == []


def test_a_rung_carries_the_facility_choices_the_scenario_declares():
    """The two fields the old construction silently dropped.

    Both are choices about this study's facilities rather than properties of
    the network, and a rung that clears without them is a different study.
    """
    rung = bl._config(PRIMARY_BAND)
    assert rung.network.load_power_factor == pytest.approx(0.8503)
    assert rung.network.inverter_smax_multiplier == pytest.approx(1.10)


def test_only_the_band_moves_between_the_two_rungs():
    """Reusing the scenario path must not disturb anything the band does not
    name."""
    primary = bl._config(PRIMARY_BAND)
    strict = bl._config(STRICT_BAND)
    assert _field_diffs(primary, strict,
                        ignore=("v_min_pu", "v_max_pu")) == []
    assert (primary.network.v_min_pu, primary.network.v_max_pu) == PRIMARY_BAND
    assert (strict.network.v_min_pu, strict.network.v_max_pu) == STRICT_BAND


def test_a_sweep_override_reaches_the_config_builder_and_not_the_rung():
    """Where an override does and does not reach, stated as the boundary.

    `_config` is public and reads the scenario cache, so a sweep built on it
    sees what the sweep wrote. `build_rung_agents` clears both caches first --
    deliberately, so a rung cannot depend on what ran before it -- and
    therefore always clears the file's model. The two behaviours are different
    on purpose, and each is asserted here rather than left to a reader.
    """
    from config_loader import reload_scenarios, set_scenario_override

    reload_scenarios()
    try:
        set_scenario_override(bl.BASE_SCENARIO,
                              "network.inverter_smax_multiplier", 1.35)
        assert bl._config(PRIMARY_BAND).network.inverter_smax_multiplier ==             pytest.approx(1.35)

        _agents, _w, built = bl.build_rung_agents("B3", PRIMARY_BAND, 123)
        assert built.network.inverter_smax_multiplier == pytest.approx(1.10),             "a rung inherited an override the previous caller left in the cache"
    finally:
        reload_scenarios()

    # With nothing written, both doors give the file's value.
    assert bl._config(PRIMARY_BAND).network.inverter_smax_multiplier ==         pytest.approx(1.10)
    _agents, _w, rung_config = bl.build_rung_agents("B3", PRIMARY_BAND, 123)
    assert rung_config.network.inverter_smax_multiplier == pytest.approx(1.10)


def test_re_applying_the_scenario_onto_a_rung_config_does_not_accumulate():
    """`get_scenario` re-applies a scenario's overrides onto the config it is
    handed, so a rung's value passes through the applier twice. Every entry is
    an absolute assignment, and this is what holds that true."""
    from scenarios import get_scenario

    config = bl._config(PRIMARY_BAND)
    before = (config.network.line_capacity_multiplier,
              config.network.inverter_smax_multiplier,
              config.network.load_power_factor)
    assert before == (pytest.approx(1.5), pytest.approx(1.10),
                      pytest.approx(0.8503))
    for _ in range(3):
        get_scenario(bl.BASE_SCENARIO, 96, config)
    after = (config.network.line_capacity_multiplier,
             config.network.inverter_smax_multiplier,
             config.network.load_power_factor)
    assert after == before, f"re-applying the scenario moved {before} to {after}"


def test_the_superseded_ladder_outputs_are_marked_as_such():
    """The recorded outputs describe the pre-S7 model, and the marker has to
    keep saying so.

    `outputs/ladder*.json` and `logs/audit_ladder_current.log` hold figures from
    before the rung config was routed through the scenario path. They were left
    on disk rather than rewritten, so the manifest beside them carries the
    reason. Its `configuration_in_force` must be what the ladder builds now --
    otherwise the note is describing a model that no longer exists either, and
    a reader would be told the current numbers are the superseded ones.
    """
    import json
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (repo / "outputs" / "ladder_superseded_manifest.json").read_text())

    assert manifest["status"] == "superseded"
    recorded = manifest["configuration_recorded"]
    current = manifest["configuration_in_force"]
    assert recorded != current, "the manifest names no actual change"

    live = bl._config(PRIMARY_BAND)
    assert current["network.load_power_factor"] == \
        pytest.approx(live.network.load_power_factor)
    assert current["network.inverter_smax_multiplier"] == \
        pytest.approx(live.network.inverter_smax_multiplier)
    assert current["built_from"] != recorded["built_from"]

    for entry in manifest["files"]:
        assert (repo / entry["path"]).exists(), entry["path"]
        # Digests are provenance, not a gate: re-running the ladder overwrites
        # these outputs, and that is the intended signal that they no longer
        # describe the recorded run. The marker above is what the test holds.
        assert len(entry["sha256"]) == 64
