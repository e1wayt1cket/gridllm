"""Every scenario's scale is declared, and none of it is inherited.

Four quantities used to be one number in `defaults.yaml`: the benchmark's
physical demand, a study's operating scale, the renewable capacity, and what
the temporal profile was allowed to change. They are separate now, and these
tests hold the separation:

  * `defaults.yaml` carries the IEEE33 benchmark and no study scale at all.
  * each scenario declares its own scale, shape, power factor and capacity;
  * a scenario that declares none is an error rather than an inheritance;
  * the legacy scenarios are unchanged to the bit.
"""

import copy

import numpy as np
import pytest

import money
from config_loader import get_scenario_cfg, load_defaults
from grid import build_base_network, native_nominal_demand_mw
from models import MarketConfig
from scenarios import get_scenario, list_scenarios, scenario_scale, \
    typical_day_config

RENEWABLE_KEYS = ("pv_bus_mw", "wind_bus_mw")


def _agents(scenario: str = "typical_day"):
    config = typical_day_config() if scenario == "typical_day" \
        else MarketConfig.from_yaml(opf_mode="socp")
    agents, _wholesale = get_scenario(scenario, money.PERIODS_PER_DAY, config)
    return agents, config


def _declared(scenario: str):
    cfg = get_scenario_cfg(scenario)
    renewable = cfg.get("renewable") or {}
    return {
        "pv": sum(float(v) for v in (renewable.get("pv_bus_mw") or {}).values()),
        "wind": sum(float(v) for v in
                    (renewable.get("wind_bus_mw") or {}).values()),
    }


# ---------------------------------------------------------------------------
# defaults.yaml carries no study scale
# ---------------------------------------------------------------------------

def test_the_defaults_carry_no_load_level_and_no_renewable_capacity():
    """A benchmark default is not a study scale.

    One number in `defaults.yaml` used to mean the benchmark's demand, the
    study's operating point and what the renewables were sized against at once,
    and no reader could tell which was in force.
    """
    defaults = load_defaults()
    assert "target_peak_mw" not in defaults["profiles"]["load"]
    for kind in ("residential", "industrial"):
        block = defaults["prosumers"][kind]
        assert "pv_capacity_factor" not in block, kind
        assert "wind_capacity_factor" not in block, kind
        for bus_ov in (block.get("bus_overrides") or {}).values():
            assert "pv_capacity_scale" not in bus_ov
            assert "wind_capacity_scale" not in bus_ov


def test_a_scenario_that_declares_no_scale_is_an_error():
    """Silence must not mean "inherit whatever the last scenario used"."""
    from scenarios import scenario_scale as ss
    for missing in ("load_scale_mode", "load_shape", "renewable"):
        cfg = {"load_scale_mode": "native_peak", "load_shape": "native",
               "renewable": {}}
        cfg.pop(missing)
        with pytest.raises(ValueError, match=missing):
            ss(cfg)


# ---------------------------------------------------------------------------
# The native research scenario
# ---------------------------------------------------------------------------

def test_the_native_day_peaks_at_the_networks_own_demand():
    """`max_t P_total(t)` is the IEEE33 nominal demand, not a study scale.

    The profile is peak-normalized, so the fleet's busiest period carries
    exactly the demand the benchmark was published with.
    """
    agents, _ = _agents()
    native = native_nominal_demand_mw(build_base_network(typical_day_config()))
    assert native == pytest.approx(3.715)

    total = np.sum([a.load_real for a in agents], axis=0)
    assert float(total.max()) == pytest.approx(native, abs=1e-9)


def test_the_native_day_uses_the_benchmark_nodal_distribution():
    """`P_i(t) = P_i_base * lambda_t` with one shared lambda_t.

    Every bus therefore sits at the same fraction of its own demand at every
    period, which is what makes the fleet's peak the sum of the bus demands
    rather than something the profile decides.
    """
    agents, _ = _agents()
    # Buses that carry load. The storage fleet's agents hold a zero profile by
    # construction, so a shape normalized by their own peak is undefined.
    load_buses = [a for a in agents
                  if float(np.max(np.asarray(a.load_real, dtype=float))) > 0.0]
    assert len(load_buses) == 32          # case33bw loads buses 1-32

    ratios = np.stack([np.asarray(a.load_real, dtype=float)
                       for a in load_buses])
    shapes = ratios / ratios.max(axis=1, keepdims=True)
    assert np.allclose(shapes, shapes[0], atol=1e-9), \
        "the buses do not share one temporal profile"

    # And each bus's peak is the network's own rated load at that bus.
    net = build_base_network(typical_day_config())
    native_by_bus = dict(zip((int(b) for b in net.load.bus), net.load.p_mw))
    spec = scenario_scale(get_scenario_cfg("typical_day"))
    assert spec["shape"] == "native"
    for a in load_buses:
        assert float(np.max(np.asarray(a.load_real, dtype=float))) \
            == pytest.approx(native_by_bus[a.bus], rel=1e-9)


def test_the_native_day_loads_are_reported_not_configured():
    """Peak, mean, energy and load factor come from the profile."""
    agents, _ = _agents()
    total = np.sum([a.load_real for a in agents], axis=0)
    peak = float(total.max())
    mean = float(total.mean())
    energy = float(total.sum()) * money.DT_HOURS

    assert peak == pytest.approx(3.715, abs=1e-9)
    assert energy == pytest.approx(mean * 24.0, rel=1e-9)
    assert 0.0 < mean / peak < 1.0, "a day cannot sit flat on its own peak"


def test_the_native_day_reactive_load_is_the_benchmarks():
    """Q scales with the same lambda_t, at the power factor that makes it 2.3 MVAr."""
    agents, config = _agents()
    total = np.sum([a.load_real for a in agents], axis=0)
    peak_p = float(total.max())

    pf = config.network.load_power_factor
    assert pf == pytest.approx(0.8503, abs=1e-4)
    tan_phi = float(np.tan(np.arccos(pf)))
    assert peak_p * tan_phi == pytest.approx(2.3, abs=0.01)


def test_the_native_day_carries_the_declared_renewable_capacity():
    agents, _ = _agents()
    declared = _declared("typical_day")
    assert declared == {"pv": 0.6, "wind": 1.0}
    assert sum(float(a.pv_capacity or 0.0)
               for a in agents) == pytest.approx(0.6)
    assert sum(float(a.wind_capacity or 0.0)
               for a in agents) == pytest.approx(1.0)


def test_renewable_capacity_is_independent_of_the_load_scale():
    """The capacity is declared, not derived from the local load.

    A capacity computed from `base_load` moves whenever the load does, which
    makes it a property of the load rather than a choice the study made and can
    defend.
    """
    agents, _ = _agents()
    net = build_base_network(typical_day_config())
    native_by_bus = dict(zip((int(b) for b in net.load.bus), net.load.p_mw))

    # The ratio a bus's array bears to its own load is not a constant, which is
    # what a capacity derived from `base_load * factor * scale` would produce.
    # Measured here: bus 21 carries 0.3 MW against 0.09 MW of load (3.3x) and
    # bus 23 carries 0.3 MW against 0.42 MW (0.7x). Two buses at the same
    # declared capacity against loads nearly five times apart cannot have been
    # sized from those loads.
    ratio_21 = 0.3 / native_by_bus[21]
    ratio_23 = 0.3 / native_by_bus[23]
    assert ratio_21 / ratio_23 > 3.0, (ratio_21, ratio_23)
    for a in agents:
        if a.pv_capacity:
            assert a.bus in (21, 23)


# ---------------------------------------------------------------------------
# The legacy scenarios are untouched
# ---------------------------------------------------------------------------

#: Measured before the revision, on the tree that still carried `target_peak_mw`.
_LEGACY = {
    "baseline": (9.9685041539, 7.4278965106, 178.2695162552),
    "high_re": (9.9685041539, 7.4278965106, 178.2695162552),
    "no_congestion": (9.9685041539, 7.4278965106, 178.2695162552),
    "tight_bottleneck": (9.9685041539, 7.4278965106, 178.2695162552),
    "re_ramp_drop": (9.9685041539, 7.4278965106, 178.2695162552),
    "re_ramp_surge": (9.9685041539, 7.4278965106, 178.2695162552),
    "peak_load": (14.9527562308, 11.1418447658, 267.4042743799),
    "congestion": (11.9631566736, 8.7293067391, 209.5033617379),
}


@pytest.mark.parametrize("scenario", sorted(_LEGACY))
def test_a_legacy_scenario_keeps_its_physical_scale(scenario):
    """Its peak, mean and daily energy are the ones it had before.

    Read from a measured table rather than recomputed, because the claim is
    that nothing about these scenarios moved -- and a recomputed expectation
    would move with them.
    """
    agents, _ = _agents(scenario)
    total = np.sum([a.load_real for a in agents], axis=0)
    peak, mean, energy = _LEGACY[scenario]
    assert float(total.max()) == pytest.approx(peak, rel=1e-9)
    assert float(total.mean()) == pytest.approx(mean, rel=1e-9)
    assert float(total.sum()) * money.DT_HOURS == pytest.approx(energy, rel=1e-9)


@pytest.mark.parametrize("scenario", sorted(_LEGACY))
def test_a_legacy_scenario_keeps_its_power_factor_and_capacity(scenario):
    agents, config = _agents(scenario)
    assert config.network.load_power_factor == pytest.approx(0.9)
    assert sum(float(a.pv_capacity or 0.0) for a in agents) \
        == pytest.approx(3.5)
    assert sum(float(a.wind_capacity or 0.0) for a in agents) \
        == pytest.approx(1.5)


def test_legacy_scenarios_say_where_their_scale_came_from():
    """The field that stops a 10 MW study scale being read as the benchmark's."""
    for scenario in _LEGACY:
        assert get_scenario_cfg(scenario)["scenario_scale_source"] \
            == "legacy_explicit"
    assert get_scenario_cfg("typical_day")["scenario_scale_source"] \
        == "ieee33_native"


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------

def test_the_inventory_shows_no_implicit_inheritance():
    """Every scenario declares its own scale, and the declaration arrives.

    The realized columns are checked against the declared ones, so a
    declaration that never reaches the agents shows up as a mismatch rather
    than as a scenario quietly running at another's scale.
    """
    import baseline_ladder as bl

    rows = bl.scenario_inventory()
    assert {r["scenario"] for r in rows} == set(list_scenarios())
    for r in rows:
        assert r["scenario_scale_source"] != "MISSING", r["scenario"]
        assert r["load_scale_mode"] in ("native_peak", "explicit_peak")
        assert r["load_shape"] in ("native", "diversified")
        assert "error" not in r, (r["scenario"], r.get("error"))
        assert r["realized_pv_mw"] == pytest.approx(
            sum(r["pv_bus_mw_declared"].values()), abs=1e-6), r["scenario"]
        assert r["realized_wind_mw"] == pytest.approx(
            sum(r["wind_bus_mw_declared"].values()), abs=1e-6), r["scenario"]


def test_only_the_research_scenario_is_at_native_scale():
    """One scenario runs at the benchmark's demand, and it is this one."""
    import baseline_ladder as bl

    rows = {r["scenario"]: r for r in bl.scenario_inventory()}
    native = [s for s, r in rows.items() if r["load_scale_mode"] == "native_peak"]
    assert native == ["typical_day"]
    assert rows["typical_day"]["realized_peak_mw"] == pytest.approx(3.715,
                                                                    abs=1e-6)
    for scenario, r in rows.items():
        if scenario in _LEGACY:
            assert r["realized_peak_mw"] > 9.0, scenario


def test_the_scale_spec_is_read_whole_from_the_scenario():
    """`scenario_scale` reports the declaration and nothing else."""
    spec = scenario_scale(get_scenario_cfg("typical_day"))
    assert spec == {"mode": "native_peak", "peak_load_mw": None,
                    "shape": "native",
                    "pv_bus_mw": {21: 0.3, 23: 0.3},
                    "wind_bus_mw": {30: 1.0}}
