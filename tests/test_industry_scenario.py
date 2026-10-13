# tests/test_industry_scenario.py
"""Contracts of the industry-shape scenario.

`typical_day_industry` is `typical_day` with exactly one thing changed: the
intraday *shape* of each bus's load. The level is deliberately untouched -- the
scenario still pins the fleet's coincident peak to the network's own nominal
demand -- so the two days are directly comparable and any difference between
them belongs to the shape.

That is what most of these tests defend. A change that let the industry curves
carry their own magnitude, that left a bus unassigned and quietly defaulted it,
or that counted the workbook's PV output twice would still produce a
plausible-looking day. It would simply no longer be the day the scenario says
it is, and nothing in a cleared result would show it.

The per-bus shape is checked through the *un-noised* path
(`_industry_load_shapes`), because the realized profiles carry one shared
multiplicative error factor per period that does not cancel under
max-normalisation. The realized day is checked through proportionality instead,
which the shared factor does preserve exactly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import yaml

from config_loader import get_scenario_cfg, reload_defaults
from grid import build_base_network, native_nominal_demand_mw
from industry_load import N_SLOTS, load_shapes_csv
from models import MarketConfig
from scenarios import _industry_load_shapes, get_scenario

SCENARIO = "typical_day_industry"
SHAPES_FILE = "data/processed/industry_typical_profiles.csv"


def _workbook_map():
    """The bus→industry assignment exactly as the scenario file declares it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((root / "config" / "scenarios.yaml").read_text(
        encoding="utf-8"))
    cfg = raw["scenarios"][SCENARIO]["industry_map"]
    return {int(k): str(v) for k, v in cfg["buses"].items()}, cfg


@pytest.fixture(scope="module")
def artifact():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / SHAPES_FILE
    if not path.exists():
        pytest.skip(f"{SHAPES_FILE} not built; run src/industry_load.py")
    return load_shapes_csv(str(path))


@pytest.fixture(scope="module")
def net():
    reload_defaults()
    return build_base_network(MarketConfig.from_yaml())


@pytest.fixture(scope="module")
def built():
    """The scenario's agents, built once."""
    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    agents, wholesale = get_scenario(SCENARIO, T=N_SLOTS)
    return agents, wholesale


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_the_scenario_is_registered_and_declares_its_own_scale():
    cfg = get_scenario_cfg(SCENARIO)
    assert cfg["load_scale_mode"] == "native_peak"
    assert cfg["load_shape"] == "industry"
    assert cfg["scenario_scale_source"] == "ieee33_native_industry_shape"
    assert "industry_map" in cfg


def test_the_mapping_is_declared_as_a_simulation_mapping():
    """The file must not read as a claim about real geographic connection."""
    _, imap = _workbook_map()
    assert imap.get("simulation_mapping") is True
    assert imap.get("shapes_file") == SHAPES_FILE


def test_every_load_bus_is_assigned_exactly_once(net, artifact):
    assignment, _ = _workbook_map()
    load_buses = {int(b) for b in net.load.bus}
    assert set(assignment) == load_buses
    assert len(assignment) == len(load_buses)


def test_the_mapped_industries_all_have_a_shape(artifact):
    assignment, imap = _workbook_map()
    for industry in set(assignment.values()):
        assert industry in artifact.shapes, industry
        assert industry not in set(imap.get("excluded_industries", [])), industry


def test_only_reliable_industries_shape_the_load_buses(artifact):
    """Exploratory industries must not silently become part of the benchmark."""
    assignment, _ = _workbook_map()
    for industry in set(assignment.values()):
        assert artifact.shapes[industry].reliable, (
            f"{industry} is below the reliability threshold and must not "
            f"shape a load bus in the benchmark scenario")


def test_the_generator_is_excluded_by_name(artifact):
    assignment, imap = _workbook_map()
    assert "分布式光伏" in imap["excluded_industries"]
    assert "分布式光伏" not in set(assignment.values())


# ---------------------------------------------------------------------------
# The assignment is enforced, not assumed
# ---------------------------------------------------------------------------

def _map_with(net, artifact, **overrides):
    assignment, imap = _workbook_map()
    cfg = dict(imap)
    cfg["buses"] = dict(assignment)
    cfg.update(overrides)
    return cfg


def test_a_bus_left_unassigned_raises(net, artifact):
    cfg = _map_with(net, artifact)
    cfg["buses"] = {b: i for b, i in cfg["buses"].items() if b != 17}
    with pytest.raises(ValueError, match="unassigned"):
        _industry_load_shapes(cfg, net, N_SLOTS)


def test_a_bus_carrying_no_load_raises(net, artifact):
    """Bus 0 is the slack; naming it must fail rather than be ignored."""
    cfg = _map_with(net, artifact)
    cfg["buses"] = dict(cfg["buses"])
    cfg["buses"][0] = "钢铁"
    with pytest.raises(ValueError, match="no load"):
        _industry_load_shapes(cfg, net, N_SLOTS)


def test_an_excluded_industry_raises(net, artifact):
    cfg = _map_with(net, artifact)
    cfg["buses"] = dict(cfg["buses"])
    cfg["buses"][17] = "分布式光伏"
    with pytest.raises(ValueError, match="excluded"):
        _industry_load_shapes(cfg, net, N_SLOTS)


def test_an_industry_with_no_shape_raises(net, artifact):
    cfg = _map_with(net, artifact)
    cfg["buses"] = dict(cfg["buses"])
    cfg["buses"][17] = "不存在的行业"
    with pytest.raises(ValueError, match="no shape"):
        _industry_load_shapes(cfg, net, N_SLOTS)


def test_a_missing_artifact_names_the_command_that_builds_it(net):
    cfg = _map_with(net, None)
    cfg["shapes_file"] = "data/processed/definitely_absent.csv"
    cfg["buses"] = {}
    with pytest.raises(FileNotFoundError, match="industry_load.py"):
        _industry_load_shapes(cfg, net, N_SLOTS)


# ---------------------------------------------------------------------------
# The level contract
# ---------------------------------------------------------------------------

def test_the_declared_scale_reaches_the_agents(built, net):
    agents, _ = built
    loads = [a for a in agents if a.load_type != "storage"]
    total = np.sum([np.asarray(a.load_real) for a in loads], axis=0)
    assert float(total.max()) == pytest.approx(native_nominal_demand_mw(net),
                                               abs=1e-6)
    assert float(total.max()) == pytest.approx(3.715, abs=1e-6)


def test_the_industry_day_peaks_where_the_native_day_peaks(built):
    """Same declared scale, so the peak is the same number in both days."""
    reload_defaults()
    native_agents, _ = get_scenario("typical_day", T=N_SLOTS)
    reload_defaults()

    def peak(agents):
        loads = [a for a in agents if a.load_type != "storage"]
        return float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())

    assert peak(built[0]) == pytest.approx(peak(native_agents), abs=1e-9)


def test_the_shape_does_not_carry_its_own_magnitude(artifact):
    """A shape scaled by any constant must produce the same day.

    This is the property that makes the shape normalisation load-bearing: the
    artifact supplies form, never level. If a change let the artifact's scale
    through, the fleet's peak would move with it.
    """
    assignment, imap = _workbook_map()
    net = build_base_network(MarketConfig.from_yaml())

    base = _industry_load_shapes(imap, net, N_SLOTS)
    scaled = {b: arr * 137.0 for b, arr in base.items()}

    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    from grid import create_agents_from_network
    from scenarios import scenario_scale
    scale = scenario_scale(get_scenario_cfg(SCENARIO))

    def peak(shapes):
        agents = create_agents_from_network(net, N_SLOTS, with_wind=True,
                                            scale=scale, load_shapes=shapes)
        loads = [a for a in agents if a.load_type != "storage"]
        return float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())

    assert peak(base) == pytest.approx(peak(scaled), abs=1e-6)


# ---------------------------------------------------------------------------
# The shapes themselves
# ---------------------------------------------------------------------------

def test_every_bus_carries_its_assigned_industry_shape(net, artifact):
    assignment, imap = _workbook_map()
    shapes = _industry_load_shapes(imap, net, N_SLOTS)
    assert set(shapes) == set(assignment)
    for bus, arr in shapes.items():
        want = artifact.shape_of(assignment[bus])
        want = want / want.max()
        assert arr.shape == (N_SLOTS,)
        np.testing.assert_allclose(arr, want, atol=1e-12, err_msg=f"bus {bus}")


def test_every_shape_is_96_finite_non_negative_slots(built):
    agents, _ = built
    for a in agents:
        if a.load_type == "storage":
            continue
        for arr in (a.load_real, a.load_forecast):
            arr = np.asarray(arr)
            assert arr.shape == (N_SLOTS,), a.name
            assert np.all(np.isfinite(arr)), a.name
            assert np.all(arr >= 0.0), a.name


def test_buses_sharing_an_industry_carry_the_same_shape(built):
    """The shared per-period error factor is exactly what makes this checkable."""
    assignment, _ = _workbook_map()
    agents, _ = built
    loads = sorted([a for a in agents if a.load_type != "storage"],
                   key=lambda a: a.bus)

    groups = {}
    for a in loads:
        groups.setdefault(assignment[int(a.bus)], []).append(a)

    for industry, members in groups.items():
        ref = np.asarray(members[0].load_real)
        for other in members[1:]:
            ratio = np.asarray(other.load_real) / ref
            assert float(ratio.std()) < 1e-9 * abs(float(ratio.mean())), (
                f"buses {members[0].bus} and {other.bus} are both {industry} "
                f"but do not carry the same shape")


def test_different_industries_do_not_collapse_to_one_shape(built):
    assignment, _ = _workbook_map()
    agents, _ = built
    by_bus = {int(a.bus): a for a in agents if a.load_type != "storage"}
    steel = next(b for b, i in assignment.items() if i == "钢铁")
    dc = next(b for b, i in assignment.items() if i == "数据中心")
    ratio = np.asarray(by_bus[dc].load_real) / np.asarray(by_bus[steel].load_real)
    assert float(ratio.std()) > 1e-6 * abs(float(ratio.mean()))


# ---------------------------------------------------------------------------
# Facilities are not double-counted
# ---------------------------------------------------------------------------

def test_renewables_are_unchanged_by_the_industry_data(built):
    """The workbook's PV is a generator and must not re-enter as a device."""
    agents, _ = built
    assert sum(float(a.pv_capacity or 0.0) for a in agents) == pytest.approx(0.6)
    assert sum(float(a.wind_capacity or 0.0) for a in agents) == pytest.approx(1.0)


def test_the_fleet_is_unchanged(built):
    agents, _ = built
    assert [a.name for a in agents if a.participant_type == "storage"] == \
        ["ESS9", "ESS14", "ESS17", "ESS27"]


# ---------------------------------------------------------------------------
# Price selection
# ---------------------------------------------------------------------------

def test_the_declared_curve_prices_the_day(built):
    """`wholesale_curve_index` is a price scenario index, not a date."""
    from dianyicha_price import SHEETS, read_curves
    _, wholesale = built
    index = int(get_scenario_cfg(SCENARIO)["wholesale_curve_index"])
    curve = read_curves()[SHEETS[index]]
    np.testing.assert_allclose(wholesale, curve.array())
    # ...and the curve is a single day that is spatially averaged, read as one
    # day by project ruling rather than confirmed from outside. The scenario is
    # priced by a SPATIAL mean, so no bus is represented by it.
    assert curve.is_single_day is True
    assert curve.temporal_statistic == "single_day"
    assert curve.spatial_statistic == "spatial_mean"
    assert curve.externally_confirmed is False


def test_the_declared_source_is_the_real_price_reader():
    """Built here rather than read from the shared fixture: `source` is written
    into a process-global cache on every build, so another test's rebuild would
    otherwise decide this assertion."""
    import wholesale_price as wp
    reload_defaults()
    wp.reset_cache()
    get_scenario(SCENARIO, T=N_SLOTS)
    assert wp.current_source() == "dianyicha"
    reload_defaults()
    wp.reset_cache()


# ---------------------------------------------------------------------------
# Other scenarios are untouched
# ---------------------------------------------------------------------------

def test_other_shapes_ignore_load_shapes_entirely():
    """Passing per-bus shapes to a `native` scenario must change nothing."""
    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    net = build_base_network(MarketConfig.from_yaml())
    from grid import create_agents_from_network
    from scenarios import scenario_scale

    scale = scenario_scale(get_scenario_cfg("typical_day"))
    stray = {b: np.linspace(0.1, 2.0, N_SLOTS) for b in range(1, 33)}

    reload_defaults()
    plain = create_agents_from_network(net, N_SLOTS, with_wind=True, scale=scale)
    reload_defaults()
    with_stray = create_agents_from_network(net, N_SLOTS, with_wind=True,
                                            scale=scale, load_shapes=stray)

    assert [a.name for a in plain] == [a.name for a in with_stray]
    for a, b in zip(plain, with_stray):
        np.testing.assert_allclose(a.load_real, b.load_real)


def test_an_unknown_load_shape_still_raises():
    from grid import _make_load_profile
    with pytest.raises(ValueError, match="unknown load shape"):
        _make_load_profile(np.arange(N_SLOTS), "residential", {}, shape="nonsense")


def test_the_native_scenario_still_carries_one_shared_shape():
    """`typical_day`'s defining property, re-checked after the change."""
    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    agents, _ = get_scenario("typical_day", T=N_SLOTS)
    loads = sorted([a for a in agents if a.load_type != "storage"],
                   key=lambda a: a.bus)
    ref = np.asarray(loads[0].load_real)
    for other in loads[1:]:
        ratio = np.asarray(other.load_real) / ref
        assert float(ratio.std()) < 1e-9 * abs(float(ratio.mean()))


def test_legacy_scenarios_keep_their_own_scale():
    reload_defaults()
    import wholesale_price as wp
    wp.reset_cache()
    for name in ("baseline", "high_re", "peak_load"):
        agents, _ = get_scenario(name, T=N_SLOTS)
        loads = [a for a in agents if a.load_type != "storage"]
        peak = float(np.sum([np.asarray(a.load_real) for a in loads], axis=0).max())
        assert peak > 9.0, name
        reload_defaults()
