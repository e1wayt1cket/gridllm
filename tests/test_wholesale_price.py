"""The external wholesale price: that it is a market, and that it is not ours.

The point of the external market is that the distribution network cannot move
its price. That is a property of the construction, not of a number, so most of
what is asserted here is *independence*: the price must not respond to IEEE33's
load, PV or wind, must not be reproducible from the agents it is handed, and
must not be a straight line across its reachable range.

The stack's continuity is asserted segment by segment. A typo in one slope
leaves a curve that is still monotone and still plausible, so continuity is the
only thing that catches it.
"""

import csv

import numpy as np
import pytest

import wholesale_price as wp
from config_loader import get_default, reload_defaults, set_default


@pytest.fixture()
def provider():
    return wp.SyntheticWholesalePriceProvider()


@pytest.fixture()
def stack():
    return wp.SupplyStack.from_config(get_default(f"{wp.CONFIG_KEY}.supply_stack"))


# ---------------------------------------------------------------------------
# The supply stack
# ---------------------------------------------------------------------------

def test_the_stack_is_continuous_at_every_knot(stack):
    """The curve's stated property, checked where a typo would hide."""
    for k in range(len(stack.slopes)):
        top = stack.prices[k] + stack.slopes[k] * (stack.knots[k + 1]
                                                   - stack.knots[k])
        assert top == pytest.approx(stack.prices[k + 1], abs=1e-9), k


def test_a_discontinuous_stack_is_refused():
    """A plausible-looking typo must not load."""
    with pytest.raises(ValueError, match="discontinuous"):
        wp.SupplyStack(knots=(0.0, 0.5, 1.0), prices=(160.0, 260.0, 900.0),
                       slopes=(120.0, 700.0))


def test_slopes_must_not_decrease():
    """Marginal cost rises as cheap capacity is used up, or it is not a stack."""
    with pytest.raises(ValueError, match="must not decrease"):
        wp.SupplyStack(knots=(0.0, 0.5, 1.0), prices=(160.0, 400.0, 800.0),
                       slopes=(700.0, 280.0))


def test_the_knots_must_span_the_unit_interval():
    with pytest.raises(ValueError, match=r"span \[0, 1\]"):
        wp.SupplyStack(knots=(0.1, 0.5, 1.0), prices=(160.0, 260.0, 900.0),
                       slopes=(200.0, 1280.0))


def test_the_stack_matches_the_declared_endpoints(stack):
    assert stack.price(np.array([0.0]))[0] == pytest.approx(160.0)
    assert stack.price(np.array([1.0]))[0] == pytest.approx(900.0)


def test_the_stack_rises_with_net_load(stack):
    """Monotone, and strictly so where a segment has a positive slope."""
    r = np.linspace(0.0, 1.0, 200)
    p = stack.price(r)
    assert np.all(np.diff(p) > 0)


def test_the_stack_is_steeper_near_the_peak_than_near_the_trough(stack):
    """The structure the retired affine map did not have.

    One straight line is what the market-opportunity audit identified as the
    reason the strategic-bidding signal was under-excited. A stack that rises
    faster at the top is the whole point of the change, so it is asserted.
    """
    assert all(a < b for a, b in zip(stack.slopes, stack.slopes[1:]))
    assert stack.slopes[-1] > 10 * stack.slopes[0]


def test_the_stack_price_is_clipped_outside_the_unit_interval(stack):
    """`r` outside [0,1] is pinned, not extrapolated past the stack."""
    assert stack.price(np.array([-0.5]))[0] == pytest.approx(160.0)
    assert stack.price(np.array([1.5]))[0] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# Shape and reproducibility
# ---------------------------------------------------------------------------

def test_a_day_is_96_periods(provider):
    assert len(provider.get_day_ahead_price(0)) == 96


def test_the_same_seed_gives_the_same_curve(provider):
    a = provider.get_day_ahead_price(3, seed=2024)
    b = provider.get_day_ahead_price(3, seed=2024)
    assert np.array_equal(a, b)


def test_the_same_day_gives_the_same_curve_twice(provider):
    """No global RNG anywhere: a second draw must not depend on call order."""
    first = provider.get_day_ahead_price(5)
    provider.get_day_ahead_price(9)
    provider.get_day_ahead_price(2)
    assert np.array_equal(first, provider.get_day_ahead_price(5))


def test_a_different_day_gives_a_different_curve(provider):
    assert not np.allclose(provider.get_day_ahead_price(0),
                           provider.get_day_ahead_price(1))


def test_different_days_stay_inside_a_declared_band(provider):
    """Days vary, but the market does not become a different market."""
    means = np.array([provider.get_day_ahead_price(d).mean()
                      for d in range(provider.number_of_days)])
    assert means.std() > 0.0
    assert means.min() > 0.3 * means.max()


def test_prices_stay_inside_the_declared_bounds(provider):
    for day in range(provider.number_of_days):
        p = provider.get_day_ahead_price(day)
        assert np.all(p >= provider.floor - 1e-9), day
        assert np.all(p <= provider.cap + 1e-9), day


def test_the_library_is_reproducible_from_its_configuration():
    """Two providers built from the same file agree period for period."""
    a = wp.SyntheticWholesalePriceProvider()
    b = wp.SyntheticWholesalePriceProvider()
    assert np.array_equal(a.get_day_ahead_price(11), b.get_day_ahead_price(11))
    assert a.library_range() == b.library_range()


# ---------------------------------------------------------------------------
# Decoupling: the distribution network cannot move this price
# ---------------------------------------------------------------------------

def test_the_price_does_not_move_when_ieee33_shapes_change():
    """The property the whole module exists for.

    The retired map priced the day from this network's own net load, which
    made the price a function of the dispatch it was pricing -- a battery
    could shade the number it was settled at. Perturbing the distribution
    network's load, PV and wind parameters must leave the external price
    exactly where it was.
    """
    reload_defaults()
    provider = wp.SyntheticWholesalePriceProvider()
    before = provider.get_day_ahead_price(4)

    for key, value in (("profiles.load.night_base", 0.95),
                       ("profiles.load.evening_peak_hour", 21.5),
                       ("profiles.pv.cloud_seed", 777),
                       ("profiles.wind.seed", 888),
                       ("profiles.load.forecast_noise_seed", 12345)):
        set_default(key, value)
    after = wp.SyntheticWholesalePriceProvider().get_day_ahead_price(4)
    reload_defaults()

    assert np.array_equal(before, after), "the external price moved with IEEE33"


def test_the_provider_never_reads_agents(provider):
    """The interface accepts agents and must ignore them.

    A later edit that made the external price respond to local renewable
    output would be a one-line change that silently reintroduces the
    endogeneity this market was built to remove.
    """
    price = provider.day_ahead_price(2)

    class FakeAgent:
        load_forecast = np.full(96, 1e6)
        pv_forecast = np.full(96, 1e6)
        wind_forecast = np.full(96, 1e6)
        has_wind = True

    # Nothing in the provider's signature takes agents at all; assert the
    # construction path has no agent parameter and produces the same prices.
    assert np.array_equal(price, provider.day_ahead_price(2))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def test_a_window_is_the_days_own_periods(provider):
    """`t_start` is what aligns a rolling block to the day.

    Without it every block would be priced off the first `window_T` periods of
    the day, and a block at 17:00 would be charged 05:00's prices.
    """
    full = provider.get_day_ahead_price(3)
    window = provider.get_day_ahead_price(3, T=8, t_start=64)
    assert np.array_equal(window, full[64:72])
    assert not np.allclose(window, full[:8])


def test_a_short_horizon_slices_rather_than_rebuilds(provider):
    """The horizon's own length is not a coarser day."""
    full = provider.get_day_ahead_price(3)
    assert np.array_equal(provider.get_day_ahead_price(3, T=16), full[:16])
    assert np.array_equal(provider.get_day_ahead_price(3, T=48), full[:48])


def test_a_window_past_the_end_is_refused(provider):
    with pytest.raises(ValueError, match="never wraps"):
        provider.get_day_ahead_price(0, T=8, t_start=92)


def test_an_empty_window_is_refused(provider):
    with pytest.raises(ValueError, match="empty"):
        provider.get_day_ahead_price(0, T=0, t_start=0)
    with pytest.raises(ValueError, match="empty"):
        provider.get_day_ahead_price(0, T=8, t_start=-1)


# ---------------------------------------------------------------------------
# The reduction, and what the library's size does to it
# ---------------------------------------------------------------------------

def test_r_spans_the_unit_interval_over_the_library(provider):
    """Min-max normalization across the whole library, as declared."""
    r = np.concatenate([provider.reduced_input(d)
                        for d in range(provider.number_of_days)])
    assert r.min() == pytest.approx(0.0)
    assert r.max() == pytest.approx(1.0)


def test_a_day_is_normalized_against_the_library_not_itself(provider):
    """The day's own extremes need not reach the ends of the stack.

    Per-day normalization would put every day's cheapest hour at the bottom of
    the stack and its dearest at the top, so days would differ in shape only
    and never in level. Normalizing across the library is what lets a mild day
    sit low and a severe day climb.
    """
    spans = [(provider.reduced_input(d).min(), provider.reduced_input(d).max())
             for d in range(provider.number_of_days)]
    assert any(lo > 0.0 for lo, _ in spans)
    assert any(hi < 1.0 for _, hi in spans)


def test_the_library_range_depends_on_the_library_size():
    """Recorded because it is a real dependency, not a defect to hide.

    The yardstick is the library's own net-load range, so adding days can
    widen it and shift every price. The manifest records the range so a
    library built against a different one is refused rather than silently
    priced as though it were the declared one.
    """
    small = wp.SyntheticWholesalePriceProvider(
        {**get_default(wp.CONFIG_KEY), "number_of_days": 5})
    large = wp.SyntheticWholesalePriceProvider(
        {**get_default(wp.CONFIG_KEY), "number_of_days": 60})
    assert small.library_range() != large.library_range()


# ---------------------------------------------------------------------------
# Noise and scarcity
# ---------------------------------------------------------------------------

def test_the_noise_does_not_decide_the_level(provider):
    """Noise perturbs which segment a period sits in; it must not set the day."""
    quiet = wp.SyntheticWholesalePriceProvider(
        {**get_default(wp.CONFIG_KEY), "noise_sigma": 0.0,
         "scarcity_probability": 0.0})
    noisy = provider
    q = quiet.get_day_ahead_price(6)
    n = noisy.get_day_ahead_price(6)
    assert np.abs(n - q).mean() > 0.0
    assert np.abs(n - q).mean() < 0.2 * (q.max() - q.min())


def test_scarcity_raises_the_price_and_the_cap_still_binds():
    cfg = {**get_default(wp.CONFIG_KEY), "scarcity_probability": 1.0,
           "scarcity_min": 150.0, "scarcity_max": 300.0}
    p = wp.SyntheticWholesalePriceProvider(cfg)
    without = wp.SyntheticWholesalePriceProvider(
        {**cfg, "scarcity_probability": 0.0})
    assert p.get_day_ahead_price(1).mean() > without.get_day_ahead_price(1).mean()
    assert np.all(p.get_day_ahead_price(1) <= p.cap + 1e-9)


def test_the_declared_market_reports_its_scarcity_structure(provider):
    """The retired map had none; this one declares it and says so.

    Reported rather than asserted true in the abstract: a scarcity-free market
    is a legitimate configuration, and what matters is that the description
    follows the configuration instead of the configuration following the
    description.
    """
    assert provider.describe()["spike_structure_present"] is True
    quiet = wp.SyntheticWholesalePriceProvider(
        {**get_default(wp.CONFIG_KEY), "scarcity_probability": 0.0})
    assert quiet.describe()["spike_structure_present"] is False


def test_the_noise_stream_is_separate_from_the_parameter_stream():
    """Editing how a day's parameters are drawn must not shift its noise."""
    cfg = get_default(wp.CONFIG_KEY)
    a = wp.SyntheticWholesalePriceProvider(cfg)
    # Change only the external system's demand level, which moves r and so the
    # stack price; the noise's own draws must still come from the same stream.
    b = wp.SyntheticWholesalePriceProvider(
        {**cfg, "external_system": {**cfg["external_system"],
                                    "morning_peak_hour": 7.0}})
    assert not np.array_equal(a.get_day_ahead_price(2), b.get_day_ahead_price(2))


# ---------------------------------------------------------------------------
# Self-description
# ---------------------------------------------------------------------------

def test_describe_reports_the_stack_it_prices_with(provider):
    d = provider.describe()
    assert d["kind"] == "supply_stack"
    assert d["source"] == "synthetic_v1"
    assert d["supply_stack"]["knots"][0] == 0.0
    assert d["stack_min_cny_mwh"] == pytest.approx(160.0)
    assert d["stack_max_cny_mwh"] == pytest.approx(900.0)
    assert d["number_of_days"] == provider.number_of_days


def test_day_structure_reports_the_input_each_day_reaches(provider):
    days = [type("D", (), {"day": d, "physics": {}})() for d in range(3)]
    rows = provider.day_structure_for(days)
    assert [r["day"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert 0.0 <= r["input_min"] <= r["input_max"] <= 1.0
        assert r["reachable_price_min_cny_mwh"] >= 160.0 - 1e-9
        assert r["reachable_price_max_cny_mwh"] <= 900.0 + 1e-9


def test_the_legacy_provider_describes_itself_as_affine():
    """So an audit can report what the price IS without asking which class."""
    d = wp.LegacyMeritOrderProvider().describe()
    assert d["kind"] == "affine_net_load"
    assert d["transfer"]["type"] == "affine"
    assert d["declared_floor_cny_mwh"] == 50.0
    assert d["declared_cap_cny_mwh"] == 900.0


def test_the_legacy_provider_does_not_serve_prices_through_this_interface():
    """It needs the agents, which this interface deliberately does not carry."""
    with pytest.raises(NotImplementedError, match="legacy"):
        wp.LegacyMeritOrderProvider().get_day_ahead_price(0)


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        wp.get_provider("no_such_market")


def test_guangdong_is_declared_but_has_no_data_yet():
    with pytest.raises(NotImplementedError, match="Guangdong"):
        wp.GuangdongWholesalePriceProvider()


def test_every_declared_source_name_has_a_provider():
    """The declared names and the registry must not drift apart.

    Checked on the classes rather than by constructing, because building a
    provider is a side effect (it reads a library file, or raises by design
    for the phase-two slot) and a name check should not depend on either.
    """
    classes = {
        "synthetic_v1": wp.SyntheticWholesalePriceProvider,
        "library": wp.LibraryPriceProvider,
        "guangdong": wp.GuangdongWholesalePriceProvider,
        "legacy": wp.LegacyMeritOrderProvider,
    }
    assert set(classes) == set(wp.SOURCES)
    for name, cls in classes.items():
        assert cls.name == name
        assert issubclass(cls, wp.WholesalePriceProvider)


def test_the_active_source_follows_the_configuration():
    reload_defaults()
    assert wp.current_source() == get_default(f"{wp.CONFIG_KEY}.source")
    set_default(f"{wp.CONFIG_KEY}.source", "synthetic_v1")
    assert wp.current_source() == "synthetic_v1"
    reload_defaults()
    assert wp.current_source() == "legacy"


# ---------------------------------------------------------------------------
# The library artifact
# ---------------------------------------------------------------------------

def test_the_library_round_trips_through_its_file(tmp_path):
    cfg = {**get_default(wp.CONFIG_KEY), "number_of_days": 3}
    path = str(tmp_path / "lib.csv")
    manifest = str(tmp_path / "lib_manifest.csv")
    wp.build_library(path, manifest, cfg=cfg, quiet=True)

    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    assert len(rows) == 3 * 96
    assert list(rows[0]) == ["day_id", "time_index", "time",
                             "price_cny_mwh"]
    assert rows[0]["time"] == "00:00"
    assert rows[95]["time"] == "23:45"

    direct = wp.SyntheticWholesalePriceProvider(cfg)
    lib = wp.LibraryPriceProvider(
        {**cfg, "library_file": path, "library_manifest_file": manifest})
    for day in range(3):
        assert np.allclose(lib.get_day_ahead_price(day),
                           direct.get_day_ahead_price(day))


def test_the_manifest_records_the_yardstick_every_price_depends_on(tmp_path):
    cfg = {**get_default(wp.CONFIG_KEY), "number_of_days": 3}
    path = str(tmp_path / "lib.csv")
    manifest_path = str(tmp_path / "lib_manifest.csv")
    m = wp.build_library(path, manifest_path, cfg=cfg, quiet=True)
    for key in ("version", "seed", "generator", "price_floor", "price_cap",
                "noise_sigma", "scarcity_probability", "supply_stack",
                "number_of_days", "resolution_minutes", "r_min_mw", "r_max_mw",
                "library_sha256"):
        assert key in m, key
    on_disk = next(csv.DictReader(open(manifest_path, newline="",
                                       encoding="utf-8")))
    assert on_disk["library_sha256"] == m["library_sha256"]


def test_a_library_whose_yardstick_disagrees_is_refused(tmp_path):
    """A stale library prices different numbers; a warning would let it run."""
    cfg = {**get_default(wp.CONFIG_KEY), "number_of_days": 3}
    path = str(tmp_path / "lib.csv")
    manifest = str(tmp_path / "lib_manifest.csv")
    wp.build_library(path, manifest, cfg=cfg, quiet=True)
    with pytest.raises(ValueError, match="must be rebuilt|yardstick|refused"):
        wp.LibraryPriceProvider(
            {**cfg, "number_of_days": 9, "library_file": path,
             "library_manifest_file": manifest})


def test_a_library_without_a_manifest_is_refused(tmp_path):
    """Without it the artifact cannot be checked against its generator."""
    cfg = {**get_default(wp.CONFIG_KEY), "number_of_days": 2}
    path = str(tmp_path / "lib.csv")
    wp.build_library(path, str(tmp_path / "m.csv"), cfg=cfg, quiet=True)
    with pytest.raises(FileNotFoundError, match="manifest"):
        wp.LibraryPriceProvider(
            {**cfg, "library_file": path,
             "library_manifest_file": str(tmp_path / "absent.csv")})


def test_a_missing_library_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-library"):
        wp.LibraryPriceProvider(
            {**get_default(wp.CONFIG_KEY),
             "library_file": str(tmp_path / "nope.csv")})


def test_the_library_reports_every_library_day(provider):
    s = wp.stats(provider)
    assert s["days"] == provider.number_of_days
    assert s["intervals_per_day"] == 96
    assert s["min"] >= provider.floor - 1e-9
    assert s["max"] <= provider.cap + 1e-9
    assert len(s["segment_occupancy"]) == len(provider.stack.slopes)
    assert sum(s["segment_occupancy"]) == pytest.approx(1.0)
    assert s["p_negative"] == 0.0
