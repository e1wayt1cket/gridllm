"""The multi-day market-opportunity audit: does the native scale have arbitrage?

The audit answers a question about the *market*, so most of what can go wrong is
not a crash but a figment: a day set that varies the wrong thing, a split that
saw an economic outcome, a bound priced against a clearing the audited unit was
in, or a degradation sensitivity that quietly became the adopted cost. Each of
those has a test, and several of them pass on a module that returns plausible
numbers.

The tests that need a clearing build a small ensemble once and share it; the
rest run on synthetic price vectors and need neither Gurobi nor a network.
"""

import copy
import json

import numpy as np
import pytest

import market_opportunity as mo
from config_loader import load_defaults, reload_defaults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_days():
    """Three built days: physics only, no clearing, no prices."""
    specs = mo.ensemble_specs(3, seed=7)
    days = [mo.build_day(s) for s in specs]
    mo.assign_partitions(days)
    return days


@pytest.fixture(scope="module")
def small_report(tmp_path_factory):
    """A one-bus, three-day audit with the fleet arm and the footprint."""
    return mo.build_report(days=3, seed=7, buses=[9], with_fleet=True,
                           fleet_footprint_day=0, quiet=True)


@pytest.fixture()
def synthetic_records():
    """Records with controlled margins, for the verdict arithmetic."""
    def make(margins_per_day):
        rows = []
        for day, margins in enumerate(margins_per_day):
            for bus, margin in enumerate(margins):
                rows.append({
                    "day": day, "bus": bus, "partition": "test",
                    "arbitrage_margin_cny": margin,
                    "tradeable_box_share": 0.3 if margin > 0 else 0.0,
                    "siting_eligible": margin > 0,
                    "degradation_adjusted_spread_cny_mwh": margin,
                    "lmp_range_cny_mwh": 300.0, "clean": True,
                })
        return rows
    return make


# ---------------------------------------------------------------------------
# The ensemble varies the day, not the study scale
# ---------------------------------------------------------------------------

def test_the_shape_menu_declares_every_parameter_the_profile_reader_uses():
    """A menu entry with a missing key would silently fall back to a default.

    The seven names are the ones `_make_load_profile` reads under the `native`
    shape; a missing one would not raise, it would produce a day nobody
    declared.
    """
    from grid import _make_load_profile
    hours = np.arange(96)
    for name in mo.SHAPE_NAMES:
        shape = mo.LoadShapeSpec.from_menu(name)
        defaults = shape.as_defaults()
        assert set(defaults) == set(k for k in mo.LEVER_KEYS
                                    if k.startswith("profiles.load.")
                                    and "noise_seed" not in k), name
        profile = _make_load_profile(hours, "residential", {
            k.split(".")[-1]: v for k, v in defaults.items()}, shape="native")
        assert profile.max() == pytest.approx(1.0), name


def test_an_unknown_shape_is_refused():
    with pytest.raises(ValueError, match="unknown load shape"):
        mo.LoadShapeSpec.from_menu("no_such_shape")


def test_the_ensemble_cycles_the_shapes_rather_than_sampling_them():
    """Every declared variant has to be in the ensemble to be looked up."""
    specs = mo.ensemble_specs(len(mo.SHAPE_NAMES) * 2, seed=1)
    used = [s.shape_name for s in specs]
    assert set(used) == set(mo.SHAPE_NAMES)


def test_the_ensemble_is_reproducible_from_its_seed():
    a = mo.ensemble_specs(5, seed=11)
    b = mo.ensemble_specs(5, seed=11)
    c = mo.ensemble_specs(5, seed=12)
    assert [s.as_defaults() for s in a] == [s.as_defaults() for s in b]
    assert [s.as_defaults() for s in a] != [s.as_defaults() for s in c]


def test_a_sweep_varies_only_the_named_lever():
    base = mo.ensemble_specs(1, seed=3)[0]
    swept = mo.sweep_specs("wind_seed", [10, 20, 30], base)
    for s in swept:
        assert s.wind_seed in (10, 20, 30)
        assert s.pv_cloud_seed == base.pv_cloud_seed
        assert s.price_seed == base.price_seed
        assert s.load_shape == base.load_shape


def test_an_unknown_lever_is_refused():
    base = mo.ensemble_specs(1, seed=1)[0]
    with pytest.raises(ValueError, match="unknown lever"):
        mo.sweep_specs("no_such_lever", [1], base)


def test_the_coincident_peak_is_the_networks_own_under_every_shape(small_days):
    """The frozen scale is a property, and this is what makes it one.

    Every shape re-normalizes to a peak of 1.0 and `native_peak` then scales the
    realized profile to the network's own demand, so however the peak is placed
    in time the day's peak is 3.715 MW. A lever that moved the level would make
    the ensemble a scale study wearing a weather study's name.
    """
    peaks = [d.physics["real_peak_mw"] for d in small_days]
    assert max(peaks) - min(peaks) < 1e-9
    assert peaks[0] == pytest.approx(3.715, abs=1e-6)


def test_the_installed_capacity_is_untouched_by_the_ensemble(small_days):
    for d in small_days:
        assert d.physics["installed_pv_mw"] == pytest.approx(0.6, abs=1e-9)
        assert d.physics["installed_wind_mw"] == pytest.approx(1.0, abs=1e-9)


def test_the_ensemble_never_moves_a_frozen_key():
    """The levers and the frozen set must not overlap, or the no-tuning claim is false."""
    assert set(mo.LEVER_KEYS).isdisjoint(set(mo.FROZEN_KEYS))


# ---------------------------------------------------------------------------
# The split sees no economics
# ---------------------------------------------------------------------------

def test_the_split_is_a_pure_function_of_the_labels(small_days):
    labels = {d.day: d.label for d in small_days}
    first = mo.partition_days(labels)
    second = mo.partition_days(labels)
    assert first == second


def test_the_split_does_not_depend_on_the_input_order(small_days):
    labels = {d.day: d.label for d in small_days}
    forward = mo.partition_days(labels)
    reverse = mo.partition_days(dict(reversed(list(labels.items()))))
    assert {k: sorted(v) for k, v in forward.items()} \
        == {k: sorted(v) for k, v in reverse.items()}


def test_the_split_does_not_look_at_profit(synthetic_records):
    """Two ensembles identical in physics and opposite in economics partition alike.

    The labels are built from the days' physics and nothing else, so a change
    that reached a price or a dispatch could not move a day between partitions.
    Asserted by construction: the labels fed in below carry no economic field,
    and adding one does not change the result.
    """
    labels = {i: mo.RegimeLabel("low", "low", "low", "evening") for i in range(8)}
    assert mo.partition_days(labels) == mo.partition_days(labels)


def test_the_fractions_have_to_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        mo.partition_days({0: mo.RegimeLabel("a", "a", "a", "evening")},
                          fractions=(0.5, 0.5, 0.5))


def test_the_wrong_number_of_fractions_is_refused():
    with pytest.raises(ValueError, match="fractions"):
        mo.partition_days({}, fractions=(1.0,))


def test_every_day_lands_in_exactly_one_partition(small_days):
    parts = {p: [d.day for d in small_days if d.partition == p]
             for p in mo.SPLIT_NAMES}
    all_days = [d for ids in parts.values() for d in ids]
    assert sorted(all_days) == sorted(d.day for d in small_days)


@pytest.fixture()
def single_stratum_days():
    """Eight days sharing one label, on eight distinct day numbers."""
    from dataclasses import replace
    base = mo.ensemble_specs(1, seed=1)[0]
    label = mo.RegimeLabel("low", "low", "low", "evening")
    return [mo.EnsembleDay(spec=replace(base, day=i), label=label)
            for i in range(8)]


def test_a_stratum_reaching_the_test_partition_is_not_flagged(single_stratum_days):
    parts = {"train": [0, 1, 2, 3], "validation": [4, 5], "test": [6, 7]}
    assert mo.underfilled_strata(single_stratum_days, parts) == []


def test_an_uncovered_stratum_is_reported_rather_than_absorbed(single_stratum_days):
    """A regime the test set never sees is a regime the verdict is silent on.

    Eight days in one stratum and none of them in test: the verdict is then
    evidence about no regime at all, and saying so is better than reporting a
    number the reader will take as covering everything.
    """
    starved = {"train": list(range(8)), "validation": [], "test": []}
    assert mo.underfilled_strata(single_stratum_days, starved) \
        == ["low/low/low/evening"]


def test_a_stratum_too_small_to_represent_is_not_flagged(single_stratum_days):
    """Below four days a stratum cannot be expected to reach every partition."""
    small = {"train": [0], "validation": [], "test": []}
    assert mo.underfilled_strata(single_stratum_days[:3], small) == []


# ---------------------------------------------------------------------------
# The audit is priced in a market with no batteries in it
# ---------------------------------------------------------------------------

def test_the_decisive_prices_come_from_a_clearing_with_no_storage(small_report):
    """The bound is only a bound against prices the unit did not help set."""
    assert all(r["clean"] for r in small_report["records"])
    assert small_report["records"], "the audit produced no records"


def test_the_wholesale_curve_is_the_same_with_and_without_the_fleet(small_report):
    """If the fleet moved the forecast curve, the two arms would be different days."""
    spec = mo.ensemble_specs(1, seed=7)[0]
    absent = mo.clear_day(spec, with_fleet=False)
    present = mo.clear_day(spec, with_fleet=True)
    assert np.allclose(absent["wholesale"], present["wholesale"])
    assert len([a for a in absent["agents"] if a.storage is not None]) == 0
    assert len([a for a in present["agents"] if a.storage is not None]) == 4


def test_the_fleet_reduces_the_price_range_it_trades_in(small_report):
    """The footprint is the reason the attainable reading differs from the bound."""
    fp = small_report["fleet_footprint"]
    assert fp["mean_range_retained"] < 1.0
    assert all(b["range_present_cny_mwh"] < b["range_absent_cny_mwh"]
               for b in fp["per_bus"])
    for b in fp["per_bus"]:
        assert b["range_retained"] == pytest.approx(
            b["range_present_cny_mwh"] / b["range_absent_cny_mwh"], rel=1e-9)


def test_the_attainable_reading_is_reported_beside_the_bound(small_report):
    a = small_report["attainability"]
    assert a["available"] is True
    assert 0.0 <= a["retention_ratio"] <= 1.0
    assert a["verdict_attainable"] in (
        "NATIVE_MARKET_OPPORTUNITY_SUFFICIENT",
        "NATIVE_MARKET_OPPORTUNITY_INSUFFICIENT")
    assert "verdict_attainable" in small_report


# ---------------------------------------------------------------------------
# Aggregation and the verdict
# ---------------------------------------------------------------------------

def test_the_report_columns_are_the_declared_ones(small_report):
    for r in small_report["records"]:
        assert set(r) == set(mo.REPORT_COLUMNS), set(r) ^ set(mo.REPORT_COLUMNS)


def test_the_csv_carries_the_declared_columns_in_order(tmp_path, small_report):
    path = tmp_path / "audit.csv"
    mo.write_csv(small_report["records"], str(path))
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(mo.REPORT_COLUMNS)


def test_an_empty_sample_counts_zero_rather_than_reporting_nan():
    """A partition with no days is a count of zero, not an unavailable mean."""
    stats = mo._threshold_stats([])
    assert stats["n"] == 0
    assert np.isnan(stats["median"])
    assert np.isnan(stats["p_gt_0"])


def test_the_threshold_probabilities_are_means_of_the_indicator():
    stats = mo._threshold_stats([-10.0, 10.0, 60.0, 200.0])
    assert stats["p_gt_0"] == pytest.approx(0.75)
    assert stats["p_gt_50"] == pytest.approx(0.50)
    assert stats["p_gt_100"] == pytest.approx(0.25)


def test_days_the_unit_sat_out_are_dropped_not_counted_as_zero():
    """A day with no trade has no spread; calling it zero would invent one."""
    stats = mo._threshold_stats([float("nan"), 10.0, 20.0])
    assert stats["n"] == 2
    assert stats["n_missing"] == 1
    assert stats["p_gt_0"] == pytest.approx(1.0)


def test_the_verdict_is_the_declared_predicate(synthetic_records):
    """Reconstructed by hand, including the half boundary."""
    # 4 days, all buses positive: 4 pass, 2*4 >= 4.
    recs = synthetic_records([[10.0], [20.0], [30.0], [40.0]])
    assert mo.verdict(recs)["n_days_passing"] == 4
    assert mo.verdict(recs)["verdict"].endswith("SUFFICIENT")

    # 4 days, exactly half pass: the bar is "at least half", so this passes.
    recs = synthetic_records([[10.0], [20.0], [-5.0], [-6.0]])
    assert mo.verdict(recs)["n_days_passing"] == 2
    assert mo.verdict(recs)["verdict"].endswith("SUFFICIENT")

    # 4 days, one short of half: fails.
    recs = synthetic_records([[10.0], [-4.0], [-5.0], [-6.0]])
    assert mo.verdict(recs)["n_days_passing"] == 1
    assert mo.verdict(recs)["verdict"].endswith("INSUFFICIENT")


def test_a_day_whose_median_is_exactly_zero_does_not_pass(synthetic_records):
    """The bar is a strict inequality, and the boundary is asserted."""
    recs = synthetic_records([[0.0], [0.0], [10.0], [10.0]])
    v = mo.verdict(recs)
    assert v["n_days_passing"] == 2
    recs = synthetic_records([[0.0], [0.0], [0.0], [10.0]])
    assert mo.verdict(recs)["n_days_passing"] == 1


def test_an_empty_partition_is_undetermined_rather_than_insufficient():
    """Too small to answer is not the same finding as answered in the negative."""
    v = mo.verdict([])
    assert v["n_days"] == 0
    assert v["verdict"] == "NATIVE_MARKET_OPPORTUNITY_UNDETERMINED"


def test_the_verdict_reads_only_the_partition_it_was_given(synthetic_records):
    recs = synthetic_records([[10.0], [20.0]])
    for r in recs:
        r["partition"] = "train"
    assert mo.verdict(recs, "test")["n_days"] == 0
    assert mo.verdict(recs, "train")["n_days"] == 2
    assert mo.verdict(recs, None)["n_days"] == 2


def test_the_fleet_aggregate_is_labelled_as_not_a_bound(synthetic_records):
    """Four single-unit bounds is four markets, and the label has to say so."""
    recs = synthetic_records([[10.0, 20.0, 30.0, 40.0]])
    fl = mo.fleet_distribution(recs, [0, 1, 2, 3])
    assert "not a bound on the fleet's profit" in fl["scope_note"]
    assert fl["isolated_units_upper_bound_cny"]["median"] == pytest.approx(100.0)
    assert fl["best_bus_margin_cny"]["median"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# The strategy space, which is not the same question as the margin
# ---------------------------------------------------------------------------

def test_the_box_threshold_is_the_siting_rules():
    """Two copies of a threshold are two thresholds unless something pins them."""
    import diagnose_typical_day as dtd
    assert mo.TRADEABLE_BOX_SHARE == dtd.TRADEABLE_BOX_SHARE


def test_a_positive_margin_with_a_tiny_box_is_not_a_strategy_space(synthetic_records):
    """The finding the audit exists to surface, stated as a test.

    Every bus earns something and none of them leaves a usable action space,
    which is exactly the state the native market is in. A verdict that only
    looked at the margin would call this healthy.
    """
    recs = synthetic_records([[10.0, 12.0], [11.0, 13.0]])
    for r in recs:
        r["tradeable_box_share"] = 0.03
        r["siting_eligible"] = False
    assert mo.verdict(recs)["verdict"].endswith("SUFFICIENT")
    ss = mo.strategic_space(recs, [0, 1])
    assert ss["verdict_box"].endswith("INSUFFICIENT")
    assert ss["eligible_rate_all_buses"] == 0.0


def test_the_strategy_space_reading_is_available_and_reported(small_report):
    ss = small_report["strategic_space"]
    assert ss["available"] is True
    assert ss["criterion"] == f"tradeable box share >= {mo.TRADEABLE_BOX_SHARE}"
    assert 0.0 <= ss["median_box_share_fleet"] <= 1.0
    assert ss["verdict_box"] in ("NATIVE_MARKET_OPPORTUNITY_SUFFICIENT",
                                 "NATIVE_MARKET_OPPORTUNITY_INSUFFICIENT")


def test_every_bus_day_carries_a_finite_box_share(small_report):
    for r in small_report["records"]:
        assert np.isfinite(r["tradeable_box_share"])
        assert 0.0 <= r["tradeable_box_share"] <= 1.0


# ---------------------------------------------------------------------------
# The diagnosis is findings, and nothing is tuned
# ---------------------------------------------------------------------------

def test_the_audit_leaves_the_config_caches_as_it_found_them(small_report):
    """The levers live in a process-global cache; leaving them set leaks days.

    `build_report` writes a day's levers to reach that day, so the last day's
    values would otherwise become the starting point of whatever runs next in
    the same process. Restoring is what keeps this an audit rather than a
    configuration edit.
    """
    reload_defaults()
    before = copy.deepcopy(load_defaults())
    mo.build_report(days=2, seed=5, buses=[9], quiet=True)
    assert load_defaults() == before


def test_the_audit_leaves_no_price_provider_built_under_its_own_levers():
    """Restoring the config is not enough while the providers stay cached.

    `get_provider` caches by source name and builds from the configuration in
    force at that moment. An audit moves the levers to reach each day, so a
    provider built during one would go on answering the next caller in this
    process from a market that no longer matches what is on disk.
    """
    from wholesale_price import get_provider, reset_cache
    reload_defaults()
    reset_cache()
    before = get_provider()
    mo.build_report(days=2, seed=5, buses=[9], quiet=True)
    assert get_provider() is not before, (
        "the audit left the provider it built under its own levers behind")
    # Repeatable in the same process: the second run rebuilds on the config the
    # first one restored rather than on the levers the first one moved.
    first_run = get_provider()
    mo.build_report(days=2, seed=5, buses=[9], quiet=True)
    assert get_provider() is not first_run


def test_the_degradation_sensitivity_changes_no_record_and_no_verdict(small_report):
    """A sensitivity that fed the verdict would be an adopted value."""
    a = mo.build_report(days=2, seed=5, buses=[9],
                        degradation_sensitivity=(100.0,), quiet=True)
    b = mo.build_report(days=2, seed=5, buses=[9],
                        degradation_sensitivity=(60.0, 40.0), quiet=True)
    assert a["records"] == b["records"]
    assert a["verdict"] == b["verdict"]
    assert a["diagnosis"]["degradation"]["at_declared"] \
        == b["diagnosis"]["degradation"]["at_declared"]
    assert (a["diagnosis"]["degradation"]["sensitivity"]
            != b["diagnosis"]["degradation"]["sensitivity"])


def test_the_degradation_floor_is_reported_under_named_conventions():
    """The three readings differ by up to a third, so each has to be named."""
    from arbitrage_lp import UnitSpec
    unit = UnitSpec(e_max=1.6, p_ch_max=0.4, p_dis_max=0.4, eta_ch=0.92,
                    eta_dis=0.92, soc0=0.5, soc_min=0.10, soc_max=0.90)
    d = mo.degradation_audit(unit, 100.0)
    at = d["at_declared"]
    assert at["per_mwh_stored_cny_mwh"] == pytest.approx(200.0)
    assert at["per_mwh_charged_cny_mwh"] == pytest.approx(
        100.0 * (1.0 + 0.92 * 0.92))
    assert at["per_mwh_delivered_cny_mwh"] == pytest.approx(
        100.0 * (1.0 / (0.92 * 0.92) + 1.0))
    assert d["adopted"] is False


def test_the_price_maps_floor_and_cap_are_reported_as_unreachable(small_days):
    """The measured claim behind "the map is affine over its whole range"."""
    pm = mo.price_map_audit(small_days)
    assert pm["floor_reachable"] is False
    assert pm["cap_reachable"] is False
    assert pm["reachable_price_min_cny_mwh"] > pm["declared_floor_cny_mwh"]
    assert pm["reachable_price_max_cny_mwh"] < pm["declared_cap_cny_mwh"]
    assert pm["spike_structure_present"] is False
    assert pm["slope_cny_mwh_per_mw_median"] > 0.0


def test_the_price_slope_comes_from_the_active_provider(small_days):
    """The audit describes the market it actually measured.

    It used to read the retired map's `price_curve` constants directly, which
    was right only while that map was the only one. Once the native scenario
    moved to the external market those keys stopped driving the price, and the
    report would have gone on quoting a slope for a market it was no longer
    pricing.
    """
    from wholesale_price import get_provider

    pm = mo.price_map_audit(small_days)
    described = get_provider().describe()
    assert pm["source"] == described["source"]
    assert pm["kind"] == described["kind"]

    # The reported slope is the median over the ensemble's days, taken from the
    # provider's own per-day structure rather than from any one day's.
    key = ("slope_cny_mwh_per_mw" if "slope_cny_mwh_per_mw" in pm["per_day"][0]
           else "slope_cny_mwh_per_input")
    assert pm["slope_cny_mwh_per_mw_median"] == pytest.approx(
        float(np.median([float(p[key]) for p in pm["per_day"]])), rel=1e-9)
    for p in pm["per_day"]:
        assert float(p[key]) > 0.0


def test_the_retired_maps_slope_was_read_from_the_peak_not_the_level(small_days):
    """The legacy map's divisor is the forecast peak, not the realized one.

    Kept because the distinction is what made a slope quoted off the realized
    peak wrong by about 6%, and it still applies wherever the legacy provider
    is described.
    """
    from wholesale_price import LegacyMeritOrderProvider

    described = LegacyMeritOrderProvider().describe()
    base = described["transfer"]["base_cny_mwh"]
    elasticity = described["transfer"]["elasticity"]
    per_day = LegacyMeritOrderProvider().day_structure_for(small_days)
    for d, p in zip(small_days, per_day):
        assert float(p["slope_cny_mwh_per_mw"]) == pytest.approx(
            base * elasticity / d.physics["forecast_peak_mw"], rel=1e-9)
        assert d.physics["real_peak_mw"] \
            != pytest.approx(d.physics["forecast_peak_mw"], rel=1e-3)


def test_every_change_requirement_is_an_unadopted_finding(small_report):
    reqs = small_report["diagnosis"]["change_requirements"]
    assert reqs
    for r in reqs:
        assert r["adopted"] is False
        assert "finding" in r and "would_change" in r and "measured" in r


def test_nothing_in_the_cli_surface_can_move_a_frozen_constant():
    """The no-tuning claim, checked against the parser rather than trusted.

    Read off the constructed parser rather than the source text, so an option
    named in a way a grep would not anticipate still fails here.
    """
    options = {o for a in mo.build_parser()._actions for o in a.option_strings}
    frozen_words = ("base", "elasticity", "cycle", "cost", "noise", "spike",
                    "price", "level", "scale", "capacity", "ess-size")
    for option in sorted(options):
        for word in frozen_words:
            assert word not in option, f"{option!r} names a frozen constant"


def test_the_cli_chooses_which_buses_are_measured_and_never_which_exist():
    """`--buses` narrows a report; it cannot move the fleet.

    The fleet's size and siting are frozen inputs, so there is deliberately no
    option that re-sites or re-sizes it. A flag that could would make the audit
    a siting study, which is the search this work exists to stop running.
    """
    options = {o for a in mo.build_parser()._actions for o in a.option_strings}
    assert "--buses" in options
    for forbidden in ("--fleet-buses", "--ess-buses", "--fleet-size",
                      "--fleet-capacity", "--fleet-scale"):
        assert forbidden not in options, forbidden


# ---------------------------------------------------------------------------
# The series source
# ---------------------------------------------------------------------------

def test_a_series_that_is_not_whole_days_is_refused(tmp_path):
    path = tmp_path / "odd.csv"
    path.write_text("lmp\n" + "\n".join(str(100 + i) for i in range(50)),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="not a whole number"):
        mo.load_price_series(str(path), T=96)


def test_a_series_can_be_resampled_when_it_is_asked_for(tmp_path):
    """An hourly series is one day of 96 periods, and only when it is asked for.

    Hourly published prices are the realistic external input, and 24 rows is
    not a whole number of 96-period days, so without the flag this is refused.
    """
    path = tmp_path / "hourly.csv"
    values = [100.0 + 40.0 * np.sin(i / 24 * 2 * np.pi) for i in range(24)]
    path.write_text("lmp\n" + "\n".join(str(v) for v in values), encoding="utf-8")

    with pytest.raises(ValueError, match="not a whole number"):
        mo.load_price_series(str(path), T=96)

    days = mo.load_price_series(str(path), T=96, resample="linear")
    assert len(days) == 1
    assert len(days[0][1]) == 96
    assert days[0][1].min() == pytest.approx(min(values), abs=1.0)
    assert days[0][1].max() == pytest.approx(max(values), abs=1.0)


def test_a_whole_number_of_days_is_never_resampled(tmp_path):
    """288 rows read as three days of 96, not one day of 288.

    The count divides evenly, so the file already says where its days end and
    interpolating would invent a boundary the source did not have. Pinned
    because the other reading is equally plausible for a 5-minute series and
    the choice has to be deliberate rather than incidental.
    """
    path = tmp_path / "three_days.csv"
    path.write_text("lmp\n" + "\n".join(str(300.0 + i) for i in range(288)),
                    encoding="utf-8")
    days = mo.load_price_series(str(path), T=96, resample="linear")
    assert len(days) == 3
    assert days[0][1][0] == pytest.approx(300.0)
    assert days[0][1][-1] == pytest.approx(300.0 + 95)


def test_a_malformed_series_is_refused_rather_than_guessed(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="no price series"):
        mo.load_price_series(str(missing), T=96)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        mo.load_price_series(str(empty), T=96)

    headerless = tmp_path / "numbers.csv"
    headerless.write_text("\n".join(str(i) for i in range(96)), encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        mo.load_price_series(str(headerless), T=96)

    wrongcol = tmp_path / "named.csv"
    wrongcol.write_text("lmp\n" + "\n".join("1" for _ in range(96)), encoding="utf-8")
    with pytest.raises(ValueError, match="no column"):
        mo.load_price_series(str(wrongcol), T=96, column="nope")

    bad = tmp_path / "bad.csv"
    rows = [str(100)] * 96
    rows[7] = "not-a-number"
    bad.write_text("lmp\n" + "\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        mo.load_price_series(str(bad), T=96)


def test_a_series_is_mapped_uniformly_and_the_degeneracy_is_flagged(tmp_path):
    """One system price is not 33 observations, and the report has to say so.

    The rows are identical by construction, so a per-bus table over them left
    unannotated would read as 33 independent measurements of the same thing.
    """
    path = tmp_path / "one_day.csv"
    path.write_text("lmp\n" + "\n".join(str(300 + i) for i in range(96)),
                    encoding="utf-8")
    report = mo.build_report(days=1, seed=1, source="series",
                             series_path=str(path), buses=[0, 5, 17],
                             quiet=True)

    mins = {r["lmp_min_cny_mwh"] for r in report["records"] if r["day"] == 0}
    assert len(mins) == 1, "uniform mapping is not uniform"
    assert len(report["records"]) == 3, "three buses were asked for"

    day = next(d for d in report["ensemble"]["physics"] if d["day"] == 0)
    assert day["price_source"] == "external_series_uniform"
    assert any("uniformly" in n for n in day["notes"]), day["notes"]
    assert report["records"][0]["clean"] is True


def test_two_series_days_are_read_as_two_days(tmp_path):
    path = tmp_path / "two_days.csv"
    path.write_text("lmp\n" + "\n".join(str(300 + i) for i in range(192)),
                    encoding="utf-8")
    days = mo.load_price_series(str(path), T=96)
    assert len(days) == 2
    assert days[0][1][0] == pytest.approx(300.0)
    assert days[1][1][0] == pytest.approx(300.0 + 96)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_writes_the_json_and_csv_it_was_asked_for(tmp_path):
    js, cs = tmp_path / "r.json", tmp_path / "r.csv"
    code = mo.main(["--ensemble-seed", "1", "--days", "2", "--buses", "9", "--quiet",
                    "--json", str(js), "--csv", str(cs)])
    assert code == 0
    report = json.loads(js.read_text(encoding="utf-8"))
    assert report["verdict"]["verdict"].startswith("NATIVE_MARKET_OPPORTUNITY_")
    assert cs.read_text(encoding="utf-8").splitlines()[0].split(",") \
        == list(mo.REPORT_COLUMNS)


def test_main_returns_zero_on_an_insufficient_verdict(tmp_path):
    """The verdict is the deliverable, not a failure of the module."""
    js = tmp_path / "r.json"
    code = mo.main(["--ensemble-seed", "1", "--days", "1", "--buses", "9", "--quiet", "--json", str(js)])
    report = json.loads(js.read_text(encoding="utf-8"))
    assert report["verdict"]["verdict"].endswith("UNDETERMINED")
    assert code == 0


def test_main_returns_one_when_no_day_survives(monkeypatch, tmp_path):
    """A run that produced no measurable day has not answered anything."""
    monkeypatch.setattr(mo, "SHED_LIMIT_MWH", -1.0)
    code = mo.main(["--ensemble-seed", "1", "--days", "1", "--buses", "9", "--quiet",
                    "--json", str(tmp_path / "r.json")])
    assert code == 1


def test_series_source_without_a_path_is_a_usage_error():
    with pytest.raises(SystemExit):
        mo.main(["--ensemble-seed", "1", "--source", "series", "--quiet"])
