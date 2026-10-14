# tests/test_industry_load.py
"""Contracts of the industry load preprocessor.

The workbook is messy in specific, measured ways — synthesised placeholder rows,
incomplete days, duplicates, a meter multiplier that agrees with the declared
rating for some meters and not others. The module's job is to publish an
intraday *shape* per industry while altering nothing and hiding nothing.

These tests pin two properties that are easy to lose:

* **Nothing is repaired.** A negative reading stays negative, an implausible
  ratio stays as it is, and the only response is a flag. If a future change
  starts clamping or imputing, that shows up here as a changed value.
* **Weighting is by the hierarchy, not by the row count.** A user with many
  meters, or many recorded days, must not outvote a user with few. The workbook
  makes this concrete: one commercial user contributes thirteen times the rows
  of another.

Most tests build records directly so they stay fast and independent of the
67 MiB workbook; the end-to-end read is marked slow.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt

import numpy as np
import pytest

import industry_load as il


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

N = il.N_SLOTS


def profile(peak_slot, base=0.1, peak=1.0):
    """A flat profile with a single spike, long enough to be a real day."""
    v = [base] * N
    v[peak_slot] = peak
    return tuple(float(x) for x in v)


def day(n):
    """The n-th day of a synthetic calendar, safely past any month boundary."""
    return dt.date(2021, 1, 1) + dt.timedelta(days=n)


def mk(account="ACC1", meter="MTR1", industry="化工业", values=None,
       ratio=1000.0, running=50000.0, date=dt.date(2021, 1, 1),
       user_name="user", location="南京", excel_row=2,
       phase=il.PHASE_ACTIVE_POWER):
    values = profile(48) if values is None else values
    return il.MeterDay(
        account=account, meter=meter, user_name=user_name, industry=industry,
        location=location, excel_row=excel_row, date=date, values=values,
        ratio=ratio, running_capacity=running, phase=phase, flags=())


def flagged(records):
    return il.flag_records(records)


# ---------------------------------------------------------------------------
# Slot axis
# ---------------------------------------------------------------------------

def test_slot_axis_is_96_columns_at_15_minutes():
    """The header must be exactly 00:00..23:45, ascending by 15 minutes."""
    header = list(il.META_COLUMNS) + [
        (dt.datetime(2000, 1, 1) + dt.timedelta(minutes=15 * i)).time()
        for i in range(N)]
    il._check_time_header(header)          # must not raise
    assert len(header) == il.COL_SLOT0 + N


def test_slot_axis_rejects_a_coarser_grid():
    """A 30-minute file must fail loudly rather than be silently accepted."""
    header = list(il.META_COLUMNS) + [
        (dt.datetime(2000, 1, 1) + dt.timedelta(minutes=30 * i)).time()
        for i in range(N)]
    with pytest.raises(ValueError, match="time axis"):
        il._check_time_header(header)


def test_slot_axis_rejects_a_short_block():
    header = list(il.META_COLUMNS) + [
        (dt.datetime(2000, 1, 1) + dt.timedelta(minutes=15 * i)).time()
        for i in range(N - 1)]
    with pytest.raises(ValueError):
        il._check_time_header(header)


# ---------------------------------------------------------------------------
# Day classification
# ---------------------------------------------------------------------------

def test_complete_incomplete_and_zero_days():
    complete = mk(meter="A", values=profile(48))
    incomplete = mk(meter="B", values=tuple(
        [None] + list(profile(48)[1:])))
    allzero = mk(meter="C", values=tuple([0.0] * N))

    a, b, c = flagged([complete, incomplete, allzero])

    assert not a.flags
    assert il.FLAG_MISSING_SLOTS in b.flags
    assert il.FLAG_ALL_ZERO in c.flags
    # A complete day is usable; the other two are not.
    assert not a.excluded and b.excluded and c.excluded


def test_empty_identity_is_dropped():
    md = mk(account="", meter="", date=None)
    md = il.flag_meter_day(md, None)
    assert il.FLAG_EMPTY_IDENTITY in md.flags
    assert md.excluded


def test_non_active_power_phase_is_dropped():
    """功率相序 other than 有功功率 has no established unit convention."""
    ok = il.flag_meter_day(mk(), None)
    other = il.flag_meter_day(mk(phase="其他"), None)

    assert il.FLAG_OTHER_PHASE not in ok.flags      # 有功功率 is the norm
    assert il.FLAG_OTHER_PHASE in other.flags
    assert other.excluded


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def test_placeholder_row_is_recognised_structurally():
    head = il.PLACEHOLDER_HEAD
    values = tuple(head) + tuple([0.0] * (N - len(head)))
    assert sum(values) == il.PLACEHOLDER_SUM
    assert il.is_placeholder(values)
    md = il.flag_meter_day(mk(values=values), None)
    assert il.FLAG_PLACEHOLDER in md.flags
    assert md.excluded


def test_placeholder_lookalike_is_not_swallowed():
    """A genuine reading that opens 1,2,3,3,3 but totals something else stays."""
    head = il.PLACEHOLDER_HEAD
    tail = [0.0] * (N - len(head))
    tail[10] = 0.5                      # breaks the exact-sum contract
    values = tuple(head) + tuple(tail)
    assert not il.is_placeholder(values)


def test_placeholder_lookalike_with_missing_slots_is_not_placeholder():
    head = il.PLACEHOLDER_HEAD
    tail = [None] + [0.0] * (N - len(head) - 1)
    assert not il.is_placeholder(tuple(head) + tuple(tail))


# ---------------------------------------------------------------------------
# Flags never repair
# ---------------------------------------------------------------------------

def test_negative_values_are_flagged_and_left_alone():
    values = list(profile(48))
    values[3] = -0.25
    values[70] = -0.5
    original = tuple(values)
    md = il.flag_meter_day(mk(industry="印刷业", values=original), None)

    assert il.FLAG_NEGATIVE_LOAD in md.flags
    assert md.values == original            # byte-for-byte unchanged
    assert min(md.values) == -0.5           # nothing clamped to zero


def test_pv_negatives_use_a_distinct_flag():
    values = list(profile(48))
    values[3] = -0.25
    md = il.flag_meter_day(
        mk(industry=il.PV_INDUSTRY, values=tuple(values)), None)
    assert il.FLAG_NEGATIVE_PV in md.flags
    assert il.FLAG_NEGATIVE_LOAD not in md.flags


def test_ratio_missing_is_flagged_not_guessed():
    md = il.flag_meter_day(mk(ratio=None), None)
    assert il.FLAG_RATIO_MISSING in md.flags
    assert md.ratio is None


def test_implausible_ratio_is_flagged_and_unchanged():
    """A ratio implying thousands of times the rating is reported, not fixed."""
    md = il.flag_meter_day(mk(ratio=1_320_000.0, running=240_000.0), None)
    assert il.FLAG_CAPACITY_INCONSISTENT in md.flags
    assert md.ratio == 1_320_000.0          # the value survives intact


def test_consistent_ratio_is_not_flagged():
    md = il.flag_meter_day(mk(ratio=100.0, running=1000.0), None)
    assert il.FLAG_CAPACITY_INCONSISTENT not in md.flags


def test_ratio_outlier_uses_the_industry_median():
    a = mk(account="A", meter="M1", industry="纺织业", ratio=100.0)
    b = mk(account="B", meter="M2", industry="纺织业", ratio=100.0)
    far = mk(account="C", meter="M3", industry="纺织业", ratio=1_000_000.0)
    out = {md.meter: md for md in flagged([a, b, far])}
    assert il.FLAG_RATIO_SUSPECT in out["M3"].flags
    assert il.FLAG_RATIO_SUSPECT not in out["M1"].flags


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_identical_repeat_is_dropped_from_shapes_but_kept_in_report():
    a = mk(meter="M1", values=profile(48))
    b = mk(meter="M1", values=profile(48))       # same meter-day, same profile
    out = {i: md for i, md in enumerate(flagged([a, b]))}
    assert il.FLAG_DUPLICATE_ROW in out[1].flags

    shapes = il.build_shapes(list(out.values()))
    s = shapes.shapes["化工业"]
    assert s.n_observations == 1                 # counted once, not twice

    # ...and the dropped duplicate is still reported.
    frame = il.exclusions_to_frame(list(out.values()))
    assert (frame["flags"].str.contains(il.FLAG_DUPLICATE_ROW)).sum() == 1


def test_conflicting_repeat_keeps_both_readings():
    """Two different profiles for one meter-day is a disagreement, not noise."""
    a = mk(meter="M1", values=profile(10))
    b = mk(meter="M1", values=profile(80))       # same key, different profile
    out = list(flagged([a, b]))
    assert il.FLAG_DUPLICATE_CONFLICT in out[1].flags
    assert not out[1].excluded
    shapes = il.build_shapes(out)
    assert shapes.shapes["化工业"].n_observations == 2


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_single_meter_day_is_normalised_to_unit_peak():
    shape = il.normalise(profile(48, base=2.0, peak=8.0))
    assert shape is not None
    assert shape.shape == (N,)
    assert shape.max() == pytest.approx(1.0)
    assert shape[48] == pytest.approx(1.0)
    assert shape[0] == pytest.approx(0.25)


def test_normalisation_is_scale_invariant():
    """The meter multiplier cancels, which is why an implausible one is safe."""
    a = il.normalise(profile(48, base=0.1, peak=1.0))
    b = il.normalise(profile(48, base=100.0, peak=1000.0))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_all_negative_day_scales_instead_of_flipping():
    """A PV day of pure reverse flow keeps its sign."""
    values = tuple(-float(v) for v in profile(48))
    shape = il.normalise(values)
    assert shape is not None
    assert shape.min() == pytest.approx(-1.0)     # the spike, still negative
    assert shape.max() == pytest.approx(-0.1)     # the floor, still negative
    assert np.all(shape <= 0)                     # nothing rectified


def test_degenerate_profile_returns_none():
    assert il.normalise(tuple([0.0] * N)) is None
    assert il.normalise(tuple([float("nan")] * N)) is None


# ---------------------------------------------------------------------------
# Hierarchical aggregation
# ---------------------------------------------------------------------------

def test_multi_meter_user_day_is_averaged_once():
    """One day with two meters is one observation, the mean of the two."""
    p_top = profile(10)
    p_bot = profile(80)
    day = il.build_shapes(flagged([
        mk(account="U1", meter="M1", date=dt.date(2021, 1, 1), values=p_top),
        mk(account="U1", meter="M2", date=dt.date(2021, 1, 1), values=p_bot),
    ]))
    curve = day.shape_of("化工业")
    expected = (il.normalise(p_top) + il.normalise(p_bot)) / 2.0
    expected = expected / expected.max()
    np.testing.assert_allclose(curve, expected, rtol=1e-12)


def test_meter_count_does_not_change_the_industry_curve():
    """A user with ten meters must not outvote a user with one."""
    one = [mk(account="U1", meter="M1", values=profile(10))]
    many_same = [mk(account="U1", meter=f"M{i}", values=profile(10))
                 for i in range(10)]
    other = [mk(account="U2", meter="N1", values=profile(80))]

    a = il.build_shapes(flagged(one + other)).shape_of("化工业")
    b = il.build_shapes(flagged(many_same + other)).shape_of("化工业")
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_day_count_does_not_change_the_industry_curve():
    """A user with many recorded days must not outvote a user with one."""
    user_a = [mk(account="U1", meter="M1", date=dt.date(2021, 1, 1),
                 values=profile(10))]
    user_b_once = [mk(account="U2", meter="N1", date=dt.date(2021, 1, 1),
                      values=profile(80))]
    user_b_many = [mk(account="U2", meter="N1",
                      date=day(i), values=profile(80))
                   for i in range(20)]

    a = il.build_shapes(flagged(user_a + user_b_once)).shape_of("化工业")
    b = il.build_shapes(flagged(user_a + user_b_many)).shape_of("化工业")
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_industries_are_aggregated_separately():
    shapes = il.build_shapes(flagged([
        mk(account="U1", meter="M1", industry="化工业", values=profile(10)),
        mk(account="U2", meter="N2", industry="钢铁", values=profile(80)),
    ]))
    assert set(shapes.industries()) == {"化工业", "钢铁"}
    assert int(np.argmax(shapes.shape_of("化工业"))) == 10
    assert int(np.argmax(shapes.shape_of("钢铁"))) == 80


def test_pv_is_kept_apart_from_the_load_pool():
    """分布式光伏 is a generator; it must never merge into a load industry."""
    shapes = il.build_shapes(flagged([
        mk(account="U1", meter="M1", industry="化工业", values=profile(10)),
        mk(account="U2", meter="N2", industry=il.PV_INDUSTRY,
           values=profile(60)),
    ]))
    assert il.PV_INDUSTRY in shapes.shapes
    assert "化工业" in shapes.shapes
    assert int(np.argmax(shapes.shape_of("化工业"))) == 10


# ---------------------------------------------------------------------------
# Reliability marking
# ---------------------------------------------------------------------------

def test_insufficient_sample_is_marked_not_filled():
    shapes = il.build_shapes(flagged([
        mk(account="U1", meter="M1", industry="电解铝", values=profile(10)),
    ]))
    s = shapes.shapes["电解铝"]
    assert s.reliable is False
    assert "limited" in s.reliability_note
    assert "users 1 < 3" in s.reliability_note


def test_industry_with_no_surviving_observation_gets_no_curve():
    """An absent curve must stay absent; copying another industry is forbidden."""
    shapes = il.build_shapes(flagged([
        mk(account="U1", meter="M1", industry="电解铝",
           values=tuple([0.0] * N)),
        mk(account="U2", meter="N2", industry="化工业", values=profile(10)),
    ]))
    assert "电解铝" not in shapes.shapes
    assert "化工业" in shapes.shapes


def test_sufficient_sample_is_marked_reliable():
    records = [mk(account=f"U{u}", meter=f"M{u}_{d}",
                  date=day(d), values=profile(20))
               for u in range(4) for d in range(40)]
    s = il.build_shapes(flagged(records)).shapes["化工业"]
    assert s.reliable is True
    assert s.n_users == 4


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

def test_every_shape_is_96_finite_points():
    records = [mk(account=f"U{u}", meter=f"M{u}", date=day(d),
                  values=profile(20 + u)) for u in range(4) for d in range(40)]
    shapes = il.build_shapes(flagged(records))
    assert shapes
    for industry in shapes.industries():
        arr = shapes.shape_of(industry)
        assert arr.shape == (N,)
        assert np.all(np.isfinite(arr))
        assert arr.max() == pytest.approx(1.0)
        assert arr.min() >= 0.0


def test_written_frames_have_the_expected_shape():
    records = flagged([
        mk(account=f"U{u}", meter=f"M{u}", date=day(d),
           values=profile(20)) for u in range(4) for d in range(40)])
    shapes = il.build_shapes(records)

    typical = il.shapes_to_frame(shapes)
    assert len(typical) == len(shapes) * N
    assert set(typical["time_index"]) == set(range(N))

    summary = il.summary_to_frame(shapes)
    assert (summary["n_slots"] == N).all()


def test_exclusions_locate_the_source_row_without_identities():
    rows = [mk(account="1234567890", meter="987654321", excel_row=77,
               values=tuple([0.0] * N))]
    frame = il.exclusions_to_frame(flagged(rows))
    assert len(frame) == 1
    assert frame.loc[0, "excel_row"] == 77          # auditable
    assert "1234567890" not in frame.to_csv(index=False)   # account not exposed
    assert frame.loc[0, "user_id"].startswith("U")
    assert frame.loc[0, "meter_id"].startswith("M")


def test_rerun_is_deterministic():
    records = [mk(account=f"U{u}", meter=f"M{u}", date=day(d),
                  values=profile(20 + u)) for u in range(4) for d in range(40)]
    first = il.build_shapes(flagged(records))
    second = il.build_shapes(flagged(records))
    assert first.industries() == second.industries()
    for industry in first.industries():
        np.testing.assert_array_equal(first.shape_of(industry),
                                      second.shape_of(industry))


# ---------------------------------------------------------------------------
# Absolute MW is refused
# ---------------------------------------------------------------------------

def test_absolute_mw_refuses_without_verification():
    shapes = il.build_shapes(flagged([
        mk(account=f"U{u}", meter=f"M{u}", date=day(d),
           values=profile(20)) for u in range(4) for d in range(40)]))
    with pytest.raises(ValueError, match="not verified"):
        il.absolute_mw(shapes, {3: "化工业"}, {3: 1.0})


def test_absolute_mw_works_once_verified():
    shapes = il.build_shapes(flagged([
        mk(account=f"U{u}", meter=f"M{u}", date=day(d),
           values=profile(20)) for u in range(4) for d in range(40)]))
    out = il.absolute_mw(shapes, {3: "化工业"}, {3: 2.0},
                         verified_industries=["化工业"])
    assert out[3].max() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Ratio evidence
# ---------------------------------------------------------------------------

def test_ratio_evidence_counts_both_verdicts():
    good = mk(account="A", meter="M1", ratio=100.0, running=1000.0)
    bad = mk(account="B", meter="M2", ratio=1_000_000.0, running=1000.0)
    ev = il.verify_ratios([good, bad])
    assert ev["meters_checked"] == 2
    assert ev["meters_consistent"] == 1
    assert ev["meters_inconsistent"] == 1


# ---------------------------------------------------------------------------
# The qualified subset: the gate the absolute load has to pass
# ---------------------------------------------------------------------------

def _one_day(meter="M1", account="A", **kwargs):
    """One meter, one date -- a single block unless a second row is added."""
    return mk(account=account, meter=meter, **kwargs)


def test_a_meter_whose_peak_contradicts_capacity_is_not_qualified():
    """max|slot| * ratio * 4 / running = 1.0 * 1e6 * 4 / 1000, far past 1.5x."""
    recs = flagged([_one_day(meter="M1", ratio=1_000_000.0, running=1000.0)])
    verdicts = il.qualified_meters(recs)
    assert verdicts[_one_day(meter="M1").meter_key]["qualified"] is False
    assert il.qualified_records(recs) == []
    assert il.qualified_block_keys(recs) == set()


def test_a_meter_whose_ratio_cannot_be_read_is_not_qualified():
    """Missing evidence is not a pass: the workbook not saying a meter's
    magnitude is not the network agreeing with it."""
    recs = flagged([_one_day(meter="M1", ratio=None)])
    verdicts = il.qualified_meters(recs)
    key = _one_day(meter="M1").meter_key
    assert verdicts[key] == {"checkable": False, "qualified": False,
                             "peak_ratio": None, "industry": "化工业"}
    assert il.qualified_records(recs) == []


def test_a_meter_whose_capacity_cannot_be_read_is_not_qualified():
    recs = flagged([_one_day(meter="M1", running=None)])
    assert il.qualified_meters(recs)[_one_day(meter="M1").meter_key]["qualified"] is False
    assert il.qualified_records(recs) == []


def test_a_clean_block_cannot_bypass_its_disqualified_meter():
    """The gate is one decision per meter. A block carrying no flag of its own
    must still be out when its meter is out -- otherwise the block-level route
    would quietly overrule the meter-level one."""
    recs = flagged([_one_day(meter="M1", ratio=1_000_000.0, running=1000.0)])
    assert not (set(recs[0].flags) & set(il.SHAPE_DROP_FLAGS))
    assert il.qualified_records(recs) == []


def test_a_qualified_meter_still_needs_a_block_that_survives_the_pool():
    """Both gates, not either: a good meter does not rescue a dropped block."""
    good = flagged([_one_day(meter="M1", date=day(0))])
    dead = flagged([_one_day(meter="M1", date=day(1),
                             values=tuple([0.0] * N))])
    assert il.qualified_block_keys(good + dead) == {(good[0].meter_key, day(0))}
    # The meter itself is unaffected by one of its days being dropped.
    assert il.qualified_meters(good + dead)[good[0].meter_key]["qualified"] is True


def test_blocks_are_counted_as_pairs_not_as_records():
    """The unit the ruling is stated in. A repeated meter-day is two records
    and one block, and reporting the first as the second is the error the
    screening report exists to prevent."""
    once = _one_day(meter="M1", date=day(0), excel_row=2)
    again = _one_day(meter="M1", date=day(0), excel_row=3,
                     values=profile(60))
    recs = flagged([once, again])
    assert il.qualified_meters(recs)[once.meter_key]["qualified"] is True
    assert recs[1].flags == (il.FLAG_DUPLICATE_CONFLICT,)
    # Two records, one block: the conflict is kept, the block is counted once.
    assert len(il.qualified_records(recs)) == 2
    assert len(il.qualified_block_keys(recs)) == 1


# ---------------------------------------------------------------------------
# End to end against the real workbook
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_workbook_end_to_end(tmp_path):
    """Read the shipped workbook and check the published contracts."""
    path = il.resolve_workbook()
    if not path.exists():                       # pragma: no cover
        pytest.skip("industry workbook not present")

    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = il.detect_sheet(wb)
    finally:
        wb.close()

    shapes = il.run(workbook=str(path), sheet=sheet,
                    out_dir=str(tmp_path), report_path=None,
                    make_figure=False)
    assert len(shapes) >= 15
    for industry in shapes.industries():
        s = shapes.shapes[industry]
        arr = np.asarray(s.shape, dtype=float)
        assert arr.shape == (N,)
        assert np.all(np.isfinite(arr))
        assert arr.max() == pytest.approx(1.0, abs=1e-9)
        assert np.all(arr >= -1e-9)

    # The two industries the audit found thin stay marked thin.
    assert shapes.shapes["电解铝"].reliable is False
    assert shapes.shapes["5G基站"].reliable is False
    # The workbook's 16 industries are all present.
    assert il.PV_INDUSTRY in shapes.shapes


@pytest.mark.slow
def test_real_workbook_qualified_subset_is_the_frozen_one():
    """The frozen screening figures, re-derived from the source workbook.

    These are the numbers the absolute-load ruling rests on, so they are
    asserted from the file rather than quoted from a report. A change to a
    tolerance, to a flag list, or to the workbook moves this test instead of
    silently moving the dataset. The two units are asserted separately and
    never as one number.
    """
    path = il.resolve_workbook()
    if not path.exists():                       # pragma: no cover
        pytest.skip("industry workbook not present")
    assert il.file_sha256(path) == (
        "63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb")

    records = il.flag_records(list(il.iter_meter_days(str(path))))

    verdicts = il.qualified_meters(records)
    checked = [v for v in verdicts.values() if v["checkable"]]
    assert len(verdicts) == 283
    assert len(checked) == 200                   # the check can reach these
    assert sum(1 for v in checked if v["qualified"]) == 87
    assert sum(1 for v in checked if not v["qualified"]) == 113
    assert len(verdicts) - len(checked) == 83    # and these it cannot

    keys = il.qualified_block_keys(records)
    assert len(keys) == 39_365                   # blocks, not records
    assert len({k[0] for k in keys}) == 84
    assert len({k[1] for k in keys}) == 730
    assert len({md.account for md in il.qualified_records(records)}) == 36
