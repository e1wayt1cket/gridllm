# tests/test_dianyicha_price.py
"""Contracts of the 电易电查 price reader and its provider.

Two things about this data are easy to get wrong in ways that leave every
number looking plausible, so they are pinned here rather than left to review:

* **The slot axis is interval-END.** A label is the close of its interval, so
  the clock runs `00:15` to `24:00` and label `i` belongs to period `i`. Reading
  an interval-START file through this reader would rotate every price by one
  period and still produce a curve of the right length in the right units.
  The reader must refuse it, not rotate it.
* **Every curve is a mean.** The workbook's own headers say so. No curve may
  be described, labelled or consumed as a measured day, and the reader has to
  carry that forward rather than leave it to whoever writes the report.

Most tests build a throwaway workbook so the failure paths can be exercised
without touching the shipped file; the real read is marked slow.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt

import numpy as np
import pytest

import dianyicha_price as dp

N = dp.N_SLOTS


# ---------------------------------------------------------------------------
# Throwaway workbooks
# ---------------------------------------------------------------------------

def _end_labels():
    """00:15 .. 23:45 as times, then 24:00 as Excel's epoch-midnight datetime."""
    labels = [(dt.datetime(2000, 1, 1)
               + dt.timedelta(minutes=15 * (i + 1))).time()
              for i in range(N - 1)]
    labels.append(dt.datetime(1900, 1, 1, 0, 0))
    return labels


def _start_labels():
    """00:00 .. 23:45 — the interval-START convention, which must be refused."""
    return [(dt.datetime(2000, 1, 1) + dt.timedelta(minutes=15 * i)).time()
            for i in range(N)]


def write_workbook(path, sheets):
    """`sheets` maps sheet name -> (labels, prices)."""
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, (labels, prices) in sheets.items():
        ws = wb.create_sheet(name)
        ws.cell(row=1, column=1, value="时段")
        ws.cell(row=1, column=2, value="日前均价")
        for i, (lab, price) in enumerate(zip(labels, prices), start=2):
            ws.cell(row=i, column=1, value=lab)
            if price is not None:
                ws.cell(row=i, column=2, value=price)
    wb.save(path)
    wb.close()
    return path


def prices(n=N):
    return [100.0 + i for i in range(n)]


@pytest.fixture
def good_book(tmp_path):
    return write_workbook(
        tmp_path / "good.xlsx",
        {"261009": (_end_labels(), prices()),
         "7days": (_end_labels(), [p + 1 for p in prices()]),
         "30days": (_end_labels(), [p + 2 for p in prices()]),
         "halfyear": (_end_labels(), [p + 3 for p in prices()])})


# ---------------------------------------------------------------------------
# Slot axis
# ---------------------------------------------------------------------------

def test_end_labels_map_positionally_onto_periods():
    """Label i closes period i; nothing is shifted by a quarter hour."""
    labels = _end_labels()
    minutes = [dp.label_minutes(c, i) for i, c in enumerate(labels)]
    assert minutes[0] == 15          # 00:15 closes period 0
    assert minutes[1] == 30
    assert minutes[-1] == 24 * 60    # 24:00 closes period 95
    assert minutes == [15 * (i + 1) for i in range(N)]
    dp._check_slot_axis(labels)      # must not raise


def test_interval_start_labels_are_refused_not_rotated(tmp_path):
    """A 00:00-start file must raise rather than be read as if it were ours."""
    path = write_workbook(tmp_path / "start.xlsx",
                          {"261009": (_start_labels(), prices())})
    with pytest.raises(ValueError, match="interval-END"):
        dp.read_curves(str(path), ("261009",))


def test_a_single_shifted_label_is_caught(tmp_path):
    labels = _end_labels()
    labels[10] = dt.time(2, 45)      # should be 02:45 -> shifted to 02:45? no:
    labels[10] = dt.time(2, 30)      # duplicate of slot 9's 02:30
    path = write_workbook(tmp_path / "shift.xlsx",
                          {"261009": (labels, prices())})
    with pytest.raises(ValueError):
        dp.read_curves(str(path), ("261009",))


def test_midnight_datetime_outside_the_last_slot_is_refused():
    with pytest.raises(ValueError, match="only.*last slot|end-of-day"):
        dp.label_minutes(dt.datetime(1900, 1, 1, 0, 0), 5)


def test_non_midnight_datetime_is_refused():
    with pytest.raises(ValueError, match="midnight"):
        dp.label_minutes(dt.datetime(2021, 5, 1, 3, 0), N - 1)


def test_non_time_label_is_refused():
    with pytest.raises(ValueError, match="not a time"):
        dp.label_minutes("00:15", 0)


def test_wrong_row_count_is_refused(tmp_path):
    path = write_workbook(tmp_path / "short.xlsx",
                          {"261009": (_end_labels()[:90], prices(90))})
    with pytest.raises(ValueError, match="96 data rows"):
        dp.read_curves(str(path), ("261009",))


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def test_blank_price_is_refused(tmp_path):
    p = prices()
    p[7] = None
    path = write_workbook(tmp_path / "blank.xlsx",
                          {"261009": (_end_labels(), p)})
    with pytest.raises(ValueError, match="blank"):
        dp.read_curves(str(path), ("261009",))


def test_non_numeric_price_is_refused(tmp_path):
    p = prices()
    p[3] = "n/a"
    path = write_workbook(tmp_path / "bad.xlsx",
                          {"261009": (_end_labels(), p)})
    with pytest.raises(ValueError, match="not a number"):
        dp.read_curves(str(path), ("261009",))


def test_missing_sheet_is_refused(good_book):
    with pytest.raises(KeyError, match="not in"):
        dp.read_curves(str(good_book), ("261009", "nosuchsheet"))


def test_values_are_read_unchanged(good_book):
    """The reader reports the file; it does not clip, scale or clean."""
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    assert c.prices[0] == pytest.approx(100.0)
    assert c.prices[-1] == pytest.approx(100.0 + N - 1)
    assert len(c.prices) == N


def test_stats_report_mean_and_extremes(good_book):
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    s = c.stats()
    arr = np.asarray(c.prices, dtype=float)
    assert s["min_cny_mwh"] == pytest.approx(arr.min())
    assert s["max_cny_mwh"] == pytest.approx(arr.max())
    assert s["mean_cny_mwh"] == pytest.approx(arr.mean())
    assert s["trough_slot"] == 0 and s["peak_slot"] == N - 1


# ---------------------------------------------------------------------------
# Every curve is a mean
# ---------------------------------------------------------------------------

def test_no_curve_claims_to_be_measured(good_book):
    curves = dp.read_curves(str(good_book), ("261009", "7days"))
    assert curves
    for name, c in curves.items():
        assert c.is_measured is False, name
        assert "mean" in c.role, name
        assert c.provenance["is_measured"] is False


def test_describe_makes_no_measured_day_claim(good_book):
    payload = dp.describe(dp.read_curves(str(good_book), ("261009",)))
    assert payload["contains_measured_daily_series"] is False
    assert "mean" in payload["note"] or "means" in payload["note"]
    for info in payload["curves"].values():
        assert info["is_measured"] is False


def test_provenance_carries_the_raw_header_label(good_book):
    """The sheet's own wording is preserved, not normalised away."""
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    assert c.provenance["header_label"] == "日前均价"
    assert c.provenance["units"] == "CNY/MWh"
    assert c.provenance["interval_convention"] == "label_is_interval_end"


# ---------------------------------------------------------------------------
# Window contract
# ---------------------------------------------------------------------------

def test_window_is_a_slice_of_the_whole_day(good_book):
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    np.testing.assert_allclose(c.window(40, 8), c.array()[40:48])


def test_window_beyond_the_day_is_refused(good_book):
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    with pytest.raises(ValueError, match="does not fit"):
        c.window(92, 8)


def test_full_day_window_is_the_whole_curve(good_book):
    c = dp.read_curves(str(good_book), ("261009",))["261009"]
    np.testing.assert_allclose(c.window(0, N), c.array())


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------

def test_provider_is_registered_under_its_name():
    import wholesale_price as wp
    assert "dianyicha" in wp.SOURCES
    assert wp.DianyichaPriceProvider.name == "dianyicha"
    assert issubclass(wp.DianyichaPriceProvider, wp.WholesalePriceProvider)


def test_provider_serves_the_curves_in_configured_order():
    import wholesale_price as wp
    wp.reset_cache()
    provider = wp.get_provider("dianyicha")
    for day_id, sheet in enumerate(dp.SHEETS):
        assert provider.curve_for(day_id).sheet == sheet


def test_provider_refuses_a_day_id_outside_the_declared_curves():
    """A stray day id must fail, not silently serve the first curve."""
    import wholesale_price as wp
    provider = wp.get_provider("dianyicha")
    with pytest.raises(ValueError, match="not a declared dianyicha curve"):
        provider.get_day_ahead_price(len(dp.SHEETS))


def test_provider_has_no_negative_day_id():
    import wholesale_price as wp
    provider = wp.get_provider("dianyicha")
    with pytest.raises(ValueError):
        provider.get_day_ahead_price(-1)


def test_provider_ignores_the_seed():
    """A recorded curve has no draw for a seed to override."""
    import wholesale_price as wp
    provider = wp.get_provider("dianyicha")
    np.testing.assert_allclose(provider.get_day_ahead_price(0, seed=1),
                               provider.get_day_ahead_price(0, seed=2))


def test_provider_window_matches_the_underlying_curve():
    import wholesale_price as wp
    provider = wp.get_provider("dianyicha")
    full = provider.get_day_ahead_price(0)
    np.testing.assert_allclose(provider.get_day_ahead_price(0, T=16, t_start=32),
                               full[32:48])


def test_provider_describe_flags_every_curve_as_a_mean():
    import wholesale_price as wp
    payload = wp.get_provider("dianyicha").describe()
    assert payload["kind"] == "recorded_curve"
    assert payload["contains_measured_daily_series"] is False
    assert all(not v["is_measured"] for v in payload["curves"].values())


def test_existing_sources_still_resolve():
    """Adding a source must not disturb the ones already there."""
    import wholesale_price as wp
    wp.reset_cache()
    assert wp.get_provider("synthetic_v1").name == "synthetic_v1"
    assert wp.get_provider("library").name == "library"
    assert wp.get_provider("legacy").name == "legacy"
    with pytest.raises(NotImplementedError):
        wp.get_provider("guangdong")


# ---------------------------------------------------------------------------
# The shipped workbook
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metadata: what each curve is, as opposed to what it is called
# ---------------------------------------------------------------------------

def test_every_curve_is_recorded_as_real_market_derived():
    """All four come from real historical market data; they differ by
    aggregation, not by provenance."""
    for name, c in dp.read_curves().items():
        assert c.source_kind == "real_market", name
        assert c.provenance["source_kind"] == "real_market", name


def test_the_trailing_curves_name_their_window():
    curves = dp.read_curves()
    assert curves["7days"].aggregation == "trailing_mean"
    assert curves["7days"].mean_window_days == 7
    assert curves["30days"].mean_window_days == 30
    assert curves["halfyear"].mean_window_days == 180


def test_the_same_day_curve_is_recorded_as_unresolved_not_as_a_day():
    """The ambiguity is carried, not resolved by assumption.

    Its sheet name embeds a date and its header says 均值. This module does not
    decide which it is; it records both readings and marks the aggregation
    unresolved, so no consumer can treat it as a confirmed daily series.
    """
    c = dp.read_curves()["261009"]
    assert c.aggregation == "unresolved"
    assert c.aggregation != "single_day"
    assert c.mean_window_days is None
    assert c.reference_period == "26/10/09"
    readings = c.provenance["aggregation_readings"]
    assert "26/10/09" in readings and "mean" in readings


def test_no_curve_is_promoted_to_measured():
    """`is_measured` means *confirmed* single-day, so nothing here sets it."""
    for name, c in dp.read_curves().items():
        assert c.is_measured is False, name
        assert c.provenance["is_measured"] is False, name
    payload = dp.describe(dp.read_curves())
    assert payload["contains_confirmed_daily_series"] is False
    assert payload["contains_measured_daily_series"] is False
    assert payload["source_kind"] == "real_market"
    assert set(payload["aggregations_present"]) == {"trailing_mean",
                                                    "unresolved"}


def test_the_trailing_windows_end_dates_are_left_unverified():
    """Inventing them would be worse than leaving them blank."""
    for name, c in dp.read_curves().items():
        assert c.window_end is None, name
        assert c.provenance["window_start"] is None, name
        assert c.provenance["window_end"] is None, name


def test_the_header_is_carried_verbatim_alongside_the_classification():
    """The workbook's own wording survives; the classification sits beside it."""
    curves = dp.read_curves()
    assert curves["261009"].provenance["header_label"] == "日前均值（CNY/MWh）"
    assert curves["7days"].provenance["header_label"] == "日前均价"
    payload = dp.describe(curves)
    assert "均值" in payload["curves"]["261009"]["header_label"]
    assert payload["curves"]["7days"]["aggregation"] == "trailing_mean"


def test_the_workbook_producer_properties_are_recorded():
    """The only producer-side metadata the file carries -- and the reason the
    window dates are absent: it is a hand-assembled extraction."""
    prov = dp.read_curves()["261009"].provenance
    assert prov["workbook_author"]
    assert prov["workbook_created"]
    assert prov["workbook_modified"]


@pytest.mark.slow
def test_real_workbook_reads_all_four_curves():
    path = dp.resolve_workbook()
    if not path.exists():                       # pragma: no cover
        pytest.skip("price workbook not present")

    curves = dp.read_curves(str(path))
    assert set(curves) == set(dp.SHEETS)
    for name, c in curves.items():
        arr = c.array()
        assert arr.shape == (N,)
        assert np.all(np.isfinite(arr))
        assert c.is_measured is False
        assert c.role == dp.ROLES[name]

    # Values measured independently from the source file.
    expected = {
        "261009": (225.90, 445.00, 386.83),
        "7days": (257.50, 477.87, 395.55),
        "30days": (333.08, 450.86, 407.82),
        "halfyear": (284.93, 393.86, 346.33),
    }
    for name, (lo, hi, mean) in expected.items():
        s = curves[name].stats()
        assert s["min_cny_mwh"] == pytest.approx(lo, abs=0.01)
        assert s["max_cny_mwh"] == pytest.approx(hi, abs=0.01)
        assert s["mean_cny_mwh"] == pytest.approx(mean, abs=0.01)

    # The raw header wording survives, including the 均值 on 261009.
    assert "均值" in curves["261009"].provenance["header_label"]
    assert curves["261009"].provenance["workbook_sha256"]
