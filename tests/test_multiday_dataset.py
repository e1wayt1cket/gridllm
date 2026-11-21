# test_multiday_dataset.py
"""The calendar labels and price-pairing rules, without the workbook.

These are the decisions the multi-day manifest depends on. They are pure, so
the failure modes that matter are testable here: a make-up workday is not a
holiday, a same-month-day pair is not necessarily the same day type, and one
real curve may not price two load days.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import multiday_dataset as md


def test_national_day_and_its_makeup_workdays_are_distinct():
    holiday = md.calendar_label("2022-10-01")
    assert holiday["holiday_class"] == "national_day"
    assert holiday["holiday_offset"] == 0
    assert holiday["is_workday"] is False
    assert holiday["day_type"] == "holiday"

    for day in ("2022-10-08", "2022-10-09"):
        makeup = md.calendar_label(day)
        assert makeup["holiday_class"] == "national_day"
        assert makeup["holiday_offset"] is None
        assert makeup["is_workday"] is True
        assert makeup["day_type"] == "weekday"


def test_spring_festival_offset_counts_from_new_years_day():
    assert md.calendar_label("2022-01-31")["holiday_offset"] == -1
    assert md.calendar_label("2022-02-01")["holiday_offset"] == 0
    assert md.calendar_label("2022-02-06")["holiday_offset"] == 5


def test_season_weekday_and_weekend_are_independent_of_holidays():
    winter = md.calendar_label("2022-12-31")
    assert winter["season"] == "winter"
    assert winter["weekday"] == "Saturday"
    assert winter["is_weekend"] is True
    assert winter["day_type"] == "weekend"

    autumn = md.calendar_label("2022-09-30")
    assert autumn["season"] == "autumn"
    assert autumn["weekday"] == "Friday"
    assert autumn["is_workday"] is True
    assert autumn["day_type"] == "weekday"


def _curve(day, curve_id="261009", temporal="single_day"):
    return md.PriceCurve(
        price_date=md.as_date(day), curve_id=curve_id, source="test",
        curve_sha256="sha-" + curve_id, temporal_statistic=temporal)


def test_same_month_day_across_years_flags_a_day_type_mismatch():
    result = md.pair_prices(["2022-10-09"], [_curve("2026-10-09")])
    pair = result["pairs"][0]
    assert pair["price_date"] == "2026-10-09"
    assert pair["price_pair_rule"] == "same_month_day_across_years"
    assert pair["price_year_gap"] == 4
    # 2022-10-09 is a National Day make-up workday; 2026-10-09 is ordinary.
    assert pair["pair_quality"] == "same_month_day_day_type_mismatch"


def test_an_exact_date_beats_every_other_rule():
    result = md.pair_prices(["2026-10-09"], [_curve("2026-10-09")])
    assert result["pairs"][0]["price_pair_rule"] == "exact_date"
    assert result["pairs"][0]["pair_quality"] == "exact"


def test_a_curve_is_never_reused_for_a_second_load_day():
    result = md.pair_prices(
        ["2022-10-09", "2023-10-09"], [_curve("2026-10-09")])
    first, second = result["pairs"]
    assert first["price_date"] == "2026-10-09"
    assert second["price_date"] == ""
    assert second["pair_reason"] == "price_curve_reuse_conflict"


def test_window_means_are_rejected_not_relabelled():
    result = md.pair_prices(
        ["2022-10-09"], [_curve("2026-10-09", temporal="window_mean")])
    assert result["pairs"][0]["pair_reason"] == "no_single_day_curve"
    assert result["rejected_curves"][0]["reason"] == "not_a_single_day_curve"


def test_holiday_relative_rule_matches_by_class_and_offset():
    result = md.pair_prices(
        ["2022-10-02"], [_curve("2021-10-03")])
    pair = result["pairs"][0]
    assert pair["price_pair_rule"] == "holiday_relative"
    assert pair["pair_quality"] == "holiday_relative"


def test_weekday_fallback_is_limited_to_seven_days():
    result = md.pair_prices(["2022-10-11"], [_curve("2022-10-18")])
    assert result["pairs"][0]["price_pair_rule"] == "weekday_fallback"
    outside = md.pair_prices(["2022-10-20"], [_curve("2022-10-12")])
    assert outside["pairs"][0]["price_date"] == ""
    assert outside["pairs"][0]["pair_reason"] == "no_single_day_curve"
