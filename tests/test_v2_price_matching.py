# test_v2_price_matching.py
"""Date, granularity and missing-marking logic for the V2 price audit.

The matching question is strict: the price must be for the same actual date, be
a real traceable single-day curve, and use the same 96 x 15-minute axis. An
adjacent date or a window mean is never a substitute.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import v2_price_matching as vpm


def _price(date="2022-04-01", periods=96, minutes=15,
           convention="interval_end", units="CNY/MWh", currency="CNY",
           kind="single_day", source_kind="real_market_data",
           traceable=True):
    return vpm.PriceDay(
        source_id="test", path="test.xlsx", sheet="s", date=date,
        periods=periods, interval_minutes=minutes,
        label_convention=convention, units=units, currency=currency,
        tax_status="unknown", kind=kind, source_kind=source_kind,
        traceable=traceable)


def test_same_date_same_axis_is_exact():
    result = vpm.match_day("2022-04-01", [_price()])
    assert result["match_class"] == vpm.EXACT
    assert result["same_date"] is True
    assert result["same_granularity"] is True
    assert "period index" in result["interval_alignment"]


def test_an_adjacent_date_is_not_a_match():
    result = vpm.match_day("2022-04-01", [_price(date="2022-04-02")])
    assert result["match_class"] == vpm.MISSING
    assert "no real single-day price" in result["reason"]


def test_a_window_mean_is_not_a_daily_price():
    result = vpm.match_day(
        "2022-04-01", [_price(kind="window_mean")])
    assert result["match_class"] == vpm.UNVERIFIABLE
    assert "traceable real single-day" in result["reason"]


def test_a_synthetic_same_date_curve_is_not_a_match():
    result = vpm.match_day(
        "2022-04-01", [_price(source_kind="synthetic")])
    assert result["match_class"] == vpm.UNVERIFIABLE


def test_a_missing_period_is_partial_not_filled():
    result = vpm.match_day("2022-04-01", [_price(periods=95)])
    assert result["match_class"] == vpm.PARTIAL
    assert "95 periods" in result["reason"]


def test_a_thirty_minute_curve_is_partial_not_resampled():
    result = vpm.match_day("2022-04-01", [_price(periods=48, minutes=30)])
    assert result["match_class"] == vpm.PARTIAL
    assert "30 min" in result["reason"]


def test_an_unknown_unit_is_unverifiable():
    result = vpm.match_day(
        "2022-04-01", [_price(units="USD/MWh", currency="USD")])
    assert result["match_class"] == vpm.UNVERIFIABLE
    assert "CNY/MWh" in result["reason"]


def test_an_untraceable_same_date_curve_is_unverifiable():
    result = vpm.match_day(
        "2022-04-01", [_price(traceable=False)])
    assert result["match_class"] == vpm.UNVERIFIABLE


def test_class_counts_cover_every_row():
    rows = [
        {"match_class": vpm.EXACT},
        {"match_class": vpm.MISSING},
        {"match_class": vpm.MISSING},
    ]
    assert vpm.count_classes(rows) == {
        vpm.EXACT: 1, vpm.PARTIAL: 0, vpm.MISSING: 2,
        vpm.UNVERIFIABLE: 0,
    }
