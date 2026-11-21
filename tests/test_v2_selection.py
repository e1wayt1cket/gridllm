# test_v2_selection.py
"""The V2 coverage core and representative-day rules, without the workbook.

These tests pin the decisions that would otherwise be buried in a long CSV:
the seasonal-coverage core is a prefix intersection, ordinary representatives
are central rather than extreme, and holiday representatives are grouped by the
real holiday instance and relative position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_v2_user_date_coverage as avc
import select_v2_seasonal_subsets as sss
import select_v2_typical_days as svd


def _meta(bus, industry="x", in_v1=False):
    return {"bus": bus, "industry": industry, "in_v1": in_v1,
            "n_meters": 1, "meters": {"M" + str(bus)}}


def test_season_counts_cover_all_four():
    counts = avc._season_counts([
        "2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"])
    assert counts == {"spring": 1, "summer": 1, "autumn": 1, "winter": 1}


def test_curve_metrics_use_the_quarter_hour_slot():
    curve = np.zeros(96)
    curve[0] = 2.0
    curve[1] = 4.0
    metrics = avc._curve_metrics(curve)
    assert metrics["peak_mw"] == pytest.approx(4.0)
    assert metrics["mean_mw"] == pytest.approx(6.0 / 96)
    assert metrics["energy_mwh"] == pytest.approx(6.0 * 0.25)


def test_provisional_core_is_the_largest_prefix_that_keeps_the_seasons():
    users = {"A": _meta(1), "B": _meta(2), "C": _meta(3)}
    dates_a = ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"]
    dates_b = dates_a + ["2022-01-02"]
    rows = []
    for date in dates_a:
        rows.append({"user_id": "A", "date": date, "all_meters_complete": True})
    for date in dates_b:
        rows.append({"user_id": "B", "date": date, "all_meters_complete": True})
    rows.append({"user_id": "C", "date": "2022-01-01",
                 "all_meters_complete": True})
    core, common, ranked, ladder = avc.provisional_core(
        users, rows, min_season_dates=1)
    assert ranked == ["B", "A", "C"]
    assert core == ["B", "A"]
    assert common == set(dates_a)
    assert ladder[-2]["k"] == 2
    assert ladder[-1]["k"] == 3
    assert ladder[-1]["min_season_dates"] == 0


def test_ordinary_representatives_are_near_the_stratum_median():
    rows = []
    for i, (peak, energy) in enumerate([
            (1.0, 10.0), (2.0, 20.0), (3.0, 30.0),
            (4.0, 40.0), (5.0, 50.0)]):
        rows.append({
            "date": f"2022-04-{i + 1:02d}", "season": "spring",
            "day_type": "weekday", "core_peak_mw": peak,
            "core_energy_mwh": energy, "holiday_class": "ordinary",
            "holiday_name": "", "holiday_offset": "",
        })
    chosen = svd._stratum_representatives(rows, "spring", "weekday", limit=3)
    assert len(chosen) == 3
    assert "2022-04-03" in {r["date"] for r in chosen}
    assert all(r["representative"] for r in chosen)


def test_holiday_representatives_use_actual_year_and_offset():
    rows = []
    for year, base in (("2021", "2021-02-11"), ("2022", "2022-01-31")):
        for offset in (-1, 0, 1):
            rows.append({
                "date": base, "season": "winter", "day_type": "holiday",
                "holiday_class": "spring_festival", "holiday_name":
                "Spring Festival", "holiday_offset": offset,
                "core_peak_mw": 1.0, "core_energy_mwh": 1.0,
            })
    chosen = svd._holiday_representatives(rows, "winter", limit=2)
    reasons = " ".join(r["selection_reason"] for r in chosen)
    assert "2021" in reasons and "2022" in reasons
    assert len(chosen) == 4


def test_seasonal_greedy_keeps_common_dates_and_respects_the_load_cap():
    users = {"A": _meta(1), "B": _meta(2), "C": _meta(3)}
    dates = ["2022-04-01", "2022-04-02"]
    data = {}
    for user, value in (("A", 1.0), ("B", 2.0), ("C", 3.0)):
        data[(user, "spring")] = {
            "dates": set(dates),
            "curves": {d: np.ones(96) * value for d in dates},
        }
    result = sss.select_group(
        users, data, "spring", "season", min_user_days=1,
        min_common_days=2, max_ratio=1.0)
    assert result["_chosen"] == ["A", "B"]
    assert result["_dates"] == dates
    assert result["users"] == 2
    assert result["median_peak_mw"] == pytest.approx(3.0)


def test_subset_metrics_sum_the_fixed_users_on_covered_days():
    users = ["A", "B"]
    meta = {"A": _meta(1, "ind_a"), "B": _meta(2, "ind_b")}
    complete = {"A": {"2022-04-01", "2022-04-02"},
                "B": {"2022-04-01"}}
    curves = {
        "2022-04-01|A": np.ones(96),
        "2022-04-01|B": np.ones(96) * 2,
        "2022-04-02|A": np.ones(96),
    }
    reps = [{"date": "2022-04-01", "season": "spring"},
            {"date": "2022-04-02", "season": "spring"}]
    result = __import__("analyze_v2_subset_feasibility")._subset_metrics(
        "AB", users, meta, complete, reps, curves)
    assert result["reps_complete"] == 1
    assert result["median_peak_mw"] == pytest.approx(3.0)
    assert result["median_energy_mwh"] == pytest.approx(3.0 * 96 * 0.25)
    assert result["top_bus"] == 2
    assert result["max_bus_energy_share"] == pytest.approx(2.0 / 3.0)
