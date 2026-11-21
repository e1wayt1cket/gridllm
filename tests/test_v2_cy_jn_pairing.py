# test_v2_cy_jn_pairing.py
"""The deterministic load-to-price matcher and its split."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pair_v2_cy_jn_2025 as pj


def _pair(load, price, pool, group="summer", used=None):
    used = used if used is not None else set()
    pairs, unpaired = pj.pair_group(group, [load], pool, used)
    return pairs, unpaired, used


def test_same_month_day_of_the_same_type_is_taken():
    # 2022-07-04 Monday, 2025-07-04 Friday: both weekdays.
    pairs, unpaired, used = _pair("2022-07-04",
                                  None, ["2025-07-04", "2025-07-09"])
    assert pairs[0].price_date == "2025-07-04"
    assert pairs[0].pairing_type == "same_month_day"
    assert not unpaired
    assert used == {"2025-07-04"}


def test_a_same_month_day_of_the_wrong_type_loses_to_a_same_type_date():
    # 2022-07-05 Tuesday; 2025-07-05 is a Saturday, 2025-07-08 a Tuesday.
    pairs, _unpaired, _used = _pair("2022-07-05",
                                    None, ["2025-07-05", "2025-07-08"])
    assert pairs[0].price_date == "2025-07-08"
    assert pairs[0].pairing_type == "same_season_same_day_type"


def test_type_beats_distance_even_when_the_distance_is_large():
    # No same-type date near the month-day; the distant same-type one still wins.
    pairs, _unpaired, _used = _pair("2022-07-05",
                                    None, ["2025-07-05", "2025-08-05"])
    assert pairs[0].price_date == "2025-08-05"
    assert pairs[0].pairing_type == "same_season_same_day_type"


def test_a_mismatch_is_recorded_when_no_same_type_date_is_left():
    pairs, _unpaired, _used = _pair("2022-07-05", None, ["2025-07-05"])
    assert pairs[0].price_date == "2025-07-05"
    assert pairs[0].pairing_type == "same_season_day_type_mismatch"


def test_the_holiday_window_matches_by_month_day_across_types():
    # 2022-10-04 Tuesday, 2025-10-04 Saturday.
    pairs, _unpaired, _used = _pair("2022-10-04", None, ["2025-10-04"],
                                    group="national_day_2022")
    assert pairs[0].price_date == "2025-10-04"
    assert pairs[0].pairing_type == "same_month_day"


def test_a_price_date_is_never_used_twice():
    used = set()
    loads = ["2022-07-04", "2022-07-11"]     # both Mondays
    pool = ["2025-07-04", "2025-07-07"]      # Friday, Monday
    pairs, unpaired = pj.pair_group("summer", loads, pool, used)
    assert not unpaired
    used_dates = [p.price_date for p in pairs]
    assert len(used_dates) == len(set(used_dates)) == 2


def test_a_load_day_with_no_price_left_is_unpaired_not_copied():
    pairs, unpaired = pj.pair_group(
        "summer", ["2022-07-04", "2022-07-11"], ["2025-07-04"], set())
    assert len(pairs) == 1
    assert len(unpaired) == 1


def test_the_split_is_deterministic_and_never_overlaps():
    pairs = [pj.Pair(f"2022-0{m}-{d:02d}", f"2025-0{m}-{d:02d}", group,
                     "same_month_day", "")
             for group, m in (("summer", 7), ("autumn", 10), ("winter", 12))
             for d in range(1, 11)]
    pairs += [pj.Pair(f"2022-10-0{d}", f"2025-10-0{d}", "national_day_2022",
                      "same_month_day", "")
              for d in range(3, 8)]
    first = pj.split_pairs(pairs, seed=7)
    second = pj.split_pairs(pairs, seed=7)
    assert first == second
    train, val, test = (set(first["train"]), set(first["validation"]),
                        set(first["test"]))
    assert not (train & val or train & test or val & test)
    assert len(first["holiday_block"]) == 5
    assert set(first["holiday_block"]).isdisjoint(train | val | test)
    assert len(train | val | test | set(first["holiday_block"])) == len(pairs)


def test_the_load_digest_is_stable_and_identity_sensitive():
    matrix = np.arange(96 * 33, dtype=float).reshape(96, 33)
    assert pj.load_curve_digest("2022-07-04", matrix) == \
        pj.load_curve_digest("2022-07-04", matrix.copy())
    assert pj.load_curve_digest("2022-07-04", matrix) != \
        pj.load_curve_digest("2022-07-05", matrix)
    changed = matrix.copy()
    changed[0, 0] += 1.0
    assert pj.load_curve_digest("2022-07-04", matrix) != \
        pj.load_curve_digest("2022-07-04", changed)
