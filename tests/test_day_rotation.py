# test_day_rotation.py
"""The episode scheduler, which knows nothing about how many days there are.

The scheduler takes the days that exist and decides an order. What this suite
pins is that it stays that way: a permutation rather than a sample, an order
that depends on the seed and not on the data, a schedule that never invents a
day it was not given, and no length baked in for a dataset to be made to fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import day_rotation as dr

THREE = ["2021-01-01", "2021-06-15", "2022-03-09"]


# ---------------------------------------------------------------------------
# What it is given
# ---------------------------------------------------------------------------

def test_an_empty_day_list_is_refused():
    """A scheduler that accepts no days runs zero episodes and looks like a run
    that finished."""
    with pytest.raises(ValueError, match="no days were supplied"):
        dr.assert_days([])


def test_a_repeated_day_is_refused():
    """A duplicated day would be visited twice, so a cycle would not be a
    permutation and a per-cycle comparison would compare day mixes."""
    with pytest.raises(ValueError, match="repeats"):
        dr.assert_days(["d1", "d2", "d1"])


def test_a_blank_day_is_refused():
    with pytest.raises(ValueError, match="contains None"):
        dr.assert_days(["d1", None])


# ---------------------------------------------------------------------------
# A cycle is a permutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 7, 12345])
def test_a_cycle_visits_every_supplied_day_exactly_once(seed):
    order = dr.cycle_days(THREE, seed, 0)
    assert sorted(order) == sorted(THREE)
    assert len(order) == len(THREE)


def test_fewer_days_than_any_expected_length_still_works():
    """Three days is three days. Nothing pads the list to a size it was not
    given, reuses a day to fill a cycle, or walks off the end."""
    for cycle in range(7):
        order = dr.cycle_days(THREE, 3, cycle)
        assert sorted(order) == sorted(THREE)
    plan = dr.episode_plan(THREE, 3, max_cycles=7)
    assert len(plan) == 7 * 3
    assert {p["day"] for p in plan} == set(THREE)


def test_one_day_is_a_valid_schedule():
    assert dr.cycle_days(["only"], 1, 0) == ["only"]
    assert dr.is_cycle_end(["only"], 0) is True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_the_same_seed_and_cycle_give_the_same_order():
    assert dr.cycle_days(THREE, 5, 2) == dr.cycle_days(THREE, 5, 2)


def test_a_cycle_does_not_depend_on_the_cycles_before_it():
    """The seed is stateless, so a run resumed at cycle 9 has the order cycle 9
    would have had."""
    first = dr.cycle_days(THREE, 5, 9)
    for _ in range(3):
        dr.cycle_days(THREE, 5, 0)
    assert dr.cycle_days(THREE, 5, 9) == first


def test_different_seeds_are_not_the_same_schedule():
    orders = {tuple(dr.cycle_days(list(range(20)), s, 0)) for s in range(8)}
    assert len(orders) > 1


def test_the_seed_stream_is_separated_from_the_training_seed():
    assert dr.cycle_seed(1, 0) != dr.cycle_seed(0, 1)
    assert dr.cycle_seed(1, 0) == dr.cycle_seed(1, 0)


# ---------------------------------------------------------------------------
# The episode plan
# ---------------------------------------------------------------------------

def test_episode_day_places_each_episode_in_a_cycle():
    days = list(range(4))
    for episode in range(9):
        slot = dr.episode_day(days, 2, episode)
        assert slot["episode"] == episode
        assert slot["cycle"] == episode // 4
        assert slot["position"] == episode % 4
        assert slot["day"] in days
        assert slot["days_per_cycle"] == 4


def test_a_negative_episode_or_cycle_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        dr.episode_day(THREE, 1, -1)
    with pytest.raises(ValueError, match="non-negative"):
        dr.cycle_days(THREE, 1, -1)


def test_cycle_end_fires_once_per_cycle():
    days = list(range(4))
    ends = [ep for ep in range(12) if dr.is_cycle_end(days, ep)]
    assert ends == [3, 7, 11]


def test_the_plan_is_the_episodes_in_order():
    plan = dr.episode_plan(THREE, 4, max_cycles=2)
    assert [p["episode"] for p in plan] == list(range(6))
    assert [p["cycle"] for p in plan] == [0, 0, 0, 1, 1, 1]
    assert sorted(p["day"] for p in plan[:3]) == sorted(THREE)


# ---------------------------------------------------------------------------
# No fixed dataset length
# ---------------------------------------------------------------------------

def test_the_module_holds_no_split_and_no_dataset_length():
    """The names that used to carry a 40-day split are gone, not merely unused.
    A constant left in place is one a later caller reaches for."""
    for gone in ("LIBRARY_DAYS", "SPLIT_SIZES", "SPLIT", "assert_protocol",
                 "split_of", "validation_days", "test_days"):
        assert not hasattr(dr, gone), gone


def test_the_scheduler_works_for_any_length():
    for n in (1, 2, 5, 37, 400):
        days = [f"day{i}" for i in range(n)]
        assert len(dr.cycle_days(days, 1, 0)) == n
