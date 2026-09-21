"""The day-rotation protocol: fixed days, seeded order, no leakage.

The protocol's whole value is that a difference between two runs is a
difference in the learning. That holds only if the days are identical across
runs and the validation days never see a gradient step, so those are what these
tests pin. A protocol that quietly sampled with replacement, or let a seed move
the data, would still produce a plausible training curve.
"""

import numpy as np
import pytest

import day_rotation as dr


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def test_the_split_is_the_declared_size():
    assert dr.assert_protocol()["sizes"] == dr.SPLIT_SIZES


def test_the_split_covers_the_library_exactly_once():
    everything = [d for days in dr.SPLIT.values() for d in days]
    assert sorted(everything) == list(range(dr.LIBRARY_DAYS))
    assert len(everything) == len(set(everything))


def test_the_partitions_do_not_overlap():
    train, val, test = (set(dr.split_of(p))
                        for p in ("train", "validation", "test"))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_an_overlapping_split_is_refused(monkeypatch):
    broken = {k: list(v) for k, v in dr.SPLIT.items()}
    broken["validation"] = list(broken["train"])
    monkeypatch.setattr(dr, "SPLIT", broken)
    with pytest.raises(ValueError, match="both"):
        dr.assert_protocol()


def test_a_short_split_is_refused(monkeypatch):
    broken = {k: list(v) for k, v in dr.SPLIT.items()}
    broken["train"] = broken["train"][:19]
    monkeypatch.setattr(dr, "SPLIT", broken)
    with pytest.raises(ValueError):
        dr.assert_protocol()


def test_the_declared_split_matches_the_one_the_audit_derives():
    """The split is declared, not recomputed -- so it is checked against its source.

    It comes from `market_opportunity`'s regime labelling over the same 40-day
    ensemble. Regenerating it is expensive (it builds every day), so it is
    written out rather than derived at run time; that only stays honest if
    something fails when the two drift apart.
    """
    import market_opportunity as mo

    specs = mo.ensemble_specs(dr.LIBRARY_DAYS, seed=mo.MASTER_SEED)
    assert all(s.day_id == s.day for s in specs), \
        "the ensemble index and the library day id must be the same number"
    days = [mo.build_day(s) for s in specs]
    mo.assign_partitions(days)
    derived = {p: sorted(d.day for d in days if d.partition == p)
               for p in mo.SPLIT_NAMES}
    assert derived == {k: sorted(v) for k, v in dr.SPLIT.items()}


# ---------------------------------------------------------------------------
# The rotation
# ---------------------------------------------------------------------------

def test_a_cycle_is_the_whole_training_set():
    for cycle in range(5):
        assert sorted(dr.cycle_days(1, cycle)) == sorted(dr.split_of("train"))


def test_a_cycle_is_a_permutation_not_a_sample():
    """No repetition within a cycle: 20 days, 20 distinct entries."""
    days = dr.cycle_days(1, 0)
    assert len(days) == 20
    assert len(set(days)) == 20


def test_every_training_day_appears_once_per_cycle():
    for cycle in range(10):
        counts = {d: 0 for d in dr.split_of("train")}
        for day in dr.cycle_days(7, cycle):
            counts[day] += 1
        assert set(counts.values()) == {1}, cycle


def test_the_same_seed_and_cycle_give_the_same_order():
    assert dr.cycle_days(3, 4) == dr.cycle_days(3, 4)


def test_different_cycles_give_different_orders():
    orders = [tuple(dr.cycle_days(1, c)) for c in range(12)]
    assert len(set(orders)) > 1
    assert len(set(orders)) >= 10, "the order barely changes between cycles"


def test_the_order_is_independent_of_the_training_random_stream():
    """Drawing exploration noise must not move a single day.

    The rotation runs on a stream of its own, tagged per `(run_seed, cycle)`,
    so how many random numbers the training loop consumed before asking for the
    schedule cannot change the schedule.
    """
    first = dr.cycle_days(2, 3)
    for _ in range(1000):
        np.random.rand()
    assert dr.cycle_days(2, 3) == first


def test_the_schedule_is_independent_of_the_order_it_is_asked_in():
    """Cycle k's order does not depend on cycles before it having run."""
    forward = [dr.cycle_days(5, c) for c in range(6)]
    backward = [dr.cycle_days(5, c) for c in reversed(range(6))]
    assert forward == list(reversed(backward))


# ---------------------------------------------------------------------------
# Seeds move the order, never the data
# ---------------------------------------------------------------------------

def test_different_seeds_move_the_order():
    orders = {tuple(dr.cycle_days(seed, 0)) for seed in range(1, 9)}
    assert len(orders) > 1


def test_every_seed_sees_exactly_the_same_days():
    """The property that makes a seed-to-seed difference a learning difference.

    If a seed could change which days exist, or how many times a day is seen
    per cycle, then a difference between two runs would be partly a difference
    in the data and the comparison would not mean what it is read as meaning.
    """
    reference = None
    for seed in (1, 2, 3, 17, 99):
        seen = []
        for cycle in range(6):
            seen.extend(dr.cycle_days(seed, cycle))
        counts = {d: seen.count(d) for d in dr.split_of("train")}
        current = tuple(sorted(counts.items()))
        if reference is None:
            reference = current
        assert current == reference, f"seed {seed} saw a different day mix"


def test_the_validation_and_test_sets_never_enter_a_training_cycle():
    """Leakage here would make the validation curve meaningless."""
    forbidden = set(dr.validation_days()) | set(dr.test_days())
    for seed in (1, 2, 3):
        for cycle in range(8):
            assert not (set(dr.cycle_days(seed, cycle)) & forbidden), \
                f"seed {seed} cycle {cycle} trained on a held-out day"


def test_the_seed_does_not_enter_the_split():
    """`day_id` is a library index, not a function of the run seed."""
    assert dr.split_of("train") == dr.split_of("train")
    assert dr.SPLIT is dr.SPLIT
    assert set(dr.split_of("train")) == set(dr.cycle_days(123456, 0))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_has_one_episode_per_day_per_cycle():
    plan = dr.episode_plan(1, max_cycles=40)
    assert len(plan) == 40 * 20
    assert [p["episode"] for p in plan] == list(range(len(plan)))


def test_the_plan_covers_every_training_day_once_per_cycle():
    plan = dr.episode_plan(4, max_cycles=6)
    for cycle in range(6):
        days = [p["day_id"] for p in plan if p["cycle"] == cycle]
        assert sorted(days) == sorted(dr.split_of("train"))


def test_the_plan_only_names_training_days():
    plan = dr.episode_plan(9, max_cycles=40)
    assert {p["split"] for p in plan} == {"train"}
    assert set(p["day_id"] for p in plan) == set(dr.split_of("train"))


def test_the_plan_is_reproducible_and_seed_tagged():
    a = dr.episode_plan(11, max_cycles=3)
    b = dr.episode_plan(11, max_cycles=3)
    c = dr.episode_plan(12, max_cycles=3)
    assert a == b
    assert a != c
    assert {p["run_seed"] for p in a} == {11}


def test_a_negative_cycle_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        dr.cycle_days(1, -1)


# ---------------------------------------------------------------------------
# The protocol has to be realizable: a day_id must actually price a day
# ---------------------------------------------------------------------------

def _arm(day_id: int):
    """Build the native day priced on one library day, through the real path."""
    from config_loader import reload_defaults, reload_scenarios
    from scenarios import get_scenario, typical_day_config
    import wholesale_price as wp

    reload_defaults()
    reload_scenarios()
    wp.reset_cache()
    config = typical_day_config(96)
    agents, wholesale = get_scenario("typical_day", 96, config, day_id=day_id)
    return agents, np.asarray(wholesale, dtype=float)


def test_a_chosen_day_id_actually_prices_that_day():
    """The whole protocol rests on this, so it is asserted rather than assumed."""
    import wholesale_price as wp

    provider = wp.SyntheticWholesalePriceProvider()
    for day in (0, 1, 7, 39):
        _, wholesale = _arm(day)
        assert np.allclose(wholesale, provider.day_ahead_price(day)), day


def test_every_training_day_is_priced_differently():
    """Twenty days that all billed the same price would be one day seen twice."""
    prices = {tuple(np.round(_arm(d)[1], 6)) for d in dr.split_of("train")}
    assert len(prices) == len(dr.split_of("train"))


def test_the_same_day_id_is_repeatable_across_builds():
    _, first = _arm(11)
    _, second = _arm(11)
    assert np.array_equal(first, second)


def test_only_the_price_moves_between_days_not_the_network():
    """A day is a price day; the distribution network is frozen."""
    a0, _ = _arm(0)
    a1, _ = _arm(1)
    assert [x.name for x in a0] == [x.name for x in a1]
    for x, y in zip(a0, a1):
        assert np.array_equal(x.load_forecast, y.load_forecast), x.name
        assert np.array_equal(x.pv_forecast, y.pv_forecast), x.name


def test_first_energy_cannot_depend_on_the_run_seed():
    """`day_id` is a library index; no run seed may enter what a day contains."""
    import wholesale_price as wp

    provider = wp.SyntheticWholesalePriceProvider()
    for day in dr.split_of("train")[:4]:
        _, wholesale = _arm(day)
        assert np.allclose(wholesale, provider.day_ahead_price(day))


def test_build_rung_agents_would_silently_price_the_wrong_day():
    """The trap this protocol has to avoid, pinned so it cannot come back.

    `build_rung_agents` calls `reload_defaults()` internally, which resets
    `wholesale_market.day_id` to the file's value. A run that set the day
    *before* calling it would train on the same day for all 800 episodes while
    reporting a 40-day protocol -- and nothing about the resulting curve would
    look wrong. The correct entry point is `get_scenario(..., day_id=...)`,
    which sets the day after any reload.
    """
    from config_loader import get_default, reload_defaults, set_default
    from baseline_ladder import PRIMARY_BAND, build_rung_agents

    reload_defaults()
    set_default("wholesale_market.day_id", 33)
    day_before = get_default("wholesale_market.day_id")
    build_rung_agents("B3", PRIMARY_BAND, 1)
    assert day_before == 33
    assert get_default("wholesale_market.day_id") != 33, (
        "build_rung_agents no longer resets the day lever; the note in "
        "day_rotation and the training protocol need revisiting")
