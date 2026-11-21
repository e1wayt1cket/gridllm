import unittest

import numpy as np

from library_rules import (HIGH_MW, MAX_METERS, POWER_FACTOR, REACTIVE_RATIO,
                           assign_buses, draw_split, native_p0,
                           reactive_from_active, select_day)


def flat(peak):
    return np.full(96, float(peak))


class SelectDayTest(unittest.TestCase):
    def test_caps_selection_at_the_load_bus_count(self):
        curves = {f"s{i:02d}": flat(0.05) for i in range(40)}
        users = {k: f"u{k}" for k in curves}
        sel = select_day("2021-07-01", curves, users)
        self.assertEqual(len(sel.sources), MAX_METERS)
        self.assertTrue(sel.hit_bus_ceiling)
        self.assertEqual(sel.status, "OUT_OF_BAND")
        self.assertEqual(sel.reason, "no_swap_below_band")

    def test_upward_swap_keeps_the_user_set(self):
        curves = {"a1": flat(1.0), "a2": flat(2.4), "b1": flat(1.0)}
        users = {"a1": "A", "a2": "A", "b1": "B"}
        sel = select_day("2021-07-02", curves, users)
        self.assertEqual(set(sel.users), {"A", "B"})
        self.assertIn("a2", sel.sources)
        self.assertNotIn("a1", sel.sources)
        self.assertAlmostEqual(sel.peak_mw, 3.4, places=6)
        self.assertEqual(sel.status, "IN_BAND")
        self.assertEqual(sel.reason, "raised_in_band")

    def test_initial_band_is_reported_without_raising(self):
        sel = select_day("2021-07-03", {"a": flat(2.0), "b": flat(1.5)}, {"a": "A", "b": "B"})
        self.assertEqual(sel.reason, "initial_in_band")
        self.assertEqual(sel.status, "IN_BAND")

    def test_oversized_meter_leaves_nothing_selectable(self):
        sel = select_day("2021-07-04", {"a": flat(HIGH_MW + 0.1)}, {"a": "A"})
        self.assertEqual(sel.sources, ())
        self.assertEqual(sel.reason, "nothing_selectable")
        self.assertEqual(sel.status, "OUT_OF_BAND")

    def test_zero_peak_meter_is_not_the_smallest(self):
        curves = {"a1": flat(0.0), "a2": flat(2.0), "b": flat(1.5)}
        sel = select_day("2021-07-05", curves, {"a1": "A", "a2": "A", "b": "B"})
        self.assertIn("a2", sel.sources)
        self.assertNotIn("a1", sel.sources)

    def test_selection_is_deterministic(self):
        curves = {"a": flat(2.0), "b": flat(1.5), "c": flat(1.4)}
        users = {"a": "A", "b": "B", "c": "C"}
        self.assertEqual(select_day("2021-07-06", curves, users),
                         select_day("2021-07-06", curves, users))


class AssignBusesTest(unittest.TestCase):
    def setUp(self):
        self.p0 = native_p0()

    def test_assignment_uses_distinct_target_buses(self):
        pi = {f"m{i}": 0.2 for i in range(5)}
        a = assign_buses(pi, self.p0)
        self.assertEqual(len(set(a.bus_of_source.values())), 5)
        self.assertTrue(all(b in a.target_share for b in a.bus_of_source.values()))

    def test_target_max_share_equals_native_at_full_coverage(self):
        pi = {f"m{i}": 1.0 for i in range(32)}
        a = assign_buses(pi, self.p0)
        self.assertAlmostEqual(a.target_max_share, 0.1131, places=3)
        self.assertAlmostEqual(a.excluded_native_share, 0.0, places=9)

    def test_partial_coverage_records_excluded_share(self):
        pi = {f"m{i}": 1.0 for i in range(28)}
        a = assign_buses(pi, self.p0)
        self.assertGreater(a.target_max_share, 0.1131)
        self.assertAlmostEqual(a.excluded_native_share, 0.0606, places=3)

    def test_swap_iteration_never_increases_deviation(self):
        pi = {"big": 0.30, "mid": 0.25, "small": 0.20, "tiny": 0.15, "dot": 0.10}
        a = assign_buses(pi, self.p0)
        self.assertLessEqual(a.d_final, a.d_init)

    def test_reactive_ratio_holds_and_zero_stays_zero(self):
        active = np.zeros((2, 33))
        active[0, 1] = 1.234
        reactive = reactive_from_active(active)
        self.assertAlmostEqual(reactive[0, 1] / active[0, 1], REACTIVE_RATIO, places=9)
        self.assertEqual(reactive[0, 0], 0.0)
        self.assertTrue((reactive[0, 2:] == 0.0).all())

    def test_declared_ratio_matches_the_power_factor(self):
        self.assertAlmostEqual(np.tan(np.arccos(POWER_FACTOR)), REACTIVE_RATIO, places=3)


class DrawSplitTest(unittest.TestCase):
    def _dates(self, months, n, year=2022):
        """Round-robin the months so every generated day number stays valid."""
        return [f"{year}-{months[i % len(months)]:02d}-{i // len(months) + 1:02d}"
                for i in range(n)]

    def _prices(self):
        return {f"{m:02d}": tuple(f"2025-{m:02d}-{d:02d}" for d in range(1, 32))
                for m in range(6, 13)}

    def test_126_days_give_90_18_18(self):
        plan = draw_split(self._dates([6, 7, 8, 9, 11, 12], 126), self._prices())
        self.assertEqual(plan.library_size, 126)
        self.assertEqual(plan.counts, {"train": 90, "validation": 18, "test": 18, "holiday": 0})

    def test_holiday_block_is_the_national_day_window(self):
        plan = draw_split(self._dates([10], 31), self._prices())
        self.assertEqual(plan.blocks["holiday"],
                         tuple(f"2022-10-0{d}" for d in range(1, 6)))

    def test_month_over_price_count_draws_and_records_discards(self):
        prices = {"06": tuple(f"2025-06-{d:02d}" for d in range(1, 15))}
        plan = draw_split(self._dates([6], 30), prices)
        self.assertEqual(plan.library_size, 14)
        self.assertEqual(len(plan.discarded["06"]), 16)
        self.assertEqual(sum(plan.counts.values()), 14)

    def test_blocks_are_disjoint_by_date_and_by_price(self):
        plan = draw_split(self._dates([6, 7, 8, 9, 10, 11, 12], 140), self._prices())
        loads = [p.split("|")[0] for pair in plan.pairs.values() for p in pair]
        prices_used = [p.split("|")[1] for pair in plan.pairs.values() for p in pair]
        self.assertEqual(len(set(loads)), len(loads))
        self.assertEqual(len(set(prices_used)), len(prices_used))
        self.assertEqual(len(loads), plan.library_size)

    def test_same_seed_gives_the_same_split(self):
        dates = self._dates([9], 20)
        self.assertEqual(draw_split(dates, self._prices()),
                         draw_split(dates, self._prices()))


if __name__ == "__main__":
    unittest.main()
