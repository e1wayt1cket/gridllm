import unittest

import numpy as np

from library_rules import HIGH_MW, MAX_METERS, select_day


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


if __name__ == "__main__":
    unittest.main()
