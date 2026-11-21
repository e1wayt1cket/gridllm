"""The screen's verdict must follow the day's own band and block, never a constant."""

from types import SimpleNamespace
import unittest

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np

import screen_library


def day(low, high):
    return SimpleNamespace(network=SimpleNamespace(voltage_min=low, voltage_max=high))


def result(volts, optimal=True):
    return {"dispatch": {"optimal": optimal, "solver_status": 2 if optimal else 3,
                         "solution_count": 1 if optimal else 0, "runtime_seconds": 1.5,
                         "voltage_sq": np.full((4, 2), volts ** 2),
                         "pcc_mw": np.array([1., 1.2]), "clearing_objective": 10.},
            "price_qualification": "SMOOTH_PASS", "qualified_for_settlement": optimal}


class RowTests(unittest.TestCase):
    def test_band_comes_from_the_day_not_a_fixed_constant(self):
        summary = screen_library.row("2021-06-04", "train", "2025-06-02",
                                     day(0.93, 1.07), result(1.06))
        self.assertTrue(summary["voltage_band_ok"])
        self.assertTrue(summary["feasible"])

    def test_voltage_above_the_declared_band_is_infeasible(self):
        summary = screen_library.row("2021-06-04", "train", "2025-06-02",
                                     day(0.93, 1.07), result(1.08))
        self.assertFalse(summary["voltage_band_ok"])
        self.assertFalse(summary["feasible"])
        self.assertTrue(summary["voltage_at_upper_bound"])

    def test_a_failed_solve_is_infeasible_whatever_the_voltage(self):
        summary = screen_library.row("2021-06-04", "train", "2025-06-02",
                                     day(0.93, 1.07), result(1.0, optimal=False))
        self.assertFalse(summary["feasible"])
        self.assertEqual(summary["solver_status"], 3)


class CountTests(unittest.TestCase):
    def test_counts_stay_per_block(self):
        rows = [screen_library.row("a", "train", "p", day(0.93, 1.07), result(1.0)),
                screen_library.row("b", "test", "q", day(0.93, 1.07), result(1.08))]
        counts = screen_library.block_counts(rows)
        self.assertEqual(counts["train"]["feasible"], 1)
        self.assertEqual(counts["test"]["feasible"], 0)
        self.assertEqual(counts["validation"]["days"], 0)


if __name__ == "__main__":
    unittest.main()
