"""Check the fixed candidate capacity and consistent active-market withdrawal."""

import unittest
from dataclasses import replace

import numpy as np

from local_da_candidate import (EXTERNAL_Q_NAME, candidate_model, price_candidate,
                                stress_cases)
from local_da_contracts import DayAheadInput, Generator, RadialNetwork, StorageBid, StorageUnit
from local_da_operating import OperatingRules, support_day


def candidate_input():
    net = RadialNetwork(33, 0, tuple((0, b) for b in range(1, 33)), np.zeros(32), np.zeros(32))
    units = tuple(StorageUnit(f"ESS{b}", b) for b in [9, 14, 17, 27])
    bids = {u.name: StorageBid(np.full(2, .4), np.full(2, .4), np.zeros(2), np.zeros(2)) for u in units}
    day = DayAheadInput(net, np.ones((2, 33)) * .01, np.ones((2, 33)) * .005,
                        np.full(2, 100.), units, (), bids, 0., 600.)
    rules = OperatingRules()
    day, ratings = support_day(day, rules)
    day = replace(day, generators=day.generators + (Generator(EXTERNAL_Q_NAME, 32, np.zeros(2), reactive_limit_mvar=.5),))
    return day, rules, ratings


class CandidateTests(unittest.TestCase):
    def test_eight_cases_preserve_load_and_candidate_resources(self):
        day, _, _ = candidate_input()
        cases = dict(stress_cases(day))
        self.assertEqual(len(cases), 8)
        for case in cases.values():
            np.testing.assert_array_equal(case.active_load_mw, day.active_load_mw)
            np.testing.assert_array_equal(case.reactive_load_mvar, day.reactive_load_mvar)
            self.assertIs(case.generators, day.generators)
        self.assertTrue(cases["terminal_soc_equal"].terminal_soc_equal)
        self.assertFalse(day.terminal_soc_equal)
        self.assertTrue((day.bids["ESS9"].charge_quantity_mw == .4).all())

    def test_zero_active_quantity_retains_identical_reactive_obligation(self):
        day, rules, ratings = candidate_input()
        withdrawn = dict(stress_cases(day))["all_zero_quantity"]
        model = candidate_model(withdrawn, rules, ratings, exact=False)
        try:
            self.assertEqual(model.charge["ESS9"][0].UB, 0.)
            self.assertAlmostEqual(model.generator_q["Q_ESS9"][0].UB, .44)
            self.assertAlmostEqual(model.generator_q["Q_ESS9"][0].LB, -.44)
            self.assertEqual(model.generator_q[EXTERNAL_Q_NAME][0].LB, 0.)
            self.assertEqual(model.generator_q[EXTERNAL_Q_NAME][0].UB, .5)
        finally:
            model.close()

    def test_external_q_cannot_absorb_or_exceed_half_mvar(self):
        day, rules, ratings = candidate_input()
        model = candidate_model(day, rules, ratings, exact=False)
        try:
            model.model.setObjective(-model.generator_q[EXTERNAL_Q_NAME][0])
            result = model.solve()
            self.assertTrue(result["optimal"])
            self.assertAlmostEqual(result["generator_q"][EXTERNAL_Q_NAME][0], .5, places=7)
            model.model.setObjective(model.generator_q[EXTERNAL_Q_NAME][0])
            result = model.solve()
            self.assertAlmostEqual(result["generator_q"][EXTERNAL_Q_NAME][0], 0., places=7)
        finally:
            model.close()

    def test_emergency_dispatch_cannot_enter_candidate_pricing(self):
        day, rules, ratings = candidate_input()
        result = price_candidate(day, replace(rules, allow_emergency_shedding=True), ratings,
                                  {"optimal": True})
        self.assertFalse(result["qualified_for_settlement"])
        self.assertEqual(result["price_qualification"], "NOT_TESTED")


if __name__ == "__main__":
    unittest.main()
