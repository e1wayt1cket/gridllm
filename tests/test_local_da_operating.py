"""Verify remapping conservation and the operating assumptions that change feasibility."""

import unittest
from dataclasses import replace

import numpy as np

from local_da_contracts import (DayAheadInput, Generator, RadialNetwork, StorageBid,
                                StorageUnit)
from local_da_operating import OperatingModel, OperatingRules, audit_operating, remap_sources, support_day


def tiny_day(load=1., reactive=.2, import_limit=None):
    net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
    unit = StorageUnit("ESS1", 1, eta_charge=1., eta_discharge=1.)
    bid = StorageBid(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
    return DayAheadInput(net, np.array([[0., load]]), np.array([[0., reactive]]),
                         np.array([100.]), (unit,), (Generator("PV1", 1, np.array([.3])),),
                         {unit.name: bid}, 0., 600., import_limit_mw=import_limit,
                         import_emergency_mw=0.)


class OperatingTests(unittest.TestCase):
    def test_historical_aggregation_and_explicit_exclusion(self):
        active = np.array([[0., .2, .4, .1], [0., .3, .5, .1]])
        selected = [dict(source_id=f"m{i}", user_id=f"u{i}", bus_zero_based=i) for i in [1, 2, 3]]
        history = [dict(user_id=f"u{i}", meter_ids=f"m{i}", destination_bus="2",
                        candidate_status="mapped", mapping_reason="") for i in [1, 2]]
        history += [dict(user_id="u3", meter_ids="m3", destination_bus="", candidate_status="unmapped",
                         mapping_reason="Historically excluded industry")]
        result, kept, excluded = remap_sources(active, selected, history)
        np.testing.assert_allclose(result[:, 2], [.6, .8])
        np.testing.assert_allclose(result.sum(axis=1), active.sum(axis=1) - active[:, 3])
        self.assertEqual((len(kept), len(excluded)), (2, 1))

    def test_unverified_source_is_rejected(self):
        with self.assertRaises(ValueError):
            remap_sources(np.zeros((1, 2)), [dict(source_id="m", user_id="u", bus_zero_based=1)], [])

    def test_emergency_serves_matching_p_and_q_fraction(self):
        day = tiny_day(import_limit=0.)
        day = replace(day, generators=())
        rules = OperatingRules(allow_emergency_shedding=True, bounded_reactive_support=False)
        model = OperatingModel(day, rules, {}, exact=True, seconds=30)
        try:
            model.model.setObjective(model.shed_energy)
            result = model.solve()
        finally:
            model.close()
        self.assertTrue(result["optimal"])
        self.assertAlmostEqual(result["shed_mw"][0, 1], 1., places=6)
        self.assertAlmostEqual(result["shed_q_mvar"][0, 1], .2, places=6)
        self.assertTrue(audit_operating(day, rules, {}, result)["passed"])

    def test_inverter_capability_is_a_circle_not_independent_q_box(self):
        day, ratings = support_day(tiny_day(), OperatingRules())
        model = OperatingModel(day, OperatingRules(), ratings, exact=False, seconds=30)
        try:
            model.generation["PV1"][0].LB = .3
            model.model.setObjective(-model.generator_q["PV1"][0])
            result = model.solve()
        finally:
            model.close()
        self.assertTrue(result["optimal"])
        self.assertAlmostEqual(result["generator_q"]["PV1"][0], np.sqrt(.33**2 - .3**2), places=6)
        self.assertEqual(result["declared_binary_variables"], 0)

    def test_storage_q_has_its_own_rating_without_active_bid(self):
        day, ratings = support_day(tiny_day(), OperatingRules())
        model = OperatingModel(day, OperatingRules(), ratings, exact=False, seconds=30)
        try:
            model.model.setObjective(-model.generator_q["Q_ESS1"][0])
            result = model.solve()
        finally:
            model.close()
        self.assertTrue(result["optimal"])
        self.assertAlmostEqual(result["generator_q"]["Q_ESS1"][0], .44, places=6)


if __name__ == "__main__":
    unittest.main()
