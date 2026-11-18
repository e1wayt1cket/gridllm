"""Analytic market cases for the new day-ahead interface."""

from dataclasses import replace
import unittest

import numpy as np

from local_da_contracts import (DayAheadInput, Generator, RadialNetwork, StorageBid,
                                StorageUnit, validate_day)
from local_da_accounting import ac_replay, physical_audit, settle_day
from local_da_clearing import ClearingModel, choose_directions, clear_day


def small_day(periods=2, *, storage=True):
    units = (StorageUnit("ESS", 0, energy_mwh=1, eta_charge=1, eta_discharge=1,
                         degradation_cny_mwh=10),) if storage else ()
    bids = {unit.name: StorageBid(np.full(periods, .4), np.full(periods, .4),
                                 np.full(periods, 50.), np.full(periods, 100.)) for unit in units}
    return DayAheadInput(RadialNetwork(1, 0, (), np.array([]), np.array([])),
                         np.ones((periods, 1)), np.zeros((periods, 1)),
                         np.resize([20., 200.], periods), units, (), bids, 0, 600,
                         terminal_soc_equal=True)


class ContractTests(unittest.TestCase):
    def test_network_band_is_the_research_band(self):
        net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
        self.assertEqual((net.voltage_min, net.voltage_max), (0.93, 1.07))

    def test_slack_voltage_defaults_to_one_pu(self):
        net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
        self.assertEqual(net.slack_vm_pu, 1.)

    def test_slack_voltage_outside_the_band_is_refused(self):
        day = small_day()
        for value in [1.08, .90]:
            with self.assertRaises(ValueError):
                validate_day(replace(day, network=replace(day.network, slack_vm_pu=value)))

    def test_shunt_must_name_a_finite_bus_inside_the_network(self):
        day = small_day()
        for shunt in [(1, .5), (0.0, .5), (0, np.nan), (0, .5, 1)]:
            with self.assertRaises(ValueError):
                validate_day(replace(day, network=replace(day.network, shunts=(shunt,))))

    def test_explicit_quantity_and_price_bounds(self):
        day = small_day()
        for bid in [replace(day.bids["ESS"], charge_quantity_mw=np.full(2, .401)),
                    replace(day.bids["ESS"], charge_price_cny_mwh=np.array([0., np.nan])),
                    replace(day.bids["ESS"], discharge_price_cny_mwh=np.full(2, 601))]:
            with self.assertRaises(ValueError):
                validate_day(replace(day, bids={"ESS": bid}))

    def test_idle_direction_always_opens_one_leg(self):
        direction, idle = choose_directions({"charge": {"ESS": np.array([0., .2, 0.])},
                                             "discharge": {"ESS": np.array([0., 0., .2])}})
        np.testing.assert_array_equal(direction["ESS"], [1, 1, -1])
        np.testing.assert_array_equal(idle["ESS"], [True, False, False])

    def test_soc_and_invalid_radial_network_rejected(self):
        day = small_day()
        with self.assertRaises(ValueError):
            validate_day(replace(day, storage=(replace(day.storage[0], soc_initial=1.1),)))
        with self.assertRaises(ValueError):
            validate_day(replace(day, network=RadialNetwork(2, 0, ((1, 0),),
                                                          np.array([.01]), np.array([.02])),
                                 active_load_mw=np.ones((2, 2)), reactive_load_mvar=np.zeros((2, 2))))


class SolverTests(unittest.TestCase):
    def test_quantity_price_sign_dual_units_and_own_payoff(self):
        day = small_day()
        result = clear_day(day, seconds=15)
        self.assertTrue(result["qualified_for_settlement"], result)
        self.assertEqual(result["dispatch"]["declared_integer_variables"], 0)
        np.testing.assert_allclose(result["pricing"]["charge"]["ESS"], [.4, 0], atol=1e-6)
        np.testing.assert_allclose(result["pricing"]["discharge"]["ESS"], [0, .4], atol=1e-6)
        np.testing.assert_allclose(result["pricing"]["lmp_cny_mwh"][:, 0], [20, 200], atol=1e-6)
        expected = .25 * (20 * 1.4 + 200 * .6 + 100 * .4 - 50 * .4)
        self.assertAlmostEqual(result["pricing"]["clearing_objective"], expected, places=5)
        ledger = settle_day(day, result, inventory_price_cny_mwh=200)
        self.assertAlmostEqual(ledger["units"]["ESS"]["net_cash_cny"], 18., places=5)
        self.assertAlmostEqual(ledger["units"]["ESS"]["degradation_cny"], 2., places=5)
        self.assertAlmostEqual(ledger["units"]["ESS"]["economic_payoff_cny"], 16., places=5)
        self.assertAlmostEqual(ledger["operator_residual_cny"], 0., places=5)

    def test_explicit_complementarity_blocks_same_period_cycle(self):
        day = small_day(periods=1)
        unit = replace(day.storage[0], eta_charge=.92, eta_discharge=.92)
        bid = StorageBid(np.array([.4]), np.array([.4]), np.array([600.]), np.array([0.]))
        day = replace(day, storage=(unit,), bids={"ESS": bid})
        model = ClearingModel(day, exact=True, seconds=15)
        try:
            result = model.solve()
        finally:
            model.close()
        self.assertTrue(result["optimal"])
        self.assertLess(abs(result["charge"]["ESS"][0]), 1e-6)
        self.assertLess(abs(result["discharge"]["ESS"][0]), 1e-6)
        self.assertEqual(result["declared_binary_variables"], 0)

    def test_four_independent_units_and_zero_quantity(self):
        day = small_day()
        units = tuple(replace(day.storage[0], name=f"ESS{i}") for i in range(4))
        bids = {u.name: day.bids["ESS"] for u in units}
        bids["ESS0"] = replace(day.bids["ESS"], charge_quantity_mw=np.zeros(2),
                               discharge_quantity_mw=np.zeros(2))
        day = replace(day, storage=units, bids=bids)
        result = clear_day(day, seconds=15)
        self.assertTrue(result["qualified_for_settlement"])
        ledger = settle_day(day, result, inventory_price_cny_mwh=200)
        self.assertEqual(set(ledger["units"]), set(bids))
        self.assertAlmostEqual(ledger["units"]["ESS0"]["economic_payoff_cny"], 0., places=6)
        self.assertTrue(all(ledger["units"][f"ESS{i}"]["economic_payoff_cny"] > 0 for i in [1, 2, 3]))

    def test_cleared_slack_voltage_follows_the_declaration(self):
        day = small_day(storage=False)
        day = replace(day, network=replace(day.network, slack_vm_pu=1.04))
        result = clear_day(day, seconds=15)
        np.testing.assert_allclose(np.sqrt(result["dispatch"]["voltage_sq"]), 1.04, atol=1e-9)

    def test_infeasible_load_has_no_price_or_settlement_fallback(self):
        day = small_day(storage=False)
        day = replace(day, import_limit_mw=0)
        result = clear_day(day, seconds=15)
        self.assertFalse(result["qualified_for_settlement"])
        self.assertNotIn("pricing", result)
        with self.assertRaisesRegex(ValueError, "Unqualified"):
            settle_day(day, result, inventory_price_cny_mwh=100)

    def test_generation_is_curtailable_and_cash_is_included(self):
        day = small_day(storage=False)
        generator = Generator("PV", 0, np.full(2, 2.), offer_cny_mwh=40.)
        day = replace(day, generators=(generator,), export_limit_mw=0,
                      external_price_cny_mwh=np.full(2, 200.))
        result = clear_day(day, seconds=15)
        self.assertTrue(result["qualified_for_settlement"])
        np.testing.assert_allclose(result["pricing"]["generation"]["PV"], 1, atol=1e-6)
        ledger = settle_day(day, result, inventory_price_cny_mwh=0)
        self.assertAlmostEqual(ledger["generators"]["PV"]["curtailment_mwh"], .5, places=5)
        self.assertAlmostEqual(ledger["generator_payment_cny"], 20., places=4)
        self.assertAlmostEqual(ledger["consumer_payment_cny"], 20., places=4)
        self.assertAlmostEqual(ledger["operator_residual_cny"], 0., places=4)

    def test_initial_inventory_is_not_free_cash_profit(self):
        day = small_day(periods=1)
        bid = StorageBid(np.array([0.]), np.array([.4]), np.array([0.]), np.array([0.]))
        day = replace(day, external_price_cny_mwh=np.array([100.]),
                      terminal_soc_equal=False, bids={"ESS": bid})
        result = clear_day(day, seconds=15)
        self.assertTrue(result["qualified_for_settlement"])
        ledger = settle_day(day, result, inventory_price_cny_mwh=100.)
        unit = ledger["units"]["ESS"]
        self.assertAlmostEqual(unit["net_cash_cny"], 10., places=5)
        self.assertAlmostEqual(unit["inventory_value_cny"], -10., places=5)
        self.assertAlmostEqual(unit["economic_payoff_cny"], -1., places=5)

    def test_quantity_and_discharge_quote_change_the_award(self):
        day = small_day()
        for bid in [replace(day.bids["ESS"], charge_quantity_mw=np.full(2, .2)),
                    replace(day.bids["ESS"], discharge_price_cny_mwh=np.full(2, 600.))]:
            candidate = replace(day, bids={"ESS": bid})
            result = clear_day(candidate, seconds=15)
            self.assertTrue(result["qualified_for_settlement"], result)
            charge = result["pricing"]["charge"]["ESS"]
            expected = .2 if bid.discharge_price_cny_mwh[0] == 100 else 0
            self.assertAlmostEqual(float(charge.sum()), expected, places=5)

    def test_declared_slack_voltage_passes_both_physical_checks(self):
        day = small_day(storage=False)
        day = replace(day, network=replace(day.network, slack_vm_pu=1.02))
        result = clear_day(day, seconds=15)
        self.assertTrue(result["physical_audit"]["passed"], result["physical_audit"])
        self.assertTrue(result["ac_replay"]["passed"], result["ac_replay"])

    def test_fixed_compensation_raises_the_cleared_voltage(self):
        net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
        day = replace(small_day(storage=False), network=net,
                      active_load_mw=np.array([[0., .5], [0., .5]]),
                      reactive_load_mvar=np.array([[0., .1], [0., .1]]))
        uncompensated = clear_day(day, seconds=15)["dispatch"]
        compensated = clear_day(replace(day, network=replace(net, shunts=((1, .1),))),
                                seconds=15)["dispatch"]
        self.assertGreater(compensated["voltage_sq"][0, 1], uncompensated["voltage_sq"][0, 1])
        self.assertLess(compensated["qcc_mvar"][0], uncompensated["qcc_mvar"][0])

    def test_declared_reactive_capability_relieves_the_slack_and_holds_its_limit(self):
        net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
        day = replace(small_day(storage=False), network=net,
                      active_load_mw=np.array([[0., .5], [0., .5]]),
                      reactive_load_mvar=np.array([[0., .1], [0., .1]]))
        bare = clear_day(day, seconds=15)["dispatch"]
        generator = Generator("G", 1, np.full(2, .5), reactive_limit_mvar=.2)
        capable = clear_day(replace(day, generators=(generator,)), seconds=15)["dispatch"]
        self.assertGreater(capable["voltage_sq"][0, 1], bare["voltage_sq"][0, 1])
        self.assertLess(capable["qcc_mvar"][0], bare["qcc_mvar"][0])
        self.assertLessEqual(float(np.abs(capable["generator_q"]["G"]).max()), .2 + 1e-6)

    def test_declared_storage_energy_caps_the_throughput(self):
        day = small_day()
        tight = replace(day, storage=(replace(day.storage[0], energy_mwh=.2),))
        loose = replace(day, storage=(replace(day.storage[0], energy_mwh=2.),))
        tight_charge = float(clear_day(tight, seconds=15)["dispatch"]["charge"]["ESS"].sum())
        loose_charge = float(clear_day(loose, seconds=15)["dispatch"]["charge"]["ESS"].sum())
        self.assertLess(tight_charge, loose_charge)

    def test_native_q_network_losses_and_nonlinear_ac_replay(self):
        day = small_day(storage=False)
        net = RadialNetwork(2, 0, ((0, 1),), np.array([.01]), np.array([.02]))
        day = replace(day, network=net, active_load_mw=np.array([[0., .5], [0., .6]]),
                      reactive_load_mvar=np.array([[0., .1], [0., .15]]))
        result = clear_day(day, seconds=15)
        self.assertTrue(result["qualified_for_settlement"], result)
        self.assertGreater(result["pricing"]["pcc_mw"][0], .5)
        self.assertGreater(result["pricing"]["lmp_cny_mwh"][0, 1], 20)
        self.assertTrue(ac_replay(day, result["pricing"])["passed"])
        mutated = dict(result["pricing"])
        mutated["qcc_mvar"] = result["pricing"]["qcc_mvar"] + .1
        self.assertFalse(physical_audit(day, mutated)["passed"])


if __name__ == "__main__":
    unittest.main()
