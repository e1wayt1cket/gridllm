"""Independent analytic checks for the unified market contract."""

from dataclasses import replace
import unittest

import numpy as np

from da_market_data import MarketRules, zero_bids
from da_market_pricing import qualify_prices
from da_market_model import MarketModel, audit
from da_market_conic import ConicMarketModel
from da_market_cutting import CuttingMarketModel
from da_market import clear_market_day
from local_da_contracts import DayAheadInput, RadialNetwork, StorageUnit, StorageBid


def analytic_day():
    unit=StorageUnit("ESS",0,energy_mwh=1.,eta_charge=1.,eta_discharge=1.)
    bid=StorageBid(np.full(2,.4),np.full(2,.4),np.full(2,50.),np.full(2,100.))
    return DayAheadInput(RadialNetwork(1,0,(),np.array([]),np.array([])),np.ones((2,1)),
                         np.zeros((2,1)),np.array([20.,200.]),(unit,),(),{"ESS":bid},0,600,
                         terminal_soc_equal=True)


class MarketModelTests(unittest.TestCase):
    def test_public_clearing_settlement_and_invalid_input(self):
        day=analytic_day()
        rules=MarketRules(external_q_bus=0,cone_form='rotated')
        result=clear_market_day(day,{},rules=rules)
        self.assertEqual(result['status'],'SETTLED',result)
        self.assertEqual(result['price_qualification'],'SMOOTH_PASS')
        self.assertAlmostEqual(result['settlement']['units']['ESS']['net_cash_cny'],18.,places=4)
        self.assertAlmostEqual(result['settlement']['units']['ESS']['economic_payoff_cny'],-2.,places=4)
        invalid=replace(day,bids={'ESS':replace(day.bids['ESS'],charge_quantity_mw=np.full(2,.5))})
        self.assertEqual(clear_market_day(invalid,{},rules=rules)['status'],'INPUT_INVALID')

    def test_cutting_cone_price_matches_loss_derivative(self):
        day=replace(analytic_day(),network=RadialNetwork(2,0,((0,1),),np.array([.01]),np.array([.02])),
                    active_load_mw=np.array([[0.,.3],[0.,.4]]),reactive_load_mvar=np.array([[0.,.1],[0.,.15]]),
                    storage=(),bids={})
        model=CuttingMarketModel(day,{},MarketRules(external_q_bus=1,cone_form='rotated'),directions={})
        try:
            result=model.solve(prices=True)
        finally:
            model.close()
        self.assertTrue(result['optimal'],result)
        self.assertTrue(audit(day,{},MarketRules(external_q_bus=1),result)['passed'])
        self.assertLess(result['cone_distance_max'],1e-10)

    def test_native_conic_prices_and_rhs_update(self):
        day=analytic_day()
        rules=MarketRules(external_q_bus=0)
        model=ConicMarketModel(day,{},rules,directions={"ESS":np.array([1,-1])})
        try:
            result=model.solve(prices=True)
            self.assertTrue(result["optimal"],result)
            self.assertTrue(audit(day,{},rules,result)["passed"])
            np.testing.assert_allclose(result["lmp_cny_mwh"][:,0],[20,200],atol=1e-5)
            model.balance[0,0].RHS+=1e-4
            changed=model.solve()
            self.assertAlmostEqual((changed["clearing_objective"]-result["clearing_objective"])/(.25e-4),20.,places=3)
        finally:
            model.close()

    def test_currency_scaling_preserves_objective_and_lmp(self):
        day=analytic_day()
        for scale in [1.,100.]:
            rules=MarketRules(external_q_bus=0,objective_scale=scale)
            model=MarketModel(day,{},rules,exact=False,directions={"ESS":np.array([1,-1])})
            try:
                result=model.solve(prices=True)
            finally:
                model.close()
            self.assertTrue(result["optimal"])
            np.testing.assert_allclose(result["lmp_cny_mwh"][:,0],[20,200],atol=1e-5)
            self.assertAlmostEqual(result["clearing_objective"],42.,places=5)
            self.assertTrue(audit(day,{},rules,result)["passed"])

    def test_standard_cone_matches_rotated_form_and_ac(self):
        source=analytic_day()
        day=replace(source,network=RadialNetwork(2,0,((0,1),),np.array([.01]),np.array([.02])),
                    active_load_mw=np.array([[0.,.3],[0.,.4]]),
                    reactive_load_mvar=np.array([[0.,.1],[0.,.15]]),storage=(),bids={})
        results=[]
        for form in ["rotated","standard"]:
            rules=MarketRules(external_q_bus=1,cone_form=form)
            model=MarketModel(day,{},rules,exact=False)
            try:
                result=model.solve(prices=True)
            finally:
                model.close()
            self.assertTrue(result["optimal"])
            self.assertTrue(audit(day,{},rules,result)["passed"])
            results.append(result)
        np.testing.assert_allclose(results[0]["pcc_mw"],results[1]["pcc_mw"],atol=1e-6)
        for result in results:
            for t,(p,q) in enumerate(zip(day.active_load_mw[:,1],day.reactive_load_mvar[:,1])):
                r,x=.01,.02
                z=r*r+x*x
                current=(1-2*(r*p+x*q)-np.sqrt((1-2*(r*p+x*q))**2-4*z*(p*p+q*q)))/(2*z)
                derivative=1+r*2*(p+r*current)/(1-2*(r*p+x*q)-2*z*current)
                expected=day.external_price_cny_mwh[t]*derivative
                self.assertLess(abs(result["lmp_cny_mwh"][t,1]-expected),max(.1,abs(expected)*.01))

    def test_qualify_prices_records_the_backend_it_ran(self):
        day=analytic_day()
        rules=MarketRules(external_q_bus=0,cone_form='rotated')
        model=MarketModel(day,{},rules,exact=False)
        try:
            dispatch=model.solve()
        finally:
            model.close()
        result=qualify_prices(day,{},rules,dispatch,backend='cutting')
        self.assertEqual(result['pricing_backend'],'cutting')
        self.assertEqual(result['price_qualification'],'SMOOTH_PASS')
        with self.assertRaises(ValueError):
            qualify_prices(day,{},rules,dispatch,backend='native')

    def test_zero_quantity_preserves_endpoint_soc(self):
        day=zero_bids(analytic_day())
        rules=MarketRules(external_q_bus=0)
        model=MarketModel(day,{},rules)
        try:
            result=model.solve()
        finally:
            model.close()
        self.assertTrue(result["optimal"])
        np.testing.assert_allclose(result["energy"]["ESS"],.5,atol=1e-7)
        # Exclusivity is an explicit disjunction over the declared quantities, so
        # the dispatch declares one binary per unit and period.
        self.assertGreater(result["declared_binary_variables"],0)
        with self.assertRaises(ValueError):
            rules.validate(replace(day,terminal_soc_equal=False))


if __name__=="__main__":
    unittest.main()
