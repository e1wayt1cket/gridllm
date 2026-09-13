# tests/test_participant_payoff.py
"""Each participant's settled payoff, and where it came from.

The clearing objective is not a payoff: it values a storage unit at that unit's
own declared bid and offer, while settlement happens at the nodal price. These
tests pin the settlement view, and in particular that a storage unit cannot earn
a consumption value it has no basis for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types

import numpy as np
import pytest

import money
import participant_payoff as pp
from models import MarketConfig, StorageSpec


def _config(cycle_cost=100.0, penalty=2500.0, terminal_value=None):
    cfg = MarketConfig(verbose=False)
    cfg.storage.cycle_cost = cycle_cost
    cfg.storage.terminal_value = terminal_value
    cfg.market_design.penalty_unserved = penalty
    return cfg


def _storage_agent(bid_value=0.0, offer_cost=0.0, e_max=1.0,
                   participant_type="storage"):
    """A battery alone: zero load, zero on-site generation."""
    zeros = np.zeros(2)
    return types.SimpleNamespace(
        name="ESS", bus=3, participant_type=participant_type,
        load_forecast=zeros, load_real=zeros,
        pv_forecast=zeros, pv_real=zeros, wind_forecast=None, wind_real=None,
        bid_value=bid_value, offer_cost=offer_cost,
        storage=StorageSpec(e_max=e_max, p_ch_max=1.0, p_dis_max=2.0,
                            eta_ch=0.95, eta_dis=0.95, soc0=0.5,
                            soc_min=0.1, soc_max=0.9))


def _storage_sched():
    """Charge 1 MW in period 0, discharge 2 MW in period 1."""
    return {
        "p_ch": np.array([1.0, 0.0]), "p_dis": np.array([0.0, 2.0]),
        "p_buy": np.array([1.0, 0.0]), "p_sell": np.array([0.0, 2.0]),
        "served": np.zeros(2), "unserved": np.zeros(2),
        "pv_used": np.zeros(2), "wind_used": np.zeros(2),
    }


# --------------------------------------------------------------------------
# Independent storage: the first case study
# --------------------------------------------------------------------------
def test_independent_storage_payoff_is_revenue_minus_cost_minus_degradation():
    """Profit = sales revenue - purchase cost - degradation, at the nodal price."""
    cfg = _config(cycle_cost=100.0)
    agent = _storage_agent()
    b = pp.participant_payoff(_storage_sched(), np.array([50.0, 300.0]),
                              agent, cfg)
    # 2 MW sold at 300 for a quarter hour, 1 MW bought at 50, degradation on
    # the 1 + 2 MW of throughput.
    assert b.revenue == pytest.approx(2.0 * 300.0 * 0.25)
    assert b.purchase_cost == pytest.approx(1.0 * 50.0 * 0.25)
    assert b.degradation_cost == pytest.approx(100.0 * 3.0 * 0.25)
    assert b.generation_cost == 0.0
    assert b.penalty == 0.0
    assert b.other_cost == 0.0
    assert b.total == pytest.approx(150.0 - 12.5 - 75.0)


def test_independent_storage_payoff_ignores_the_declared_bid():
    """A battery must not earn the clearing objective's charge-bid credit.

    The objective credits an agent for declaring a high charge bid, but the unit
    is settled at the nodal price, so that credit is not money. A payoff that
    read bid_value or offer_cost would hand the unit a consumption value it has
    no basis for, which is exactly the error that made storage churn.
    """
    cfg = _config()
    sched = _storage_sched()
    lmp = np.array([50.0, 300.0])
    truthful = pp.participant_payoff(sched, lmp, _storage_agent(0.0, 0.0),
                                     cfg)
    shaded = pp.participant_payoff(
        sched, lmp, _storage_agent(720.0, 300.0), cfg)
    assert shaded.total == pytest.approx(truthful.total)
    assert shaded.revenue == pytest.approx(truthful.revenue)
    assert shaded.purchase_cost == pytest.approx(truthful.purchase_cost)


def test_independent_storage_reports_energy_and_equivalent_cycles():
    cfg = _config()
    b = pp.participant_payoff(_storage_sched(), np.array([50.0, 300.0]),
                              _storage_agent(e_max=1.0), cfg)
    assert b.energy_bought_mwh == pytest.approx(1.0 * 0.25)
    assert b.energy_sold_mwh == pytest.approx(2.0 * 0.25)
    assert b.charge_mwh == pytest.approx(0.25)
    assert b.discharge_mwh == pytest.approx(0.5)
    # One full cycle moves 2 * e_max through the battery.
    assert b.equivalent_cycles == pytest.approx((0.25 + 0.5) / (2.0 * 1.0))


def test_equivalent_cycles_is_zero_without_a_battery():
    a = _storage_agent(participant_type="prosumer")
    a.storage = None
    b = pp.participant_payoff(_storage_sched(), np.zeros(2), a, _config())
    assert b.equivalent_cycles == 0.0


# --------------------------------------------------------------------------
# Prosumer: the general ledger
# --------------------------------------------------------------------------
def test_prosumer_payoff_values_served_load_and_charges_generation():
    cfg = _config(cycle_cost=100.0, penalty=2500.0)
    zeros = np.zeros(1)
    agent = types.SimpleNamespace(
        name="P", bus=0, participant_type="prosumer",
        load_forecast=np.ones(1), load_real=np.ones(1),
        pv_forecast=zeros, pv_real=zeros, wind_forecast=None, wind_real=None,
        bid_value=400.0, offer_cost=100.0,
        storage=StorageSpec(e_max=1.0, p_ch_max=1.0, p_dis_max=1.0,
                            eta_ch=1.0, eta_dis=1.0, soc0=0.5,
                            soc_min=0.0, soc_max=1.0))
    sched = {"p_ch": np.array([0.0]), "p_dis": np.array([1.0]),
             "p_buy": np.array([0.0]), "p_sell": np.array([1.0]),
             "served": np.array([2.0]), "unserved": np.array([1.0]),
             "pv_used": np.array([3.0]), "wind_used": np.array([1.0])}
    b = pp.participant_payoff(sched, np.array([200.0]), agent, cfg)
    assert b.revenue == pytest.approx(1.0 * 200.0 * 0.25 + 400.0 * 2.0 * 0.25)
    assert b.purchase_cost == 0.0
    assert b.generation_cost == pytest.approx(100.0 * 4.0 * 0.25)
    assert b.degradation_cost == pytest.approx(100.0 * 1.0 * 0.25)
    assert b.penalty == pytest.approx(2500.0 * 1.0 * 0.25)
    assert b.total == pytest.approx(
        (50.0 + 200.0) - 0.0 - 100.0 - 25.0 - 625.0)


# --------------------------------------------------------------------------
# Dispatch and the declared stub
# --------------------------------------------------------------------------
def test_payoff_model_dispatch_follows_participant_type():
    assert isinstance(pp.payoff_model_for(_storage_agent()),
                      pp.IndependentStoragePayoff)
    assert isinstance(pp.payoff_model_for(_storage_agent(
        participant_type="prosumer")), pp.ProsumerPayoff)


def test_payoff_model_defaults_to_prosumer_for_a_legacy_agent():
    """An agent built before participant_type existed is a general site."""
    legacy = _storage_agent()
    del legacy.participant_type
    assert isinstance(pp.payoff_model_for(legacy), pp.ProsumerPayoff)


def test_retailer_payoff_is_a_declared_stub():
    retailer = _storage_agent(participant_type="retailer")
    with pytest.raises(NotImplementedError, match="retail tariff"):
        pp.payoff_model_for(retailer).payoff(_storage_sched(), np.zeros(2),
                                             retailer, _config())


def test_unknown_participant_type_is_rejected():
    with pytest.raises(ValueError, match="unknown participant_type"):
        pp.payoff_model_for(_storage_agent(participant_type="aggregator"))


# --------------------------------------------------------------------------
# Terminal state of charge
# --------------------------------------------------------------------------
def test_soc_neutral_adjustment_is_zero_at_the_starting_soc():
    assert pp.soc_neutral_adjustment(0.5, 0.5, 2.0, 400.0) == pytest.approx(0.0)


def test_soc_neutral_adjustment_values_the_residual_energy():
    # Ending 0.2 of a 2 MWh battery fuller is 0.4 MWh of unsold energy.
    assert pp.soc_neutral_adjustment(0.7, 0.5, 2.0, 400.0) \
        == pytest.approx(0.4 * 400.0)
    # Running it down is a charge against the arm that did so.
    assert pp.soc_neutral_adjustment(0.3, 0.5, 2.0, 400.0) \
        == pytest.approx(-0.4 * 400.0)


def test_terminal_adjustment_stays_out_of_the_decomposition():
    """The components must sum to the cash profit, not to the adjusted one."""
    b = pp.PayoffBreakdown(revenue=100.0, purchase_cost=20.0,
                           degradation_cost=5.0, terminal_adjustment=-30.0)
    assert b.total == pytest.approx(75.0)
    assert b.soc_neutral_total == pytest.approx(45.0)
    d = b.as_dict()
    assert d["total"] == pytest.approx(75.0)


def test_terminal_price_falls_back_to_the_mean_wholesale():
    cfg = _config(terminal_value=None)
    assert pp.resolve_terminal_price(cfg, np.array([100.0, 300.0])) \
        == pytest.approx(200.0)
    assert pp.resolve_terminal_price(_config(terminal_value=123.0)) \
        == pytest.approx(123.0)
    with pytest.raises(ValueError, match="no terminal price"):
        pp.resolve_terminal_price(_config(terminal_value=None), None)


def test_breakdown_is_money_not_per_period_power():
    """The period length is applied exactly once, inside the ledger."""
    cfg = _config(cycle_cost=0.0)
    b = pp.participant_payoff(_storage_sched(), np.full(2, 100.0),
                              _storage_agent(), cfg)
    # 2 MWh sold and 1 MWh bought at 100 CNY/MWh.
    assert b.revenue == pytest.approx(2.0 * 100.0 * money.DT_HOURS)
    assert b.total == pytest.approx((2.0 - 1.0) * 100.0 * money.DT_HOURS)
