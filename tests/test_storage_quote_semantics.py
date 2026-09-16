"""What a storage unit's quotes mean, and what they must not mean.

A battery alone declares two reservation prices in CNY/MWh. A load declares a
multiplier on its own willingness to pay. The two ontologies are selected by
participant type and never by whether an agent happens to carry a battery, so a
prosumer's on-site battery keeps the declared prices its load and generation
give it. These tests pin the selection, the truthful point, and the refusals
that stop a caller from silently getting the wrong one.
"""

import types

import numpy as np
import pytest

import money
from dispatch_core import quote_arrays, check_storage_quote_signs
from models import MarketConfig, StorageSpec
from rl_spec import (
    ACTION_BID_OFFER_V1,
    ACTION_STORAGE_QUOTE_V2,
    action_spec_for,
)

T = 4


def _config(cycle_cost=100.0):
    config = MarketConfig(opf_mode="socp")
    config.storage.cycle_cost = cycle_cost
    return config


def _agent(bid_value=0.0, offer_cost=0.0, participant_type="storage",
           with_storage=True):
    return types.SimpleNamespace(
        name="ESS", bus=3, bid_value=bid_value, offer_cost=offer_cost,
        participant_type=participant_type,
        storage=StorageSpec(e_max=1.0, p_ch_max=1.0, p_dis_max=1.0,
                            eta_ch=0.92, eta_dis=0.92, soc0=0.5,
                            soc_min=0.1, soc_max=0.9) if with_storage else None)


# ---------------------------------------------------------------------------
# The two ontologies
# ---------------------------------------------------------------------------
def test_a_storage_agent_declares_absolute_prices():
    bid, offer = quote_arrays(_agent(), {"ESS": {"charge_price": 20.0,
                                          "discharge_price": 90.0}},
                              T, _config())
    np.testing.assert_allclose(bid, np.full(T, 20.0))
    np.testing.assert_allclose(offer, np.full(T, 90.0))


def test_a_storage_agent_can_declare_per_period_prices():
    bid, offer = quote_arrays(
        _agent(),
        {"ESS": {"charge_price": np.array([1.0, 2.0, 3.0, 4.0]),
                  "discharge_price": np.array([9.0, 8.0, 7.0, 6.0])}},
        T, _config())
    np.testing.assert_allclose(bid, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(offer, [9.0, 8.0, 7.0, 6.0])


def test_an_absent_declaration_is_the_truthful_one():
    """No premium named is no premium asked for."""
    bid, offer = quote_arrays(_agent(), {}, T, _config())
    assert np.all(bid == 0.0) and np.all(offer == 0.0)
    assert list(ACTION_STORAGE_QUOTE_V2.truthful_array()) == [0.0, 0.0]


def test_a_prosumer_with_a_battery_keeps_the_multiplier_ontology():
    """Its declared prices are meaningful for it, and are not replaced.

    Selecting on `a.storage is not None` instead of on participant type would
    sweep every prosumer battery into the storage ontology and take away the
    willingness to pay its load actually has.
    """
    agent = _agent(bid_value=600.0, offer_cost=200.0,
                   participant_type="prosumer")
    bid, offer = quote_arrays(agent, {"ESS": {"bid_mult": 1.5, "offer_adder": 30.0}},
                              T, _config())
    np.testing.assert_allclose(bid, np.full(T, 900.0))
    np.testing.assert_allclose(offer, np.full(T, 230.0))


def test_a_load_agent_is_unaffected():
    bid, offer = quote_arrays(_agent(bid_value=580.0, participant_type="prosumer",
                                     with_storage=False),
                              {"ESS": {"bid_mult": 1.0, "offer_adder": 0.0}}, T,
                              _config())
    np.testing.assert_allclose(bid, np.full(T, 580.0))
    np.testing.assert_allclose(offer, np.full(T, 0.0))


# ---------------------------------------------------------------------------
# Refusals: getting the wrong ontology must be loud
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("legacy", [
    {"bid_mult": 1.5},
    {"offer_adder": 10.0},
    {"bid_mult": 1.5, "offer_adder": 10.0},
])
def test_the_retired_pair_is_rejected_for_a_storage_agent(legacy):
    """A caller still shading a multiplier would otherwise get truthful quotes
    and never learn its action had stopped doing anything."""
    with pytest.raises(KeyError, match="bid_mult|offer_adder"):
        quote_arrays(_agent(), {"ESS": legacy}, T, _config())


def test_a_partial_declaration_is_rejected():
    with pytest.raises(KeyError, match="both"):
        quote_arrays(_agent(), {"ESS": {"charge_price": 10.0}}, T, _config())


def test_a_quote_below_the_exactness_floor_is_rejected():
    """The decomposition of charge and discharge into one net flow is exact
    only while both objective coefficients stay negative.

    Two ways to lose that. A negative quote turns one coefficient positive
    outright. And a zero quote with no degradation charge leaves it at zero, so
    the auxiliary is no longer pinned to the net flow and the reported charge
    and discharge may exceed it -- the degeneracy that would let a unit's state
    of charge drift from the flow the balance sees.
    """
    assert check_storage_quote_signs(0.0, 0.0, 100.0, "ok") is None
    with pytest.raises(ValueError, match="non-negative"):
        check_storage_quote_signs(-1.0, 0.0, 100.0, "negative")
    with pytest.raises(ValueError, match="strictly negative"):
        check_storage_quote_signs(0.0, 0.0, 0.0, "no premium, no degradation")
    with pytest.raises(ValueError, match="strictly negative"):
        check_storage_quote_signs(0.0, 0.0, -50.0, "negative degradation")


# ---------------------------------------------------------------------------
# Spec selection agrees with the solver's selection
# ---------------------------------------------------------------------------
def test_spec_selection_matches_the_participant_type():
    assert action_spec_for(_agent()) is ACTION_STORAGE_QUOTE_V2
    assert action_spec_for(
        _agent(participant_type="prosumer")) is ACTION_BID_OFFER_V1


def test_the_truthful_point_is_inside_every_registered_action_space():
    from rl_spec import ACTION_SPECS
    for spec in ACTION_SPECS.values():
        assert set(spec.truthful) == set(spec.action_names)
        for name in spec.action_names:
            lo, hi = spec.bounds[name]
            assert lo <= spec.truthful[name] <= hi
