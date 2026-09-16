# tests/test_nash_action_space.py
"""The best-response search is bounded by each agent's own action space.

`nash.py` used to branch on `agent.is_prosumer` and read the bounds off
`config.market_design.{bid_mult,offer_adder}_range`, which meant a storage unit
was searched over a multiplier it no longer declares. The search is now driven
by the agent's spec, and these tests hold it there without running an OPF.
"""

import numpy as np
import pytest

from models import MarketConfig
from nash import NashEquilibriumTester, _generate_variations
from rl_spec import ACTION_BID_OFFER_V1, ACTION_STORAGE_QUOTE_V2


class _Agent:
    """Only the surface the search reads: a name and a participant type."""

    def __init__(self, name, participant_type="storage"):
        self.name = name
        self.participant_type = participant_type


T = 96
STORAGE = _Agent("ESS6", "storage")
PROSUMER = _Agent("Bus5R", "prosumer")


def _strategy(agent, spec, scale=1.0):
    return {agent.name: {n: np.full(T, spec.truthful[n] * scale, dtype=float)
                         for n in spec.action_names}}


def _assert_in_bounds(agent, actions):
    spec = ACTION_STORAGE_QUOTE_V2 if agent is STORAGE else ACTION_BID_OFFER_V1
    assert set(actions) == set(spec.action_names), (
        f"{agent.name} was varied over {sorted(actions)}, but declares "
        f"{sorted(spec.action_names)}")
    for n in spec.action_names:
        lo, hi = spec.bounds[n]
        arr = np.asarray(actions[n], dtype=float)
        assert arr.shape == (T,)
        assert arr.min() >= lo - 1e-9 and arr.max() <= hi + 1e-9


@pytest.mark.parametrize("agent", [STORAGE, PROSUMER])
def test_variations_stay_inside_the_agents_own_action_space(agent):
    spec = (ACTION_STORAGE_QUOTE_V2 if agent is STORAGE
            else ACTION_BID_OFFER_V1)
    base = _strategy(agent, spec)

    variants = _generate_variations(base, agent, MarketConfig(),
                                    num_variations=40)

    assert len(variants) == 40
    for var in variants:
        _assert_in_bounds(agent, var[agent.name])


def test_variations_include_the_truthful_corner_for_a_battery():
    """The truthful point is a corner of the storage box, so a deterministic
    vertex must land on it -- otherwise the search never evaluates the
    declaration everything else is measured against."""
    base = _strategy(STORAGE, ACTION_STORAGE_QUOTE_V2)

    variants = _generate_variations(base, STORAGE, MarketConfig(),
                                    num_variations=8)

    truthful = ACTION_STORAGE_QUOTE_V2.truthful
    found = any(
        all(np.allclose(var[STORAGE.name][n], truthful[n]) for n in truthful)
        for var in variants)
    assert found


def test_variations_do_not_touch_other_agents_declarations():
    """A best response moves one agent; everyone else's profile is fixed."""
    base = _strategy(STORAGE, ACTION_STORAGE_QUOTE_V2)
    base[PROSUMER.name] = {n: np.full(T, 7.0)
                           for n in ACTION_BID_OFFER_V1.action_names}

    variants = _generate_variations(base, STORAGE, MarketConfig(),
                                    num_variations=12)

    for var in variants:
        assert set(var[PROSUMER.name]) == set(ACTION_BID_OFFER_V1.action_names)
        for n in ACTION_BID_OFFER_V1.action_names:
            assert np.all(var[PROSUMER.name][n] == 7.0)


def test_cache_key_separates_agents_by_the_actions_they_declare():
    """A profile whose declared price moved must not hit the old cache entry.

    The key used to hash `bid_mult` for everyone and `offer_adder` for
    prosumers, so a storage unit's second price was not in the key at all.
    """
    tester = NashEquilibriumTester([STORAGE, PROSUMER], None, T=T, stage="DA")
    base = _strategy(STORAGE, ACTION_STORAGE_QUOTE_V2)
    base[PROSUMER.name] = {n: np.full(T, 1.0)
                           for n in ACTION_BID_OFFER_V1.action_names}

    def key(strategy):
        return tester._make_cache_key(STORAGE.name, strategy)

    initial = key(base)

    moved_charge = {k: dict(v) for k, v in base.items()}
    moved_charge[STORAGE.name]["charge_price"] = np.full(T, 5.0)
    assert key(moved_charge) != initial

    moved_discharge = {k: dict(v) for k, v in base.items()}
    moved_discharge[STORAGE.name]["discharge_price"] = np.full(T, 5.0)
    assert key(moved_discharge) != initial

    moved_other = {k: dict(v) for k, v in base.items()}
    moved_other[PROSUMER.name]["offer_adder"] = np.full(T, 5.0)
    assert key(moved_other) != initial
