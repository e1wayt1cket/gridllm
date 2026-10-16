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


# ---------------------------------------------------------------------------
# The cache key is canonical, and its tolerance is stated
# ---------------------------------------------------------------------------

def _two_agent_strategy(storage_values, prosumer_values, kind=np.asarray):
    """The same profile built with whatever container the caller passes."""
    return {
        STORAGE.name: {n: kind(storage_values) for n in
                       ACTION_STORAGE_QUOTE_V2.action_names},
        PROSUMER.name: {n: kind(prosumer_values) for n in
                        ACTION_BID_OFFER_V1.action_names},
    }


def _tester():
    return NashEquilibriumTester([STORAGE, PROSUMER], None, T=T, stage="DA")


def test_the_same_profile_keys_the_same_from_a_list_tuple_or_array():
    """A caller's choice of container is not part of the question."""
    tester = _tester()
    values = list(np.linspace(0.0, 10.0, T))
    keys = {
        "array": tester._make_cache_key(
            STORAGE.name, _two_agent_strategy(values, values, np.asarray)),
        "list": tester._make_cache_key(
            STORAGE.name, _two_agent_strategy(values, values, list)),
        "tuple": tester._make_cache_key(
            STORAGE.name, _two_agent_strategy(values, values, tuple)),
    }
    assert keys["array"] == keys["list"] == keys["tuple"], (
        "the same profile produced different keys from different containers")


def test_the_key_holds_python_floats_and_not_numpy_scalars():
    """NumPy scalars compare and hash like floats, but they are not the same
    objects, and a key whose equality rests on that is one upgrade from
    changing silently."""
    key = _tester()._make_cache_key(
        STORAGE.name, _two_agent_strategy(np.zeros(T), np.ones(T)))
    for part in key[1:]:
        assert isinstance(part, tuple)
        for v in part:
            assert type(v) is float, f"{v!r} is {type(v).__name__}, not float"


def test_a_difference_the_search_can_act_on_is_not_shared():
    """The quantization may only merge profiles the search cannot tell apart."""
    from nash import _KEY_DECIMALS, _rhobeg_for
    from rl_spec import ACTION_STORAGE_QUOTE_V2 as spec

    tester = _tester()
    base = np.zeros(T)
    other = base.copy()
    step = 10.0 ** (-_KEY_DECIMALS) * 10.0
    other[0] = step
    assert tester._make_cache_key(STORAGE.name,
                                  _two_agent_strategy(base, base)) != \
        tester._make_cache_key(STORAGE.name,
                               _two_agent_strategy(other, base)), (
            f"a {step} move on a quote produced the same key")

    # And the tolerance really is below what the search can resolve.
    assert step < _rhobeg_for(spec) / 1000.0


def test_a_difference_below_the_tolerance_does_share_a_key():
    """Stated rather than accidental: this is what the cache buys."""
    from nash import _KEY_DECIMALS

    tester = _tester()
    base = np.zeros(T)
    nudged = base.copy()
    nudged[0] = 10.0 ** (-(_KEY_DECIMALS + 2))
    assert tester._make_cache_key(STORAGE.name,
                                  _two_agent_strategy(base, base)) == \
        tester._make_cache_key(STORAGE.name,
                               _two_agent_strategy(nudged, base))


def test_the_key_is_stable_across_instances_and_orders_of_construction():
    """No object identity and no `repr` may reach the key."""
    values = np.linspace(0.0, 10.0, T)
    first = _tester()._make_cache_key(
        STORAGE.name, _two_agent_strategy(values, values))
    second = NashEquilibriumTester([PROSUMER, STORAGE], None, T=T,
                                   stage="DA")._make_cache_key(
        STORAGE.name, _two_agent_strategy(values, values))
    assert first == second
    assert "_Agent" not in repr(first) and "0x" not in repr(first)


def test_a_cache_hit_is_answered_from_the_canonical_form():
    """A profile equivalent to a cached one must hit, and not re-evaluate."""
    tester = _tester()
    values = np.linspace(0.0, 10.0, T)
    key = tester._make_cache_key(
        STORAGE.name, _two_agent_strategy(values, values, np.asarray))
    sentinel = ("cached", "payoff")
    tester._payoff_cache[key] = sentinel

    list_built = _two_agent_strategy(values, values, list)
    assert tester._best_response(STORAGE, list_built, num_variations=1) \
        is sentinel


def test_a_profile_missing_a_declared_action_is_refused():
    """Otherwise it keys as an empty tuple, which collides with every other
    empty tuple and caches a payoff for a profile that declares nothing."""
    tester = _tester()
    incomplete = {STORAGE.name: {"charge_price": np.zeros(T)}}
    # The refusal names the agent and the action it is missing, so a caller
    # reading the message knows which leg of whose quote to supply.
    with pytest.raises(ValueError, match=r"declares no '\w+' for \w+"):
        tester._make_cache_key(STORAGE.name, incomplete)


def test_a_non_finite_action_is_refused_rather_than_keyed():
    """NaN is not equal to itself, so such a key would never hit again and the
    cache would miss forever while looking like it was working."""
    tester = _tester()
    bad = np.zeros(T)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tester._make_cache_key(STORAGE.name,
                               _two_agent_strategy(bad, np.ones(T)))
    bad[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        tester._make_cache_key(STORAGE.name,
                               _two_agent_strategy(bad, np.ones(T)))
