# tests/test_baselines.py
"""The baseline arms an AI-assisted policy is measured against.

A baseline that quietly drifts is worse than no baseline: it changes the number
the paper reports without changing anything anyone would think to look at. These
pin the arms' outputs as functions of the observation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import baselines
from rl_spec import OBS_V3


def _obs(soc=0.5, price=1.0, average=1.0, load=0.5, re=0.3):
    """An observation vector with the named features set to given values."""
    idx = {n: i for i, n in enumerate(OBS_V3.feature_order)}
    v = np.zeros(OBS_V3.total_dim, dtype=np.float32)
    v[idx["soc"]] = soc
    v[idx["last_lmp_norm"]] = price
    v[idx["avg_lmp_norm"]] = average
    v[idx["load_feat"]] = load
    v[idx["re_feat"]] = re
    return v


def test_feature_index_covers_every_named_feature():
    for name in OBS_V3.feature_order:
        assert name in baselines._FEATURE_INDEX


def test_truthful_declares_the_stated_prices():
    p = baselines.make_baseline_policy("truthful")
    for obs in (_obs(), _obs(soc=0.05), _obs(price=3.0), _obs(soc=0.99)):
        assert np.allclose(p(obs), [1.0, 0.0])


def test_rule_charges_a_flat_battery_and_discharges_a_full_one():
    p = baselines.make_baseline_policy("rule")
    # State of charge outranks price: a flat battery charges even at a high
    # price, a full one discharges even at a low price.
    assert p(_obs(soc=0.10, price=3.0, average=1.0))[0] > 1.0
    assert p(_obs(soc=0.90, price=0.2, average=1.0))[1] == 0.0
    assert p(_obs(soc=0.90, price=0.2, average=1.0))[0] < 1.0


def test_rule_uses_price_when_the_battery_is_between_its_limits():
    p = baselines.make_baseline_policy("rule")
    cheap = p(_obs(soc=0.5, price=0.5, average=1.0))
    dear = p(_obs(soc=0.5, price=2.0, average=1.0))
    # Cheap energy: bid up to charge, and ask a lot to discharge. Dear: reverse.
    assert cheap[0] > dear[0]
    assert cheap[1] > dear[1]


def test_rule_is_a_pure_function_of_the_observation():
    p = baselines.make_baseline_policy("rule")
    obs = _obs(soc=0.6, price=0.7, average=1.0)
    assert np.allclose(p(obs), p(obs))


def test_myopic_ignores_the_state_of_charge():
    p = baselines.make_baseline_policy("myopic")
    flat = _obs(soc=0.02, price=0.5, average=1.0)
    full = _obs(soc=0.98, price=0.5, average=1.0)
    assert np.allclose(p(flat), p(full))


def test_myopic_reacts_to_the_price():
    p = baselines.make_baseline_policy("myopic")
    cheap = p(_obs(price=0.5, average=1.0))
    dear = p(_obs(price=2.0, average=1.0))
    assert cheap[0] > dear[0]
    assert cheap[1] > dear[1]


def test_every_baseline_stays_inside_the_declared_action_bounds():
    lo = np.array([0.3, 0.0])
    hi = np.array([1.8, 50.0])
    for name in baselines.BASELINE_NAMES:
        p = baselines.make_baseline_policy(name)
        for soc in (0.0, 0.5, 1.0):
            for price in (0.0, 0.5, 1.0, 2.0, 3.0):
                a = p(_obs(soc=soc, price=price, average=1.0))
                assert a.shape == (2,)
                assert np.all(a >= lo) and np.all(a <= hi), (name, soc, price, a)


def test_zero_average_price_does_not_produce_a_ratio_blowup():
    for name in baselines.BASELINE_NAMES:
        a = baselines.make_baseline_policy(name)(_obs(price=1.0, average=0.0))
        assert np.all(np.isfinite(a))


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        baselines.make_baseline_policy("oracle")
